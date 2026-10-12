"""Reference data from georay (German state data via georaffer, or local raster files)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from typing import Any

import numpy as np
from pyproj import CRS

from scar.core.camera import CameraModel
from scar.geodesy.world_frame import WorldFrame

from .base import ReferenceView


def round_125(x: float) -> float:
    """Nearest value of the 1-2-5 series (..., 0.5, 1, 2, 5, 10, ...)."""
    if x <= 0:
        raise ValueError("x must be positive")
    base = 10.0 ** math.floor(math.log10(x))
    return min((1 * base, 2 * base, 5 * base, 10 * base), key=lambda v: abs(v - x))


@dataclass(frozen=True)
class GeorayReferenceConfig:
    """How reference views are cut out.

    Data comes from georaffer (``regions``) unless ``image_files``/``dsm_files`` are given, which
    are used through georay's ``RasterFileSource`` (any projected CRS). DSM heights must be in the
    vertical reference of the sequence's world frame.

    Resolution: ``fixed`` uses ``pixels_per_meter``; ``match_image`` picks the 1-2-5 value closest
    to the aerial image's ground resolution, clamped to ``[min_ppm, max_ppm]`` (legacy behaviour).
    """

    regions: tuple[str, ...] = ("nrw", "rlp")
    imagery_from: tuple[int, int | None] = (2015, None)
    dsm_from: tuple[int, int | None] | None = None
    image_files: tuple[str, ...] = ()
    dsm_files: tuple[str, ...] = ()
    tile_size_m: float = 1000.0
    tile_cache_dir: str = "./tiles"
    resolution: str = "match_image"
    pixels_per_meter: float = 2.0
    min_ppm: float = 1.0
    max_ppm: float = 5.0
    bbox_transform: str = "yaw_enclosing"
    bbox_enlargement: float = 0.0
    north_aligned_square_size: float | None = None
    ranking_order: str = "SEASON_YEAR_STATE"
    preferred_date: date | None = None
    preferred_regions: tuple[str, ...] = ("nrw",)
    dsm_interpolation: str = "nearest"
    show_progress: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolution not in ("fixed", "match_image"):
            raise ValueError("resolution must be 'fixed' or 'match_image'")
        if self.bbox_transform not in ("direct", "north_aligned", "yaw_enclosing"):
            raise ValueError(f"Unknown bbox_transform {self.bbox_transform!r}")


class GeorayReferenceProvider:
    """Adapter from SCAR's :class:`~scar.references.base.ReferenceProvider` interface to georay."""

    def __init__(self, world_frame: WorldFrame, config: GeorayReferenceConfig | None = None, source=None) -> None:
        self.world_frame = world_frame
        self.config = config or GeorayReferenceConfig()
        self.source = source or self._create_source(self.config)
        if not CRS.from_user_input(self.source.crs).equals(world_frame.pyproj_crs):
            raise ValueError(f"Reference data CRS {self.source.crs} differs from the world frame CRS {world_frame.crs}")
        self._renderers: dict[CameraModel, Any] = {}
        self._bbox_transform = self._create_bbox_transform(self.config)
        self._origin = np.asarray(world_frame.origin, dtype=float)

    def render(self, T_world_camera: np.ndarray, camera: CameraModel, *, pixels_per_meter: float | None = None,
        timestamp_ns: int | None = None) -> ReferenceView:
        from georay.types.pose import Pose6D

        T_absolute = np.asarray(T_world_camera, dtype=float).copy()
        T_absolute[:3, 3] += self._origin
        pose = Pose6D.from_transformation_matrix(T_absolute)
        renderer = self._renderer(camera)
        config = self.config

        footprint = renderer.compute_bounding_box(pose=pose, preferred_date=config.preferred_date,
            preferred_regions=list(config.preferred_regions))
        if pixels_per_meter is None:
            pixels_per_meter = self.resolution_for(camera, footprint.as_array())

        result = renderer.render_from_bbox(
            bbox=footprint,
            pose=pose,
            pixels_per_meter=pixels_per_meter,
            bbox_transform=self._bbox_transform,
            preferred_date=config.preferred_date,
            preferred_regions=list(config.preferred_regions),
            render_image=True,
            render_dsm=True,
            dsm_interpolation=config.dsm_interpolation,
        )

        to_local = np.array([[1.0, 0.0, -self._origin[0]], [0.0, 1.0, -self._origin[1]], [0.0, 0.0, 1.0]])
        dsm = result.dsm.rasters[0].astype(np.float64) - self._origin[2]
        if dsm.shape != result.image.rasters[0].shape[:2]:
            raise RuntimeError("georay returned image and DSM on different grids")

        return ReferenceView(
            image=result.image.rasters[0],
            dsm=dsm,
            pixel_to_world=to_local @ result.image.pixel_to_utm_matrix,
            footprint=result.image.footprint.as_array() - self._origin[:2],
            metadata={
                **config.metadata,
                "backend": "georay",
                "pixels_per_meter": pixels_per_meter,
                "raw_footprint": (footprint.as_array() - self._origin[:2]).tolist(),
                "image_tiles": _tile_summary(result.image.meta),
                "dsm_tiles": _tile_summary(result.dsm.meta),
            },
        )

    def resolution_for(self, camera: CameraModel, footprint_xy: np.ndarray) -> float:
        """Pixels per meter for a camera footprint (UL, UR, LR, LL corners)."""
        config = self.config
        if config.resolution == "fixed":
            return config.pixels_per_meter
        diagonal = float(np.linalg.norm(np.asarray(footprint_xy)[2] - np.asarray(footprint_xy)[0]))
        ppm = round_125(float(np.hypot(camera.height, camera.width)) / diagonal)
        return float(min(max(ppm, config.min_ppm), config.max_ppm))

    def _renderer(self, camera: CameraModel):
        if camera not in self._renderers:
            from georay.geometry.distortion import (
                NoDistortion,
                RadialTangentialDistortion,
            )
            from georay.geometry.pinhole_camera import PinholeCamera
            from georay.raster.reference_tile_selector import (
                ClosestReferenceTileSelector,
            )
            from georay.reference_view_renderer import ReferenceViewRenderer
            from georay.types.tile import RankingOrder

            distortion = RadialTangentialDistortion.from_opencv(list(camera.distortion)) if camera.distortion \
                else NoDistortion()
            georay_camera = PinholeCamera(image_width=camera.width, image_height=camera.height, fx=camera.fx,
                fy=camera.fy, cx=camera.cx, cy=camera.cy, distortion=distortion)
            self._renderers[camera] = ReferenceViewRenderer(
                camera=georay_camera,
                source=self.source,
                tile_selector=ClosestReferenceTileSelector(ranking_order=RankingOrder[self.config.ranking_order]),
                tile_cache_dir=self.config.tile_cache_dir,
                show_progress=self.config.show_progress,
            )
        return self._renderers[camera]

    @staticmethod
    def _create_source(config: GeorayReferenceConfig):
        from georay.raster.sources import GeorafferSource, RasterFileSource

        if config.image_files or config.dsm_files:
            return RasterFileSource(list(config.image_files), list(config.dsm_files), tile_size_m=config.tile_size_m,
                cache_dir=f"{config.tile_cache_dir}/files", recording_date=config.preferred_date)
        return GeorafferSource(regions=list(config.regions), imagery_from=config.imagery_from,
            dsm_from=config.dsm_from, tile_path=config.tile_cache_dir)

    @staticmethod
    def _create_bbox_transform(config: GeorayReferenceConfig):
        from georay.spatial.bounding_box_transform import (
            DirectBoundingBoxTransform,
            NorthAlignedBoundingBoxTransform,
            YawEnclosingRectangularBoundingBoxTransform,
        )

        if config.bbox_transform == "direct":
            return DirectBoundingBoxTransform()
        if config.bbox_transform == "north_aligned":
            return NorthAlignedBoundingBoxTransform(enlargement=config.bbox_enlargement,
                convert_to_square=config.north_aligned_square_size is not None,
                square_size=config.north_aligned_square_size)
        return YawEnclosingRectangularBoundingBoxTransform(enlargement=config.bbox_enlargement,
            should_enforce_aspect_ratio=False, aspect_ratio=Fraction(4, 3))


def _tile_summary(meta) -> list[dict[str, Any]]:
    if not meta:
        return []
    return [{"coord": list(tile.coord), "region": tile.region, "year": tile.year,
        "recording_date": tile.recording_date.isoformat() if tile.recording_date else None} for tile in meta[0].tiles]
