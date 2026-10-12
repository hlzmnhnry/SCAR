"""Assemble and run the SCAR pipeline from a :class:`~scar.config.schema.RunConfig`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from scar.config.schema import RunConfig
from scar.core.types import Frame, Sequence
from scar.geodesy import HeightReference, WorldFrame
from scar.io import load_camera, load_extrinsics, load_sequence, save_sequence
from scar.sources.images import FileListImageSource
from scar.sources.poses import PoseTable

log = logging.getLogger(__name__)

ANCHORS_FILE = "anchors.json"


@dataclass
class PreparedSequence:
    sequence: Sequence
    images: FileListImageSource

    def read_image(self, frame: Frame) -> np.ndarray:
        return self.images.read(frame.image_path)


def world_frame(config: RunConfig) -> WorldFrame:
    world = config.world
    if world.height == "geoid":
        grid = config.path(world.geoid_grid) if world.geoid_grid and Path(world.geoid_grid).suffix else world.geoid_grid
        height = HeightReference.geoid(str(grid), world.height_name)
    else:
        height = HeightReference(world.height, name=world.height_name)
    return WorldFrame(crs=world.crs, height=height, apply_meridian_convergence=world.meridian_convergence)


def image_source(config: RunConfig) -> FileListImageSource:
    images = config.sequence.images
    if images.type == "directory":
        return FileListImageSource.from_directory(config.path(images.directory), time_unit=images.time_unit,
            suffix=images.suffix)
    if images.type == "ltc":
        return FileListImageSource.from_csv(config.path(images.csv), timestamp_column="timestamp", path_column=None,
            path_template="{index}.png", time_unit="us", root=config.path(images.root))
    return FileListImageSource.from_csv(config.path(images.csv), timestamp_column=images.timestamp_column,
        path_column=images.path_column, path_template=images.path_template, time_unit=images.time_unit,
        root=config.path(images.root))


def pose_priors(config: RunConfig, world: WorldFrame, timestamps_ns: np.ndarray) -> list:
    from scar.core.types import PosePrior
    from scar.datasets.generic_csv import read_pose_csv
    from scar.datasets.ltc import read_ins_pose_table
    from scar.io.legacy import legacy_ltc_body_poses, read_legacy_ins

    poses = config.sequence.poses
    csv_path = config.path(poses.csv)
    query = np.asarray(timestamps_ns, dtype=np.int64) + round(poses.time_offset_s * 1e9)

    if poses.type == "ltc_legacy":
        return [PosePrior(T) for T in legacy_ltc_body_poses(read_legacy_ins(csv_path), query // 1000)]

    table: PoseTable = read_ins_pose_table(csv_path, world, poses.ltc_height) if poses.type == "ltc" \
        else read_pose_csv(csv_path, poses.format, world)
    max_gap = round(poses.max_gap_s * 1e9)
    return [table.prior_at(int(t), mode=poses.interpolation, max_gap_ns=max_gap) for t in query]


def _seconds_to_ns(value: str | None) -> int | None:
    return None if value is None else int(Decimal(str(value)) * 1_000_000_000)


def prepare_sequence(config: RunConfig) -> PreparedSequence:
    """Frames (selected images with pose priors), initial camera and extrinsics."""
    sequence_config = config.sequence
    world = world_frame(config)
    images = image_source(config)

    selection = sequence_config.selection
    indices = images.select(start_ns=_seconds_to_ns(selection.start), end_ns=_seconds_to_ns(selection.end),
        step=selection.step,
        interval_ns=None if selection.interval_s is None else round(selection.interval_s * 1e9),
        tolerance_ns=None if selection.tolerance_s is None else round(selection.tolerance_s * 1e9))
    if len(indices) == 0:
        raise ValueError("No images selected; check sequence.selection and the image timestamps")

    timestamps = images.timestamps_ns[indices]
    priors = pose_priors(config, world, timestamps)
    frames = tuple(Frame(frame_id=i, timestamp_ns=int(t), image_path=images.paths[index], pose_prior=prior)
        for i, (index, t, prior) in enumerate(zip(indices, timestamps, priors, strict=True)))

    sequence = Sequence(
        camera=load_camera(config.path(sequence_config.camera), sequence_config.camera_key),
        extrinsics=load_extrinsics(config.path(sequence_config.extrinsics), sequence_config.camera_key),
        frames=frames, world_frame=world, metadata={"name": config.name},
    )
    return PreparedSequence(sequence=sequence, images=images)


def image_mask(config: RunConfig, width: int, height: int) -> np.ndarray | None:
    """Pixels to ignore: a mask image (non-zero = ignore) and/or a hot-pixel list (``x,y`` per line)."""
    sequence_config = config.sequence
    mask = np.zeros((height, width), dtype=bool)
    if sequence_config.mask:
        image = cv2.imread(str(config.path(sequence_config.mask)), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (height, width):
            raise ValueError(f"Mask {sequence_config.mask} must be a {width}x{height} image")
        mask |= image > 0
    if sequence_config.hot_pixels:
        points = np.loadtxt(config.path(sequence_config.hot_pixels), delimiter=",", ndmin=2).astype(int)
        inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        mask[points[inside, 1], points[inside, 0]] = True
        radius = sequence_config.hot_pixel_radius
        if radius > 0:
            kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask if mask.any() else None


def reference_provider(config: RunConfig, world: WorldFrame):
    """Georay reference provider with the configuration's paths resolved."""
    from dataclasses import replace

    from scar.references import GeorayReferenceProvider

    reference = replace(config.reference,
        image_files=tuple(str(config.path(p)) for p in config.reference.image_files),
        dsm_files=tuple(str(config.path(p)) for p in config.reference.dsm_files),
        tile_cache_dir=str(config.path(config.reference.tile_cache_dir)))
    return GeorayReferenceProvider(world, reference)


def make_matcher(config: RunConfig):
    """The matcher of ``collect.matcher``."""
    from scar.collect import ImageMatchingModelsMatcher, OpenCVSiftMatcher

    collect = config.collect
    if collect.matcher == "opencv-sift":
        return OpenCVSiftMatcher()
    return ImageMatchingModelsMatcher(collect.matcher, device=collect.device)


def run_collect(config: RunConfig, resume: bool = False, on_frame: Callable | None = None,
    prepared: PreparedSequence | None = None) -> Sequence:
    from scar.collect import AnchorCollector

    prepared = prepared or prepare_sequence(config)
    sequence = prepared.sequence
    output = config.path(config.output)

    references = reference_provider(config, sequence.world_frame)

    collect = config.collect
    collector = AnchorCollector(collect.settings, make_matcher(config), references, prepared.read_image,
        on_frame=on_frame,
        image_mask=image_mask(config, sequence.camera.width, sequence.camera.height))

    log.info("collecting anchors for %d frames", len(sequence.frames))
    result = collector.collect(sequence, checkpoint_directory=output / "collect", resume=resume)
    save_sequence(result, output / ANCHORS_FILE)
    return result


def run_anchor_noise(config: RunConfig, anchors: Sequence | None = None, noise_config=None):
    """Re-estimate the anchors of a run from later observations; writes ``anchor_noise.json``."""
    from scar.evaluate import estimate_anchor_noise

    output = config.path(config.output)
    anchors = anchors or load_sequence(output / ANCHORS_FILE)
    prepared = prepare_sequence(config)
    references = reference_provider(config, anchors.world_frame)
    result = estimate_anchor_noise(anchors, make_matcher(config), references, prepared.read_image,
        config.collect.settings, noise_config)
    result.save(output / "anchor_noise.json")
    return result


def run_optimize(config: RunConfig, anchors: Sequence | None = None):
    from scar.io.results import save_result
    from scar.optimize import load_preset, run_calibration

    output = config.path(config.output)
    anchors = anchors or load_sequence(output / ANCHORS_FILE)
    preset_name = config.optimize.preset
    preset_path = config.path(preset_name) if Path(preset_name).suffix else preset_name
    result = run_calibration(anchors, load_preset(preset_path))
    save_result(result, output / "calibration", preset_name=preset_name)
    return result
