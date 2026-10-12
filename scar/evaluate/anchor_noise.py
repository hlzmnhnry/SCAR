"""Empirical noise of collected anchors by re-matching tracked observations.

An anchor is lifted to 3D once, from the reference match in the frame where it was collected. Later
frames observe the same ground point through feature tracking. Matching such a later frame against
its own rendered reference and lifting the tracked pixel again gives an independent second estimate
of the anchor. The differences between these re-estimates and the anchors measure the precision of
the whole chain (matching, tracking, DSM lookup) and can be compared with the landmark prior of the
optimization.

The tracked pixel is generally not a matched keypoint: its reference position is interpolated with a
local affine fit to the inlier matches around it, which works with any matcher.

Interpretation: with the same reference imagery for collection and re-matching, errors shared by
both (georeferencing of the orthophoto, DSM errors) cancel, so the result is a precision, not an
absolute accuracy. Re-match against different imagery (e.g. another year) to include them. If
anchor and re-estimate errors are independent and alike, the anchor noise is the spread of the
differences divided by sqrt(2).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from scar.collect.collector import AnchorCollector, CollectConfig
from scar.collect.filters import homography_inliers, slope_mask
from scar.collect.matching import Matcher, Matches
from scar.core.types import Frame, Sequence
from scar.references.base import ReferenceProvider, ReferenceView

log = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AnchorNoiseConfig:
    """Which observations are re-estimated and how the reference position of a pixel is found."""

    every_nth_frame: int = 1
    # re-estimate only observations at least this many frames after the anchor was collected
    min_frame_gap: int = 1
    # inlier matches within this image radius of a tracked pixel define its local affine transfer
    neighbor_radius_px: float = 12.0
    min_neighbors: int = 4
    max_fit_rms_px: float = 1.5
    # differences beyond this are counted as outliers (reported, not removed from the robust statistics)
    outlier_threshold_m: float = 5.0

    def __post_init__(self) -> None:
        if self.every_nth_frame < 1 or self.min_frame_gap < 1:
            raise ValueError("every_nth_frame and min_frame_gap must be at least 1")
        if self.min_neighbors < 3:
            raise ValueError("an affine transfer needs at least 3 neighbouring matches")


@dataclass
class AnchorNoiseResult:
    """Re-estimates of anchors (one row per re-estimated observation)."""

    landmark_ids: np.ndarray
    frame_ids: np.ndarray
    anchors_xyz: np.ndarray  # (N, 3) anchor positions (local world coordinates)
    estimates_xyz: np.ndarray  # (N, 3) re-estimated positions
    neighbors: np.ndarray  # matches used for the affine transfer
    fit_rms_px: np.ndarray
    frames_matched: int = 0
    frames_skipped: int = 0
    config: AnchorNoiseConfig = field(default_factory=AnchorNoiseConfig)

    @property
    def deltas(self) -> np.ndarray:
        """Re-estimate minus anchor (N, 3) [m]."""
        return self.estimates_xyz - self.anchors_xyz

    def summary(self) -> dict:
        deltas = self.deltas
        if len(deltas) == 0:
            return {"estimates": 0, "landmarks": 0, "frames_matched": self.frames_matched,
                "frames_skipped": self.frames_skipped}
        median = np.median(deltas, axis=0)
        sigma = MAD_TO_SIGMA * np.median(np.abs(deltas - median), axis=0)
        horizontal = np.linalg.norm(deltas[:, :2], axis=1)
        threshold = self.config.outlier_threshold_m
        outliers = (horizontal > threshold) | (np.abs(deltas[:, 2]) > threshold)
        return {
            "estimates": len(deltas),
            "landmarks": len(np.unique(self.landmark_ids)),
            "frames_matched": self.frames_matched,
            "frames_skipped": self.frames_skipped,
            "bias_m": median.tolist(),
            "sigma_difference_m": sigma.tolist(),
            "sigma_anchor_m": (sigma / np.sqrt(2.0)).tolist(),
            "median_horizontal_difference_m": float(np.median(horizontal)),
            "outlier_fraction": float(outliers.mean()),
        }

    def save(self, path: str | Path) -> None:
        rows = [{"landmark_id": int(lid), "frame_id": int(fid), "anchor": a.tolist(), "estimate": e.tolist(),
            "neighbors": int(n), "fit_rms_px": float(r)}
            for lid, fid, a, e, n, r in zip(self.landmark_ids, self.frame_ids, self.anchors_xyz, self.estimates_xyz,
                self.neighbors, self.fit_rms_px, strict=True)]
        Path(path).write_text(json.dumps({"config": asdict(self.config), "summary": self.summary(), "estimates": rows},
            indent=1))


def _local_affine(image_points: np.ndarray, reference_points: np.ndarray, query: np.ndarray,
    config: AnchorNoiseConfig) -> tuple[np.ndarray | None, int, float]:
    """Reference position of ``query`` from an affine fit to the matches around it."""
    near = np.linalg.norm(image_points - query, axis=1) <= config.neighbor_radius_px
    count = int(near.sum())
    if count < config.min_neighbors:
        return None, count, np.nan
    source = image_points[near] - query
    design = np.column_stack([source, np.ones(count)])
    solution, *_ = np.linalg.lstsq(design, reference_points[near], rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((design @ solution - reference_points[near]) ** 2, axis=1))))
    if rms > config.max_fit_rms_px:
        return None, count, rms
    return solution[2], count, rms  # the query is the origin of the centred fit


def _filtered_matches(collector: AnchorCollector, sequence: Sequence, reference: ReferenceView,
    image: np.ndarray) -> Matches | None:
    """Reference matches of a frame with the collection's filters (no feature budget)."""
    config = collector.config
    matches = collector.matcher.match(collector.preprocess(reference.image), image)
    if len(matches) < config.min_detections:
        return None
    if config.homography_filter:
        inliers = homography_inliers(matches.reference, sequence.camera.undistort_pixels(matches.image),
            config.homography_threshold_px, config.homography_confidence, config.homography_max_iterations)
        if inliers is None or inliers.sum() < config.min_inliers:
            return None
        matches = matches.subset(inliers)
    if config.slope_filter:
        matches = matches.subset(slope_mask(reference.dsm, matches.reference + config.reference_pixel_offset,
            config.slope_window, config.slope_threshold_m))
    return matches


def estimate_anchor_noise(anchors: Sequence, matcher: Matcher, references: ReferenceProvider,
    read_image: Callable[[Frame], np.ndarray], collect_config: CollectConfig | None = None,
    config: AnchorNoiseConfig | None = None) -> AnchorNoiseResult:
    """Re-estimate the anchors of ``anchors`` (collected sequence) from later, tracked observations.

    ``collect_config`` should be the configuration the anchors were collected with (matching
    filters, pixel convention, height lookup); ``references`` may use other imagery than the
    collection to include errors of the reference data.
    """
    collect_config = collect_config or CollectConfig()
    config = config or AnchorNoiseConfig()
    collector = AnchorCollector(collect_config, matcher, references, read_image)

    observations = anchors.observations
    frame_index = {frame.frame_id: i for i, frame in enumerate(anchors.frames)}
    order = np.argsort([frame_index[int(f)] for f in observations.frame_ids], kind="stable")
    birth: dict[int, int] = {}
    for k in order:
        birth.setdefault(int(observations.landmark_ids[k]), frame_index[int(observations.frame_ids[k])])
    xyz = dict(zip(anchors.landmarks.ids.tolist(), anchors.landmarks.xyz, strict=True))

    rows: list[tuple] = []
    matched = skipped = 0
    for index, frame in enumerate(anchors.frames):
        if index % config.every_nth_frame:
            continue
        in_frame = np.flatnonzero(observations.frame_ids == frame.frame_id)
        in_frame = in_frame[[index - birth[int(observations.landmark_ids[k])] >= config.min_frame_gap
            for k in in_frame]] if len(in_frame) else in_frame
        if len(in_frame) == 0:
            continue

        T_world_camera = frame.pose_prior.T_world_camera(anchors.extrinsics)
        try:
            reference = references.render(T_world_camera, anchors.camera, timestamp_ns=frame.timestamp_ns)
        except (ValueError, RuntimeError) as error:
            log.warning("frame %d: no reference (%s)", frame.frame_id, error)
            skipped += 1
            continue
        matches = _filtered_matches(collector, anchors, reference, collector.preprocess(read_image(frame)))
        if matches is None or len(matches) < config.min_neighbors:
            skipped += 1
            continue
        matched += 1

        for k in in_frame:
            uv = observations.uv[k]
            reference_pixel, count, rms = _local_affine(matches.image, matches.reference, uv, config)
            if reference_pixel is None:
                continue
            pixel = (reference_pixel + collect_config.reference_pixel_offset)[None]
            height = reference.heights(pixel, collect_config.height_lookup)[0]
            if not np.isfinite(height):
                continue
            landmark_id = int(observations.landmark_ids[k])
            estimate = np.array([*reference.pixels_to_world(pixel)[0], height])
            rows.append((landmark_id, frame.frame_id, xyz[landmark_id], estimate, count, rms))
        log.info("frame %d: %d re-estimates so far", frame.frame_id, len(rows))

    if rows:
        lids, fids, a, e, n, r = (np.asarray(column) for column in zip(*rows, strict=True))
    else:
        lids = fids = n = np.zeros(0, dtype=int)
        a = e = np.zeros((0, 3))
        r = np.zeros(0)
    return AnchorNoiseResult(landmark_ids=lids.astype(int), frame_ids=fids.astype(int), anchors_xyz=a.reshape(-1, 3),
        estimates_xyz=e.reshape(-1, 3), neighbors=n.astype(int), fit_rms_px=r.astype(float), frames_matched=matched,
        frames_skipped=skipped, config=config)
