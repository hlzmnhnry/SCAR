"""Collection of georeferenced anchors: match frames against references, track, lift to 3D."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from scar.core.types import Frame, Landmarks, Observations, Sequence
from scar.io.sequence import load_sequence, save_sequence
from scar.references.base import ReferenceProvider, ReferenceView

from .filters import homography_inliers, select_new_anchors, slope_mask
from .matching import Matcher, Matches
from .tracking import TrackingConfig, track

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectConfig:
    """Parameters of anchor collection (defaults: CVPRW 2026 LTC configuration)."""

    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8
    tracking: bool = True
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    collect_every_nth_frame: int = 2
    min_detections: int = 50
    min_inliers: int = 20
    homography_filter: bool = True
    homography_threshold_px: float = 3.0
    homography_confidence: float = 0.95
    homography_max_iterations: int = 2000
    slope_filter: bool = True
    slope_window: int = 15
    slope_threshold_m: float = 5.0
    min_confidence: float = 0.5
    max_features: int = 500
    min_pixel_distance: float = 30.0
    height_lookup: str = "nearest"
    # matchers report keypoints with pixel centres at integers (OpenCV convention); references map
    # pixel (i, j) to the area [i, i+1) x [j, j+1). The CVPRW 2026 pipeline used 0.0 here.
    reference_pixel_offset: float = 0.5
    legacy_feature_budget: bool = False
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        if self.min_detections <= self.min_inliers:
            raise ValueError("min_detections must be greater than min_inliers")
        if self.height_lookup not in ("nearest", "bilinear"):
            raise ValueError("height_lookup must be 'nearest' or 'bilinear'")

    @classmethod
    def from_dict(cls, data: dict) -> CollectConfig:
        data = dict(data)
        tracking = data.pop("tracking_config", None)
        return cls(tracking_config=TrackingConfig(**tracking) if tracking else TrackingConfig(), **data)


@dataclass
class FrameReport:
    """What happened in one frame; handed to the ``on_frame`` callback (e.g. a viewer)."""

    frame: Frame
    image: np.ndarray
    reference: ReferenceView | None = None
    tracked_ids: np.ndarray | None = None
    tracked_points: np.ndarray | None = None
    matches: Matches | None = None
    new_landmark_ids: np.ndarray | None = None
    num_tracked: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    num_new: int = 0
    skipped_reason: str | None = None


@dataclass
class _State:
    landmark_ids: list[int] = field(default_factory=list)
    landmark_xyz: list[np.ndarray] = field(default_factory=list)
    obs_frame: list[int] = field(default_factory=list)
    obs_landmark: list[int] = field(default_factory=list)
    obs_uv: list[np.ndarray] = field(default_factory=list)
    tracked_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    tracked_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    next_id: int = 0
    last_frame_index: int = -1


class AnchorCollector:
    """Collects anchors for the frames of a sequence (frame poses and camera are initial guesses).

    For every frame: track the current anchors from the previous frame, and every
    ``collect_every_nth_frame`` frames (while fewer than ``max_features`` are tracked) match the
    image against a rendered reference, filter the matches and lift new anchors with the DSM.
    """

    def __init__(self, config: CollectConfig, matcher: Matcher, references: ReferenceProvider,
        read_image: Callable[[Frame], np.ndarray], on_frame: Callable[[FrameReport], None] | None = None,
        image_mask: np.ndarray | None = None) -> None:
        """``image_mask`` (bool, image size) marks pixels whose features are ignored, e.g. hot pixels."""
        self.config = config
        self.matcher = matcher
        self.references = references
        self.read_image = read_image
        self.on_frame = on_frame
        self.image_mask = None if image_mask is None else np.asarray(image_mask, dtype=bool)
        self._clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid, config.clahe_tile_grid))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.dtype != np.uint8:
            raise ValueError("Images must be 8-bit")
        return self._clahe.apply(image)

    def collect(self, sequence: Sequence, checkpoint_directory: str | Path | None = None, resume: bool = False) -> Sequence:
        """Return ``sequence`` with collected landmarks and observations."""
        checkpoint_directory = Path(checkpoint_directory) if checkpoint_directory else None
        state = _State()
        previous_image = None
        if resume:
            if checkpoint_directory is None:
                raise ValueError("Resuming needs a checkpoint directory")
            state = self._load_checkpoint(checkpoint_directory)
            if state.last_frame_index >= 0:
                previous_image = self.preprocess(self.read_image(sequence.frames[state.last_frame_index]))
            log.info("resuming after frame %d with %d anchors", state.last_frame_index, len(state.landmark_ids))

        for index in range(state.last_frame_index + 1, len(sequence.frames)):
            frame = sequence.frames[index]
            image = self.preprocess(self.read_image(frame))
            report = FrameReport(frame=frame, image=image)

            if self.config.tracking and previous_image is not None and len(state.tracked_ids):
                self._track(sequence, frame, previous_image, image, state, report)
            elif not self.config.tracking:
                state.tracked_ids, state.tracked_points = np.zeros(0, dtype=int), np.zeros((0, 2))

            if len(state.tracked_ids) < self.config.max_features and index % self.config.collect_every_nth_frame == 0:
                self._collect_new(sequence, frame, image, state, report)

            report.tracked_ids, report.tracked_points = state.tracked_ids.copy(), state.tracked_points.copy()
            if self.on_frame is not None:
                self.on_frame(report)

            previous_image = image
            state.last_frame_index = index
            if checkpoint_directory is not None and (index + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(sequence, state, checkpoint_directory)

        result = self._to_sequence(sequence, state)
        if checkpoint_directory is not None:
            self._save_checkpoint(sequence, state, checkpoint_directory)
        return result

    def _track(self, sequence, frame, previous_image, image, state, report) -> None:
        points, keep = track(previous_image, image, state.tracked_points, sequence.camera, self.config.tracking_config)
        keep &= ~self._masked(points)
        state.tracked_ids = state.tracked_ids[keep]
        state.tracked_points = points[keep]
        for landmark_id, uv in zip(state.tracked_ids, state.tracked_points, strict=True):
            state.obs_frame.append(frame.frame_id)
            state.obs_landmark.append(int(landmark_id))
            state.obs_uv.append(uv)
        report.num_tracked = len(state.tracked_ids)

    def _collect_new(self, sequence, frame, image, state, report) -> None:
        config = self.config
        T_world_camera = frame.pose_prior.T_world_camera(sequence.extrinsics)
        try:
            reference = self.references.render(T_world_camera, sequence.camera, timestamp_ns=frame.timestamp_ns)
        except (ValueError, RuntimeError) as error:
            report.skipped_reason = f"no reference: {error}"
            log.warning("frame %d: %s", frame.frame_id, report.skipped_reason)
            return
        report.reference = reference

        matches = self.matcher.match(self.preprocess(reference.image), image)
        report.num_matches = len(matches)
        if len(matches) < config.min_detections:
            report.skipped_reason = f"{len(matches)} matches < {config.min_detections}"
            return

        if config.homography_filter:
            inliers = homography_inliers(matches.reference, sequence.camera.undistort_pixels(matches.image),
                config.homography_threshold_px, config.homography_confidence, config.homography_max_iterations)
            if inliers is None or inliers.sum() < config.min_inliers:
                report.skipped_reason = "too few homography inliers"
                return
            matches = matches.subset(inliers)
        report.num_inliers = len(matches)

        if config.slope_filter:
            matches = matches.subset(slope_mask(reference.dsm, matches.reference + config.reference_pixel_offset,
                config.slope_window, config.slope_threshold_m))

        matches = matches.subset(~self._masked(matches.image))
        chosen = select_new_anchors(matches.image, matches.confidence, state.tracked_points, config.min_pixel_distance,
            config.max_features, config.min_confidence, config.legacy_feature_budget)
        matches = matches.subset(chosen)
        report.matches = matches

        reference_pixels = matches.reference + config.reference_pixel_offset
        xy = reference.pixels_to_world(reference_pixels)
        z = reference.heights(reference_pixels, config.height_lookup)
        valid = np.isfinite(z)

        new_ids = []
        for world_xy, height, uv in zip(xy[valid], z[valid], matches.image[valid], strict=True):
            landmark_id = state.next_id
            state.next_id += 1
            new_ids.append(landmark_id)
            state.landmark_ids.append(landmark_id)
            state.landmark_xyz.append(np.array([world_xy[0], world_xy[1], height]))
            state.obs_frame.append(frame.frame_id)
            state.obs_landmark.append(landmark_id)
            state.obs_uv.append(uv)

        state.tracked_ids = np.concatenate([state.tracked_ids, np.asarray(new_ids, dtype=int)])
        state.tracked_points = np.vstack([state.tracked_points, matches.image[valid]]) if len(new_ids) \
            else state.tracked_points
        report.new_landmark_ids = np.asarray(new_ids, dtype=int)
        report.num_new = len(new_ids)
        log.info("frame %d: %d matches, %d inliers, %d new anchors, %d tracked", frame.frame_id, report.num_matches,
            report.num_inliers, report.num_new, len(state.tracked_ids))

    def _masked(self, pixels: np.ndarray) -> np.ndarray:
        """True for pixels (OpenCV convention) that lie in the ignore mask or outside the image."""
        if self.image_mask is None:
            return np.zeros(len(pixels), dtype=bool)
        height, width = self.image_mask.shape
        cols = np.round(np.asarray(pixels)[:, 0]).astype(int)
        rows = np.round(np.asarray(pixels)[:, 1]).astype(int)
        outside = (cols < 0) | (rows < 0) | (cols >= width) | (rows >= height)
        masked = outside.copy()
        masked[~outside] = self.image_mask[rows[~outside], cols[~outside]]
        return masked

    def _to_sequence(self, sequence: Sequence, state: _State) -> Sequence:
        landmarks = Landmarks(np.asarray(state.landmark_ids, dtype=np.int64), np.asarray(state.landmark_xyz).reshape(-1, 3))
        observations = Observations(np.asarray(state.obs_frame, dtype=np.int64), np.asarray(state.obs_landmark, dtype=np.int64),
            np.asarray(state.obs_uv).reshape(-1, 2))
        metadata = {**sequence.metadata, "collect_config": _jsonable(asdict(self.config))}
        return sequence.replace(landmarks=landmarks, observations=observations, metadata=metadata)

    def _save_checkpoint(self, sequence: Sequence, state: _State, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_sequence(self._to_sequence(sequence, state), directory / "anchors.json")
        (directory / "collect_state.json").write_text(json.dumps({
            "last_frame_index": state.last_frame_index,
            "next_id": state.next_id,
            "tracked_ids": state.tracked_ids.tolist(),
            "tracked_points": state.tracked_points.tolist(),
        }))

    @staticmethod
    def _load_checkpoint(directory: Path) -> _State:
        partial = load_sequence(directory / "anchors.json")
        saved = json.loads((directory / "collect_state.json").read_text())
        return _State(
            landmark_ids=partial.landmarks.ids.tolist(),
            landmark_xyz=list(partial.landmarks.xyz),
            obs_frame=partial.observations.frame_ids.tolist(),
            obs_landmark=partial.observations.landmark_ids.tolist(),
            obs_uv=list(partial.observations.uv),
            tracked_ids=np.asarray(saved["tracked_ids"], dtype=int),
            tracked_points=np.asarray(saved["tracked_points"], dtype=float).reshape(-1, 2),
            next_id=int(saved["next_id"]),
            last_frame_index=int(saved["last_frame_index"]),
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
