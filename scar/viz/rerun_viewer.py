"""Live visualization of collection and optimization with rerun (https://rerun.io)."""

from __future__ import annotations

import colorsys

import numpy as np

from scar.collect.collector import FrameReport


def _colors(ids: np.ndarray) -> np.ndarray:
    """Stable colour per anchor id."""
    hues = (np.asarray(ids, dtype=np.int64) * 0.61803398875) % 1.0
    return np.asarray([[int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.8, 1.0)] for h in hues], dtype=np.uint8)


class RerunViewer:
    """Streams :class:`~scar.collect.collector.FrameReport` s and optimization results to rerun.

    3D content is shifted by the first camera position (float32 precision in the viewer).
    """

    def __init__(self, application_id: str = "scar", spawn: bool = True, save: str | None = None) -> None:
        import rerun as rr

        self.rr = rr
        rr.init(application_id, spawn=spawn and save is None)
        if save is not None:
            if spawn and hasattr(rr, "set_sinks"):
                rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(save)))
            else:
                rr.save(str(save))
        self._offset: np.ndarray | None = None
        self._landmarks: dict[int, np.ndarray] = {}
        self._trajectory: list[np.ndarray] = []

    def _local(self, points: np.ndarray) -> np.ndarray:
        if self._offset is None:
            self._offset = np.asarray(points, dtype=float).reshape(-1, 3)[0].copy()
        return (np.asarray(points, dtype=float) - self._offset).astype(np.float32)

    def on_frame(self, report: FrameReport, T_world_camera: np.ndarray | None = None, camera=None) -> None:
        rr = self.rr
        frame = report.frame
        rr.set_time("frame", sequence=frame.frame_id)
        rr.set_time("time", timestamp=frame.timestamp_ns * 1e-9)

        rr.log("image", rr.Image(report.image))
        if report.tracked_points is not None and len(report.tracked_points):
            rr.log("image/anchors", rr.Points2D(report.tracked_points, colors=_colors(report.tracked_ids), radii=2.5))
        rr.log("stats/tracked", rr.Scalars(report.num_tracked))
        rr.log("stats/new_anchors", rr.Scalars(report.num_new))
        if report.num_matches:
            rr.log("stats/matches", rr.Scalars(report.num_matches))
            rr.log("stats/inliers", rr.Scalars(report.num_inliers))

        if report.reference is not None:
            rr.log("reference", rr.Image(report.reference.image))
            if report.matches is not None and len(report.matches):
                rr.log("reference/matches", rr.Points2D(report.matches.reference, colors=[255, 200, 0], radii=3.0))
            if report.new_landmark_ids is not None and len(report.new_landmark_ids):
                xy = report.reference.pixels_to_world(report.matches.reference[: len(report.new_landmark_ids)])
                heights = report.reference.heights(report.matches.reference[: len(report.new_landmark_ids)])
                for landmark_id, point in zip(report.new_landmark_ids, np.column_stack([xy, heights]), strict=True):
                    self._landmarks[int(landmark_id)] = point

        if T_world_camera is not None:
            position = self._local(T_world_camera[:3, 3][None])[0]
            self._trajectory.append(position)
            rr.log("world/camera", rr.Transform3D(translation=position, mat3x3=T_world_camera[:3, :3].astype(np.float32)))
            if camera is not None:
                rr.log("world/camera/image", rr.Pinhole(image_from_camera=camera.K, resolution=[camera.width, camera.height],
                    image_plane_distance=20.0))
            rr.log("world/trajectory", rr.LineStrips3D([np.asarray(self._trajectory)], colors=[[80, 160, 255]]))
        if self._landmarks:
            ids = np.fromiter(self._landmarks, dtype=np.int64)
            rr.log("world/anchors", rr.Points3D(self._local(np.asarray(list(self._landmarks.values()))),
                colors=_colors(ids), radii=0.8))

    def on_calibration(self, result) -> None:
        """Log stage errors and the optimized scene (after :func:`scar.optimize.run_calibration`)."""
        rr = self.rr
        for index, stage in enumerate(result.stages):
            for iteration, error in enumerate(stage.report.errors):
                rr.set_time("optimization_iteration", sequence=iteration)
                rr.log(f"optimization/{index}_{stage.name}", rr.Scalars(error))
        state = result.stages[-1].state
        rr.log("world/optimized/anchors", rr.Points3D(self._local(state.landmark_xyz), colors=[[255, 255, 255]], radii=0.6),
            static=True)
        rr.log("world/optimized/trajectory", rr.LineStrips3D([self._local(state.T_world_camera[:, :3, 3])],
            colors=[[255, 120, 60]]), static=True)
