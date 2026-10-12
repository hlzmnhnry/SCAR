"""Writing calibration results to a run directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scar.optimize.runner import CalibrationResult
from scar.optimize.stages import CALIBRATION_PARAMETERS

from .calibration import save_calibration


def save_result(result: CalibrationResult, directory: str | Path, preset_name: str | None = None) -> Path:
    """Write ``calibration.yaml``, ``stages.json`` and ``estimates.npz`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    stages = [
        {
            "name": stage.name,
            "intrinsics": [stage.state.camera.fx, stage.state.camera.fy, stage.state.camera.cx, stage.state.camera.cy],
            "distortion": list(stage.state.camera.distortion),
            "skew": stage.state.skew,
            "initial_error": stage.report.initial_error,
            "final_error": stage.report.final_error,
            "iterations": stage.report.iterations,
        }
        for stage in result.stages
    ]

    metadata = {
        "preset": preset_name,
        "frames": result.problem.num_frames,
        "landmarks": result.problem.num_landmarks,
        "observations": result.problem.num_observations,
        "final_error": result.stages[-1].report.final_error,
    }
    covariance = result.stages[-1].calibration_covariance
    if covariance is not None:
        metadata["intrinsics_covariance"] = {"order": list(CALIBRATION_PARAMETERS), "matrix": covariance.tolist()}
    if result.extrinsics is not None:
        residuals = result.extrinsics.residuals
        metadata["extrinsics_fit"] = {
            "method": "least_squares",
            "rms_translation_m": float(np.sqrt(np.mean(np.sum(residuals[:, :3] ** 2, axis=1)))),
            "rms_rotation_deg": float(np.degrees(np.sqrt(np.mean(np.sum(residuals[:, 3:] ** 2, axis=1))))),
        }

    save_calibration(directory / "calibration.yaml", camera=result.camera, extrinsics=result.calibrated_extrinsics,
        metadata=metadata)
    (directory / "stages.json").write_text(json.dumps(stages, indent=2))

    state = result.stages[-1].state
    np.savez_compressed(directory / "estimates.npz", frame_ids=result.problem.frame_ids,
        timestamps_ns=result.problem.timestamps_ns, T_world_camera=state.T_world_camera,
        landmark_ids=result.problem.landmark_ids, landmark_xyz=state.landmark_xyz)
    return directory
