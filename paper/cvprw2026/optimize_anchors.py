"""Optimize collected anchors with a preset and compare with the CVPRW 2026 reference result.

    pixi run python paper/cvprw2026/optimize_anchors.py output/scar/2022-10-19-14-14-59/train_legacy/anchors.json \
        --golden tests/regression/golden/ltc_2022-10-19-14-14-59_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scar.core.se3 import rotation_angle
from scar.io import load_sequence
from scar.io.results import save_result
from scar.optimize import load_preset, run_calibration

KEYS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]


def reprojection_errors(result) -> np.ndarray:
    problem, state = result.problem, result.stages[-1].state
    errors = np.empty(problem.num_observations)
    for frame in range(problem.num_frames):
        rows = np.flatnonzero(problem.observation_frame == frame)
        projected = state.camera.project(state.landmark_xyz[problem.observation_landmark[rows]], state.T_world_camera[frame])
        errors[rows] = np.linalg.norm(projected - problem.observation_uv[rows], axis=1)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--preset", default="cvprw2026")
    parser.add_argument("--golden", type=Path, default=None, help="tests/regression/golden/<name> (without suffix)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sequence = load_sequence(args.anchors)
    result = run_calibration(sequence, load_preset(args.preset))
    output = args.output or args.anchors.parent / f"calibration_{args.preset}"
    save_result(result, output, preset_name=args.preset)

    camera = result.camera
    errors = reprojection_errors(result)
    summary = {
        "frames": result.problem.num_frames, "landmarks": result.problem.num_landmarks,
        "observations": result.problem.num_observations,
        "intrinsics": dict(zip(KEYS, [camera.fx, camera.fy, camera.cx, camera.cy, *camera.distortion], strict=True)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_mad_px": float(np.median(np.abs(errors - np.median(errors)))),
        "extrinsics_rotation_change_deg": float(np.degrees(rotation_angle(result.T_body_camera[:3, :3],
            sequence.extrinsics.T_body_camera[:3, :3]))),
    }
    if args.golden:
        golden = json.loads(args.golden.with_suffix(".json").read_text())
        reference = golden["stages"]["full"]["camera_calibration"]
        T_golden = np.asarray(golden["T_ins_cam"])
        summary["golden_difference"] = {
            **{key: summary["intrinsics"][key] - reference[key] for key in KEYS},
            "extrinsics_rotation_deg": float(np.degrees(rotation_angle(result.T_body_camera[:3, :3], T_golden[:3, :3]))),
            "extrinsics_translation_m": float(np.linalg.norm(result.T_body_camera[:3, 3] - T_golden[:3, 3])),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
