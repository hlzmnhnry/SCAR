"""Condense legacy SCAR stage results into compact golden references.

The legacy pipeline writes one ~40 MB JSON per optimization stage. For
regression tests we only keep what a re-implementation has to reproduce:
the calibration after every stage and the final camera poses, landmarks and
recovered extrinsics.

Usage:
    python tests/regression/make_golden.py <result_dir> <name> [--note TEXT]

Writes ``tests/regression/golden/<name>.npz`` and ``<name>.json``.
"""

import argparse
import json
from glob import glob
from os.path import basename, dirname, join

import numpy as np

STAGES = ["camera_pose_rotation", "camera_pose_translation", "landmark", "calibration", "full", "ls"]
CALIBRATION_KEYS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
GOLDEN_DIR = join(dirname(__file__), "golden")


def latest_stage_file(result_dir: str, stage: str) -> str:
    # stage results are named '<stage>_<YYYY-mm-dd_HH-MM-SS>.json'
    candidates = sorted(f for f in glob(join(result_dir, f"{stage}_*.json"))
                        if basename(f)[len(stage) + 1].isdigit())
    if not candidates:
        raise FileNotFoundError(f"No result for stage '{stage}' in {result_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir")
    parser.add_argument("name")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    summary = {"name": args.name, "note": args.note, "stages": {}}
    arrays = {}

    for stage in STAGES:
        path = latest_stage_file(args.result_dir, stage)
        with open(path) as f:
            result = json.load(f)

        calibration = result["camera_calibration"]
        summary["stages"][stage] = {
            "source_file": basename(path),
            "camera_calibration": {k: calibration[k] for k in CALIBRATION_KEYS},
        }

        if stage in ("full", "ls"):
            image_indices = result["image_indices"]
            landmark_ids = sorted(result["world_points"], key=int)
            arrays[f"{stage}_image_indices"] = np.asarray(image_indices)
            arrays[f"{stage}_camera_poses"] = np.asarray([result["camera_poses"][str(i)] for i in image_indices])
            arrays[f"{stage}_landmark_ids"] = np.asarray([int(i) for i in landmark_ids])
            arrays[f"{stage}_landmarks"] = np.asarray([result["world_points"][i] for i in landmark_ids])
            arrays[f"{stage}_calibration"] = np.asarray([calibration[k] for k in CALIBRATION_KEYS])
            summary["stages"][stage].update(num_frames=len(image_indices), num_landmarks=len(landmark_ids))

        if stage == "ls":
            arrays["T_ins_cam"] = np.asarray(result["T_ins_cam"])
            summary["T_ins_cam"] = result["T_ins_cam"]

    np.savez_compressed(join(GOLDEN_DIR, f"{args.name}.npz"), **arrays)
    with open(join(GOLDEN_DIR, f"{args.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote golden reference '{args.name}' to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
