"""Visual localization metric of the paper on the legacy VL anchors (RoMa, one frame every 10).

Compares the Kalibr calibration, the legacy (golden) SCAR calibration and the ablation variants V0
and V4, each evaluated with its own pose processing.

    pixi run python paper/cvprw2026/evaluate_localization.py 2022-10-19-14-14-59
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np

from scar.core.camera import CameraModel
from scar.evaluate import localize
from scar.io import load_camera, load_extrinsics
from scar.io.legacy import load_legacy_anchors, sequence_from_anchors

from ablation import VARIANTS, PoseSource, with_priors, world_frame  # noqa: E402 (same directory)

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign")
    args = parser.parse_args()
    campaign = args.campaign

    tracks = glob(str(REPO / "output" / campaign / "geofeatures" / "PT_*_roma_vl.json"))[0]
    anchors = load_legacy_anchors(tracks, tracks.replace("/PT_", "/WP_"))
    source = PoseSource(REPO / "data" / campaign / "ins.csv")
    kalibr_camera = load_camera(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_camera.json")
    kalibr_T = load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json").T_body_camera

    golden = json.loads((REPO / "tests/regression/golden" / f"ltc_{campaign}_train.json").read_text())
    c = golden["stages"]["full"]["camera_calibration"]
    golden_camera = CameraModel(1600, 1100, c["fx"], c["fy"], c["cx"], c["cy"], (c["k1"], c["k2"], c["p1"], c["p2"]))

    def ablation_calibration(variant: str):
        data = json.loads((REPO / "output" / "ablation" / campaign / f"{variant}.json").read_text())
        fx, fy, cx, cy, *dist = data["intrinsics"]
        return CameraModel(1600, 1100, fx, fy, cx, cy, tuple(dist)), np.asarray(data["T_body_camera"])

    cases = [("Kalibr", "V0_legacy", kalibr_camera, kalibr_T), ("legacy SCAR", "V0_legacy", golden_camera, np.asarray(golden["T_ins_cam"])),
        ("V0 legacy", "V0_legacy", *ablation_calibration("V0_legacy")),
        ("V4 corrected", "V4_interpolation", *ablation_calibration("V4_interpolation"))]

    base = sequence_from_anchors(anchors, None, kalibr_camera, load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json"),
        world_frame(False), priors=source.priors(VARIANTS["V0_legacy"], anchors.image_timestamps_ns))
    print(f"{campaign}: {len(base.frames)} frames, {len(base.observations)} correspondences")
    for name, variant, camera, T in cases:
        variant_spec = VARIANTS[variant]
        sequence = with_priors(base, source.priors(variant_spec, anchors.image_timestamps_ns),
            world_frame(variant_spec.get("convergence", False)))
        summary = localize(sequence, camera, T).summary()
        accuracy = summary["accuracy"]
        print(f"  {name:13s} rot {summary['median_rotation_error_deg']:.3f} deg, trans {summary['median_translation_error_m']:.2f} m | "
              f"2m/2deg {accuracy['2.0m/2.0deg']:.3f}, 5m/5deg {accuracy['5.0m/5.0deg']:.3f}, 10m/10deg {accuracy['10.0m/10.0deg']:.3f} "
              f"({summary['localized']}/{summary['frames']})")


if __name__ == "__main__":
    main()
