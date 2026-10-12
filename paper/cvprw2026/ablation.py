"""Ablation of the pose processing corrections (W1, W2, W4, W5) on the LTC campaigns.

Anchors are fixed (collected with ``--mode default``); only the pose priors derived from the INS
data change between variants:

    V0  legacy: float32 lat/lon, no meridian convergence, INS altitude as height, nearest sample
    V1  + float64 (W1)
    V2  + meridian convergence (W2)
    V3  + heights: altitude + undulation -> DHHN2016 via GCG2016 (W4)
    V4  + linear/slerp interpolation (W5)
    V5  V4 poses, preset 'joint': extrinsics estimated in the factor graph (W9)

For each campaign and variant: cvprw2026 optimization on the train anchors, reprojection of the test
anchors with the same pose processing, extrinsics relative to Kalibr, and the translation/time
offset between INS and reference data on the test sequence.

    pixi run python paper/cvprw2026/ablation.py --campaigns 2022-10-19-14-14-59 --output output/ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scar.core.types import Frame, PosePrior, Sequence
from scar.datasets.ltc import read_ins_pose_table
from scar.evaluate import ReprojectionStatistics, estimate_pose_offset, reprojection_errors
from scar.geodesy import HeightReference, WorldFrame
from scar.io import load_extrinsics, load_sequence
from scar.io.legacy import legacy_ltc_body_poses, read_legacy_ins
from scar.optimize import load_preset, run_calibration

REPO = Path(__file__).resolve().parents[2]
GEOID = REPO / "data" / "quasigeoid" / "de" / "linux" / "geotiff" / "GCG2016v2023.tif"
CAMPAIGNS = ["2022-02-23-11-45-35", "2022-02-23-15-53-20", "2022-06-10-10-42-22", "2022-06-14-10-16-48",
    "2022-10-19-14-14-59", "2024-01-29-12-31-01"]
VARIANTS = {
    "V0_legacy": {"legacy": True},
    "V1_float64": {"convergence": False, "height": "world", "interpolation": "nearest"},
    "V2_convergence": {"convergence": True, "height": "world", "interpolation": "nearest"},
    "V3_height": {"convergence": True, "height": "undulation", "interpolation": "nearest"},
    "V4_interpolation": {"convergence": True, "height": "undulation", "interpolation": "linear"},
    "V5_joint": {"convergence": True, "height": "undulation", "interpolation": "linear", "preset": "joint"},
}


def world_frame(convergence: bool) -> WorldFrame:
    return WorldFrame(crs="EPSG:25832", height=HeightReference.geoid(str(GEOID), "DHHN2016"),
        apply_meridian_convergence=convergence)


class PoseSource:
    """Pose priors of one variant for arbitrary image timestamps (tables are cached)."""

    def __init__(self, ins_csv: Path) -> None:
        self.ins_csv = ins_csv
        self._legacy = None
        self._tables = {}

    def priors(self, variant: dict, timestamps_ns: np.ndarray) -> list[PosePrior]:
        if variant.get("legacy"):
            if self._legacy is None:
                self._legacy = read_legacy_ins(self.ins_csv)
            return [PosePrior(T) for T in legacy_ltc_body_poses(self._legacy, np.asarray(timestamps_ns) // 1000)]
        key = (variant["convergence"], variant["height"])
        if key not in self._tables:
            self._tables[key] = read_ins_pose_table(self.ins_csv, world_frame(variant["convergence"]), variant["height"])
        return [self._tables[key].prior_at(int(t), mode=variant["interpolation"], max_gap_ns=50_000_000)
            for t in timestamps_ns]


def with_priors(sequence: Sequence, priors: list[PosePrior], world: WorldFrame) -> Sequence:
    frames = tuple(Frame(frame_id=f.frame_id, timestamp_ns=f.timestamp_ns, image_path=f.image_path, pose_prior=p,
        metadata=f.metadata) for f, p in zip(sequence.frames, priors, strict=True))
    return sequence.replace(frames=frames, world_frame=world)


def body_rotvec_deg(T: np.ndarray, reference: np.ndarray) -> list[float]:
    """Rotation from ``reference`` to ``T`` about the body axes (forward, right, down) [deg]."""
    return np.degrees(Rotation.from_matrix(T[:3, :3] @ reference[:3, :3].T).as_rotvec()).tolist()


def run(campaign: str, variant_name: str, sources: dict, output: Path) -> dict:
    variant = VARIANTS[variant_name]
    world = world_frame(variant.get("convergence", False))
    anchors = {part: load_sequence(REPO / "output" / "scar" / campaign / f"{part}_default" / "anchors.json")
        for part in ("train", "test")}
    source = sources.setdefault(campaign, PoseSource(REPO / "data" / campaign / "ins.csv"))
    sequences = {part: with_priors(seq, source.priors(variant, np.asarray([f.timestamp_ns for f in seq.frames])), world)
        for part, seq in anchors.items()}

    result = run_calibration(sequences["train"], load_preset(variant.get("preset", "cvprw2026")))
    camera, T = result.camera, result.T_body_camera
    test = ReprojectionStatistics.of(reprojection_errors(sequences["test"], camera, T))
    offset = estimate_pose_offset(sequences["test"], camera, T)
    offset_rot = estimate_pose_offset(sequences["test"], camera, T, estimate_rotation=True)

    kalibr = load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json").T_body_camera
    summary = {
        "campaign": campaign, "variant": variant_name,
        "intrinsics": [camera.fx, camera.fy, camera.cx, camera.cy, *camera.distortion],
        "T_body_camera": T.tolist(),
        "extrinsics_std_deg": None if result.stages[-1].extrinsics_covariance is None else
            np.degrees(np.sqrt(np.diag(result.stages[-1].extrinsics_covariance)[:3])).tolist(),
        "intrinsics_std": None if result.stages[-1].calibration_covariance is None else
            np.sqrt(np.diag(result.stages[-1].calibration_covariance)).tolist(),
        "extrinsics_vs_kalibr_deg": body_rotvec_deg(T, kalibr),
        "test_reprojection": {"median": test.median, "mad": test.mad, "mean": test.mean, "count": test.count},
        "offset": {"delta_m": offset.delta.tolist(), "delta_std_m": offset.delta_std.tolist(), "dt_s": offset.dt,
            "dt_std_s": offset.dt_std, "median_after": offset.after.median},
        "offset_with_rotation": {"delta_m": offset_rot.delta.tolist(), "delta_std_m": offset_rot.delta_std.tolist(),
            "dt_s": offset_rot.dt, "dt_std_s": offset_rot.dt_std, "omega_deg": offset_rot.omega_deg.tolist(),
            "omega_std_deg": offset_rot.omega_std_deg.tolist(), "median_after": offset_rot.after.median},
        "train": {"frames": result.problem.num_frames, "anchors": result.problem.num_landmarks,
            "observations": result.problem.num_observations},
    }
    path = output / campaign / f"{variant_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaigns", nargs="+", default=CAMPAIGNS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output", type=Path, default=REPO / "output" / "ablation")
    args = parser.parse_args()

    sources: dict = {}
    for campaign in args.campaigns:
        for variant in args.variants:
            s = run(campaign, variant, sources, args.output)
            ev = s["extrinsics_vs_kalibr_deg"]
            print(f"{campaign} {variant:17s} test {s['test_reprojection']['median']:6.3f} px | "
                  f"vs Kalibr [{ev[0]:+.3f} {ev[1]:+.3f} {ev[2]:+.3f}] deg | offset "
                  f"{np.round(s['offset']['delta_m'], 3).tolist()} m, dt {1000 * s['offset']['dt_s']:+.1f} ms | with rotation "
                  f"{np.round(s['offset_with_rotation']['delta_m'], 3).tolist()} "
                  f"± {np.round(s['offset_with_rotation']['delta_std_m'], 3).tolist()} m, "
                  f"omega {np.round(s['offset_with_rotation']['omega_deg'], 3).tolist()} deg", flush=True)


if __name__ == "__main__":
    main()
