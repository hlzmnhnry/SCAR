"""Markdown tables from the results of ablation.py.

    pixi run python paper/cvprw2026/ablation_report.py output/ablation
"""

# ruff: noqa: RUF001  (dashes and sigma in the Markdown output are intended)
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ["V0_legacy", "V1_float64", "V2_convergence", "V3_height", "V4_interpolation", "V5_joint"]


def load(directory: Path) -> dict:
    results: dict = {}
    for path in sorted(directory.glob("*/*.json")):
        data = json.loads(path.read_text())
        results.setdefault(data["campaign"], {})[data["variant"]] = data
    return results


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = load(args.directory)
    campaigns = sorted(results)
    variants = [v for v in VARIANTS if any(v in results[c] for c in campaigns)]

    print("Test reprojection error, median [px] (relative change to V0)\n")
    rows = []
    for campaign in campaigns:
        base = results[campaign].get("V0_legacy", {}).get("test_reprojection", {}).get("median")
        row = [campaign]
        for variant in variants:
            value = results[campaign].get(variant, {}).get("test_reprojection", {}).get("median")
            row.append("–" if value is None else f"{value:.2f}" + (f" ({100 * (value / base - 1):+.0f} %)" if base and variant != "V0_legacy" else ""))
        rows.append(row)
    complete = [c for c in campaigns if all(v in results[c] for v in variants)]
    if complete:
        rows.append([f"mean ({len(complete)} campaigns)"] + [f"{np.mean([results[c][v]['test_reprojection']['median'] for c in complete]):.2f}"
            for v in variants])
    print(table(["campaign", *variants], rows))

    print("\nExtrinsics relative to Kalibr, rotation about body axes (roll, pitch, yaw) [deg]\n")
    rows = []
    for campaign in campaigns:
        row = [campaign]
        for variant in ("V1_float64", "V4_interpolation", "V5_joint"):
            value = results[campaign].get(variant, {}).get("extrinsics_vs_kalibr_deg")
            row.append("–" if value is None else ", ".join(f"{v:+.2f}" for v in value))
        rows.append(row)
    print(table(["campaign", "V1 (no convergence)", "V4 (corrected)", "V5 (joint)"], rows))

    print("\nOffsets on the test sequence with rotation correction (V4): delta E, N, U [m], dt [ms], omega [deg]\n")
    rows = []
    for campaign in campaigns:
        result = results[campaign].get("V4_interpolation")
        if result is None:
            continue
        offset = result["offset_with_rotation"]
        rows.append([campaign, ", ".join(f"{v:+.2f}" for v in offset["delta_m"]), f"{1000 * offset['dt_s']:+.1f}",
            ", ".join(f"{v:+.3f}" for v in offset["omega_deg"])])
    print(table(["campaign", "delta [m]", "dt [ms]", "omega [deg]"], rows))

    if any("V5_joint" in results[c] for c in campaigns):
        print("\nIntrinsics V4 (cvprw2026) vs. V5 (joint), with 1-sigma of V5\n")
        rows = []
        for campaign in campaigns:
            if "V5_joint" not in results[campaign] or "V4_interpolation" not in results[campaign]:
                continue
            a, b = results[campaign]["V4_interpolation"]["intrinsics"], results[campaign]["V5_joint"]["intrinsics"]
            std = results[campaign]["V5_joint"]["intrinsics_std"]
            std = [std[i] for i in (0, 1, 3, 4)] if std else [float("nan")] * 4
            ext = results[campaign]["V5_joint"]["extrinsics_std_deg"] or [float("nan")] * 3
            rows.append([campaign, f"{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}, {a[3]:.2f}",
                f"{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}", ", ".join(f"{s:.2f}" for s in std),
                ", ".join(f"{s:.3f}" for s in ext)])
        print(table(["campaign", "V4 fx, fy, cx, cy", "V5 fx, fy, cx, cy", "V5 σ [px]", "V5 σ extrinsics [deg]"], rows))


if __name__ == "__main__":
    main()
