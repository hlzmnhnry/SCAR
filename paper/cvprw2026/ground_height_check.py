"""Height reference of the LTC INS data, checked on the ground (W4).

While the aircraft stands still at the airfield, the INS height minus the terrain height is the
height of the INS/antenna reference point above ground. It must be plausible (about 1-3 m for the
microlight) and the same in all campaigns. Two readings of the SBG ``altitude`` are compared:

- ``undulation``: height above SBG's geoid model, i.e. ellipsoidal = altitude + undulation,
  converted to DHHN2016 with the GCG2016 quasigeoid (as ``scar.datasets.ltc`` does),
- ``legacy``: altitude used directly as DHHN2016 height (CVPRW 2026 pipeline).

Terrain heights come from a processed georaffer DSM tile (DHHN2016, 1 px/m).

    pixi run python paper/cvprw2026/ground_height_check.py /path/to/ltc_data \
        --dsm ~/georay/tiles/processed/dsm/1000
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from pyproj import Transformer

from scar.geodesy import geoid_undulation

REPO = Path(__file__).resolve().parents[2]
GEOID = REPO / "data" / "quasigeoid" / "de" / "linux" / "geotiff" / "GCG2016v2023.tif"
AIRFIELD = (375941.0, 5596606.0)  # Moenchsheide, ETRS89 / UTM 32N


def standstill_segments(t, speed, min_duration_s=20.0, threshold=0.2):
    still = speed < threshold
    segments, start = [], None
    for i, flag in enumerate(np.append(still, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if t[i - 1] - t[start] >= min_duration_s:
                segments.append((start, i))
            start = None
    return segments


def dsm_height(tiles: Path, x: float, y: float) -> tuple[float, float]:
    """Bilinear DSM height and local relief (std within 5 m) at (x, y)."""
    import rasterio

    for path in tiles.glob(f"*_32_{int(x // 1000) * 1000}_{int(y // 1000) * 1000}_*.tif"):
        with rasterio.open(path) as dataset:
            data = dataset.read(1).astype(float)
            if dataset.nodata is not None:
                data[data == dataset.nodata] = np.nan
            col, row = ~dataset.transform * (x, y)  # continuous, pixel centres at +0.5
            c, r = col - 0.5, row - 0.5
            c0, r0 = int(np.floor(c)), int(np.floor(r))
            dc, dr = c - c0, r - r0
            patch = data[r0 : r0 + 2, c0 : c0 + 2]
            height = (1 - dr) * ((1 - dc) * patch[0, 0] + dc * patch[0, 1]) + dr * ((1 - dc) * patch[1, 0] + dc * patch[1, 1])
            window = data[int(row) - 5 : int(row) + 6, int(col) - 5 : int(col) + 6]
            return float(height), float(np.nanstd(window))
    raise FileNotFoundError(f"no DSM tile for ({x:.0f}, {y:.0f}) in {tiles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ltc_data", type=Path, help="folder with the campaigns' full ins.csv")
    parser.add_argument("--dsm", type=Path, default=Path("~/georay/tiles/processed/dsm/1000").expanduser())
    args = parser.parse_args()

    to_utm = Transformer.from_crs("EPSG:4937", "EPSG:25832", always_xy=True)  # SAPOS: ETRS89
    print(f"{'campaign':20s} {'segment':>9s} {'dist':>6s} {'relief':>6s} {'undul.':>7s} {'N_GCG':>7s} "
          f"{'h(undulation)':>13s} {'h(legacy)':>9s}")
    for campaign in sorted(p for p in args.ltc_data.iterdir() if re.match(r"\d{4}-\d{2}-\d{2}-", p.name)):
        with (campaign / "ins.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        def column(name: str, rows=rows) -> np.ndarray:
            return np.asarray([row[name] for row in rows], dtype=float)

        t = column("timestamp") * 1e-6
        speed = np.linalg.norm(np.column_stack([column("velocity.x"), column("velocity.y"), column("velocity.z")]), axis=1)
        lat, lon, alt, und = column("latitude"), column("longitude"), column("altitude"), column("undulation")

        for start, end in standstill_segments(t, speed):
            x, y = to_utm.transform(np.median(lon[start:end]), np.median(lat[start:end]))
            distance = float(np.hypot(x - AIRFIELD[0], y - AIRFIELD[1]))
            if distance > 3000:
                continue
            H_ground, relief = dsm_height(args.dsm, x, y)
            N = float(geoid_undulation(np.median(lat[start:end]), np.median(lon[start:end]), str(GEOID)))
            h_alt, h_und = float(np.median(alt[start:end])), float(np.median(und[start:end]))
            above_undulation = (h_alt + h_und) - (H_ground + N)
            above_legacy = h_alt - H_ground
            print(f"{campaign.name:20s} {t[end - 1] - t[start]:7.0f} s {distance:5.0f}m {relief:5.2f}m {h_und:7.3f} {N:7.3f} "
                  f"{above_undulation:11.2f} m {above_legacy:7.2f} m")


if __name__ == "__main__":
    main()
