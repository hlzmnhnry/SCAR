# Reproducing the CVPRW 2026 results with the scar library

Reference: tag `v1.0.0-cvprw2026` (c99fed6) with pygeon `3c5aae8`; golden results in
`tests/regression/golden` (see its README).

## Optimization (from the legacy anchors)

`run_calibration(sequence, load_preset("cvprw2026"))` on the legacy anchors and legacy INS poses:

| Environment | Difference to the legacy results |
| --- | --- |
| `pixi -e repro` (GTSAM 4.2.0, build of the original environment) | identical (intrinsics, poses, landmarks), extrinsics 1e-15 |
| `pixi` default (GTSAM 4.3.0) | intrinsics ≤ 6e-8, camera positions ≤ 1.2e-6 m, extrinsics ≤ 5e-7° |

Checked by `tests/regression/test_optimization.py` for 2022-02-23-11-45-35 and 2022-10-19-14-14-59 (train).

## Full pipeline (new anchor collection)

`paper/cvprw2026/collect_ltc.py` collects anchors with the library (georay references from georaffer
tiles, SuperPoint+LightGlue), `optimize_anchors.py` runs the `cvprw2026` preset, and
`evaluate_reprojection.py` reports the reprojection error of the *legacy test anchors* under each
calibration (INS poses, median ± MAD over all observations).

Bit-identical anchors are not expected: reference tiles are selected by georay instead of pygeon,
footprints come from DSM ray marching instead of precomputed bounding boxes, and GPU matching is not
deterministic. `--mode legacy` keeps the legacy collection conventions (W10, W11).

### 2022-10-19-14-14-59, train, `--mode legacy`

| | legacy | scar | scar − legacy |
| --- | --- | --- | --- |
| anchors / observations (after filtering) | 4444 / 158507 | 3582 / 122435 | |
| fx, fy [px] | 1319.12, 1321.19 | 1319.22, 1321.44 | +0.10, +0.25 |
| cx, cy [px] | 801.40, 560.80 | 801.38, 560.71 | −0.02, −0.08 |
| extrinsics rotation / translation | | | 0.027°, 0.6 mm |
| test reprojection, median ± MAD [px] | 6.04 ± 2.43 | 6.15 ± 2.44 | +0.11 |

Kalibr (initial calibration) on the same test anchors: 10.32 ± 3.55 px.

### Summary (train sequences, `cvprw2026` preset)

| Campaign | Anchors | Calibration | fx, fy − legacy [px] | extrinsics vs. legacy | test reprojection median ± MAD [px] |
| --- | --- | --- | --- | --- | --- |
| 2022-10-19-14-14-59 | legacy (4444) | legacy | | | 6.04 ± 2.43 |
| | scar `--mode legacy` (3582) | scar | +0.10, +0.25 | 0.027°, 0.6 mm | 6.15 ± 2.44 |
| | scar `--mode default` (3554) | scar | −0.05, +0.15 | 0.026°, 0.4 mm | 6.04 ± 2.43 |
| | – | Kalibr | | | 10.32 ± 3.55 |
| 2022-02-23-11-45-35 | legacy (4830) | legacy | | | 4.38 ± 2.01 |
| | scar `--mode legacy` (2231) | scar | +0.25, +0.97 | 0.025°, 0.2 mm | 4.33 ± 2.03 |
| | – | Kalibr | | | 47.81 ± 6.74 |

Anchor counts after the preset's filtering. Test reprojection uses the legacy test anchors of each
campaign (which themselves carry the W10 half-pixel shift) and legacy INS poses.

Observation: for 2022-02-23-11-45-35 the library collects fewer new anchors per frame than the legacy
run (median 4 vs. 10; tracks are equally long, ~37 vs. 40 frames), i.e. fewer matches survive the
filters. Candidate causes: different reference tiles (georay vs. pygeon selection) and DSM gaps,
which the NaN-aware slope filter now excludes. The resulting calibration is unaffected (see above).

### Campaigns without legacy anchors (`--mode default`)

No legacy anchors of these campaigns are available, so they were collected anew (train and test,
images from the SCAR data release). The calibrations published with the LTC dataset
(`calibration/{camera,extrinsics}.json`) are the SCAR results of the paper and serve as reference.
Test reprojection uses the new test anchors of each campaign (median ± MAD over all observations).

| Campaign | anchors (train, filtered) | extrinsics vs. published | Kalibr | published | scar |
| --- | --- | --- | --- | --- | --- |
| 2022-02-23-15-53-20 | 1345 | 0.051°, 1.0 mm | 48.28 ± 6.70 | 5.32 ± 2.61 | 5.38 ± 2.68 |
| 2022-06-10-10-42-22 | 1255 | 0.053°, 0.9 mm | 48.57 ± 5.31 | 4.93 ± 2.20 | 4.92 ± 2.20 |
| 2022-06-14-10-16-48 | 2739 | 0.304°, 0.7 mm | 47.69 ± 7.37 | 7.24 ± 2.94 | 5.49 ± 2.69 |
| 2024-01-29-12-31-01 | 1711 | 0.071°, 0.9 mm | 17.22 ± 5.56 | 7.07 ± 2.80 | 6.85 ± 3.03 |

For 2022-06-14-10-16-48 the new calibration differs by 0.3° from the published one and reprojects the
test anchors clearly better (5.49 vs. 7.24 px). The test anchors come from the same pipeline as the
new calibration (W10-corrected conventions), which may favour it slightly; the size of the gap
suggests the published extrinsics of this campaign are not optimal. To be checked in P6 with
independent test data (e.g. visual localization).

## Visual localization with the library (P7)

`scar.evaluate.localize` (OpenCV PnP RANSAC + LM refinement instead of pycolmap; 4 px, 200 points and
50 inliers minimum as in the paper) on the legacy VL anchors (RoMa, 1000 frames per campaign), each
calibration with its own pose processing (`evaluate_localization.py`). Median rotation / translation
error and the share of frames within 2 m/2°, 5 m/5°, 10 m/10°:

| campaign | calibration | rotation [deg] | translation [m] | 2m/2° | 5m/5° | 10m/10° |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-02-23-11-45-35 | Kalibr | 2.417 | 2.55 | 0.000 | 0.781 | 0.933 |
| | legacy SCAR | 0.373 | 2.48 | 0.353 | 0.799 | 0.937 |
| | V4 (corrected) | 0.323 | 2.42 | 0.396 | 0.781 | 0.935 |
| 2022-10-19-14-14-59 | Kalibr | 0.732 | 5.51 | 0.082 | 0.441 | 0.787 |
| | legacy SCAR | 0.528 | 5.70 | 0.115 | 0.427 | 0.783 |
| | V4 (corrected) | 0.550 | 5.85 | 0.104 | 0.425 | 0.786 |

The magnitudes match the paper (rotation error reduced from about 2.4° to 0.3–0.6°, translation error
essentially unchanged).
