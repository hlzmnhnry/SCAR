# Geodetic corrections: ablation log

Tracks how the corrections W1–W7 of the refactoring plan affect SCAR. The `cvprw2026` preset keeps the
published behaviour; defaults of the generic pipeline are chosen from the results collected here.

| ID | Correction | Legacy behaviour |
| --- | --- | --- |
| W1 | read latitude/longitude in float64 | pygeon reads them as float32; `utm` then computes partly in float32 |
| W2 | rotate orientations from true north into grid axes (meridian convergence) | not applied |
| W3 | explicit horizontal datum of the INS positions | WGS84/ETRS89 treated as identical |
| W4 | convert INS heights into the DSM's vertical datum (DHHN2016) | INS `altitude` used unchanged |
| W5 | interpolate poses (position linear, rotation slerp) | nearest INS sample |
| W6 | bilinear, pixel-centre-correct DSM lookup, nodata masking | nearest pixel, area/point mix-up in georay |
| W7 | pose prior covariance in world axes / from INS accuracies | fixed sigmas in camera axes |
| W10 | reference keypoints: pixel centres at integers (matcher) vs. area convention (reference grid) | not converted |
| W11 | feature budget counted once per new anchor | counted twice |
| W12 | project anchors in float64 during evaluation | world points passed to OpenCV as float32 |

## 1. Input level (2026-09-28)

Differences of the initial body poses to the legacy poses for the 1000 training frames of
2022-10-19-14-14-59 (`1666183007634000`–`1666183207435392`, every 10th image). Legacy poses are
reproduced bit-exactly by `scar.io.legacy`. Each row adds one correction on top of W1.

| Variant | max \|dE\| [m] | mean dN [m] | max \|dN\| [m] | mean dU [m] | mean / max dR [deg] |
| --- | --- | --- | --- | --- | --- |
| W1 float64 | 0.057 | +0.217 | 0.978 | 0.000 | 0.000 / 0.000 |
| W1 + W5 interpolation | 0.060 | +0.215 | 0.977 | 0.000 | 0.000 / 0.005 |
| W1 + W2 convergence | 0.057 | +0.217 | 0.978 | 0.000 | 1.385 / 1.406 |
| W1 + W4 `altitude + undulation` → DHHN2016 (GCG2016) | 0.057 | +0.217 | 0.978 | +0.342 | 0.000 / 0.000 |

Notes:

- W1 is larger than the float32 quantization of latitude alone (0.42 m steps), because the `utm`
  package keeps computing in float32 when given float32 scalars.
- W4 assumes the SBG `altitude` is a height above the geoid whose undulation is in the `undulation`
  column (47.7–47.8 m here; GCG2016 gives 47.48 m at the airfield). This is a hypothesis to be tested
  with ground segments (section 6).
- W3: resolved, the RTK corrections came from SAPOS, so the INS positions are ETRS89 (section 2).

## 2. Calibration level (2026-09-29, `paper/cvprw2026/ablation.py`)

Setup: for each of the six LTC campaigns, anchors of the training and test segments (1000 frames
each, collected with W10/W11 corrected) are fixed. Variants differ only in how the INS data become
pose priors; each variant calibrates on the training anchors (`cvprw2026` preset) and is evaluated by
reprojecting the test anchors with the *same* pose processing. Tables: `ablation_report.py`.

| Variant | Pose processing |
| --- | --- |
| V0 | legacy (float32, no convergence, altitude as height, nearest sample) |
| V1 | + float64 (W1) |
| V2 | + meridian convergence (W2) |
| V3 | + altitude + undulation → DHHN2016 via GCG2016 (W4) |
| V4 | + linear/slerp interpolation (W5) |
| V5 | V4 poses, preset `joint` (extrinsics in the factor graph, W9) |

Test reprojection error, median [px] (change relative to V0):

| campaign | V0 | V1 | V2 | V3 | V4 | V5 |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-02-23-11-45-35 | 4.59 | 4.48 (−2 %) | 3.64 (−21 %) | 3.66 (−20 %) | 3.66 (−20 %) | 3.72 (−19 %) |
| 2022-02-23-15-53-20 | 5.38 | 5.32 (−1 %) | 4.17 (−22 %) | 4.21 (−22 %) | 4.21 (−22 %) | 4.12 (−23 %) |
| 2022-06-10-10-42-22 | 4.92 | 4.91 (0 %) | 4.23 (−14 %) | 4.25 (−14 %) | 4.25 (−14 %) | 4.30 (−13 %) |
| 2022-06-14-10-16-48 | 5.49 | 5.53 (+1 %) | 3.70 (−33 %) | 3.74 (−32 %) | 3.74 (−32 %) | 3.66 (−33 %) |
| 2022-10-19-14-14-59 | 5.74 | 5.70 (−1 %) | 5.30 (−8 %) | 5.27 (−8 %) | 5.27 (−8 %) | 5.89 (+3 %) |
| 2024-01-29-12-31-01 | 6.85 | 7.00 (+2 %) | 5.12 (−25 %) | 5.13 (−25 %) | 5.13 (−25 %) | 5.51 (−20 %) |
| mean | 5.49 | 5.49 | **4.36** | 4.38 | 4.38 | 4.53 |

Extrinsics relative to the Kalibr target calibration (October 2022), rotation about the body axes
roll, pitch, yaw [deg]:

| campaign | V1 (no convergence) | V4 (corrected) | V5 (joint) |
| --- | --- | --- | --- |
| 2022-02-23-11-45-35 | +1.75, −0.21, +1.63 | +1.77, −0.18, +0.22 | +1.82, −0.16, +0.23 |
| 2022-02-23-15-53-20 | +1.74, −0.18, +1.64 | +1.73, −0.14, +0.23 | +1.83, −0.18, +0.23 |
| 2022-06-10-10-42-22 | +1.80, −0.15, +1.52 | +1.76, −0.10, +0.11 | +1.91, −0.21, +0.12 |
| 2022-06-14-10-16-48 | +1.72, −0.10, +1.64 | +1.73, −0.08, +0.26 | +1.87, −0.09, +0.28 |
| 2022-10-19-14-14-59 | +0.07, −0.01, +0.62 | +0.07, +0.03, −0.76 | +0.07, −0.05, −0.78 |
| 2024-01-29-12-31-01 | −0.01, −0.22, +1.44 | +0.04, −0.14, +0.06 | +0.01, −0.16, +0.08 |

Findings:

- **W2 is the dominant correction.** Rotating the (true-north) INS orientations into grid axes lowers
  the test error in every campaign, by 21 % on average (8–33 %; 6 of 6 campaigns improve, one-sided
  sign test p = 1/64 ≈ 0.016). Presumably the calibration cannot absorb the 1.37° heading mismatch
  because the pose priors of the staged schedule allow only small rotation changes and a rotation
  about the world vertical is not a rotation about the body axis once the aircraft banks or pitches.
- **The published LTC extrinsics contain the convergence angle.** Without the correction the yaw
  differs from Kalibr by about 1.6° in all campaigns of 2022-02 to 2022-06 and 2024-01; with it,
  by 0.06–0.26°. (2022-10-19, the Kalibr session, is the exception with −0.76°.)
- **Roll changes between June and October 2022.** Feb–Jun 2022 show +1.75° roll relative to Kalibr,
  October 2022 and January 2024 about 0°, consistently in all variants. The camera was re-mounted
  between the June and October 2022 campaigns (confirmed by the data owners); Kalibr was recorded
  after the re-mount. SCAR thus detects the re-mount from flight data alone.
- **W1, W4, W5 hardly change the calibration** (≤ 2 %; W5 none, as camera and INS samples almost
  coincide). W1 still matters for georeferencing (0.2–1 m positions, section 1) and W4 is the
  physically right height handling (section 6).
- **V5 (joint) does not generalize better on real data** (mean 4.53 vs. 4.38 px), although it recovers
  intrinsics and extrinsics on synthetic flights (section 3). Its extrinsics agree with V4 within about
  0.1°, but it raises fx by +1.5 to +5.7 px (≈ 0.3 %) in every campaign with a formal σ of 0.13 px,
  i.e. the campaign-to-campaign spread (±1.7 px) is far larger than the marginal covariance. Together
  with the offsets below this points to a systematic scale/height inconsistency between INS and
  reference data that the joint model resolves through the focal length.

Offsets on the test segments with calibration fixed (V4), estimating a world translation, a time
offset and a camera-frame rotation correction (`scar.evaluate.estimate_pose_offset`):

| campaign | delta E, N, U [m] | dt [ms] | omega [deg] |
| --- | --- | --- | --- |
| 2022-02-23-11-45-35 | −0.22, −0.26, −0.91 | +25.3 | −0.143, +0.039, −0.068 |
| 2022-02-23-15-53-20 | −0.11, +0.44, −1.52 | −34.4 | +0.120, +0.009, +0.016 |
| 2022-06-10-10-42-22 | −0.46, −0.51, −1.02 | −14.2 | +0.026, +0.027, −0.276 |
| 2022-06-14-10-16-48 | +0.78, −0.30, −0.82 | +2.7 | −0.003, −0.045, +0.008 |
| 2022-10-19-14-14-59 | +0.21, −0.27, −0.53 | +26.3 | −0.174, +0.139, +0.090 |
| 2024-01-29-12-31-01 | +0.25, −0.76, −0.72 | +4.1 | +0.106, +0.042, +0.070 |

- **Vertical:** in all campaigns the cameras would have to be 0.5–1.5 m lower to fit the anchors best
  (equivalently the focal length ~0.2–0.3 % larger, cf. V5). The ground check (section 6) shows no such
  offset at standstill, so the cause is in flight (e.g. GNSS/INS height, lever arm, DSM of the anchor
  points being above the imaged texture, focus). Open.
- **Horizontal (W3):** the offsets scatter by about ±0.5 m without a common direction. The ~0.8 m
  north-east shift expected if WGS84/ITRF positions were used against ETRS89 reference data is not
  visible. This is consistent with the RTK corrections: the LTC flights used SAPOS (confirmed by the
  data owners), whose positions are ETRS89 like the reference data. `scar.datasets.ltc` therefore
  declares the INS positions as ETRS89 (EPSG:4937). W3 is resolved for LTC; for other data the source
  CRS must be set according to the correction service.
- **Time offset:** not identifiable on these segments (−34 … +26 ms, no consistent sign).

## 3. Synthetic flights: what the staged schedule can recover (2026-09-28)

Simulated nadir flight (`tests/synthetic/simulation.py`): 40 frames, 600 m above undulating terrain, 0.5 px
pixel noise, anchors with σ = (0.25, 0.25, 0.1) m, navigation noise 0.05° / 2 cm, true intrinsics and
extrinsics known. Errors after the `cvprw2026` preset:

| Initial error | fx | fy | cx | cy [px] | extrinsic rotation [deg] |
| --- | --- | --- | --- | --- | --- |
| none | +0.02 | 0.00 | −0.03 | −0.01 | 0.02 |
| extrinsics (1, −0.8, 1.5)° | +0.02 | 0.00 | −0.03 | −0.01 | 1.97 → 0.02 |
| intrinsics (+3, −2, +2, −2) px | +2.37 | −0.40 | +1.71 | −1.61 | 0.00 → 0.12 |

- The schedule recovers extrinsic errors of a few degrees reliably.
- Errors of the initial intrinsics are largely *not* corrected: the landmark stage absorbs the
  resulting misfit (XY landmark prior 0.2 m), and the tight intrinsics priors of the later stages
  (fx: 5 px, then 0.2 px) keep them near the start. A principal point offset leaks into the
  extrinsic rotation (nadir coupling of cx/cy with roll/pitch).
- A single joint stage with the *true* noise models (world-axes navigation covariance, anchor
  covariance, loose intrinsics prior, extrinsics known) estimates the intrinsics to 0.1–0.3 px and its
  marginal covariance is consistent (mean NEES 7.5 for 8 parameters over 20 runs; see
  `tests/synthetic/test_recovery.py`). Estimating the extrinsics jointly in the graph (W9) is needed
  before this can replace the staged schedule on real data.

## 4. W6: DSM pixel conventions (2026-09-28)

- georaffer DSM tiles are tagged `AREA_OR_POINT=Point` with the tie point at the centre of the first
  pixel (e.g. 357000.5 / 5612999.5); GDAL/rasterio read them as covering exactly their 1 km cell. This
  matches georay's area convention (pixel (i, j) covers [i, i+1) x [j, j+1)). No shift in the data.
- georay now verifies each tile's georeferencing against its grid cell when mosaicking.
- georay's bilinear height lookup in the footprint ray marcher treated values as located at integer
  pixel coordinates, i.e. shifted by half a pixel (fixed in georay `feature/raster-sources`). It only
  affects footprints, not SCAR's anchors.
- SCAR's legacy anchor lifting (`dsm[floor(v), floor(u)]`) takes the value of the pixel containing
  the point, which is consistent with the area convention; the remaining W6 question is nearest vs.
  bilinear lookup (P4/P6).
- DSM nodata (-9999) is now NaN in georay outputs. In the legacy pipeline the slope filter rejects
  matches next to nodata (huge min-max difference), but a match inside a larger nodata area passes
  it and would be lifted to -9999 m. None of the local legacy anchor sets contains such a point
  (all heights between 29 and 197 m), so the published results are not affected.

## 5. W10/W11: anchor collection (2026-09-28)

- **W10.** SuperPoint (LightGlue implementation) and OpenCV report keypoints with pixel centres at
  integer coordinates. The reference grids of pygeon and georay map pixel `(i, j)` to the area
  `[i, i+1) x [j, j+1)`, i.e. its centre is at `(i + 0.5, j + 0.5)`. The legacy pipeline passed keypoints
  unchanged, which places anchors `0.5 / ppm` too far west and `0.5 / ppm` too far north
  (0.25 m each at 2 px/m, 0.1 m at 5 px/m) and makes the nearest-pixel DSM lookup pick a neighbour for
  half of the positions. Synthetic check (`tests/synthetic/test_collection.py`, exact rendering, 0.5 m GSD):
  mean anchor error (dE, dN) = (−0.03, +0.02) m with the offset vs. (−0.28, +0.27) m without; median
  horizontal error 0.09 m vs. 0.39 m. `CollectConfig.reference_pixel_offset` is 0.5 by default and 0.0
  for reproducing the legacy collection.
- The same convention question arises when cutouts are warped (yaw-aligned footprints, as used for
  LTC): OpenCV's `warpAffine` samples with pixel centres at integers, while the warp matrix was built
  from area-convention corner coordinates. In pygeon (legacy) this combines with the unconverted
  keypoints into exactly the grid-axis shift above, independent of the cutout rotation. georay had
  the same mismatch between warped content and its georeferencing (e.g. 0.09 m / 0.16 m for a 30°
  cutout at 2 px/m); fixed in georay `bca1b36` (`test_warped_cutout_content_matches_its_georeferencing`).
- **W11.** `filter_by_min_distance` counted every accepted anchor twice against `maximum_features`,
  so at most about half of the free budget was filled per frame. `CollectConfig.legacy_feature_budget`
  reproduces this.
- **W12.** `evaluate_reprojection.py` cast world points to float32 before `cv2.projectPoints`
  (0.5 m resolution in UTM northing). On the 2022-10-19 test anchors this changes the median
  reprojection error by about 0.02 px (6.041 vs. 6.065 px for the legacy calibration): negligible.

## 6. Ground check of the INS heights (W4, `ground_height_check.py`)

At standstill on the airfield (7 segments, 6 campaigns, parking positions up to 170 m apart) the INS
height minus the DSM height (bDOM, RLP, 1 m) is

- 0.54–0.66 m if `altitude + undulation` is taken as ellipsoidal height and converted to DHHN2016
  with GCG2016 (as `scar.datasets.ltc` does),
- 0.31–0.43 m if `altitude` is used directly as DHHN2016 height (legacy),
- about −47 m if `altitude` were ellipsoidal (ruled out).

Both remaining readings are consistent across campaigns (±0.04 m) and plausible for an INS in the
fuselage; they differ by the difference between SBG's geoid model (undulation 47.90 m) and GCG2016
(47.67 m). The height of the INS above ground was not measured (roughly 0.5 m according to the data
owners), so the data cannot decide between them. The conversion via GCG2016 is kept because the DSM
heights are DHHN2016 normal heights; the remaining uncertainty of the INS heights is about ±0.2 m.

## 7. Recommendations

- Default pose processing: float64, meridian convergence, explicit height conversion, interpolation
  (V4). Preset `default` = `cvprw2026` schedule; `joint` stays experimental until the fx/height
  inconsistency is understood.
- LTC dataset: the published extrinsics are consistent with the dataset's pose convention (no
  meridian convergence); in grid axes they differ by about 1.4° in yaw. Decision (2026-10-02): the
  published extrinsics stay unchanged; pygeon keeps the uncorrected convention as default and offers
  the correction as an option for extrinsics estimated from corrected poses.
- Separate calibrations per camera-mount configuration (before/after the re-mount between June and
  October 2022) are required; the October 2022 Kalibr calibration only applies after the re-mount.
