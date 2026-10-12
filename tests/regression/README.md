# Regression references

`golden/` holds compact references of the legacy pipeline. They are used to verify that the refactored
pipeline reproduces the published behaviour (preset `cvprw2026`). Each reference contains:

- a `<name>.json` with provenance and the calibration after every stage,
- a `<name>.npz` with the final (`full`, `ls`) camera poses, landmarks and `T_ins_cam`.

They were created with `make_golden.py` from the stage results of the legacy scripts.
`*_input_poses.npz` hold the initial camera poses the legacy optimization started from
(`make_golden_input_poses.py`, pygeon `VGLIterator(frame=CAMERA)` with the Kalibr extrinsics, now `paper/cvprw2026/calibration/kalibr_extrinsics.json`).

Run the tests with `pixi run test`. Tests marked `regression` skip when the local data they need is missing.
`pixi run -e repro test` uses GTSAM 4.2.0 in the exact build of the original environment; there the
cvprw2026 preset reproduces the references bit for bit. With GTSAM 4.3 (default environment) results
differ by solver round-off (≤ 6e-8 in the intrinsics, ≤ 1.2e-6 m in camera positions).

## Provenance

| Reference | Code | Input anchors (`output/<campaign>/geofeatures/`) |
| --- | --- | --- |
| `ltc_2022-02-23-11-45-35_train` | tag `v1.0.0-cvprw2026` (c99fed6), pygeon `3c5aae8` | `*_1645614710400000_1645614910199892_10_superpoint_lightglue_train` |
| `ltc_2022-10-19-14-14-59_train` | tag `v1.0.0-cvprw2026` (c99fed6), pygeon `3c5aae8` | `*_1666183007634000_1666183207435392_10_superpoint_lightglue_train` |

The LTC references were regenerated on 2026-09-28 by running the stage chain
(`camera_pose_rotation → camera_pose_translation → landmark → calibration → full → ls`) with the configs in
`config/` and `config/<campaign>/` of the tag (now summarized in `paper/cvprw2026/campaigns.yaml`), writing to `output/<campaign>/optimization/golden_p0`. For 2022-02-23-11-45-35
the rerun is bit-identical to the results from 2026-04-10 (`optimization/basic`).

The pygeon version matters: the pygeon currently installed in the `scar` environment (eccv-rebuttal state)
expects `ins_with_pos_acc.csv` and `data/meta/corine_landcover_superclasses.json` and fails on `data/`. The
references were therefore produced with pygeon `3c5aae8` on `PYTHONPATH`
(`git -C ~/pygeon archive 3c5aae8 pygeon | tar -x -C <dir>`).

## Findings from creating the baseline

- **INS precision changed during the project.** Before cbbf66a (2026-03-10, an ancestor of c99fed6) SCAR read
  `ins.csv` with pandas (float64). From cbbf66a on it uses pygeon, which reads latitude/longitude as float32
  (≈0.42 m quantization in northing at 50.6° N). The references above include this effect.
- **Published LTC extrinsics are SCAR results.** `pygeon/data/2022-02-23-11-45-35/calibration/extrinsics.json`
  differs from this reference's `T_ins_cam` by 0.06° / 0.7 mm.
- **Configs differ from the values stated in the paper.** Rotation stage: 0.5° pose rotation prior (paper Σ_calib: 5°),
  Huber 5 (paper: 3). Landmark stage: 0.2 m XY prior (paper: σ_XY ≈ 0.25 m). The `cvprw2026` preset reproduces the
  configs, since these produced the references.
- Frozen anchors are only available locally for 2022-02-23-11-45-35 and 2022-10-19-14-14-59; the other four LTC
  campaigns need the anchors from the SCAR data release.
