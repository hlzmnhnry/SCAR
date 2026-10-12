"""The cvprw2026 preset must reproduce the legacy optimization chain.

In the `repro` pixi environment (GTSAM 4.2.0, same build as the original environment) results are
identical. With newer GTSAM versions they agree within solver round-off (observed: 6e-8 in the
intrinsics, 1.2e-6 m in camera positions); the tolerances below leave a margin of ~10x.
"""

import importlib.metadata
import json
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from scar.core.se3 import rotation_angle
from scar.io import load_camera, load_extrinsics
from scar.io.legacy import legacy_ltc_sequence, load_legacy_anchors
from scar.optimize import load_preset, run_calibration

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden"
CAMPAIGNS = ["2022-02-23-11-45-35", "2022-10-19-14-14-59"]
CALIBRATION_KEYS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]

EXACT = importlib.metadata.version("gtsam").startswith("4.2")
TOLERANCE = {
    "calibration": 1e-12 if EXACT else 1e-6,
    "position_m": 1e-12 if EXACT else 1e-5,
    "rotation_deg": 1e-10 if EXACT else 1e-5,
    "landmark_m": 1e-12 if EXACT else 1e-6,
    "extrinsics": 1e-12 if EXACT else 1e-6,
}

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module", params=CAMPAIGNS)
def reproduction(request):
    campaign = request.param
    tracks = glob(str(REPO / "output" / campaign / "geofeatures" / "PT_*_superpoint_lightglue_train.json"))
    ins = REPO / "data" / campaign / "ins.csv"
    if not tracks or not ins.exists():
        pytest.skip(f"legacy anchors or INS data of {campaign} not available")

    anchors = load_legacy_anchors(tracks[0], tracks[0].replace("/PT_", "/WP_"))
    sequence = legacy_ltc_sequence(anchors, ins, load_camera(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_camera.json"),
        load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json"))

    result = run_calibration(sequence, load_preset("cvprw2026"))
    golden = json.loads((GOLDEN / f"ltc_{campaign}_train.json").read_text())
    arrays = np.load(GOLDEN / f"ltc_{campaign}_train.npz")
    return result, golden, arrays


def test_every_stage_reproduces_the_intrinsics(reproduction):
    result, golden, _ = reproduction
    assert [stage.name for stage in result.stages] == [name for name in golden["stages"] if name != "ls"]
    for stage in result.stages:
        camera = stage.state.camera
        ours = [camera.fx, camera.fy, camera.cx, camera.cy, *camera.distortion]
        reference = [golden["stages"][stage.name]["camera_calibration"][key] for key in CALIBRATION_KEYS]
        np.testing.assert_allclose(ours, reference, rtol=0, atol=TOLERANCE["calibration"], err_msg=stage.name)


def test_final_poses_and_landmarks(reproduction):
    result, _, arrays = reproduction
    state = result.stages[-1].state

    np.testing.assert_array_equal(result.problem.timestamps_ns // 1000, arrays["full_image_indices"])
    # the legacy code renumbered landmarks 0..M-1 in id order; the problem keeps the original ids
    np.testing.assert_array_equal(arrays["full_landmark_ids"], np.arange(result.problem.num_landmarks))
    assert np.all(np.diff(result.problem.landmark_ids) > 0)

    np.testing.assert_allclose(state.T_world_camera[:, :3, 3], arrays["full_camera_poses"][:, :3, 3],
        rtol=0, atol=TOLERANCE["position_m"])
    angles = np.degrees(rotation_angle(state.T_world_camera[:, :3, :3], arrays["full_camera_poses"][:, :3, :3]))
    assert angles.max() <= TOLERANCE["rotation_deg"]
    np.testing.assert_allclose(state.landmark_xyz, arrays["full_landmarks"], rtol=0, atol=TOLERANCE["landmark_m"])


def test_recovered_extrinsics(reproduction):
    result, _, arrays = reproduction
    assert result.extrinsics.success
    np.testing.assert_allclose(result.T_body_camera, arrays["T_ins_cam"], rtol=0, atol=TOLERANCE["extrinsics"])
