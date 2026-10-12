"""The legacy importers must reproduce what the legacy pipeline computed, bit for bit."""

from pathlib import Path

import numpy as np
import pytest

from scar.io import load_extrinsics
from scar.io.legacy import legacy_ltc_body_poses, read_legacy_ins

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden"
CAMPAIGNS = ["2022-02-23-11-45-35", "2022-10-19-14-14-59"]

pytestmark = pytest.mark.regression


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_legacy_ltc_camera_poses_match_pygeon(campaign):
    ins_path = REPO / "data" / campaign / "ins.csv"
    if not ins_path.exists():
        pytest.skip(f"{ins_path} not available")

    golden = np.load(GOLDEN / f"ltc_{campaign}_train_input_poses.npz")
    extrinsics = load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json")

    T_world_camera = legacy_ltc_body_poses(read_legacy_ins(ins_path), golden["timestamps_us"]) @ extrinsics.T_body_camera
    np.testing.assert_array_equal(T_world_camera, golden["T_world_camera"])
