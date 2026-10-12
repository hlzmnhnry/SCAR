"""The whole pipeline through the CLI on the synthetic custom-region demo (offline, CPU)."""

import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("rasterio")
pytest.importorskip("georay")

from scar.cli import main
from scar.io import load_camera, load_extrinsics

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "custom_region_geotiff"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("make_demo_data", EXAMPLE / "make_demo_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    directory = tmp_path_factory.mktemp("demo_run")
    module = load_demo_module()
    module.make_demo_data(directory / "demo", num_images=24)
    shutil.copyfile(EXAMPLE / "config.yaml", directory / "config.yaml")
    return module, directory


def test_check_run_and_export(demo, capsys):
    module, directory = demo
    config = directory / "config.yaml"

    assert main(["check", str(config)]) == 0
    assert "24 frames" in capsys.readouterr().out

    assert main(["run", str(config)]) == 0
    calibration = directory / "runs" / "demo" / "calibration" / "calibration.yaml"
    errors = module.compare(directory / "demo", calibration)
    assert errors["initial_extrinsic_error_deg"] > 1.4
    assert errors["extrinsic_error_deg"] < 0.05
    assert np.all(np.abs(errors["focal_error_px"]) < 1.0)

    assert main(["evaluate", str(directory / "runs" / "demo" / "anchors.json"), "--calibration", str(calibration),
        "--offsets"]) == 0
    assert "reprojection" in capsys.readouterr().out

    kalibr = directory / "camchain.yaml"
    assert main(["export", str(calibration), "--format", "kalibr", "--output", str(kalibr)]) == 0
    assert load_camera(kalibr) == load_camera(calibration)
    np.testing.assert_allclose(load_extrinsics(kalibr).T_body_camera, load_extrinsics(calibration).T_body_camera,
        atol=1e-12)

    assert main(["export", str(calibration), "--format", "ltc", "--output", str(directory / "ltc")]) == 0
    assert load_camera(directory / "ltc" / "camera.json") == load_camera(calibration)

    opencv = directory / "camera_opencv.yaml"
    assert main(["export", str(calibration), "--format", "opencv", "--output", str(opencv)]) == 0
    storage = cv2.FileStorage(str(opencv), cv2.FILE_STORAGE_READ)
    np.testing.assert_allclose(storage.getNode("camera_matrix").mat(), load_camera(calibration).K)


def test_config_errors_are_reported(tmp_path, capsys):
    data = yaml.safe_load((EXAMPLE / "config.yaml").read_text())
    data["sequence"]["poses"]["format"]["time_unt"] = "s"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data))
    assert main(["check", str(tmp_path / "config.yaml")]) == 1
    assert "did you mean 'time_unit'" in capsys.readouterr().err


def test_init_writes_templates(tmp_path):
    assert main(["init", str(tmp_path / "flight.yaml"), "--template", "event"]) == 0
    assert "reconstructed_event_images" in (tmp_path / "flight.yaml").read_text()
    assert main(["init", str(tmp_path / "flight.yaml")]) == 1
