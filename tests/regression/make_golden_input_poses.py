"""Record the initial camera poses the legacy pipeline derived from INS data.

Must run in the legacy environment with pygeon 3c5aae8 on PYTHONPATH (see README).
For every image timestamp of a legacy anchor file it stores the pose that
`sfm_optimization.py::load_dataset` used as prior: pygeon's
`VGLIterator(frame=CAMERA).get_by_timestamp(ts, max_dt_us=0)["pose"]`.

Usage:
    python tests/regression/make_golden_input_poses.py <data_directory> <campaign> \
        <pixel_tracks.json> <extrinsics.json> <name>
"""

import argparse
import json
from os.path import dirname, join
from pathlib import Path

import numpy as np
from pygeon.datasets.vgl_iterator import VGLIterator
from pygeon.enums import CampaignDate, FrameType

GOLDEN_DIR = join(dirname(__file__), "golden")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_directory")
    parser.add_argument("campaign")
    parser.add_argument("pixel_tracks")
    parser.add_argument("extrinsics")
    parser.add_argument("name")
    args = parser.parse_args()

    with open(args.pixel_tracks) as f:
        pixel_tracks = json.load(f)
    timestamps = sorted({int(ts) for observations in pixel_tracks.values() for ts, _ in observations})

    extrinsics = Path(args.extrinsics)
    iterator = VGLIterator(
        campaign=CampaignDate(args.campaign),
        data_directory=Path(args.data_directory),
        frame=FrameType.CAMERA,
        image_indexed_naming=True,
        with_landcover=False,
        with_image=False,
        camera_calibration_directory=extrinsics.parent,
        camera_file_basename="camera_matrix_guess_kalibr.json",
        extrinsics_calibration_directory=extrinsics.parent,
        extrinsics_file_basename=extrinsics.name,
    )

    poses = np.asarray([iterator.get_by_timestamp(ts_us=ts, nearest_only=True, max_dt_us=0)["pose"]
                        for ts in timestamps])

    np.savez_compressed(join(GOLDEN_DIR, f"{args.name}.npz"), timestamps_us=np.asarray(timestamps), T_world_camera=poses)
    print(f"Wrote {len(timestamps)} poses to {GOLDEN_DIR}/{args.name}.npz")


if __name__ == "__main__":
    main()
