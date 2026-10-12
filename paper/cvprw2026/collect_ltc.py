"""Collect anchors for an LTC sequence of the CVPRW 2026 evaluation with the scar library.

Uses the time ranges and collection settings of campaigns.yaml (from the legacy configuration), the
Kalibr initial calibration and legacy LTC poses. ``--mode legacy`` reproduces the legacy collection settings
(reference keypoints without half-pixel offset, doubled feature budget), ``--mode default`` uses
the corrected ones.

    python paper/cvprw2026/collect_ltc.py 2022-10-19-14-14-59 train --mode legacy \
        --images data/2022-10-19-14-14-59/images --output output/scar/2022-10-19-14-14-59/train_legacy
"""

from __future__ import annotations

import argparse
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path

from scar.collect import AnchorCollector, CollectConfig, ImageMatchingModelsMatcher
from scar.datasets.ltc import image_reader, ltc_sequence, ltc_world_frame
from scar.io import load_camera, load_extrinsics
from scar.references import GeorayReferenceConfig, GeorayReferenceProvider

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign")
    parser.add_argument("part", choices=["train", "test"])
    parser.add_argument("--mode", choices=["legacy", "default"], default="legacy")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--tiles", default=str(Path("~/georay/tiles").expanduser()))
    parser.add_argument("--matcher", default="superpoint-lightglue")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    campaigns = yaml.safe_load((Path(__file__).parent / "campaigns.yaml").read_text())
    start_us, end_us = (int(t) for t in campaigns["campaigns"][args.campaign][args.part])
    base = campaigns["collection"]

    sequence = ltc_sequence(args.data / args.campaign, load_camera(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_camera.json"),
        load_extrinsics(REPO / "paper" / "cvprw2026" / "calibration" / "kalibr_extrinsics.json"), start_us=start_us,
        end_us=end_us, image_step=int(campaigns["image_step"]), poses="legacy",
        world_frame=ltc_world_frame(apply_meridian_convergence=False))
    if args.max_frames:
        sequence = sequence.replace(frames=sequence.frames[: args.max_frames])

    campaign_date = datetime.strptime(args.campaign[:10], "%Y-%m-%d").date()
    references = GeorayReferenceProvider(sequence.world_frame, GeorayReferenceConfig(
        regions=("nrw", "rlp"), imagery_from=(2010, None), tile_cache_dir=args.tiles, resolution="match_image",
        bbox_transform="yaw_enclosing", ranking_order="SEASON_YEAR_STATE", preferred_date=campaign_date,
        preferred_regions=("nrw",)))

    tracking, matching = base["tracking"], base["image_matching"]
    config = CollectConfig(
        collect_every_nth_frame=matching["collect_new_geofeatures_every_nth_image"],
        min_detections=matching["minimum_detections"], min_inliers=matching["minimum_inliers"],
        homography_threshold_px=matching["hom_magsac_reprojection_threshold"],
        slope_window=matching["slope_window"], slope_threshold_m=matching["slope_threshold"],
        min_confidence=matching["minimum_confidence"], max_features=tracking["maximum_features"],
        min_pixel_distance=tracking["minimum_pixel_distance"],
        reference_pixel_offset=0.0 if args.mode == "legacy" else 0.5,
        legacy_feature_budget=args.mode == "legacy",
    )

    collector = AnchorCollector(config, ImageMatchingModelsMatcher(args.matcher), references,
        image_reader(args.images))
    start = time.time()
    result = collector.collect(sequence, checkpoint_directory=args.output, resume=args.resume)
    print(f"{len(result.frames)} frames, {len(result.landmarks)} anchors, {len(result.observations)} observations "
          f"in {time.time() - start:.0f} s -> {args.output}/anchors.json")


if __name__ == "__main__":
    main()
