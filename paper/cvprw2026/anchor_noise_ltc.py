"""Anchor noise of an LTC sequence by re-matching tracked observations (scar.evaluate.anchor_noise).

Re-estimates the anchors of an ``anchors.json`` collected with ``collect_ltc.py``, using the
collection settings stored in the file. By default the same reference imagery is used (precision);
``--imagery`` restricts the re-matching to other orthophoto years, so that differences between
imagery epochs enter the result. Successor of the legacy ``scripts/estimate_world_point_noise.py``.

    pixi run -e gpu python paper/cvprw2026/anchor_noise_ltc.py \
        output/scar/2022-10-19-14-14-59/train_default/anchors.json \
        --images data/2022-10-19-14-14-59/images --every 5
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from scar.collect import CollectConfig, ImageMatchingModelsMatcher
from scar.datasets.ltc import image_reader
from scar.evaluate import AnchorNoiseConfig, estimate_anchor_noise
from scar.io import load_sequence
from scar.references import GeorayReferenceConfig, GeorayReferenceProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tiles", default=str(Path("~/georay/tiles").expanduser()))
    parser.add_argument("--matcher", default="superpoint-lightglue")
    parser.add_argument("--every", type=int, default=5, help="re-match every n-th frame")
    parser.add_argument("--min-frame-gap", type=int, default=1)
    parser.add_argument("--imagery", nargs=2, type=int, metavar=("FROM", "TO"), default=None,
        help="orthophoto years for re-matching (default: as in the collection)")
    parser.add_argument("--output", type=Path, default=None, help="default: anchor_noise.json next to the anchors")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    anchors = load_sequence(args.anchors)
    collect_config = CollectConfig.from_dict(anchors.metadata["collect_config"])
    campaign_date = date.fromisoformat(str(anchors.metadata["campaign"])[:10])

    references = GeorayReferenceProvider(anchors.world_frame, GeorayReferenceConfig(
        regions=("nrw", "rlp"), imagery_from=tuple(args.imagery) if args.imagery else (2010, None),
        tile_cache_dir=args.tiles, resolution="match_image", bbox_transform="yaw_enclosing",
        ranking_order="SEASON_YEAR_STATE", preferred_date=campaign_date, preferred_regions=("nrw",)))

    result = estimate_anchor_noise(anchors, ImageMatchingModelsMatcher(args.matcher), references,
        image_reader(args.images), collect_config,
        AnchorNoiseConfig(every_nth_frame=args.every, min_frame_gap=args.min_frame_gap))

    output = args.output or args.anchors.with_name("anchor_noise.json")
    result.save(output)
    summary = result.summary()
    print(json.dumps({key: np.round(value, 3).tolist() if isinstance(value, (list, float)) else value
        for key, value in summary.items()}, indent=1))
    print(f"-> {output}")


if __name__ == "__main__":
    main()
