import numpy as np
import pytest
from scipy.spatial import KDTree

from scar.collect import local_relief, select_new_anchors, slope_mask
from scar.references import ReferenceView, round_125


def legacy_filter_by_min_distance(kp0, kp1, matches, matches_confidence, existing_points, d_min, max_features,
    minimum_confidence=0.0):
    """Copy of scripts/collect_data.py::filter_by_min_distance at tag v1.0.0-cvprw2026 (prints removed)."""
    num_existing = existing_points.shape[0]
    if max_features is not None and num_existing >= max_features:
        return np.empty((0, 2)), np.empty((0, 2))
    valid_idx = np.where(matches_confidence >= minimum_confidence)[0]
    if valid_idx.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    tree = KDTree(existing_points) if num_existing > 0 else None
    idx_sorted = valid_idx[np.argsort(-matches_confidence[valid_idx])]
    new_kp0, new_kp1 = [], []
    for idx in idx_sorted:
        if max_features is not None and (num_existing + len(new_kp0)) >= max_features:
            break
        pt = kp1[matches[idx]]
        if tree is not None:
            dist, _ = tree.query(pt, k=1)
            if dist < d_min:
                continue
        new_kp0.append(kp0[idx])
        new_kp1.append(pt)
        num_existing += 1
        existing_points = np.vstack([existing_points, pt.reshape(1, 2)])
        tree = KDTree(existing_points)
    return np.array(new_kp0), np.array(new_kp1)


@pytest.mark.parametrize("seed", range(4))
def test_anchor_selection_matches_legacy_with_legacy_budget(seed):
    rng = np.random.default_rng(seed)
    image_points = rng.uniform(0, 600, (400, 2))
    reference_points = rng.uniform(0, 600, (400, 2))
    confidence = rng.uniform(0, 1, 400)
    existing = rng.uniform(0, 600, (30, 2))

    _, expected = legacy_filter_by_min_distance(reference_points, image_points, np.arange(400), confidence, existing,
        d_min=30.0, max_features=120, minimum_confidence=0.3)
    chosen = select_new_anchors(image_points, confidence, existing, 30.0, 120, 0.3, legacy_budget=True)
    np.testing.assert_array_equal(image_points[chosen].reshape(-1, 2), expected.reshape(-1, 2))


def test_anchor_selection_budget():
    points = np.column_stack([np.arange(100) * 100.0, np.zeros(100)])
    confidence = np.ones(100)
    assert len(select_new_anchors(points, confidence, np.zeros((0, 2)), 30.0, max_features=40)) == 40
    assert len(select_new_anchors(points, confidence, np.zeros((0, 2)), 30.0, max_features=40, legacy_budget=True)) == 20


def test_slope_mask_rejects_relief_and_missing_heights():
    dsm = np.full((60, 60), 100.0)
    dsm[10:14, 10:14] = 120.0  # a building
    dsm[40:45, 40:45] = np.nan  # a gap
    points = np.array([[12.0, 12.0], [20.0, 12.0], [30.0, 30.0], [46.0, 42.0], [80.0, 5.0]])  # (46, 42): gap within the window
    np.testing.assert_array_equal(slope_mask(dsm, points, window=5, threshold_m=5.0), [False, True, True, False, False])
    assert np.isinf(local_relief(dsm, 5)[42, 42])


def test_reference_view_heights_and_mapping():
    dsm = np.tile(np.arange(4, dtype=float) + 0.5, (3, 1))  # height = x of the pixel centre
    view = ReferenceView(image=np.zeros((3, 4, 3), np.uint8), dsm=dsm,
        pixel_to_world=np.array([[2.0, 0.0, 100.0], [0.0, -2.0, 50.0], [0.0, 0.0, 1.0]]))
    pixels = np.array([[1.5, 1.5], [2.2, 0.5], [9.0, 1.0]])
    np.testing.assert_allclose(view.heights(pixels, "nearest")[:2], [1.5, 2.5])
    np.testing.assert_allclose(view.heights(pixels, "bilinear")[:2], [1.5, 2.2])
    assert np.isnan(view.heights(pixels)[2])
    np.testing.assert_allclose(view.pixels_to_world(pixels[:1]), [[103.0, 47.0]])
    np.testing.assert_allclose(view.world_to_pixels(view.pixels_to_world(pixels)), pixels)
    assert view.ground_sampling_distance == pytest.approx(2.0)


def test_round_125():
    assert [round_125(x) for x in (0.7, 1.4, 2.6, 3.6, 8.0)] == [0.5, 1.0, 2.0, 5.0, 10.0]
