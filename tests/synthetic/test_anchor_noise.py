"""Anchor noise by re-matching tracked observations, on an exactly rendered synthetic scene."""

import numpy as np
import pytest

from scar.collect import AnchorCollector, CollectConfig, OpenCVSiftMatcher
from scar.core.types import Landmarks
from scar.evaluate import AnchorNoiseConfig, estimate_anchor_noise
from scar.evaluate.anchor_noise import _local_affine

from .scene import CAMERA, OrthoReferenceProvider, flight, render_aerial, texture

COLLECT = CollectConfig(collect_every_nth_frame=1, max_features=300, min_pixel_distance=15.0, min_confidence=0.0)


@pytest.fixture(scope="module")
def scene():
    ortho = texture()
    sequence = flight(num_frames=8)
    images = {frame.frame_id: render_aerial(ortho, CAMERA, frame.pose_prior.T_world_body, seed=frame.frame_id)
        for frame in sequence.frames}
    anchors = AnchorCollector(COLLECT, OpenCVSiftMatcher(), OrthoReferenceProvider(ortho),
        read_image=lambda frame: images[frame.frame_id]).collect(sequence)
    return ortho, images, anchors


def noise(scene, anchors=None, **config):
    ortho, images, collected = scene
    return estimate_anchor_noise(anchors or collected, OpenCVSiftMatcher(), OrthoReferenceProvider(ortho),
        lambda frame: images[frame.frame_id], COLLECT, AnchorNoiseConfig(**config))


def test_exact_scene_has_small_anchor_noise(scene):
    result = noise(scene)
    summary = result.summary()
    assert summary["estimates"] > 300 and summary["landmarks"] > 100
    assert np.all(np.abs(summary["bias_m"]) < 0.1)
    assert np.all(np.asarray(summary["sigma_anchor_m"][:2]) < 0.3)  # below the orthophoto GSD of 0.5 m
    np.testing.assert_allclose(result.deltas[:, 2], 0.0, atol=1e-9)  # flat DSM
    assert summary["outlier_fraction"] < 0.02


def test_only_later_observations_are_re_estimated(scene):
    _, _, anchors = scene
    result = noise(scene, min_frame_gap=2)
    first = {}
    for frame_id, landmark_id in zip(anchors.observations.frame_ids, anchors.observations.landmark_ids, strict=True):
        first.setdefault(int(landmark_id), int(frame_id))
    gaps = np.array([f - first[int(lid)] for lid, f in zip(result.landmark_ids, result.frame_ids, strict=True)])
    assert len(gaps) and gaps.min() >= 2


def test_perturbed_anchors_are_detected(scene):
    """Anchors displaced by 1 m (std, horizontal) show up as 1 m differences."""
    _, _, anchors = scene
    rng = np.random.default_rng(0)
    xyz = anchors.landmarks.xyz.copy()
    xyz[:, :2] += rng.normal(0.0, 1.0, (len(xyz), 2))
    perturbed = anchors.replace(landmarks=Landmarks(anchors.landmarks.ids, xyz))
    sigma = np.asarray(noise(scene, perturbed).summary()["sigma_difference_m"])
    np.testing.assert_allclose(sigma[:2], 1.0, rtol=0.2)


def test_local_affine_transfer_is_exact_for_affine_matches():
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 30, (20, 2))
    A, t = np.array([[1.9, 0.3], [-0.2, 2.1]]), np.array([100.0, 50.0])
    query = np.array([15.0, 15.0])
    estimate, count, rms = _local_affine(image, image @ A.T + t, query, AnchorNoiseConfig(neighbor_radius_px=20.0))
    np.testing.assert_allclose(estimate, A @ query + t)
    assert count >= 4 and rms < 1e-9
