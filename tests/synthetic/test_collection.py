"""Anchor collection on an exactly rendered synthetic scene."""

import numpy as np
import pytest

from scar.collect import AnchorCollector, CollectConfig, OpenCVSiftMatcher

from .scene import (
    CAMERA,
    OrthoReferenceProvider,
    flight,
    ground_points,
    render_aerial,
    texture,
)


@pytest.fixture(scope="module")
def scene():
    ortho = texture()
    sequence = flight()
    images = {frame.frame_id: render_aerial(ortho, CAMERA, frame.pose_prior.T_world_body, seed=frame.frame_id)
        for frame in sequence.frames}
    return ortho, sequence, images


def collect(scene, **overrides):
    ortho, sequence, images = scene
    config = CollectConfig(collect_every_nth_frame=1, max_features=300, min_pixel_distance=15.0, min_confidence=0.0,
        **overrides)
    collector = AnchorCollector(config, OpenCVSiftMatcher(), OrthoReferenceProvider(ortho),
        read_image=lambda frame: images[frame.frame_id])
    return collector.collect(sequence)


def anchor_errors(result):
    """Anchor position minus the true ground point seen at its first observation (per anchor)."""
    frames = result.frame_by_id
    errors = []
    for landmark_id, xyz in zip(result.landmarks.ids, result.landmarks.xyz, strict=True):
        first = np.flatnonzero(result.observations.landmark_ids == landmark_id)[0]
        frame = frames[int(result.observations.frame_ids[first])]
        truth = ground_points(CAMERA, frame.pose_prior.T_world_body, result.observations.uv[first][None])[0]
        errors.append(xyz - truth)
    return np.asarray(errors)


def test_anchors_are_unbiased_and_tracked(scene):
    result = collect(scene)
    assert len(result.landmarks) > 200
    assert len(result.observations) > 2 * len(result.landmarks)  # anchors are tracked over frames

    errors = anchor_errors(result)
    assert np.all(np.abs(np.mean(errors[:, :2], axis=0)) < 0.05)
    assert np.median(np.linalg.norm(errors[:, :2], axis=1)) < 0.4
    np.testing.assert_allclose(errors[:, 2], 0.0, atol=1e-9)

    # tracked observations stay consistent with the true geometry
    frames = result.frame_by_id
    index = {int(i): k for k, i in enumerate(result.landmarks.ids)}
    residuals = [np.linalg.norm(CAMERA.project(result.landmarks.xyz[index[int(lid)]][None],
        frames[int(fid)].pose_prior.T_world_body)[0] - uv)
        for fid, lid, uv in zip(result.observations.frame_ids, result.observations.landmark_ids,
            result.observations.uv, strict=True)]
    assert np.median(residuals) < 1.0


def test_legacy_pixel_convention_shifts_anchors_by_half_a_pixel(scene):
    """Without the +0.5 px offset anchors move by (-GSD/2, +GSD/2), as in the CVPRW 2026 pipeline."""
    errors = anchor_errors(collect(scene, reference_pixel_offset=0.0))
    np.testing.assert_allclose(np.mean(errors[:, :2], axis=0), [-0.25, 0.25], atol=0.05)


def test_resume_continues_exactly(scene, tmp_path):
    ortho, sequence, images = scene
    config = CollectConfig(collect_every_nth_frame=1, max_features=300, min_pixel_distance=15.0, min_confidence=0.0,
        checkpoint_every=1)

    def collector():
        return AnchorCollector(config, OpenCVSiftMatcher(), OrthoReferenceProvider(ortho),
            read_image=lambda frame: images[frame.frame_id])

    full = collector().collect(sequence)
    collector().collect(sequence.replace(frames=sequence.frames[:3]), checkpoint_directory=tmp_path)
    resumed = collector().collect(sequence, checkpoint_directory=tmp_path, resume=True)

    np.testing.assert_array_equal(resumed.landmarks.ids, full.landmarks.ids)
    np.testing.assert_allclose(resumed.landmarks.xyz, full.landmarks.xyz)
    np.testing.assert_array_equal(resumed.observations.frame_ids, full.observations.frame_ids)
    np.testing.assert_allclose(resumed.observations.uv, full.observations.uv)


def test_image_mask_excludes_features(scene):
    ortho, sequence, images = scene
    mask = np.zeros((CAMERA.height, CAMERA.width), dtype=bool)
    mask[:, : CAMERA.width // 2] = True  # ignore the left half
    collector = AnchorCollector(CollectConfig(collect_every_nth_frame=1, max_features=300, min_pixel_distance=15.0,
        min_confidence=0.0), OpenCVSiftMatcher(), OrthoReferenceProvider(ortho),
        read_image=lambda frame: images[frame.frame_id], image_mask=mask)
    result = collector.collect(sequence)
    assert len(result.observations) > 100
    assert np.all(np.round(result.observations.uv[:, 0]) >= CAMERA.width // 2)
