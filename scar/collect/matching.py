"""Feature matching between a reference orthophoto and an aerial image."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Matches:
    """Corresponding pixels in the reference (orthophoto) and the aerial image."""

    reference: np.ndarray  # (N, 2)
    image: np.ndarray  # (N, 2)
    confidence: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.reference)

    def subset(self, mask_or_index) -> Matches:
        return Matches(self.reference[mask_or_index], self.image[mask_or_index], self.confidence[mask_or_index])


class Matcher(Protocol):
    def match(self, reference: np.ndarray, image: np.ndarray) -> Matches:
        """Match two 8-bit grayscale images."""
        ...


def _get_matcher_factory():
    try:  # the package was renamed from image-matching-models to vismatch
        from vismatch import get_matcher
    except ImportError:
        from matching import get_matcher
    return get_matcher


class ImageMatchingModelsMatcher:
    """Matchers of the ``vismatch`` package (formerly ``image-matching-models``).

    Examples: superpoint-lightglue, roma.
    Match confidences are set to 1, as in the CVPRW 2026 pipeline: the package does not expose
    comparable confidences across matchers.
    """

    def __init__(self, name: str = "superpoint-lightglue", device: str = "auto", **kwargs) -> None:
        import torch

        get_matcher = _get_matcher_factory()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.device = device
        self._torch = torch
        self._matcher = get_matcher(name, device=device, **kwargs)

    def _to_tensor(self, image: np.ndarray):
        # same as torchvision's ToTensor for 8-bit images: CHW float32 in [0, 1]; vismatch >= 1.3
        # requires three channels (grayscale matchers convert back with weights summing to 1)
        array = np.ascontiguousarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        return self._torch.from_numpy(array).permute(2, 0, 1).float().div(255)

    def match(self, reference: np.ndarray, image: np.ndarray) -> Matches:
        if self.name.startswith("roma"):
            # RoMa loads images from files itself
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / "reference.png", Path(directory) / "image.png"]
                cv2.imwrite(str(paths[0]), reference)
                cv2.imwrite(str(paths[1]), image)
                inputs = [self._matcher.load_image(str(path), resize=None) for path in paths]
        else:
            inputs = [self._to_tensor(reference), self._to_tensor(image)]

        result = self._matcher(*inputs)
        kp_reference = np.asarray(result["matched_kpts0"], dtype=float).reshape(-1, 2)
        kp_image = np.asarray(result["matched_kpts1"], dtype=float).reshape(-1, 2)
        return Matches(kp_reference, kp_image, np.ones(len(kp_reference)))


class OpenCVSiftMatcher:
    """SIFT + ratio test (OpenCV only): a CPU baseline without deep-learning dependencies.

    Confidence is ``1 - ratio`` of the Lowe ratio test.
    """

    def __init__(self, max_features: int = 4000, ratio: float = 0.8) -> None:
        self.ratio = ratio
        self._sift = cv2.SIFT_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, reference: np.ndarray, image: np.ndarray) -> Matches:
        kp_ref, desc_ref = self._sift.detectAndCompute(reference, None)
        kp_img, desc_img = self._sift.detectAndCompute(image, None)
        empty = Matches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
        if desc_ref is None or desc_img is None or len(kp_ref) < 2 or len(kp_img) < 2:
            return empty

        pairs = [m for m in self._matcher.knnMatch(desc_ref, desc_img, k=2) if len(m) == 2]
        good = [(a, a.distance / max(b.distance, 1e-12)) for a, b in pairs if a.distance < self.ratio * b.distance]
        if not good:
            return empty
        reference_pts = np.asarray([kp_ref[a.queryIdx].pt for a, _ in good], dtype=float)
        image_pts = np.asarray([kp_img[a.trainIdx].pt for a, _ in good], dtype=float)
        return Matches(reference_pts, image_pts, np.asarray([1.0 - r for _, r in good]))
