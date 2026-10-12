"""Filters for reference-to-image matches and selection of new anchors."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def homography_inliers(reference: np.ndarray, image_undistorted: np.ndarray, threshold_px: float = 3.0,
    confidence: float = 0.95, max_iterations: int = 2000) -> np.ndarray | None:
    """Inlier mask of a MAGSAC++ homography between reference and (undistorted) image pixels.

    A homography is a valid model for near-nadir views of moderately flat terrain. Returns None if
    no model was found.
    """
    if len(reference) < 4:
        return None
    _, mask = cv2.findHomography(np.asarray(reference, dtype=np.float64), np.asarray(image_undistorted, dtype=np.float64),
        method=cv2.USAC_MAGSAC, ransacReprojThreshold=threshold_px, confidence=confidence, maxIters=max_iterations)
    return None if mask is None else mask.ravel().astype(bool)


def local_relief(dsm: np.ndarray, window: int = 15) -> np.ndarray:
    """Largest height difference to any neighbour within a ``window`` x ``window`` box.

    Pixels whose window touches missing heights (NaN) get infinite relief.
    """
    dsm = np.asarray(dsm, dtype=np.float32)
    missing = ~np.isfinite(dsm)
    filled = np.where(missing, 0.0, dsm).astype(np.float32)
    kernel = np.ones((window, window), dtype=np.uint8)
    relief = np.maximum(cv2.dilate(filled, kernel) - filled, filled - cv2.erode(filled, kernel))
    if missing.any():
        relief[cv2.dilate(missing.astype(np.uint8), kernel).astype(bool)] = np.inf
    return relief


def slope_mask(dsm: np.ndarray, reference: np.ndarray, window: int = 15, threshold_m: float = 5.0) -> np.ndarray:
    """True for reference pixels whose surrounding relief is at most ``threshold_m``."""
    relief = local_relief(dsm, window)
    rows = np.floor(np.asarray(reference)[:, 1]).astype(int)
    cols = np.floor(np.asarray(reference)[:, 0]).astype(int)
    inside = (rows >= 0) & (cols >= 0) & (rows < relief.shape[0]) & (cols < relief.shape[1])
    mask = np.zeros(len(rows), dtype=bool)
    mask[inside] = relief[rows[inside], cols[inside]] <= threshold_m
    return mask


def select_new_anchors(image_points: np.ndarray, confidence: np.ndarray, existing: np.ndarray,
    min_distance_px: float = 30.0, max_features: int | None = 500, min_confidence: float = 0.0,
    legacy_budget: bool = False) -> np.ndarray:
    """Greedily pick candidates (by confidence) that keep ``min_distance_px`` to all tracked points.

    Returns indices into ``image_points``. With ``legacy_budget`` each accepted point is counted
    twice against ``max_features``, reproducing the CVPRW 2026 implementation (which therefore
    added at most about half of the free budget per frame).
    """
    existing = np.asarray(existing, dtype=float).reshape(-1, 2)
    num_existing = len(existing)
    if max_features is not None and num_existing >= max_features:
        return np.zeros(0, dtype=int)

    valid = np.where(confidence >= min_confidence)[0]
    if valid.size == 0:
        return np.zeros(0, dtype=int)

    order = valid[np.argsort(-confidence[valid])]
    selected: list[int] = []
    points = list(existing)
    tree = cKDTree(existing) if num_existing else None

    for index in order:
        used = num_existing + (2 * len(selected) if legacy_budget else len(selected))
        if max_features is not None and used >= max_features:
            break
        point = image_points[index]
        if tree is not None and tree.query(point, k=1)[0] < min_distance_px:
            continue
        selected.append(int(index))
        points.append(point)
        tree = cKDTree(np.asarray(points))

    return np.asarray(selected, dtype=int)
