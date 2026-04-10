import numpy as np

from scipy.spatial.transform import Rotation

def simple_pose_error(P1: np.ndarray, P2: np.ndarray):

    assert P1.shape == (4, 4) and P2.shape == (4, 4), "Inputs must be 4x4 pose matrices"

    # extract rotation and translation
    R1, t1 = P1[:3, :3], P1[:3, 3]
    R2, t2 = P2[:3, :3], P2[:3, 3]

    # rotation error: angle of relative rotation
    rotvec = Rotation.from_matrix(R1.T @ R2).as_rotvec()
    rot_error_deg = np.degrees(np.linalg.norm(rotvec))

    # translation error: euclidean distance
    trans_error = np.linalg.norm(t1 - t2)

    return rot_error_deg, trans_error

def full_pose_error(P1: np.ndarray, P2: np.ndarray):
    
    assert P1.shape == (4, 4) and P2.shape == (4, 4), "Inputs must be 4x4 pose matrices"

    # extract rotation and translation
    R1, t1 = P1[:3, :3], P1[:3, 3]
    R2, t2 = P2[:3, :3], P2[:3, 3]
    
    translation_difference = t2 - t1

    # rotation error: angle of relative rotation
    rotation_difference = Rotation.from_matrix(R1.T @ R2).as_euler("xyz", degrees=True)

    return rotation_difference, translation_difference

def error_statistics(errors: np.ndarray):
    
    assert errors.ndim == 1, "'errors' should be 1-dim array of errors"
    
    q25, q50, q75 = np.quantile(errors, [0.25, 0.5, 0.75])
    mad = np.median(np.abs(errors - q50))  # median absolute deviation
    
    assert q50 == np.median(errors).item(), "Q50 is expected to be equal to median"
    
    return {
        "mean": float(np.mean(errors)),
        "median": float(q50),
        "maximum": float(np.max(errors)),
        "minimum": float(np.min(errors)),
        "std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "quantiles": [float(q25), float(q50), float(q75)],
        "iqr": float(q75 - q25),
        "mad": float(mad),
    }
