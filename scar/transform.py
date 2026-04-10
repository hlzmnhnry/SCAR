import utm
import numpy as np

from scipy.spatial.transform import Rotation

def inverse_transformation(T: np.ndarray):
    """ Get inverse of transformation matrix T

    Args:
        T (np.ndarray): Transformation matrix

    Returns:
    
        np.ndarray: Inverse transformation matrix T^{-1}
    """
    
    R, t = T[:3, :3], T[:3, 3].reshape(3, 1)
    
    T_inv = np.vstack([np.hstack([R.T, -R.T @ t]),
        np.array([0., 0., 0., 1.])])
    
    return T_inv

def create_pose_matrix(rotation_matrix: np.ndarray, translation_vector: np.ndarray):

    assert rotation_matrix.shape == (3, 3)
    assert translation_vector.shape in [(3,), (3, 1)]

    translation_vector = translation_vector.flatten()

    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = translation_vector

    return pose_matrix

def orientation_ned_to_enu(angle_x: float, angle_y: float, angle_z: float, degrees_in=False,
    degrees_out=False):

    # rotation matrix to convert from NED to ENU (and vice versa)    
    # x- and y-axis are switched and z is flipped
    R_ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    # orientation in NED frame as rotation matrix
    orientation_ned = Rotation.from_euler("xyz", [angle_x, angle_y, angle_z],
        degrees=degrees_in).as_matrix()
    
    # apply conversion from NED to ENU
    orientation_enu = R_ned_to_enu @ orientation_ned
    
    return Rotation.from_matrix(orientation_enu).as_euler("xyz", degrees=degrees_out)

def lla_ned_euler_to_enu_pose(latitude: float, longitude: float, altitude: float,
    angle_x_rad: float, angle_y_rad: float, angle_z_rad: float):
    
    angles = orientation_ned_to_enu(angle_x=angle_x_rad, angle_y=angle_y_rad,
        angle_z=angle_z_rad, degrees_in=False, degrees_out=False)
    
    # get orientation as quaternions
    rot_quat = Rotation.from_euler("xyz", angles, degrees=False).as_quat(scalar_first=False,
        canonical=True)

    utm_x, utm_y, = utm.from_latlon(latitude, longitude)[:2]
    position = np.array([utm_x, utm_y, altitude], dtype=float)

    return position, rot_quat

def lla_ned_quat_to_enu_pose(latitude: float, longitude: float, altitude: float,
    quat_x: float, quat_y: float, quat_z: float, quat_w: float):
    
    R_ned = Rotation.from_quat(np.array([quat_x, quat_y, quat_z, quat_w]),
        scalar_first=False).as_matrix()

    # rotation matrix to convert from NED to ENU (and vice versa)    
    # x- and y-axis are switched and z is flipped
    R_ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    
    rot_quat = Rotation.from_matrix(R_ned_to_enu @ R_ned).as_quat(scalar_first=False)
    
    utm_x, utm_y, = utm.from_latlon(latitude, longitude)[:2]
    position = np.array([utm_x, utm_y, altitude], dtype=float)
    
    return position, rot_quat

def add_pose_noise(pose: np.ndarray, translation_noise_std: float = 0.0, rotation_noise_deg: float = 0.0):

    assert pose.shape == (4, 4), "'pose' must be a 4x4 pose matrix"

    noisy_pose = pose.copy()

    if translation_noise_std > 0:
        delta_t = np.random.normal(0.0, translation_noise_std, size=3)
        noisy_pose[:3, 3] += delta_t

    if rotation_noise_deg > 0:

        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)

        angle_rad = np.deg2rad(np.random.normal(0.0, rotation_noise_deg))

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R_delta = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        noisy_pose[:3, :3] = R_delta @ noisy_pose[:3, :3]

    return noisy_pose
