import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import makedirs
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from os.path import join, isfile, isdir
from scipy.optimize import least_squares
from scar import load_image_indices, extract_timestamp, inverse_transformation, full_pose_error

from pygeon.datasets.vgl_iterator import VGLIterator
from pygeon.io.calibrations import load_extrinsics
from pygeon.types import CameraModel, Extrinsics
from pygeon.enums import CampaignDate, FrameType

DATETIME_STRING = None

def cam_model_to_result_json(cam: CameraModel):
    
    camera_calibration = {
        "fx": cam.K[0, 0],
        "fy": cam.K[1, 1],
        "cx": cam.K[0, 2],
        "cy": cam.K[1, 2],
        "k1": cam.dist[0],
        "k2": cam.dist[1],
        "p1": cam.dist[2],
        "p2": cam.dist[3],
        "skew": cam.K[0, 1]
    }
    
    return camera_calibration

def is_config_valid(config: Dict):
    """check fields of config dictionary"""
    return True

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def _hat(w: np.ndarray):
    return np.array([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],  0.0]], dtype=np.float64)

def _vee(W: np.ndarray):
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)

def pose_expmap(xi: np.ndarray, eps: float = 1e-9):

    v = xi[:3]
    w = xi[3:]
    theta = np.linalg.norm(w)
    W = _hat(w)

    T = np.eye(4, dtype=np.float64)

    if theta < eps:
        
        # series expansions: exp(W) approx. I + W + 1/2 W^2,  V approx. I + 1/2 W + 1/6 W^2
        W2 = W @ W
        R = np.eye(3) + W + 0.5 * W2
        V = np.eye(3) + 0.5 * W + (1.0/6.0) * W2
    else:
        th2 = theta * theta
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / th2
        C = (theta - np.sin(theta)) / (theta * th2)
        W2 = W @ W
        R = np.eye(3) + A * W + B * W2
        V = np.eye(3) + B * W + C * W2

    T[:3, :3] = R
    T[:3, 3]  = V @ v

    return T

def pose_logmap(T: np.ndarray, eps: float = 1e-9):

    # orthonormalize R to avoid numerical drift
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = T[:3, 3]

    # rotation log
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < eps:
        # for small angles: ln(R) approx. 1/2 (R - R^T)
        W = 0.5 * (R - R.T)
        w = _vee(W)
        # V^{-1} ≈ I - 1/2 W + 1/12 W^2
        W2 = W @ W
        V_inv = np.eye(3) - 0.5 * W + (1.0/12.0) * W2
    else:
        # ln(R) = (theta/(2 sin(theta))) (R - R^T)
        W = (theta / (2.0 * np.sin(theta))) * (R - R.T) # this is hat(w)
        w = _vee(W)
        # V^{-1} = I - 1/2 W + β W^2,   β = 1/theta^2 - (1 + cos(theta))/(2 theta sin(theta))
        th2 = theta * theta
        beta = (1.0 / th2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        V_inv = np.eye(3) - 0.5 * W + beta * (W @ W)

    v = V_inv @ t

    return np.hstack([v, w])

def optimize_ins_cam_extrinsic_scipy(P_ins_list: List[np.ndarray],
    P_cam_opt_list: List[np.ndarray], T_ins_cam_guess: np.ndarray, method="lm",
    loss="linear", jac="3-point", W_T: float = 1.0, W_R: float = 1.0):
    
    if method == "lm":
        assert loss == "linear", "Levenberg-Marquardt supports only linear loss"

    T0  = T_ins_cam_guess.copy()
    
    def residual(delta_xi: np.ndarray):
        
        T = T0 @ pose_expmap(delta_xi)
        errors = []
        
        for P_ins, P_cam_opt in zip(P_ins_list, P_cam_opt_list):
            
            T_pred = P_ins @ T
            T_err = inverse_transformation(P_cam_opt) @ T_pred
            
            err = pose_logmap(T_err)
            err[:3] *= W_T
            err[3:] *= W_R
            
            errors.append(err)
        
        return np.concatenate(errors)
    
    delta0 = np.zeros(6)
    result = least_squares(residual, delta0, method=method, loss=loss, jac=jac)
    T_opt  = T0 @ pose_expmap(result.x)

    return T_opt

def format_matrix_for_code(mat: np.ndarray, precision: int = 8, formatted: str = ""):
    
    for row in mat:
        formatted += "[" + ", ".join(f"{val:.{precision}g}" for val in row) + "]\n"
    
    return formatted

def show_pose_residuals(camera_poses: List[Dict], camera_poses_new: Dict, config: Dict,
    subtitle: str = "", save_key: str = ""):
    
    global DATETIME_STRING
    
    rotation_deltas: List[np.ndarray] = []
    translation_deltas: List[np.ndarray] = []

    for pose_camera in camera_poses:
        
        pose_camera_ls = np.array(camera_poses_new[pose_camera["image_index"]])

        rot_diff, trans_diff = full_pose_error(pose_camera["pose"], pose_camera_ls)
        rotation_deltas.append(rot_diff)
        translation_deltas.append(trans_diff)

    rotation_deltas = np.array(rotation_deltas)
    translation_deltas = np.array(translation_deltas)
    
    rotation_delta_norms = np.linalg.norm(rotation_deltas, axis=1)
    translation_delta_norms = np.linalg.norm(translation_deltas, axis=1)
    
    ### figure for translation residuals ###
    
    fig_xyz, axes_xyz = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
    fig_xyz.canvas.manager.set_window_title(f"Translational residuals ({subtitle})")

    ax_x: plt.Axes = axes_xyz[0]
    ax_y: plt.Axes = axes_xyz[1]
    ax_z: plt.Axes = axes_xyz[2]
    ax_total: plt.Axes = axes_xyz[3]

    ax_x.plot(translation_deltas[:, 0], label="X", c="royalblue")
    ax_x.axhline(np.mean(translation_deltas[:, 0]), color="r", linestyle="--", label="Mean")
    ax_x.set_xlabel("Pose index")
    ax_x.set_ylabel("Delta (m)")
    ax_x.grid(True)
    ax_x.legend()

    ax_y.plot(translation_deltas[:, 1], label="Y", c="orange")
    ax_y.axhline(np.mean(translation_deltas[:, 1]), color="r", linestyle="--", label="Mean")
    ax_y.set_xlabel("Pose index")
    ax_y.set_ylabel("Delta (m)")
    ax_y.grid(True)
    ax_y.legend()
    
    ax_z.plot(translation_deltas[:, 2], label="Z", c="mediumseagreen")
    ax_z.axhline(np.mean(translation_deltas[:, 2]), color="r", linestyle="--", label="Mean")
    ax_z.set_xlabel("Pose index")
    ax_z.set_ylabel("Delta (m)")
    ax_z.grid(True)
    ax_z.legend()
    
    ax_total.plot(translation_delta_norms, label="L2-Norm", c="mediumpurple")
    ax_total.axhline(np.mean(translation_delta_norms), color="r", linestyle="--", label="Mean")
    ax_total.set_xlabel("Pose index")
    ax_total.set_ylabel("Delta (m)")
    ax_total.grid(True)
    ax_total.legend()
    
    for ax in [ax_x, ax_y, ax_z, ax_total]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontweight("bold")
        offset_text.set_color("red")

    fig_xyz.suptitle(f"Translational residuals ({subtitle})")
    fig_xyz.tight_layout()

    ### figure for rotational residuals ###
    
    fig_rpy, axes_rpy = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
    fig_rpy.canvas.manager.set_window_title(f"Rotational residuals ({subtitle})")

    ax_rx: plt.Axes = axes_rpy[0]
    ax_ry: plt.Axes = axes_rpy[1]
    ax_rz: plt.Axes = axes_rpy[2]
    ax_rtotal: plt.Axes = axes_rpy[3]
    
    ax_rx.plot(rotation_deltas[:, 0], label="Roll (X)", c="royalblue")
    ax_rx.axhline(np.mean(rotation_deltas[:, 0]), color="r", linestyle="--", label="Mean")
    ax_rx.set_xlabel("Pose index")
    ax_rx.set_ylabel("Delta (deg)")
    ax_rx.grid(True)
    ax_rx.legend()

    ax_ry.plot(rotation_deltas[:, 1], label="Pitch (Y)", c="orange")
    ax_ry.axhline(np.mean(rotation_deltas[:, 1]), color="r", linestyle="--", label="Mean")
    ax_ry.set_xlabel("Pose index")
    ax_ry.set_ylabel("Delta (deg)")
    ax_ry.grid(True)
    ax_ry.legend()
    
    ax_rz.plot(rotation_deltas[:, 2], label="Yaw (Z)", c="mediumseagreen")
    ax_rz.axhline(np.mean(rotation_deltas[:, 2]), color="r", linestyle="--", label="Mean")
    ax_rz.set_xlabel("Pose index")
    ax_rz.set_ylabel("Delta (deg)")
    ax_rz.grid(True)
    ax_rz.legend()
    
    ax_rtotal.plot(rotation_delta_norms, label="L2-Norm", c="mediumpurple")
    ax_rtotal.axhline(np.mean(rotation_delta_norms), color="r", linestyle="--", label="Mean")
    ax_rtotal.set_xlabel("Pose index")
    ax_rtotal.set_ylabel("Delta (deg)")
    ax_rtotal.grid(True)
    ax_rtotal.legend()

    for ax in [ax_rx, ax_ry, ax_rz, ax_rtotal]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontweight("bold")
        offset_text.set_color("red")

    fig_rpy.suptitle(f"Rotational residuals ({subtitle})")
    fig_rpy.tight_layout()
    
    if config.get("save_plots", False):
        fig_xyz.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_pose_residuals_{save_key}_xyz_{DATETIME_STRING}.png"), dpi=300)
        fig_rpy.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_pose_residuals_{save_key}_rpy_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig_xyz.show()
        fig_rpy.show()

def load_data(campaign: CampaignDate, config: Dict):
    
    ls_config: Dict[str, Any] = config["least_squares"]
    dataset_config: Dict[str, Any] = config["dataset"]
    
    ### load optimized data ###
    
    input_file_type = ls_config["result_input_file"]["type"]
    input_filename = None
    
    if input_file_type not in ["path", "load_latest_from_prefix"]:
        raise ValueError("Unknown file input type")
    
    if input_file_type == "load_latest_from_prefix":
    
        glob_directory = ls_config["result_input_file"]["input_directory"]
        if not isdir(glob_directory):
            raise ValueError(f"'{glob_directory}' is not a directory")
        
        prefix = ls_config["result_input_file"]["prefix"]
        possible_files = list(glob(join(glob_directory, f"{prefix}_*.json")))
        
        if len(possible_files) == 0:
            raise ValueError(f"There is no file with prefix '{prefix}' in '{glob_directory}'")
        
        input_filename = max(possible_files, key=extract_timestamp)
    
    elif input_file_type == "path":
        input_filename = ls_config["result_input_file"]["path"]
    
    if not isfile(input_filename):
        raise ValueError(f"'{input_filename}' does not exist")
    
    print(f"\nReading optimization result from {input_filename}...")
    
    with open(input_filename, "r") as fin:
        results = json.load(fin)

    cal = results["camera_calibration"]
    K = np.array([[cal["fx"], 0., cal["cx"]], [0., cal["fy"], cal["cy"]], [0., 0., 1.]], dtype=float)
    dist = np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"]], dtype=float)
    
    cam_model = CameraModel(K=K, dist=dist, size=(1600, 1100))
    
    image_indices = results["image_indices"]
    pixel_tracks = results["pixel_tracks"]
    world_points = results["world_points"]

    if load_image_indices(pixel_coords=pixel_tracks) != image_indices:
        raise ValueError("There is an inconsistency in the provided image indices and pixel coordinates")

    ### load extrinsics ###
    
    extrinsics_calibration_file = None  
    
    if "extrinsics_calibration_file" not in dataset_config:
        raise ValueError("'read_extrinsics_from_file' not in 'extrinsics_file' in config")
    
    extrinsics_calibration_file = dataset_config["extrinsics_calibration_file"]
    print(f"\nReading extrinsics from {extrinsics_calibration_file}...")
    
    if extrinsics_calibration_file is None:
        raise ValueError("Calibrations could not be loaded successfully")
    
    extrinsics_calibration_file: str
    
    extrinsics_calibration_directory = Path(extrinsics_calibration_file).parent \
        if extrinsics_calibration_file.endswith(".json") else None
    extrinsics_file_basename = Path(extrinsics_calibration_file).parts[-1] \
        if extrinsics_calibration_file.endswith(".json") else "extrinsics.json"

    extrinsics: Extrinsics = load_extrinsics(calib_dir=extrinsics_calibration_directory,
        extrinsics_file_basename=extrinsics_file_basename)
    
    ### load raw data ###
    
    print(f"\nLoading campaign {campaign.value} data...")
    
    image_directory = Path(dataset_config["image_directory"]) \
        if "image_directory" in dataset_config else None
    data_directory = Path(dataset_config["data_directory"]) \
        if "data_directory" in dataset_config else None
    
    it = VGLIterator(
        campaign=campaign,
        data_directory=data_directory,
        image_directory=image_directory,
        frame=FrameType.BODY,
        image_indexed_naming=True,
        with_landcover=False,
        with_image=False,
        camera_calibration_directory=Path("config/"),           # NOTE: calibration not used here, use kalibr as dummy
        camera_file_basename="camera_matrix_guess_kalibr.json", # NOTE: calibration not used here, use kalibr as dummy
        extrinsics_calibration_directory=extrinsics_calibration_directory,
        extrinsics_file_basename=extrinsics_file_basename
    )

    ### create pose lists ###
    
    camera_poses_optimized: List[Dict] = []

    ins_poses_groundtruth: List[Dict] = []
    camera_poses_groundtruth: List[Dict] = []

    for image_ts in image_indices:
        
        ### read optimized poses ###
        
        camera_poses_optimized.append({
            "image_index": image_ts,
            "pose": np.asarray(results["camera_poses"][str(image_ts)], dtype=float)
        })
        
        sample = it.get_by_timestamp(ts_us=image_ts, nearest_only=True, max_dt_us=0)
        
        ### read original poses ###
        
        ins_poses_groundtruth.append({
            "image_index": image_ts,
            "pose": sample["pose"]
        })
        
        P_cam = sample["pose"] @ extrinsics.T_bc
        camera_poses_groundtruth.append({
            "image_index": image_ts,
            "pose": P_cam
        })

    return world_points, pixel_tracks, ins_poses_groundtruth, camera_poses_optimized, \
        camera_poses_groundtruth, image_indices, cam_model, extrinsics

def main():

    parser = argparse.ArgumentParser(description="SCAR LS estimation of extrinsics", add_help=True)
    parser.add_argument("--base-config-file", type=str, required=True, help="File to read base configuration parameters from")
    parser.add_argument("--config-file", type=str, required=True, help="File to read campaign-specific configuration parameters from")
    
    args = parser.parse_args()
    
    if not isfile(args.base_config_file):
        raise ValueError("Provided base config file does not exist")
    
    if not isfile(args.config_file):
        raise ValueError("Provided config file does not exist")

    config: Dict = json.load(open(args.base_config_file))
    override: Dict = json.load(open(args.config_file))

    config = deep_update(config, override)
    dataset_config: Dict[str, Any] = config["dataset"]
    ls_config: Dict[str, Any] = config["least_squares"]
    
    if not is_config_valid(config=config):
        raise ValueError("Provided config file is not valid")
    
    global DATETIME_STRING
    DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    campaign = CampaignDate(dataset_config["campaign"])
    config["output_directory"] = join("output", campaign.value, "optimization", config["output_directory"])
    
    world_points, pixel_tracks, ins_poses_gt, camera_poses_opt, camera_poses_gt, \
        image_indices, camera_model, t_ins_cam_guess = load_data(config=config, campaign=campaign)

    t_ins_cam_guess: Extrinsics
    camera_model: CameraModel

    P_ins_list: List[np.ndarray] = []
    P_cam_opt_list: List[np.ndarray] = []

    for i, image_index in enumerate(image_indices):
        
        pose_ins = ins_poses_gt[i]
        pose_opt = camera_poses_opt[i]
        
        if pose_ins["image_index"] != image_index or pose_opt["image_index"] != image_index:
            raise ValueError("Image indices do not match")
        
        P_ins_list.append(pose_ins["pose"])
        P_cam_opt_list.append(pose_opt["pose"])

    parameters: Dict[str, Any] = ls_config.get("parameters", {})

    T_ins_cam_optimized = optimize_ins_cam_extrinsic_scipy(P_ins_list=P_ins_list,
        P_cam_opt_list=P_cam_opt_list, T_ins_cam_guess=t_ins_cam_guess.T_bc,
        method=parameters["method"], loss=parameters["loss"],
        jac=parameters["jac"], W_T=parameters.get("W_T", 1.0),
        W_R=parameters.get("W_R", 1.0),
    )
    
    ### run small sanity check for SE(3) logic ###
    
    rng = np.random.default_rng(0)
    xi = np.hstack([0.3 * rng.standard_normal(3), 0.1 * rng.standard_normal(3)])
    T = pose_expmap(xi)
    xi_rt = pose_logmap(T)
    
    if not np.allclose(xi, xi_rt, atol=1e-9):
        raise ValueError("Something is faulty with SE(3) logic")
    
    ### apply newly determined extrinsics from LS ###
    
    camera_poses_ls: Dict[int, List[List[float]]] = {}

    for i, image_index in enumerate(image_indices):
        
        pose_ins = ins_poses_gt[i]
        if pose_ins["image_index"] != image_index:
            raise ValueError("Image indices do not match")

        camera_poses_ls[image_index] = (pose_ins["pose"] @ T_ins_cam_optimized).tolist()

    ### visualization ###

    np.set_printoptions(suppress=True)

    print(f"T_ins_cam (initial):\n{format_matrix_for_code(t_ins_cam_guess.T_bc)}")
    print(f"T_ins_cam (optimized):\n{format_matrix_for_code(T_ins_cam_optimized)}")
    
    show_pose_residuals(camera_poses=camera_poses_gt,
        camera_poses_new={pose['image_index']: pose["pose"] \
        for pose in camera_poses_opt}, subtitle="Before OPT/LS",
        save_key="before", config=config)
    
    show_pose_residuals(camera_poses=camera_poses_opt,
        camera_poses_new=camera_poses_ls, subtitle="After OPT/LS",
        save_key="after", config=config)
    
    if config.get("show_plots", True):
        plt.show()
    
    print("Exporting LS-results as .json file...")
    
    ### export results ###
    
    if config.get("export_results", False):
        
        results = {
            "image_indices": image_indices,
            "camera_poses": camera_poses_ls,
            "world_points": world_points,
            "pixel_tracks": pixel_tracks,
            "camera_calibration": cam_model_to_result_json(camera_model),
            "T_ins_cam": T_ins_cam_optimized.tolist()
        }
        
        makedirs(config["output_directory"], exist_ok=True)
        
        filename = f"{config['output_prefix']}_{DATETIME_STRING}.json"
        
        with open(join(config["output_directory"], filename), "w+") as fout:
            json.dump(results, fout, indent=4)

        print(f"\nWrote LS-results to {join(config['output_directory'], filename)}")

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
