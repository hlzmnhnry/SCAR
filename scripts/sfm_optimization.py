import json
import gtsam # type: ignore
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from glob import glob
from tqdm import tqdm
from os import makedirs
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from os.path import join, isfile, isdir

from scar import filter_frames_with_few_features, \
    filter_points_with_few_observations, remap_ids_sequentially, load_image_indices, \
    extract_timestamp, full_pose_error, filter_frames_in_range, add_pose_noise

from pygeon.types import CameraModel
from pygeon.enums import CampaignDate, FrameType
from pygeon.io.calibrations import load_camera_model
from pygeon.datasets.vgl_iterator import VGLIterator

DATETIME_STRING = None

def is_config_valid(config: Dict):
    """check fields of config dictionary"""
    
    if "data_source" not in config:
        raise ValueError("No data source provided")
    
    if config["data_source"] not in ["dataset", "optimization_result"]:
        raise ValueError("Unknown 'data_source' for SFM optimization, choose either 'dataset' or 'optimization_result'")
    
    return True

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def print_camera_calibration(K: np.ndarray, dist: np.ndarray, name: str,
    precision=2, total_length=10):
    
    assert K.shape == (3, 3), "Invalid camera calibration matrix"
    
    print(f"{name} (3x3 + dist.):")
    
    for i in range(K.shape[0]):
        row = " ".join(f"{val:>{total_length}.{precision}f}" for val in K[i, :3])
        line_str = f"[ {row} ]"
        print(line_str)

    line_length = len(line_str)
    length_per_dist_elem = line_length - (len(dist) + 1) - 2
    length_per_dist_elem /= len(dist)
    length_per_dist_elem = np.floor(length_per_dist_elem).astype(int).item()
    
    line_length = len(line_str)
    print("#" * line_length)
    
    row = " ".join(f"{val:>{length_per_dist_elem}.{precision}f}" for val in dist.flatten())
    line_str = f"[ {row}{' ' * (line_length - length_per_dist_elem * len(dist) - (len(dist) + 1) - 2)} ]"
    print(line_str, "\n")

def load_reprojection_kernel(config: Dict):

    optimization_config: Dict[str, Any] = config["optimization"]

    sigmas = optimization_config["reprojection_noise"]
    base_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    kernel_name = optimization_config.get("kernel", "None")
    kernel_param = optimization_config.get("kernel_param", 1.0)

    if kernel_name in [None, "None", ""]:
        return base_noise

    kernel_map = {
        "Huber": gtsam.noiseModel.mEstimator.Huber,
        "Cauchy": gtsam.noiseModel.mEstimator.Cauchy,
        "GemanMcClure": gtsam.noiseModel.mEstimator.GemanMcClure
    }

    if kernel_name not in kernel_map:
        raise ValueError(f"Unknown kernel type '{kernel_name}'")

    kernel = kernel_map[kernel_name].Create(kernel_param)
    
    return gtsam.noiseModel.Robust.Create(kernel, base_noise)

def run_optimization(graph: gtsam.NonlinearFactorGraph, initial_estimates: gtsam.Values,
    method: str = "LM", verbosity: str = "SUMMARY"):
    
    errors_per_iteration: List[float] = []

    def record_error(iteration, errorBefore, errorAfter):
        
        if iteration == 1:
            errors_per_iteration.append(errorBefore)
        
        # error is current sum of (squared) residuals
        errors_per_iteration.append(errorAfter)
        
        print(f"Iteration #{iteration}, error before: {errorBefore:.2f}, error after: {errorAfter:.2f}")
        
    if method == "LM":
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM(verbosity)
        try:
            params.iterationHook = record_error
        except AttributeError:
            pass
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        
    elif method == "Dogleg":
        params = gtsam.DoglegParams()
        params.setVerbosityDL(verbosity)
        try:
            params.iterationHook = record_error
        except AttributeError:
            pass
        optimizer = gtsam.DoglegOptimizer(graph, initial_estimates, params)

    elif method == "GN":
        params = gtsam.GaussNewtonParams()
        params.setVerbosity(verbosity)
        try:
            params.iterationHook = record_error
        except AttributeError:
            pass
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimates, params)

    else:
        raise ValueError(f"Unknown optimization method '{method}', choose 'LM', 'Dogleg' or 'GN'")

    result = optimizer.optimize()

    return result, errors_per_iteration

def show_errors_per_iteration(errors_per_iteration: List[float], config: Dict):
    
    global DATETIME_STRING
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.canvas.manager.set_window_title("Errors for each iteration")
    
    ax.plot(errors_per_iteration, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_xticks(range(len(errors_per_iteration)))
    ax.set_ylabel("Cost (sum of sq. residuals)")
    
    ax.set_title("Optimization error")
    ax.grid(True, axis="y")
    ax.set_yscale("log")
    
    # visible range: +- one magnitude 
    epsilon = 1e-12
    ymin = max(min(errors_per_iteration), epsilon)
    ymax = max(errors_per_iteration)
    ax.set_ylim(ymin / 10, ymax * 10)

    for i, err in enumerate(errors_per_iteration):
        ax.text(i, err * 1.1, f"{err:.2f}",
            ha="center", va="bottom", fontsize=10, color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            clip_on=False)
    
    fig.tight_layout()
    
    if config.get("save_plots", False):
        fig.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_errors_per_iteration_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig.show()

def show_pose_changes(camera_poses: Dict, image_indices: List[int],
    values: gtsam.Values, config: Dict):
    
    global DATETIME_STRING
    
    rotation_deltas: List[np.ndarray] = []
    translation_deltas: List[np.ndarray] = []

    for i, pose_camera in enumerate(camera_poses):
        
        local_pose_index = image_indices.index(pose_camera["image_index"])
        assert i == local_pose_index, "Local pose index does not match position in list"
        
        key_camera = gtsam.symbol("C", i)
        P_cam_est = values.atPose3(key_camera).matrix()
        
        rot_diff, trans_diff = full_pose_error(pose_camera["pose"], P_cam_est)
        rotation_deltas.append(rot_diff)
        translation_deltas.append(trans_diff)

    rotation_deltas = np.array(rotation_deltas)
    translation_deltas = np.array(translation_deltas)
    
    rotation_delta_norms = np.linalg.norm(rotation_deltas, axis=1)
    translation_delta_norms = np.linalg.norm(translation_deltas, axis=1)
    
    ### figure for translation changes ###
    
    fig_xyz, axes_xyz = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
    fig_xyz.canvas.manager.set_window_title("Translational camera pose changes")

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

    fig_xyz.suptitle("Translational camera pose changes")
    fig_xyz.tight_layout()

    ### figure for rotation changes ###
    
    fig_rpy, axes_rpy = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
    fig_rpy.canvas.manager.set_window_title("Rotational camera pose changes")

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

    fig_rpy.suptitle("Rotational camera pose changes")
    fig_rpy.tight_layout()
    
    if config.get("save_plots", False):
        fig_xyz.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_pose_changes_xyz_{DATETIME_STRING}.png"), dpi=300)
        fig_rpy.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_pose_changes_rpy_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig_xyz.show()
        fig_rpy.show()

def show_landmark_changes(world_points: Dict[str, List[float]], values: gtsam.Values, config: Dict):

    global DATETIME_STRING

    translation_errors: List[float] = []
    translation_deltas: List[np.ndarray] = []

    for pid, point in world_points.items():
        
        if not pid.isdigit():
            raise ValueError("Point ID is expected to be an integer")
        key_landmark = gtsam.symbol("W", int(pid))
        landmark_est: np.ndarray = np.asarray(values.atPoint3(key_landmark)).reshape(3)

        delta = landmark_est.flatten() - np.array(point).flatten()
        translation_deltas.append(delta)

        error = np.linalg.norm(delta)
        translation_errors.append(error)

    translation_deltas = np.array(translation_deltas)
    translation_norms = np.linalg.norm(translation_deltas, axis=1)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
    fig.canvas.manager.set_window_title("Landmark position changes")

    ax_x: plt.Axes = axes[0]
    ax_y: plt.Axes = axes[1]
    ax_z: plt.Axes = axes[2]
    ax_total: plt.Axes = axes[3]

    ax_x.plot(translation_deltas[:, 0], label="X", c="royalblue")
    ax_x.axhline(np.mean(translation_deltas[:, 0]), color="r", linestyle="--", label="Mean")
    ax_x.set_xlabel("Landmark index")
    ax_x.set_ylabel("Delta (m)")
    ax_x.grid(True)
    ax_x.legend()
    
    ax_y.plot(translation_deltas[:, 1], label="Y", c="orange")
    ax_y.axhline(np.mean(translation_deltas[:, 1]), color="r", linestyle="--", label="Mean")
    ax_y.set_xlabel("Landmark index")
    ax_y.set_ylabel("Delta (m)")
    ax_y.grid(True)
    ax_y.legend()
    
    ax_z.plot(translation_deltas[:, 2], label="Z", c="mediumseagreen")
    ax_z.axhline(np.mean(translation_deltas[:, 2]), color="r", linestyle="--", label="Mean")
    ax_z.set_xlabel("Landmark index")
    ax_z.set_ylabel("Delta (m)")
    ax_z.grid(True)
    ax_z.legend()
    
    ax_total.plot(translation_norms, label="L2-Norm", c="mediumpurple")
    ax_total.axhline(np.mean(translation_norms), color="r", linestyle="--", label="Mean")
    ax_total.set_xlabel("Landmark index")
    ax_total.set_ylabel("Delta (m)")
    ax_total.grid(True)
    ax_total.legend()
    
    for ax in [ax_x, ax_y, ax_z, ax_total]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontweight("bold")
        offset_text.set_color("red")

    fig.suptitle("Landmark position changes")
    fig.tight_layout()
    
    if config.get("save_plots", False):
        fig.savefig(join(config["output_directory"],
            f"{config['output_prefix']}_landmark_changes_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig.show()

def load_dataset(config: Dict):
    
    dataset_config: Dict[str, Any] = config["dataset"]
    sequence_config: Dict[str, Any] = config["sequence"]
    campaign = CampaignDate(dataset_config["campaign"])
    
    print(f"\nLoading campaign {campaign.value} data...")
    
    data_directory = Path(dataset_config["data_directory"]) \
        if "data_directory" in dataset_config else None
    
    ### read calibration data ###
    
    camera_calibration_file, extrinsics_calibration_file = None, None
    
    if "camera_calibration_file" not in dataset_config:
        raise ValueError("'camera_calibration_file' not in dataset config")
    
    if "extrinsics_calibration_file" not in dataset_config:
        raise ValueError("'read_extrinsics_from_file' not in 'extrinsics_file' in config")
    
    camera_calibration_file = dataset_config["camera_calibration_file"]
    print(f"\nReading camera calibration from {camera_calibration_file}...")

    extrinsics_calibration_file = dataset_config["extrinsics_calibration_file"]
    print(f"\nReading extrinsics from {extrinsics_calibration_file}...")
    
    if camera_calibration_file is None or extrinsics_calibration_file is None:
        raise ValueError("Calibrations could not be loaded successfully")
    
    cam_calib_dir = Path(camera_calibration_file).parent
    cam_file_basename = Path(camera_calibration_file).name
    camera_model: CameraModel = load_camera_model(calib_dir=cam_calib_dir, camera_file_basename=cam_file_basename)
    
    ext_calib_dir = Path(extrinsics_calibration_file).parent
    ext_file_basename = Path(extrinsics_calibration_file).name
    
    ### load sequence data and check sanity ###
    
    input_folder = join("output", campaign.value, "geofeatures")
    pixel_tracks: dict = json.load(open(join(input_folder, f"PT_{campaign.value}_{config['sequence']['sequence_id']}.json")))
    world_points: dict = json.load(open(join(input_folder, f"WP_{campaign.value}_{config['sequence']['sequence_id']}.json")))

    if len(pixel_tracks) != len(world_points):
        raise ValueError("Each world point must have at least one observation")
    
    pt_point_ids = [int(point_id) for point_id in pixel_tracks.keys()]
    wp_point_ids = [int(point_id) for point_id in world_points.keys()]
    
    if not (np.unique(pt_point_ids).tolist() == pt_point_ids and np.unique(wp_point_ids).tolist() == wp_point_ids):
        raise ValueError("Point IDs are not unique (either in PC or in WP file)")
    
    pc_maximum_id, wp_maximum_id = max(pt_point_ids), max(wp_point_ids)
    
    if not(pc_maximum_id == wp_maximum_id and wp_maximum_id == len(world_points) - 1):
        raise ValueError(f"ID mixup in input sequence, 'pc_maximum_id'={pc_maximum_id}, 'wp_maximum_id'={wp_maximum_id}")
    
    num_points = len(world_points)
    observations_per_points = [len(obs) for obs in pixel_tracks.values()]
    
    print(f"\nLoaded {num_points} points from sequence '{config['sequence']['sequence_id']}'")
    print(f"Total of {sum(observations_per_points)} observations")
    
    ### (optionally) clip sequence to provided frame range ###
    
    if "sequence_clipping" in sequence_config:

        clipping_config: Dict = sequence_config.get("sequence_clipping", {})
        image_indices: List[int] = sorted(load_image_indices(pixel_coords=pixel_tracks))
        if len(image_indices) == 0:
            raise ValueError("No image indices available")

        local_start = int(clipping_config.get("local_index_start", 0))
        local_end = clipping_config.get("local_index_end", -1)
        local_end = int(local_end) if local_end is not None else -1

        if local_end == -1:
            local_end = len(image_indices) - 1

        if not(0 <= local_start < len(image_indices)):
            raise ValueError("'local_index_start' must be within [0, len(IMAGE_INDICES)-1]")
        if not (0 <= local_end < len(image_indices)):
            raise ValueError("'local_index_end' must be within [0, len(IMAGE_INDICES)-1]")
        if not (local_start <= local_end):
            raise ValueError("'local_index_start' must be <= 'local_index_end'")

        image_index_start = image_indices[local_start]
        image_index_end = image_indices[local_end]
        num_points = len(world_points)

        pixel_tracks, world_points = filter_frames_in_range(pixel_coords=pixel_tracks, world_points=world_points,
            image_index_start=image_index_start, image_index_end=image_index_end)
        pixel_tracks, world_points = remap_ids_sequentially(pixel_tracks, world_points)[:2]

        print(f"\nClipped frame range to {image_index_start}-{image_index_end} (local {local_start}-{local_end}). Removed {num_points - len(world_points)} points.")

        image_indices = sorted(load_image_indices(pixel_coords=pixel_tracks))
        num_points = len(world_points)

    ### remove points that only have few observations ###
    
    pixel_tracks, world_points = filter_points_with_few_observations(pixel_tracks, world_points,
        minimum_observations=sequence_config["filtering"]["minimum_point_observations"])
    pixel_tracks, world_points = remap_ids_sequentially(pixel_tracks, world_points)[:2]
    
    print(f"\nRemoved {num_points - len(world_points)} points as they have less than {config['sequence']['filtering']['minimum_point_observations']} observations")
    
    ### remove frames that contain only few points ###
    
    pixel_tracks, world_points, removed_frames = filter_frames_with_few_features(pixel_tracks, world_points,
        minimum_features_per_frame=sequence_config["filtering"]["minimum_features_per_frame"])
    pixel_tracks, world_points = remap_ids_sequentially(pixel_tracks, world_points)[:2]

    print(f"Removed {len(removed_frames)} frames as they contain less than {config['sequence']['filtering']['minimum_features_per_frame']} points")
    
    image_indices: List = load_image_indices(pixel_coords=pixel_tracks)
    
    num_points = len(world_points)
    observations_per_points = [len(obs) for obs in pixel_tracks.values()]
    
    print(f"\nRemaining number of frames: {len(image_indices)}")
    print(f"Remaining number of points: {num_points}")
    print(f"Remaining number of observations: {sum(observations_per_points)}")
    
    pixel_tracks: Dict[str, List[List[int, List[float]]]]
    world_points: Dict[str, List[float]]
    
    ### reading groundtruth data for camera poses ###
    
    it = VGLIterator(
        campaign=campaign,
        data_directory=data_directory,
        frame=FrameType.CAMERA,
        image_indexed_naming=True,
        with_landcover=False,
        with_image=False,
        camera_calibration_directory=cam_calib_dir,
        camera_file_basename=cam_file_basename,
        extrinsics_calibration_directory=ext_calib_dir,
        extrinsics_file_basename=ext_file_basename
    )
    
    camera_poses: List[Dict[int, np.ndarray]] = []
    
    for _, image_ts in enumerate(image_indices):
        
        sample = it.get_by_timestamp(ts_us=image_ts, nearest_only=True, max_dt_us=0)
        P_cam = sample["pose"]
        
        if "additive_pose_noise" in config["dataset"]:
            
            noise_config: Dict = config["dataset"]["additive_pose_noise"]
            translation_noise_std = noise_config.get("translation", 0.)
            rotation_noise_deg = noise_config.get("rotation", 0.)
            
            P_cam = add_pose_noise(P_cam, translation_noise_std=translation_noise_std,
                rotation_noise_deg=rotation_noise_deg)
        
        camera_poses.append({
            "image_index": image_ts,
            "pose": P_cam
        })

    return world_points, pixel_tracks, image_indices, camera_poses, camera_model

def load_optimization_result(config: Dict):
    
    optimization_config: Dict[str, Any] = config["optimization"]
    
    ### load optimized data ###
    
    input_file_type = optimization_config["result_input_file"]["type"]
    input_filename = None
    
    if input_file_type not in ["path", "load_latest_from_prefix"]:
        raise ValueError("Unknown file input type")
    
    if input_file_type == "load_latest_from_prefix":
    
        glob_directory = optimization_config["result_input_file"]["input_directory"]
        if not isdir(glob_directory):
            raise ValueError(f"'{glob_directory}' is not a directory")
        
        prefix = optimization_config["result_input_file"]["prefix"]
        possible_files = list(glob(join(glob_directory, f"{prefix}_*.json")))
        
        if len(possible_files) == 0:
            raise ValueError(f"There is no file with prefix '{prefix}' in '{glob_directory}'")
        
        input_filename = max(possible_files, key=extract_timestamp)
    
    elif input_file_type == "path":
        input_filename = optimization_config["result_input_file"]["path"]
    
    if not isfile(input_filename):
        raise ValueError(f"'{input_filename}' does not exist")
        
    print(f"\nReading previous optimization result from {input_filename}...")
    
    with open(input_filename, "r") as fin:
        results = json.load(fin)
        
    image_indices = results["image_indices"]
    pixel_tracks = results["pixel_tracks"]
    
    if load_image_indices(pixel_coords=pixel_tracks) != image_indices:
        raise ValueError("There is an inconsistency in the provided image indices and pixel coordinates")
    
    camera_poses: List[Dict] = []
    for image_index in image_indices:
        camera_poses.append({
            "image_index": image_index,
            "pose": np.asarray(results["camera_poses"][str(image_index)], dtype=float)
        })
    
    world_points = results["world_points"]
    
    cal = results["camera_calibration"]
    K = np.array([[cal["fx"], 0., cal["cx"]], [0., cal["fy"], cal["cy"]], [0., 0., 1.]], dtype=float)
    dist = np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"]], dtype=float)
    
    cam_model = CameraModel(K=K, dist=dist, size=(1600, 1100))
    
    return world_points, pixel_tracks, image_indices, camera_poses, cam_model

def main():

    parser = argparse.ArgumentParser(description="SCAR SFM optimization", add_help=True)
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
    
    if not is_config_valid(config=config):
        raise ValueError("Provided config file is not valid")
    
    if config["data_source"] == "optimization_result":
        world_points, pixel_tracks, image_indices, camera_poses, camera_model = load_optimization_result(config=config)
    elif config["data_source"] == "dataset":
        world_points, pixel_tracks, image_indices, camera_poses, camera_model = load_dataset(config=config)
            
    global DATETIME_STRING
    DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    campaign = CampaignDate(dataset_config["campaign"])
    config["output_directory"] = join("output", campaign.value, "optimization", config["output_directory"])

    ### create factor-graph for optimization ###

    print("\nConstructing factor-graph...")

    graph = gtsam.NonlinearFactorGraph()

    ### insert priors for camera poses ###
    
    optimization_config: Dict = config.get("optimization", {})
    
    if optimization_config["priors_on_camera_poses"]:
        
        print("'--> Inserting priors on camera poses...")

        for i, image_index in tqdm(enumerate(image_indices), total=len(image_indices)):
            
            pose_camera = camera_poses[i]
            if pose_camera["image_index"] != image_index:
                raise ValueError("INS groundtruth and image do not match")
            
            key_camera = gtsam.symbol("C", i)
            P3_cam = gtsam.Pose3(pose_camera["pose"])
            
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas([
                *np.deg2rad(optimization_config["camera_pose_orientation_noise_degree"]),
                *optimization_config["camera_pose_position_noise"]
            ])
            
            graph.add(gtsam.PriorFactorPose3(key_camera, P3_cam, pose_noise))

    ### insert priors for landmarks ###

    if optimization_config["priors_on_landmarks"]:
        
        print("'--> Inserting priors on landmarks...")

        for pid, point in tqdm(world_points.items()):
            
            if not pid.isdigit():
                raise ValueError("Point ID is expected to be an integer")
            key_landmark = gtsam.symbol("W", int(pid))
            P3_landmark = gtsam.Point3(*point)
            
            landmark_noise = gtsam.noiseModel.Diagonal.Sigmas([
                *optimization_config["landmark_position_noise"]
            ])
            
            graph.add(gtsam.PriorFactorPoint3(key_landmark, P3_landmark, landmark_noise))

    ### insert prior for camera calibration ###

    key_calibration = gtsam.symbol("K", 0)

    K0 = gtsam.Cal3DS2(
        fx=camera_model.K[0, 0],
        fy=camera_model.K[1, 1],
        s=0.,
        u0=camera_model.K[0, 2],
        v0=camera_model.K[1, 2],
        k1=camera_model.dist[0],
        k2=camera_model.dist[1],
        p1=camera_model.dist[2],
        p2=camera_model.dist[3]
    )
    
    if optimization_config["prior_on_camera_calibration"]:
        
        print("'--> Inserting prior on camera calibration...")
        progress_bar = tqdm(total=1)
    
        calibration_noise = gtsam.noiseModel.Diagonal.Sigmas([
            optimization_config["camera_calibration_prior_noise"]["fx"],
            optimization_config["camera_calibration_prior_noise"]["fy"],
            optimization_config["camera_calibration_prior_noise"]["skew"],
            optimization_config["camera_calibration_prior_noise"]["cx"],
            optimization_config["camera_calibration_prior_noise"]["cy"],
            optimization_config["camera_calibration_prior_noise"]["k1"],
            optimization_config["camera_calibration_prior_noise"]["k2"],
            optimization_config["camera_calibration_prior_noise"]["p1"],
            optimization_config["camera_calibration_prior_noise"]["p2"]
        ])
        
        graph.add(gtsam.PriorFactorCal3DS2(key_calibration, K0, calibration_noise))
        
        progress_bar.update(1)
        progress_bar.close()

    ### insert SFM factors ###
    
    print("'--> Inserting SFM factors...")
    
    reprojection_noise = load_reprojection_kernel(config=config)
    
    num_observations = sum([len(obs) for obs in pixel_tracks.values()])
    progress_bar = tqdm(total=num_observations)
    
    for pid, observations in pixel_tracks.items():
        
        if not pid.isdigit():
            raise ValueError("Point ID is expected to be an integer")
        key_landmark = gtsam.symbol("W", int(pid))
        
        for image_index, (u, v) in observations:
            
            local_pose_index = image_indices.index(image_index)
            key_camera = gtsam.symbol("C", local_pose_index)
            
            measurement = gtsam.Point2(float(u), float(v))
            
            factor = gtsam.GeneralSFMFactor2Cal3DS2(
                measurement,
                reprojection_noise,
                key_camera,
                key_landmark,
                key_calibration
            )
            
            graph.add(factor)
            progress_bar.update(1)

    progress_bar.close()

    ### create inital estimates ###
    
    print("\nInserting initial values...")
    
    values = gtsam.Values()
    values.insert(key_calibration, K0)
    
    ### initial camera pose values ###
    
    for i, image_index in enumerate(image_indices):
            
        pose_camera = camera_poses[i]
        if pose_camera["image_index"] != image_index:
            raise ValueError("INS groundtruth and image do not match")
        
        key_camera = gtsam.symbol("C", i)
        P3_cam = gtsam.Pose3(pose_camera["pose"])
        
        values.insert(key_camera, P3_cam)
        
    ### initial landmark values ###

    for pid, point in world_points.items():
        
        if not pid.isdigit():
            raise ValueError("Point ID is expected to be an integer")
        key_landmark = gtsam.symbol("W", int(pid))
        P3_landmark = gtsam.Point3(*point)
        
        values.insert(key_landmark, P3_landmark)
        
    ### optimization ###
        
    print("\nRunning optimization...")
        
    result, errors_per_iteration = run_optimization(graph=graph,
        initial_estimates=values, method="Dogleg")
        
    ### visualization ###
    
    if config.get("export_results", False) or config.get("save_plots", False):
        makedirs(config["output_directory"], exist_ok=True)
    
    print("\nVisualizing results...\n")
    
    print_camera_calibration(K=K0.K(), dist=K0.k(), name="Initial camera intrinsics",
        precision=5, total_length=15)
    
    K1 = result.atCal3DS2(key_calibration)
    print_camera_calibration(K=K1.K(), dist=K1.k(), name="Optimized camera intrinsics",
        precision=5, total_length=15)
    
    show_errors_per_iteration(errors_per_iteration=errors_per_iteration, config=config)
    show_pose_changes(camera_poses=camera_poses, image_indices=image_indices, values=result, config=config)
    show_landmark_changes(world_points=world_points, values=result, config=config)

    if config.get("show_plots", True):
        plt.show()
    
    print("Exporting optimization results as .json file...")
    
    ### export results ###
    
    if config.get("export_results", False):
    
        camera_poses: Dict[int, List[List[float]]] = {}
        
        for i, image_index in enumerate(image_indices):
            
            key_camera = gtsam.symbol("C", i)
            P_cam_est: np.ndarray = result.atPose3(key_camera).matrix()
            
            camera_poses[image_index] = P_cam_est.tolist()
            
        _world_points: Dict[str, List[float]] = {}
        
        for pid in world_points.keys():
            
            if not pid.isdigit():
                raise ValueError("Point ID is expected to be an integer")
            key_landmark = gtsam.symbol("W", int(pid))
            
            landmark_est: np.ndarray = result.atPoint3(key_landmark)
            _world_points[pid] = landmark_est.tolist()
        
        camera_calibration = {
            "fx": result.atCal3DS2(key_calibration).K()[0, 0],
            "fy": result.atCal3DS2(key_calibration).K()[1, 1],
            "cx": result.atCal3DS2(key_calibration).K()[0, 2],
            "cy": result.atCal3DS2(key_calibration).K()[1, 2],
            "k1": result.atCal3DS2(key_calibration).k()[0],
            "k2": result.atCal3DS2(key_calibration).k()[1],
            "p1": result.atCal3DS2(key_calibration).k()[2],
            "p2": result.atCal3DS2(key_calibration).k()[3],
            "skew": result.atCal3DS2(key_calibration).K()[0, 1]
        }
        
        results = {
            "image_indices": image_indices,
            "camera_poses": camera_poses,
            "world_points": _world_points,
            "pixel_tracks": pixel_tracks,
            "camera_calibration": camera_calibration
        }
        
        filename = f"{config['output_prefix']}_{DATETIME_STRING}.json"
        
        with open(join(config["output_directory"], filename), "w+") as fout:
            json.dump(results, fout, indent=4)
            
        print(f"\nWrote optimization results to {join(config['output_directory'], filename)}")

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
