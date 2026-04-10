import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import makedirs
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from scipy.spatial.transform import Rotation
from os.path import join, isfile, isdir, dirname, basename

from pygeon.datasets.vgl_iterator import VGLIterator    # type: ignore
from pygeon.enums import FrameType, CampaignDate, is_valid_campaign_date_str
from pygeon.types import Extrinsics, CameraModel

from scar import load_image_indices, to_framewise_pixel_and_world_points, inverse_transformation, \
    draw_reprojection_errors, extract_timestamp, error_statistics, create_paper_reprojection_image

WINDOW_NAME = "REPROJECTION"
DATETIME_STRING = None
VIDEO_WRITER = None
NUMBER_FRAMES = 0

def is_config_valid(config: Dict):
    """check fields of config dictionary"""
    
    if config["data_source"] not in ["dataset", "optimization_result"]:
        raise ValueError("Unknown 'data_source' for reprojection evaluation, choose either 'dataset' or 'optimization_result'")

    if "output_directory" not in config:
        raise ValueError("No output folder name provided")

    ### dataset config ###

    dataset_config = config["dataset"]
    
    if not is_valid_campaign_date_str(dataset_config["campaign"]):
        raise ValueError(f"'{dataset_config['campaign']}' is not a supported campaign date")

    return True

def deep_update(base, update):
    """overwrite base config entries"""
    
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v

    return base

def load_file_from_type(config: Dict | str):
    
    if isinstance(config, str):
        return config # filename
    
    input_file_type = config["type"]
    input_filename = None
    
    if input_file_type not in ["path", "load_latest_from_prefix"]:
        raise ValueError("Unknown file input type")
    
    if input_file_type == "load_latest_from_prefix":
    
        glob_directory = config["input_directory"]
        if not isdir(glob_directory):
            raise ValueError(f"'{glob_directory}' is not a directory")
        
        prefix = config["prefix"]
        possible_files = list(glob(join(glob_directory, f"{prefix}_*.json")))
        
        if len(possible_files) == 0:
            raise ValueError(f"There is no file with prefix '{prefix}' in '{glob_directory}'")
        
        input_filename = max(possible_files, key=extract_timestamp)
    
    elif input_file_type == "path":
        input_filename = config["path"]
    
    if not isfile(input_filename):
        raise ValueError(f"'{input_filename}' does not exist")
    
    return input_filename

def read_opencv_yaml_matrix(path: str, key: str):
    
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    
    if not fs.isOpened():
        raise IOError(f"Cannot open {path}")
    
    node = fs.getNode(key)
    
    if node.empty():
        fs.release()
        raise KeyError(f"Key '{key}' not found in {path}")
    
    mat = node.mat()
    fs.release()
    
    return mat

def read_camera_calibration(config: Dict):
    
    camera_matrix, distortion = None, None
    
    if "camera_calibration" in config: # default
        
        calibration = config["camera_calibration"]
        
        camera_matrix = np.array([
            [calibration["fx"], 0., calibration["cx"]],
            [0., calibration["fy"], calibration["cy"]],
            [0., 0., 1.]
        ])
        
        distortion = np.array(
            [calibration["k1"], calibration["k2"], calibration["p1"], calibration["p2"]]
        )
        
    elif "model" in config: # COLMAP output file
        
        assert config["model"] == "CameraModelId.OPENCV", "Only 'OPENCV' camera model supported"

        camera_matrix = np.array([
            [config["fx"], 0., config["cx"]],
            [0., config["fy"], config["cy"]],
            [0., 0., 1.]
        ])
        
        distortion = np.array(
            [config["k1"], config["k2"], config["p1"], config["p2"]]
        )
        
    return camera_matrix, distortion

def show_reprojection_summary(projection_errors: List[float], framewise_errors: Dict[int, List[float]],
    pointwise_errors: Dict[int, List[float]], config: Dict):
    
    global DATETIME_STRING
    
    # framewise mean errors
    frame_indices = sorted(framewise_errors.keys())
    frame_means = [np.mean(framewise_errors[i]) for i in frame_indices]
    
    if not frame_indices:
        return

    # pointwise mean errors
    point_ids = sorted(pointwise_errors.keys())
    point_means = [np.mean(pointwise_errors[pid]) for pid in point_ids]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
    fig.canvas.manager.set_window_title("Reprojection error summary")
    
    ax1: plt.Axes = axes[0]
    ax2: plt.Axes = axes[1]
    ax3: plt.Axes = axes[2]

    ax1.plot(frame_indices, frame_means, marker="o", c="teal")
    ax1.axhline(np.mean(frame_means), color="r", linestyle="--", label="Mean")
    ax1.set_xticks(frame_indices[::len(frame_indices)//10])
    ax1.set_title("Mean reprojection error per frame")
    ax1.set_ylabel("Error (px)")
    ax1.grid(True, axis="y")
    ax1.legend()

    ax2.plot(point_ids, point_means, marker="x", c="coral")
    ax2.axhline(np.mean(point_means), color="r", linestyle="--", label="Mean")
    ax2.set_xticks(point_ids[::len(point_ids)//10])
    ax2.set_title("Mean reprojection error per point")
    ax2.set_xlabel("Point ID")
    ax2.set_ylabel("Error (px)")
    ax2.grid(True, axis="y")
    ax2.legend()
    
    ax3.hist(projection_errors, bins=50, color="gold", edgecolor="black")
    ax3.axvline(np.mean(projection_errors), color="r", linestyle="--", label="Mean")
    ax3.set_title("Histogram of all reprojection errors")
    ax3.set_xlabel("Error (px)")
    ax3.set_ylabel("Count")
    ax3.grid(True, axis="y")
    ax3.legend()

    for ax in [ax1, ax2, ax3]:
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontweight("bold")
        offset_text.set_color("red")

    fig.suptitle("Reprojection error summary")
    fig.tight_layout()
    
    if config.get("save_plots", False):
        output_prefix = config.get("output_prefix", "")
        fig.savefig(join(config["output_directory"],
            f"{output_prefix}{'_' if len(output_prefix) > 0 else ''}reprojection_summary_{DATETIME_STRING}.png"), dpi=300)
    
    if config.get("show_plots", True):
        fig.show()

def show_reprojections(config: Dict, campaign: CampaignDate, image_index: int, points: np.ndarray,
    projected_points: np.ndarray, point_ids: np.ndarray):
    
    global DATETIME_STRING

    image_path = join(config["dataset"]["data_directory"], f"{campaign.value}", "images", f"{image_index}.png")
    assert isfile(image_path), f"Image file '{image_path}' could not be found"
    
    image_capture = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    
    if config.get("create_paper_reprojection_image", False):
        create_paper_reprojection_image(image=image_capture, points=points,
            projected_points=projected_points, image_index=image_index, vmin=0., vmax=50.)
    
    visualization_image = draw_reprojection_errors(image=image_capture, points=points,
        projected_points=projected_points, point_ids=point_ids)
    
    video_config: Dict = config.get("video", {})
    
    if video_config.get("generate_video", False):
        
        global VIDEO_WRITER, NUMBER_FRAMES
        
        output_prefix = config.get("output_prefix", "")
        filename = f"{output_prefix}{'_' if len(output_prefix) > 0 else ''}reprojection_{DATETIME_STRING}.mp4"

        video_path = join(config["output_directory"], filename)
        maximum_number_frames = video_config.get("maximum_number_frames", np.inf)
        
        if len(dirname(video_path)) > 0:
            makedirs(dirname(video_path), exist_ok=True)

        if VIDEO_WRITER is None and NUMBER_FRAMES <= maximum_number_frames:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
            fps = video_config.get("fps", 10)
            height, width = visualization_image.shape[:2]
            VIDEO_WRITER = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        if NUMBER_FRAMES <= maximum_number_frames:
            VIDEO_WRITER.write(visualization_image)
            NUMBER_FRAMES += 1
        elif VIDEO_WRITER is not None:
            print(f"\nReached {maximum_number_frames} frames, closing video...")
            VIDEO_WRITER.release()
            VIDEO_WRITER = None
    
    cv2.imshow(WINDOW_NAME, visualization_image)
    cv2.waitKey(1)

def evaluate_dataset(config: Dict):
    
    dataset_config: Dict[str, Any] = config["dataset"]
    sequence_config: Dict[str, Any] = config["sequence"]
    campaign = CampaignDate(dataset_config["campaign"])
    
    print(f"\nLoading campaign {campaign.value} data...")
    
    image_directory = Path(dataset_config["image_directory"]) \
        if "image_directory" in dataset_config else None
    data_directory = Path(dataset_config["data_directory"]) \
        if "data_directory" in dataset_config else None

    ### read calibration data ###
        
    camera_calibration_file, extrinsics_calibration_file = None, None
    
    if "camera_calibration_file" not in dataset_config:
        raise ValueError("'camera_calibration_file' not in dataset config")
    
    if "extrinsics_calibration_file" not in dataset_config:
        raise ValueError("'read_extrinsics_from_file' not in in config")
    
    camera_calibration_file = load_file_from_type(config=dataset_config["camera_calibration_file"])
    print(f"\nReading camera calibration from {camera_calibration_file}...")

    extrinsics_calibration_file = load_file_from_type(config=dataset_config["extrinsics_calibration_file"])
    print(f"\nReading extrinsics from {extrinsics_calibration_file}...")
    
    if camera_calibration_file is None or extrinsics_calibration_file is None:
        raise ValueError("Calibrations could not be loaded successfully")
    
    camera_calibration_file: str
    extrinsics_calibration_file: str
    
    camera_calibration_directory = Path(camera_calibration_file).parent \
        if "colmap_sfm" not in camera_calibration_file and not basename(camera_calibration_file).startswith("ls_") else Path("config/")
    camera_file_basename = Path(camera_calibration_file).parts[-1] \
        if "colmap_sfm" not in camera_calibration_file and not basename(camera_calibration_file).startswith("ls_") else "camera_matrix_guess_kalibr.json"
    
    extrinsics_calibration_directory = Path(extrinsics_calibration_file).parent \
        if extrinsics_calibration_file.endswith(".json") and not basename(extrinsics_calibration_file).startswith("ls_") else Path("config/")
    extrinsics_file_basename = Path(extrinsics_calibration_file).parts[-1] \
        if extrinsics_calibration_file.endswith(".json") and not basename(extrinsics_calibration_file).startswith("ls_") else "extrinsics_guess_kalibr.json"
    
    it = VGLIterator(
        campaign=campaign,
        data_directory=data_directory,
        image_directory=image_directory,
        frame=FrameType.CAMERA,
        image_indexed_naming=True,
        with_landcover=False,
        with_image=False,
        camera_calibration_directory=camera_calibration_directory,
        camera_file_basename=camera_file_basename,
        extrinsics_calibration_directory=extrinsics_calibration_directory,
        extrinsics_file_basename=extrinsics_file_basename
    )
    
    if extrinsics_calibration_file.endswith(".yaml"): # VINS
        T_bc = read_opencv_yaml_matrix(extrinsics_calibration_file, key="body_T_cam0")
        T_cb = inverse_transformation(T_bc)
        it.ext = Extrinsics(T_bc=T_bc, T_cb=T_cb)
    elif basename(extrinsics_calibration_file).startswith("ls_"):
        T_bc = np.array(json.load(open(extrinsics_calibration_file))["T_ins_cam"])
        T_cb = inverse_transformation(T_bc)
        it.ext = Extrinsics(T_bc=T_bc, T_cb=T_cb)
    
    if "colmap_sfm" in camera_calibration_file or basename(camera_calibration_file).startswith("ls_"): # COLMAP or opt. result
        cal = json.load(open(camera_calibration_file))
        cal = cal["camera_calibration"] if basename(camera_calibration_file).startswith("ls_") else cal
        K = np.array([[cal["fx"], 0., cal["cx"]], [0., cal["fy"], cal["cy"]], [0., 0., 1.]], dtype=float)
        dist = np.array([cal["k1"], cal["k2"], cal["p1"], cal["p2"]], dtype=float)
        it.cam = CameraModel(K=K, dist=dist, size=(1600, 1100))    
    
    ### load sequence data and check sanity ###
    
    pixel_tracks, world_points = None, None
    
    if sequence_config.get("read_points_from_result_file", False):
        
        if "points_file" not in sequence_config:
            raise ValueError("Set 'read_points_from_result_file' to true, but no 'points_file' in config")

        points_file_path = load_file_from_type(config=config["points_file"])
        print(f"\nReading points from {points_file_path}...")
        
        pixel_tracks: dict = json.load(open(points_file_path))["pixel_tracks"]
        world_points: dict = json.load(open(points_file_path))["world_points"]
        
    else:
        input_folder = join("output", campaign.value, "geofeatures")
        pixel_tracks: dict = json.load(open(join(input_folder, f"PT_{campaign.value}_{config['sequence']['sequence_id']}.json")))
        world_points: dict = json.load(open(join(input_folder, f"WP_{campaign.value}_{config['sequence']['sequence_id']}.json")))

    if pixel_tracks is None:
        raise ValueError("Pixel coordinates could not be loaded successfully")
    
    if world_points is None:
        raise ValueError("World points could not be loaded successfully")
    
    if len(pixel_tracks) != len(world_points):
        raise ValueError("Each world point must have at least one observation")
    
    pc_point_ids = [int(point_id) for point_id in pixel_tracks.keys()]
    wp_point_ids = [int(point_id) for point_id in world_points.keys()]
    
    if not (np.unique(pc_point_ids).tolist() == pc_point_ids and np.unique(wp_point_ids).tolist() == wp_point_ids):
        raise ValueError("Point IDs are not unique (either in PC or in WP file)")
    
    pc_maximum_id, wp_maximum_id = max(pc_point_ids), max(wp_point_ids)
    
    if not (pc_maximum_id == wp_maximum_id and wp_maximum_id == len(world_points) - 1):
        raise ValueError(f"ID mixup in input sequence, 'pc_maximum_id'={pc_maximum_id}, 'wp_maximum_id'={wp_maximum_id}")
    
    num_points = len(world_points)
    observations_per_points = [len(obs) for obs in pixel_tracks.values()]
    
    print(f"\nLoaded {num_points} points from sequence '{config['sequence']['sequence_id']}'")
    print(f"Total of {sum(observations_per_points)} observations")

    ### visualize reprojections frame-wise and calculate errors ###

    image_indices: List = load_image_indices(pixel_coords=pixel_tracks)

    pixel_tracks_FW, world_points_fw, point_ids_fw = to_framewise_pixel_and_world_points(
        pixel_tracks, world_points, image_indices, return_point_ids=True)

    # save all individual reprojection errors    
    all_reprojection_errors: List[float] = []
    
    # save reprojection errors per frame and world point
    framewise_errors: Dict[int, List[float]] = {}
    pointwise_errors: Dict[int, List[float]] = {}

    for i, image_ts in enumerate(image_indices):
        
        objectPoints = np.array(world_points_fw[i], dtype=np.float32)
        sample = it.get_by_timestamp(ts_us=image_ts, nearest_only=True, max_dt_us=0)
        
        pose = inverse_transformation(sample["pose"])
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec(degrees=False).reshape(3, 1)
        tvec = pose[:3, 3].reshape(3, 1)
        
        projected_points: np.ndarray = cv2.projectPoints(
            objectPoints=objectPoints,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=it.cam.K,
            distCoeffs=it.cam.dist,
        )[0]
        
        projected_points = projected_points.squeeze(axis=1)
        imagePoints = np.array(pixel_tracks_FW[i], dtype=np.float32)
        
        error: np.ndarray = np.linalg.norm(projected_points - imagePoints, axis=1)
        all_reprojection_errors.extend(error.tolist())
        
        framewise_errors[image_ts] = error.tolist()

        for j, pid in enumerate(point_ids_fw[i]):
            pointwise_errors.setdefault(int(pid), []).append(float(error[j]))
        
        print(f"\n######### 'image_ts' = {image_ts} #########")
        print(f"mean reprojection error = {np.mean(error):.4f} px")
        
        if config["visualize_projection"]:
            # raise NotImplementedError
            show_reprojections(config=config, image_index=sample["image_index"], points=imagePoints,
                projected_points=projected_points, point_ids=point_ids_fw[i], campaign=campaign)
            
    return all_reprojection_errors, framewise_errors, pointwise_errors

def evaluate_optimization_result(config: Dict):

    optimization_config: Dict[str, Any] = config["optimization"]
    
    ### load optimized data ###
    
    input_filename = load_file_from_type(config=optimization_config["result_input_file"])
    
    with open(input_filename, "r") as fin:
        results = json.load(fin)
        
    ### visualize reprojections frame-wise and calculate errors ###
    
    print(f"\nReading optimization result from {input_filename}...")

    image_indices = results["image_indices"]
    pixel_tracks = results["pixel_tracks"]
    
    if load_image_indices(pixel_coords=pixel_tracks) != image_indices:
        raise ValueError("There is an inconsistency in the provided image indices and pixel coordinates")
    
    camera_poses = results["camera_poses"]
    world_points = results["world_points"]
    
    if results["camera_calibration"]["skew"] > 1e-6:
        raise ValueError("Skew cannot be visualized")
    camera_matrix, distortion = read_camera_calibration(config=results)

    pixel_tracks_fw, world_points_fw, point_ids_fw = to_framewise_pixel_and_world_points(
        pixel_tracks, world_points, image_indices, return_point_ids=True)
    
    # save all individual reprojection errors    
    all_reprojection_errors: List[float] = []
    
    # save reprojection errors per frame and world point
    framewise_errors: Dict[int, List[float]] = {}
    pointwise_errors: Dict[int, List[float]] = {}
    
    for i, image_index in enumerate(image_indices):

        objectPoints = np.array(world_points_fw[i], dtype=np.float32)
        camera_pose = np.asarray(camera_poses[str(image_index)], dtype=float)
        camera_pose = inverse_transformation(camera_pose)

        rvec = Rotation.from_matrix(camera_pose[:3, :3]).as_rotvec(degrees=False).reshape(3, 1)
        tvec = camera_pose[:3, 3].reshape(3, 1)
        
        projected_points: np.ndarray = cv2.projectPoints(
            objectPoints=objectPoints,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
        )[0]
        
        projected_points = projected_points.squeeze(axis=1)
        imagePoints = np.array(pixel_tracks_fw[i], dtype=np.float32)
        
        error: np.ndarray = np.linalg.norm(projected_points - imagePoints, axis=1)
        all_reprojection_errors.extend(error.tolist())
        
        framewise_errors[image_index] = error.tolist()

        for j, pid in enumerate(point_ids_fw[i]):
            pointwise_errors.setdefault(int(pid), []).append(float(error[j]))
        
        print(f"\n######### 'image_index' = {image_index} #########")
        print(f"mean reprojection error = {np.mean(error):.4f} px")
        
        if config["visualize_projection"]:
            raise NotImplementedError
            show_reprojections(config=config, image_index=image_index, points=imagePoints,
                projected_points=projected_points, point_ids=point_ids_fw[i])
            
    return all_reprojection_errors, framewise_errors, pointwise_errors

def main():
    
    parser = argparse.ArgumentParser(description="SCAR data reprojection", add_help=True)
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
    
    global DATETIME_STRING
    DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    
    campaign = CampaignDate(dataset_config["campaign"])
    config["output_directory"] = join("output", campaign.value, "reprojection", config["output_directory"])
    
    if config.get("visualize_projection", False):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    if config.get("export_results", False) or config.get("save_plots", False):
        makedirs(config["output_directory"], exist_ok=True)
    
    errors, fw_errors, pw_errors = None, None, None
    
    if config["data_source"] == "optimization_result":
        raise NotImplementedError
        errors, fw_errors, pw_errors = evaluate_optimization_result(config=config)
    elif config["data_source"] == "dataset":
        errors, fw_errors, pw_errors = evaluate_dataset(config=config)
    
    statistics = error_statistics(errors=np.array(errors).flatten())
    
    print(f"\nMEAN REPROJECTION ERROR ACROSS ALL POINTS: {np.mean(errors):.4f} px\n")

    results = {
        "errors_statistics": statistics,
        "errors": errors,
        "framewise_errors": fw_errors,
        "pointwise_errors": pw_errors
    }
    
    if "outlier_thresholds" in config:
        
        errors = np.asarray(errors)
        filtered_errors = {}
        
        for t in config["outlier_thresholds"]:
            filtered = errors[errors <= t]
            if len(filtered) > 0:
                filtered_error = float(np.mean(filtered))
                filtered_errors[str(t)] = filtered_error
                print(f"For treshold outlier rejection 'error <= {t}': reprojection error = {filtered_error:.4f}")
            else:
                print(f"No points that fulfill 'error <= {t}'")
                
        results["filtered_errors"] = filtered_errors
    
    show_reprojection_summary(projection_errors=errors, framewise_errors=fw_errors,
        pointwise_errors=pw_errors, config=config)
    
    if config.get("show_plots", True):
        plt.show()
    
    cv2.destroyAllWindows()

    global VIDEO_WRITER

    if VIDEO_WRITER is not None:
        print("\nClosing video...")
        VIDEO_WRITER.release()
        VIDEO_WRITER = None
        
    if config.get("export_results", False):
        
        output_prefix = config.get("output_prefix", "")
        filename = f"{output_prefix}{'_' if len(output_prefix) > 0 else ''}reprojection_{DATETIME_STRING}.json"
        
        with open(join(config["output_directory"], filename), "w+") as fout:
            json.dump(results, fout, indent=4)

        print(f"\nWrote reprojection results to {join(config['output_directory'], filename)}")

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
