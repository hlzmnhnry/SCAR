import numpy as np

from collections import defaultdict
from typing import List

def filter_points_with_few_observations(pixel_coords: dict, world_points: dict,
    minimum_observations: int):

    filtered_pixel_coords = {}
    filtered_world_points = {}

    for point_id, observations in pixel_coords.items():
        
        if len(observations) >= minimum_observations and point_id in world_points:
            
            filtered_pixel_coords[point_id] = observations
            filtered_world_points[point_id] = world_points[point_id]

    return filtered_pixel_coords, filtered_world_points

def filter_frames_with_few_features(pixel_coords: dict, world_points: dict,
    minimum_features_per_frame: int):

    frame_counts = defaultdict(int)

    for observations in pixel_coords.values():
        for appearance in observations:
            frame_id = appearance[0]
            frame_counts[frame_id] += 1

    # get frames that have at least 'minimum_features_per_frame' features
    valid_frames = {frame_id for frame_id, count in frame_counts.items() if count >= minimum_features_per_frame}
    removed_frames = sorted(set(frame_counts.keys()) - valid_frames)

    filtered_pixel_coords = {}
    filtered_world_points = {}

    for point_id, observations in pixel_coords.items():
        
        # for each point, only keep observations that are in valid frames
        filtered_observations = [obs for obs in observations if obs[0] in valid_frames]
        
        # only keep point if it has any observation left
        if filtered_observations:
            filtered_pixel_coords[point_id] = filtered_observations
            filtered_world_points[point_id] = world_points[point_id]

    return filtered_pixel_coords, filtered_world_points, removed_frames

def filter_frames_in_range(pixel_coords: dict, world_points: dict, image_index_start: int,
    image_index_end: int):
    
    filtered_pixel_coords = {}
    filtered_world_points = {}

    for point_id, observations in pixel_coords.items():
        
        # filter observations to be within frame range
        filtered_observations = [
            obs for obs in observations
            if image_index_start <= obs[0] <= image_index_end
        ]

        # keep only if at least one observation is in the range
        if filtered_observations:
            filtered_pixel_coords[point_id] = filtered_observations
            filtered_world_points[point_id] = world_points[point_id]

    return filtered_pixel_coords, filtered_world_points

def remap_ids_sequentially(pixel_coords: dict, world_points: dict):

    new_pixel_coords = {}
    new_world_points = {}
    
    sorted_old_ids = sorted(pixel_coords.keys(), key=int)
    id_mapping = {str(old_id): str(new_id) for new_id, old_id in enumerate(sorted_old_ids)}

    for old_id, new_id in id_mapping.items():
        new_pixel_coords[new_id] = pixel_coords[old_id]
        new_world_points[new_id] = world_points[old_id]

    return new_pixel_coords, new_world_points, id_mapping

def load_image_indices(pixel_coords: dict):
    
    image_indices: List = []

    for pixel_point in pixel_coords.values():
        for appearance in pixel_point:
            image_indices.append(appearance[0])

    return sorted(np.unique(image_indices).tolist())

def to_framewise_pixel_and_world_points(pixel_coords: dict, world_points: dict,
    image_indices: List, return_point_ids: bool = False):
    
    # create an ordererd list for each frame, listing all pixels/points/ids
    pixel_coords_framewise = [[] for _ in range(len(image_indices))]
    world_points_framewise = [[] for _ in range(len(image_indices))]
    point_ids_framewise = [[] for _ in range(len(image_indices))]

    for point_id in world_points.keys():
        
        world_point = world_points[point_id]
        
        for appearance in pixel_coords[point_id]:
            
            assert len(appearance) == 2, "Expected: frame id and pixel coordinate"

            image_index = appearance[0]
            pixel_point = appearance[1]

            # get correct list to insert to
            frame_index = image_indices.index(image_index)

            pixel_coords_framewise[frame_index].append(pixel_point)
            world_points_framewise[frame_index].append(world_point)
            point_ids_framewise[frame_index].append(point_id)

    if return_point_ids:
        return pixel_coords_framewise, world_points_framewise, point_ids_framewise

    return pixel_coords_framewise, world_points_framewise
