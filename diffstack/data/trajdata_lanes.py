import torch
import numpy as np

from typing import Dict, Iterable
from time import time
from trajdata.caching.scene_cache import SceneCache

from trajdata.utils.arr_utils import angle_wrap
from trajdata.utils.map_utils import get_polyline_headings
from trajdata.data_structures.collation import CustomCollateData
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.maps.vec_map import VectorMap, RoadLane, Polyline

# from trajdata.maps.map_kdtree import LaneCenterKDTree
from trajdata.utils.arr_utils import (
    batch_nd_transform_points_np,
    batch_nd_transform_angles_np,
    batch_nd_transform_points_angles_np,
    angle_wrap,
)
from diffstack.utils.geometry_utils import batch_proj
from diffstack.utils.utils import convert_state_pred2plan, lane_frenet_features_simple


# # For nuscenes-based implementation
# from nuscenes.map_expansion.map_api import NuScenesMap, locations as nusc_map_names
# from diffstack.data.manual_preprocess_plan_data import get_lane_reference_points


class LanesList(list, CustomCollateData):
    @staticmethod
    def __collate__(elements: list) -> any:
        return LanesList(
            [[torch.as_tensor(pts) for pts in lanelist] for lanelist in elements]
        )

    def __to__(self, device, non_blocking=False):
        # Always keep on cpu
        del device
        del non_blocking
        return self


def get_lane_reference_points_from_polylines(
    global_xyh: np.ndarray,
    lane_polylines_xyh: Iterable[np.ndarray],
    max_num_points: int = 16,
    resolution_meters: float = 1.0,
):
    # Previosly we did an interpolation of 1m. This was important for cost functions based on N ref points,
    # but its irrelevant for cost functions using projection of the expert state onto the lane segment.
    # The projection will remain the same regardless of interpolated points.
    # The only difference would be in headings, which are not interpolated this way. One solution is to do
    # heading interpolation after projection.

    if not lane_polylines_xyh:
        return np.zeros((global_xyh.shape[0], 0, 3), dtype=np.float32)

    # Code v5: keep top-n closest lane points, deprioritize lane points with over 45 heading difference
    pts_xyh = np.concatenate(lane_polylines_xyh, axis=0)

    # Filter distance, set2set [traj, lanept]
    d2mat = np.square(pts_xyh[None, :, :2] - global_xyh[:, None, :2]).sum(-1)
    heading_diff_mat = pts_xyh[None, :, 2] - global_xyh[:, None, 2]
    heading_diff_mat = np.minimum(
        np.abs(angle_wrap(heading_diff_mat)),  # same direction
        np.abs(angle_wrap(heading_diff_mat + np.pi)),
    )  # opposite direction
    # Add a large constant if heading differs by over 45 degrees
    d2mat += (heading_diff_mat > np.pi / 4).astype(d2mat.dtype) * 1e6
    closest_ind = np.argsort(d2mat, axis=-1)[:, :max_num_points]
    pts_xyh = pts_xyh[closest_ind]  # (traj, max_num_points, 4)

    return pts_xyh


def get_lane_projection_points(
    vec_map: VectorMap,
    ego_histories: np.ndarray,
    ego_futures: np.ndarray,
    agent_from_world_tf: np.ndarray,
) -> np.ndarray:
    """Points along a lane that are closest to GT future ego states."""

    if vec_map is None:
        return None
    # Get present and future for the ego agent.
    ego_pres_future = np.concatenate([ego_histories[-1:], ego_futures], axis=0)
    ego_xyhv = convert_state_pred2plan(ego_pres_future)
    world_from_agent_tf = np.linalg.inv(agent_from_world_tf)
    ego_xyh_world = batch_nd_transform_points_angles_np(
        ego_xyhv[:, :3], world_from_agent_tf
    )

    # trajlen = ego_xyh_world.shape[0]
    # if trajlen == 0:
    #     lane_points_xyh_world = []
    # else:
    ego_xyz_world = np.concatenate(
        (ego_xyh_world[:, :2], np.zeros_like(ego_xyh_world[:, :1])), axis=-1
    )
    closest_lanes = vec_map.get_closest_unique_lanes(ego_xyz_world)
    lane_polylines_xyh = [lane.center.points[:, (0, 1, 3)] for lane in closest_lanes]
    lane_points_xyh_world = get_lane_reference_points_from_polylines(
        ego_xyh_world, lane_polylines_xyh, max_num_points=1
    )

    if lane_points_xyh_world.shape[1] == 0:
        # No lanes.
        return None
    else:
        # World to agent coordinates
        lane_points_xyh = batch_nd_transform_points_angles_np(
            lane_points_xyh_world[..., :3], agent_from_world_tf[None]
        )

        lane_projection_points = lane_frenet_features_simple(
            ego_xyhv[..., :3], lane_points_xyh[:, :]
        )

        assert lane_projection_points.shape == (ego_futures.shape[0] + 1, 3)

    return lane_projection_points.astype(np.float32)


def get_goal_lanes(
    vec_map: VectorMap,
    goal_xyvvaahh: np.ndarray,
    agent_from_world_tf: np.ndarray,
    goal_to_lane_range: float = 20.0,
    max_lateral_dist: float = 4.5,
    max_heading_delta: float = np.pi / 4,
):
    assert goal_xyvvaahh.shape[-1] == 8  # xyvvaahh

    goal_xyhv = convert_state_pred2plan(goal_xyvvaahh)
    world_from_agent_tf = np.linalg.inv(agent_from_world_tf)
    goal_xyh_world = batch_nd_transform_points_angles_np(
        goal_xyhv[:3], world_from_agent_tf
    )

    # Find lanes in range
    goal_xyz_world = np.concatenate(
        (goal_xyh_world[:2], np.zeros_like(goal_xyh_world[:1])), axis=-1
    )
    near_lanes = vec_map.get_lanes_within(goal_xyz_world, dist=goal_to_lane_range)
    near_lanes_xyh_world = [lane.center.points[:, (0, 1, 3)] for lane in near_lanes]

    # Filter
    lanes_xyh_world = []
    for lane_xyh in near_lanes_xyh_world:
        delta_x, delta_y, dpsi = batch_proj(goal_xyh_world, lane_xyh)
        if (
            abs(dpsi[0]) < max_heading_delta
            and np.min(np.abs(delta_y)) < max_lateral_dist
        ):
            lanes_xyh_world.append(lane_xyh)

    # World to agent coordinates
    goal_lanes = [
        batch_nd_transform_points_angles_np(lane_xyh, agent_from_world_tf[None])
        for lane_xyh in lanes_xyh_world
    ]

    return LanesList(goal_lanes)
