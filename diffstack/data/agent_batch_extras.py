import torch
import numpy as np
from typing import Dict, Iterable, Union

from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.agent import AgentType

from diffstack.utils.utils import move_list_element_to_front
from diffstack.data.trajdata_lanes import get_goal_lanes, get_lane_projection_points, LanesList


def robot_selector(element: AgentBatchElement):
    # Find most relevant neighbor
    dists = []
    inds = []
    for n_i in range(len(element.neighbor_futures)):
        # Filter vehicles
        if element.neighbor_types_np[n_i] != AgentType.VEHICLE:
            continue

        # Filter incomplete future
        if element.neighbor_futures[n_i].shape[0] <  element.agent_future_np.shape[0]:
            continue

        # Filter parked vehicles
        # We used to do this for all dataset variants EXCEPT for v7
        # 
        # # Implementation 1: compute distance for valid history and future
        # # Agent is considered to be parked if it moves less then 1m from the beginning of history to the end of future.
        # start_to_end_dist = np.linalg.norm(element.neighbor_histories[n_i][0, :2] - element.neighbor_futures[n_i][-1, :2])
        # if start_to_end_dist < 1.:
        #     continue
        #
        # Implementation 2: use pre-computed metainfo based on entire valid trajectory
        if element.neighbor_meta_dicts[n_i]['is_stationary']:
            continue

        # Distance from predicted agent
        dist = np.square(element.agent_future_np[:, :2] - element.neighbor_futures[n_i][:, :2])  # [1:] exclude current state for vehicle future
        dist = np.min(dist.sum(axis=-1), axis=-1)  # sum over states, min over time
        inds.append(n_i)
        dists.append(dist)
        
    if dists:
        plan_i = inds[np.argmin(np.array(dists))]  # neighbor that gets closest to current node
    else:
        # No neighbors or all futures are incomplete
        plan_i = -1     


    element.extras['robot_ind'] = plan_i
    return element


def make_robot_the_first(element: AgentBatchElement):
    """Reorder neighbors such that the first neighbor is the proxy robot agent for planning."""

    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, do nothing
        pass
    else:
        element.neighbor_futures = move_list_element_to_front(element.neighbor_futures, robot_ind)
        element.neighbor_future_lens_np = np.array(move_list_element_to_front(element.neighbor_future_lens_np, robot_ind))
        element.neighbor_future_extents = move_list_element_to_front(element.neighbor_future_extents, robot_ind)
        element.neighbor_histories = move_list_element_to_front(element.neighbor_histories, robot_ind)
        element.neighbor_history_lens_np = np.array(move_list_element_to_front(element.neighbor_history_lens_np, robot_ind))
        element.neighbor_history_extents = move_list_element_to_front(element.neighbor_history_extents, robot_ind)
        element.neighbor_types_np = np.array(move_list_element_to_front(element.neighbor_types_np, robot_ind))
        element.neighbor_meta_dicts = move_list_element_to_front(element.neighbor_meta_dicts, robot_ind)
        element.extras["robot_ind"] = 0
    return element


def remove_parked(element: AgentBatchElement):
    is_parked = np.array([meta_dict['is_stationary'] for meta_dict in element.neighbor_meta_dicts])
    get_filtered_list = lambda x: [x[i] for i in range(element.num_neighbors) if not is_parked[i]]
    get_filtered_np = lambda x: x[np.logical_not(is_parked)]

    element.neighbor_histories = get_filtered_list(element.neighbor_histories)
    element.neighbor_history_extents = get_filtered_list(element.neighbor_history_extents)
    element.neighbor_history_lens_np = get_filtered_np(element.neighbor_history_lens_np)

    element.neighbor_futures = get_filtered_list(element.neighbor_futures)
    element.neighbor_future_extents = get_filtered_list(element.neighbor_future_extents)
    element.neighbor_future_lens_np = get_filtered_np(element.neighbor_future_lens_np)

    element.neighbor_meta_dicts = get_filtered_list(element.neighbor_meta_dicts)
    element.neighbor_types_np = get_filtered_np(element.neighbor_types_np)

    element.num_neighbors = len(element.neighbor_types_np)

    return element


def augment_with_point_goal(element: AgentBatchElement):
    """Add goal information for planning."""

    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy goal info
        goal = np.full((8, ), np.nan, dtype=np.float32)
    else:
        # Goal is the gt state at the end of the planning horizon.
        goal = element.neighbor_futures[robot_ind][-1].astype(np.float32)
    element.extras["goal"] = goal
    return element


def augment_with_goal_lanes(element: AgentBatchElement, goal_to_lane_range: float = 20., max_lateral_dist: float = 4.5, max_heading_delta: float = np.pi/4):
    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy lane info
        goal_lanes = LanesList([])
    else:
        goal_lanes = get_goal_lanes(
            element.vec_map, element.extras['goal'], element.agent_from_world_tf,
            goal_to_lane_range=goal_to_lane_range, max_lateral_dist=max_lateral_dist, max_heading_delta=max_heading_delta)

    element.extras["lanes_near_goal"] = goal_lanes
    return element


def augment_with_lanes(element: AgentBatchElement, make_missing_lane_invalid: bool = True):
    """Add lane information for planning."""

    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy lane info
        lane_projection_points = None
    else:
        lane_projection_points = get_lane_projection_points(
            element.vec_map,
            element.neighbor_histories[robot_ind], element.neighbor_futures[robot_ind], 
            element.agent_from_world_tf)
    if lane_projection_points is None:
        lane_projection_points = np.full((element.agent_future_len + 1, 3), np.nan, dtype=np.float32)
        if make_missing_lane_invalid:
            # We set robot_idx to -1 to indicate that the sample is invalid
            element.extras['robot_ind'] = -1

    element.extras["lane_projection_points"] = lane_projection_points
    
    return element


def get_filter_func(ego_valid=False, pred_near_ego=False, lane_near_ego=False, pred_not_parked=False):
    """ 
    """
    # shortcut no filtering
    if not ego_valid and not pred_near_ego and not lane_near_ego and not pred_not_parked:
        return None
        
    def filter_fn(element: AgentBatchElement) -> bool:
        if ego_valid and element.extras['robot_ind'] < 0:
            return False 
            
        if pred_not_parked:
            # # Implementation 1: compute distance for valid history and future
            # start_to_end_dist = np.linalg.norm(element.agent_history_np[0, :2] - element.agent_future_np[-1, :2])            
            # if start_to_end_dist < 1.:
            #     return False

            # Implementation 2: use pre-computed metainfo based on entire valid trajectory
            if element.agent_meta_dict['is_stationary']:
                return False

        if pred_near_ego:
            # Only keep if the closest distance betwen ego and predicted agent for future steps is under 10 meters.
            assert ego_valid, "Assert: it only make sense to use pred_near_ego if we filter samples with valid ego."
            shorter_future_len = min(element.agent_future_len, element.neighbor_future_lens_np[element.extras['robot_ind']])
            ego_futures = element.neighbor_futures[element.extras['robot_ind']][:shorter_future_len, :2]
            pred_futures = element.agent_future_np[:shorter_future_len, :2]
            dists = np.linalg.norm(ego_futures - pred_futures, axis=-1)
            min_ego_pred_dist = np.min(dists)
            if min_ego_pred_dist > 10.:
                return False
        if lane_near_ego:
            raise NotImplementedError()
        return True
    return filter_fn


