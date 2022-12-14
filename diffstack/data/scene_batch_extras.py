import torch
import numpy as np
from typing import Dict, Iterable, Union, List, Optional

from trajdata.data_structures.batch_element import SceneBatchElement
from trajdata.data_structures.agent import AgentType

from diffstack.utils.utils import move_list_element_to_front
from diffstack.data.trajdata_lanes import get_goal_lanes, get_lane_projection_points, LanesList


def role_selector(element: SceneBatchElement, pred_agent_types: List[AgentType] = (AgentType.VEHICLE, )):
    # Find ego
    agent_names = [agent.name for agent in element.agents]
    ego_i = next(i for i, name in enumerate(agent_names) if name == "ego")

    # Find pred agent that is closest to ego
    dists = []
    inds = []
    for n_i in range(len(element.agent_futures)):
        if n_i == ego_i:
            continue 

        # Filter what agents we want to predict
        if element.agent_types_np[n_i] not in pred_agent_types:
            continue

        # Filter incomplete future
        if element.agent_future_lens_np[n_i] < element.agent_future_lens_np[ego_i]:
            continue

        # Filter parked vehicles
        if element.agent_meta_dicts[n_i]['is_stationary']:
            continue

        # Distance from predicted agent
        dist = np.square(element.agent_futures[ego_i][:, :2] - element.agent_futures[n_i][:, :2])
        dist = np.min(dist.sum(axis=-1), axis=-1)  # sum over states, min over time
        inds.append(n_i)
        dists.append(dist)
        
    if dists:
        pred_i = inds[np.argmin(np.array(dists))]  # neighbor that gets closest to current node
    else:
        # No neighbors or all futures are incomplete
        ego_i = -1
        pred_i = -1     

    element.extras['robot_ind'] = ego_i
    element.extras['pred_agent_ind'] = pred_i
    return element


def make_robot_the_first(element: SceneBatchElement, extras_key: str = "robot_ind"):
    """Reorder neighbors such that the first neighbor is the proxy robot agent for planning."""
    ind = element.extras[extras_key]
    if ind < 0:
        # No robot, do nothing
        pass
    else:
        element.agent_futures = move_list_element_to_front(element.agent_futures, ind)
        element.agent_future_lens_np = np.array(move_list_element_to_front(element.agent_future_lens_np, ind))
        element.agent_future_extents = move_list_element_to_front(element.agent_future_extents, ind)
        element.agent_histories = move_list_element_to_front(element.agent_histories, ind)
        element.agent_history_lens_np = np.array(move_list_element_to_front(element.agent_history_lens_np, ind))
        element.agent_history_extents = move_list_element_to_front(element.agent_history_extents, ind)
        element.agent_types_np = np.array(move_list_element_to_front(element.agent_types_np, ind))
        element.agent_meta_dicts = move_list_element_to_front(element.agent_meta_dicts, ind)
        element.extras[extras_key] = 0
    return element


def remove_parked(element: SceneBatchElement, keep_agent_ind: Optional[int] = None):
    is_parked = np.array([meta_dict['is_stationary'] for meta_dict in element.agent_meta_dicts])
    if keep_agent_ind is not None and keep_agent_ind >= 0:
        is_parked[keep_agent_ind] = False
    get_filtered_list = lambda x: [x[i] for i in range(element.num_agents) if not is_parked[i]]
    get_filtered_np = lambda x: x[np.logical_not(is_parked)]

    element.agent_histories = get_filtered_list(element.agent_histories)
    element.agent_history_extents = get_filtered_list(element.agent_history_extents)
    element.agent_history_lens_np = get_filtered_np(element.agent_history_lens_np)

    element.agent_futures = get_filtered_list(element.agent_futures)
    element.agent_future_extents = get_filtered_list(element.agent_future_extents)
    element.agent_future_lens_np = get_filtered_np(element.agent_future_lens_np)

    element.agent_meta_dicts = get_filtered_list(element.agent_meta_dicts)
    element.agent_types_np = get_filtered_np(element.agent_types_np)

    if element.map_patches is not None:
        element.map_patches = get_filtered_list(element.map_patches)

    element.num_agents = len(element.agent_types_np)
    return element


def augment_with_point_goal(element: SceneBatchElement):
    """Add goal information for planning."""

    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy goal info
        goal = np.full((8, ), np.nan, dtype=np.float32)
    else:
        # Goal is the gt state at the end of the planning horizon.
        goal = element.agent_futures[robot_ind][-1].astype(np.float32)
    element.extras["goal"] = goal
    return element



def augment_with_goal_lanes(element: SceneBatchElement, goal_to_lane_range: float = 20., max_lateral_dist: float = 4.5, max_heading_delta: float = np.pi/4):
    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy lane info
        goal_lanes = LanesList([])
    else:
        goal_lanes = get_goal_lanes(
            element.vec_map, element.extras['goal'], element.centered_agent_from_world_tf,
            goal_to_lane_range=goal_to_lane_range, max_lateral_dist=max_lateral_dist, max_heading_delta=max_heading_delta)

    element.extras["lanes_near_goal"] = goal_lanes
    return element


def augment_with_lanes(element: SceneBatchElement, make_missing_lane_invalid: bool = True):
    """Add lane information for planning."""
    if element.num_agents==0:
        return element
    robot_ind = element.extras['robot_ind']
    if robot_ind < 0:
        # No robot, create dummy lane info
        lane_projection_points = None
    else:
        lane_projection_points = get_lane_projection_points(
            element.vec_map,
            element.agent_histories[robot_ind], element.agent_futures[robot_ind], 
            element.centered_agent_from_world_tf)
    if lane_projection_points is None:
        max_future_len = element.agent_future_lens_np.max()
        lane_projection_points = np.full((max_future_len + 1, 3), np.nan, dtype=np.float32)
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
        
    def filter_fn(element: SceneBatchElement) -> bool:
        robot_ind = element.extras['robot_ind']
        pred_agent_ind = element.extras['pred_agent_ind']

        if ego_valid and (robot_ind < 0 or pred_agent_ind < 0):
            return False 
            
        if pred_not_parked:
            # Implementation 2: use pre-computed metainfo based on entire valid trajectory
            if element.agent_meta_dicts[pred_agent_ind]['is_stationary']:
                return False

        if pred_near_ego:
            # Only keep if the closest distance betwen ego and predicted agent for future steps is under 10 meters.
            assert ego_valid, "Assert: it only make sense to use pred_near_ego if we filter samples with valid ego."
            shorter_future_len = min(element.agent_future_lens_np[pred_agent_ind], element.agent_future_lens_np[robot_ind])
            ego_futures = element.agent_futures[robot_ind][:shorter_future_len, :2]
            pred_futures = element.agent_futures[pred_agent_ind][:shorter_future_len, :2]
            dists = np.linalg.norm(ego_futures - pred_futures, axis=-1)
            min_ego_pred_dist = np.min(dists)
            if min_ego_pred_dist > 10.:
                return False
        if lane_near_ego:
            raise NotImplementedError()
        return True
    return filter_fn
