import os
import time
import torch
import numpy as np
import dill
import json

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
from tqdm import tqdm
from typing import List, Dict

from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.agent import AgentType
from trajdata.utils.arr_utils import batch_nd_transform_points_angles_pt, transform_matrices, batch_nd_transform_points_pt
from trajdata.utils.arr_utils import PadDirection, transform_matrices

from diffstack.data.trajdata_lanes import LanesList
from diffstack.utils import visualization as plan_vis
from diffstack.utils.utils import batch_derivative_of, angle_wrap, collate

from diffstack.modules.predictors.trajectron_utils.environment import Environment


standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def prepare_cache_to_avdata(rank, hyperparams, args, diffstack):
    """
    Loads cache file used for Corl22 paper and converts to AgentBatch.

    The data is identical apart from numerical differences which was verified by manually
    comparing inputs/outputs for 
    - `corl22_unified_reproduce` branch commit 3703d7adb94f98a749465fc7dec7da4f4a2bb2e3
    - `clean` branch commit 4edf81c4ea742a862e9600558864d193ff9346c8
    I used the local `./cache/nuScenes_mini_val.pkl.6.mpc1.20.20.v6.cached.data.pkl` file.
    Randomly picked inputs were the same, and planning metrics were very close over the 
    validation data when using the `Unified Train GTpred Fan-MPC manualdata compare` launch
    config, 
        --predictor=gt 
        --planner=fan_mpc
        --history_sec=4.0
        --prediction_sec=3.0
        --augment_input_noise=0.0

    Fan related metrics:
    fan_converged 100: 0.8839       | plan.fan_valid 100: 0.8839
    plan_class_hcost 100: 0.8021    | plan.class_hcost 100: 0.8023
    
    Mpc related metrics: difference is larger but this could be due to small 
    numerical differences that have a large influence on mpc.
    plan_hcost 100: 0.7412          | plan.hcost 100: 0.7417
    lan_converged 100: 0.6161       | plan.converged 100: 0.5893

    It is hard to reproduce results with T++ training because the random seeds will differ.
    """
    _, train_split = hyperparams["train_data"].split('-')
    _, eval_split = hyperparams["eval_data"].split('-')
    cached_train_data_path = os.path.join(
        os.path.expanduser(hyperparams["cached_data_dir"]),
        f"cached_nuScenes_{train_split}.pkl")
    cached_eval_data_path = os.path.join(
        os.path.expanduser(hyperparams["cached_data_dir"]), 
        f"cached_nuScenes_{eval_split}.pkl")

    if not os.path.exists(cached_eval_data_path):
        raise ValueError(f"No file: {cached_eval_data_path}")
    if not os.path.exists(cached_train_data_path):
        raise ValueError(f"No file: {cached_train_data_path}")

    # # Load training data.
    with open(cached_train_data_path, 'rb') as f:
        train_dataset = dill.load(f)
        train_dataset = list(train_dataset)[0]

    # Load eval data
    with open(cached_eval_data_path, 'rb') as f:
        eval_dataset = dill.load(f)
        eval_dataset = list(eval_dataset)[0]

    # Filter data
    filter_fn = get_filter_func(
        node_type=train_dataset.node_type, 
        plan_node_types=hyperparams["plan_node_types"],
        get_neighbor_idx_for_planning_fn=get_neighbor_idx_for_planning,
        plan_valid=hyperparams["filter_plan_valid"], 
        plan_converged=hyperparams["filter_plan_converged"], plan_relevant=hyperparams["filter_plan_relevant"],
        lane_near=hyperparams["filter_lane_near"])
    if filter_fn is not None:
        train_dataset.filter(filter_fn, verbose=(rank == 0))

    filter_fn = get_filter_func(
        node_type=eval_dataset.node_type, 
        plan_node_types=hyperparams["plan_node_types"],
        get_neighbor_idx_for_planning_fn=get_neighbor_idx_for_planning,
        legacy_valid_set=True)
    eval_dataset.filter(filter_fn, verbose=(rank == 0))

    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=hyperparams["world_size"],
            rank=rank
        )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=hyperparams["world_size"],
        rank=rank
    )

    def my_collate(*args):
        manual_batch = collate(*args)
        agent_batch = convert_manual_batch_to_agentbatch(manual_batch, hyperparams=hyperparams)
        return agent_batch

    train_dataloader = DataLoader(train_dataset,
                                        collate_fn=my_collate,
                                        pin_memory=False if hyperparams['device'] == 'cpu' else True,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=False,
                                        num_workers=hyperparams['preprocess_workers'],
                                        sampler=train_sampler)

    eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=my_collate,
                                pin_memory=False if hyperparams['device'] == 'cpu' else True,
                                batch_size=hyperparams['eval_batch_size'],
                                shuffle=False,
                                num_workers=hyperparams['preprocess_workers'],
                                sampler=eval_sampler)

    input_wrapper = lambda batch: {"batch": batch}

    return train_dataloader, train_sampler, train_dataset, eval_dataloader, eval_sampler, eval_dataset, input_wrapper


def get_filter_func(node_type, plan_node_types, get_neighbor_idx_for_planning_fn, plan_valid=False, plan_converged=False, plan_relevant=False, min_history_len=None, lane_near=False, legacy_valid_set=False):
    """ 
    sample: single non-batched input from dataset
    return: True if sample should be kept in dataset
    """

    # shortcut no filtering
    if not plan_valid and not plan_converged and not plan_relevant and min_history_len is None and not legacy_valid_set:
        return None
        
    def fn(sample):
        (first_history_index,
            x_t, y_t, x_st_t, y_st_t,
            neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
            neighbors_edge_value,
            robot_traj_st_t,
            map, neighbors_future_data, plan_data) = sample

        # Prediction reletad filters
        if legacy_valid_set and x_t.shape[1] - first_history_index < 8:
            # TODO this is only kept to reproduce validation set in the paper. x_t.shape is used incorrectly
            #      x_t.shape[1]=8 because its state dimensions, min_history_len=8, so in effect this requires first_history_index==0
            return False

        if min_history_len is not None and x_t.shape[0] - first_history_index < min_history_len:
            return False

        # Planning related filters
        if node_type not in plan_node_types:
            # Don't filter if we don't plan for this node type.
            return True
        if plan_valid:
            ni = get_neighbor_idx_for_planning_fn(plan_data)
            if ni < 0:
                return False
            # TODO temp fix for preprocessing not filtering invalid futures
            vehicle_future_f = neighbors_future_data[(str(node_type), 'VEHICLE')]
            if vehicle_future_f[int(ni)].shape[1]>=9 and torch.isnan(vehicle_future_f[int(ni)][:, :(4+2+3)]).any():
                return False      
        if plan_converged:
            # _, _, _, _, _, _, _, _, gtplan_x, gtplan_u, gtplan_converged = self.decode_plan_inputs(plan_data=plan_data.unsqueeze(0))  # dummy batch
            gtplan_converged = plan_data['gt_plan_converged']
            if gtplan_converged < 0.5:
                return False
        if plan_relevant:
            regret = plan_data["nopred_plan_hcost"] - plan_data["gt_plan_hcost"]
            if regret <= 0.:  # 0.001:
                return False
        if lane_near:
            raise NotImplementedError()
        return True
    return fn


def get_neighbor_idx_for_planning(plan_data, plan_agent_choice="most_relevant"):
    # Chose plan_node, context_nodes.
    if plan_agent_choice == "most_relevant":
        plan_neigbors = plan_data["most_relevant_idx"]
        # plan_neigbors = plan_data[..., 0].int()
    elif plan_agent_choice == "ego":
        plan_neigbors = plan_data["robot_idx"]
        # plan_neigbors = plan_data[..., 1].int()
    else:
        raise ValueError("Unknown plan_agent_choice: {}".format(plan_agent_choice))
    return plan_neigbors


def standardized_manual_state(x, x_origin, agent_type: str, dt: float, only2d=False):
    attention_radius = dict()
    attention_radius["PEDESTRIAN"] = 20.0
    attention_radius["VEHICLE"] = 30.0    
    state_dict = {
      "PEDESTRIAN": {
        "position": ["x", "y"],
        "velocity": ["x", "y"],
        "acceleration": ["x", "y"],
        # "augment": ["ego_indicator"]      
      },
      "VEHICLE": {
        "position": ["x", "y"],
        "velocity": ["x", "y"],
        "acceleration": ["x", "y"],
        "heading": ["°", "d°"],
        # "augment": ["ego_indicator"]      
      }
    }

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization, dt=dt) 
    std_dict = {}
    _, std_dict["PEDESTRIAN"] = env.get_standardize_params(state_dict["PEDESTRIAN"], "PEDESTRIAN")   
    _, std_dict["VEHICLE"] = env.get_standardize_params(state_dict["VEHICLE"], "VEHICLE")   

    std = np.array(std_dict[agent_type], dtype=np.float32)
    std[0:2] = attention_radius[agent_type]
    rel_state = np.zeros_like(x, dtype=np.float32)
    if only2d:
        rel_state_dims = 2
    else:
        rel_state_dims = np.min((x.shape[-1],  x_origin.shape[-1]))

    rel_state[:, :rel_state_dims] = x_origin[:rel_state_dims]
    x_standardized = (x - rel_state) / std[None, :x.shape[-1]]
    return x_standardized


def unstandardized_manual_state(x_standardized, x_origin, agent_type: str, dt: float, only2d=False):
    attention_radius = dict()
    attention_radius["PEDESTRIAN"] = 20.0
    attention_radius["VEHICLE"] = 30.0    
    state_dict = {
      "PEDESTRIAN": {
        "position": ["x", "y"],
        "velocity": ["x", "y"],
        "acceleration": ["x", "y"],
        # "augment": ["ego_indicator"]      
      },
      "VEHICLE": {
        "position": ["x", "y"],
        "velocity": ["x", "y"],
        "acceleration": ["x", "y"],
        "heading": ["°", "d°"],
        # "augment": ["ego_indicator"]      
      }
    }

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization, dt=dt) 
    std_dict = {}
    _, std_dict["PEDESTRIAN"] = env.get_standardize_params(state_dict["PEDESTRIAN"], "PEDESTRIAN")   
    _, std_dict["VEHICLE"] = env.get_standardize_params(state_dict["VEHICLE"], "VEHICLE")   

    std = np.array(std_dict[agent_type], dtype=np.float32)
    std[0:2] = attention_radius[agent_type]
    rel_state = np.zeros_like(x_standardized, dtype=np.float32)
    if only2d:
        rel_state_dims = 2
    else:
        rel_state_dims = np.min((x_standardized.shape[-1],  x_origin.shape[-1]))

    rel_state[:, :rel_state_dims] = x_origin[:rel_state_dims]
    x_global = x_standardized * std[None] + rel_state
    return x_global


def check_consistent(states, agent_type:str, dt):
    if agent_type == "VEHICLE":
        x, y, vx, vy, ax, ay, h, dh = torch.unbind(states[..., :8], dim=-1)
        h_from_v = torch.atan2(vy, vx)
        h_err = ((h-h_from_v + np.pi) % (2*np.pi) - np.pi).abs()
    else:
        x, y, vx, vy, ax, ay = torch.unbind(states[..., :6], dim=-1)
        h_err = torch.zeros((1, ))

    delta_x = (x[1:] - x[:-1]) / dt
    delta_y = (y[1:] - y[:-1]) / dt
    vx_err = (vx[1:]-delta_x)
    vy_err = (vy[1:]-delta_y)

    print ("h", h_err, h_err.max())
    print ("vx", vx_err, vx_err.max())
    print ("vy", vy_err, vy_err.max())
    return h_err, vx_err, vy_err


def convert_manual_hist_to_trajdata_hist(traj, agent_type: AgentType):
    if agent_type == AgentType.VEHICLE:
        # "VEHICLE": { "position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"], "heading": ["°", "d°"], "augment": ["ego_indicator"]      
        # History
        x, y, vx, vy, ax, ay, h, dh = torch.unbind(traj[..., :8], dim=-1)
        trajdata_hist = torch.stack((x, y, vx, vy, ax, ay, torch.sin(h), torch.cos(h)), dim=-1)
    elif agent_type == AgentType.PEDESTRIAN:
        # PEDESTRIAN:  "position": ["x", "y"], "velocity": ["x", "y"],  "acceleration": ["x", "y"], "augment": ["ego_indicator"]      
        # History.  There is no heading so recover it from vx/vy            
        x, y, vx, vy, ax, ay = torch.unbind(traj[..., :6], dim=-1)
        h = torch.atan2(vy, vx)
        trajdata_hist = torch.stack((x, y, vx, vy, ax, ay, torch.sin(h), torch.cos(h)), dim=-1)         
    else:
        assert False    

    return trajdata_hist


def convert_trajdata_hist_to_manual_hist(traj, agent_type: AgentType, dt: float):
    if agent_type == AgentType.VEHICLE:
        # "VEHICLE": { "position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"], "heading": ["°", "d°"], "augment": ["ego_indicator"]      
        # History
        x, y, vx, vy, ax, ay, sinh, cosh = torch.unbind(traj, dim=-1)
        h = angle_wrap(torch.atan2(sinh, cosh))
        dh = angle_wrap(batch_derivative_of(h[..., None], dt)).squeeze(-1)
        # Set dh to zero for steps that are nan
        dh[torch.logical_not(h.isnan())] = torch.nan_to_num(dh, 0.)[torch.logical_not(h.isnan())]
        manual_hist = torch.stack((x, y, vx, vy, ax, ay, h ,dh), dim=-1)
    elif agent_type == AgentType.PEDESTRIAN:
        # PEDESTRIAN:  "position": ["x", "y"], "velocity": ["x", "y"],  "acceleration": ["x", "y"], "augment": ["ego_indicator"]      
        # History.  There is no heading so recover it from vx/vy            
        x, y, vx, vy, ax, ay, sinh, cosh = torch.unbind(traj, dim=-1)
        manual_hist = torch.stack((x, y, vx, vy, ax, ay), dim=-1)         
    else:
        assert False    

    return manual_hist


def convert_manual_fut_to_trajdata_fut(traj, agent_type: AgentType, ph:int, dt:float, num_lane_points=16):
    """
    traj: [..., T, future_state_dim]
    """
    if agent_type == AgentType.VEHICLE:
        x_fut, u_gt, lanes, x_proj, u_t_fitted_dh, u_t_fitted_a, lane_points = torch.split(
                traj, (4, 2, 3, 2, ph+1, ph+1, num_lane_points*3), dim=-1,)
        x, y, h, v = torch.unbind(x_fut, dim=-1)
        vx = v * torch.cos(h)
        vy = v * torch.sin(h)
        ax = batch_derivative_of(vx[:, None], dt).squeeze(1)
        ay = batch_derivative_of(vy[:, None], dt).squeeze(1)
        trajdata_fut = torch.stack((x, y, vx, vy, ax, ay, torch.sin(h), torch.cos(h)), dim=-1)
    elif agent_type == AgentType.PEDESTRIAN:
        # There is no heading, nor velocity, so recover these.
        x, y = torch.unbind(traj, dim=-1)
        vx = batch_derivative_of(x[..., None], dt).squeeze(-1)
        vy = batch_derivative_of(y[..., None], dt).squeeze(-1)
        h = torch.atan2(vy, vx)
        ax = batch_derivative_of(vx[..., None], dt).squeeze(-1)
        ay = batch_derivative_of(vy[..., None], dt).squeeze(-1)
        trajdata_fut = torch.stack((x, y, vx, vy, ax, ay, torch.sin(h), torch.cos(h)), dim=-1)    
        lanes = None
        lane_points = None            
    else:
        assert False    

    return trajdata_fut, lanes, lane_points


class ManualInputsList(list):
    def __to__(self, device, non_blocking=False):
        # keep on cpu
        return self


def origin_to_tf(origin_xyh):
    translate_mat = transform_matrices(
        angles=torch.zeros_like(origin_xyh[..., 2]), 
        translations=-origin_xyh[..., :2])

    rot_mat = transform_matrices(
        angles=-origin_xyh[..., 2], 
        translations=None)

    trans_rot_mat = torch.bmm(rot_mat, translate_mat)  # first translate then rotate

    return trans_rot_mat, rot_mat


def transform_states_xyvvaahh(traj_xyvvaahh: torch.Tensor, origin_xyh: torch.Tensor) -> torch.Tensor:
    """
    traj_xyvvaahh: [..., state_dim] where state_dim = [x, y, vx, vy, ax, ay, sinh, cosh]
    """
    tf_mat, rot_mat = origin_to_tf(origin_xyh)

    xy, vv, aa, hh = torch.split(traj_xyvvaahh, (2, 2, 2, 2), dim=-1)
    xy = batch_nd_transform_points_pt(xy, tf_mat)
    vv = batch_nd_transform_points_pt(vv, rot_mat)
    aa = batch_nd_transform_points_pt(aa, rot_mat)
    # hh: sinh, cosh instead of cosh, sinh, so we use flip
    hh = batch_nd_transform_points_pt(hh.flip(-1), rot_mat).flip(-1)

    return torch.concat((xy, vv, aa, hh), dim=-1)

def transform_agentbatch_coordinate_frame(agent_batch: AgentBatch, origin_xyh: torch.Tensor, extras_transform_fn: Dict) -> AgentBatch:
    """
    Args:
        origin_xyh: desired origin (x, y, heading) defined in the current coordinate frame.
    """
    if agent_batch.maps is not None:
        raise NotImplementedError

    tf_mat, _ = origin_to_tf(origin_xyh)

    return AgentBatch(
        data_idx=agent_batch.data_idx,
        scene_ts=agent_batch.scene_ts,
        dt=agent_batch.dt,
        agent_name=agent_batch.agent_name,
        agent_type=agent_batch.agent_type,
        curr_agent_state=agent_batch.curr_agent_state,  # this is always defined in the `global` coordinate frame for some reason
        agent_hist=transform_states_xyvvaahh(agent_batch.agent_hist, origin_xyh),
        agent_hist_extent=agent_batch.agent_hist_extent,
        agent_hist_len=agent_batch.agent_hist_len,
        agent_fut=transform_states_xyvvaahh(agent_batch.agent_fut, origin_xyh),
        agent_fut_extent=agent_batch.agent_fut_extent,
        agent_fut_len=agent_batch.agent_fut_len,
        num_neigh=agent_batch.num_neigh,
        neigh_types=agent_batch.neigh_types,
        neigh_hist=transform_states_xyvvaahh(agent_batch.neigh_hist, origin_xyh),
        neigh_hist_extents=agent_batch.neigh_hist_extents,
        neigh_hist_len=agent_batch.neigh_hist_len,
        neigh_fut=transform_states_xyvvaahh(agent_batch.neigh_fut, origin_xyh),
        neigh_fut_extents=agent_batch.neigh_fut_extents,
        neigh_fut_len=agent_batch.neigh_fut_len,
        robot_fut=transform_states_xyvvaahh(agent_batch.robot_fut, origin_xyh)
            if agent_batch.robot_fut is not None
            else None,
        robot_fut_len=agent_batch.robot_fut_len,
        maps=agent_batch.maps,  # TODO
        maps_resolution=agent_batch.maps_resolution,  # TODO
        rasters_from_world_tf=agent_batch.rasters_from_world_tf, # TODO
        agents_from_world_tf=torch.bmm(tf_mat, agent_batch.agents_from_world_tf),  # TODO test
        scene_ids=agent_batch.scene_ids,
        history_pad_dir=agent_batch.history_pad_dir,
        extras={
            key: extras_transform_fn[key](val)
            for key, val in agent_batch.extras.items()},
    )    

def convert_manual_batch_to_agentbatch(manual_data_batch: List, hyperparams: Dict, agent_centric_frame=False, pad_direction=PadDirection.AFTER) -> AgentBatch:

    (first_history_index,
        x_t, y_t, x_st_t, y_st_t,
        neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
        neighbors_edge_value,
        robot_traj_st_t,
        map, neighbors_future_data, plan_data) = manual_data_batch

    batch_size = x_st_t.shape[0]
    ph = hyperparams["prediction_horizon"]
    dt = hyperparams["dt"]
    num_lane_points = 16
    agent_hist_len = x_st_t.shape[1] - first_history_index
    assert list(neighbors_data_st.keys())[0][0] == "VEHICLE"
    agent_type = torch.tensor([AgentType.VEHICLE] * batch_size)

    # Combine different edge neighbors into a single list
    # Deep copy
    neigh_data_st = [list(temp) for temp in neighbors_data_st[("VEHICLE", "VEHICLE")]]
    neigh_fut_data = [list(temp) for temp in neighbors_future_data[("VEHICLE", "VEHICLE")]]
    neigh_types_list = [[AgentType.VEHICLE] * len(neigh_data_st[b_i]) for b_i in range(batch_size)]
    for b_i in range(batch_size):
        neigh_data_st[b_i].extend(neighbors_data_st[("VEHICLE", "PEDESTRIAN")][b_i])
        neigh_fut_data[b_i].extend(neighbors_future_data[("VEHICLE", "PEDESTRIAN")][b_i])
        neigh_types_list[b_i].extend([AgentType.PEDESTRIAN] * len(neighbors_data_st[("VEHICLE", "PEDESTRIAN")][b_i]))
    num_neigh = torch.tensor([len(neigh) for neigh in neigh_data_st])
    max_num_neigh = torch.max(num_neigh).item()
    # We dont need to orgainze edge values into a single list because it is already combined for vehicles and pedestrians
    neigh_edge_list = list(neighbors_edge_value[("VEHICLE", "VEHICLE")])

    # Unstandardize neigh_data
    neigh_data = [[] for _ in range(batch_size)]
    x_origin_batch = x_t[:, -1, :].cpu().numpy()
    for b_i in range(batch_size):
        # # For agent history
        x_t_conv = unstandardized_manual_state(x_st_t[b_i], x_origin_batch[b_i], AgentType.VEHICLE.name, dt, only2d=True)
        assert torch.logical_or(torch.isclose(x_t_conv, x_t[b_i]), torch.isnan(x_t_conv)).all()
        # check_consistent(x_t_conv, "VEHICLE", dt)

        for n_i in range(len(neigh_data_st[b_i])):
            x_conv = unstandardized_manual_state(neigh_data_st[b_i][n_i], x_origin_batch[b_i], AgentType(neigh_types_list[b_i][n_i]).name, dt, only2d=False)
            # check_consistent(x_conv, AgentType(neigh_types[b_i][n_i]).name, dt)
            # Check current pose is the same in history and future
            assert torch.isclose(x_conv[-1, :2], neigh_fut_data[b_i][n_i][0, :2]).all()
            neigh_data[b_i].append(x_conv)
            
    # Move plan agent to first neighbor
    robot_ind = torch.full((batch_size, ), -1, dtype=torch.int)
    for b_i in range(batch_size):
        plan_i = plan_data['most_relevant_idx'][b_i]
        if plan_i >= 0:
            neigh_data[b_i] = [neigh_data[b_i][plan_i]] + [
                neigh_data[b_i][n_i] 
                for n_i in range(len(neigh_data[b_i])) if n_i != plan_i]
            neigh_fut_data[b_i] = [neigh_fut_data[b_i][plan_i]] + [
                neigh_fut_data[b_i][n_i] 
                for n_i in range(len(neigh_fut_data[b_i])) if n_i != plan_i]
            robot_ind[b_i] = 0
            neigh_edge_list[b_i] = [neigh_edge_list[b_i][plan_i]] + [
                neigh_edge_list[b_i][n_i] 
                for n_i in range(len(neigh_edge_list[b_i])) if n_i != plan_i]

    # Convert to agentbatch tensors
    neigh_hist = torch.full((batch_size, max_num_neigh, hyperparams["maximum_history_length"]+1, 8), torch.nan)
    neigh_fut = torch.full((batch_size, max_num_neigh, ph, 8), torch.nan)
    neigh_hist_len = torch.full((batch_size, max_num_neigh), 0, dtype=torch.int64)
    neigh_fut_len = torch.full((batch_size, max_num_neigh), 0, dtype=torch.int64)
    neigh_types = torch.full((batch_size, max_num_neigh), -1, dtype=torch.int)
    neigh_edge = torch.full((batch_size, max_num_neigh), torch.nan)

    lanes_batch = torch.full((batch_size, ph+1, 3), torch.nan)
    lane_points_batch = torch.full((batch_size, ph+1, num_lane_points, 3), torch.nan)

    for b_i in range(len(neigh_data)):
        for n_i in range(len(neigh_data[b_i])):
            # Convert state representation - TODO move out function, apply to agent states too.
            trajdata_hist = convert_manual_hist_to_trajdata_hist(neigh_data[b_i][n_i], neigh_types_list[b_i][n_i])
            trajdata_fut, lanes, lane_points = convert_manual_fut_to_trajdata_fut(
                neigh_fut_data[b_i][n_i], neigh_types_list[b_i][n_i], ph, dt)

            # Find actual history/future length. Invalid states are represented by zeros.
            invalid_t_mask = torch.logical_or(
                torch.isclose(trajdata_hist[:, :2], torch.zeros(()), atol=1e-4).all(dim=1),
                torch.isnan(trajdata_hist[:, 0])
            )
            invalid_t_count = invalid_t_mask.sum()
            trajdata_hist = trajdata_hist[invalid_t_count:]
            invalid_t_mask = torch.logical_or(
                torch.isclose(trajdata_fut[:, :2], torch.zeros(()), atol=1e-4).all(dim=1),
                torch.isnan(trajdata_fut[:, 0])
            )                
            invalid_t_count = invalid_t_mask.sum()
            trajdata_fut = trajdata_fut[:trajdata_fut.shape[0]-invalid_t_count]
            
            # Use padding_direction.AFTER 
            if pad_direction == pad_direction.AFTER:
                neigh_hist[b_i, n_i, :trajdata_hist.shape[0]] = trajdata_hist
            else:
                neigh_hist[b_i, n_i, neigh_hist.shape[2]-trajdata_hist.shape[0]:] = trajdata_hist
            neigh_fut[b_i, n_i, :trajdata_fut[1:].shape[0]] = trajdata_fut[1:]
            neigh_hist_len[b_i, n_i] = trajdata_hist.shape[0]
            neigh_fut_len[b_i, n_i] = trajdata_fut.shape[0]
            neigh_types[b_i, n_i] = neigh_types_list[b_i][n_i]
            neigh_edge[b_i, n_i] = neigh_edge_list[b_i][n_i]

            if n_i == robot_ind[b_i]:
                lanes_batch[b_i] = lanes

                lane_points = lane_points.reshape(list(lane_points.shape[:-1]) + [num_lane_points, 3])
                lane_points_batch[b_i] = lane_points
    
    agent_hist = convert_manual_hist_to_trajdata_hist(x_t, AgentType.VEHICLE)
    # Agenet future is only xy, same as the history for a pedestrian, so we
    # purposefully use the history converter function with PEDESTRIAN the future of a VEHICLE agent.
    agent_fut, _, _ = convert_manual_fut_to_trajdata_fut(y_t, AgentType.PEDESTRIAN, ph, dt)

    # Convert to PadDirection.AFTER
    if pad_direction == PadDirection.AFTER:
        for b_i in range(batch_size):
            new_hist = torch.full_like(agent_hist[b_i], torch.nan)
            new_hist[:agent_hist_len[b_i]] = agent_hist[b_i, (agent_hist.shape[1]-agent_hist_len[b_i]):]
            agent_hist[b_i] = new_hist

    curr_agent_state_world = agent_hist[torch.arange(batch_size), agent_hist_len-1]
    curr_agent_state_world[:, :2] += plan_data["scene_offset"]

    agents_from_world_tf = transform_matrices(
        angles=torch.zeros((batch_size, )),
        translations=-plan_data["scene_offset"],
    )

    lanes_near_goal = plan_data["most_relevant_nearby_lanes"]  # shallow copy
    
    extras = {
        "robot_ind": robot_ind,
        "goal": neigh_fut[:, 0, -1, :],  # xyvvaahh
        "lane_projection_points": lanes_batch,  # xyh
        "lanes_near_goal": LanesList(lanes_near_goal),  # list of xyh
        "neigh_edge_weight": neigh_edge,
        "manual_inputs": ManualInputsList(manual_data_batch),
    }

    agent_batch = AgentBatch(
        data_idx=torch.full((batch_size,), np.nan),
        scene_ts=torch.tensor([0] * batch_size),
        dt=torch.tensor([hyperparams["dt"]] * batch_size),
        agent_name=["dummy"] * batch_size,
        agent_type=agent_type,
        curr_agent_state=curr_agent_state_world,
        agent_hist=agent_hist,
        agent_hist_len=agent_hist_len,
        agent_fut=agent_fut,
        agent_fut_len=torch.tensor([hyperparams["prediction_horizon"]] * batch_size, dtype=torch.int),        
        agent_hist_extent=torch.full((batch_size,), np.nan),
        agent_fut_extent=torch.full((batch_size,), np.nan),
        num_neigh=num_neigh,
        neigh_types=neigh_types,
        neigh_hist=neigh_hist,
        neigh_hist_len=neigh_hist_len,
        neigh_fut=neigh_fut,
        neigh_fut_len=neigh_fut_len,
        neigh_hist_extents=torch.full((batch_size,), np.nan),
        neigh_fut_extents=torch.full((batch_size,), np.nan),
        robot_fut=None,
        robot_fut_len=None,
        maps=None,
        map_names=None,
        vector_maps=None,
        maps_resolution=None,
        rasters_from_world_tf=torch.full((batch_size,), np.nan),
        agents_from_world_tf=agents_from_world_tf,
        scene_ids=[None for _ in range(batch_size)],
        history_pad_dir=pad_direction,
        extras=extras,
    )

    # # Convert everything to agent centric.
    if agent_centric_frame:
        agent_state_t = agent_hist[torch.arange(batch_size), agent_hist_len-1]
        origin_xyh = torch.concat((
            agent_state_t[:, :2],
            torch.atan2(agent_state_t[:, -2], agent_state_t[:, -1]).unsqueeze(-1)  # h = atan2(sinh, cosh)
        ), dim=-1)
        tf_mat, rot_mat = origin_to_tf(origin_xyh)

        extras_tf_fn = {
            "robot_ind": lambda x: x,
            "goal": lambda x: transform_states_xyvvaahh(x, origin_xyh),
            "lane_projection_points": lambda x: 
                batch_nd_transform_points_angles_pt(x, tf_mat),
            "lanes_near_goal": lambda xlistbatch: LanesList(
                [[batch_nd_transform_points_angles_pt(x, tf_mat[b_i]) for x in xlistbatch[b_i]] for b_i in range(batch_size)])  
        }

        agent_batch = transform_agentbatch_coordinate_frame(agent_batch, origin_xyh, extras_tf_fn)

    return agent_batch

