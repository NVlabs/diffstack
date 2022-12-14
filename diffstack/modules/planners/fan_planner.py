import torch
import functools

from diffstack.modules.planners.fan_planner_utils import SplinePlanner
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

import matplotlib.pyplot as plt

VISUALIZE = False
USE_CPU = True


class FanPlanner(torch.nn.Module):

    def __init__(self, ph, dt, device):
        super().__init__()
        self.spline_planner = SplinePlanner(device=('cpu' if USE_CPU else device), dt=dt)
        self.ph = ph
        self.dt = dt
        self.scenes = None
    
    @staticmethod
    @functools.lru_cache(maxsize=None) 
    def get_map(raw_data_path, map_name):
        return NuScenesMap(dataroot=raw_data_path, map_name=map_name)

    def forward(self, x_init_batch, x_goal_batch, cost_obj, dyn_obj, cost_inputs, relevant_lanes_batch=None, candidate_trajs_batch=None, is_valid_batch=None):
        device = x_init_batch.device
        batch_size = x_init_batch.shape[0]
        plan_info_dict = {'traj_xu': [], 'traj_cost': []}

        if candidate_trajs_batch is not None:
            # Optionall pass in candidates directly
            assert len(candidate_trajs_batch) == batch_size
            assert len(is_valid_batch) == batch_size
            
            fan_ctrl_xu_batch = candidate_trajs_batch
            is_valid_batch = is_valid_batch

        else:
            assert len(relevant_lanes_batch) == batch_size
            fan_ctrl_xu_batch, is_valid_batch = self.get_candidate_trajectories(x_init_batch, relevant_lanes_batch) 

        # Get costs
        traj_cost_batch = self.get_cost_for_trajs(fan_ctrl_xu_batch, cost_obj, cost_inputs)
        plan_info_dict['traj_xu'] =  fan_ctrl_xu_batch  # intentionally dont detach because we will backprop through these
        plan_info_dict['traj_cost'] = traj_cost_batch   # intentionally dont detach because we will backprop through these
        
        # Choose the lowest cost as a default planning output
        #   but we also return all candidates and all costs for differnet loss functions.
        plan_xu_batch = []
        plan_cost_batch = []
        for batch_i in range(batch_size):
            fan_ctrl_xu = plan_info_dict['traj_xu'][batch_i]
            traj_cost = traj_cost_batch[batch_i]
            if is_valid_batch[batch_i]:
                best_ind = torch.argmin(traj_cost, dim=0)
                plan_cost = traj_cost[best_ind]
                plan_xu = fan_ctrl_xu[best_ind]
            else:
                plan_cost = traj_cost.squeeze(0)
                plan_xu = fan_ctrl_xu.squeeze(0)

            plan_xu_batch.append(plan_xu)
            plan_cost_batch.append(plan_cost)

        plan_xu = torch.stack(plan_xu_batch, dim=1)  # T, b, 6
        plan_x, plan_u = torch.split(plan_xu, (4, 2), dim=-1)
        plan_cost = torch.stack(plan_cost_batch, dim=0)
        if not isinstance(is_valid_batch, torch.Tensor):
            is_valid_batch = torch.tensor(is_valid_batch, device=device).bool()

        return plan_x, plan_u, plan_cost, is_valid_batch, plan_info_dict

    def get_candidate_trajectories(self, x_init_batch, relevant_lanes_batch):
        device = x_init_batch.device
        batch_size = x_init_batch.shape[0]

        x, y, h, v = torch.unbind(x_init_batch, dim=-1)
        x_init_xyvh_batch = torch.stack([x, y, v, h], dim=-1)

        if USE_CPU:
            relevant_lanes_batch = [[pts.to('cpu') for pts in relevant_lanes] for relevant_lanes in relevant_lanes_batch]
            x_init_xyvh_batch = x_init_xyvh_batch.to('cpu')

        # Split over batch
        fan_ctrl_xu_batch = []
        is_converged_batch = []
        for batch_i in range(batch_size):
            relevant_lanes = relevant_lanes_batch[batch_i]
            if len(relevant_lanes)==0:
                fan_ctrl_xu_batch.append(torch.zeros((1, self.ph+1, 6), device=device).detach())
                is_converged_batch.append(False)          
                continue

            x_init_xyvh = x_init_xyvh_batch[batch_i].unsqueeze(0)
            fan_trajs, _ = self.spline_planner.gen_trajectory_batch(x_init_xyvh, self.ph * self.dt, relevant_lanes) 
            assert len(fan_trajs) == 1
            fan_trajs = fan_trajs[0]

            if fan_trajs.shape[0] > 0:
                xy = fan_trajs[..., :2]
                vel = fan_trajs[..., 2:3]
                acce = fan_trajs[..., 3:4]
                yaw = fan_trajs[..., 4:5]
                yaw_rate = fan_trajs[..., 5:6]
                fan_xu = torch.cat((xy, yaw, vel, yaw_rate, acce), -1)  # (N, T+1, xu=6)
            else:
                fan_xu = torch.zeros((0, self.ph+1, 8), dtype=torch.float, device=device)  # (N, T+1, xu)

            if USE_CPU:
                # Move back to the gpu
                # fan_trajs = fan_trajs.to(device)
                fan_xu = fan_xu.to(device)

            num_candidates = fan_xu.shape[0]
            if num_candidates == 0:
                # Skip if there are no valid candidates.
                # This happens quite frequently e.g. at low velocities, where
                # reaching points from the lane center are dynamically infeasible
                fan_ctrl_xu_batch.append(torch.zeros((1, self.ph+1, 6), device=device))
                is_converged_batch.append(False)                          
                continue

            # Convert fan xu splines to state and control.
            #   The control part of the spline is the acceleration and steering at time t, not the same as control command.
            #   We can best track the trajectory by commanding at t for the target control as (u_t + u_{t+1})/2
            #   In a previous version of the code u[t] = u[t+1] was used
            fan_x, fan_u = torch.split(fan_xu, (4, 2), dim=-1)
            ctrl_u = torch.cat(((fan_u[:,:-1] + fan_u[:, 1:])*0.5, fan_u[:, -1:]), dim=1)  # last control doesnt matter
            fan_ctrl_xu = torch.cat((fan_x, ctrl_u), dim=-1)
            fan_ctrl_xu_batch.append(fan_ctrl_xu)
            is_converged_batch.append(True)

        return fan_ctrl_xu_batch, is_converged_batch  # TODO there is no need to return trees

    def get_cost_for_trajs(self, fan_ctrl_xu_batch, cost_obj, cost_inputs):

        batch_size = len(fan_ctrl_xu_batch)
        gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, lane_points = cost_inputs

        traj_cost_batch = []
        for batch_i in range(batch_size):
            fan_ctrl_xu = fan_ctrl_xu_batch[batch_i]
            num_candidates = fan_ctrl_xu.shape[0]
            cost_inputs_i =  (None if gt_neighbors_batch is None else gt_neighbors_batch[:, :, batch_i].unsqueeze(2).tile((1, 1, num_candidates, 1)), 
                              None if mus_batch is None else mus_batch[:, :, batch_i].unsqueeze(2).tile((1, 1, num_candidates, 1, 1)), 
                              None if probs_batch is None else probs_batch[:, batch_i].unsqueeze(1).tile((1, num_candidates, 1)), 
                              goal_batch[batch_i].unsqueeze(0).tile((num_candidates, 1)),  
                              lanes[:, batch_i].unsqueeze(1).tile((1, num_candidates, 1)), 
                              lane_points[:, batch_i].unsqueeze(1).tile((1, num_candidates, 1, 1)) if lane_points is not None else None, 
                              )

            traj_cost = cost_obj(fan_ctrl_xu.transpose(1, 0), cost_inputs_i)  # T, b
            traj_cost = torch.sum(traj_cost, dim=0)  # b,
            traj_cost_batch.append(traj_cost)

        return traj_cost_batch
        
