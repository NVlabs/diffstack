import os
import random
import torch
import time
import numpy as np

from scipy.optimize import minimize
from collections import defaultdict
from typing import Dict, Union, Tuple, Any, Optional, Iterable
from trajdata.data_structures.batch import PadDirection, SceneBatch
from diffstack.utils.utils import batch_select


def compute_ade_pt(predicted_trajs, gt_traj):
    error = torch.linalg.norm(predicted_trajs - gt_traj, dim=-1)
    ade = torch.mean(error, axis=-1)
    return ade.flatten()


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde_pt(predicted_trajs, gt_traj):
    final_error = torch.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[:, -1], dim=-1)
    return final_error.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_nll_pt(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(predicted_dist.log_prob(gt_traj), min=-20.)
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz[0], -log_p_y_xz_final[0]


def compute_nll(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(predicted_dist.log_prob(torch.as_tensor(gt_traj)), min=-20.)
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz[0].numpy(), -log_p_y_xz_final[0].numpy()


def compute_prediction_metrics(prediction_output_dict,
                                futures,
                                y_dists=None,
                                keep_indices=None):
    ade_errors = compute_ade_pt(prediction_output_dict, futures)
    fde_errors = compute_fde_pt(prediction_output_dict, futures)
    if y_dists:
        nll_means, nll_finals = compute_nll_pt(y_dists, futures)

    if keep_indices is not None:
        return {'ade': ade_errors[keep_indices], 
                'fde': fde_errors[keep_indices],
                'nll_mean': nll_means[keep_indices], 
                'nll_final': nll_finals[keep_indices]}
    else:
        return {'ade': ade_errors, 
                'fde': fde_errors,
                'nll_mean': nll_means, 
                'nll_final': nll_finals}


def split_predicted_agent_extents(
    batch: SceneBatch, 
    num_dist_agents: int = 1, 
    num_single_agents: Optional[int] = None, 
    max_num_agents: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get extents from scene batch.
    
    Returns three tensors:
        - ego_extents
        - pred_agent_extents
        - ml_agent_extents
    """
    # TODO it should be the responsibility of the predictor to associate agent to its role.
    # Here we just assume a fixed order
    robot_ind = batch.extras["robot_ind"]
    pred_ind = batch.extras["pred_agent_ind"]
    assert (robot_ind == 0).all() and (pred_ind == 1).all(), "Agent roles are assumed to be hardcoded"
    assert batch.history_pad_dir == PadDirection.AFTER
    if num_single_agents is None:
        if max_num_agents is None:
            raise ValueError("Must specify either num_ml_agents or max_num_agents")
        num_single_agents = max_num_agents + 1 - num_dist_agents

    agent_extent = batch_select(batch.agent_hist_extent, index=batch.agent_hist_len-1, batch_dims=2)  # b, N, t, (length, width)
    ego_extent = agent_extent[:, 0]
    dist_extents = agent_extent[:, 1:1+num_dist_agents]
    dist_extents = torch.nn.functional.pad(dist_extents, (0, 0, 0, num_dist_agents-dist_extents.shape[1]), 'constant', torch.nan)

    single_extents = agent_extent[:, 1+num_dist_agents:(1+num_dist_agents+num_single_agents)]
    single_extents = torch.nn.functional.pad(single_extents, (0, 0, 0, num_single_agents-single_extents.shape[1]), 'constant', torch.nan)
    # ml_extents = agent_extent[:, 2:(2+MAX_PLAN_NEIGHBORS+1)]
    # ml_extents = torch.nn.functional.pad(ml_extents, (0, 0, 0, MAX_PLAN_NEIGHBORS+1-ml_extents.shape[1]), 'constant', torch.nan)
    
    return ego_extent, dist_extents, single_extents
