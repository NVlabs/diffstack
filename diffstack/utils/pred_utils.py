import os
import random
import torch
import time
import numpy as np

from scipy.optimize import minimize
from collections import defaultdict
from typing import Dict, Union, Tuple, Any, Optional, Iterable



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

