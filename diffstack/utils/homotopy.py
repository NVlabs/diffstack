from Pplan.Sampling.spline_planner import SplinePlanner
from Pplan.Sampling.trajectory_tree import TrajTree

import torch
import numpy as np
import diffstack.utils.geometry_utils as GeoUtils
from diffstack.utils.geometry_utils import ratan2
from diffstack.utils.tree import Tree
from typing import List
from enum import IntEnum


HOMOTOPY_THRESHOLD = np.pi / 6


class HomotopyType(IntEnum):
    """
    Homotopy class between two paths
    STATIC: relatively small wraping angle
    CW: clockwise
    CCW: counter-clockwise
    """

    STATIC = 0
    CW = 1
    CCW = 2

    @staticmethod
    def enforce_symmetry(x, mode="U"):
        assert x.shape[-2] == x.shape[-1]
        xT = x.transpose(-1, -2).clone()

        diag_mask = torch.eye(x.shape[-1], device=x.device).bool().expand(*x.shape)
        x.masked_fill_(diag_mask, HomotopyType.STATIC)

        if mode == "U":
            # force symmetry based on upper triangular matrix
            triangle = torch.tril(torch.ones_like(x), diagonal=-1)
            x = x * (1 - triangle) + xT * triangle
        elif mode == "L":
            # force symmetry based on lower triangular matrix
            triangle = torch.triu(torch.ones_like(x), diagonal=1)
            x = x * (1 - triangle) + xT * triangle
        return x


def mag_integral(path0, path1, mask=None):
    if isinstance(path0, torch.Tensor):
        delta_path = path0 - path1
        close_flag = torch.norm(delta_path, dim=-1) < 1e-3
        angle = ratan2(delta_path[..., 1], delta_path[..., 0]).masked_fill(
            close_flag, 0
        )
        delta_angle = GeoUtils.round_2pi(angle[..., 1:] - angle[..., :-1])
        if mask is not None:
            if mask.ndim == delta_angle.ndim - 1:
                delta_angle = delta_angle * mask[..., None]
            elif mask.ndim == delta_angle.ndim:
                diff_mask = mask[..., 1:] * mask[..., :-1]
                delta_angle = delta_angle * diff_mask
        angle_diff = torch.sum(delta_angle, dim=-1)

    elif isinstance(path0, torch.ndarray):
        delta_path = path0 - path1
        close_flag = (np.norm(delta_path, dim=-1) < 1e-3).float()
        angle = np.arctan2(delta_path[..., 1], delta_path[..., 0]) * (1 - close_flag)
        delta_angle = GeoUtils.round_2pi(angle[..., 1:] - angle[..., :-1])
        if mask is not None:
            if mask.ndim == delta_angle.ndim - 1:
                delta_angle = delta_angle * mask[..., None]
            elif mask.ndim == delta_angle.ndim:
                diff_mask = mask[..., 1:] * mask[..., :-1]
                delta_angle = delta_angle * diff_mask
        angle_diff = np.sum(delta_angle, axis=-1)
    return angle_diff


def identify_homotopy(
    ego_path: torch.Tensor, obj_paths: torch.Tensor, threshold=HOMOTOPY_THRESHOLD
):
    """Identifying homotopy classes for the ego

    Args:
        ego_path (torch.Tensor): B x T x 2
        obj_paths (torch.Tensor): B x M x N x T x 2
    """
    b, M, N = obj_paths.shape[:3]
    angle_diff = mag_integral(ego_path[:, None, None], obj_paths)
    homotopy = torch.zeros([b, M, N], device=ego_path.device)
    homotopy[angle_diff >= threshold] = HomotopyType.CCW
    homotopy[angle_diff <= -threshold] = HomotopyType.CW
    homotopy[(angle_diff > -threshold) & (angle_diff < threshold)] = HomotopyType.STATIC

    return angle_diff, homotopy


def identify_pairwise_homotopy(
    path: torch.Tensor, threshold=HOMOTOPY_THRESHOLD, mask=None
):
    """
    Args:
        path (torch.Tensor): B x N x T x 2
    """
    b, N, T = path.shape[:3]
    if mask is not None:
        mask = mask[:, None] * mask[:, :, None]
    angle_diff = mag_integral(path[:, :, None], path[:, None], mask=mask)
    homotopy = torch.zeros([b, N, N], device=path.device)
    homotopy[angle_diff >= threshold] = HomotopyType.CCW
    homotopy[angle_diff <= -threshold] = HomotopyType.CW
    homotopy[(angle_diff > -threshold) & (angle_diff < threshold)] = HomotopyType.STATIC

    return angle_diff, homotopy
