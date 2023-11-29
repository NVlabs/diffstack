"""
Adapted from https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/evaluation/metrics.py
"""


from typing import Callable, Dict, List, Union
import torch
import numpy as np

from diffstack.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
    get_box_world_coords,
)
from diffstack.utils.loss_utils import log_normal
from diffstack.modules.predictors.trajectron_utils.model.components.gmm2d import GMM2D

from trajdata import SceneBatch


metric_signature = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
]


def _assert_shapes(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> None:
    """
    Check the shapes of args required by metrics
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
    """
    assert (
        len(pred.shape) == 4
    ), f"expected 3D (BxMxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Batch x Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (
        batch_size,
        num_modes,
    ), f"expected 2D (Batch x Modes) array for confidences, got {confidences.shape}"

    assert np.allclose(np.sum(confidences, axis=1), 1), "confidences should sum to 1"
    assert avails.shape == (
        batch_size,
        future_len,
    ), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(ground_truth).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"


def batch_neg_multi_log_likelihood(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    For more details about used loss function and reformulation, please see
    https://github.com/lyft/l5kit/blob/master/competition.md.
    Args:
        ground_truth (np.ndarray): array of shape (batchsize)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batchsize)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape ((batchsize)xmodes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batchsize)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: negative log-likelihood for this batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = np.max(
        error, axis=-1, keepdims=True
    )  # error are negative at this point, so max() gives the minimum one
    error = (
        -np.log(np.sum(np.exp(error - max_value), axis=-1)) - max_value
    )  # reduce modes
    return error


def batch_rmse(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> np.ndarray:
    """
    Return the root mean squared error, computed using the stable nll
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: negative log-likelihood for this batch, an array of float numbers
    """
    nll = batch_neg_multi_log_likelihood(ground_truth, pred, confidences, avails)
    _, _, future_len, _ = pred.shape

    return np.sqrt(2 * nll / future_len)


def batch_prob_true_mode(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> np.ndarray:
    """
    Return the probability of the true mode
    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: a (modes) numpy array
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = np.max(
        error, axis=-1, keepdims=True
    )  # error are negative at this point, so max() gives the minimum one

    error = np.exp(error - max_value) / np.sum(np.exp(error - max_value))
    return error


def batch_time_displace(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
) -> np.ndarray:
    """
    Return the displacement at timesteps T
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: a (batch)x(timesteps) numpy array
    """
    true_mode_error = batch_prob_true_mode(ground_truth, pred, confidences, avails)
    true_mode_error = true_mode_error[:, :, np.newaxis]  # add timesteps axis

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability
    return np.sum(true_mode_error * np.sqrt(error), axis=1)  # reduce modes


def batch_average_displacement_error(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = np.mean(error, axis=-1)  # average over timesteps
    if mode == "oracle":
        error = np.min(error, axis=1)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error*confidences, axis=1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def batch_final_displacement_error(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: final displacement error (FDE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    inds = np.arange(0, pred.shape[2])
    inds = (avails > 0) * inds  # [B, (A), T] arange indices with unavailable indices set to 0
    last_inds = inds.max(axis=-1)
    last_inds = np.tile(last_inds[:, np.newaxis, np.newaxis],(1,pred.shape[1],1))
    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords
    
    
    error = np.sum(
        ((ground_truth - pred) * avails) ** 2, axis=-1
    )  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)

    # error = error[:, :, -1]  # use last timestep
    error = np.take_along_axis(error,last_inds,axis=2).squeeze(-1)
    if mode == "oracle":
        error = np.min(error, axis=-1)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=-1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def batch_average_diversity(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Compute the distance among trajectory samples averaged across time steps
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode: option are "mean" (average distance) and "max" (distance between
            the two most distinctive samples).
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    # compute pairwise distances
    error = np.linalg.norm(
        pred[:, np.newaxis, :] - pred[:, :, np.newaxis], axis=-1
    )  # [B, M, M, T]
    error = np.mean(error, axis=-1)  # average over timesteps
    error = error.reshape([error.shape[0], -1])  # [B, M * M]
    if mode == "max":
        error = np.max(error, axis=-1)
    elif mode == "mean":
        error = np.mean(error, axis=-1)
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def batch_final_diversity(
    ground_truth: np.ndarray,
    pred: np.ndarray,
    confidences: np.ndarray,
    avails: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Compute the distance among trajectory samples at the last timestep
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode: option are "mean" (average distance) and "max" (distance between
            the two most distinctive samples).
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)
    # compute pairwise distances at the last time step
    pred = pred[..., -1]
    error = np.linalg.norm(
        pred[:, np.newaxis, :] - pred[:, :, np.newaxis], axis=-1
    )  # [B, M, M]
    error = error.reshape([error.shape[0], -1])  # [B, M * M]
    if mode == "max":
        error = np.max(error, axis=-1)
    elif mode == "mean":
        error = np.mean(error, axis=-1)
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def single_mode_metrics(
    metrics_func, ground_truth: np.ndarray, pred: np.ndarray, avails: np.ndarray
):
    """
    Run a metrics with single mode by inserting a mode dimension

    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(time)x(2D coords)
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): Optional, set to None when not applicable
            calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: metrics values
    """
    pred = pred[:, None]
    conf = np.ones((pred.shape[0], 1))
    kwargs = dict(ground_truth=ground_truth, pred=pred, confidences=conf, avails=avails)
    return metrics_func(**kwargs)


def batch_pairwise_collision_rate(agent_edges, collision_funcs=None):
    """
    Count number of collisions among edge pairs in a batch
    Args:
        agent_edges (dict): A dict that maps collision types to box locations
        collision_funcs (dict): A dict of collision functions (implemented in diffstack.utils.geometric_utils)

    Returns:
        collision loss (torch.Tensor)
    """
    if collision_funcs is None:
        collision_funcs = {
            "VV": VEH_VEH_collision,
            "VP": VEH_PED_collision,
            "PV": PED_VEH_collision,
            "PP": PED_PED_collision,
        }

    coll_rates = {}
    for et, fun in collision_funcs.items():
        edges = agent_edges[et]
        dis = fun(
            edges[..., 0:3],
            edges[..., 3:6],
            edges[..., 6:8],
            edges[..., 8:],
        )
        dis = dis.min(-1)[0]  # reduction over time
        if isinstance(dis, np.ndarray):
            coll_rates[et] = np.sum(dis <= 0) / float(dis.shape[0])
        else:
            coll_rates[et] = torch.sum(dis <= 0) / float(dis.shape[0])
    return coll_rates


def batch_pairwise_collision_rate_masked(agent_edges, type_mask,collision_funcs=None):
    """
    Count number of collisions among edge pairs in a batch
    Args:
        agent_edges (dict): A dict that maps collision types to box locations
        collision_funcs (dict): A dict of collision functions (implemented in diffstack.utils.geometric_utils)

    Returns:
        collision loss (torch.Tensor)
    """
    if collision_funcs is None:
        collision_funcs = {
            "VV": VEH_VEH_collision,
            "VP": VEH_PED_collision,
            "PV": PED_VEH_collision,
            "PP": PED_PED_collision,
        }
    coll_rates = {}
    for et, fun in collision_funcs.items():
        if et in type_mask and type_mask[et].sum()>0:
            dis = fun(
                agent_edges[..., 0:3],
                agent_edges[..., 3:6],
                agent_edges[..., 6:8],
                agent_edges[..., 8:],
            )
            dis = dis.min(-1)[0]  # reduction over time
            if isinstance(dis, np.ndarray):
                coll_rates[et] = np.sum((dis <= 0)*type_mask[et]) / type_mask[et].sum()
            else:
                coll_rates[et] = torch.sum((dis <= 0)*type_mask[et]) / type_mask[et].sum()
    return coll_rates


def batch_detect_off_road(positions, drivable_region_map):
    """
    Compute whether the given positions are out of drivable region
    Args:
        positions (torch.Tensor): a position (x, y) in rasterized frame [B, ..., 2]
        drivable_region_map (torch.Tensor): binary drivable region maps [B, H, W]

    Returns:
        off_road (torch.Tensor): whether each given position is off-road [B, ...]
    """
    assert positions.shape[0] == drivable_region_map.shape[0]
    assert drivable_region_map.ndim == 3
    b, h, w = drivable_region_map.shape
    positions_flat = positions[..., 1].long() * w + positions[..., 0].long()
    if positions_flat.ndim == 1:
        positions_flat = positions_flat[:, None]
    drivable = torch.gather(
        drivable_region_map.flatten(start_dim=1),  # [B, H * W]
        dim=1,
        index=positions_flat.long().flatten(start_dim=1),  # [B, (all trailing dim flattened)]
    ).reshape(*positions.shape[:-1])
    return 1 - drivable.float()


def batch_detect_off_road_boxes(positions, yaws, extents, drivable_region_map):
    """
    Compute whether boxes specified by (@positions, @yaws, and @extents) are out of drivable region.
    A box is considered off-road if at least one of its corners are out of drivable region
    Args:
        positions (torch.Tensor): agent centroid (x, y) in rasterized frame [B, ..., 2]
        yaws (torch.Tensor): agent yaws in rasterized frame [B, ..., 1]
        extents (torch.Tensor): agent extents [B, ..., 2]
        drivable_region_map (torch.Tensor): binary drivable region maps [B, H, W]

    Returns:
        box_off_road (torch.Tensor): whether each given box is off-road [B, ...]
    """
    boxes = get_box_world_coords(positions, yaws, extents)  # [B, ..., 4, 2]
    off_road = batch_detect_off_road(boxes, drivable_region_map)  # [B, ..., 4]
    box_off_road = off_road.sum(dim=-1) > 0.5
    return box_off_road.float()


def GMM_loglikelihood(x, m, v, pi, avails=None, mode="mean"):
    """
    Log probability of tensor x under a uniform mixture of Gaussians.
    Adapted from CS 236 at Stanford.
    Args:
        x (torch.Tensor): tensor with shape (B, D)
        m (torch.Tensor): means tensor with shape (B, M, D) or (1, M, D), where
            M is number of mixture components
        v (torch.Tensor): variances tensor with shape (B, M, D) or (1, M, D) where
            M is number of mixture components
        logpi (torch.Tensor): log probability of the modes (B,M)
        detach (bool): option whether to detach all modes but the best one
        mode (string): mode of loss, sum or max

    Returns:
        -log_prob (torch.Tensor): log probabilities of shape (B,)
    """

    if v is None:
        v = torch.ones_like(m)

    # (B , D) -> (B , 1, D)
    x = x.unsqueeze(1)
    # (B, 1, D) -> (B, M, D) -> (B, M)
    if avails is not None:
        avails = avails.unsqueeze(1)
    log_prob = log_normal(x, m, v, avails=avails)
    if mode == "sum":
        loglikelihood = (pi*log_prob).sum(1)
    elif mode == "mean":
        loglikelihood = (pi*log_prob).mean(1)
    elif mode == "max":
        loglikelihood = (pi*log_prob).max(1)
    return loglikelihood


class DistanceBuffer(object):
    """ class that stores the distance given x,y location
    """
    def __init__(self):
        self._buffer = dict()

    def __getitem__(self,key):
        if key in self._buffer:
            return self._buffer[key]
        else:
            return self.update(key)

    def update(self,key):
        dis = np.linalg.norm(key)
        self._buffer[key] = dis
        self._buffer[-key] = dis
        return dis


class RandomPerturbation(object):
    """
    Add Gaussian noise to the trajectory
    """
    def __init__(self, std: np.ndarray):
        assert std.shape == (3,) and np.all(std >= 0)
        self.std = std

    def perturb(self, obs):
        """Given the observation object, add Gaussian noise to positions and yaws

        Args:
            obs(Dict[torch.tensor]): observation dict

        Returns:
            obs(Dict[torch.tensor]): perturbed observation
        """
        obs = dict(obs)
        target_traj = np.concatenate((obs["fut_pos"], obs["fut_yaw"]), axis=-1)
        std = np.ones_like(target_traj) * self.std[None, :]
        noise = np.random.normal(np.zeros_like(target_traj), std)
        target_traj += noise
        obs["fut_pos"] = target_traj[..., :2]
        obs["fut_yaw"] = target_traj[..., :1]
        return obs


class OrnsteinUhlenbeckPerturbation(object):
    """
    Add Ornstein-Uhlenbeck noise to the trajectory
    """
    def __init__(self,theta,sigma):
        """
        Args:
            theta (np.ndarray): converging factor of the OU process
            sigma (np.ndarray): magnitude of the Gaussian noise added at each step
        """
        assert theta.shape == (3,) and sigma.shape == (3,)
        self.theta = theta
        self.sigma = sigma
        self.buffers = dict()

    def perturb(self,obs):
        """Given the observation object, add Gaussian noise to positions and yaws

        Args:
            obs(Dict[torch.tensor]): observation dict

        Returns:
            obs(Dict[torch.tensor]): perturbed observation
        """
        if isinstance(obs["fut_pos"],np.ndarray):
            target_traj = np.concatenate((obs["fut_pos"], obs["fut_yaw"]), axis=-1)
            bs = target_traj.shape[0]
            T = target_traj.shape[-2]
            if bs in self.buffers:
                buffer = self.buffers[bs]
            else:
                buffer = [np.zeros([bs,3])]
                self.buffers[bs]=buffer
            while len(buffer)<T:
                buffer.append(buffer[-1]-self.theta*buffer[-1]+np.random.randn(bs,3)*self.sigma)
            noise = np.stack(buffer,axis=1)
            target_traj += noise[...,:T,:]
            obs["fut_pos"] = target_traj[..., :2].astype(np.float32)
            obs["fut_yaw"] = target_traj[..., 2:].astype(np.float32)
            buffer.pop(0)
        elif isinstance(obs["fut_pos"],torch.Tensor):

            target_traj = torch.cat((obs["fut_pos"], obs["fut_yaw"]), dim=-1)
            bs = target_traj.shape[0]
            T = target_traj.shape[-2]
            if bs in self.buffers:
                buffer = self.buffers[bs]
            else:
                buffer = [torch.zeros([bs,3])]
                self.buffers[bs]=buffer
            while len(buffer)<T:
                buffer.append(buffer[-1]-self.theta*buffer[-1]+torch.randn(bs,3)*self.sigma)
            noise = torch.stack(buffer,dim=1)
            target_traj += noise[...,:T,:].to(target_traj.device)
            obs["fut_pos"] = target_traj[..., :2].type(torch.float32)
            obs["fut_yaw"] = target_traj[..., 2:].type(torch.float32)
            buffer.pop(0)
        return obs

class DynOrnsteinUhlenbeckPerturbation(object):
    """
    Add Ornstein-Uhlenbeck noise to the action of a Unicycle model trajectory
    """
    def __init__(self,theta,sigma,dyn):
        """
        Args:
            theta (np.ndarray): converging factor of the OU process
            sigma (np.ndarray): magnitude of the Gaussian noise added at each step
        """
        assert theta.shape == (2,) and sigma.shape == (2,)
        self.theta = theta
        self.sigma = sigma
        self.buffers = dict()
        self.dyn = dyn

    def perturb(self,obs):
        """Given the observation object, add Gaussian noise to action sequence

        Args:
            obs(Dict[torch.tensor]): observation dict

        Returns:
            obs(Dict[torch.tensor]): perturbed observation
        """
        pos = obs["fut_pos"]
        yaw = obs["fut_yaw"]
        mask = obs["fut_mask"]
        dt = obs["step_time"]
        
        if isinstance(pos,np.ndarray):
            state = self.dyn.get_state(pos,yaw,mask)
            curr_state = state[...,0,:]
            u0 = self.dyn.inverse_dyn(state[...,:-1,:],state[...,1:,:])

            bs = u0.shape[0]
            T = u0.shape[-2]
            if bs in self.buffers:
                buffer = self.buffers[bs]
            else:
                buffer = [np.zeros([bs,self.dyn.udim])]
                self.buffers[bs]=buffer
            while len(buffer)<T:
                buffer.append(buffer[-1]-self.theta*buffer[-1]+np.random.randn(bs,self.dyn.udim)*self.sigma)
            noise = np.stack(buffer,axis=1)
            u_pert = u0 + noise[...,:T,:]
            traj_pert = self.dyn.forward_dynamics(curr_state,u_pert)
            traj_pert = np.concatenate((curr_state.unsqueeze(-2),traj_pert),-2)
            obs["fut_pos"] = self.dyn.state2pos(traj_pert)
            obs["fut_yaw"] = self.dyn.state2yaw(traj_pert)
            buffer.pop(0)
        elif isinstance(obs["fut_pos"],torch.Tensor):
            
            state = self.dyn.get_state(pos,yaw,mask)
            curr_state = state[...,0,:]
            u0 = self.dyn.inverse_dyn(state[...,:-1,:],state[...,1:,:])

            bs = u0.shape[0]
            T = u0.shape[-2]
            if bs in self.buffers:
                buffer = self.buffers[bs]
            else:
                buffer = [torch.zeros([bs,self.dyn.udim])]
                self.buffers[bs]=buffer
            while len(buffer)<T:
                buffer.append(buffer[-1]-self.theta*buffer[-1]+torch.randn(bs,self.dyn.udim)*self.sigma)
            noise = torch.stack(buffer,dim=1)
            u_pert = u0 + noise[...,:T,:].to(u0.device)

            traj_pert = self.dyn.forward_dynamics(curr_state,u_pert)
            traj_pert = torch.cat((curr_state.unsqueeze(-2),traj_pert),-2)
            obs["fut_pos"] = self.dyn.state2pos(traj_pert)
            obs["fut_yaw"] = self.dyn.state2yaw(traj_pert)
            buffer.pop(0)
        return obs


def compute_min_afde_k(
    trajs: torch.Tensor, 
    log_probs: torch.Tensor, 
    gt_trajs: torch.Tensor, 
    k: Union[int, List[int]]
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict of ade and fde metrics out of top-k most likely predictions.
    Args:
        trajs: tensor, (b, n, s, k, t, state)
        log_probs: tensor, (b, n, {s,1}, k, {t,1})
        gt_trajs: tensor, (b, n, t, state)
        k: int or list of ints for top k predictions
    Returns:
        dict with keys ['ade_top_k', 'fde_top_k'] and values of tensor, (b, n, state)
    """
    assert trajs.ndim == 6 and log_probs.ndim == 5 and gt_trajs.ndim == 4
    sorted_idxs = torch.argsort(log_probs[..., [0], :], dim=-1, descending=True)
    sorted_means = torch.gather(trajs, dim=-3, index=sorted_idxs.unsqueeze(-1).expand(
        (-1, trajs.shape[1], -1, -1, trajs.shape[4], trajs.shape[5])))

    errors = torch.linalg.norm(sorted_means - gt_trajs[:, :, None, None], dim=-1)  # b, n, s, k, t

    if isinstance(k, int):
        k = [k]
    res = {}
    for ki in k:
        top_k_errors = errors[..., :ki, :]

        ade_k = torch.mean(top_k_errors, dim=-1)  # b, n, s, k
        min_ade_k = torch.min(ade_k, dim=-1).values  # b, n, s

        fde_k = top_k_errors[..., -1]    # b, n, s, k
        min_fde_k = torch.min(fde_k, dim=-1).values  # b, n, s

        res[f"ade_top_{ki}"] = min_ade_k
        res[f"fde_top_{ki}"] = min_fde_k

    return res


## The next set of functions are replicates from the trajectron repo.

def compute_ade_pt(predicted_trajs, gt_traj):
    error = torch.linalg.norm(predicted_trajs - gt_traj, dim=-1)
    ade = torch.mean(error, axis=-1)
    return ade.flatten()


def compute_fde_pt(predicted_trajs, gt_traj):
    final_error = torch.linalg.norm(predicted_trajs[..., -1, :] - gt_traj[..., -1, :], dim=-1)
    return final_error.flatten()


def compute_nll_pt(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(predicted_dist.log_prob(gt_traj), min=-20.)
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz.sum(dim=0), -log_p_y_xz_final.sum(dim=0)


def compute_min_afde_k_pt(predicted_dist, gt_traj, k):
    means = predicted_dist.mus[..., :gt_traj.shape[-1]]
    probs = predicted_dist.pis_cat_dist.probs
    sorted_idxs = torch.argsort(probs[..., [0], :], dim=-1, descending=True)
    sorted_means = torch.gather(means, dim=-2, index=sorted_idxs.unsqueeze(-1).expand((-1, -1, means.shape[2], -1, means.shape[4])))

    errors = torch.linalg.norm(sorted_means - gt_traj.unsqueeze(-2), dim=-1)
    top_k_errors = errors[..., :k]

    ade_k = torch.mean(top_k_errors, dim=-2)
    min_ade_k = torch.min(ade_k, dim=-1).values

    fde_k = top_k_errors[..., -1, :]
    min_fde_k = torch.min(fde_k, dim=-1).values

    return min_ade_k.flatten(), min_fde_k.flatten()


def compute_nll(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(predicted_dist.log_prob(torch.as_tensor(gt_traj)), min=-20.)
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz[0].numpy(), -log_p_y_xz_final[0].numpy()


def compute_prediction_metrics(futures,
                                prediction_output_dict=None,
                                y_dists=None):
    eval_ret = dict()
    if prediction_output_dict is not None:
        ade_errors = compute_ade_pt(prediction_output_dict, futures)
        fde_errors = compute_fde_pt(prediction_output_dict, futures)
        
        eval_ret['ml_ade'] = ade_errors
        eval_ret['ml_fde'] = fde_errors
    
    if y_dists is not None:
        nll_means, nll_finals = compute_nll_pt(y_dists, futures)
        min_ade_5, min_fde_5 = compute_min_afde_k_pt(y_dists, futures, k=5)
        min_ade_10, min_fde_10 = compute_min_afde_k_pt(y_dists, futures, k=10)
        
        eval_ret['min_ade_5'] = min_ade_5
        eval_ret['min_fde_5'] = min_fde_5
        eval_ret['min_ade_10'] = min_ade_10
        eval_ret['min_fde_10'] = min_fde_10
        eval_ret['nll_mean'] = nll_means
        eval_ret['nll_final'] = nll_finals

    return eval_ret


def scene_centric_prediction_metrics(scene_batch: SceneBatch, pred_outputs: Dict) -> Dict[str, torch.Tensor]:
    
    futures = scene_batch.agent_fut[:, :, :, :2].transpose(0, 1)  # n, b, t, 2
    if "pred_dist_with_ego" not in pred_outputs or pred_outputs["pred_dist_with_ego"] is None:
        return {}
    pred_dist = pred_outputs["pred_dist_with_ego"]

    # Metrics for all agents
    metrics_dict = compute_prediction_metrics(futures, prediction_output_dict=None, y_dists=pred_dist)
    # Average over agents
    metrics_dict = {k: v.mean(dim=0) for k, v in metrics_dict.items()}

    # Ego metrics
    ego_pred_dist = GMM2D(pred_dist.log_pis[:1], pred_dist.mus[:1], pred_dist.log_sigmas[:1], pred_dist.corrs[:1])
    ego_metrics_dict = compute_prediction_metrics(futures[0], prediction_output_dict=None, y_dists=ego_pred_dist)
    metrics_dict.update({"ego_"+k: v for k, v in ego_metrics_dict.items()})

    # Move to cpu
    metrics_dict = {k: v.detach().cpu() for k, v in metrics_dict.items()}

    return metrics_dict
