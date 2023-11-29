import os
import random
import torch
import time
import numpy as np
import pickle
import dill
import collections.abc

from datetime import timedelta
from collections import defaultdict, OrderedDict
from scipy.interpolate import interp1d
from typing import Dict, Union, Tuple, Any, Optional, Iterable, List
from torch.utils.data._utils.collate import default_collate

from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

from diffstack.utils.geometry_utils import batch_rotate_2D
# Expose a couple of util functions defined in different submodules.
from trajdata.utils.arr_utils import angle_wrap, batch_select
from diffstack.modules.predictors.trajectron_utils.model.components import GMM2D
container_abcs = collections.abc


# Distributed
import torch.distributed
try:
    import gpu_affinity
    USE_GPU_AFFINITY = True
except:
    USE_GPU_AFFINITY = False


def initialize_torch_distributed(local_rank: int):
    if torch.cuda.is_available():
        backend = 'nccl'
        # Set gpu affinity so that the optimal memory segment is used for multi-gpu training
        # https://gitlab-master.nvidia.com/dl/gwe/gpu_affinity 
        if USE_GPU_AFFINITY:
            gpu_affinity.set_affinity(local_rank, int(os.environ["WORLD_SIZE"]))
    else:
        backend = 'gloo'
    
    torch.distributed.init_process_group(backend=backend,
                            init_method='env://',
                            # default timeout torch.distributed.default_pg_timeout=1800 (sec, =30mins)
                            # increase timeout for datacaching where workload for different gpus can be very different
                            timeout=timedelta(hours=10))  # 10h                             


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepeare_torch_env(rank, hyperparams):
    if torch.cuda.is_available() and hyperparams.run["device"] != 'cpu':
        hyperparams.run["device"] = f'cuda:{rank}'
        torch.cuda.set_device(rank)
    else:
        hyperparams.run["device"] = f'cpu'

    if hyperparams.run["seed"] is not None:
        set_all_seeds(hyperparams.run["seed"])

class CudaTimer(object):
    def __init__(self, enabled=True):
        self.enabled=enabled
        self.timers=defaultdict(list)

    @staticmethod
    def cuda_sync_maybe():
        try:
            torch.cuda.synchronize()
        except:
            pass

    def start(self, name):
        if self.enabled:
            self.cuda_sync_maybe()
            self.timers[name].append(time.time())
    
    def end(self, name):
        if self.enabled:
            assert name in self.timers
            self.cuda_sync_maybe()
            self.timers[name].append(time.time())
    
    def print(self, names=None):
        if self.enabled:
            if names is None:
                names = self.timers.keys()
            s = "Timer " + " ".join([f"{name}={self.timers[name][-1]-self.timers[name][0]:.3f}" for name in names])
            print (s)
            self.timers.clear()


def merge_dicts_with_prefix(prefix_dict: Dict[str, Dict[str, Any]], separator: str = ".") -> Dict[str, Any]:
    output = {}
    for prefix, dict_instance in prefix_dict.items():
        for k, v in dict_instance.items():
            output[f"{prefix}{separator}{k}"] = v
    return output


def batch_derivative_of(states, dt = 1.):
    """
    states: [..., T, state_dim]
    dt: time difference between states in input trajectory
    """
    diff = states[..., 1:, :] - states[..., :-1, :]
    # Add first state derivative
    if isinstance(states, torch.Tensor):
        diff = torch.cat((diff[..., :1, :], diff), dim=-2)
    else:
        diff = np.concatenate((diff[..., :1, :], diff), axis=-2)
    return diff / dt


def subsample_future(x: torch.Tensor, new_horizon: int, current_horizon: int):
    return subsample_traj(x, new_horizon, current_horizon, is_future=True)


def subsample_history(x: torch.Tensor, new_horizon: int, current_horizon: int):
    return subsample_traj(x, new_horizon, current_horizon, is_future=False)


def subsample_traj(x: torch.Tensor, new_horizon: int, current_horizon: int, is_future: bool = True):
    """x: [..., planh+1, :]"""
    assert x.shape[-2] == current_horizon + 1
    if current_horizon != new_horizon:
        if new_horizon == 0:
            subsample_inds = [0]
        else:
            assert current_horizon % new_horizon == 0, f"planning horizon ({new_horizon}) needs to be a multiple of prediction horizon ({current_horizon})"
            subsample_gap = current_horizon // new_horizon
            subsample_inds = list(range(0, current_horizon+1, subsample_gap))  # for gap=2 [0,1,2,3,4,5] --> [0,2,4]
        if not is_future:
            # for gap=2 [0,1,2,3,4,5] --> [0,2,4] --> [-1, -3, -5] --> [-5, -3, -1] (equivalent --> [1, 3, 5])
            subsample_inds = [-ind-1 for ind in subsample_inds]
            subsample_inds.reverse()
            
        assert len(subsample_inds) == new_horizon+1
        if x.ndim == 2:
            return x[subsample_inds]
        else:
            return x.transpose(-2, 0)[subsample_inds].transpose(-2, 0)
    else:
        return x


def normalize_angle(h):
    return (h + np.pi) % (2.0 * np.pi) - np.pi  


def traj_xyhvv_to_pred(traj, dt):
    # Input: [x, y, h, vx, vy] 
    # Output prediction state: ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sintheta', 'costheta']
    x, y, h, vx, vy = np.split(traj, 5, axis=-1)
    ax = batch_derivative_of(vx, dt=dt)
    ay = batch_derivative_of(vy, dt=dt)
    pred_state = np.concatenate((
        x, y, vx, vy, ax, ay, np.sin(h), np.cos(h)
    ), axis=-1)
    return pred_state


def traj_pred_to_xyhvv(traj):
    # Input prediction state: ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sintheta', 'costheta']
    # Output: [x, y, h, vx, vy] 
    x, y, vx, vy, ax, ay, sinh, cosh = np.split(traj, 8, axis=-1)
    h = np.arctan2(sinh, cosh)
    return np.concatenate((x, y, h, vx, vy), axis=-1)


def traj_xy_to_xyh(traj: Union[np.ndarray, torch.Tensor]):
    xy = traj
    dxy = xy[..., 1:, :2] - xy[..., :-1, :2]
    if isinstance(traj, torch.Tensor):
        h = torch.atan2(dxy[..., 1], dxy[..., 0])[..., None]  # TODO invalid for near-zero velocity
        h = torch.concat((h, h[..., -1:, :]), dim=-2)  # extend time
        return torch.concat((xy, h), dim=-1)
    else:
        h = np.arctan2(dxy[..., 1], dxy[..., 0])[..., None]  # TODO invalid for near-zero velocity
        h = np.concatenate((h, h[..., -1:, :]), axis=-2)  # extend time
        return np.concatenate((xy, h), axis=-1)


def traj_xyh_to_xyhv(traj: Union[np.ndarray, torch.Tensor], dt: float):
    xyhvv = traj_xyh_to_xyhvv(traj, dt)
    return traj_xyhvv_to_xyhv(xyhvv)


def traj_xyh_to_xyhvv(traj: Union[np.ndarray, torch.Tensor], dt: float):
    if isinstance(traj, torch.Tensor):
        x, y, h = torch.split(traj, 1, dim=-1)
        vx = batch_derivative_of(x, dt)
        vy = batch_derivative_of(y, dt)
        return torch.concat((x, y, h, vx, vy), dim=-1)
    else:
        x, y, h = np.split(traj, 3, axis=-1)
        vx = batch_derivative_of(x, dt)
        vy = batch_derivative_of(y, dt)
        return np.concatenate((x, y, h, vx, vy), axis=-1)


def traj_xyhv_to_xyhvv(traj):
    x, y, h, v = np.split(traj, 4, axis=-1)
    vx = v * np.cos(h)
    vy = v * np.sin(h)
    return np.concatenate((x, y, h, vx, vy), axis=-1)


def traj_xyhvv_to_xyhv(traj: Union[np.ndarray, torch.Tensor]):
    # Use only the forward velocity component and ignore sideway velocity.
    if isinstance(traj, torch.Tensor):
        x, y, h, vx, vy = torch.split(traj, 1, dim=-1)
        v_xy = batch_rotate_2D(torch.stack((vx, vy),-1), -h)  # forward and sideway velocity
        v = v_xy[...,0]
        return torch.concat((x, y, h, v), dim=-1)
    else:
        x, y, h, vx, vy = np.split(traj, 5, axis=-1)
        v_xy = batch_rotate_2D(np.stack((vx, vy),-1), -h)  # forward and sideway velocity
        v = v_xy[...,0]
        return np.concatenate((x, y, h, v), axis=-1)


def closest_lane_state(global_state: np.ndarray, nusc_map: NuScenesMap):
    nearest_lane = nusc_map.get_closest_lane(x=global_state[0], y=global_state[1])
    lane_rec = nusc_map.get_arcline_path(nearest_lane)
    closest_state, _ = arcline_path_utils.project_pose_to_lane(global_state, lane_rec)
    return closest_state

def lane_frenet_features_simple(ego_state: np.ndarray, lane_states: np.ndarray, plot=False):
    """Taking the equation from the "Line defined by two points" section of
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    as well as this answer: https://stackoverflow.com/a/6853926
    """
    if ego_state.ndim == 2:
        # Recursive callse, iterate over batch/time dimension
        assert ego_state.shape[0] == lane_states.shape[0]
        out = [lane_frenet_features_simple(ego_state[i], lane_states[i], plot=plot) for i in range(ego_state.shape[0])]
        return np.stack(out, axis=0)

    assert lane_states.ndim == 2 and ego_state.ndim == 1

    # Simplified 
    i = np.argmin(np.square(lane_states[:, :2] - ego_state[None, :2]).sum(-1))
    lane_pt = lane_states[i]
    lane_pt_xy = lane_pt[:2]
    lane_pt_h = lane_pt[2]
    v = np.array([np.cos(lane_pt_h), np.sin(lane_pt_h)])
    proj_len = ((ego_state[:2] - lane_pt_xy) * v).sum(-1)  # equivalent of dot(xy-vect_xy, v)
    proj_onto_lane = v * proj_len + lane_pt_xy

    # Debug
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        pts = np.stack([lane_pt_xy - 20*v, lane_pt_xy + 1*v])
        plt.plot(pts[:, 0], pts[:, 1], label="lane line")
        plt.scatter(ego_state[0], ego_state[1], label='ego')
        plt.scatter(proj_onto_lane[0], proj_onto_lane[1], label='proj')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.show()

    return np.array([proj_onto_lane[0], proj_onto_lane[1], lane_pt_h])


def closest_lane(ego_xy: torch.Tensor, lane_points: torch.Tensor):
    ind = torch.argmin(torch.square(lane_points[..., :2] - ego_xy[..., :2].unsqueeze(-2)).sum(-1), axis=-1)
    # Workaround for lack of gather_nd and no broadcasting of gather
    # TODO look for faster implementation of gather_nd
    lane_comps = torch.unbind(lane_points, axis=-1)
    lane_comps = [torch.gather(x, x.ndim-1, ind.unsqueeze(-1)) for x in lane_comps]
    return torch.cat(lane_comps, axis=-1)


def closest_lane_np(ego_xy: np.ndarray, lane_points_list: Iterable[np.ndarray]):
    state_to_lane_dist2 = [
        np.square(lane_points[..., :2] - ego_xy[np.newaxis,..., :2]).sum(-1).min(-1) 
        for lane_points in lane_points_list]
    return np.argmin(state_to_lane_dist2) if len(state_to_lane_dist2)>0 else None


def get_pointgoal_from_onroute_lanes(ego_state_xyhv: np.ndarray, lanes_xyh: List[np.ndarray], dt: float, future_len: int, max_vel: float = 29.0, target_acc: float = 1.5) -> np.ndarray:
    # TODO we should move this logic to the planner

    assert ego_state_xyhv.ndim == 1 and ego_state_xyhv.shape[-1] == 4
    ego_x, ego_y, ego_h, ego_v = ego_state_xyhv

    if len(lanes_xyh) == 0:
        # No lanes. Set goal as current location.
        goal_point_xyh = np.stack((ego_x, ego_y, ego_h), axis=-1)
        ref_traj_xyh = np.repeat(goal_point_xyh[None], future_len+1, axis=0)
        print ("WARNING: no lane for inferring goal")
        return ref_traj_xyh, goal_point_xyh

    # Target final velocity, based on accelearting but remaining under max_vel limit
    target_v = np.minimum(ego_v + target_acc * future_len * dt, max_vel)
    avg_v = 0.5 * (target_v + ego_v)
    target_pathlen = avg_v * future_len * dt

    closest_lane_ind = closest_lane_np(ego_state_xyhv[:3], lanes_xyh)
    closest_lane = lanes_xyh[closest_lane_ind]

    state_to_lane_dist2 = np.square(closest_lane[:, :2] - ego_state_xyhv[None, :2]).sum(-1)
    closest_point_ind = np.argmin(state_to_lane_dist2)

    # Assume polyline is ordered by the lane direction. 
    if closest_point_ind < closest_lane.shape[0]-1:
        future_lane = closest_lane[closest_point_ind:]
        # Get distance along lane. 
        # TODO compute distance from current state, not first lane point
        step_len = np.concatenate([[0.], np.linalg.norm(future_lane[1:] - future_lane[:-1], axis=-1)])
    else:
        # TODO again we should compute distance from current state, negative for first state, positive for second
        future_lane = closest_lane[closest_point_ind-1:]
        step_len = np.concatenate([[0.], np.linalg.norm(future_lane[1:] - future_lane[:-1], axis=-1)])        
    
    dist_along_lane = np.cumsum(step_len)

    # Find connecting lane
    del closest_lane
    while dist_along_lane[-1] < target_pathlen:
        # TODO manually find continuation of lane, i.e. the lane with first point closest to last point of our lane.
        first_lane_points = np.array([lane[0] for lane in lanes_xyh])
        d = np.linalg.norm(first_lane_points[:, :2] - future_lane[-1, None, :2], axis=-1)
        closest_lane_ind = np.argmin(d)

        if d[closest_lane_ind] > 3:  # 3m maximum to treat them as connected lanes  
            # No more lanes, we need to stop at the end of current lane.
            break

        future_lane = np.concatenate((future_lane, lanes_xyh[closest_lane_ind]), axis=0)
        # TODO compute distance from current state, not first lane point
        step_len = np.concatenate([[0.], np.linalg.norm(future_lane[1:] - future_lane[:-1], axis=-1)])
        dist_along_lane = np.cumsum(step_len)

    del closest_lane_ind

    # Find lane point closest to our target distance.
    # TODO this could be done more efficiently, use lane util functions
    target_lane_point_ind = np.argmin(np.abs(dist_along_lane - target_pathlen))
    goal_point_xyh = future_lane[target_lane_point_ind]

    # Recomput target pathlen based on goal point.
    target_pathlen = dist_along_lane[target_lane_point_ind]
    avg_v = target_pathlen / (future_len * dt)
    target_v = 2 * avg_v - ego_v
    a = (target_v - ego_v) / (future_len * dt)

    # For each future time t, what is the intended length of traversed path.
    delta_t = np.arange(future_len+1) * dt
    delta_pathlen_t = (delta_t * a * 0.5 + ego_v) * delta_t 

    # Interpolate lane at these delta lenth points.
    # TODO we need to unwrap angles, do interpolation, and then wrap them again.
    interp_fn = interp1d(dist_along_lane, future_lane, bounds_error=False, assume_sorted=True, axis=0)  # nan for extrapolation
    # interp_fn = interp1d(dist_along_lane, future_lane, fill_value="extrapolate", assume_sorted=True, axis=0)

    ref_traj_xyh = interp_fn(delta_pathlen_t)

    return ref_traj_xyh, goal_point_xyh


def lat_long_distances(x: torch.Tensor, y: torch.Tensor, vect_x: torch.Tensor, vect_y: torch.Tensor, vect_h: torch.Tensor):
    sin_lane_h = torch.sin(vect_h)
    cos_lane_h = torch.cos(vect_h)
    d_long = (x - vect_x) * cos_lane_h + (y - vect_y) * sin_lane_h
    d_lat = -(x - vect_x) * sin_lane_h + (y - vect_y) * cos_lane_h
    return d_lat, d_long


def lane_frenet_features(ego_state: np.ndarray, lane_states: np.ndarray):
    """Taking the equation from the "Line defined by two points" section of
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    as well as this answer: https://stackoverflow.com/a/6853926
    """
    x1s = lane_states[:-1, 0]
    y1s = lane_states[:-1, 1]
    h1s = lane_states[:-1, 2]

    x2s = lane_states[1:, 0]
    y2s = lane_states[1:, 1]
    h2s = lane_states[1:, 2]

    A = ego_state[0] - x1s
    B = ego_state[1] - y1s

    C = x2s - x1s
    D = y2s - y1s

    dot = A * C + B * D
    len_sq = C * C + D * D
    params = np.ma.masked_invalid(np.divide(dot, len_sq, out=np.full_like(dot, np.nan), where=np.abs(len_sq) >= 1e-3))

    if (params < 0).all():
        seg_idx = np.argmax(params)
        lane_x = x1s[seg_idx]
        lane_y = y1s[seg_idx]
        lane_h = h1s[seg_idx]
    elif (params > 1).all():
        seg_idx = np.argmin(params)
        lane_x = x2s[seg_idx]
        lane_y = y2s[seg_idx]
        lane_h = h2s[seg_idx]
    else:
        seg_idx = np.argmin(np.abs(params))
        lane_x = x1s[seg_idx] + params[seg_idx] * C[seg_idx]
        lane_y = y1s[seg_idx] + params[seg_idx] * D[seg_idx]
        lane_h = h1s[seg_idx] + params[seg_idx] * (h2s[seg_idx] - h1s[seg_idx])

    # plot_lane_frenet(lane_states, ego_state, np.array([xx, yy, hh]), seg_idx)
    return lane_x, lane_y, lane_h, seg_idx


def np_rbf(input: np.ndarray, center: Union[np.ndarray, float] = 0.0, scale: Union[np.ndarray, float] = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return np.exp(-0.5*np.square(input - center).sum(-1)/scale)


def pt_rbf(input: torch.Tensor, center: Union[torch.Tensor, float] = 0.0, scale: Union[torch.Tensor, float] = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return torch.exp(-0.5*torch.square(input - center).sum(-1)/scale)


def wrap(angles: Union[torch.Tensor, np.ndarray]):
    return (angles + np.pi) % (2 * np.pi) - np.pi

def angle_wrap(angles: Union[torch.Tensor, np.ndarray]):
    return wrap(angles)


# Custom torch functions that support jit.trace
class _tracable_exp_fn(torch.autograd.Function):
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return x.exp()

    def backward(ctx, dl_dx):
        x, = ctx.saved_tensors
        return _tracable_exp_fn.apply(x) * dl_dx
tracable_exp = _tracable_exp_fn.apply

tracable_sqrt = lambda x: torch.pow(x, 0.5)

tracable_norm = lambda x, dim: tracable_sqrt(x.square().sum(dim=dim)) 

def tracable_rbf(input: torch.Tensor, center: Union[torch.Tensor, float] = 0.0, scale: Union[torch.Tensor, float] = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return tracable_exp(-0.5*tracable_sqrt(input - center).sum(-1)/scale)



def ensure_length_nd(x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
    if extra_info is not None:
        ep_lens = extra_info['ep_lengths']

        x_reshaped = x[..., :ep_lens+1, :]
        u_reshaped = u[..., :ep_lens+1, :]
        # Again, this is one more timesteps than there should be for u, 
        # the last is all zero, and is ignored in the creation of B later.
    
        return x_reshaped, u_reshaped
    else:
        return x, u


def prediction_diversity(y_dists):
    # Variance of predictions at the final time step
    mus = y_dists.mus  # 1, b, T, N, 2
    log_pis = y_dists.log_pis  # 1, b, T, N

    xy = mus.squeeze(0)[:, -1]  # b, N, 2
    probs = torch.exp(log_pis.squeeze(0)[:, -1]) # b, N
    # weighted mean final xy
    xy_mean = (xy * probs.unsqueeze(-1)).sum(dim=1)   # b, 2
    dist_from_mean = torch.linalg.norm(xy_mean.unsqueeze(1) - xy, dim=-1)  # b, N
    # variance of eucledian distances from the (weighted) mean prediction
    # TODO this metric doesnt have a proper statistical interpretation
    # To make it interpretable follow something like 
    # similarity measure in DPP https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9387598 
    # diversity energy https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_LookOut_Diverse_Multi-Future_Prediction_and_Planning_for_Self-Driving_ICCV_2021_paper.pdf 
    var = (probs * torch.square(dist_from_mean)).sum(dim=-1)

    return var  


def convert_state_pred2plan(x_pred: Union[torch.Tensor, np.ndarray]):
    """ Transform 

    Prediction input: 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'sintheta', 'costheta'
    Planning input: x, y, theta, v
    """
    if isinstance(x_pred, torch.Tensor):
        x_plan = torch.stack([
            x_pred[..., 0],  # x
            x_pred[..., 1],  # y
            torch.atan2(x_pred[..., 6], x_pred[..., 7]),  # theta
            torch.linalg.norm(x_pred[..., 2:4], dim=-1),  # v
        ], dim=-1)
    else:
        x_plan = np.stack([
            x_pred[..., 0],  # x
            x_pred[..., 1],  # y
            np.arctan2(x_pred[..., 6], x_pred[..., 7]),  # theta
            np.linalg.norm(x_pred[..., 2:4], axis=-1),  # v
        ], axis=-1)
    return x_plan    


def gmms_from_single_futures(x_xyhv: torch.Tensor, dt: float):
    assert x_xyhv.ndim == 4  # N, b, T, 4
    assert x_xyhv.shape[-1] == 4  # xyhv
    ph = x_xyhv.shape[-2]
    
    mus = x_xyhv.unsqueeze(3)  # (N, b, T, 1, 4)
    log_pis = torch.zeros(mus.shape[:-1], dtype=mus.dtype, device=mus.device)
    log_sigmas = torch.log((torch.arange(1, ph+1, dtype=mus.dtype, device=mus.device) * dt)**2*2)
    log_sigmas = log_sigmas.reshape(1, 1, ph, 1, 1).repeat((x_xyhv.shape[0], x_xyhv.shape[1], 1, 1, 2))
    corrs = 0. * torch.ones(mus.shape[:-1], dtype=mus.dtype, device=mus.device)  # TODO not sure what is reasonable
        
    y_dists = GMM2D(log_pis, mus, log_sigmas, corrs)
    return y_dists


def gmm_concat_as_modes(gmms: Iterable[GMM2D], probs: Iterable[float]):
    gmm_joint = GMM2D(
        log_pis=torch.concat([gmm.log_pis + np.log(p) for gmm, p in zip(gmms, probs)], dim=-1),
        mus=torch.concat([gmm.mus for gmm in gmms], dim=-2),
        log_sigmas=torch.concat([gmm.sigmas for gmm in gmms], dim=-2),
        corrs=torch.concat([gmm.corrs for gmm in gmms], dim=-1),
    )
    return gmm_joint


def gmm_concat_as_agents(gmms: Iterable[GMM2D]):
    gmm_joint = GMM2D(
        log_pis=torch.concat([gmm.log_pis for gmm in gmms], dim=0),
        mus=torch.concat([gmm.mus for gmm in gmms], dim=0),
        log_sigmas=torch.concat([gmm.log_sigmas for gmm in gmms], dim=0),
        corrs=torch.concat([gmm.corrs for gmm in gmms], dim=0),
    )
    return gmm_joint
  

def gmm_extend(gmm: GMM2D, num_modes: int):
    gmm = GMM2D(
        log_pis=torch.nn.functional.pad(gmm.log_pis, (0, num_modes), mode="constant", value=-np.inf),
        mus=torch.nn.functional.pad(gmm.mus, (0, 0, 0, num_modes), mode="constant", value=np.nan),
        log_sigmas=torch.nn.functional.pad(gmm.log_sigmas, (0, 0, 0, num_modes), mode="constant", value=np.nan),
        corrs=torch.nn.functional.pad(gmm.corrs, (0, num_modes), mode="constant", value=0.),
    )
    return gmm
    

def move_list_element_to_front(a: list, i: int) -> list:
    a = [a[i]] + [a[j] for j in range(len(a)) if j != i]
    return a

def soft_min(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((y-x)/2)
    exp2 = expfun((x-y)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)

def soft_max(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((x-y)/2)
    exp2 = expfun((y-x)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)
def soft_sat(x,x_min=None,x_max=None,gamma=5):
    if x_min is None and x_max is None:
        return x
    elif x_min is None and x_max is not None:
        return soft_min(x,x_max,gamma)
    elif x_min is not None and x_max is None:
        return soft_max(x,x_min,gamma)
    else:
        if isinstance(x_min,torch.Tensor) or isinstance(x_min,np.ndarray):
            assert (x_max>x_min).all()
        else:
            assert x_max>x_min
        xc = x - (x_min+x_max)/2
        if isinstance(x,torch.Tensor):
            return xc/(torch.pow(1+torch.pow(torch.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        elif isinstance(x,np.ndarray):
            return xc/(np.power(1+np.power(np.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        else:
             raise Exception("data type not supported")


def recursive_dict_list_tuple_apply(x, type_func_dict, ignore_if_unspecified=False):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.
        ignore_if_unspecified (bool): ignore an item if its type is unspecified by the type_func_dict

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert list not in type_func_dict
    assert tuple not in type_func_dict
    assert dict not in type_func_dict
    assert torch.nn.ParameterDict not in type_func_dict
    assert torch.nn.ParameterList not in type_func_dict

    if isinstance(x, (dict, OrderedDict, torch.nn.ParameterDict)):
        new_x = (
            OrderedDict()
            if isinstance(x, OrderedDict)
            else dict()
        )
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict, ignore_if_unspecified)
        return new_x
    elif isinstance(x, (list, tuple, torch.nn.ParameterList)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict, ignore_if_unspecified) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            if ignore_if_unspecified:
                return x
            else:
                raise NotImplementedError("Cannot handle data type %s" % str(type(x)))

def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    assert begin_axis < end_axis
    assert begin_axis >= 0
    assert end_axis <= len(x.shape)
    assert isinstance(target_dims, (tuple, list))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i >= end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def yaw_from_pos(pos: torch.Tensor, dt, yaw_correction_speed=0.):
    """
    Compute yaws from position sequences. Optionally suppress yaws computed from low-velocity steps

    Args:
        pos (torch.Tensor): sequence of positions [..., T, 2]
        dt (float): delta timestep to compute speed
        yaw_correction_speed (float): zero out yaw change when the speed is below this threshold (noisy heading)

    Returns:
        accum_yaw (torch.Tensor): sequence of yaws [..., T-1, 1]
    """

    pos_diff = pos[..., 1:, :] - pos[..., :-1, :]
    yaw = torch.atan2(pos_diff[..., 1], pos_diff[..., 0])
    delta_yaw = torch.cat((yaw[..., [0]], yaw[..., 1:] - yaw[..., :-1]), dim=-1)
    speed = torch.norm(pos_diff, dim=-1) / dt
    delta_yaw[speed < yaw_correction_speed] = 0.
    accum_yaw = torch.cumsum(delta_yaw, dim=-1)
    return accum_yaw[..., None]
    

def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors)
    
    Parameters
    ----------
    data: any picklable object
    
    Returns
    --------
    list[data]
        List of data gathered from each rank
    """
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return [data]

    # Serialize to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Receive Tensor from all ranks
    # We pad the tensor because torch all_gather does not support gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size, )).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


# Wrapper around dict to identify batchable dict data.
class batchable_dict(dict):
    pass

class batchable_list(list):
    pass

class batchable_nonuniform_tensor(torch.Tensor):
    pass


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, str) or elem.__class__.__name__ == "batchable_list" or elem.__class__.__name__ == "batchable_nonuniform_tensor":
        # TODO isinstance(elem, batchable_nonuniform_tensor) is never true, perhaps some import path comparison issue
        return dill.dumps(batch) if torch.utils.data.get_worker_info() else batch
    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4: # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
                                                                     scene_pts=torch.Tensor(scene_pts),
                                                                     patch_size=patch_size[0],
                                                                     rotation=heading_angle)
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif elem.__class__.__name__ == "batchable_dict":
        # We dill the dictionary for the same reason as the neighbors structure (see below).
        # Unlike for neighbors where we keep a list, here we collate elements recursively
        data_dict = {key: collate([d[key] for d in batch]) for key in elem}
        return dill.dumps(data_dict) if torch.utils.data.get_worker_info() else data_dict
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    try:
        return default_collate(batch)
    except RuntimeError:
        # This happens when tensors are not of the same shape.
        return dill.dumps(batch) if torch.utils.data.get_worker_info() else batch


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))

def removeprefix(line, prefix):
    if line.startswith(prefix):
        line_new = line[len(prefix):]
    return line_new