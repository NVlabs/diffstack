import os
import random
import torch
import torch.distributed as dist
import time
import numpy as np
import pickle
import dill
import collections.abc
import datetime

from collections import defaultdict
from typing import Dict, Union, Tuple, Any, Optional, Iterable
from torch.utils.data._utils.collate import default_collate

from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

# Expose a couple of util functions defined in different submodules.
from trajdata.utils.arr_utils import angle_wrap
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
                            timeout=datetime.timedelta(hours=10))  # 10h                             


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepeare_torch_env(rank, hyperparams):
    if torch.cuda.is_available() and hyperparams["device"] != 'cpu':
        hyperparams["device"] = f'cuda:{rank}'
        torch.cuda.set_device(rank)
    else:
        hyperparams["device"] = f'cpu'

    if hyperparams["seed"] is not None:
        set_all_seeds(hyperparams["seed"])

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


def subsample_traj(x, predh, planh):
    assert x.shape[0] == planh + 1
    if planh != predh:
        assert planh % predh == 0, f"planning horizon ({predh}) needs to be a multiple of prediction horizon ({predh})"
        subsample_gap = planh // predh
        subsample_inds = list(range(0, planh+1, subsample_gap))  # for gap=2 [0,1,2,3,4] --> [0,2,4]
        assert len(subsample_inds) == predh+1
        return x[subsample_inds]
    else:
        return x

def normalize_angle(h):
    return (h + np.pi) % (2.0 * np.pi) - np.pi  


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
    return np.argmin(state_to_lane_dist2)


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


def move_list_element_to_front(a: list, i: int) -> list:
    a = [a[i]] + [a[j] for j in range(len(a)) if j != i]
    return a
    

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
    world_size = dist.get_world_size()

    if world_size == 1:
        return [data]

    # Serialize to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
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
    dist.all_gather(tensor_list, tensor)

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

