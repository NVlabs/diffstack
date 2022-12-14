import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple

from trajdata.maps.map_api import VectorMap
from trajdata.maps.vec_map_elements import RoadLane, Polyline
from diffstack.utils.utils import angle_wrap


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_interpolating_spline(state_0, state_f, tf):
    dx0, dy0 = state_0[..., 2] * \
        torch.cos(state_0[..., 3]), state_0[..., 2] * \
        torch.sin(state_0[..., 3])
    dxf, dyf = state_f[..., 2] * \
        torch.cos(state_f[..., 3]), state_f[..., 2] * \
        torch.sin(state_f[..., 3])
    tf = tf * torch.ones_like(state_0[..., 0])
    return (
        torch.stack(cubic_spline_coefficients(
            state_0[..., 0], dx0, state_f[..., 0], dxf, tf), -1),
        torch.stack(cubic_spline_coefficients(
            state_0[..., 1], dy0, state_f[..., 1], dyf, tf), -1),
        tf,
    )


def compute_spline_xyvaqrt(x_coefficients, y_coefficients, tf, N=10):
    t = torch.arange(N).unsqueeze(0).to(tf.device) * tf.unsqueeze(-1) / (N - 1)
    tp = t[..., None] ** torch.arange(4).to(tf.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]
                                       ).to(tf.device) * torch.arange(4).to(tf.device)
    ddtp = t[..., None] ** torch.tensor([0, 0, 0, 1]).to(
        tf.device) * torch.tensor([0, 0, 2, 6]).to(tf.device)
    x_coefficients = x_coefficients.unsqueeze(-1)
    y_coefficients = y_coefficients.unsqueeze(-1)
    vx = dtp @ x_coefficients
    vy = dtp @ y_coefficients
    v = torch.hypot(vx, vy)
    v_pos = torch.clip(v, min=1e-4)
    ax = ddtp @ x_coefficients
    ay = ddtp @ y_coefficients
    a = (ax * vx + ay * vy) / v_pos
    r = (-ax * vy + ay * vx) / (v_pos ** 2)
    yaw = torch.atan2(vy, vx)  # TODO(pkarkus) this is invalid for v=0
    return torch.cat((
        tp @ x_coefficients,
        tp @ y_coefficients,
        v,
        a,
        yaw,
        r,
        t.unsqueeze(-1),
    ), -1)
    

def interp_lanes(lane, extrapolate=True):
    """ generate interpolants for lanes 

    Args:
        lane (np.array()): [Nx3]

    Returns:

    """
    if isinstance(lane, torch.Tensor):
        lane = lane.cpu().numpy()
    ds = np.cumsum(
        np.hstack([0., np.linalg.norm(lane[1:, :2]-lane[:-1, :2], axis=-1)]))

    if extrapolate:
        # Allow extrapolation:
        return interp1d(ds, lane, fill_value="extrapolate", assume_sorted=True, axis=0), lane[0]
    else:
        # Nans for extrapolation
        return interp1d(ds, lane, bounds_error=False, assume_sorted=True, axis=0), lane[0]


def batch_rotate_2D(xy, theta):
    if isinstance(xy, torch.Tensor):
        x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
        y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
        return torch.stack([x1, y1], dim=-1)
    elif isinstance(xy, np.ndarray):
        x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
        y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
        return np.concatenate((x1[..., None], y1[..., None]), axis=-1)


class SplinePlanner(object):
    def __init__(self, device, dx_grid=None, dy_grid=None, acce_grid=None, dyaw_grid=None, max_steer=0.5, max_rvel=8,
                 acce_bound=[-6, 4], vbound=[-10, 30], spline_order=3, dt=0.5):
        self.spline_order = spline_order
        self.device = device
        assert spline_order == 3
        if dx_grid is None:
            self.dx_grid = torch.tensor([-4., 0, 4.]).to(self.device)
            # self.dx_grid = torch.tensor([0.]).to(self.device)
        else:
            self.dx_grid = dx_grid
        if dy_grid is None:
            self.dy_grid = torch.tensor([-4., -2., 0, 2., 4.]).to(self.device)
        else:
            self.dy_grid = dy_grid
        if acce_grid is None:
            self.acce_grid = torch.tensor([-3., -2., -1., -0.5, 0., 0.5, 1., 2.]).to(self.device)
            # self.acce_grid = torch.tensor([-1., 0., 1.]).to(self.device)
        else:
            self.acce_grid = acce_grid
        self.d_lane_lat_grid = torch.tensor([-0.5, 0, 0.5]).to(self.device)

        if dyaw_grid is None:
            self.dyaw_grid = torch.tensor(
                [-np.pi / 12, 0, np.pi / 12]).to(self.device)
        else:
            self.dyaw_grid = dyaw_grid
        self.max_steer = max_steer
        self.max_rvel = max_rvel
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.dt = dt

    def calc_trajectories(self, x0, tf, xf):
        if x0.ndim == 1:
            x0_tile = x0.tile(xf.shape[0], 1)
            xc, yc, tf_vect = compute_interpolating_spline(x0_tile, xf, tf)
        elif x0.ndim == xf.ndim:
            xc, yc, tf_vect = compute_interpolating_spline(x0, xf, tf)
        else:
            raise ValueError("wrong dimension for x0")
        traj = compute_spline_xyvaqrt(xc, yc, tf_vect, N=round(tf/self.dt) + 1)  # +1 for t0
        return traj

    def gen_terminals_lane_original(self, x0, tf, lanes):
        if lanes is None:
            return self.gen_terminals(x0, tf)

        gs = [self.dx_grid.shape[0], self.acce_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(1, 1, gs[1], 1).flatten()
        dv = self.acce_grid[None, None, :, None].repeat(
            gs[0], 1, 1, 1).flatten()*tf

        delta_x = list()

        assert x0.ndim in [1, 2], "x0 must have dimension 1 or 2"

        is_batched = (x0.ndim > 1)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        for lane in lanes:
            f, p_start = lane
            p_start = torch.from_numpy(p_start).to(x0.device)
            offset = x0[:, :2]-p_start[None, :2]
            s_offset = offset[:, 0] * \
                torch.cos(p_start[2])+offset[:, 1]*torch.sin(p_start[2])
            ds = (dx+dv/2*tf).unsqueeze(0)+x0[:, 2:3]*tf
            ss = ds + s_offset.unsqueeze(-1)
            xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                torch.float).to(x0.device)
            delta_x.append(torch.cat((xyyaw[..., :2], dv.tile(
                x0.shape[0], 1).unsqueeze(-1)+x0[:, None, 2:3], xyyaw[..., 2:]), -1))

        delta_x = torch.cat(delta_x, -2)

        if not is_batched:
            delta_x = delta_x.squeeze(0)
        return delta_x

    def gen_terminals_lane(self, x0, tf, lanes):
        if lanes is None:
            return self.gen_terminals(x0, tf)

        gs = [self.d_lane_lat_grid.shape[0], self.acce_grid.shape[0]]
        dlat = self.d_lane_lat_grid[:, None, None, None].repeat(1, 1, gs[1], 1).flatten()
        dv = self.acce_grid[None, None, :, None].repeat(
            gs[0], 1, 1, 1).flatten()*tf  

        delta_x = list()

        assert x0.ndim in [1, 2], "x0 must have dimension 1 or 2"
        is_batched = (x0.ndim > 1)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        for lane in lanes:
            f, p_start = lane  # f: interplation function f(ds)--> lane_x,y,yaw
            if isinstance(p_start, np.ndarray):
                p_start = torch.from_numpy(p_start)
            p_start = p_start.to(x0.device)
            offset = x0[:, :2]-p_start[None, :2]
            s_offset = offset[:, 0] * \
                torch.cos(p_start[2])+offset[:, 1]*torch.sin(p_start[2])  # distance projected onto lane, from its starting point

            # TODO this can be wildly inaccurate when lane is strongly curved
            # instead we should use the projection of current state onto lane, and find the distance along lane
            # we can do that by storing cumsum along lane, and assuming straight path from the closest lane point

            v0 = x0[:, 2:3]
            vf = v0 + dv.unsqueeze(0)
            # Replace negative velocity with stopping
            vf = torch.maximum(vf, torch.zeros_like(vf))

            ds = (v0 + vf) * 0.5 * tf  # delta distance along lane. dx is from grid, dv is final velocity from grid, average delta vel is dv/2, x0[:, 2:3] current velo 
            ss = ds + s_offset.unsqueeze(-1)  # target distance along lane
            xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                torch.float).to(x0.device)  # interpolate lane for target ds 

            # y offset in the direction of lane normal
            dlat_xy = dlat.unsqueeze(0).unsqueeze(-1) * torch.stack([torch.sin(xyyaw[..., 2]), -torch.cos(xyyaw[..., 2])], dim=-1)
                
            target_xyvh = torch.cat((
                    xyyaw[..., :1] + dlat_xy[..., :1], 
                    xyyaw[..., 1:2] + dlat_xy[..., 1:2], 
                    x0[:, None, 2:3] + dv.tile(x0.shape[0], 1).unsqueeze(-1),
                    xyyaw[..., 2:]
                ), -1)  # xyh --> xyvh. insert target velo = d0+dv

            # Filter nans (extrapolation) and negative vel target
            assert target_xyvh.shape[0] == 1, "No batch support for now, it would need ragged tensor"
            target_xyvh = target_xyvh.squeeze(0)  # N, 4
            target_xyvh = target_xyvh[torch.logical_not(target_xyvh[:, 2].isnan()) & (target_xyvh[:, 2] >= 0)]  # N*, 4
            target_xyvh = target_xyvh.unsqueeze(0)  # 1, N*, 4
            
            delta_x.append(target_xyvh)

        delta_x = torch.cat(delta_x, -2)

        if not is_batched:
            delta_x = delta_x.squeeze(0)
        return delta_x

    def gen_terminals(self, x0, tf):
        gs = [self.dx_grid.shape[0], self.dy_grid.shape[0],
              self.acce_grid.shape[0], self.dyaw_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(
            1, gs[1], gs[2], gs[3]).flatten()
        dy = self.dy_grid[None, :, None, None].repeat(
            gs[0], 1, gs[2], gs[3]).flatten()
        dv = tf * self.acce_grid[None, None, :,
                                 None].repeat(gs[0], gs[1], 1, gs[3]).flatten()
        dyaw = self.dyaw_grid[None, None, None, :].repeat(
            gs[0], gs[1], gs[2], 1).flatten()
        delta_x = torch.stack([dx, dy, dv, dyaw], -1)

        if x0.ndim == 1:
            xy = torch.cat(
                (delta_x[:, 0:1] + delta_x[:, 2:3] / 2 * tf + x0[2:3] * tf, delta_x[:, 1:2]), -1)
            rotated_xy = batch_rotate_2D(xy, x0[3]) + x0[:2]
            return torch.cat((rotated_xy, delta_x[:, 2:] + x0[2:]), -1) + x0[None, :]
        elif x0.ndim == 2:

            delta_x = torch.tile(delta_x, [x0.shape[0], 1, 1])
            xy = torch.cat(
                (delta_x[:, :, 0:1] + delta_x[:, :, 2:3] / 2 * tf + x0[:, None, 2:3] * tf, delta_x[:, :, 1:2]), -1)
            rotated_xy = batch_rotate_2D(
                xy, x0[:, 3:4]) + x0[:, None, :2]

            return torch.cat((rotated_xy, delta_x[:, :, 2:] + x0[:, None, 2:]), -1) + x0[:, None, :]
        else:
            raise ValueError("x0 must have dimension 1 or 2")

    def feasible_flag(self, traj):
        feas_flag = ((traj[..., 2] >= self.vbound[0]) & (traj[..., 2] < self.vbound[1]) &
                     (traj[..., 3] >= self.acce_bound[0]) & (traj[..., 3] <= self.acce_bound[1]) &
                     (torch.abs(traj[..., 5] * traj[..., 2]) <= self.max_rvel) & (
            torch.abs(traj[..., 2]) * self.max_steer >= torch.abs(traj[..., 5]))).all(1)
        return feas_flag

    def gen_trajectories(self, x0, tf, lanes=None, dyn_filter=True):
        if lanes is None:
            xf = self.gen_terminals(x0, tf)
        else:
            lane_interp = [interp_lanes(lane, extrapolate=False) for lane in lanes]

            xf = self.gen_terminals_lane(
                x0, tf, lane_interp)

        # x, y, v, a, yaw,r, t
        traj = self.calc_trajectories(x0, tf, xf)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
            return traj[feas_flag, :], xf[feas_flag, :]
        else:
            return traj, xf

    def gen_trajectory_batch(self, x0_set, tf, lanes=None, dyn_filter=True):
        # x0_set states (n, 4) for x, y, vel, yaw

        if lanes is None:
            xf_set = self.gen_terminals(x0_set, tf)
        else:
            # Do not allow extrapolation, will return nan
            lane_interp = [interp_lanes(lane, extrapolate=False) for lane in lanes]
            # x, y, v, yaw
            xf_set = self.gen_terminals_lane(x0_set, tf, lane_interp)

        num_node = x0_set.shape[0]
        num = xf_set.shape[1]
        x0_tiled = torch.tile(x0_set, [num, 1])
        xf_tiled = xf_set.reshape(-1, xf_set.shape[-1])
        # x, y, v, a, yaw,r, t
        traj = self.calc_trajectories(x0_tiled, tf, xf_tiled)

        # yaw values are incorrect when v=0, correct it by taking yaw at t-1
        yaw_tm1 = x0_tiled[:, 3]  # this is x, y, v, yaw
        for t in range(traj.shape[1]):
            # traj is x, y, v, a, yaw,r, t
            invalid_yaw_flag = torch.isclose(traj[:, t, 2], torch.zeros((), dtype=traj.dtype, device=traj.device))
            traj[invalid_yaw_flag, t, 4] = yaw_tm1[invalid_yaw_flag]
            yaw_tm1 = traj[:, t, 4]

        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
        else:
            feas_flag = torch.ones(
                num * num_node, dtype=torch.bool).to(x0_set.device)
        feas_flag = feas_flag.reshape(num_node, num)
        traj = traj.reshape(num_node, num, *traj.shape[1:])
        return [traj[i, feas_flag[i]] for i in range(num_node)], xf_tiled

    def gen_trajectory_tree(self, x0, tf, n_layers, dyn_filter=True):
        trajs = list()
        nodes = [x0[None, :]]
        for i in range(n_layers):
            xf = self.gen_terminals(nodes[i], tf)
            x0i = torch.tile(nodes[i], [xf.shape[1], 1])
            xf = xf.reshape(-1, xf.shape[-1])

            traj = self.calc_trajectories(x0i, tf, xf)
            if dyn_filter:
                feas_flag = self.feasible_flag(traj)
                traj = traj[feas_flag]
                xf = xf[feas_flag]

            trajs.append(traj)

            nodes.append(xf.reshape(-1, xf.shape[-1]))
        return trajs, nodes[1:]

