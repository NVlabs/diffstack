from diffstack.dynamics.base import DynType, Dynamics
from diffstack.utils.math_utils import soft_sat
import torch
import numpy as np


class DoubleIntegrator(Dynamics):
    def __init__(self, dt, name="DI_model", abound=None, vbound=None):
        self._name = name
        self._type = DynType.DI
        self.xdim = 4
        self.udim = 2
        self.dt = dt
        self.cyclic_state = list()
        self.vbound = vbound
        self.abound = abound

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            return np.hstack((x[..., 2:], u))
        elif isinstance(x, torch.Tensor):
            return torch.cat((x[..., 2:], u), dim=-1)
        else:
            raise NotImplementedError

    def step(self, x, u, bound=True, return_jacobian=False):
        if self.abound is None:
            bound = False

        if isinstance(x, np.ndarray):
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)
            xn = np.hstack(
                (
                    (x[..., 2:4] + 0.5 * u * self.dt) * self.dt + x[..., 0:2],
                    x[..., 2:4] + u * self.dt,
                )
            )
            if return_jacobian:
                raise NotImplementedError
            else:
                return xn
        elif isinstance(x, torch.Tensor):
            if bound:
                lb, ub = self.ubound(x)
                u = torch.clip(u, min=lb, max=ub)
            xn = torch.clone(x)
            xn[..., 0:2] += (x[..., 2:4] + 0.5 * u * self.dt) * self.dt
            xn[..., 2:4] += u * self.dt
            if return_jacobian:
                jacx = torch.cat(
                    [
                        torch.zeros([4, 2]),
                        torch.cat([torch.eye(2) * self.dt, torch.zeros([2, 2])], 0),
                    ],
                    -1,
                ) + torch.eye(4)
                jacx = torch.tile(jacx, (*x.shape[:-1], 1, 1)).to(x.device)

                jacu = torch.cat(
                    [torch.eye(2) * self.dt**2 * 2, torch.eye(2) * self.dt], 0
                )
                jacu = torch.tile(jacu, (*x.shape[:-1], 1, 1)).to(x.device)
                return xn, jacx, jacu
            else:
                return xn
        else:
            raise NotImplementedError

    def get_x_Gaussian_from_u(self, x, mu_u, var_u):
        mu_x, _, jacu = self.step(x, mu_u, bound=False, return_jacobian=True)

        var_u_mat = torch.diag_embed(var_u)
        var_x = torch.matmul(torch.matmul(jacu, var_u_mat), jacu.transpose(-1, -2))
        return mu_x, var_x

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if self.vbound is None:
            if isinstance(x, np.ndarray):
                lb = np.ones_like(x[..., 2:]) * self.abound[0]
                ub = np.ones_like(x[..., 2:]) * self.abound[1]

            elif isinstance(x, torch.Tensor):
                lb = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 0]
                ).to(x.device)
                ub = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 1]
                ).to(x.device)

            else:
                raise NotImplementedError
        else:
            if isinstance(x, np.ndarray):
                lb = (x[..., 2:] > self.vbound[0]) * self.abound[0]
                ub = (x[..., 2:] < self.vbound[1]) * self.abound[1]

            elif isinstance(x, torch.Tensor):
                lb = (
                    x[..., 2:] > torch.from_numpy(self.vbound[0]).to(x.device)
                ) * torch.from_numpy(self.abound[0]).to(x.device)
                ub = (
                    x[..., 2:] < torch.from_numpy(self.vbound[1]).to(x.device)
                ) * torch.from_numpy(self.abound[1]).to(x.device)
            else:
                raise NotImplementedError
        return lb, ub

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        # return torch.atan2(x[..., 3:], x[..., 2:3])
        return torch.zeros_like(x[..., 0:1])

    def inverse_dyn(self, x, xp):
        return (xp[..., 2:] - x[..., 2:]) / self.dt

    def calculate_vel(self, pos, yaw, mask):
        vel = (pos[..., 1:, :] - pos[..., :-1, :]) / self.dt
        if isinstance(pos, torch.Tensor):
            # right finite difference velocity
            vel_r = torch.cat((vel[..., 0:1, :], vel), dim=-2)
            # left finite difference velocity
            vel_l = torch.cat((vel, vel[..., -1:, :]), dim=-2)
            mask_r = torch.roll(mask, 1, dims=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask

            mask_l = torch.roll(mask, -1, dims=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                (mask_l & mask_r).unsqueeze(-1) * (vel_r + vel_l) / 2
                + (mask_l & (~mask_r)).unsqueeze(-1) * vel_l
                + (mask_r & (~mask_l)).unsqueeze(-1) * vel_r
            )
        elif isinstance(pos, np.ndarray):
            # right finite difference velocity
            vel_r = np.concatenate((vel[..., 0:1, :], vel), axis=-2)
            # left finite difference velocity
            vel_l = np.concatenate((vel, vel[..., -1:, :]), axis=-2)
            mask_r = np.roll(mask, 1, axis=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask
            mask_l = np.roll(mask, -1, axis=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                np.expand_dims(mask_l & mask_r, -1) * (vel_r + vel_l) / 2
                + np.expand_dims(mask_l & (~mask_r), -1) * vel_l
                + np.expand_dims(mask_r & (~mask_l), -1) * vel_r
            )
        else:
            raise NotImplementedError
        return vel

    def get_state(self, pos, yaw, dt, mask):
        vel = self.calculate_vel(pos, yaw, mask)
        if isinstance(vel, np.ndarray):
            return np.concatenate((pos, vel), -1)
        elif isinstance(vel, torch.Tensor):
            return torch.cat((pos, vel), -1)

    def forward_dynamics(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        include_step0: bool = False,
    ):
        if include_step0:
            raise NotImplementedError

        if isinstance(u, np.ndarray):
            u = np.clip(u, self.abound[0], self.abound[1])
            delta_v = np.cumsum(u * self.dt, -2)
            vel = x0[..., np.newaxis, 2:] + delta_v
            vel = np.clip(vel, self.vbound[0], self.vbound[1])
            delta_xy = np.cumsum(vel * self.dt, -2)
            xy = x0[..., np.newaxis, :2] + delta_xy

            traj = np.concatenate((xy, vel), -1)
        elif isinstance(u, torch.Tensor):
            u = soft_sat(u, self.abound[0], self.abound[1])
            delta_v = torch.cumsum(u * self.dt, -2)
            vel = x0[..., 2:].unsqueeze(-2) + delta_v
            vel = soft_sat(vel, self.vbound[0], self.vbound[1])
            delta_xy = torch.cumsum(vel * self.dt, -2)
            xy = x0[..., :2].unsqueeze(-2) + delta_xy

            traj = torch.cat((xy, vel), -1)
        return traj
