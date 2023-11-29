from diffstack.dynamics.base import DynType, Dynamics
from diffstack.utils.math_utils import soft_sat
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.autograd.functional import jacobian
import diffstack.utils.geometry_utils as GeoUtils


class Unicycle(Dynamics):
    def __init__(
        self,
        dt,
        name=None,
        max_steer=0.5,
        max_yawvel=8,
        acce_bound=[-6, 4],
        vbound=[-10, 30],
    ):
        self.dt = dt
        self._name = name
        self._type = DynType.UNICYCLE
        self.xdim = 4
        self.udim = 2
        self.cyclic_state = [3]
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.max_steer = max_steer
        self.max_yawvel = max_yawvel

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            theta = x[..., 3:4]
            dxdt = np.hstack(
                (np.cos(theta) * x[..., 2:3], np.sin(theta) * x[..., 2:3], u)
            )
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (torch.cos(theta) * x[..., 2:3], torch.sin(theta) * x[..., 2:3], u),
                dim=-1,
            )
        else:
            raise NotImplementedError
        return dxdt

    def step(self, x, u, bound=True, return_jacobian=False):
        assert x.shape[:-1] == u.shape[:-1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)

            theta = x[..., 3:4]
            cos_theta_p = np.cos(theta) - 0.5 * np.sin(theta) * u[..., 1:] * self.dt
            sin_theta_p = np.sin(theta) + 0.5 * np.cos(theta) * u[..., 1:] * self.dt
            vel_p = x[..., 2:3] + u[..., 0:1] * self.dt * 0.5

            dxdt = np.hstack(
                (
                    cos_theta_p * vel_p,
                    sin_theta_p * vel_p,
                    u,
                )
            )
            xp = x + dxdt * self.dt
            if return_jacobian:
                d_cos_theta_p_d_theta = (
                    -np.sin(theta) - 0.5 * np.cos(theta) * u[..., 1:] * self.dt
                )
                d_sin_theta_p_d_theta = (
                    np.cos(theta) - 0.5 * np.sin(theta) * u[..., 1:] * self.dt
                )
                d_vel_p_d_a = 0.5 * self.dt
                d_cos_theta_p_d_yaw = -0.5 * np.sin(theta) * self.dt
                d_sin_theta_p_d_yaw = 0.5 * np.cos(theta) * self.dt
                jacx = np.tile(np.eye(4), (*x.shape[:-1], 1, 1))
                jacx[..., 0, 2:3] = cos_theta_p * self.dt
                jacx[..., 0, 3:4] = vel_p * self.dt * d_cos_theta_p_d_theta
                jacx[..., 1, 2:3] = sin_theta_p * self.dt
                jacx[..., 1, 3:4] = vel_p * self.dt * d_sin_theta_p_d_theta

                jacu = np.zeros((*x.shape[:-1], 4, 2))
                jacu[..., 0, 0:1] = cos_theta_p * self.dt * d_vel_p_d_a
                jacu[..., 0, 1:2] = vel_p * self.dt * d_cos_theta_p_d_yaw
                jacu[..., 1, 0:1] = sin_theta_p * self.dt * d_vel_p_d_a
                jacu[..., 1, 1:2] = vel_p * self.dt * d_sin_theta_p_d_yaw
                jacu[..., 2, 0:1] = self.dt
                jacu[..., 3, 1:2] = self.dt

                return xp, jacx, jacu
            else:
                return xp
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            if bound:
                lb, ub = self.ubound(x)
                # s = (u - lb) / torch.clip(ub - lb, min=1e-3)
                # u = lb + (ub - lb) * torch.sigmoid(s)
                u = torch.clip(u, lb, ub)

            theta = x[..., 3:4]
            cos_theta_p = (
                torch.cos(theta) - 0.5 * torch.sin(theta) * u[..., 1:] * self.dt
            )
            sin_theta_p = (
                torch.sin(theta) + 0.5 * torch.cos(theta) * u[..., 1:] * self.dt
            )
            vel_p = x[..., 2:3] + u[..., 0:1] * self.dt * 0.5
            dxdt = torch.cat(
                (
                    cos_theta_p * vel_p,
                    sin_theta_p * vel_p,
                    u,
                ),
                dim=-1,
            )
            xp = x + dxdt * self.dt
            if return_jacobian:
                d_cos_theta_p_d_theta = (
                    -torch.sin(theta) - 0.5 * torch.cos(theta) * u[..., 1:] * self.dt
                )
                d_sin_theta_p_d_theta = (
                    torch.cos(theta) - 0.5 * torch.sin(theta) * u[..., 1:] * self.dt
                )
                d_vel_p_d_a = 0.5 * self.dt
                d_cos_theta_p_d_yaw = -0.5 * torch.sin(theta) * self.dt
                d_sin_theta_p_d_yaw = 0.5 * torch.cos(theta) * self.dt
                eye4 = torch.tile(torch.eye(4, device=x.device), (*x.shape[:-1], 1, 1))
                jacxy = torch.zeros((*x.shape[:-1], 4, 2), device=x.device)
                zeros21 = torch.zeros((*x.shape[:-1], 2, 1), device=x.device)
                jacv = (
                    torch.cat(
                        [cos_theta_p.unsqueeze(-2), sin_theta_p.unsqueeze(-2), zeros21],
                        -2,
                    )
                    * self.dt
                )
                jactheta = (
                    torch.cat(
                        [
                            (vel_p * d_cos_theta_p_d_theta).unsqueeze(-2),
                            (vel_p * d_sin_theta_p_d_theta).unsqueeze(-2),
                            zeros21,
                        ],
                        -2,
                    )
                    * self.dt
                )
                jacx = torch.cat([jacxy, jacv, jactheta], -1) + eye4
                # jacx = torch.tile(torch.eye(4,device=x.device), (*x.shape[:-1], 1, 1))
                # jacx[...,0,2:3] = cos_theta_p*self.dt
                # jacx[...,0,3:4] = vel_p*self.dt*d_cos_theta_p_d_theta
                # jacx[...,1,2:3] = sin_theta_p*self.dt
                # jacx[...,1,3:4] = vel_p*self.dt*d_sin_theta_p_d_theta

                jacxy_a = (
                    torch.cat(
                        [cos_theta_p.unsqueeze(-2), sin_theta_p.unsqueeze(-2)], -2
                    )
                    * self.dt
                    * d_vel_p_d_a
                )
                jacxy_yaw = (
                    torch.cat(
                        [
                            (vel_p * d_cos_theta_p_d_yaw).unsqueeze(-2),
                            (vel_p * d_sin_theta_p_d_yaw).unsqueeze(-2),
                        ],
                        -2,
                    )
                    * self.dt
                )
                eye2 = torch.tile(torch.eye(2, device=x.device), (*x.shape[:-1], 1, 1))
                jacu = torch.cat(
                    [torch.cat([jacxy_a, jacxy_yaw], -1), eye2 * self.dt], -2
                )
                # jacu = torch.zeros((*x.shape[:-1], 4, 2),device=x.device)
                # jacu[...,0,0:1] = cos_theta_p*self.dt*d_vel_p_d_a
                # jacu[...,0,1:2] = vel_p*self.dt*d_cos_theta_p_d_yaw
                # jacu[...,1,0:1] = sin_theta_p*self.dt*d_vel_p_d_a
                # jacu[...,1,1:2] = vel_p*self.dt*d_sin_theta_p_d_yaw
                # jacu[...,2,0:1] = self.dt
                # jacu[...,3,1:2] = self.dt
                return xp, jacx, jacu
            else:
                return xp
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
        if isinstance(x, np.ndarray):
            v = x[..., 2:3]
            vclip = np.clip(np.abs(v), a_min=0.1, a_max=None)

            yawbound = np.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            acce_lb = np.clip(
                np.clip(self.vbound[0] - v, None, self.acce_bound[1]),
                self.acce_bound[0],
                None,
            )
            acce_ub = np.clip(
                np.clip(self.vbound[1] - v, self.acce_bound[0], None),
                None,
                self.acce_bound[1],
            )
            lb = np.concatenate((acce_lb, -yawbound), -1)
            ub = np.concatenate((acce_ub, yawbound), -1)
            return lb, ub
        elif isinstance(x, torch.Tensor):
            v = x[..., 2:3]
            vclip = torch.clip(torch.abs(v), min=0.1)
            yawbound = torch.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            yawbound = torch.clip(yawbound, min=0.1)
            acce_lb = torch.clip(
                torch.clip(self.vbound[0] - v, max=self.acce_bound[1]),
                min=self.acce_bound[0],
            )
            acce_ub = torch.clip(
                torch.clip(self.vbound[1] - v, min=self.acce_bound[0]),
                max=self.acce_bound[1],
            )
            lb = torch.cat((acce_lb, -yawbound), dim=-1)
            ub = torch.cat((acce_ub, yawbound), dim=-1)
            return lb, ub

        else:
            raise NotImplementedError

    def uniform_sample_xp(self, x, num_sample):
        if isinstance(x, torch.Tensor):
            u_lb, u_ub = self.ubound(x)
            u_sample = torch.rand(
                *x.shape[:-1], num_sample, self.udim, device=x.device
            ) * (u_ub - u_lb).unsqueeze(-2) + u_lb.unsqueeze(-2)
            xp = self.step(
                x.unsqueeze(-2).repeat_interleave(num_sample, -2), u_sample, bound=False
            )
        elif isinstance(x, np.ndarray):
            u_lb, u_ub = self.ubound(x)
            u_sample = np.random.uniform(
                u_lb[..., None, :],
                u_ub[..., None, :],
                (*x.shape[:-1], num_sample, self.udim),
            )
            xp = self.step(
                x[..., None, :].repeat(num_sample, -2), u_sample, bound=False
            )
        else:
            raise NotImplementedError
        return xp

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        return x[..., 3:]

    @staticmethod
    def state2vel(x):
        return x[..., 2:3]

    @staticmethod
    def state2xyvsc(x):
        return torch.cat([x[..., :3], torch.sin(x[..., 3:]), torch.cos(x[..., 3:])], -1)

    @staticmethod
    def combine_to_state(xy, vel, yaw):
        if isinstance(xy, torch.Tensor):
            return torch.cat((xy, vel, yaw), -1)
        elif isinstance(xy, np.ndarray):
            return np.concatenate((xy, vel, yaw), -1)

    def calculate_vel(self, pos, yaw, mask, dt=None):
        if dt is None:
            dt = self.dt
        if isinstance(pos, torch.Tensor):
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
                yaw[..., 1:, :]
            )
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
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * np.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * np.sin(yaw[..., 1:, :])
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

    def inverse_dyn(self, x, xp, dt=None, mask=None):
        if dt is None:
            dt = self.dt
        dx = torch.cat(
            [xp[..., 2:3] - x[..., 2:3], GeoUtils.round_2pi(xp[..., 3:] - x[..., 3:])],
            -1,
        )
        u = dx / dt
        if mask is not None:
            u = u * mask[..., None]
        return u

    def get_state(self, pos, yaw, mask, dt=None):
        if dt is None:
            dt = self.dt
        vel = self.calculate_vel(pos, yaw, mask, dt)
        if isinstance(vel, np.ndarray):
            return np.concatenate((pos, vel, yaw), -1)
        elif isinstance(vel, torch.Tensor):
            return torch.cat((pos, vel, yaw), -1)

    @staticmethod
    def get_axay(x, u):
        yaw = Unicycle.state2yaw(x)
        vel = Unicycle.state2vel(x)
        acce = u[..., 0:1]
        r = u[..., 1:]
        stack_fun = torch.stack if isinstance(x, torch.Tensor) else np.stack
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        return stack_fun(
            [
                acce * cos(yaw) - vel * r * sin(yaw),
                acce * sin(yaw) + vel * r * cos(yaw),
            ],
            -1,
        )

    def forward_dynamics(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        mode="parallel",
        bound=True,
    ):
        """
        Integrate the state forward with initial state x0, action u
        Args:
            initial_states (Torch.tensor): state tensor of size [B, (A), 4]
            actions (Torch.tensor): action tensor of size [B, (A), T, 2]
        Returns:
            state tensor of size [B, (A), T, 4]
        """
        if mode == "chain":
            num_steps = u.shape[-2]
            x = [x0] + [None] * num_steps
            for t in range(num_steps):
                x[t + 1] = self.step(x[t], u[..., t, :], bound=bound)

            return torch.stack(x[1:], dim=-2)

        else:
            assert mode in ["parallel", "partial_parallel"]
            with torch.no_grad():
                num_steps = u.shape[-2]
                b = x0.shape[0]
                device = x0.device

                mat = torch.ones(num_steps + 1, num_steps + 1, device=device)
                mat = torch.tril(mat)
                mat = mat.repeat(b, 1, 1)

                mat2 = torch.ones(num_steps, num_steps + 1, device=device)
                mat2_h = torch.tril(mat2, diagonal=1)
                mat2_l = torch.tril(mat2, diagonal=-1)
                mat2 = torch.logical_xor(mat2_h, mat2_l).float() * 0.5
                mat2 = mat2.repeat(b, 1, 1)
                if x0.ndim == 3:
                    mat = mat.unsqueeze(1)
                    mat2 = mat2.unsqueeze(1)

            acc = u[..., :1]
            yaw = u[..., 1:]
            if bound:
                acc_clipped = soft_sat(acc, self.acce_bound[0], self.acce_bound[1])
            else:
                acc_clipped = acc

            if mode == "parallel":
                acc_paded = torch.cat(
                    (x0[..., -2:-1].unsqueeze(-2), acc_clipped * self.dt), dim=-2
                )

                v_raw = torch.matmul(mat, acc_paded)
                v_clipped = soft_sat(v_raw, self.vbound[0], self.vbound[1])
            else:
                v_clipped = [x0[..., 2:3]] + [None] * num_steps
                for t in range(num_steps):
                    vt = v_clipped[t]
                    acc_clipped_t = soft_sat(
                        acc_clipped[:, t], self.vbound[0] - vt, self.vbound[1] - vt
                    )
                    v_clipped[t + 1] = vt + acc_clipped_t * self.dt
                v_clipped = torch.stack(v_clipped, dim=-2)

            v_avg = torch.matmul(mat2, v_clipped)

            v = v_clipped[..., 1:, :]
            if bound:
                with torch.no_grad():
                    v_earlier = v_clipped[..., :-1, :]

                    yawbound = torch.minimum(
                        self.max_steer * torch.abs(v_earlier),
                        self.max_yawvel / torch.clip(torch.abs(v_earlier), min=0.1),
                    )
                    yawbound_clipped = torch.clip(yawbound, min=0.1)

                yaw_clipped = soft_sat(yaw, -yawbound_clipped, yawbound_clipped)
            else:
                yaw_clipped = yaw
            yawvel_paded = torch.cat(
                (x0[..., -1:].unsqueeze(-2), yaw_clipped * self.dt), dim=-2
            )
            yaw_full = torch.matmul(mat, yawvel_paded)
            yaw = yaw_full[..., 1:, :]

            # print('before clip', torch.cat((acc[0], yawvel[0]), dim=-1))
            # print('after clip', torch.cat((acc_clipped[0], yawvel_clipped[0]), dim=-1))

            yaw_earlier = yaw_full[..., :-1, :]
            vx = v_avg * torch.cos(yaw_earlier)
            vy = v_avg * torch.sin(yaw_earlier)
            v_all = torch.cat((vx, vy), dim=-1)

            # print('initial_states[0, -2:]', initial_states[0, -2:])
            # print('vx[0, :5]', vx[0, :5])

            v_all_paded = torch.cat(
                (x0[..., :2].unsqueeze(-2), v_all * self.dt), dim=-2
            )
            x_and_y = torch.matmul(mat, v_all_paded)
            x_and_y = x_and_y[..., 1:, :]

            x_all = torch.cat((x_and_y, v, yaw), dim=-1)
            return x_all

    # def propagate_and_linearize(self,x0,u,dt=None):
    #     if dt is None:
    #         dt = self.dt
    #     xp,_,_ = self.forward_dynamics(x0,u,dt,mode="chain")
    #     xl = torch.cat([x0.unsqueeze(1),xp[:,:-1]],1)
    #     A,B = jacobian(lambda x,u: self.step(x,u,dt),(xl,u))
    #     A = A.diagonal(dim1=0,dim2=3).diagonal(dim1=0,dim2=2).permute(2,3,0,1)
    #     B = B.diagonal(dim1=0,dim2=3).diagonal(dim1=0,dim2=2).permute(2,3,0,1)
    #     C = xp - (A@xl.unsqueeze(-1)+B@u.unsqueeze(-1)).squeeze(-1)
    #     return xp,A,B,C


class Unicycle_xyvsc(Dynamics):
    def __init__(
        self, dt,name = None, max_steer=0.5, max_yawvel=8, acce_bound=[-6, 4], vbound=[-10, 30]
    ):
        self.dt = dt
        self._name = name
        self._type = DynType.UNICYCLE
        self.xdim = 5
        self.udim = 2
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.max_steer = max_steer
        self.max_yawvel = max_yawvel

    def step(self, x, u, bound=True, return_jacobian=False):
        assert x.shape[:-1] == u.shape[:-1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)

            s = x[..., 3:4]
            c = x[..., 4:5]
            c_step = c-0.5*s*u[...,1:]*self.dt
            s_step = s+0.5*c*u[...,1:]*self.dt
            vel_step = x[..., 2:3] + u[..., 0:1] * self.dt * 0.5
            yaw = u[..., 1:2]
            cp = c*np.cos(yaw*self.dt)-s*np.sin(yaw*self.dt)
            sp = s*np.cos(yaw*self.dt)+c*np.sin(yaw*self.dt)
            dx = np.hstack(
                (
                    c_step * vel_step*self.dt,
                    s_step * vel_step*self.dt,
                    u[...,]*self.dt,
                    sp-s,
                    cp-c,
                )
            )
            xp = x + dx
            if return_jacobian:
                raise NotImplementedError
                # d_cos_theta_p_d_theta = -np.sin(theta)-0.5*np.cos(theta)*u[...,1:]*self.dt
                # d_sin_theta_p_d_theta = np.cos(theta)-0.5*np.sin(theta)*u[...,1:]*self.dt
                # d_vel_p_d_a = 0.5*self.dt
                # d_cos_theta_p_d_yaw = -0.5*np.sin(theta)*self.dt
                # d_sin_theta_p_d_yaw = 0.5*np.cos(theta)*self.dt
                # jacx = np.tile(np.eye(4), (*x.shape[:-1], 1, 1))
                # jacx[...,0,2:3] = cos_theta_p*self.dt
                # jacx[...,0,3:4] = vel_p*self.dt*d_cos_theta_p_d_theta
                # jacx[...,1,2:3] = sin_theta_p*self.dt
                # jacx[...,1,3:4] = vel_p*self.dt*d_sin_theta_p_d_theta

                # jacu = np.zeros((*x.shape[:-1], 4, 2))
                # jacu[...,0,0:1] = cos_theta_p*self.dt*d_vel_p_d_a
                # jacu[...,0,1:2] = vel_p*self.dt*d_cos_theta_p_d_yaw
                # jacu[...,1,0:1] = sin_theta_p*self.dt*d_vel_p_d_a
                # jacu[...,1,1:2] = vel_p*self.dt*d_sin_theta_p_d_yaw
                # jacu[...,2,0:1] = self.dt
                # jacu[...,3,1:2] = self.dt

                # return xp, jacx, jacu
            else:
                return xp
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            if bound:
                lb, ub = self.ubound(x)
                # s = (u - lb) / torch.clip(ub - lb, min=1e-3)
                # u = lb + (ub - lb) * torch.sigmoid(s)
                u = torch.clip(u, lb, ub)

            s = x[..., 3:4]
            c = x[..., 4:5]
            c_step = c-0.5*s*u[...,1:]*self.dt
            s_step = s+0.5*c*u[...,1:]*self.dt
            vel_step = x[..., 2:3] + u[..., 0:1] * self.dt * 0.5
            yaw = u[..., 1:2]
            cp = c*torch.cos(yaw*self.dt)-s*torch.sin(yaw*self.dt)
            sp = s*torch.cos(yaw*self.dt)+c*torch.sin(yaw*self.dt)
            dx = torch.cat(
                (
                    c_step * vel_step*self.dt,
                    s_step * vel_step*self.dt,
                    u[...,:1]*self.dt,
                    sp-s,
                    cp-c,
                ),-1
            )
            xp = x + dx
            if return_jacobian:
                raise NotImplementedError
                # d_cos_theta_p_d_theta = -torch.sin(theta)-0.5*torch.cos(theta)*u[...,1:]*self.dt
                # d_sin_theta_p_d_theta = torch.cos(theta)-0.5*torch.sin(theta)*u[...,1:]*self.dt
                # d_vel_p_d_a = 0.5*self.dt
                # d_cos_theta_p_d_yaw = -0.5*torch.sin(theta)*self.dt
                # d_sin_theta_p_d_yaw = 0.5*torch.cos(theta)*self.dt
                # eye4 = torch.tile(torch.eye(4,device=x.device), (*x.shape[:-1], 1, 1))
                # jacxy = torch.zeros((*x.shape[:-1], 4, 2),device=x.device)
                # zeros21 = torch.zeros((*x.shape[:-1], 2, 1),device=x.device)
                # jacv = torch.cat([cos_theta_p.unsqueeze(-2),sin_theta_p.unsqueeze(-2),zeros21],-2)*self.dt
                # jactheta = torch.cat([(vel_p*d_cos_theta_p_d_theta).unsqueeze(-2),(vel_p*d_sin_theta_p_d_theta).unsqueeze(-2),zeros21],-2)*self.dt
                # jacx = torch.cat([jacxy,jacv,jactheta],-1)+eye4
                # # jacx = torch.tile(torch.eye(4,device=x.device), (*x.shape[:-1], 1, 1))
                # # jacx[...,0,2:3] = cos_theta_p*self.dt
                # # jacx[...,0,3:4] = vel_p*self.dt*d_cos_theta_p_d_theta
                # # jacx[...,1,2:3] = sin_theta_p*self.dt
                # # jacx[...,1,3:4] = vel_p*self.dt*d_sin_theta_p_d_theta
                

                
                # jacxy_a = torch.cat([cos_theta_p.unsqueeze(-2),sin_theta_p.unsqueeze(-2)],-2)*self.dt*d_vel_p_d_a
                # jacxy_yaw = torch.cat([(vel_p*d_cos_theta_p_d_yaw).unsqueeze(-2),(vel_p*d_sin_theta_p_d_yaw).unsqueeze(-2)],-2)*self.dt
                # eye2 = torch.tile(torch.eye(2,device=x.device), (*x.shape[:-1], 1, 1))
                # jacu = torch.cat([torch.cat([jacxy_a,jacxy_yaw],-1),eye2*self.dt],-2)
                # # jacu = torch.zeros((*x.shape[:-1], 4, 2),device=x.device)
                # # jacu[...,0,0:1] = cos_theta_p*self.dt*d_vel_p_d_a
                # # jacu[...,0,1:2] = vel_p*self.dt*d_cos_theta_p_d_yaw
                # # jacu[...,1,0:1] = sin_theta_p*self.dt*d_vel_p_d_a
                # # jacu[...,1,1:2] = vel_p*self.dt*d_sin_theta_p_d_yaw
                # # jacu[...,2,0:1] = self.dt
                # # jacu[...,3,1:2] = self.dt
                return xp, jacx, jacu
            else:
                return xp
        else:
            raise NotImplementedError

    # def get_x_Gaussian_from_u(self,x,mu_u,var_u):
    #     mu_x,_,jacu = self.step(x, mu_u, bound=False, return_jacobian=True)
        
    #     var_u_mat = torch.diag_embed(var_u)
    #     var_x = torch.matmul(torch.matmul(jacu,var_u_mat),jacu.transpose(-1,-2))
    #     return mu_x,var_x
    
    def __call__(self, x, u):
        return self.step(x, u)

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if isinstance(x, np.ndarray):
            v = x[..., 2:3]
            vclip = np.clip(np.abs(v), a_min=0.1, a_max=None)
            yawbound = np.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            acce_lb = np.clip(
                np.clip(self.vbound[0] - v, None, self.acce_bound[1]),
                self.acce_bound[0],
                None,
            )
            acce_ub = np.clip(
                np.clip(self.vbound[1] - v, self.acce_bound[0], None),
                None,
                self.acce_bound[1],
            )
            lb = np.concatenate((acce_lb, -yawbound),-1)
            ub = np.concatenate((acce_ub, yawbound),-1)
            return lb, ub
        elif isinstance(x, torch.Tensor):
            v = x[..., 2:3]
            vclip = torch.clip(torch.abs(v),min=0.1)
            yawbound = torch.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            yawbound = torch.clip(yawbound, min=0.1)
            acce_lb = torch.clip(
                torch.clip(self.vbound[0] - v, max=self.acce_bound[1]),
                min=self.acce_bound[0],
            )
            acce_ub = torch.clip(
                torch.clip(self.vbound[1] - v, min=self.acce_bound[0]),
                max=self.acce_bound[1],
            )
            lb = torch.cat((acce_lb, -yawbound), dim=-1)
            ub = torch.cat((acce_ub, yawbound), dim=-1)
            return lb, ub

        else:
            raise NotImplementedError
    def uniform_sample_xp(self,x,num_sample):
        if isinstance(x,torch.Tensor):
            u_lb,u_ub = self.ubound(x)
            u_sample = torch.rand(*x.shape[:-1],num_sample,self.udim).to(x.device)*(u_ub-u_lb).unsqueeze(-2)+u_lb.unsqueeze(-2)
            xp = self.step(x.unsqueeze(-2).repeat_interleave(num_sample,-2),u_sample,bound=False)
        elif isinstance(x,np.ndarray):
            u_lb,u_ub = self.ubound(x)
            u_sample = np.random.uniform(u_lb[...,None,:],u_ub[...,None,:],(*x.shape[:-1],num_sample,self.udim))
            xp = self.step(x[...,None,:].repeat(num_sample,-2),u_sample,bound=False)
        else:
            raise NotImplementedError
        return xp
    
    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2sc(x):
        return x[..., 3:]
    
    @staticmethod
    def state2yaw(x):
        arctanfun = GeoUtils.ratan2 if isinstance(x,torch.Tensor) else np.arctan2
        return arctanfun(x[...,3:4],x[...,4:5])
    
    @staticmethod
    def state2vel(x):
        return x[..., 2:3]
    
    @staticmethod
    def combine_to_state(xy,vel,yaw):
        if isinstance(xy,torch.Tensor):
            return torch.cat((xy,vel,torch.sin(yaw),torch.cos(yaw)),-1)
        elif isinstance(xy,np.ndarray):
            return np.concatenate((xy,vel,np.sin(yaw),np.cos(yaw)),-1)

    def calculate_vel(self, pos, yaw, mask, dt=None):
        if dt is None:
            dt = self.dt
        if isinstance(pos, torch.Tensor):
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
                yaw[..., 1:, :]
            )
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
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * np.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * np.sin(yaw[..., 1:, :])
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
                np.expand_dims(mask_l & mask_r,-1) * (vel_r + vel_l) / 2
                + np.expand_dims(mask_l & (~mask_r),-1) * vel_l
                + np.expand_dims(mask_r & (~mask_l),-1) * vel_r
            )
        else:
            raise NotImplementedError
        return vel

    def inverse_dyn(self,x,xp,dt=None):
        if dt is None:
            dt = self.dt
        acce = (xp[2:3]-x[2:3])/dt
        arctanfun = GeoUtils.ratan2 if isinstance(x,torch.Tensor) else np.arctan2
        catfun = torch.cat if isinstance(x,torch.Tensor) else np.concatenate
        yawrate = (arctanfun(xp[3:4],xp[4:5])-arctanfun(x[3:4],x[4:5]))/dt
        return catfun([acce,yawrate],-1)
    

    def get_state(self,pos,yaw,mask,dt=None):
        if dt is None:
            dt = self.dt
        vel = self.calculate_vel(pos, yaw, mask,dt)
        return self.combine_to_state(pos,vel,yaw)

    def forward_dynamics(self,
                         x0: torch.Tensor,
                         u: torch.Tensor,
                         mode="chain",
                         bound = True,
                        ):
    
        """
        Integrate the state forward with initial state x0, action u
        Args:
            initial_states (Torch.tensor): state tensor of size [B, (A), 4]
            actions (Torch.tensor): action tensor of size [B, (A), T, 2]
        Returns:
            state tensor of size [B, (A), T, 4]
        """
        if mode=="chain":
            num_steps = u.shape[-2]
            x = [x0] + [None] * num_steps
            for t in range(num_steps):
                x[t + 1] = self.step(x[t], u[..., t, :],bound=bound)

            return torch.stack(x[1:], dim=-2)

        else:
            raise NotImplementedError






def test():
    model = Unicycle(0.1)
    x0 = torch.tensor([[1, 2, 3, 4]]).repeat_interleave(3, 0)
    u = torch.tensor([[1, 2]]).repeat_interleave(3, 0)
    x, jacx, jacu = model.step(x0, u, return_jacobian=True)


if __name__ == "__main__":
    test()
