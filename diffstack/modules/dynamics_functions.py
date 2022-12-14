import numpy as np
import torch
from typing import Dict, Iterable, Optional, Union, Any, Tuple

from diffstack.utils.utils import angle_wrap


class LinearizedDynamics(torch.nn.Module):
    # Wrapper around nn.Module, just so that we can identify it by type in mpc.py.
    def linearized(self, x, u, diff):
        raise NotImplementedError


class ExtendedUnicycleDynamics(LinearizedDynamics):
    """
    This class borrows from mpc the linearization with analytically computed gradients.
        """
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def dyn_fn(self, x, u, return_grad=False):
        x_p, y_p, phi, v = torch.unbind(x, dim=-1)
        omega, a = torch.unbind(u, dim=-1)
        dt = self.dt

        # if ego_pred_type == 'const_vel':
        #     return torch.stack([x_p + v * torch.cos(phi) * dt,
        #                         y_p + v * torch.sin(phi) * dt,
        #                         phi * torch.ones_like(a),
        #                         v], dim=-1)

        mask = torch.abs(omega) <= 1e-2
        omega = ~mask * omega + mask * 1e-2    # TODO why 1? shouldnt it be 0? i guess doesnt matter because we will not use it

        phi_p_omega_dt = angle_wrap(phi + omega * dt)

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_phi_p_omega_dt = torch.sin(phi_p_omega_dt)
        cos_phi_p_omega_dt = torch.cos(phi_p_omega_dt)
        a_over_omega = a / omega
        dsin_domega = (sin_phi_p_omega_dt - sin_phi) / omega
        dcos_domega = (cos_phi_p_omega_dt - cos_phi) / omega

        with torch.no_grad():
            # "This function cannot be differentiated "
            # "because for small omega we lost the dependency on omega. "
            # "Instead we need to manually compute the limit of gradient wrt. omega when omega --> 0."
            d1 = torch.stack([(x_p
                                + (a_over_omega) * dcos_domega
                                + v * dsin_domega
                                + (a_over_omega) * sin_phi_p_omega_dt * dt),
                                (y_p
                                - v * dcos_domega
                                + (a_over_omega) * dsin_domega
                                - (a_over_omega) * cos_phi_p_omega_dt * dt),
                                phi_p_omega_dt,
                                v + a * dt], dim=-1)                
            d2 = torch.stack([x_p + v * cos_phi * dt + (a / 2) * cos_phi * dt ** 2,
                                y_p + v * sin_phi * dt + (a / 2) * sin_phi * dt ** 2,
                                phi * torch.ones_like(a),
                                v + a * dt], dim=-1)
            next_states = torch.where(~mask.unsqueeze(-1), d1, d2)

        if return_grad:
            one = torch.ones_like(x_p)
            zero = torch.zeros_like(x_p)
            a_over_omega2 = a_over_omega / omega
            v_over_omega = v / omega

            # derivatives wrt. state dimensions
            g_state1 = torch.stack([
                torch.stack([one, zero, zero, zero], axis=-1),
                torch.stack([zero, one, zero, zero], axis=-1),
                torch.stack([
                    dcos_domega*v + dt*a_over_omega*cos_phi_p_omega_dt + a_over_omega2*(sin_phi - sin_phi_p_omega_dt), 
                    dcos_domega*a_over_omega + dt*a_over_omega*sin_phi_p_omega_dt - (v_over_omega)*(sin_phi - sin_phi_p_omega_dt), 
                    one, 
                    zero], axis=-1),
                torch.stack([
                    dsin_domega, 
                    (cos_phi - cos_phi_p_omega_dt) / omega, 
                    zero, 
                    one], axis=-1),
            ], axis=-1)  # ..., 4, 4

            g_control1 = torch.stack([
                # derivatives wrt. control dimensions
                torch.stack([
                    - 2.* dcos_domega * a_over_omega2 - dsin_domega * v_over_omega + a_over_omega * (dt * dt) * cos_phi_p_omega_dt - 2. * dt * a_over_omega2 * sin_phi_p_omega_dt + dt * v_over_omega * cos_phi_p_omega_dt, 
                    - 2.* dsin_domega * a_over_omega2 + dcos_domega * v_over_omega + a_over_omega * (dt * dt) * sin_phi_p_omega_dt + 2. * dt * a_over_omega2 * cos_phi_p_omega_dt + dt * v_over_omega * sin_phi_p_omega_dt, 
                    one * dt, 
                    zero], axis=-1),
                torch.stack([
                    (dcos_domega + dt * sin_phi_p_omega_dt) / omega, 
                    (dsin_domega - dt * cos_phi_p_omega_dt) / omega, 
                    zero,
                    one * dt], axis=-1),        

            ], axis=-1)   # ...., 4 (states), 2 (controls)

            # derivatives wrt. state dimensions
            g_state2 = torch.stack([
                torch.stack([one, zero, zero, zero], axis=-1),
                torch.stack([zero, one, zero, zero], axis=-1),
                torch.stack([
                    (-0.5*dt*a - v) * sin_phi*dt, 
                    (0.5*dt*a + v) * cos_phi*dt, 
                    one, 
                    zero], axis=-1),
                torch.stack([
                    dt * cos_phi, 
                    dt * sin_phi, 
                    zero, 
                    one], axis=-1),
            ], axis=-1)  # ..., 4, 4

            g_control2 = torch.stack([
                # derivatives wrt. control dimensions
                torch.stack([
                    -(1./3.*dt*a + 0.5*v) * dt*dt*sin_phi,
                    (1./3.*dt*a + 0.5*v) * dt*dt*cos_phi,
                    one * dt,
                    zero], axis=-1),
                torch.stack([
                    0.5*dt*dt*cos_phi,
                    0.5*dt*dt*sin_phi,
                    zero,
                    one * dt], axis=-1),        

            ], axis=-1)   # ...., 4 (states), 2 (controls)

            g_state = torch.where(~mask.unsqueeze(-1).unsqueeze(-1), g_state1, g_state2)
            g_control = torch.where(~mask.unsqueeze(-1).unsqueeze(-1), g_control1, g_control2)

            return next_states, g_state, g_control
        else:
            return next_states        

    def forward(self, x, u):
        squeeze = x.ndimension() == 1
        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert u.ndimension() == 2

        # TODO clamp control
        # u = torch.clamp(u, -self.max_torque, self.max_torque)[:,0]

        state = self.dyn_fn(x, u)

        if squeeze:
            state = state.squeeze(0)

        return state

    def linearized(self, x, u, diff):
        # unroll trajectory through time
        grad_x_list = []
        grad_u_list = []
        T = u.shape[0]
        x_unroll = [x[0]]
        for t in range(T-1):
            new_x, grad_x, grad_u = self.dyn_fn(x_unroll[-1], u[t], return_grad=True)
            grad_x_list.append(grad_x)
            grad_u_list.append(grad_u)
            x_unroll.append(new_x)
        grad_x = torch.stack(grad_x_list, dim=0)
        grad_u = torch.stack(grad_u_list, dim=0)
        x_unroll = torch.stack(x_unroll, dim=0)

        F = torch.cat((grad_x, grad_u), dim=-1)
        f = x_unroll[1:] - torch.matmul(grad_x, x_unroll[:-1].unsqueeze(-1)).squeeze(-1) - torch.matmul(grad_u, u[:-1].unsqueeze(-1)).squeeze(-1)

        if not diff:
            F = F.detach()
            f = f.detach()

        return F, f

    def linearized_autodiff(self, x, u, diff):
        # This is a duplacate of mpc.lineare_dynamics

        dynamics = self.dyn_fn
        n_batch = x[0].size(0)
        T = u.shape[0]

        with torch.enable_grad():
            # TODO: This is inefficient and confusing.
            x_init = x[0]
            x = [x_init]
            F, f = [], []
            for t in range(T):
                if t < T-1:
                    xt = torch.autograd.Variable(x[t], requires_grad=True)
                    ut = torch.autograd.Variable(u[t], requires_grad=True)
                    # xut = torch.cat((xt, ut), 1)
                    new_x = dynamics(xt, ut)

                    # Linear dynamics approximation.
                    Rt, St = [], []
                    for j in range(new_x.shape[1]):  # n_state
                        Rj, Sj = torch.autograd.grad(
                            new_x[:,j].sum(), [xt, ut],
                            retain_graph=True)
                        if not diff:
                            Rj, Sj = Rj.data, Sj.data
                        Rt.append(Rj)
                        St.append(Sj)
                    Rt = torch.stack(Rt, dim=1)
                    St = torch.stack(St, dim=1)

                    Ft = torch.cat((Rt, St), 2)
                    F.append(Ft)

                    if not diff:
                        xt, ut, new_x = xt.data, ut.data, new_x.data
                    ft = new_x - Rt.bmm(xt.unsqueeze(2)).squeeze(2) - St.bmm(ut.unsqueeze(2)).squeeze(2)
                    f.append(ft)

                if t < T-1:
                    x.append(new_x if not new_x.requires_grad else new_x.detach())

            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            if not diff:
                F, f = list(map(torch.autograd.Variable, [F, f]))
            return F, f



def extended_unicycle_dyn_fn(x: Union[torch.Tensor, np.ndarray],
                                u: Union[torch.Tensor, np.ndarray], 
                                dt: float, 
                                ret_np: bool,
                                ego_pred_type: str = 'motion_plan'):
    x_p = torch.as_tensor(x[..., 0])
    y_p = torch.as_tensor(x[..., 1])
    phi = torch.as_tensor(x[..., 2])
    v = torch.as_tensor(x[..., 3])
    dphi = torch.as_tensor(u[..., 0])
    a = torch.as_tensor(u[..., 1])

    if ego_pred_type == 'const_vel':
        return torch.stack([x_p + v * torch.cos(phi) * dt,
                            y_p + v * torch.sin(phi) * dt,
                            phi * torch.ones_like(a),
                            v], dim=-1)

    mask = torch.abs(dphi) <= 1e-2
    dphi = ~mask * dphi + mask * 1

    phi_p_omega_dt = angle_wrap(phi + dphi * dt)
    dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
    dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

    d1 = torch.stack([(x_p
                        + (a / dphi) * dcos_domega
                        + v * dsin_domega
                        + (a / dphi) * torch.sin(phi_p_omega_dt) * dt),
                        (y_p
                        - v * dcos_domega
                        + (a / dphi) * dsin_domega
                        - (a / dphi) * torch.cos(phi_p_omega_dt) * dt),
                        phi_p_omega_dt,
                        v + a * dt], dim=-1)
    d2 = torch.stack([x_p + v * torch.cos(phi) * dt + (a / 2) * torch.cos(phi) * dt ** 2,
                        y_p + v * torch.sin(phi) * dt + (a / 2) * torch.sin(phi) * dt ** 2,
                        phi * torch.ones_like(a),
                        v + a * dt], dim=-1)

    next_states = torch.where(~mask.unsqueeze(-1), d1, d2)
    if ret_np:
        return next_states.numpy()
    else:
        return next_states
