import torch
import numpy as np
import math, copy, time
import abc
from copy import deepcopy


class DynType:
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """

    UNICYCLE = 1
    SI = 2
    DI = 3
    BICYCLE = 4
    DDI = 5


class Dynamics(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dt, name, **kwargs):
        self.dt = dt
        self._name = name
        self.xdim = 4
        self.udim = 2

    @abc.abstractmethod
    def __call__(self, x, u):
        return

    @abc.abstractmethod
    def step(self, x, u, dt, bound=True):
        return

    def name(self):
        return self._name

    @abc.abstractmethod
    def type(self):
        return

    @abc.abstractmethod
    def ubound(self, x):
        return

    @staticmethod
    def state2pos(x):
        return

    @staticmethod
    def state2yaw(x):
        return

    @staticmethod
    def get_state(pos,yaw,dt,mask):
        return
    
    def get_input_violation(self,x,u):
        lb, ub = self.ubound(x)
        return torch.maximum((lb-u).clip(min=0.0), (u-ub).clip(min=0.0))
    
    
    def forward_dynamics(self,x0: torch.Tensor,u: torch.Tensor, include_step0: bool = False):
        """
        Integrate the state forward with initial state x0, action u
        Args:
            initial_states (Torch.tensor): state tensor of size [B, (A), 4]
            u (Torch.tensor): action tensor of size [B, (A), T, 2]
            include_step0 (bool): the output trajectory will include the current state if true.
        Returns:
            state tensor of size [B, (A), T, 4]
        """
        num_steps = u.shape[-2]
        x = [x0] + [None] * num_steps
        for t in range(num_steps):
            x[t + 1] = self.step(x[t], u[..., t, :])

        if include_step0:
            x = torch.stack(x, dim=-2)
        else:
            x = torch.stack(x[1:], dim=-2)
        return x



