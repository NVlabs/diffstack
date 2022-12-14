
import torch
import numpy as np

from diffstack.modules.cost_functions.cost_functions import LinearCost1, LinearCostAngleBug, InterpretableLinearCost1


def get_cost_object(plan_cost_mode, control_limits, is_trainable, device):

    if plan_cost_mode == 'default':
        cost_class = LinearCost1
        # With the distance-based prediction term, num_trajs 128, motion_plan ego preds, L-BFGS lr=0.95
        cost_kwargs = dict(
            theta=torch.from_numpy(np.array([1.721, 0.652, 0., 8.755, 1.362, 1.440])).float().to(device))
    elif plan_cost_mode == 'interpretable':
        # Mpc1 cost
        cost_class = InterpretableLinearCost1
        cost_kwargs = dict(
            theta=torch.from_numpy(np.array([1.0] * 8)).float().to(device),
            )
    elif plan_cost_mode == 'corl_default':   # default for corl paper
        cost_class = LinearCostAngleBug
        cost_kwargs = dict(
            theta=torch.from_numpy(np.array([0.3, 0.3, 0.5, 1.0, 20.0])).float().to(device),
            control_limits=control_limits,
            )
    elif plan_cost_mode == 'corl_default_angle_fix':   # fixed angle wrap
        cost_class = LinearCost1
        cost_kwargs = dict(
            theta=torch.from_numpy(np.array([0.3, 0.3, 0.5, 1.0, 20.0])).float().to(device),
            control_limits=control_limits,
            )
    else:
        raise NotImplementedError(plan_cost_mode)

    return cost_class(**cost_kwargs, is_trainable=is_trainable)