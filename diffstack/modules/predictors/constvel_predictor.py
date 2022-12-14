import torch
import numpy as np
from typing import Dict, Optional, Union, Any

from trajdata.data_structures.batch import AgentBatch

from mpc import util as mpc_util

from diffstack.modules.module import Module, DataFormat, RunMode
from diffstack.modules.dynamics_functions import ExtendedUnicycleDynamics
from diffstack.modules.predictors.trajectron_utils.model.components import GMM2D
from diffstack.utils.pred_utils import compute_prediction_metrics

class ConstVelPredictor(Module):

    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["agent_batch", "loss_weights"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["pred_dist", "pred_ml", "metrics:train", "metrics:validate"])

    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super().__init__(model_registrar, hyperparams, log_writer, device)

        self.dyn_obj = ExtendedUnicycleDynamics(dt=self.hyperparams['dt'])

    def train(self, inputs: Dict) -> Dict:
        outputs = self.infer(inputs)
        outputs["loss"] = torch.zeros((), device=self.device)
        return outputs

    def validate(self, inputs: Dict) -> Dict:
        batch: AgentBatch = inputs["agent_batch"]
        outputs = self.infer(inputs)
        outputs["metrics"] = self.validation_metrics(outputs['pred_dist'], outputs['pred_ml'], batch.agent_fut)
        return outputs        

    def infer(self, inputs: Dict):
        batch: AgentBatch = inputs["agent_batch"]
        pred_ml, pred_dist = self.constvel_predictions(batch.agent_hist[:, -1], batch.agent_fut.shape[1])
        return {"pred_dist": pred_dist, "pred_ml": pred_ml}

    def constvel_predictions(self, curr_agent_state: torch.Tensor, prediction_horizon: int):
        # {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y'], 'heading': ['°', 'd°'], 'augment': ['ego_indicator']}
        pos_xy, vel_xy, acc_xy, yaw, dyaw = torch.split_with_sizes(curr_agent_state[..., :8], (2, 2, 2, 1, 1), dim=-1)
        vel = torch.linalg.norm(vel_xy, dim=-1).unsqueeze(-1)

            # state: x, y, yaw, vel; control: yaw_rate, acc
        x0 = torch.cat((pos_xy, yaw, vel), dim=-1)  # (b, 4)
        u_zeros = torch.zeros((prediction_horizon+1, x0.shape[0], 2), dtype=x0.dtype, device=x0.device)
            
        x_constvel = mpc_util.get_traj(prediction_horizon+1, u_zeros, x0, self.dyn_obj)  # (T+1, b, 4)
        pred_constvel = x_constvel[1:, :, :2]  # (T, b, 2)
        
        # # Simple alternative
        # x, y, heading, v = torch.unbind(x0, dim=-1)
        # vx = v * 0.5 * torch.cos(heading)
        # vy = v * 0.5 * torch.sin(heading)
        # x_const = x.unsqueeze(0) + vx.unsqueeze(0) * torch.arange(prediction_horizon1).float().to(self.device).unsqueeze(1)
        # y_const = y.unsqueeze(0) + vy.unsqueeze(0) * torch.arange(prediction_horizon+1).float().to(self.device).unsqueeze(1)
        # # heading_const = torch.repeat_interleave(heading.unsqueeze(0), prediction_horizon+1, dim=0)
        # # v_const = torch.repeat_interleave(v.unsqueeze(0), prediction_horizon+1, dim=0)
        # # traj_const = torch.stack([x_const, y_const, heading_const, v_const], dim=-1)
        # traj_const = torch.stack([x_const, y_const], dim=-1)
        # traj_const = traj_const[1:]
        # assert torch.isclose(pred_constvel, traj_const).all()
        
        pred_constvel = pred_constvel.transpose(0, 1).unsqueeze(0)  # (1, b, T, 2)

        mus = pred_constvel.unsqueeze(3)  # (1, b, T, 1, 2)
        log_pis = torch.zeros((1, x0.shape[0], prediction_horizon, 1), dtype=x0.dtype, device=x0.device)
            # log_sigmas = torch.log(torch.tensor([0.0393,  0.4288,  1.6322,  4.1350,  8.4635, 15.1796]), dtype=x0.dtype, device=x0.device)
        log_sigmas = torch.log(torch.tensor((self.hyperparams['dt']*np.arange(7))[1:]**2*2, dtype=x0.dtype, device=x0.device))
        log_sigmas = log_sigmas.reshape(1, 1, prediction_horizon, 1, 1).repeat((1, x0.shape[0], 1, 1, 2))
        corrs = 0. * torch.ones((1, x0.shape[0], prediction_horizon, 1), dtype=x0.dtype, device=x0.device)  # TODO not sure what is reasonable
            
        y_dists = GMM2D(log_pis, mus, log_sigmas, corrs)
        return pred_constvel, y_dists

    def validation_metrics(self, pred_dist, pred_ml, agent_fut):
        # Compute default metrics
        metrics = compute_prediction_metrics(pred_ml, agent_fut[..., :2], y_dists=pred_dist)
        return metrics
