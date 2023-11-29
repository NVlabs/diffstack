import torch
import numpy as np
from diffstack.dynamics.unicycle import Unicycle
from diffstack.modules.module import Module, DataFormat
from diffstack.utils.batch_utils import batch_utils
from trajdata.data_structures.batch import AgentBatch,SceneBatch
from diffstack.modules.predictors.trajectron_utils.model.components import GMM2D
import diffstack.utils.tensor_utils as TensorUtils
from typing import Dict, Optional, Union, Any


class KinematicTreeModel(Module):

    @property
    def input_format(self) -> DataFormat:
        if self.scene_centric:
            return DataFormat(["scene_batch"])
        else:
            return DataFormat(["agent_batch"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["pred_dist",  "pred_ml", "pred_single", "metrics"])

    def __init__(self,model_registrar, config, log_writer, device, input_mappings={}):
        super().__init__(model_registrar, config, log_writer, device, input_mappings)
        self.config = config
        self.step_time = config.step_time
        if "dynamics" in config:
            if config.dynamics.type=="Unicycle":
                self.dyn = Unicycle(self.step_time,max_steer=config.dynamics.max_steer,max_yawvel=config.dynamics.max_yawvel,acce_bound=config.dynamics.acce_bound)
            else:
                raise NotImplementedError
        else:
            self.dyn=Unicycle(self.step_time)
        
        self.stage=config["stage"]
        self.num_frames_per_stage = config["num_frames_per_stage"]
        self.M = config["M"]
        self.scene_centric = config["scene_centric"]
        self.only_branch_closest = config.only_branch_closest if "only_branch_closest" in config else self.scene_centric
        self.bu = batch_utils(rasterize_mode="none",parse=True)

    def train_step(self, inputs: Dict) -> Dict:
        outputs = self.infer_step(inputs)
        outputs["loss"] = torch.zeros((), device=self.device)
        return outputs

    def validate_step(self, inputs: Dict) -> Dict:
        outputs = self.infer_step(inputs)
        outputs["metrics"] = {}
        return outputs        

    def infer_step(self, inputs: Dict):
        if self.scene_centric:
            batch = inputs["scene_batch"]
        else:
            batch = inputs["agent_batch"]
        if isinstance(batch, dict) and "batch" in batch:
            batch = batch ["batch"]
        batch = self.bu.parse_batch(batch)
        pred_ml, pred_dist = self.kinematic_predictions(batch)

        # Dummy single agent prediction. 
        agent_fut = batch["agent_fut"]
        pred_single = torch.full([agent_fut.shape[0], 0, agent_fut.shape[2], 4], dtype=agent_fut.dtype, device=agent_fut.device, fill_value=torch.nan)

        output = dict(pred_dist=pred_dist, pred_ml=pred_ml, pred_single=pred_single)
        if self.scene_centric:
            output["scene_batch"]=batch
        else:
            output["agent_batch"]=batch
        return output
        
    def kinematic_predictions(self,batch):
        if batch["agent_hist"].shape[-1] == 7:  # x, y, vx, vy, ax, ay, heading
            yaw = batch["agent_hist"][...,6:7]
        else:
            assert batch["agent_hist"].shape[-1] == 8 # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
            yaw = torch.atan2(batch["agent_hist"][..., [-2]], batch["agent_hist"][..., [-1]])
        pos = batch["agent_hist"][...,:2]
        speed = torch.norm(batch["agent_hist"][..., 2:4], dim=-1,keepdim=True)
        curr_states = torch.cat([pos, yaw, speed], -1)[...,-1,:]

        bs = curr_states.shape[0]
        
        device = curr_states.device
        assert self.M<=4
        if self.scene_centric:
            Na = batch["agent_hist"].shape[1]
            inputs = torch.zeros([bs,self.M,Na,2]).to(device)
            if self.only_branch_closest:
                if batch["agent_hist"].shape[1] == 1:
                    branch_idx = 0
                else:
                    dis = torch.norm(batch["agent_hist"][0,1:,-1,:2]-batch["agent_hist"][0,:1,-1,:2],dim=-1)
                    branch_idx = dis.argmin()+1
            else:
                branch_idx = 1 if Na>1 else 0
            if self.M>1:
                inputs[:,1,branch_idx,1]=-0.02*curr_states[:,branch_idx,2]
            if self.M>2:
                inputs[:,2,branch_idx,1]=0.02*curr_states[:,branch_idx,2]
            if self.M>3:
                inputs[:,3,branch_idx,0]=torch.clip(-curr_states[:,branch_idx,2]*0.5,min=-3)
            
        else:
            inputs = torch.zeros([bs,self.M,2]).to(device)
            if self.M>1:
                inputs[:,1,1]=-0.02*curr_states[:,2]
            if self.M>2:
                inputs[:,2,1]=0.02*curr_states[:,2]
            if self.M>3:
                inputs[:,3,0]=torch.clip(-curr_states[:,2]*0.5,min=-3)
            assert self.M<=4
        
        
        inputs = inputs.unsqueeze(-2).repeat_interleave(self.num_frames_per_stage,-2)

        T = self.num_frames_per_stage*self.stage
        pred_by_stage = list()
        for stage in range(self.stage):
            inputs_i = inputs.repeat_interleave(self.M**stage,0)
            if stage==0:
                curr_state_i = curr_states
            else:
                curr_state_i = pred_by_stage[stage-1][...,-1,:]
            pred_i = self.dyn.forward_dynamics(curr_state_i.repeat_interleave(self.M,0),TensorUtils.join_dimensions(inputs_i,0,2))
            pred_by_stage.append(pred_i)
        
        for stage in range(self.stage):
            if self.scene_centric:
                pred_by_stage[stage]=pred_by_stage[stage].reshape([bs,-1,Na,self.num_frames_per_stage,4]).repeat_interleave(self.M**(self.stage-1-stage),1)
            else:
                pred_by_stage[stage]=pred_by_stage[stage].reshape([bs,-1,self.num_frames_per_stage,4]).repeat_interleave(self.M**(self.stage-1-stage),1)
        pred_traj = torch.cat(pred_by_stage,-2)  # b x N x T x D
        if self.scene_centric:
            pred_traj = pred_traj.reshape(bs,self.M**self.stage,Na,T,-1)
        prob = torch.ones([bs,self.M**self.stage]).to(device) # b x N
        prob = prob/prob.sum(-1,keepdim=True)
        if self.scene_centric:
            mus = pred_traj[...,[0,1,2,3]].permute(2,0,3,1,4)  # (Na, b, T, M**stage, 4)
            Tf = pred_traj.shape[-2]
            log_pis = torch.log(prob)[None,:,None,:].repeat_interleave(Tf,-2).repeat_interleave(Na,0)
            log_sigmas = torch.log(torch.tensor((self.config['dt']*np.arange(Tf+1))[1:]**2*2, dtype=curr_states.dtype, device=curr_states.device))
            log_sigmas = log_sigmas.reshape(1, 1, Tf, 1, 1).repeat((Na, bs, 1, self.M**self.stage, 2))
            corrs = 0. * torch.ones((1, bs, Tf, self.M**self.stage), dtype=curr_states.dtype, device=curr_states.device)  # TODO not sure what is reasonable
            y_dists = GMM2D(log_pis, mus, log_sigmas, corrs)
        else:
            mus = pred_traj[...,[0,1,2,3]].transpose(1,2).unsqueeze(0)  # (1, b, T, M**stage, 4)
            Tf = pred_traj.shape[-2]
            log_pis = torch.log(prob)[None,:,None,:].repeat_interleave(Tf,-2)
            log_sigmas = torch.log(torch.tensor((self.config['dt']*np.arange(Tf+1))[1:]**2*2, dtype=curr_states.dtype, device=curr_states.device))
            log_sigmas = log_sigmas.reshape(1, 1, Tf, 1, 1).repeat((1, bs, 1, self.M**self.stage, 2))
            corrs = 0. * torch.ones((1, bs, Tf, self.M**self.stage), dtype=curr_states.dtype, device=curr_states.device)  # TODO not sure what is reasonable
                
            y_dists = GMM2D(log_pis, mus, log_sigmas, corrs)
        return pred_traj, y_dists

    def validation_metrics(self, pred_dist, pred_ml, agent_fut):
        # Compute default metrics
        metrics = compute_prediction_metrics(pred_ml, agent_fut[..., :2], y_dists=pred_dist)
        return metrics