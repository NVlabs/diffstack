import torch
import numpy as np
from typing import Dict, Optional, Union, Any

from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.agent import AgentType

from diffstack.modules.cost_functions.cost_selector import get_cost_object
from diffstack.modules.cost_functions.linear_base_cost import LinearBaseCost
from diffstack.modules.dynamics_functions import ExtendedUnicycleDynamics
from diffstack.modules.planners.fan_planner import FanPlanner
from diffstack.modules.planners.mpc_utils.trajcost_mpc import TrajCostMPC, GradMethods

from diffstack.utils.utils import  CudaTimer, subsample_traj, convert_state_pred2plan, all_gather, restore, batchable_dict, batchable_nonuniform_tensor
from diffstack.modules.module import Module, DataFormat, RunMode


MAX_PLAN_NEIGHBORS = 16

timer = CudaTimer(enabled=False)
mse_fn = torch.nn.MSELoss(reduction='none')


class FanMpcPlanner(Module):
    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["agent_batch", "pred_dist"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["plan_xu", "valid", "converged", "loss:train", "loss:validate", "loss_batch:train", "loss:validate", "metrics:train", "metrics:validate"])

    def __init__(self, model_registrar, hyperparams, log_writer, device, input_mappings = {}):
        super().__init__(model_registrar, hyperparams, log_writer, device, input_mappings)

        self.ph = int(hyperparams['prediction_sec'] / hyperparams['dt'])
        self.planh =  self.ph * int(hyperparams['dt'] / hyperparams['plan_dt'])
        
        # Dynamics and cost
        self.dyn_obj = ExtendedUnicycleDynamics(dt=self.hyperparams['plan_dt'])
        # Load cost object from model registrat.
        # TODO we should check if the loaded model had the same cost function setting.
        self.cost_obj = self.model_registrar.get_model("planner_cost", self.get_plan_cost(hyperparams['plan_cost'], is_trainable=hyperparams["train_plan_cost"]))
        self.interpretable_cost_obj = self.model_registrar.get_model("interpretable_cost", self.get_plan_cost('interpretable', is_trainable=False))

        self.u_lower = torch.from_numpy(np.array([self.cost_obj.control_limits['min_heading_change'], self.cost_obj.control_limits['min_a']])).float().to(self.device)
        self.u_upper = torch.from_numpy(np.array([self.cost_obj.control_limits['max_heading_change'], self.cost_obj.control_limits['max_a']])).float().to(self.device)

        # TODO dummy normalization 
        self.normalize_std = torch.ones((2, ), dtype=torch.float, device=self.device)

        # Fan planner
        self.fan_obj = FanPlanner(ph=self.planh, dt=hyperparams['plan_dt'], device=device)


        # Controller
        if self.cost_obj.control_limits is not None:
            u_lower = self.u_lower
            u_upper = self.u_upper
        else:
            u_lower = None
            u_upper = None

        self.mpc_obj = TrajCostMPC(
            4, 2, self.planh+1,
            u_init=None,
            # TODO are there actual limits? SLSQP doesn't seem to constrain u, only through cost.
            # u_lower=None, u_upper=None,
            u_lower=u_lower, 
            u_upper=u_upper,
            lqr_iter=self.hyperparams['plan_lqr_max_iters'],  # 50 def for pendulum
            verbose=-1,  # no output, 0 warnings
            exit_unconverged=False,
            detach_unconverged=False,  # Manually detach instead 
            linesearch_decay=0.2,
            max_linesearch_iter=self.hyperparams['plan_lqr_max_linesearch_iters'],
            grad_method=GradMethods.AUTO_DIFF,
            eps=self.hyperparams['plan_lqr_eps'],
            n_batch=1,  # we will update this manually every time before calling MPC
        )

    def get_plan_cost(self, plan_cost_mode: str, is_trainable: bool) -> LinearBaseCost:
        control_limits = self.hyperparams["dynamic"]["VEHICLE"]["limits"]
        return get_cost_object(plan_cost_mode, control_limits, is_trainable=is_trainable, device=self.device)
        
    def train(self, inputs: Dict) -> Dict:
        batch: AgentBatch = inputs["agent_batch"]
        pred_dist = inputs["pred_dist"]

        # TODO currently we use future_mode to decide what prediction gets used. Instead we should
        # implement these as different predictors. 
        if self.hyperparams["predictor"] == "nopred":
            future_mode="nopred"
        elif self.hyperparams["predictor"] == "gt":
            future_mode="gt"
        elif self.hyperparams["predictor"] == "blind":
            future_mode="none"
        elif self.hyperparams["predictor"] == "tpp_nogt":
            future_mode="pred_nogt"
        else:
            future_mode="pred"
        init_mode=self.hyperparams["plan_init"]
        loss_mode=self.hyperparams["plan_loss"]

        # For now we only support one type of prediction agent in the batch.
        node_types = batch.agent_types()
        if len(node_types) > 1:
            raise NotImplementedError("Mixing agent types for prediction in a batch is not supported.")
        node_type = node_types[0]

        if node_type.name in self.hyperparams["plan_node_types"]:  
            plan_loss_batch, plan_converged, metrics, plan_info = self._plan_loss(batch, pred_dist, None, init_mode, future_mode, loss_mode, return_iters=False)
            plan_valid = plan_info['plan_batch_filter']
            plan_xu = plan_info['plan_xu']  # TODO this is a detached tensor
        else:
            plan_loss_batch = torch.zeros((0,), device=self.device)
            plan_valid = torch.zeros((0,), dtype=torch.bool, device=self.device)
            plan_converged = torch.zeros((0,), dtype=torch.bool, device=self.device)
            metrics = {}

        # loss as metric
        metrics["plan_loss"] = plan_loss_batch

        # Use simple mean as loss. It makes sense in that gradients in the batch will be averaged.
        plan_loss = plan_loss_batch.mean()
        outputs = {"plan_xu": plan_xu, "valid": plan_valid, "converged": plan_converged, "loss": plan_loss, "loss_batch": plan_loss_batch, "metrics": metrics}

        # TODO later the fan planner and mpc should be different subcomponents. For now just hack the output 
        # as if the fan planner produced its candidates.
        if "traj_xu" in plan_info:
            outputs["fan.candidates_xu"] = plan_info["traj_xu"]

        return outputs

    def validate(self, inputs: Dict) -> Dict:
        return self.train(inputs)

    def infer(self, inputs: Dict) -> Dict:
        return self.train(inputs)

    def _plan_loss(self, batch: AgentBatch, y_dist, pred_extra=None, init_mode="fitted", future_mode="pred", loss_mode="mse", return_iters=False):
        timer.start("prepare")
        plan_batch_filter, plan_ego_batch, lanes, goal_batch, plan_mus_batch, plan_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, fan_inputs = self.prepare_plan_instance(
            batch, y_dist, build_fan_inputs=(self.hyperparams["planner"] in ["fan", "fan_mpc"]))
        lane_points = None            
        x_gt = plan_ego_batch[1:]  
        timer.end("prepare")

        if plan_ego_batch.shape[1] == 0:
            # If we cannot plan for anything in batch, create a dummy zero loss
            plan_loss_batch = torch.zeros((1,), device=self.device)
            plan_converged = torch.zeros((1, ), dtype=torch.bool, device=self.device)
            plan_info = {}
            metrics = {}
            print ("Could not plan for anything in batch.")

        else:
            # Plan
            timer.start("plan")
            plan_x, plan_u, plan_cost, plan_converged, plan_info = self.plan_batch(
                plan_ego_batch[0], lanes, goal_batch, plan_mus_batch, plan_logp_batch, 
                empty_mus_batch, empty_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, 
                future_mode=future_mode, init_mode=init_mode, planner=self.hyperparams["planner"], 
                pred_extra=pred_extra,
                fan_inputs=fan_inputs,
                plan_batch_filter=plan_batch_filter,                
                return_iters=return_iters)  # (T, b, ...) 

            plan_sub_x = subsample_traj(plan_x, self.ph, self.planh)
            # plan_sub_u = subsample_traj(plan_u, self.ph, self.planh)
            timer.end("plan")

            timer.start("metrics")
            # Plan error metrics
            plan_mse_batch = mse_fn(plan_sub_x[1:, :, :2], x_gt[..., :2]).sum(dim=-1)  # (T, b)
            plan_unbiased_d1_batch = torch.sqrt(plan_mse_batch).mean(dim=0)
            plan_unbiased_d2_batch = plan_mse_batch.mean(dim=0)  # (b, )
            # For fan planner mse is only meaningful for 'converged' cases.
            # We will replace with zero for unconverged samples, so batch dimension is preserved, but this does introduce bias.
            if self.hyperparams["planner"] == "fan":
                plan_unbiased_d1_batch[torch.logical_not(plan_converged)] = 0.
                plan_unbiased_d2_batch[torch.logical_not(plan_converged)] = 0.

            # Cost(plan, gtfutures)
            plan_xu = torch.cat((plan_x, plan_u), -1)
            plan_hcost_batch = self.cost_obj(plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points))  # (T, b)
            plan_info['plan_xu'] = plan_xu.detach()

            # Trajectory fan related metrics
            if self.hyperparams['planner'] in ['fan', 'fan_mpc']:
                # Pad and stack trajectory candidates 
                traj_xu = torch.nn.utils.rnn.pad_sequence(plan_info['traj_xu'], batch_first=True, padding_value=torch.inf)  # b, N, T+1, 6
                traj_xu_sub = subsample_traj(traj_xu.transpose(2, 0), self.ph, self.planh)  # T, N, b, 6
                traj_cost = torch.nn.utils.rnn.pad_sequence(plan_info['traj_cost'], batch_first=True, padding_value=torch.inf)  # b, N

                ##########
                # Label: closest based on last state distance
                trajdist2 = torch.square(traj_xu_sub[1:, :, :, :2] - x_gt[:, :, :2].unsqueeze(1)).sum(-1)  # T, N, b
                trajmse = torch.mean(trajdist2, dim=0)
                label_mse = torch.argmin(trajmse, dim=0)  # b,
                plan_info["label_mse"] = label_mse.detach()

                # goaldist2 = torch.square(traj_xu[:, :, -1, :2] - x_gt[-1, :, :2].unsqueeze(1)).sum(-1)  # b, N
                # goaldist2 = trajdist2[-1]  # b, N
                # label_goaldist = torch.argmin(goaldist2, dim=1)  # b,
                # plan_info["label_goaldist"] = label_goaldist.detach()

                goaldist2 = trajdist2[-1]  # N, b
                label_goaldist = torch.argmin(goaldist2, dim=0)  # b,
                plan_info["label_goaldist"] = label_goaldist.detach()

                # label_xy = traj_xu[torch.arange(traj_xu.shape[0]), label_goaldist]  # b, T, 2
                # label_plan_mse = mse_fn(label_xy.transpose(1, 0)[1:, :, :2], x_gt[..., :2]).sum(dim=-1)  # (T, b)

                # Cross entropy, based on lowest mse
                class_mse_loss = torch.nn.functional.cross_entropy(-traj_cost, label_mse, reduction='none')
                class_mse_loss = torch.nan_to_num(class_mse_loss, nan=0.)

                # Cross entropy, based on closest at goal label
                class_goaldist_loss = torch.nn.functional.cross_entropy(-traj_cost, label_goaldist, reduction='none')
                class_goaldist_loss = torch.nan_to_num(class_goaldist_loss, nan=0.)

                ##########
                timer.start("hcost")
                # TODO this is now slow (50% of compute spent here)
                # it would be easy to speed up by skipping padding and batch loop, and instead concatenate all while remembering lengths, and recover with split
                # Get hindsight costs for all traj candidates
                traj_hcost = self.fan_obj.get_cost_for_trajs(
                    plan_info['traj_xu'], self.cost_obj, 
                    cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points))  # (b)(N)
                plan_info["traj_hcost"] = [x.detach() for x in traj_hcost]
                traj_hcost = torch.nn.utils.rnn.pad_sequence(traj_hcost, batch_first=True, padding_value=torch.inf)  # b, N 
                timer.end("hcost")
                # print (len(plan_info['traj_xu']), plan_info['traj_xu'][0].device, plan_all_gt_neighbors_batch.device, goal_batch.device, lane_points.device)
                
                # Label: based on lowest hindsight cost
                label_hcost = torch.argmin(traj_hcost, dim=1)  # b,
                plan_info["label_hcost"] = label_hcost.detach()

                # Hcost of fan's choice 
                fan_choice = torch.argmin(traj_cost, dim=1)  # b,
                fan_choice_hcost = traj_hcost[torch.arange(traj_hcost.shape[0]).to(self.device), fan_choice]

                # Cross entropy, based on hindsight cost label
                class_hcost_loss = torch.nn.functional.cross_entropy(-traj_cost, label_hcost, reduction='none')
                class_hcost_loss = torch.nan_to_num(class_hcost_loss, nan=0.)

                target_probs = torch.nn.functional.softmax(-traj_cost, dim=-1) 
                logits = -traj_cost.nan_to_num(posinf=1e10)
                dist_hcost_loss = torch.nn.functional.cross_entropy(logits, target_probs, reduction='none')
                dist_hcost_loss = torch.nan_to_num(dist_hcost_loss, nan=0.)

                # Maxmargin
                if traj_cost.shape[1] < 2: 
                    # Need at least two candidates
                    maxmargin_goaldist = torch.full((traj_xu.shape[0], ), torch.inf, dtype=torch.float, device=self.device)
                else:
                    # Lowest cost other than the closest
                    sorted_idx = torch.argsort(traj_cost, dim=1)
                    top1 = sorted_idx[:, 0]
                    top2 = sorted_idx[:, 1]
                    lowest_other = torch.where(top1 == label_goaldist, top2, top1) 
                    lowest_other_cost = torch.gather(traj_cost, dim=1, index=lowest_other.unsqueeze(-1)).squeeze(-1)

                    closest_cost = torch.gather(traj_cost, dim=1, index=label_goaldist.unsqueeze(-1)).squeeze(-1)

                    # Maxmargin loss
                    maxmargin_goaldist = closest_cost - lowest_other_cost
                    maxmargin_goaldist = torch.nan_to_num(maxmargin_goaldist, nan=0., posinf=0., neginf=0.)

            else:
                class_goaldist_loss = torch.zeros_like(plan_unbiased_d1_batch)
                maxmargin_goaldist = torch.zeros_like(plan_unbiased_d1_batch)

            # # Debug converged rate
            # print (plan_converged.float().mean(), plan_converged)
            # print (plan_x[:, 0])

            # Resolve multi-component loss modes
            if loss_mode in ["joint_hcost", "joint-hcost"]:
                loss_modes = ["class_hcost", "hcost"]  # order is important for scaler2
            elif loss_mode == "joint_hcost2":
                loss_modes = ["class_hcost2", "hcost"]  # order is important for scaler2
            elif loss_mode in ["joint_mse", "joint-mse"]:
                loss_modes = ["class_mse", "mse"]  # order is important for scaler2
            elif loss_mode == "joint_mse2":
                loss_modes = ["class_mse2", "mse"]  # order is important for scaler2
            else:
                loss_modes = [loss_mode]

            # Loss function 
            plan_loss_components = []
            for loss_mode_i in loss_modes:

                #### Trajfan planner losses
                if loss_mode_i == "class_goaldist":
                    plan_loss_batch = class_goaldist_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        plan_converged = torch.zeros_like(plan_converged)
                    else:
                        plan_converged = plan_converged & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * plan_converged.float() + plan_loss_batch.detach() * (1.-plan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "class_mse":
                    plan_loss_batch = class_mse_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        plan_converged = torch.zeros_like(plan_converged)
                    else:
                        plan_converged = plan_converged & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * plan_converged.float() + plan_loss_batch.detach() * (1.-plan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "class_mse2":
                    plan_loss_batch = class_mse_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        fan_converged = torch.zeros_like(plan_info["fan_converged"])
                    else:
                        fan_converged = plan_info["fan_converged"] & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * fan_converged.float() + plan_loss_batch.detach() * (1.-fan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "class_hcost":
                    plan_loss_batch = class_hcost_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        plan_converged = torch.zeros_like(plan_converged)
                    else:
                        plan_converged = plan_converged & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * plan_converged.float() + plan_loss_batch.detach() * (1.-plan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "class_hcost2":
                    plan_loss_batch = class_hcost_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        fan_converged = torch.zeros_like(plan_info["fan_converged"])
                    else:
                        fan_converged = plan_info["fan_converged"] & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * fan_converged.float() + plan_loss_batch.detach() * (1.-fan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "dist_hcost":
                    plan_loss_batch = dist_hcost_loss
                    # Mask valid: the first two candidate costs should be finite
                    if traj_cost.shape[1] < 2:
                        plan_converged = torch.zeros_like(plan_converged)
                    else:
                        plan_converged = plan_converged & traj_cost[:, 1].isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * plan_converged.float() + plan_loss_batch.detach() * (1.-plan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "maxmargin_goaldist":
                    plan_loss_batch = maxmargin_goaldist
                    # Mask valids
                    plan_converged = plan_converged & lowest_other_cost.isfinite()
                    # Not sure this is needed, could be already taken care of by nan replacement
                    plan_loss_batch = plan_loss_batch * plan_converged.float() + plan_loss_batch.detach() * (1.-plan_converged.float())
                    plan_loss_components.append(plan_loss_batch)

                #### MPC planner losses
                elif loss_mode_i == "mse":
                    plan_loss_batch = plan_mse_batch
                    plan_loss_batch = plan_loss_batch.mean(dim=0)  # reduce over time keep batch (b,)
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "mse-bias":
                    gtplan_mse_batch = mse_fn(gtplan_x[1:, :, :2], x_gt[..., :2]).sum(dim=-1)
                    plan_loss_batch = (plan_mse_batch - gtplan_mse_batch)
                    plan_loss_batch = plan_loss_batch.mean(dim=0)  # reduce over time keep batch (b,)
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "hcost":
                    assert not self.hyperparams['train_plan_cost']  # loss function cannot depend on other learned parameter
                    plan_loss_batch = plan_hcost_batch
                    plan_loss_batch = plan_loss_batch.mean(dim=0)  # reduce over time keep batch (b,)
                    plan_loss_components.append(plan_loss_batch)

                elif loss_mode_i == "hcost-bias":
                    assert not self.hyperparams['train_plan_cost']  # loss function cannot depend on other learned parameter
                    # # Rerun planner with gt future instead of using cached result
                    # gtplan_x, gtplan_u, _, _, _ = self.plan_batch(
                    #     plan_ego_time_batch, plan_mus_batch, plan_logp_batch, 
                    #     empty_mus_batch, empty_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, 
                    #     future_mode="gt", init_mode=init_mode, planner=self.hyperparams["planner"], plan_gt_u=gtplan_u, 
                    #     plan_data=plan_data, return_iters=False)  # (T, b, ...) 
                    
                    gtplan_xu = torch.cat((gtplan_x, gtplan_u), -1)
                    gtplan_hcost_batch = self.cost_obj(gtplan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points))
                    # plan_loss_batch = plan_hcost_batch - gtplan_hcost_batch
                    plan_loss_batch = subsample_traj(plan_hcost_batch, self.ph, self.planh) - gtplan_hcost_batch
                    plan_loss_batch = plan_loss_batch.mean(dim=0)  # reduce over time keep batch (b,)
                    plan_loss_components.append(plan_loss_batch)

                else:
                    raise ValueError("Unknown plan_loss / loss_mode arg: %s"%loss_mode_i)

            # Combine loss components
            if len(plan_loss_components) == 1:
                plan_loss_batch = plan_loss_components[0]
            elif len(plan_loss_components) == 2:
                # Second scaler will be used for mpc loss
                plan_loss_batch = plan_loss_components[0] + self.hyperparams['plan_loss_scaler2'] * plan_loss_components[1]
            else:
                raise NotImplementedError("More than 2 plan loss components")

            # Return metrics and planning internals
            plan_info['plan_batch_filter'] = plan_batch_filter.detach()
            plan_info['converged'] = plan_converged.detach()  
            plan_info['fan_converged'] = plan_info["fan_converged"].detach()  # will be set to all True if not fan planner

            if return_iters:
                plan_hcost_components = self.cost_obj(
                    plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points),
                    keep_components=True)  # (T, b, c)
                plan_icost_components = self.interpretable_cost_obj(
                    plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points),
                    keep_components=True)  # (T, b, c)

                plan_info['x_gt'] = x_gt.detach()
                # FIXN
                plan_info['all_gt_neighbors'] = plan_all_gt_neighbors_batch.detach()
                plan_info['gt_neighbors'] = plan_gt_neighbors_batch.detach()
                # detach_list = lambda l: [t.detach() for t in l]
                # plan_iters['all_gt_neighbors'] = detach_list(plan_all_gt_neighbors_batch)
                # plan_iters['gt_neighbors'] = detach_list(plan_gt_neighbors_batch)
                plan_info['lanes'] = lanes.detach()
                plan_info['lane_points'] = lane_points.detach() if lane_points is not None else None
                plan_info['plan_loss'] = plan_loss_batch.detach()
                plan_info['hcost_components'] = plan_hcost_components.detach()
                plan_info['icost_components'] = plan_icost_components.detach()
                plan_info['ego_pred_gt_dist'] = torch.square(torch.square(batch.agent_fut[plan_batch_filter, :, :2].transpose(1, 0) - x_gt[:, :, :2]).sum(dim=-1).min(dim=0).values)

                plan_and_fan_valid = plan_batch_filter.detach().clone()
                plan_and_fan_valid[plan_batch_filter] = torch.logical_and(plan_batch_filter[plan_batch_filter], plan_info["fan_converged"])
                plan_info['plan_and_fan_valid'] = plan_and_fan_valid.detach()

            metrics = {
                "unbiased_d1": plan_unbiased_d1_batch,  # (b, )
                "unbiased_d2": plan_unbiased_d2_batch,  # (b, )  # TODO this is the same as mse
                "cost": plan_cost,  # (b,)
                "mse": plan_mse_batch.mean(0),  # (b,)  
                "hcost": plan_hcost_batch.mean(0),  #(b,)
                "fan_valid": plan_info["fan_converged"].detach(),
                "converged": plan_info["converged"].detach(),
            }
            if self.hyperparams['planner'] in ['fan', 'fan_mpc']:
                metrics.update({
                    "class_goaldist": class_goaldist_loss,
                    "class_hcost": class_hcost_loss,
                    "dist_hcost_loss": dist_hcost_loss,
                    "maxmargin_goaldist": maxmargin_goaldist,
                    "class_mse": class_mse_loss,
                    "fan_hcost": fan_choice_hcost.detach(),
                })

            timer.end("metrics")

        timer.print()

        return plan_loss_batch, plan_converged, metrics, plan_info

    def prepare_plan_instance(self, batch: AgentBatch, y_dist, build_fan_inputs: bool = False):
        # Prepare planning instance.
        batch_size = batch.agent_fut.shape[0]

        if y_dist is not None:
            mus = y_dist.mus.squeeze(0)  # (b, t, K, 2)
            log_pis = y_dist.log_pis.squeeze(0) # (b, t, K)
            # # Component logprobability should be the same through time
            # assert torch.isclose(log_pis[:, 0, :], log_pis[:, -1, :]).all()
            log_pis = log_pis[:, 0]  # (b, K)
        else:
            mus = torch.full((batch_size, batch.agent_fut.shape[1], 1, 2), torch.nan, device=self.device)  # b, T, K, 2
            log_pis = torch.full((batch_size, batch.agent_fut.shape[1]), torch.nan, device=self.device)

        # # Assume that robot is always the first neighbor, i.e. robot_ind is always 0 or negative
        # assert (batch.extras['robot_ind'] <= 0).all()

        plan_batch_filter = (batch.extras['robot_ind'] >= 0)

        plan_mus_batch = mus.unsqueeze(0).transpose(1, 2)[:, :, plan_batch_filter]   # N, T, b, K, 2
        plan_logp_batch = log_pis.unsqueeze(0)[:, plan_batch_filter]  # N, b, K

        # TODO(pkarkus) In preprocessing neighbors should be ordered according to dist from ego
        # # Choose N most relevant
        # if len(others_x) > MAX_PLAN_NEIGHBORS:
        #     # Choose the most relevant, based on minimum distance to gt
        #     plan_ego_x = plan_ego_f[..., 1:, :2].unsqueeze(0)  # (1, T, 2)
        #     dists = torch.square(others_x - plan_ego_x).sum(-1)  # (N, T)
        #     dists = torch.min(dists, dim=-1).values  # (N, )
        #     sorted_idx = torch.argsort(dists)
        #     others_x = others_x[sorted_idx[:MAX_PLAN_NEIGHBORS]]

        # Combine current step and future
        # neigh_pres = batch.neigh_hist[:, :, batch.neigh_hist_len-1].unsqueeze(2)
        # neigh_pres_fut = torch.concat([neigh_pres, batch.neigh_fut], dim=2)  # b, N, T+1, 8
        # agent_pres = batch.agent_hist[:, batch.agent_hist_len].unsqueeze(1)
        # agent_pres_fut = torch.concat([agent_pres, batch.agent_fut], dim=1)  # # b, T+1, 8


        # we only want neigh_fut for vehicles
        neigh_fut = batch.neigh_fut.clone()
        nonvehicle_filter = (batch.neigh_types.int() != AgentType.VEHICLE)
        neigh_fut[nonvehicle_filter] = torch.nan

        # Convert state representation
        # ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'sintheta', 'costheta'] --> [x, y, theta, v]
        pred_fut = convert_state_pred2plan(batch.agent_fut[plan_batch_filter])  # b, N, T, 4
        neigh_fut = convert_state_pred2plan(neigh_fut[plan_batch_filter])  # b, N, T, 4
        ego_hist = convert_state_pred2plan(batch.neigh_hist[plan_batch_filter, 0])  # b, T, 4
        # To get the present time step we need to index with history len, Because history is right-padded
        pres_history_ind = batch.neigh_hist_len.to(self.device)[plan_batch_filter, 0].unsqueeze(-1).unsqueeze(-1)-1
        ego_pres = torch.gather(ego_hist, 1, torch.repeat_interleave(pres_history_ind, 4, dim=2)).squeeze(1)  # b, 4

        # # Choose N most relevant
        neigh_fut = neigh_fut[:, :MAX_PLAN_NEIGHBORS+1]  # b, N, T, 8
        # Extend to fix size. pad syntax: (last_l, last_r, second_last_l, second_last_r)
        neigh_fut = torch.nn.functional.pad(neigh_fut, (0, 0, 0, 0, 0, MAX_PLAN_NEIGHBORS+1-neigh_fut.shape[1]), 'constant', torch.nan)
        plan_ego_batch = neigh_fut[:, 0, :, :].transpose(0, 1)  # T, b, 4
        plan_gt_neighbors_batch = neigh_fut[:, 1:, :, :2].transpose(0,1).transpose(1,2)  # N, T, b, 2
        # Combine predicted agent gt future with neighbor futures
        plan_all_gt_neighbors_batch = torch.concat([pred_fut.unsqueeze(0)[..., :2].transpose(1 ,2), plan_gt_neighbors_batch], dim=0)  # N+1, T, b, 2
        # Extend plan agent state with present
        plan_ego_batch = torch.concat([ego_pres.unsqueeze(0), plan_ego_batch], dim=0)  # T+1, b, 4

        empty_mus_batch = torch.zeros((0, plan_mus_batch.shape[1], batch_size, plan_mus_batch.shape[3], 2), dtype=plan_mus_batch.dtype, device=self.device)
        empty_logp_batch = torch.zeros((0, batch_size, plan_mus_batch.shape[3]), dtype=plan_mus_batch.dtype, device=self.device)

        # Fill relevant lane data if not cached
        if build_fan_inputs: 
            lanes_near_goal_filtered = [batch.extras['lanes_near_goal'][i] for i in range(batch_size) if plan_batch_filter[i]]
        else:
            lanes_near_goal_filtered = None
        fan_inputs = {'lanes_near_goal_filtered': lanes_near_goal_filtered}

        lanes = batch.extras['lane_projection_points'][plan_batch_filter].transpose(1, 0)  # b, T, 3 --> T, b, 3
        goal = batch.extras['goal'][plan_batch_filter, ..., :2]  # b, 2

        return plan_batch_filter, plan_ego_batch, lanes, goal, plan_mus_batch, plan_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, fan_inputs

    def plan_batch(self, x_init_batch, lanes, goal_batch, mus_batch, logp_batch, empty_mus_batch, empty_logp_batch, gt_neighbors_batch, all_gt_neighbors_batch, future_mode, init_mode, planner="mpc", pred_extra=None, fan_inputs=None, plan_batch_filter=None,  gt_plan_u=None, nopred_plan_u=None,  return_iters=False):
        """
        """
        batch_size = x_init_batch.shape[0]

        u_fitted = None
        lane_points = None
        x_goal_batch = None
        plan_data = None

        # Choose initialization
        if init_mode == "fitted":
            assert not torch.isnan(u_fitted).any()
            u_init = u_fitted
            assert nopred_plan_u.shape[0] == self.planh+1, "u_init trajlen does not match planh"
        elif init_mode == "gtplan":
            u_init = gt_plan_u  # (ph, b, 2)
            assert nopred_plan_u.shape[0] == self.planh+1, "u_init trajlen does not match planh"
        elif init_mode == "nopred_plan":
            u_init = nopred_plan_u  # (ph, b, 2)
            assert nopred_plan_u.shape[0] == self.planh+1, "u_init trajlen does not match planh"
        elif init_mode == "zero":
            u_init = torch.zeros((self.planh+1, batch_size, 2), device=self.device)  # (ph, b, 2)
        else:
            raise NotImplementedError("Unknown plan_init: %s"%init_mode)

        # Chooise planner inputs
        if future_mode == "pred":
            pass
        elif future_mode == "nopred":
            # drop the predicted agent, include the other gt agents
            mus_batch = empty_mus_batch
            logp_batch = empty_logp_batch
        elif future_mode == "gt":
            # replace prediction with gt
            mus_batch = empty_mus_batch
            logp_batch = empty_logp_batch
            gt_neighbors_batch = all_gt_neighbors_batch
        elif future_mode == "none":
            # drop all predicted and gt agent futures
            mus_batch = empty_mus_batch
            logp_batch = empty_logp_batch
            # MAXN
            # gt_neighbors_batch = [torch.zeros((0, self.ph, 2), dtype=u_init.dtype, device=self.device)] * len(mus_batch)
            gt_neighbors_batch = None  # torch.zeros((MAX_PLAN_NEIGHBORS, self.ph, mus_batch.shape[2], 2), dtype=u_init.dtype, device=self.device) 
        elif future_mode == "pred_nogt":
            # drop gt agent futures but keep prediction
            gt_neighbors_batch = None 
        else:
            raise ValueError("Unknown value use_future=%s"%str(future_mode))
        
        # MAXN
        # probs_batch = [torch.exp(logp) for logp in logp_batch]
        probs_batch = torch.exp(logp_batch)

        cost_inputs = (gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, lane_points)

        # Run planner
        if planner == "mpc":
            self.mpc_obj.n_batch = batch_size
            self.mpc_obj.u_init = u_init.detach()                
            planned_x, planned_u, planned_cost, is_converged, iters = self.mpc_obj(x_init_batch, self.cost_obj, self.dyn_obj, cost_inputs, return_converged=True, return_iters=return_iters)
            planned_x = self.mpc_obj.detach_unconverged_tensor(planned_x, is_converged)
            planned_u = self.mpc_obj.detach_unconverged_tensor(planned_u, is_converged)
            self.mpc_obj.u_init = None
            iters['fan_converged'] = torch.ones_like(is_converged)  # only to report for eval stats

        elif planner == "fan":  
            assert (self.hyperparams['plan_agent_choice'] == 'most_relevant'), "Only supported because relevant_lanes are for the most relevant agent."
            if 'fan_candidate_trajs_filtered' in fan_inputs:           
                planned_x, planned_u, planned_cost, is_converged, iters = self.fan_obj(
                    x_init_batch, x_goal_batch, self.cost_obj, self.dyn_obj, cost_inputs, candidate_trajs_batch=fan_inputs['fan_candidate_trajs_filtered'], is_valid_batch=fan_inputs['fan_valid_filtered'])
            else:
                planned_x, planned_u, planned_cost, is_converged, iters = self.fan_obj(
                    x_init_batch, x_goal_batch, self.cost_obj, self.dyn_obj, cost_inputs, relevant_lanes_batch=fan_inputs['lanes_near_goal_filtered'])
            iters['fan_converged'] = is_converged  # only for compatibility with fan_mpc planner when used in class_hcost2 loss

        elif planner == "fan_mpc":
            # Do trajectory-fan planning, and initialize MPC with the planned trajectory.
            # Almost the same code duplicated as above.
            assert (self.hyperparams['plan_agent_choice'] == 'most_relevant'), "Only supported because relevant_lanes are for the most relevant agent."
            if 'fan_candidate_trajs_filtered' in fan_inputs:              
                fan_planned_x, fan_planned_u, fan_planned_cost, fan_converged, fan_iters = self.fan_obj(
                    x_init_batch, x_goal_batch, self.cost_obj, self.dyn_obj, cost_inputs, candidate_trajs_batch=fan_inputs['fan_candidate_trajs_filtered'], is_valid_batch=fan_inputs['fan_valid_filtered'])
            else:
                fan_planned_x, fan_planned_u, fan_planned_cost, fan_converged, fan_iters = self.fan_obj(
                    x_init_batch, x_goal_batch, self.cost_obj, self.dyn_obj, cost_inputs, relevant_lanes_batch=fan_inputs['lanes_near_goal_filtered'])

            # Use planner's output as initialization for MPC
            #   fan_planned_u is already a best control that approximates the spline when unrolled, taking u[t] = (u[t] + u[t+1])/2
            #   detach: fan_planned_u is chosen from a set of candidates, its normally not a function of learnable parameter, so we could only 
            #   backprop if we used a weighted sum type of output in the trajfan planner
            u_init = fan_planned_u.detach()

            # # Debug
            # torch.set_printoptions(precision=10, linewidth=160)
            # print (u_init.sum())
            # print (x_init_batch.sum())
            # print (mus_batch.sum())
            # print (probs_batch.sum())
            # print (goal_batch.sum())
            # print (torch.nan_to_num(lanes, 0.19).sum())
            # print (torch.nan_to_num(gt_neighbors_batch, 0.0019).sum())

            # # Manually fix seed
            # import random
            # seed = 100
            # random.seed(seed)
            # np.random.seed(seed)
            # torch.manual_seed(seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(seed)

            # MPC
            self.mpc_obj.n_batch = batch_size
            self.mpc_obj.u_init = u_init.detach()                
            planned_x, planned_u, planned_cost, mpc_converged, iters = self.mpc_obj(x_init_batch, self.cost_obj, self.dyn_obj, cost_inputs, return_converged=True, return_iters=return_iters)
            self.mpc_obj.u_init = None

            # print (planned_x.sum())

            # Consider converged only those that are converged (valid) for fan planner and converged for MPC
            is_converged = torch.logical_and(fan_converged, mpc_converged)
            planned_x = self.mpc_obj.detach_unconverged_tensor(planned_x, is_converged)
            planned_u = self.mpc_obj.detach_unconverged_tensor(planned_u, is_converged)            

            # Merge planning info
            iters.update(fan_iters)
            # # Intentionally not detaching these so we can backprop through a fan-planner loss optionally
            # iters['fan_planned_x'] = fan_planned_x
            # iters['fan_planned_u'] = fan_planned_u
            # iters['fan_planned_cost'] = fan_planned_cost
            iters['fan_converged'] = fan_converged
            if return_iters:
                iters['mpc_converged'] = mpc_converged
                                         
        else:
            raise NotImplementedError("Unknown planner: %s"%str(planner))

        # Add detailed cost info
        if return_iters:
            cost_components = self.cost_obj(torch.cat((planned_x, planned_u), dim=-1), cost_inputs=cost_inputs, keep_components=True)
            iters['cost_components'] = cost_components.detach()
            if planner in ["fan_mpc"]:
                fan_cost_components = self.cost_obj(torch.cat((fan_planned_x, fan_planned_u), dim=-1), cost_inputs=cost_inputs, keep_components=True)
                iters['fan_cost_components'] = fan_cost_components.detach()
            if planner in ["mpc", "fan_mpc"]:
                iters['lane_targers'] = lanes.detach()

        # # Gradcheck
        # from torch.autograd import gradcheck
        # from torch.autograd.gradcheck import get_analytical_jacobian, get_numerical_jacobian
        # for batch_i in tqdm(range(105, batch_size)):   # 121 has large error
        #     self.mpc_obj.n_batch = 1
        #     self.mpc_obj.u_init = u_init.detach().double()[:, batch_i:batch_i+1]
        #     convert_list = lambda l: l[batch_i].double()
        #     def wrapped_func(mus_batch, probs_batch): 
        #         planx, planu, cost, converged = self.mpc_obj(x_init_batch[batch_i:batch_i+1].double(), self.cost_obj, self.dyn_obj, (self.cost_theta.double(), [convert_list(gt_neighbors_batch)], [mus_batch], [probs_batch],  goal_batch[batch_i:batch_i+1].double(), lanes[:, batch_i:batch_i+1].double()), return_converged=True)
        #         if not converged[0]:
        #             planx = planx * 0
        #             planu = planu * 0
        #         return planx, planu
        #     inputs = (convert_list(mus_batch), convert_list(probs_batch))
        #     assert gradcheck(wrapped_func, inputs, eps=1e-6, atol=0.1, rtol=0.01)
        #     # assert gradcheck(wrapped_func, inputs, eps=1e-6, atol=1e-4)
        # print ("Gradient check passed")

        return planned_x, planned_u, planned_cost, is_converged, iters

    def augment_sample_with_dummy_plan_info(self, sample, ego_traj=None):
        ph = self.ph

        (first_history_index,
            x_t, y_t, x_st_t, y_st_t,
            neighbors_data_st,
            neighbors_edge_value,
            robot_traj_st_t,
            map_input, neighbors_future_data, plan_data) = sample
        assert isinstance(plan_data, batchable_dict)

        if plan_data['most_relevant_idx'] < 0:
            ego_gt_x = torch.zeros((ph+1, 4))
        else:
            ego_gt_xy = neighbors_future_data[('VEHICLE', 'VEHICLE')][plan_data['most_relevant_idx']][..., :2]
            ego_gt_x = torch.cat((ego_gt_xy, torch.zeros_like(ego_gt_xy)), dim=-1)  # (..., 4)
        plan_data['gt_plan_x']=ego_gt_x # torch.zeros((ph+1, 4))
        plan_data['gt_plan_u']=torch.zeros((ph+1, 2))
        plan_data['gt_plan_hcost']=torch.zeros(())
        plan_data['gt_plan_converged']=torch.zeros(())
        plan_data['nopred_plan_x']=ego_gt_x  # torch.zeros((ph+1, 4))
        plan_data['nopred_plan_u']=torch.zeros((ph+1, 2))
        plan_data['nopred_plan_hcost']=torch.zeros(())
        plan_data['nopred_plan_converged']=torch.zeros(())

        if ego_traj is not None:
            # Find ego state in neighbors
            all_neighbors = np.stack(neighbors_future_data[('VEHICLE', 'VEHICLE')], axis=0)
            # last ego_traj step should match current state
            dists = np.abs(all_neighbors[:, 0, :2] - ego_traj[None, -1, :2]).sum(1)
            assert np.isclose(dists.min(), 0), "could not find ego in neighbours structure"
            plan_neighbor = np.argmin(dists)

            plan_data['most_relevant_idx'] = torch.Tensor([int(plan_neighbor)]).int().squeeze(0)
            plan_data['robot_idx'] = torch.Tensor([int(plan_neighbor)]).int().squeeze(0)
                                    
        return (first_history_index,
            x_t, y_t, x_st_t, y_st_t,
            neighbors_data_st, 
            neighbors_edge_value,
            robot_traj_st_t,
            map_input, neighbors_future_data, plan_data)

    def plan_for_gt(self, batch, node_type, exclude_pred_agent=False, fan_candidates=False):
        """
        """
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
         neighbors_edge_value,
         robot_traj_st_t,
         map, neighbors_future_data, plan_data) = batch
        batch_size = x_t.shape[0]

        # x = x_t.to(self.device)
        y_gt = y_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Restore encodings
        neighbors_future_data = restore(neighbors_future_data)

        assert node_type == "VEHICLE" 
        # TODO support planning for pedestrian prediction
        # we can only plan for a vehicle but we can use pedestrian prediction.

        # Prepare planning instance.
        plan_batch_filter, plan_ego_batch, plan_mus_batch, plan_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch  = self.prepare_plan_instance(
            node_type, neighbors_future_data, plan_data, y_gt, y_dist=None, update_fan_inputs=((self.hyperparams['planner'] in ['fan', 'fan_mpc']) or fan_candidates))
        pred_T = y_gt.shape[1]  # b, T, state_dim

        plan_x_full = torch.full((pred_T + 1, batch_size, 4), torch.nan, dtype=torch.float, device=self.device)
        plan_u_full = torch.full((pred_T + 1, batch_size, 2), torch.nan, dtype=torch.float, device=self.device)
        plan_hcost_full = torch.full((batch_size, ), torch.nan, dtype=torch.float, device=self.device)
        plan_converged_full = torch.zeros((batch_size, ), dtype=torch.bool, device=self.device)
        fan_candidates_list = [torch.zeros((1, pred_T+1, 6), device=self.device)] * batch_size
        fan_valid_full = torch.zeros((batch_size, ), dtype=torch.bool, device=self.device)

        # MAXN
        if  plan_ego_batch.shape[1] > 0:
            # plan_ego_time_batch = torch.stack(plan_ego_batch, dim=1)  # (T, b, ...)
            plan_ego_time_batch = plan_ego_batch  # (T, b, ...)
            x_gt, u_gt, lanes, x_proj, u_fitted, x_init_batch, goal_batch, lane_points,  _, _, _ = self.decode_plan_inputs(plan_ego_time_batch)

            plan_x, plan_u, plan_cost, plan_converged, _ = self.plan_batch(
                plan_ego_time_batch, empty_mus_batch, empty_logp_batch, 
                empty_mus_batch, empty_logp_batch, plan_gt_neighbors_batch, (plan_gt_neighbors_batch if exclude_pred_agent else plan_all_gt_neighbors_batch), 
                future_mode="gt", init_mode="zero", planner="mpc", gt_plan_u=None, nopred_plan_u=None,
                plan_data=plan_data, return_iters=False)  # (T, b, ...) 

            plan_xu = torch.cat((plan_x, plan_u), -1)
            plan_hcost = self.cost_obj(plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points))  # (T, b)
            plan_hcost = torch.mean(plan_hcost, dim=0)  # (b,)

            plan_x_full[:, plan_batch_filter] = plan_x
            plan_u_full[:, plan_batch_filter] = plan_u
            plan_hcost_full[plan_batch_filter] = plan_hcost
            plan_converged_full[plan_batch_filter] = plan_converged

            if fan_candidates:
                fan_ctrl_xu_batch, is_converged_batch = self.fan_obj.get_candidate_trajectories(x_init_batch, relevant_lanes_batch=plan_data['most_relevant_nearby_lanes_filtered'])

                for i, batch_i in enumerate(np.arange(batch_size)[plan_batch_filter.cpu().numpy()]):
                    fan_candidates_list[batch_i] = fan_ctrl_xu_batch[i]
                    fan_valid_full[batch_i] = is_converged_batch[i]

        return plan_x_full, plan_u_full, plan_converged_full, plan_hcost_full, plan_batch_filter, fan_candidates_list, fan_valid_full
