import torch

from collections import OrderedDict
from typing import Dict, Optional, Union, Any, List, Set

from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement

from diffstack.modules.module import Module, ModuleSequence, DataFormat, RunMode
from diffstack.modules.predictors.constvel_predictor import ConstVelPredictor
from diffstack.modules.predictors.trajectron_predictor import TrajectronPredictor, TrajectronPredictorWithCacheData
from diffstack.modules.planners.fan_mpc_planner import FanMpcPlanner

from diffstack.utils.utils import merge_dicts_with_prefix, restore
from diffstack.utils.visualization import visualize_plan_batch


class DiffStack(ModuleSequence):

    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["batch"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["pred.pred_dist", "pred.pred_ml", "plan.plan_xu", "loss:train", "loss:validate"])

    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super().__init__([], model_registrar, hyperparams, log_writer, device, input_mappings={})
        
        # Build the stack into the components ordered dictionary.
        self.components = OrderedDict()

        # Initialize predictor
        if self.hyperparams["predictor"] == "constvel":
            self.pred_obj = ConstVelPredictor(
                model_registrar, hyperparams, log_writer, device, 
                input_mappings={"agent_batch": "batch"})
            assert not self.hyperparams["train_pred"]
            scene_centric = False
        elif self.hyperparams["predictor"] in ["tpp", "gt", "nopred"]:            
            self.pred_obj = TrajectronPredictor(
                model_registrar, hyperparams, log_writer, device, 
                input_mappings={"agent_batch": "batch"})
            scene_centric = False
        elif self.hyperparams["predictor"] in ["tpp_cache"]:            
            self.pred_obj = TrajectronPredictorWithCacheData(
                model_registrar, hyperparams, log_writer, device, 
                input_mappings={"agent_batch": "input.batch"})
            scene_centric = False
        else:
            raise ValueError(f"Unknown predictor={self.hyperparams['predictor']}")

        self.components["pred"] = self.pred_obj

        # Initialize planner/controller
        if self.hyperparams["planner"] in ["", "none"] or not self.hyperparams["plan_train"]:
            # TODO Implement a dummy planner with zero outputs, and support different training and inference configs for plan_train parameter
            raise NotImplementedError
        elif self.hyperparams["planner"] in ["none", "mpc", "fan", "fan_mpc"]:
            self.planner_obj = FanMpcPlanner(
                model_registrar, hyperparams, log_writer, device, 
                input_mappings={"agent_batch": "pred.agent_batch" if scene_centric else "input.batch"})
        else:
            raise ValueError(f"Unknown planner={self.hyperparams['planner']}")
        self.components["plan"] = self.planner_obj

        # Verify inputs/outputs with dry run
        self.dry_run(input_keys=['batch', 'loss_weights'])

        # Initialize training counters
        self.curr_plan_loss_scaler = None
        self.set_curr_epoch(self.curr_epoch)  # will initialize curr_plan_loss_scaler
        self.set_annealing_params()        

    def set_curr_epoch(self, curr_epoch):
        super().set_curr_epoch(curr_epoch)

        # update plan_loss_scaler
        scale_start = self.hyperparams['plan_loss_scale_start']
        scale_end = self.hyperparams['plan_loss_scale_end']
        nominal_scaler = self.hyperparams['plan_loss_scaler']
        if scale_end < 0 or curr_epoch >= scale_end:
            self.curr_plan_loss_scaler = nominal_scaler
        elif curr_epoch <= scale_start:
            self.curr_plan_loss_scaler = 0.
        else:
            assert scale_start < scale_end
            # get percentage where we are in full range
            self.curr_plan_loss_scaler = nominal_scaler * float(curr_epoch-scale_start) / float(scale_end-scale_start)
        print ("Setting plan_loss_scaler=%f"%self.curr_plan_loss_scaler)

    def train(self, inputs: Dict):
        batch: AgentBatch = inputs["batch"]
        batch.to(self.device)

        # Add bias to prediction targets optionally
        agent_fut_unbiased = batch.agent_fut.clone()
        batch.agent_fut = self.bias_labels_maybe(agent_fut_unbiased)

        # Extend input with prediction loss importance weights
        pred_loss_weights = self.prediction_importance_weights(
            batch, self.hyperparams["pred_loss_weights"], self.hyperparams["pred_loss_temp"])
        inputs['loss_weights'] = pred_loss_weights

        # Run forward pass of the stack 
        outputs = self.sequence_components(inputs, run_mode=RunMode.TRAIN) 
                
        # Losses
        loss_components = []
        if self.hyperparams["train_pred"]:
            loss_components.append(outputs['pred.loss'] * self.hyperparams["pred_loss_scaler"])
        loss_components.append(outputs['plan.loss'] * self.curr_plan_loss_scaler)
        loss = torch.stack(loss_components).sum()
        outputs.update({"loss": loss})

        # Log
        self.write_training_logs(batch, outputs)

        return outputs

    def validate(self, inputs: Dict):
        """Run the stack forward and compute metrics.
        """
        batch: AgentBatch = inputs["batch"]
        batch.to(self.device)

        # Add bias to prediction targets optionally
        agent_fut_unbiased = batch.agent_fut.clone()
        batch.agent_fut = self.bias_labels_maybe(agent_fut_unbiased)

        # Run forward pass of the stack 
        outputs = self.sequence_components(inputs, run_mode=RunMode.VALIDATE) 

        outputs["metrics"] = merge_dicts_with_prefix({"pred": outputs['pred.metrics'], "plan": outputs['plan.metrics']})

        return outputs

    def infer(self, inputs: Dict):
        """Run the stack forward without computing metrics or losses.
        """
        batch: AgentBatch = inputs["batch"]
        batch.to(self.device)

        # Add bias to prediction targets optionally
        agent_fut_unbiased = batch.agent_fut.clone()
        batch.agent_fut = self.bias_labels_maybe(agent_fut_unbiased)

        # Run forward pass of the stack 
        outputs = self.sequence_components(inputs, run_mode=RunMode.INFER) 

        return outputs

    def write_training_logs(self, batch, outputs):
        # Log planning loss. Logs for prediction loss are added in mgcvae.loss()
        if self.log_writer is not None:
            plan_active = float(outputs['plan.loss_batch'].shape[0])/float(batch.agent_hist.shape[0])
            plan_converged_rate = outputs['plan.converged'].float().mean()       
            plan_converged_mean_loss = outputs['plan.loss_batch'][outputs['plan.converged']].mean()
                
            self.log_writer.log({
                'Plan/loss/plan': outputs['plan.loss'].detach().item(),
                'Plan/loss/plan_conv': plan_converged_mean_loss.detach().item(),
                'Plan/loss/pred': outputs['pred.loss'].detach().item(),
                'Plan/loss/total': outputs['loss'].detach().item(),        
                'Plan/loss/scaler': self.curr_plan_loss_scaler,
                'Plan/active': plan_active,        
                'Plan/converged': plan_converged_rate.detach().item(),
                }, step=self.curr_iter, commit=False)

            self.log_writer.log({
                'Plan/'+k: v for k, v in self.planner_obj.cost_obj.get_params_log().items()
                }, step=self.curr_iter, commit=False)

    def get_params_summary_text(self):
        return self.planner_obj.cost_obj.get_params_summary_str()

    def _deprecated_validation(self, batch: AgentBatch, return_plot_data=False):
        batch.to(self.device)
        plot_data = {}

        # Add bias to prediction targets optionally
        agent_fut_unbiased = batch.agent_fut.clone()
        batch.agent_fut = self.bias_labels_maybe(agent_fut_unbiased)

        node_types = batch.agent_types()
        if len(node_types) > 1:
            raise NotImplementedError("Mixing agent types for prediction in a batch is not supported.")
        node_type = node_types[0]

        # Prediction, use the most likely latent mode for predictions, all modes for y_dists.
        predictions, y_dist, pred_extra = self.pred_obj._run_prediction(batch, prediction_horizon=batch.agent_fut.shape[1])

        # planning
        if self.hyperparams["planner"] not in ["", "none"] and node_type.name in self.hyperparams["plan_node_types"]:  
            # TODO support planning for pedestrian prediction
            # we can only plan for a vehicle but we can use pedestrian prediction.
            
            if self.hyperparams["predictor"] == "nopred":
                future_mode="nopred"
            elif self.hyperparams["predictor"] == "gt":
                future_mode="gt"
            elif self.hyperparams["predictor"] == "blind":
                future_mode="none"
            else:
                future_mode="pred"

            plan_loss_batch, plan_converged_batch, plan_metrics, plan_info = self.planner_obj._plan_loss(
                batch, y_dist, init_mode=self.hyperparams["plan_init"], loss_mode=self.hyperparams["plan_loss"], future_mode=future_mode, pred_extra=pred_extra, return_iters=True)  #(b, )
            if not plan_metrics:
                # emptly plan batch
                metrics_dict = {}
                plan_and_fan_valid = None
            else:
                metrics_dict = dict(
                    plan_loss=plan_loss_batch, 
                    plan_converged=plan_converged_batch,
                    plan_cost=plan_metrics['cost'],
                    plan_unbiased_d1=plan_metrics['unbiased_d1'], 
                    plan_unbiased_d2=plan_metrics['unbiased_d2'], 
                    # TODO better names
                    # plan_mse_d1=plan_metrics['unbiased_d1'], 
                    # plan_mse=plan_metrics['mse'], 
                    plan_hcost=plan_metrics['hcost'],
                    plan_valid=plan_info['plan_batch_filter'],
                    ego_pred_gt_dist=plan_info['ego_pred_gt_dist'],
                    fan_converged=plan_info['fan_converged'], 
                ) 
                if 'class_goaldist' in plan_metrics:
                    metrics_dict.update({"plan_"+k: plan_metrics[k] for k in ["class_goaldist", "class_hcost", "dist_hcost_loss", "maxmargin_goaldist", "class_mse", "fan_hcost"]})
                plan_and_fan_valid=plan_info['plan_and_fan_valid']

                # HACK
                cost_components_batch = plan_info['cost_components'][1:].mean(0)  # mean over future
                hcost_components_batch = plan_info['hcost_components'][1:].mean(0)   # mean over future        
                icost_components_batch = plan_info['icost_components'][1:].mean(0)  # mean over future
                for i in range(cost_components_batch.shape[-1]):
                    metrics_dict[f"costcomp_{i}"] = cost_components_batch[:, i]
                    metrics_dict[f"hcostcomp_{i}"] = hcost_components_batch[:, i]
                for i in range(icost_components_batch.shape[-1]):
                    metrics_dict[f"icostcomp_{i}"] = icost_components_batch[:, i] * 1000.0  # scale by 10-3, sum over 6 future steps

                # Debug comparisons with different planner inputs
                _, _, nopred_plan_metrics, nopred_plan_info = self.planner_obj._plan_loss(
                    batch, y_dist, init_mode=self.hyperparams["plan_init"], loss_mode=self.hyperparams["plan_loss"], future_mode="nopred", pred_extra=pred_extra, return_iters=True)  #(b, )

                # _, _, nof_plan_metrics, nof_plan_info = self.planner_obj._plan_loss(
                #    batch, y_dists, init_mode=self.hyperparams["plan_init"], loss_mode=self.hyperparams["plan_loss"], future_mode="none",  return_iters=True)  #(b, )

                _, gt_plan_converged, gt_plan_metrics, gt_plan_info = self.planner_obj._plan_loss(
                    batch, y_dist, init_mode=self.hyperparams["plan_init"], loss_mode=self.hyperparams["plan_loss"], future_mode="gt", pred_extra=pred_extra, return_iters=True)  #(b, )

                metrics_dict.update(dict(
                    plan_unbiased_nopred_d1=nopred_plan_metrics['unbiased_d1'], plan_unbiased_nopred_d2=nopred_plan_metrics['unbiased_d2'], plan_nopred_hcost=nopred_plan_metrics['hcost'], 
                    # plan_unbiased_nof_d1=nof_plan_metrics['unbiased_d1'], plan_unbiased_nof_d2=nof_plan_metrics['unbiased_d2'], plan_nof_hcost=nof_plan_metrics['hcost'], 
                    plan_unbiased_gt_d1=gt_plan_metrics['unbiased_d1'], plan_unbiased_gt_d2=gt_plan_metrics['unbiased_d2'], plan_gt_hcost=gt_plan_metrics['hcost'], 
                    plan_gt_converged=gt_plan_converged,
                ))

                valid_filter = plan_metrics["fan_valid"]
                metrics_dict.update({k+"_valid": v[valid_filter] for k, v in plan_metrics.items()})

            if return_plot_data:                  
                plot_data = {
                     'predictions': predictions.detach(),
                     'y_dists': y_dist,
                     # 'y_for_pred': y_for_pred.detach(),
                     'plan': (plan_metrics, plan_info),
                     'nopred_plan':  (nopred_plan_metrics, nopred_plan_info,),
                     # 'nof_plan': (nof_plan_metrics, nof_plan_info,),
                     'gt_plan': (gt_plan_metrics, gt_plan_info),
                    }
        else:
            metrics_dict = {}
            plan_and_fan_valid = None

        # Compute default metrics
        pred_metrics_dict = compute_prediction_metrics(predictions, batch.agent_fut[..., :2], y_dists=y_dist)
        metrics_dict.update(pred_metrics_dict)
        unbiased_pred_metrics = compute_prediction_metrics(predictions, agent_fut_unbiased[..., :2], y_dists=y_dist)
        metrics_dict.update({k + "_unbiased": v for k, v in unbiased_pred_metrics.items()})

        if plan_and_fan_valid is not None:
            for pred_metric_name in ['ml_ade', 'ml_fde', 'nll_mean', 'nll_final']:
                metrics_dict[pred_metric_name+"_valid"] = metrics_dict[pred_metric_name][plan_and_fan_valid]

        return metrics_dict, plot_data

    def augment_states_with_ego_indicator(self, x, x_st_t, neighbor_states, plan_data):
        """Set an ego-indicator for prediction based on which agent we will treat as ego in the planner."""

        raise NotImplementedError("Not implemented for trajdata input")

        if self.hyperparams['pred_ego_indicator'] == 'none':
            # Do nothing
            return x, x_st_t, neighbor_states

        # Choose which neighbor is ego
        if self.hyperparams['pred_ego_indicator'] == 'robot':
            ego_node_type = "VEHICLE"
            # ego_inds = torch.unbind(plan_data[:, 1].int(), 0)
            ego_inds = plan_data['robot_idx'].int()
        elif self.hyperparams['pred_ego_indicator'] == 'most_relevant':
            ego_node_type = "VEHICLE"
            # ego_inds = torch.unbind(plan_data[:, 0].int(), 0)
            ego_inds = plan_data['most_relevant_idx'].int()
        else:
            raise ValueError(f"Unknown pred_ego_indicator {self.hyperparams['pred_ego_indicator']}")
        
        ext_neighbor_states = {}
        for edge_type, node_neighbor_states in neighbor_states.items():
            ext_node_neighbor_states = []
            for batch_i, neighbor_state in enumerate(node_neighbor_states):
                ego_ind = ego_inds[batch_i]
                # TODO only supports vehicle
                if edge_type[1] == ego_node_type:
                    ext_node_neighbor_states.append([torch.nn.functional.pad(neighbor_state[i], (0, 1), value=(ego_ind == i).float()) for i in range(len(neighbor_state))])
                else:
                    ext_node_neighbor_states.append([torch.nn.functional.pad(neighbor_state[i], (0, 1), value=0.) for i in range(len(neighbor_state))])
            ext_neighbor_states[edge_type] = ext_node_neighbor_states

        x = torch.nn.functional.pad(x, (0, 1))
        x_st_t = torch.nn.functional.pad(x_st_t, (0, 1))

        return x, x_st_t, ext_neighbor_states

    def bias_labels_maybe(self, y_t):
        if self.hyperparams["bias_predictions"]:
            y_t = y_t + 2.
        return y_t

    def prediction_importance_weights(self, batch: AgentBatchElement, pred_loss_weights: str, temperature: float):
        if pred_loss_weights == "none":
            return None

        raise NotImplementedError("Not implemented for trajdata input")
        
        batch_size = y_gt.shape[0]
        zero_weights = torch.zeros((batch_size, ), dtype=y_gt.dtype, device=y_gt.device)

        node_types = batch.agent_types()
        if len(node_types) > 1:
            raise NotImplementedError("Mixing agent types for prediction in a batch is not supported.")
        node_type = node_types[0]

        plan_batch_filter, plan_ego_batch, plan_mus_batch, plan_logp_batch, plan_gt_neighbors_batch, plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch = self.prepare_plan_instance(
            node_type, neighbors_future_data, plan_data, y_gt, None, update_fan_inputs=False)

        if plan_ego_batch.shape[1] == 0:
            print ("Warning: no valid planning example in batch, keeping all prediction weights zero.")
            return zero_weights

        if pred_loss_weights == "dist":
            ego_gt_future = plan_ego_batch[1:, ..., :2]  # (T, b_plan, 2)
            agent_gt_future = y_gt[plan_batch_filter, ..., :2].transpose(1, 0)  # (T, b,_plan, 2)

            dists = torch.square(agent_gt_future[..., :2] - ego_gt_future[:, :, :2]).sum(dim=-1).sqrt()  # (T, b)
            dists = torch.min(dists, dim=0).values  # min over time, (b, )
            weights = torch.exp(-dists * temperature)

        elif pred_loss_weights == "grad":
            x_gt, u_gt, lanes, x_proj, u_fitted, x_init_batch, goal_batch, lane_points, _, _, _ = self.decode_plan_inputs(plan_ego_batch)
            plan_xu_gt = plan_ego_batch[..., :6]   # decode_plan removes t0 but we need it here, so take it from the original feature vector

            cost_inputs = (None, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points)  # gt neighbors will be replaced
            # Take only the gt future for the predicted agent, which is the last in plan_all_gt_neighbors_batch
            pred_gt_neighbor = plan_all_gt_neighbors_batch[-1:]
            
            grads = self.planner_obj.cost_obj.gt_neighbors_gradient(plan_xu_gt, cost_inputs, pred_gt_neighbor)  # (1, t, b, 2)

            grad_norm = torch.linalg.norm(grads.squeeze(0), dim=-1)  # norm over x, y  (t, b)
            grad_norm = torch.mean(grad_norm, dim=0)  # over time (b, )
            dists = grad_norm   # for debug

            # temperature: equivalent of weights = exp(log(grad^2) * temp). 
            # The 2x scaler creates a similar histogram for temp=1 in the case of dist and grad.
            weights = torch.pow(grad_norm, 2 * temperature)

        else:
            raise ValueError(f"Unknown setting pred_loss_weights={pred_loss_weights}")
        
        # extend to full batch, examples with no valid ego will get zero weight
        weights_full = zero_weights
        weights_full[plan_batch_filter] = weights
        # normalize
        weights_full = weights_full / weights_full.sum()

        return weights_full

    def visualize_plan(self, nusc_maps, scenes, batch, plot_data, titles, plot_styles, num_plots=-1):

        raise NotImplementedError("Not implemented for trajdata input")

        # shortcut
        if not plot_styles:
            return {}, []

        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, neighbors_future_data, plan_data) = batch

        # Restore encodings
        neighbors_data_st = restore(neighbors_data_st)
        neighbors_edge_value = restore(neighbors_edge_value)
        plan_data = restore(plan_data)

        x = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_t = y_t.to(self.device)

        x, x_st_t, neighbors_data_st = self.augment_states_with_ego_indicator(x, x_st_t, neighbors_data_st, plan_data)

        output, plotted_inds = visualize_plan_batch(nusc_maps, scenes, x_t, y_t, plan_data, plot_data, titles, plot_styles, num_plots, self.hyperparams['planner'], self.ph, self.planh)
        
        return output, plotted_inds

