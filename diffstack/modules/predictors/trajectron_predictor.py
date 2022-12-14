import itertools
import torch
import numpy as np
from typing import Dict, Optional, Union, Any
from contextlib import nullcontext

from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.agent import AgentType
from trajdata.utils.arr_utils import PadDirection
from diffstack.utils.pred_utils import compute_prediction_metrics
from diffstack.utils.utils import set_all_seeds, restore

from diffstack.modules.module import Module, DataFormat, RunMode
from diffstack.modules.predictors.trajectron_utils.node_type import NodeTypeEnum
from diffstack.modules.predictors.trajectron_utils.model.mgcvae import MultimodalGenerativeCVAE

from diffstack.data.cached_nusc_as_trajdata import standardized_manual_state, convert_trajdata_hist_to_manual_hist


class TrajectronPredictor(Module):
    # TODO (pkarkus) add support for multiple predicted agent types. Add a function that separates metrics per agent type.

    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["agent_batch", "loss_weights:train"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["pred_dist", "pred_ml", "loss:train", "metrics:train", "metrics:validate"])

    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device, input_mappings = {}):
        super().__init__(model_registrar, hyperparams, log_writer, device, input_mappings)

        # Prediction state variables
        self.pred_state = self.hyperparams['pred_state']
        self.state = self.hyperparams['state'] 
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        assert self.state_length['PEDESTRIAN'] in [6, 7]

        # Validate that hyperparameters are consistent
        node_type_state = self.state["VEHICLE"]
        if self.hyperparams["pred_ego_indicator"] == "none":
            if "augment" in node_type_state and "ego_indicator" in node_type_state["augment"]:
                raise ValueError("Inconsistent setting: state variables include ego_indicator but pred_ego_indicator=none")
        else:
            if "augment" not in node_type_state or "ego_indicator" not in node_type_state["augment"]:
                raise ValueError("Inconsistent setting: state variables do not include ego_indicator but pred_ego_indicator != none")
        assert np.isclose(hyperparams['dt'] / hyperparams['plan_dt'], int(hyperparams['dt'] / hyperparams['plan_dt'])), "dt must be a multiple of plan_dt"

        node_types = NodeTypeEnum(["VEHICLE", "PEDESTRIAN"])
        edge_types = list(itertools.product(node_types, repeat=2))
        # Build models for each agent type
        self.node_models_dict = torch.nn.ModuleDict()  
        for node_type in node_types:
            # Only add a Model for NodeTypes we want to predict
            class EnvClass:
                dt = self.hyperparams["dt"]
                robot_type = "VEHICLE"
            self.hyperparams["minimum_history_length"] = 1
            self.hyperparams["maximum_history_length"] = int(self.hyperparams["history_sec"] // self.hyperparams["dt"])
            self.hyperparams["prediction_horizon"] = int(self.hyperparams["prediction_sec"] // self.hyperparams["dt"])
            self.hyperparams["use_map_encoding"] = self.hyperparams["map_encoding"]
            self.hyperparams["p_z_x_MLP_dims"] = None if self.hyperparams["p_z_x_MLP_dims"] == 0 else self.hyperparams["p_z_x_MLP_dims"]
            self.hyperparams["q_z_xy_MLP_dims"] = None if self.hyperparams["q_z_xy_MLP_dims"] == 0 else self.hyperparams["q_z_xy_MLP_dims"]
             

            if node_type.name in self.pred_state.keys():
                self.node_models_dict[node_type.name] = MultimodalGenerativeCVAE(
                    EnvClass(),
                     node_type, self.model_registrar, self.hyperparams, self.device, edge_types, 
                    log_writer=(WrappedLogWriter(self.log_writer) if self.log_writer is not None else None)
                    )

    def set_curr_iter(self, curr_iter):
        super().set_curr_iter(curr_iter)
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[str(node_type)].step_annealers()

    def train(self, inputs: Dict):
        batch: AgentBatch = inputs["agent_batch"]
        loss_weights: torch.Tensor = inputs["loss_weights"]

        # Make sure there is only one agent type
        agent_types = batch.agent_types()
        if len(agent_types) > 1:
            raise NotImplementedError("Mixing agent types for prediction in a batch is not supported.")
        agent_type: AgentType = agent_types[0]

        # Choose model for agent type
        model: MultimodalGenerativeCVAE = self.node_models_dict[agent_type.name]

        inputs, inputs_st, first_history_indices, labels, labels_st, neighbors, neighbors_edge_value = self.parse_batch(batch)

        # Compute training loss
        with nullcontext() if self.hyperparams["train_pred"] else torch.no_grad():
            loss, y_dist, (x_tensor, ) =  model.train_loss(
                inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot=None,
                   map=None,
                   prediction_horizon=self.hyperparams["prediction_horizon"],
                   ret_dist=True, loss_weights=loss_weights)

        return {"pred_dist": y_dist, "loss": loss}

    def validate(self, inputs: Dict):
        batch: AgentBatch = inputs['agent_batch']
        pred_ml, pred_dist, extra_output = self._run_prediction(batch, prediction_horizon=batch.agent_fut.shape[1])
        metrics = self.validation_metrics(pred_dist, pred_ml, batch.agent_fut)
        return {"pred_dist": pred_dist, "pred_ml": pred_ml, "metrics": metrics}

    def infer(self, inputs: Dict):
        batch: AgentBatch = inputs['agent_batch']
        pred_ml, pred_dist, extra_output = self._run_prediction(batch, prediction_horizon=batch.agent_fut.shape[1])
        return {"pred_dist": pred_dist, "pred_ml": pred_ml}

    def validation_metrics(self, pred_dist, pred_ml, agent_fut):
        # Compute default metrics
        metrics = compute_prediction_metrics(pred_ml, agent_fut[..., :2], y_dists=pred_dist)
        return metrics

    def _run_prediction(self, batch: AgentBatch, prediction_horizon: int):
      
        agent_types = batch.agent_types()
        if len(agent_types) > 1:
            raise NotImplementedError("Mixing agent types for prediction in a batch is not supported.")
        node_type: AgentType = agent_types[0]

        model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]
 
        inputs, inputs_st, first_history_indices, labels, labels_st, neighbors, neighbors_edge_value = self.parse_batch(batch)

        # Run forward pass, use the most likely latent mode.
        predictions = model.predict(
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot=None,
                map=None,            
                prediction_horizon=prediction_horizon,
                num_samples=1,
                z_mode=True,
                gmm_mode=True,
                full_dist=False,
                output_dists=False)

        # Run forward pass again, but this time output all modes of z.
        y_dists, _, extra_output  = model.predict(
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot=None,
                map=None,                        
                prediction_horizon=prediction_horizon,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                output_dists=True,
                output_extra=True)     
         
        return predictions, y_dists, extra_output

    def parse_batch(self, batch: AgentBatch):
        """ Converts batch input to trajectron input """
        extended_addition_filter: torch.Tensor = torch.tensor(self.hyperparams['edge_addition_filter'], dtype=torch.float)
        extended_addition_filter = torch.nn.functional.pad(extended_addition_filter, (0, batch.agent_hist.shape[1] - extended_addition_filter.shape[0]), mode='constant', value=1.0)

        if batch.history_pad_dir == PadDirection.AFTER:
            bs = batch.agent_hist.shape[0]
            x = convert_trajdata_hist_to_manual_hist(batch.agent_hist.cpu(), AgentType.VEHICLE, self.hyperparams["dt"])
            x = history_padding_last_to_first(x, batch.agent_hist_len.cpu())
            y = batch.agent_fut[..., :2].cpu()

            x_origin_batch = x[:, -1, :].cpu().numpy()
            x_st_t = torch.stack([
                standardized_manual_state(x[bi], x_origin_batch[bi], "VEHICLE", self.hyperparams["dt"], only2d=True) 
                for bi in range(bs)], dim=0)
            y_st_t = torch.stack([
                standardized_manual_state(y[bi], x_origin_batch[bi], "VEHICLE", self.hyperparams["dt"], only2d=True) 
                for bi in range(bs)], dim=0)

            first_history_index = batch.agent_hist.shape[1] - batch.agent_hist_len.cpu()
            neighbors_data_st = {("VEHICLE", "VEHICLE"): [[] for _ in range(bs)], ("VEHICLE", "PEDESTRIAN"): [[] for _ in range(bs)]}

            batch_neigh_hist = batch.neigh_hist.cpu()
            for bi in range(bs):
                for ni in range(batch.num_neigh[bi]):
                    nhist = batch_neigh_hist[bi, ni]
                    agent_type = AgentType(int(batch.neigh_types[bi, ni].cpu()))
                    nhist = convert_trajdata_hist_to_manual_hist(nhist, agent_type, self.hyperparams["dt"])
                    nhist = history_padding_last_to_first(nhist, batch.neigh_hist_len[bi, ni].cpu())
                    nhist_st = standardized_manual_state(nhist, x_origin_batch[bi], agent_type.name, self.hyperparams["dt"], only2d=False)
                    if self.hyperparams["pred_ego_indicator"] != "none":
                        # augment with ego indicator
                        nhist_st = torch.nn.functional.pad(nhist_st, (0, 1), value=float(ni == 0))
                    neighbors_data_st[("VEHICLE", agent_type.name)][bi].append(nhist_st)

            # Convert edge weights. They are the same for vehciles and pedestriancs.
            if self.hyperparams["dynamic_edges"] == "yes":
                edge_weight = [batch.extras["neigh_edge_weight"][bi][:batch.num_neigh[bi]] for bi in range(bs)]
                neighbors_edge_value = {("VEHICLE", "VEHICLE"): edge_weight, ("VEHICLE", "PEDESTRIAN"): edge_weight}
            else:
                neighbors_edge_value = {("VEHICLE", "VEHICLE"): None, ("VEHICLE", "PEDESTRIAN"): None}
        else:
            assert False

        # augment with ego indicator
        if self.hyperparams["pred_ego_indicator"] != "none":
            x = torch.nn.functional.pad(x, (0, 1), value=0.)
            x_st_t = torch.nn.functional.pad(x_st_t, (0, 1), value=0.)

        x = x.to(self.device)
        y = y.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)

        return x, x_st_t, first_history_index, y, y_st_t, neighbors_data_st, neighbors_edge_value


class TrajectronPredictorWithCacheData(TrajectronPredictor):

    def parse_batch(self, batch: AgentBatch):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
         neighbors_edge_value,
         robot_traj_st_t,
         map, neighbors_future_data, plan_data) = batch.extras["manual_inputs"]

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Restore encodings
        neighbors_data_st = restore(neighbors_data_st)
        neighbors_edge_value = restore(neighbors_edge_value)
        neighbors_future_data = restore(neighbors_future_data)
        plan_data = restore(plan_data)

        # augment with ego indicator
        if self.hyperparams["pred_ego_indicator"] != "none":
            ext_neighbor_states = {}
            for edge_type, node_neighbor_states in neighbors_data_st.items():
                ext_node_neighbor_states = []
                for batch_i, neighbor_state in enumerate(node_neighbor_states):
                    ego_ind = plan_data['most_relevant_idx'][batch_i].int()
                    if edge_type[1] == "VEHICLE":
                        ext_node_neighbor_states.append([torch.nn.functional.pad(neighbor_state[i], (0, 1), value=(ego_ind == i).float()) for i in range(len(neighbor_state))])
                    else:
                        ext_node_neighbor_states.append([torch.nn.functional.pad(neighbor_state[i], (0, 1), value=0.) for i in range(len(neighbor_state))])
                ext_neighbor_states[edge_type] = ext_node_neighbor_states
            neighbors_data_st = ext_neighbor_states

            x = torch.nn.functional.pad(x, (0, 1))
            x_st_t = torch.nn.functional.pad(x_st_t, (0, 1))   

        return x, x_st_t, first_history_index, y, y_st_t, neighbors_data_st, neighbors_edge_value


class WrappedLogWriter():
    def __init__(self, wandb_writer) -> None:
         self.wandb_writer = wandb_writer
    
    def add_scalar(self, name, value, iter):
        self.wandb_writer.log({name: value}, step=iter, commit=False)

    def add_histogram(self, *args):
        pass

    def add_image(self, *args):
        pass


def history_padding_last_to_first(hist_last, history_len):
    if hist_last.ndim > 2:
        # recursive to itself
        return torch.stack([
            history_padding_last_to_first(hist_last[bi], history_len[bi])
            for bi in range(hist_last.shape[0])], dim=0)
    else:
        hist_first = torch.full_like(hist_last, 0.)
        hist_first[hist_first.shape[0]-history_len:] = hist_last[:history_len]
        return hist_first