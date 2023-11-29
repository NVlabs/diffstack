import torch
import torch.nn as nn
import numpy as np
from diffstack.modules.module import Module, DataFormat, RunMode

from diffstack.utils.utils import traj_xyh_to_xyhv, removeprefix
import diffstack.utils.tensor_utils as TensorUtils
import diffstack.utils.geometry_utils as GeoUtils
from diffstack.utils.batch_utils import batch_utils
from diffstack.models.agentformer import AgentFormer
from trajdata.data_structures.batch import SceneBatch
from diffstack.modules.predictors.trajectron_utils.model.components import GMM2D
import diffstack.utils.tensor_utils as TensorUtils
from typing import Dict, Any


from diffstack.utils.loss_utils import (
    collision_loss,
)


from trajdata.data_structures import StateTensor, AgentType


from diffstack.utils.lane_utils import SimpleLaneRelation
from diffstack.utils.homotopy import (
    identify_pairwise_homotopy,
)
import diffstack.utils.metrics as Metrics


class AgentFormerTrafficModel(Module):
    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["scene_batch"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(
            [
                "mixed_pred_ml:validate",
                "mixed_pred_ml:infer",
                "metrics:train",
                "metrics:validate",
                "step_time",
            ]
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_predictor_prediction_loss"}

    def __init__(self, model_registrar, cfg, log_writer, device, input_mappings={}):
        super(AgentFormerTrafficModel, self).__init__(
            model_registrar, cfg, log_writer, device, input_mappings=input_mappings
        )

        self.bu = batch_utils(rasterize_mode="none")
        self.modality_shapes = self.bu.get_modality_shapes(cfg)
        # assert modality_shapes["image"][0] == 15
        self.nets = nn.ModuleDict()

        self.bu = batch_utils(parse=True, rasterize_mode="none")
        self.nets["policy"] = AgentFormer(cfg)
        self.cfg = cfg

        if "checkpoint" in cfg and cfg.checkpoint["enabled"]:
            checkpoint = torch.load(cfg.checkpoint["path"])
            predictor_dict = {
                removeprefix(k, "components.predictor."): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("components.predictor.")
            }
            self.load_state_dict(predictor_dict)

        self.device = device
        self.nets["policy"] = self.nets["policy"].to(self.device)

    def set_eval(self):
        self.nets["policy"].eval()

    def set_train(self):
        self.nets["policy"].train()

    def _run_forward(self, inputs: Dict, run_mode: RunMode, **kwargs) -> Dict:
        if isinstance(inputs["scene_batch"], dict) and "batch" in inputs["scene_batch"]:
            parsed_batch = self.bu.parse_batch(
                inputs["scene_batch"]["batch"].astype(torch.float)
            )
        else:
            parsed_batch = self.bu.parse_batch(
                inputs["scene_batch"].astype(torch.float)
            )

        # tic = torch_utils.tic(timer=self.hyperparams.debug.timer)
        parsed_batch_torch = {
            k: v.to(self.device)
            for k, v in parsed_batch.items()
            if isinstance(v, torch.Tensor)
        }
        parsed_batch.update(parsed_batch_torch)
        output = self.nets["policy"](
            parsed_batch, predict=(run_mode == RunMode.INFER), **kwargs
        )
        # torch_utils.toc(tic, name="prediction model", timer=self.hyperparams.debug.timer)

        # Convert to standard prediction output
        trajs_xyh = output["trajectories"]  # b, Ne, mode, N_agent, t, xyh
        trajs_xyh = trajs_xyh[:, 0]  # b, K, Na, T, 3
        trajs_xyh = trajs_xyh.transpose(1, 2)  # b, Na, K, T, 3
        trajs_xyh = trajs_xyh[:, :, None]  # # b, Na, S=1, K, T, 3
        # Infer velocity from xy
        dt = self.hyperparams["step_time"]  # hyperparams is AgentFormerConfig
        trajs_xyhv = traj_xyh_to_xyhv(trajs_xyh, dt)
        trajs_xyhv = StateTensor.from_array(trajs_xyhv, format="x,y,h,v_lon")

        log_probs = torch.log(output["p"])  # b, Ne, mode
        log_probs = log_probs[:, :1, None, :, None]  # b, 1, S, K, T=1

        mus = output["trajectories"][:, 0, ..., :2].permute(
            2, 0, 3, 1, 4
        )  # (Na,b,T,M,2)
        Na, bs, Tf, M = mus.shape[:4]
        log_pis = (
            torch.log(output["p"][None, :, None, 0])
            .repeat_interleave(Na, 0)
            .repeat_interleave(Tf, 2)
        )
        log_sigmas = torch.zeros_like(mus)
        corrs = torch.zeros_like(log_pis)
        y_dists = GMM2D(log_pis, mus, log_sigmas, corrs)
        if "state_trajectory" in output and output["state_trajectory"] is not None:
            state_traj = output["state_trajectory"][None, :, 0]
            # changing state order
            state_traj = torch.cat(
                [state_traj[..., :2], state_traj[..., 3:], state_traj[..., 2:3]], -1
            )
            output["pred_ml"] = state_traj
        else:
            output["pred_ml"] = output["trajectories"][None, :, 0]
        output["pred_dist"] = y_dists

        output.update(dict(data_batch=parsed_batch))
        if run_mode == RunMode.INFER:
            # Convert to standardized prediction output
            dt = self.hyperparams["step_time"]  # hyperparams is AgentFormerConfig
            mus_xyh = output["trajectories"]  # b, Ne, mode, N_agent, t, xyh
            log_pis = torch.log(output["p"])  # b, Ne, mode

            # pred_dist: GMM
            mus_xyh = mus_xyh[:, 0]  # b, mode, N_agent, t, xyh
            mus_xyh = mus_xyh.permute(2, 0, 1, 3, 4)  # (N_agent, b, mode, T, xyh)
            # Infer velocity from xy
            mus_xyhv = traj_xyh_to_xyhv(mus_xyh, dt)
            mus_xyhv = mus_xyhv.transpose(2, 3)  # (N_agent, b, T, mode, xyhv)

            # Currently we simply treat joint distribtion as agent-wise marginals.
            log_pis = (
                log_pis[:, 0]
                .reshape(1, log_pis.shape[0], 1, log_pis.shape[2])
                .repeat(mus_xyhv.shape[0], 1, mus_xyhv.shape[2], 1)
            )  # n, b, T, mode
            log_sigmas = torch.log(
                (
                    torch.arange(
                        1,
                        mus_xyhv.shape[2] + 1,
                        dtype=mus_xyhv.dtype,
                        device=mus_xyhv.device,
                    )
                    * dt
                )
                ** 2
                * 2
            )
            log_sigmas = log_sigmas.reshape(1, 1, mus_xyhv.shape[2], 1, 1).repeat(
                (mus_xyhv.shape[0], mus_xyhv.shape[1], 1, mus_xyhv.shape[3], 2)
            )
            corrs = 0.0 * torch.ones(
                mus_xyhv.shape[:-1], dtype=mus_xyhv.dtype, device=mus_xyhv.device
            )

            pred_dist_with_ego = GMM2D(log_pis, mus_xyhv, log_sigmas, corrs)

            # drop ego
            if isinstance(inputs["scene_batch"], SceneBatch):
                assert (inputs["scene_batch"].extras["robot_ind"] <= 0).all()
            pred_dist = GMM2D(log_pis, mus_xyhv, log_sigmas, corrs)

            ml_mode_ind = torch.argmax(log_pis, dim=-1)  # n, b, T
            # pred_ml = batch_select(mus_xyhv, ml_mode_ind, 3)  # n, b, T, 4
            pred_ml = mus_xyhv.permute(1, 3, 0, 2, 4)

            # Dummy single agent prediction.
            if isinstance(inputs["scene_batch"], SceneBatch):
                agent_fut = inputs["scene_batch"].agent_fut
            else:
                agent_fut = inputs["scene_batch"]["agent_fut"]
            pred_single = torch.full(
                [agent_fut.shape[0], 0, agent_fut.shape[2], 4],
                dtype=agent_fut.dtype,
                device=agent_fut.device,
                fill_value=torch.nan,
            )

            output["pred_dist"] = pred_dist
            output["pred_dist_with_ego"] = pred_dist_with_ego
            output["pred_ml"] = pred_ml
            output["pred_single"] = pred_single
            output["metrics"] = {}
        else:
            output["pred_dist"] = None
            output["pred_dist_with_ego"] = None
            output["pred_ml"] = None
            output["pred_single"] = None
            output["metrics"] = {}
        output["step_time"] = self.cfg["step_time"]
        return output

    def compute_losses(self, pred_batch, inputs):
        return self.nets["policy"].compute_losses(pred_batch, None)

    def compute_metrics(self, pred_batch, data_batch):
        EPS = 1e-3
        metrics = dict()
        # calculate GT lane mode and homotopy
        batch = pred_batch["data_batch"]
        fut_mask = batch["fut_mask"]
        mode_valid_flag = fut_mask.all(-1)
        B, N, Tf = batch["agent_fut"].shape[:3]
        traj = pred_batch["trajectories"].view(B, -1, N, Tf, 3)
        if True:
            lane_mask = batch["lane_mask"]
            fut_xy = batch["agent_fut"][..., :2]
            fut_sc = batch["agent_fut"][..., 6:8]
            fut_sc = GeoUtils.normalize_sc(fut_sc)
            fut_xysc = torch.cat([fut_xy, fut_sc], -1)

            end_points = fut_xysc[:, :, -1]  # Only look at final time for GT!
            lane_xyh = batch["lane_xyh"]
            M = lane_xyh.size(1)
            lane_xysc = torch.cat(
                [
                    lane_xyh[..., :2],
                    torch.sin(lane_xyh[..., 2:3]),
                    torch.cos(lane_xyh[..., 2:3]),
                ],
                -1,
            )

            GT_lane_mode, _ = SimpleLaneRelation.categorize_lane_relation_pts(
                end_points.reshape(B * N, 1, 4),
                lane_xysc.repeat_interleave(N, 0),
                fut_mask.any(-1).reshape(B * N, 1),
                lane_mask.repeat_interleave(N, 0),
                force_select=False,
            )
            # You could have two lanes that it is both on

            GT_lane_mode = GT_lane_mode.squeeze(-2).argmax(-1).reshape(B, N, M)
            GT_lane_mode = torch.cat(
                [GT_lane_mode, (GT_lane_mode == 0).all(-1, keepdim=True)], -1
            )

            angle_diff, GT_homotopy = identify_pairwise_homotopy(fut_xy, mask=fut_mask)
            GT_homotopy = GT_homotopy.type(torch.int64).reshape(B, N, N)
            pred_batch["GT_lane_mode"] = GT_lane_mode
            pred_batch["GT_homotopy"] = GT_homotopy

            pred_xysc = torch.cat(
                [traj[..., :2], torch.sin(traj[..., 2:3]), torch.cos(traj[..., 2:3])],
                -1,
            )
            DS = pred_xysc.size(1)

            end_points = pred_xysc[:, :, :, -1]  # Only look at final time

            pred_lane_mode, _ = SimpleLaneRelation.categorize_lane_relation_pts(
                end_points.reshape(B * N * DS, 1, 4),
                lane_xysc.repeat_interleave(N * DS, 0),
                fut_mask.any(-1).repeat_interleave(DS, 0).reshape(B * DS * N, 1),
                lane_mask.repeat_interleave(DS * N, 0),
                force_select=False,
            )
            # You could have two lanes that it is both on

            pred_lane_mode = pred_lane_mode.squeeze(-2).argmax(-1).reshape(B, DS, N, M)
            pred_lane_mode = torch.cat(
                [pred_lane_mode, (pred_lane_mode == 0).all(-1, keepdim=True)], -1
            )

            angle_diff, pred_homotopy = identify_pairwise_homotopy(
                pred_xysc[..., :2].view(B * DS, N, Tf, 2),
                mask=fut_mask.repeat_interleave(DS, 0),
            )
            pred_homotopy = pred_homotopy.type(torch.int64).reshape(B, DS, N, N)
            ML_homotopy_flag = (pred_homotopy[:, 0] == GT_homotopy).all(-1)

            ML_homotopy_flag.masked_fill_(torch.logical_not(mode_valid_flag), True)

            metrics["ML_homotopy_correct_rate"] = TensorUtils.to_numpy(
                (ML_homotopy_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            all_homotopy_flag = (pred_homotopy == GT_homotopy[:, None]).any(1).all(-1)
            all_homotopy_flag.masked_fill_(torch.logical_not(mode_valid_flag), True)
            metrics["all_homotopy_correct_rate"] = TensorUtils.to_numpy(
                (all_homotopy_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            ML_lane_mode_flag = (pred_lane_mode[:, 0] == GT_lane_mode).all(-1)
            all_lane_mode_flag = (
                (pred_lane_mode == GT_lane_mode[:, None]).any(1).all(-1)
            )
            metrics["ML_lane_mode_correct_rate"] = TensorUtils.to_numpy(
                (ML_lane_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            metrics["all_lane_mode_correct_rate"] = TensorUtils.to_numpy(
                (all_lane_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            ML_scene_mode_flag = (pred_homotopy[:, 0] == GT_homotopy).all(-1) & (
                pred_lane_mode[:, 0] == GT_lane_mode
            ).all(-1)
            metrics["ML_scene_mode_correct_rate"] = TensorUtils.to_numpy(
                (ML_scene_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            all_scene_mode_flag = (
                (pred_homotopy == GT_homotopy[:, None]).all(-1)
                & (pred_lane_mode == GT_lane_mode[:, None]).all(-1)
            ).any(1)
            metrics["all_scene_mode_correct_rate"] = TensorUtils.to_numpy(
                (all_scene_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()

        if "GT_homotopy" in pred_batch:
            # train/validation mode

            agent_fut, fut_mask, pred_traj = TensorUtils.to_numpy(
                (batch["agent_fut"], batch["fut_mask"], pred_batch["trajectories"])
            )
            pred_traj = pred_traj.reshape([B, -1, N, Tf, 3])
            if pred_traj.shape[-2] != agent_fut.shape[-2]:
                return metrics
            a2a_valid_flag = mode_valid_flag.unsqueeze(-1) * mode_valid_flag.unsqueeze(
                -2
            )

            extent = batch["extent"]
            DS = traj.size(1)
            pred_edges = self.bu.generate_edges(
                batch["agent_type"].repeat_interleave(DS, 0),
                extent.repeat_interleave(DS, 0),
                TensorUtils.join_dimensions(traj[..., :2], 0, 2),
                TensorUtils.join_dimensions(traj[..., 2:3], 0, 2),
                batch_first=True,
            )

            coll_loss, dis = collision_loss(pred_edges=pred_edges, return_dis=True)
            dis_padded = {
                k: v.nan_to_num(1.0).view(B, DS, -1, Tf) for k, v in dis.items()
            }
            edge_mask = {
                k: torch.logical_not(v.isnan().any(-1)).view(B, DS, -1)
                for k, v in dis.items()
            }
            for k in dis_padded:
                metrics["collision_rate_ML_" + k] = TensorUtils.to_numpy(
                    (dis_padded[k][:, 0] < 0).sum()
                    / (edge_mask[k][:, 0].sum() + EPS)
                    / Tf
                ).item()
                metrics[f"collision_rate_all_{DS}_mode_{k}"] = TensorUtils.to_numpy(
                    (dis_padded[k] < 0).sum() / (edge_mask[k].sum() + EPS) / Tf
                ).item()
            metrics["collision_rate_ML"] = TensorUtils.to_numpy(
                sum([(v[:, 0] < 0).sum() for v in dis_padded.values()])
                / Tf
                / (sum([edge_mask[k][:, 0].sum() for k in edge_mask]) + EPS)
            ).item()
            metrics[f"collision_rate_all_{DS}_mode"] = TensorUtils.to_numpy(
                sum([(v < 0).sum() for v in dis_padded.values()])
                / Tf
                / (sum([edge_mask[k].sum() for k in edge_mask]) + EPS)
            ).item()

            confidence = np.ones([B * N, 1])
            Nmode = pred_traj.shape[1]
            agent_type = TensorUtils.to_numpy(batch["agent_type"])
            vehicle_mask = (
                (agent_type == AgentType.VEHICLE)
                | (agent_type == AgentType.BICYCLE)
                | (agent_type == AgentType.MOTORCYCLE)
            )
            pedestrian_mask = agent_type == AgentType.PEDESTRIAN
            agent_mask = fut_mask.any(-1)
            dt = self.cfg.step_time
            for Tsecond in [3.0, 4.0, 5.0, 8.0]:
                if Tf < Tsecond / dt:
                    continue
                Tf_bar = int(Tsecond / dt)

                ADE = Metrics.batch_average_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 0, ..., :Tf_bar, :2].reshape(B * N, 1, Tf_bar, 2),
                    confidence,
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="mean",
                ).reshape(B, N)
                allADE = ADE.sum() / (agent_mask.sum() + EPS)
                vehADE = (ADE * vehicle_mask * agent_mask).sum() / (
                    (vehicle_mask * agent_mask).sum() + EPS
                )
                pedADE = (ADE * pedestrian_mask * agent_mask).sum() / (
                    (pedestrian_mask * agent_mask).sum() + EPS
                )
                FDE = Metrics.batch_final_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 0, ..., :Tf_bar, :2].reshape(B * N, 1, Tf_bar, 2),
                    confidence,
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="mean",
                ).reshape(B, N)

                allFDE = FDE.sum() / (agent_mask.sum() + EPS)
                vehFDE = (FDE * vehicle_mask * agent_mask).sum() / (
                    (vehicle_mask * agent_mask).sum() + EPS
                )
                pedFDE = (FDE * pedestrian_mask * agent_mask).sum() / (
                    (pedestrian_mask * agent_mask).sum() + EPS
                )
                metrics[f"ML_ADE@{Tsecond}"] = allADE
                metrics[f"ML_FDE@{Tsecond}"] = allFDE
                metrics[f"ML_vehicle_ADE@{Tsecond}"] = vehADE
                metrics[f"ML_vehicle_FDE@{Tsecond}"] = vehFDE
                metrics[f"oracle_pedestrian_ADE@{Tsecond}"] = pedADE
                metrics[f"oracle_pedestrian_FDE@{Tsecond}"] = pedFDE

                ADE = Metrics.batch_average_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[..., :Tf_bar, :2]
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(B * N, -1, Tf_bar, 2),
                    confidence.repeat(Nmode, 1) / Nmode,
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="oracle",
                ).reshape(B, N)
                FDE = Metrics.batch_final_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[..., :Tf_bar, :2]
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(B * N, -1, Tf_bar, 2),
                    confidence.repeat(Nmode, 1) / Nmode,
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="oracle",
                ).reshape(B, N)

                allADE = ADE.sum() / (agent_mask.sum() + EPS)
                vehADE = (ADE * vehicle_mask * agent_mask).sum() / (
                    (vehicle_mask * agent_mask).sum() + EPS
                )
                pedADE = (ADE * pedestrian_mask * agent_mask).sum() / (
                    (pedestrian_mask * agent_mask).sum() + EPS
                )
                allFDE = FDE.sum() / (agent_mask.sum() + EPS)
                vehFDE = (FDE * vehicle_mask * agent_mask).sum() / (
                    (vehicle_mask * agent_mask).sum() + EPS
                )
                pedFDE = (FDE * pedestrian_mask * agent_mask).sum() / (
                    (pedestrian_mask * agent_mask).sum() + EPS
                )

                metrics[f"min_ADE@{Tsecond}s"] = allADE
                metrics[f"min_FDE@{Tsecond}s"] = allFDE
                metrics[f"min_vehicle_ADE@{Tsecond}s"] = vehADE
                metrics[f"min_vehicle_FDE@{Tsecond}s"] = vehFDE
                metrics[f"min_pedestrian_ADE@{Tsecond}s"] = pedADE
                metrics[f"min_pedestrian_FDE@{Tsecond}s"] = pedFDE

        return metrics
