import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import os
from functools import partial
from typing import Dict
from collections import OrderedDict, defaultdict

from diffstack.dynamics.unicycle import Unicycle, Unicycle_xyvsc
from diffstack.dynamics.unicycle import Unicycle
from diffstack.modules.module import Module, DataFormat, RunMode

from diffstack.utils.utils import removeprefix
import diffstack.utils.tensor_utils as TensorUtils
import diffstack.utils.geometry_utils as GeoUtils
import diffstack.utils.metrics as Metrics
from diffstack.utils.geometry_utils import ratan2
import diffstack.utils.lane_utils as LaneUtils
from diffstack.utils.batch_utils import batch_utils
import diffstack.utils.model_utils as ModelUtils

from trajdata.utils.map_utils import LaneSegRelation
from diffstack.modules.predictors.trajectron_utils.model.components import GMM2D
from diffstack.utils.vis_utils import plot_scene_open_loop, animate_scene_open_loop

from diffstack.utils.loss_utils import collision_loss, loss_clip
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path


from diffstack.models.CTT import (
    CTT,
    TFvars,
    FeatureAxes,
    GNNedges,
)
from trajdata.data_structures import AgentType
from diffstack.utils.homotopy import (
    HomotopyType,
    identify_pairwise_homotopy,
    HOMOTOPY_THRESHOLD,
)
from bokeh.plotting import figure, output_file, save
from bokeh.models import Range1d
from bokeh.io import export_png


class CTTTrafficModel(Module):
    @property
    def input_format(self) -> DataFormat:
        return DataFormat(["scene_batch"])

    @property
    def output_format(self) -> DataFormat:
        return DataFormat(["pred", "pred_dist", "pred_ml", "sample_modes", "step_time"])

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_predictor_marginal_lm_loss"}

    def __init__(self, model_registrar, cfg, log_writer, device, input_mappings={}):
        super(CTTTrafficModel, self).__init__(
            model_registrar, cfg, log_writer, device, input_mappings=input_mappings
        )

        self.bu = batch_utils(rasterize_mode="none")
        self.modality_shapes = self.bu.get_modality_shapes(cfg)

        self.hist_lane_relation = LaneUtils.LaneRelationFromCfg(cfg.hist_lane_relation)
        self.fut_lane_relation = LaneUtils.LaneRelationFromCfg(cfg.fut_lane_relation)
        self.cfg = cfg
        self.nets = nn.ModuleDict()

        self.create_nets(cfg)
        self.fp16 = cfg.fp16 if "fp16" in cfg else False

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

        # setup loss functions

        self.lane_mode_loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.01
        )
        self.homotopy_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.01)
        self.joint_mode_loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.05
        )
        self.traj_loss = nn.MSELoss(reduction="none")
        self.unicycle_model = Unicycle(cfg.step_time)

    def create_nets(self, cfg):
        enc_var_axes = {
            TFvars.Agent_hist: "B,A,T,F",
            TFvars.Lane: "B,L,F",
        }
        agent_ntype = len(AgentType)
        auxdim_l = 4 * cfg.num_lane_pts  # (x,y,s,c) x num_pts

        agent_raw_dim = agent_ntype + 5  # agent type, l,w,v,a,r

        enc_attn_attributes = OrderedDict()
        # each attention operation is done on a pair of variables, e.g. (agent_hist, agent_hist), on a specific axis, e.g. T
        # to specify the recipe for factorized attention, we need to specify the following:
        # 1. the dimension of the edge feature
        # 2. the edge function (if any)
        # 3. the number of separate attention mechanisms (in case we need to separate the self-attention from the cross-attention or more generally, if we need to separate the attention for different types of edges)
        # 4. whether to use normalization for the embedding

        d_lm_hist = len(self.hist_lane_relation)
        d_lm_fut = len(self.fut_lane_relation)
        d_ll = len(LaneSegRelation)
        d_static = 2 + 1 + len(AgentType)
        a2a_edge_func = partial(
            ModelUtils.agent2agent_edge,
            scale=cfg.encoder.edge_scale,
            clip=cfg.encoder.edge_clip,
        )
        l2l_edge_func = partial(
            ModelUtils.lane2lane_edge,
            scale=cfg.encoder.edge_scale,
            clip=cfg.encoder.edge_clip,
        )
        if cfg.a2l_edge_type == "proj":
            a2l_edge_func = partial(
                ModelUtils.agent2lane_edge_proj,
                scale=cfg.encoder.edge_scale,
                clip=cfg.encoder.edge_clip,
            )
            a2l_edge_dim = cfg.edge_dim.a2l
        elif cfg.a2l_edge_type == "attn":
            self.nets["a2l_edge"] = ModelUtils.Agent2Lane_emb_attn(
                edge_dim=4,
                n_embd=cfg.a2l_n_embd,
                agent_feat_dim=d_static,
                xy_scale=cfg.encoder.edge_scale,
                xy_clip=cfg.encoder.edge_clip,
                Lmax=cfg.num_lane_pts,
            )
            a2l_edge_dim = cfg.a2l_n_embd
            a2l_edge_func = self.nets["a2l_edge"].forward

        self.edge_func = dict(a2a=a2a_edge_func, l2l=l2l_edge_func, a2l=a2l_edge_func)
        enc_attn_attributes[(TFvars.Agent_hist, TFvars.Agent_hist, FeatureAxes.T)] = (
            cfg.edge_dim.a2a,
            a2a_edge_func,
            1,
            False,
        )
        enc_attn_attributes[(TFvars.Agent_hist, TFvars.Agent_hist, FeatureAxes.A)] = (
            cfg.edge_dim.a2a,
            a2a_edge_func,
            cfg.attn_ntype.a2a,
            False,
        )
        enc_attn_attributes[(TFvars.Lane, TFvars.Lane, FeatureAxes.L)] = (
            cfg.edge_dim.l2l + d_ll,
            None,
            cfg.attn_ntype.l2l,
            False,
        )
        enc_attn_attributes[
            (TFvars.Agent_hist, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L))
        ] = (a2l_edge_dim + d_lm_hist, None, cfg.attn_ntype.a2l, False)

        enc_transformer_kwargs = dict(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            PE_mode=cfg.PE_mode,
            use_rpe_net=cfg.use_rpe_net,
            attn_attributes=enc_attn_attributes,
            var_axes=enc_var_axes,
            attn_pdrop=cfg.encoder.attn_pdrop,
            resid_pdrop=cfg.encoder.resid_pdrop,
            MAX_T=cfg.future_num_frames + cfg.history_num_frames + 2,
        )

        lane_margins_dim = 5

        # decoder design
        dec_attn_attributes = OrderedDict()

        dec_attn_attributes[
            (TFvars.Agent_future, TFvars.Agent_hist, (FeatureAxes.T, FeatureAxes.T))
        ] = (cfg.edge_dim.a2a, a2a_edge_func, 1, False)

        dec_attn_attributes[
            (TFvars.Agent_future, TFvars.Agent_future, FeatureAxes.A)
        ] = (cfg.edge_dim.a2a + len(HomotopyType) * 2, None, 2, False)
        dec_attn_attributes[
            (TFvars.Agent_future, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L))
        ] = (a2l_edge_dim + d_lm_fut + lane_margins_dim, None, 1, False)
        dec_var_axes = {
            TFvars.Agent_hist: "B,A,T,F",
            TFvars.Lane: "B,L,F",
            TFvars.Agent_future: "B,A,T,F",
        }
        dec_transformer_kwargs = dict(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            PE_mode=cfg.PE_mode,
            use_rpe_net=cfg.use_rpe_net,
            attn_attributes=dec_attn_attributes,
            var_axes=dec_var_axes,
            attn_pdrop=cfg.decoder.attn_pdrop,
            resid_pdrop=cfg.decoder.resid_pdrop,
        )
        # embedding functions of raw variables
        embed_funcs = {
            TFvars.Agent_hist: ModelUtils.Agent_emb(agent_raw_dim, cfg.n_embd),
            TFvars.Lane: ModelUtils.Lane_emb(
                auxdim_l,
                cfg.n_embd,
                xy_scale=cfg.encoder.edge_scale,
                xy_clip=cfg.encoder.edge_clip,
            ),
            TFvars.Agent_future: ModelUtils.Agent_emb(agent_raw_dim, cfg.n_embd),
            GNNedges.Agenthist2Agenthist: ModelUtils.Agent2Agent_emb(
                cfg.edge_dim.a2a,
                cfg.n_embd,
                xy_scale=cfg.encoder.edge_scale,
                xy_clip=cfg.encoder.edge_clip,
            ),
            GNNedges.Agentfuture2Agentfuture: ModelUtils.Agent2Agent_emb(
                cfg.edge_dim.a2a + len(HomotopyType) * 2,
                cfg.n_embd,
                xy_scale=cfg.encoder.edge_scale,
                xy_clip=cfg.encoder.edge_clip,
            ),
        }
        if cfg.a2l_edge_type == "proj":
            embed_funcs.update(
                {
                    GNNedges.Agenthist2Lane: ModelUtils.Agent2Lane_emb_proj(
                        cfg.edge_dim.a2l,
                        cfg.n_embd,
                        xy_scale=cfg.encoder.edge_scale,
                        xy_clip=cfg.encoder.edge_clip,
                    ),
                    GNNedges.Agentfuture2Lane: ModelUtils.Agent2Lane_emb_proj(
                        cfg.edge_dim.a2l + d_lm_fut + lane_margins_dim,
                        cfg.n_embd,
                        xy_scale=cfg.encoder.edge_scale,
                        xy_clip=cfg.encoder.edge_clip,
                    ),
                }
            )
        else:
            embed_funcs.update(
                {
                    GNNedges.Agenthist2Lane: ModelUtils.Agent2Lane_emb_attn(
                        4,
                        cfg.a2l_n_embd,
                        d_static,
                        output_dim=cfg.n_embd,
                        xy_scale=cfg.encoder.edge_scale,
                        xy_clip=cfg.encoder.edge_clip,
                    ),
                    GNNedges.Agentfuture2Lane: ModelUtils.Agent2Lane_emb_attn(
                        4,
                        cfg.a2l_n_embd,
                        d_static,
                        output_dim=cfg.n_embd,
                        aux_edge_dim=d_lm_fut + lane_margins_dim,
                        xy_scale=cfg.encoder.edge_scale,
                        xy_clip=cfg.encoder.edge_clip,
                    ),
                }
            )

        enc_GNN_attributes = OrderedDict()
        enc_GNN_attributes[(GNNedges.Agenthist2Agenthist, "edge", None)] = (
            [cfg.n_embd * 2],
            None,
            None,
            None,
        )
        enc_GNN_attributes[(GNNedges.Agenthist2Lane, "edge", None)] = (
            [cfg.n_embd * 2],
            None,
            None,
            None,
        )
        enc_GNN_attributes[(GNNedges.Agenthist2Lane, "node", TFvars.Agent_hist)] = (
            [cfg.n_embd * 2],
            None,
            cfg.encoder.pooling,
            None,
        )
        enc_GNN_attributes[
            (GNNedges.Agenthist2Agenthist, "node", TFvars.Agent_hist)
        ] = ([cfg.n_embd * 2], None, cfg.encoder.pooling, None)

        node_n_embd = defaultdict(lambda: cfg.n_embd)
        edge_n_embd = defaultdict(lambda: cfg.n_embd)

        enc_edge_var = {
            GNNedges.Agenthist2Agenthist: (TFvars.Agent_hist, TFvars.Agent_hist),
            GNNedges.Agenthist2Lane: (TFvars.Agent_hist, TFvars.Lane),
        }
        dec_edge_var = {
            GNNedges.Agenthist2Agenthist: (TFvars.Agent_hist, TFvars.Agent_hist),
            GNNedges.Agenthist2Lane: (TFvars.Agent_hist, TFvars.Lane),
            GNNedges.Agentfuture2Agentfuture: (
                TFvars.Agent_future,
                TFvars.Agent_future,
            ),
            GNNedges.Agentfuture2Lane: (TFvars.Agent_future, TFvars.Lane),
        }
        enc_GNN_kwargs = dict(
            var_axes=enc_var_axes,
            GNN_attributes=enc_GNN_attributes,
            node_n_embd=node_n_embd,
            edge_n_embd=edge_n_embd,
            edge_var=enc_edge_var,
        )

        JM_GNN_attributes = OrderedDict()
        JM_GNN_attributes[(GNNedges.Agenthist2Agenthist, "edge", None)] = (
            [cfg.n_embd * 2],
            None,
            None,
            None,
        )
        JM_GNN_attributes[(GNNedges.Agenthist2Lane, "edge", None)] = (
            [cfg.n_embd * 2],
            None,
            None,
            None,
        )
        JM_GNN_attributes[(GNNedges.Agenthist2Lane, "node", TFvars.Agent_hist)] = (
            [cfg.n_embd * 2],
            None,
            cfg.encoder.pooling,
            None,
        )
        JM_GNN_attributes[(GNNedges.Agenthist2Agenthist, "node", TFvars.Agent_hist)] = (
            [cfg.n_embd * 2],
            None,
            cfg.encoder.pooling,
            None,
        )

        JM_edge_n_embd = {
            GNNedges.Agenthist2Agenthist: cfg.n_embd + cfg.encoder.mode_embed_dim,
            GNNedges.Agenthist2Lane: cfg.n_embd + cfg.encoder.mode_embed_dim,
        }

        JM_GNN_kwargs = dict(
            var_axes=enc_var_axes,
            GNN_attributes=JM_GNN_attributes,
            node_n_embd=node_n_embd,
            edge_n_embd=JM_edge_n_embd,
            edge_var=enc_edge_var,
        )

        enc_output_params = dict(
            pooling_T=cfg.encoder.pooling,
            Th=cfg.history_num_frames + 1,
            lane_mode=dict(
                n_head=4,
                hidden_dim=[cfg.n_embd],
            ),
            homotopy=dict(
                n_head=4,
                hidden_dim=[cfg.n_embd],
            ),
            joint_mode=dict(
                n_head=4,
                jm_GNN_nblock=cfg.encoder.jm_GNN_nblock,
                GNN_kwargs=JM_GNN_kwargs,
                num_joint_samples=cfg.encoder.num_joint_samples,
                num_joint_factors=cfg.encoder.num_joint_factors,
            ),
            mode_embed_dim=cfg.encoder.mode_embed_dim,
            null_lane_mode=cfg.encoder.null_lane_mode,
            PE_mode=cfg.PE_mode,
        )
        self.null_lane_mode = cfg.encoder.null_lane_mode

        dec_GNN_attributes = OrderedDict()
        if cfg.decoder.GNN_enabled:
            dec_GNN_attributes[(GNNedges.Agentfuture2Agentfuture, "edge", None)] = (
                [cfg.n_embd * 2],
                None,
                None,
                None,
            )
            dec_GNN_attributes[(GNNedges.Agentfuture2Lane, "edge", None)] = (
                [cfg.n_embd * 2],
                None,
                None,
                None,
            )
            dec_GNN_attributes[
                (GNNedges.Agentfuture2Lane, "node", TFvars.Agent_future)
            ] = ([cfg.n_embd * 2], None, cfg.decoder.pooling, None)
            dec_GNN_attributes[
                (GNNedges.Agentfuture2Agentfuture, "node", TFvars.Agent_future)
            ] = ([cfg.n_embd * 2], None, cfg.decoder.pooling, None)

        dec_GNN_kwargs = dict(
            var_axes=dec_var_axes,
            GNN_attributes=dec_GNN_attributes,
            node_n_embd=node_n_embd,
            edge_n_embd=edge_n_embd,
            edge_var=dec_edge_var,
        )

        dyn = dict()
        name_type_table = {
            "vehicle": AgentType.VEHICLE,
            "pedestrian": AgentType.PEDESTRIAN,
            "bicycle": AgentType.BICYCLE,
            "motorcycle": AgentType.MOTORCYCLE,
        }

        for k, v in cfg.decoder.dyn.items():
            if v == "unicycle":
                dyn[name_type_table[k]] = Unicycle(cfg.step_time)
            elif v == "unicycle_xyvsc":
                dyn[name_type_table[k]] = Unicycle_xyvsc(cfg.step_time)
            elif v == "DI_unicycle":
                dyn[name_type_table[k]] = Unicycle(cfg.step_time, max_steer=1e3)
            else:
                dyn[name_type_table[k]] = None

        self.weighted_consistency_loss = cfg.weighted_consistency_loss
        dec_output_params = dict(
            arch=cfg.decoder.arch,
            dyn=dyn,
            num_layers=cfg.decoder.num_layers,
            lstm_hidden_size=cfg.decoder.lstm_hidden_size,
            mlp_hidden_dims=cfg.decoder.mlp_hidden_dims,
            traj_dim=cfg.decoder.traj_dim,
            dt=cfg.step_time,
            Tf=cfg.future_num_frames,
            decode_num_modes=cfg.decoder.decode_num_modes,
            AR_step_size=cfg.decoder.AR_step_size,
            AR_update_mode=cfg.decoder.AR_update_mode,
            LR_sample_hack=cfg.LR_sample_hack,
            dec_rounds=cfg.decoder.dec_rounds,
        )

        assert (
            cfg.decoder.AR_step_size == 1
        )  # for now, we only support AR_step_size=1 for non-auto-regressive mode
        self.Tf_mode = cfg.future_num_frames
        self.nets["policy"] = CTT(
            n_embd=cfg.n_embd,
            embed_funcs=embed_funcs,
            enc_nblock=cfg.enc_nblock,
            dec_nblock=cfg.dec_nblock,
            enc_transformer_kwargs=enc_transformer_kwargs,
            enc_GNN_kwargs=enc_GNN_kwargs,
            dec_transformer_kwargs=dec_transformer_kwargs,
            dec_GNN_kwargs=dec_GNN_kwargs,
            enc_output_params=enc_output_params,
            dec_output_params=dec_output_params,
            hist_lane_relation=self.hist_lane_relation,
            fut_lane_relation=self.fut_lane_relation,
            max_joint_cardinality=self.cfg.max_joint_cardinality,
            classify_a2l_4all_lanes=self.cfg.classify_a2l_4all_lanes,
            edge_func=self.edge_func,
        )

    def set_eval(self):
        self.nets["policy"].eval()

    def set_train(self):
        self.nets["policy"].train()

    def _run_forward(self, inputs: Dict, run_mode: RunMode, **kwargs) -> Dict:
        if "parsed_batch" in inputs:
            parsed_batch = inputs["parsed_batch"]
        else:
            if (
                isinstance(inputs["scene_batch"], dict)
                and "batch" in inputs["scene_batch"]
            ):
                parsed_batch = self.bu.parse_batch(
                    inputs["scene_batch"]["batch"].astype(torch.float)
                )
            else:
                parsed_batch = self.bu.parse_batch(
                    inputs["scene_batch"].astype(torch.float)
                )

        if self.fp16:
            for k, v in parsed_batch.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    parsed_batch[k] = v.half()

        # tic = torch_utils.tic(timer=self.hyperparams.debug.timer)
        parsed_batch_torch = {
            k: v.to(self.device)
            for k, v in parsed_batch.items()
            if isinstance(v, torch.Tensor)
        }

        parsed_batch.update(parsed_batch_torch)
        dt = parsed_batch["dt"][0].item()
        B, N, Th = parsed_batch["agent_hist"].shape[:3]

        Tf = parsed_batch["agent_fut"].size(2)
        device = parsed_batch["agent_hist"].device
        agent_type = (
            F.one_hot(
                parsed_batch["agent_type"].masked_fill(
                    parsed_batch["agent_type"] < 0, 0
                ),
                len(AgentType),
            )
            .float()
            .to(device)
        )
        agent_type.masked_fill_((parsed_batch["agent_type"] < 0).unsqueeze(-1), 0)
        agent_size = parsed_batch["extent"][..., :2]
        static_feature = torch.cat([agent_size, agent_type], dim=-1)
        hist_mask = parsed_batch["hist_mask"]
        static_feature_tiled_h = static_feature.unsqueeze(2).repeat_interleave(
            Th, 2
        ) * hist_mask.unsqueeze(-1)

        hist_xy = parsed_batch["agent_hist"][..., :2]
        hist_yaw = torch.arctan2(
            parsed_batch["agent_hist"][..., 6:7], parsed_batch["agent_hist"][..., 7:8]
        )
        hist_sc = parsed_batch["agent_hist"][..., 6:8]
        # normalize hist_sc
        hist_sc = GeoUtils.normalize_sc(hist_sc)

        x_hist_uni = self.unicycle_model.get_state(hist_xy, hist_yaw, hist_mask)
        u_mask = hist_mask[..., :-1] * hist_mask[..., 1:]
        u_hist_uni = self.unicycle_model.inverse_dyn(
            x_hist_uni[..., :-1, :], x_hist_uni[..., 1:, :], mask=u_mask
        )
        u_hist_uni = torch.cat([u_hist_uni[..., :1, :], u_hist_uni], dim=-2)

        hist_v = self.unicycle_model.state2vel(x_hist_uni)
        hist_acce = u_hist_uni[..., :1]

        hist_xyvsc = torch.cat([hist_xy, hist_v, hist_sc], dim=-1)
        hist_xysc = torch.cat([hist_xy, hist_sc], dim=-1)
        hist_h = ratan2(hist_sc[..., :1], hist_sc[..., 1:2])
        hist_r = (GeoUtils.round_2pi(hist_h[..., 1:, :] - hist_h[..., :-1, :])) / dt
        hist_r = torch.cat([hist_r[..., 0:1, :], hist_r], dim=-2)
        hist_feature = torch.cat(
            [hist_v, hist_acce, hist_r, static_feature_tiled_h], dim=-1
        )
        lane_xyh = parsed_batch["lane_xyh"]
        M = lane_xyh.size(1)
        lane_xysc = torch.cat(
            [
                lane_xyh[..., :2],
                torch.sin(lane_xyh[..., 2:3]),
                torch.cos(lane_xyh[..., 2:3]),
            ],
            -1,
        )
        lane_adj = parsed_batch["lane_adj"].type(torch.int64)
        lane_mask = parsed_batch["lane_mask"]

        raw_vars = {
            TFvars.Agent_hist: hist_feature,
            TFvars.Lane: lane_xysc.reshape(B, M, -1),
        }
        hist_aux = torch.cat([hist_xyvsc, static_feature_tiled_h], -1)
        lane_aux = lane_xysc.view(B, M, -1)

        aux_xs = {
            TFvars.Agent_hist: hist_aux,
            TFvars.Lane: lane_aux,
        }
        a2l_edge = self.edge_func["a2l"](
            hist_aux.transpose(1, 2).reshape(B * Th, N, -1),
            lane_aux.repeat_interleave(Th, 0),
        )

        hist_lane_agent_flag, _ = self.hist_lane_relation.categorize_lane_relation_pts(
            TensorUtils.join_dimensions(hist_xysc, 0, 2),
            lane_xysc.repeat_interleave(N, 0),
            TensorUtils.join_dimensions(hist_mask, 0, 2),
            lane_mask.repeat_interleave(N, 0),
        )

        hist_lane_agent_flag = hist_lane_agent_flag.view(B, N, M, Th, -1)
        a2l_edge = torch.cat(
            [
                a2l_edge,
                hist_lane_agent_flag.permute(0, 3, 1, 2, 4).reshape(B * Th, N, M, -1),
            ],
            -1,
        )
        l2l_edge = self.edge_func["l2l"](lane_aux, lane_aux)
        l2l_edge = torch.cat([l2l_edge, F.one_hot(lane_adj, len(LaneSegRelation))], -1)
        agent_hist_mask = parsed_batch["hist_mask"]

        # cross_masks = {
        #     (TFvars.Agent_hist,TFvars.Lane,(FeatureAxes.A,FeatureAxes.L)): [agent_lane_mask1,agent_lane_mask2],
        # }
        cross_masks = dict()
        var_masks = {
            TFvars.Agent_hist: agent_hist_mask.float(),
            TFvars.Lane: lane_mask.float(),
        }
        enc_edges = {
            (TFvars.Agent_hist, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L)): a2l_edge,
            (TFvars.Lane, TFvars.Lane, FeatureAxes.L): l2l_edge,
        }

        frame_indices = {
            TFvars.Agent_hist: torch.arange(Th, device=device)[None, None, :].repeat(
                B, N, 1
            ),
        }

        if (
            "agent_fut" in parsed_batch
            and "fut_mask" in parsed_batch
            and parsed_batch["fut_mask"].any()
            and run_mode in [RunMode.TRAIN, RunMode.VALIDATE]
        ):
            fut_xy = parsed_batch["agent_fut"][..., :2]
            fut_sc = parsed_batch["agent_fut"][..., 6:8]
            fut_sc = GeoUtils.normalize_sc(fut_sc)
            fut_xysc = torch.cat([fut_xy, fut_sc], -1)
            fut_mask = parsed_batch["fut_mask"]

            mode_valid_flag = fut_mask.all(-1)
            end_points = fut_xysc[:, :, -1]  # Only look at final time for GT!

            GT_lane_mode, _ = self.fut_lane_relation.categorize_lane_relation_pts(
                end_points.reshape(B * N, 1, 4),
                lane_xysc.repeat_interleave(N, 0),
                fut_mask.any(-1).reshape(B * N, 1),
                lane_mask.repeat_interleave(N, 0),
                force_select=not self.null_lane_mode,
            )
            # You could have two lanes that it is both on

            GT_lane_mode = GT_lane_mode.squeeze(-2).argmax(-1).reshape(B, N, M)
            if self.null_lane_mode:
                GT_lane_mode = torch.cat(
                    [GT_lane_mode, (GT_lane_mode == 0).all(-1, keepdim=True)], -1
                )

            angle_diff, GT_homotopy = identify_pairwise_homotopy(fut_xy, mask=fut_mask)
            GT_homotopy = GT_homotopy.type(torch.int64).reshape(B, N, N)
        else:
            GT_lane_mode = None
            GT_homotopy = None
        center_from_agents = parsed_batch["center_from_agents"]
        if run_mode == RunMode.INFER:
            num_samples = kwargs.get("num_samples", 10)
        else:
            num_samples = None
        vars, mode_pred = self.nets["policy"](
            raw_vars,
            aux_xs,
            var_masks,
            cross_masks,
            frame_indices,
            agent_type=agent_type,
            enc_edges=enc_edges,
            GT_lane_mode=GT_lane_mode,
            GT_homotopy=GT_homotopy,
            center_from_agents=center_from_agents,
            num_samples=num_samples,
        )
        output_keys = [
            "trajectories",
            "inputs",
            "states",
            "input_violation",
            "jerk",
            "type_mask",
        ]
        output = {k: v for k, v in vars.items() if k in output_keys}
        if run_mode == RunMode.INFER:
            B, num_samples, N, Tf = output["trajectories"].shape[:-1]
            log_pis = (
                mode_pred["joint_pred"]
                .view(1, B, 1, num_samples)
                .expand(N, B, Tf, num_samples)
            )
            if "states" in output:
                state_xyhv = torch.cat(
                    [
                        output["states"][..., :2],
                        output["states"][..., 3:4],
                        output["states"][..., 2:3],
                    ],
                    -1,
                )
            else:
                # calculate velocity from trajectories
                pred_vel = self.unicycle_model.calculate_vel(
                    output["trajectories"][..., :2],
                    output["trajectories"][..., 2:],
                    hist_mask.any(-1)[:, None, :, None]
                    .repeat_interleave(num_samples, 1)
                    .repeat_interleave(Tf, -1),
                )
                state_xyhv = torch.cat([output["trajectories"], pred_vel], -1)
            mus_xyhv = state_xyhv.permute(2, 0, 3, 1, 4)
            log_sigmas = torch.zeros_like(mus_xyhv)
            corrs = torch.zeros_like(mus_xyhv[..., 0])
            pred_dist = GMM2D(log_pis, mus_xyhv, log_sigmas, corrs)
            output["pred_dist"] = pred_dist
            output["pred_ml"] = state_xyhv
            # output["pred_single"] = torch.full([B,0,Tf,4],np.nan,device=device)
            output["sample_modes"] = dict(
                lane_mode=mode_pred["lane_mode_sample"],
                homotopy=mode_pred["homotopy_sample"],
            )
            # print(output["inputs"][...,1].abs().max())
        else:
            if "trajectories" in output:
                # pad trajectories to Tf in case of anomaly
                if output["trajectories"].shape[-2] != Tf:
                    breakpoint()
                    dynamic_outputs = {
                        k: v
                        for k, v in output.items()
                        if k
                        in [
                            "inputs",
                            "states",
                            "input_violation",
                            "jerk",
                            "trajectories",
                        ]
                    }
                    if output["trajectories"].shape[-2] < Tf:
                        func = (
                            lambda x: x
                            if x.shape[-2] == Tf
                            else torch.cat(
                                [
                                    x,
                                    x[..., -1:, :].repeat_interleave(
                                        Tf - x.shape[-2], -2
                                    ),
                                ],
                                -2,
                            )
                        )
                    elif output["trajectories"].shape[-2] > Tf:
                        func = lambda x: x[..., :Tf, :]
                    dynamic_outputs = TensorUtils.recursive_dict_list_tuple_apply(
                        dynamic_outputs,
                        {
                            torch.Tensor: func,
                            type(None): lambda x: x,
                        },
                    )
                    output.update(dynamic_outputs)
                dec_xysc = torch.cat(
                    [
                        output["trajectories"][..., :2],
                        torch.sin(output["trajectories"][..., 2:3]),
                        torch.cos(output["trajectories"][..., 2:3]),
                    ],
                    -1,
                )
                if dec_xysc.shape[-2] != Tf:
                    pass
                else:
                    if dec_xysc.ndim == 4:
                        DS = 1
                        dec_xysc = dec_xysc.unsqueeze(1)
                    elif dec_xysc.ndim == 5:
                        DS = dec_xysc.size(1)  # decode sample

                    (
                        dec_lm,
                        dec_lm_margin,
                    ) = self.fut_lane_relation.categorize_lane_relation_pts(
                        dec_xysc[:, :, :, -1].view(B * DS * N, 1, 4),
                        lane_xysc.repeat_interleave(N * DS, 0),
                        fut_mask.repeat_interleave(DS, 0)
                        .any(-1)
                        .reshape(B * DS * N, 1),
                        lane_mask.repeat_interleave(N * DS, 0),
                        force_select=False,
                        force_unique=False,
                        const_override=dict(Y_near_thresh=1.0, X_rear_thresh=5.0),
                    )

                    y_dev, psi_dev = LaneUtils.get_ypsi_dev(
                        dec_xysc.view(B * DS * N, -1, 4),
                        lane_xysc.repeat_interleave(N * DS, 0),
                    )

                    dec_angle_diff, dec_homotopy = identify_pairwise_homotopy(
                        dec_xysc[..., :2].reshape(B * DS, N, Tf, -1),
                        mask=fut_mask.any(-1).repeat_interleave(DS, 0),
                    )
                    dec_homotopy_margin = torch.stack(
                        [
                            HOMOTOPY_THRESHOLD - dec_angle_diff.abs(),
                            -dec_angle_diff - HOMOTOPY_THRESHOLD,
                            dec_angle_diff - HOMOTOPY_THRESHOLD,
                        ],
                        -1,
                    )
                    output["dec_lm_margin"] = dec_lm_margin.view(B, DS, N, M, -1)
                    output["dec_lane_y_dev"] = y_dev.view(B, DS, N, M, Tf, -1)
                    output["dec_lane_psi_dev"] = psi_dev.view(B, DS, N, M, Tf, -1)
                    output["dec_homotopy_margin"] = dec_homotopy_margin.view(
                        B, DS, N, N, -1
                    )
                    output["dec_lm"] = dec_lm.view(B, DS, N, M, -1)
                    output["dec_homotopy"] = dec_homotopy.view(B, DS, N, N, -1)

            output.update(mode_pred)
            output["GT_lane_mode"] = GT_lane_mode
            output["GT_homotopy"] = GT_homotopy
            output["mode_valid_flag"] = mode_valid_flag
            output["batch"] = parsed_batch
        output["step_time"] = self.cfg.step_time
        return output

    def compute_metrics(self, pred_batch, data_batch):
        EPS = 1e-3
        metrics = dict()
        if "GT_homotopy" in pred_batch:
            # train/validation mode
            batch = pred_batch["batch"]
            agent_fut, fut_mask, pred_traj = TensorUtils.to_numpy(
                (batch["agent_fut"], batch["fut_mask"], pred_batch["trajectories"])
            )
            if pred_traj.shape[-2] != agent_fut.shape[-2]:
                return metrics
            mode_valid_flag = pred_batch["mode_valid_flag"]
            a2a_valid_flag = mode_valid_flag.unsqueeze(-1) * mode_valid_flag.unsqueeze(
                -2
            )

            metrics["joint_mode_accuracy"] = TensorUtils.to_numpy(
                torch.softmax(pred_batch["joint_pred"], dim=1)[:, 0].mean()
            ).item()
            metrics["joint_mode_correct_rate"] = TensorUtils.to_numpy(
                (
                    pred_batch["joint_pred"][:, 0]
                    == pred_batch["joint_pred"].max(dim=1)[0]
                )
                .float()
                .mean()
            ).item()

            lane_mode_correct_flag = pred_batch["GT_lane_mode"].argmax(-1) == (
                pred_batch["lane_mode_pred"].argmax(-1).squeeze(-1)
            )
            metrics["pred_lane_mode_correct_rate"] = TensorUtils.to_numpy(
                (lane_mode_correct_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            metrics["lane_mode_accuracy"] = TensorUtils.to_numpy(
                (
                    (
                        torch.softmax(pred_batch["lane_mode_pred"], dim=-1).squeeze(-2)
                        * pred_batch["GT_lane_mode"]
                    ).sum(-1)
                    * mode_valid_flag
                ).sum()
                / mode_valid_flag.sum()
            ).item()

            homotopy_correct_flag = pred_batch["GT_homotopy"] == (
                pred_batch["homotopy_pred"].argmax(-1)
            )

            metrics["pred_homotopy_correct_rate"] = TensorUtils.to_numpy(
                (homotopy_correct_flag * a2a_valid_flag).sum() / a2a_valid_flag.sum()
            ).item()
            metrics["homotopy_accuracy"] = TensorUtils.to_numpy(
                (
                    (
                        torch.softmax(pred_batch["homotopy_pred"], dim=-1)
                        * F.one_hot(pred_batch["GT_homotopy"], len(HomotopyType))
                    ).sum(-1)
                    * a2a_valid_flag
                ).sum()
                / a2a_valid_flag.sum()
            ).item()

            B, N, Tf = agent_fut.shape[:3]
            extent = batch["extent"]
            traj = pred_batch["trajectories"]
            DS = traj.size(1)
            GT_homotopy = pred_batch["GT_homotopy"]
            GT_lane_mode = pred_batch["GT_lane_mode"]
            pred_xysc = torch.cat(
                [traj[..., :2], torch.sin(traj[..., 2:3]), torch.cos(traj[..., 2:3])],
                -1,
            ).view(B, -1, N, Tf, 4)
            end_points = pred_xysc[:, :, :, -1]  # Only look at final time
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
            pred_lane_mode, _ = self.fut_lane_relation.categorize_lane_relation_pts(
                end_points.reshape(B * N * DS, 1, 4),
                lane_xysc.repeat_interleave(N * DS, 0),
                batch["fut_mask"]
                .any(-1)
                .repeat_interleave(DS, 0)
                .reshape(B * DS * N, 1),
                batch["lane_mask"].repeat_interleave(DS * N, 0),
                force_select=False,
            )
            # You could have two lanes that it is both on

            pred_lane_mode = pred_lane_mode.squeeze(-2).argmax(-1).reshape(B, DS, N, M)
            pred_lane_mode = torch.cat(
                [pred_lane_mode, (pred_lane_mode == 0).all(-1, keepdim=True)], -1
            )

            angle_diff, pred_homotopy = identify_pairwise_homotopy(
                pred_xysc[..., :2].view(B * DS, N, Tf, 2),
                mask=batch["fut_mask"].repeat_interleave(DS, 0),
            )
            pred_homotopy = pred_homotopy.type(torch.int64).reshape(B, DS, N, N)
            ML_homotopy_flag = (pred_homotopy[:, 1] == GT_homotopy).all(-1)

            ML_homotopy_flag.masked_fill_(torch.logical_not(mode_valid_flag), True)
            metrics["ML_homotopy_correct_rate"] = TensorUtils.to_numpy(
                (ML_homotopy_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            all_homotopy_flag = (
                (pred_homotopy[:, 1:] == GT_homotopy[:, None]).any(1).all(-1)
            )
            all_homotopy_flag.masked_fill_(torch.logical_not(mode_valid_flag), True)
            metrics["all_homotopy_correct_rate"] = TensorUtils.to_numpy(
                (all_homotopy_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            ML_lane_mode_flag = (pred_lane_mode[:, 1] == GT_lane_mode).all(-1)
            all_lane_mode_flag = (
                (pred_lane_mode[:, 1:] == GT_lane_mode[:, None]).any(1).all(-1)
            )
            metrics["ML_lane_mode_correct_rate"] = TensorUtils.to_numpy(
                (ML_lane_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            metrics["all_lane_mode_correct_rate"] = TensorUtils.to_numpy(
                (all_lane_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            ML_scene_mode_flag = (pred_homotopy[:, 1] == GT_homotopy).all(-1) & (
                pred_lane_mode[:, 1] == GT_lane_mode
            ).all(-1)
            metrics["ML_scene_mode_correct_rate"] = TensorUtils.to_numpy(
                (ML_scene_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            all_scene_mode_flag = (
                (pred_homotopy[:, 1:] == GT_homotopy[:, None]).all(-1)
                & (pred_lane_mode[:, 1:] == GT_lane_mode[:, None]).all(-1)
            ).any(1)
            metrics["all_scene_mode_correct_rate"] = TensorUtils.to_numpy(
                (all_scene_mode_flag * mode_valid_flag).sum() / mode_valid_flag.sum()
            ).item()
            pred_edges = self.bu.generate_edges(
                batch["agent_type"].repeat_interleave(DS, 0),
                extent.repeat_interleave(DS, 0),
                TensorUtils.join_dimensions(traj[..., :2], 0, 2),
                TensorUtils.join_dimensions(traj[..., 2:3], 0, 2),
                batch_first=True,
            )
            pred_edges = {k: v for k, v in pred_edges.items() if k != "PP"}

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
                    (dis_padded[k][:, 1] < 0).sum()
                    / (edge_mask[k][:, 1].sum() + EPS)
                    / Tf
                ).item()
                metrics[f"collision_rate_all_{DS}_mode_{k}"] = TensorUtils.to_numpy(
                    (dis_padded[k] < 0).sum() / (edge_mask[k].sum() + EPS) / Tf
                ).item()
            metrics["collision_rate_ML"] = TensorUtils.to_numpy(
                sum([(v[:, 1] < 0).sum() for v in dis_padded.values()])
                / Tf
                / (sum([edge_mask[k][:, 1].sum() for k in edge_mask]) + EPS)
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
            for Tsecond in [3.0, 4.0, 5.0, 6.0, 8.0]:
                if Tf < Tsecond / dt:
                    continue
                Tf_bar = int(Tsecond / dt)
                # oracle mode means the trajectory decoded under the GT scene mode
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
                metrics[f"oracle_ADE@{Tsecond}s"] = allADE
                metrics[f"oracle_FDE@{Tsecond}s"] = allFDE
                metrics[f"oracle_vehicle_ADE@{Tsecond}s"] = vehADE
                metrics[f"oracle_vehicle_FDE@{Tsecond}s"] = vehFDE
                metrics[f"oracle_pedestrian_ADE@{Tsecond}s"] = pedADE
                metrics[f"oracle_pedestrian_FDE@{Tsecond}s"] = pedFDE

                # the second mode is the most likely mode (first is GT)
                ADE = Metrics.batch_average_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 1, ..., :Tf_bar, :2].reshape(B * N, 1, Tf_bar, 2),
                    confidence,
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="mean",
                ).reshape(B, N)
                FDE = Metrics.batch_final_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 1, ..., :Tf_bar, :2].reshape(B * N, 1, Tf_bar, 2),
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
                allFDE = FDE.sum() / (agent_mask.sum() + EPS)
                vehFDE = (FDE * vehicle_mask * agent_mask).sum() / (
                    (vehicle_mask * agent_mask).sum() + EPS
                )
                pedFDE = (FDE * pedestrian_mask * agent_mask).sum() / (
                    (pedestrian_mask * agent_mask).sum() + EPS
                )

                metrics[f"ML_ADE@{Tsecond}s"] = allADE
                metrics[f"ML_FDE@{Tsecond}s"] = allFDE
                metrics[f"ML_vehicle_ADE@{Tsecond}s"] = vehADE
                metrics[f"ML_vehicle_FDE@{Tsecond}s"] = vehFDE
                metrics[f"ML_pedestrian_ADE@{Tsecond}s"] = pedADE
                metrics[f"ML_pedestrian_FDE@{Tsecond}s"] = pedFDE

                # minADE and minFDE are calculated excluding the oracle mode
                ADE = Metrics.batch_average_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 1:, ..., :Tf_bar, :2]
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(B * N, -1, Tf_bar, 2),
                    confidence.repeat(Nmode - 1, 1) / (Nmode - 1),
                    fut_mask.reshape(B * N, Tf)[..., :Tf_bar],
                    mode="oracle",
                ).reshape(B, N)
                FDE = Metrics.batch_final_displacement_error(
                    agent_fut[..., :Tf_bar, :2].reshape(B * N, Tf_bar, 2),
                    pred_traj[:, 1:, ..., :Tf_bar, :2]
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(B * N, -1, Tf_bar, 2),
                    confidence.repeat(Nmode - 1, 1) / (Nmode - 1),
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

        if "dec_lm" in pred_batch:
            DS = pred_batch["dec_lm"].shape[1]
            M = batch["lane_xyh"].shape[-2]
            if self.cfg.classify_a2l_4all_lanes:
                dec_lm, dec_cond_lm, lane_mask = TensorUtils.to_numpy(
                    (
                        pred_batch["dec_lm"],
                        F.one_hot(
                            pred_batch["dec_cond_lm"][..., :M],
                            len(self.fut_lane_relation),
                        ),
                        batch["lane_mask"],
                    )
                )
                mask = (
                    (agent_mask[:, :, None] * lane_mask[:, None])
                    .reshape(B, 1, N, M)
                    .repeat(DS, 1)
                )
                lm_consistent_rate = ((dec_lm == dec_cond_lm).all(-1) * mask).sum() / (
                    (mask).sum() + EPS
                )
            else:
                dec_lm, dec_cond_lm, lane_mask = TensorUtils.to_numpy(
                    (
                        pred_batch["dec_lm"],
                        pred_batch["dec_cond_lm"],
                        batch["lane_mask"],
                    )
                )
                dec_lm = dec_lm.argmax(-1)

                if self.null_lane_mode:
                    # augment the null lane
                    dec_lm = np.concatenate(
                        [dec_lm, (dec_lm == 0).all(-1, keepdims=True)], -1
                    )
                    lane_mask = np.concatenate(
                        [lane_mask, np.ones([B, 1], dtype=bool)], -1
                    )
                mask = (
                    (agent_mask[:, :, None] * lane_mask[:, None])
                    .reshape(B, 1, N, -1)
                    .repeat(DS, 1)
                )
                lm_consistent_rate = ((dec_lm == dec_cond_lm) * mask).sum() / (
                    (mask).sum() + EPS
                )

            metrics["lm_consistent_rate"] = lm_consistent_rate
        if "dec_homotopy" in pred_batch:
            dec_homotopy, dec_cond_homotopy = TensorUtils.to_numpy(
                (pred_batch["dec_homotopy"], pred_batch["dec_cond_homotopy"])
            )
            DS = dec_homotopy.shape[1]
            mask = (
                (agent_mask[:, :, None] * agent_mask[:, None])
                .reshape(B, 1, N, N)
                .repeat(DS, 1)
            )

            homo_consistent_rate = (
                (dec_homotopy.squeeze(-1) == dec_cond_homotopy) * mask
            ).sum() / ((mask).sum() + EPS)
            metrics["homo_consistent_rate"] = homo_consistent_rate

        return metrics

    def compute_losses(self, pred_batch, inputs):
        EPS = 1e-3
        batch = pred_batch["batch"]
        lm_K = len(self.fut_lane_relation)
        homo_K = len(HomotopyType)
        GT_homotopy = F.one_hot(pred_batch["GT_homotopy"], homo_K).float()
        GT_lane_mode = F.one_hot(pred_batch["GT_lane_mode"], lm_K).float()
        mode_valid_flag = pred_batch["mode_valid_flag"]
        dt = self.cfg.step_time
        Tf = batch["agent_fut"].shape[-2]
        if not self.cfg.classify_a2l_4all_lanes:
            # We perform softmax (in cross entropy) over the lane segments
            mask_noton_mode = torch.ones(
                len(self.fut_lane_relation), dtype=torch.bool, device=self.device
            )
            mask_noton_mode[self.fut_lane_relation.NOTON] = False
        DS = pred_batch["trajectories"].size(1)
        future_mask = batch["fut_mask"]
        agent_mask = batch["hist_mask"].any(-1)
        # mode probability loss

        B, N, Mext = pred_batch["GT_lane_mode"].shape[:3]
        if self.null_lane_mode:
            M = Mext - 1
        else:
            M = Mext
        # marginal mode prediction probability
        lane_mode_pred = pred_batch["lane_mode_pred"]
        homotopy_pred = pred_batch["homotopy_pred"]
        # marginal probability loss, use "subtract max" trick for stable cross entropy

        B, N = agent_mask.shape
        if not self.cfg.classify_a2l_4all_lanes:
            # We perform softmax (in cross entropy) over the lane segments
            GT_lane_mode_marginal = torch.masked_select(
                GT_lane_mode, mask_noton_mode[None, None, None, :]
            ).view(*GT_lane_mode.shape[:3], -1)
            GT_lane_mode_marginal = GT_lane_mode_marginal.swapaxes(
                -1, -2
            )  # Not onehot anymore

            lane_mode_pred = lane_mode_pred.reshape(-1, Mext)
            marginal_lm_loss = self.lane_mode_loss(
                lane_mode_pred - lane_mode_pred.max(dim=-1, keepdim=True)[0],
                GT_lane_mode_marginal.reshape(-1, Mext),
            )
            lm_mask = mode_valid_flag.flatten().repeat_interleave(lm_K - 1, 0)
        else:
            lane_mode_pred = lane_mode_pred.reshape(-1, lm_K)
            marginal_lm_loss = self.lane_mode_loss(
                lane_mode_pred - lane_mode_pred.max(dim=-1, keepdim=True)[0],
                GT_lane_mode.reshape(-1, lm_K),
            )
            lm_mask = mode_valid_flag.flatten().repeat_interleave(M, 0)
            raise NotImplementedError

        marginal_homo_loss = self.homotopy_loss(
            homotopy_pred.reshape(-1, homo_K)
            - homotopy_pred.reshape(-1, homo_K).max(dim=-1, keepdim=True)[0],
            GT_homotopy.reshape(-1, homo_K),
        )

        homo_mask = (
            mode_valid_flag.unsqueeze(2) * mode_valid_flag.unsqueeze(1)
        ).flatten()
        marginal_lm_loss = (marginal_lm_loss * lm_mask).sum() / (lm_mask.sum() + EPS)
        marginal_homo_loss = (marginal_homo_loss * homo_mask).sum() / (
            homo_mask.sum() + EPS
        )

        # joint probability
        joint_logpi = pred_batch["joint_pred"]
        GT_joint_mode = torch.zeros_like(joint_logpi)
        GT_joint_mode[
            ..., 0
        ] = 1  # FIXME: Shouldn't this be only if the first element is the GT?
        joint_prob_loss = self.joint_mode_loss(
            (joint_logpi - joint_logpi.max(dim=-1, keepdim=True)[0]).nan_to_num(0),
            GT_joint_mode,
        ).mean()

        # decoded trajectory consistency loss
        dec_normalized_prob = pred_batch["dec_cond_prob"] / pred_batch[
            "dec_cond_prob"
        ].sum(-1, keepdim=True)

        if "dec_lm_margin" in pred_batch:
            mask = (agent_mask.unsqueeze(2) * batch["lane_mask"].unsqueeze(1)).view(
                B, 1, N, M, 1
            )
            if self.null_lane_mode:
                mask = torch.cat([mask, torch.ones_like(mask[:, :, :, :1])], -2)
            noton_mask = torch.ones(len(self.fut_lane_relation), device=mask.device)
            noton_mask[self.fut_lane_relation.NOTON] = 0
            mask = mask * noton_mask
            if self.null_lane_mode:
                # the margin for null lane is the inverse of the maximum of all lane margins
                null_lm_margin = (
                    -pred_batch["dec_lm_margin"].max(-2)[0]
                    * (1 - noton_mask)[None, None, None, :]
                    + -pred_batch["dec_lm_margin"].min(-2)[0]
                    * noton_mask[None, None, None, :]
                )
                dec_lm_margin = torch.cat(
                    [pred_batch["dec_lm_margin"], null_lm_margin.unsqueeze(-2)], -2
                )
            else:
                dec_lm_margin = pred_batch["dec_lm_margin"] - self.cfg.lm_margin_offset

            if self.weighted_consistency_loss:
                lm_consistency_loss = (
                    loss_clip(
                        F.relu(
                            -dec_lm_margin
                            * pred_batch["dec_cond_lm"].unsqueeze(-1)
                            * mask
                        ),
                        max_loss=4.0,
                    ).sum(-1)
                    * dec_normalized_prob[..., None, None]
                ).sum() / B  # FIXME: should this now be over M? (with self.cfg.classify_a2l_4all_lanes)
            else:
                lm_consistency_loss = (
                    (
                        loss_clip(
                            F.relu(
                                -dec_lm_margin
                                * pred_batch["dec_cond_lm"].unsqueeze(-1)
                                * mask
                            ),
                            max_loss=4.0,
                        ).sum(-1)
                    ).sum()
                    / B
                    / DS
                )

        else:
            lm_consistency_loss = torch.tensor(0.0, device=self.device)
        if "dec_lane_y_dev" in pred_batch:
            mask = (agent_mask.unsqueeze(2) * batch["lane_mask"].unsqueeze(1)).view(
                B, 1, N, M
            )
            time_weight = torch.arange(1, Tf + 1, device=self.device) / Tf**2
            psi_dev = pred_batch["dec_lane_psi_dev"].squeeze(-1)
            y_dev = pred_batch["dec_lane_y_dev"].squeeze(-1)
            yaw_dev_loss = (
                psi_dev.abs()
                * mask[..., :M, None]
                * pred_batch["dec_cond_lm"][..., :M, None]
                * time_weight[None, None, None, None, :]
            ).sum() / (mask[..., :M].sum() + EPS)
            y_dev_loss = (
                y_dev.abs()
                * mask[..., :M, None]
                * pred_batch["dec_cond_lm"][..., :M, None]
                * time_weight[None, None, None, None, :]
            ).sum() / (mask[..., :M].sum() + EPS)
        else:
            yaw_dev_loss = torch.tensor(0.0, device=self.device)
            y_dev_loss = torch.tensor(0.0, device=self.device)
        if "dec_homotopy_margin" in pred_batch:
            mask = (agent_mask.unsqueeze(2) * agent_mask.unsqueeze(1))[
                :, None, :, :, None
            ]
            if self.weighted_consistency_loss:
                homotopy_consistency_loss = (
                    loss_clip(
                        F.relu(
                            -pred_batch["dec_homotopy_margin"]
                            * F.one_hot(
                                pred_batch["dec_cond_homotopy"], len(HomotopyType)
                            )
                            * mask
                        ),
                        max_loss=4.0,
                    ).sum(-1)
                    * dec_normalized_prob[..., None, None]
                ).sum() / B
            else:
                homotopy_consistency_loss = (
                    (
                        loss_clip(
                            F.relu(
                                -pred_batch["dec_homotopy_margin"]
                                * F.one_hot(
                                    pred_batch["dec_cond_homotopy"],
                                    len(HomotopyType),
                                )
                                * mask
                            ),
                            max_loss=4.0,
                        ).sum(-1)
                    ).sum()
                    / B
                    / DS
                )
        else:
            homotopy_consistency_loss = torch.tensor(0.0, device=self.device)

        # trajectory reconstruction loss
        traj = pred_batch["trajectories"]
        if traj.shape[-2] != batch["agent_fut"].shape[-2]:
            xy_loss = torch.tensor(0.0, device=traj.device)
            yaw_loss = torch.tensor(0.0, device=traj.device)
            coll_loss = torch.tensor(0.0, device=traj.device)
            acce_reg_loss = torch.tensor(0.0, device=traj.device)
            steering_reg_loss = torch.tensor(0.0, device=traj.device)
            input_violation_loss = torch.tensor(0.0, device=traj.device)
            jerk_loss = torch.tensor(0.0, device=traj.device)
        else:
            # only penalize the trajectory under GT mode
            traj_GT_mode = pred_batch["trajectories"][:, 0]
            GT_xy = batch["agent_fut"][..., :2]
            GT_h = ratan2(batch["agent_fut"][..., 6:7], batch["agent_fut"][..., 7:8])

            xy_loss = self.traj_loss(GT_xy, traj_GT_mode[..., :2])
            xy_loss = xy_loss.norm(dim=-1)
            xy_loss = (xy_loss * future_mask).sum() / (future_mask.sum() + EPS)
            yaw_loss = torch.abs(
                GeoUtils.round_2pi(traj_GT_mode[..., 2:3] - GT_h)
            ).squeeze(-1)
            yaw_loss = (yaw_loss * future_mask).sum() / (future_mask.sum() + EPS)

            # collision loss
            extent = batch["extent"]
            DS = traj.size(1)
            pred_edges = self.bu.generate_edges(
                batch["agent_type"].repeat_interleave(DS, 0),
                extent.repeat_interleave(DS, 0),
                TensorUtils.join_dimensions(traj[..., :2], 0, 2),
                TensorUtils.join_dimensions(traj[..., 2:3], 0, 2),
                batch_first=True,
            )

            edge_weight = pred_batch["dec_cond_prob"].flatten().view(-1, 1)

            coll_loss = collision_loss(
                pred_edges={k: v for k, v in pred_edges.items() if k != "PP"},
                weight=edge_weight,
            ).nan_to_num(0)
            # diff = torch.norm((traj[:,1,...,:2]-traj[:,2,...,:2])*agent_mask[...,None,None])+torch.norm((traj[:,2,...,:2]-traj[:,3,...,:2])*agent_mask[...,None,None])+torch.norm((traj[:,3,...,:2]-traj[:,4,...,:2])*agent_mask[...,None,None])
            # print(diff)
            acce_reg_loss = torch.tensor(0.0, device=traj.device)
            steering_reg_loss = torch.tensor(0.0, device=traj.device)
            input_violation_loss = torch.tensor(0.0, device=traj.device)
            jerk_loss = torch.tensor(0.0, device=traj.device)

            type_mask = pred_batch["type_mask"]
            inputs = pred_batch["inputs"]
            input_violation = (
                pred_batch["input_violation"]
                if "input_violation" in pred_batch
                else dict()
            )
            jerk = pred_batch["jerk"] if "jerk" in pred_batch else dict()

            for k, dyn in self.nets["policy"].dyn.items():
                if type_mask[k].sum() == 0:
                    continue
                if isinstance(dyn, Unicycle) or isinstance(dyn, Unicycle_xyvsc):
                    acce_reg_loss += (
                        inputs[k][..., 0:1].norm(dim=-1).mean(-1) * type_mask[k]
                    ).sum() / (type_mask[k].sum() + EPS)
                    steering_reg_loss += (
                        inputs[k][..., 1:2].norm(dim=-1).mean(-1) * type_mask[k]
                    ).sum() / (type_mask[k].sum() + EPS)

                if k in input_violation:
                    input_violation_loss += (
                        input_violation[k].sum(-1).mean(-1) * type_mask[k]
                    ).sum() / (type_mask[k].sum() + EPS)

                if k in jerk:
                    jerk_loss += (
                        (jerk[k] ** 2).sum(-1).mean(-1) * type_mask[k]
                    ).sum() / (type_mask[k].sum() + EPS)

        losses = dict(
            marginal_lm_loss=marginal_lm_loss,
            marginal_homo_loss=marginal_homo_loss,
            l2_reg=lane_mode_pred.nan_to_num(0).norm(dim=-1).mean()
            + homotopy_pred.nan_to_num(0).norm(dim=-1).mean()
            + joint_logpi.nan_to_num(0).abs().mean(),
            joint_prob_loss=joint_prob_loss,
            lm_consistency_loss=lm_consistency_loss,
            homotopy_consistency_loss=homotopy_consistency_loss,
            yaw_dev_loss=yaw_dev_loss,
            y_dev_loss=y_dev_loss,
            xy_loss=xy_loss,
            yaw_loss=yaw_loss,
            coll_loss=coll_loss,
            acce_reg_loss=acce_reg_loss,
            steering_reg_loss=steering_reg_loss,
            input_violation_loss=input_violation_loss,
            jerk_loss=jerk_loss,
        )
        for k, v in losses.items():
            if v.isinf() or v.isnan():
                raise ValueError(f"{k} becomes NaN")
        return losses

    def log_pred_image(
        self,
        batch,
        pred_batch,
        batch_idx,
        logger,
        log_all_image=False,
        savegif=False,
        **kwargs,
    ):
        if batch_idx == 0:
            return

        try:
            parsed_batch = pred_batch["batch"]
            indices = (
                list(range(parsed_batch["hist_mask"].shape[0]))
                if log_all_image
                else [parsed_batch["hist_mask"][..., -1].sum(-1).argmax().item()]
            )
            world_from_agent = TensorUtils.to_numpy(parsed_batch["world_from_agent"])
            center_xy = TensorUtils.to_numpy(
                parsed_batch["centered_agent_state"][:, :2]
            )
            centered_agent_state = TensorUtils.to_numpy(
                parsed_batch["centered_agent_state"]
            )
            world_yaw = np.arctan2(
                centered_agent_state[:, -2], centered_agent_state[:, -1]
            )
            curr_posyaw = torch.cat(
                [
                    parsed_batch["agent_hist"][..., :2],
                    torch.arctan2(
                        parsed_batch["agent_hist"][..., -2],
                        parsed_batch["agent_hist"][..., -1],
                    )[..., None],
                ],
                -1,
            )
            NS = pred_batch["trajectories"].shape[1]
            traj = torch.cat(
                [
                    curr_posyaw[:, None, :, -1:].repeat_interleave(NS, 1),
                    pred_batch["trajectories"],
                ],
                -2,
            )
            traj = TensorUtils.to_numpy(traj)
            world_traj_xy = GeoUtils.batch_nd_transform_points_np(
                traj[..., :2], world_from_agent[:, None, None]
            )
            world_traj_yaw = traj[..., 2] + world_yaw[:, None, None, None]
            world_traj = np.concatenate([world_traj_xy, world_traj_yaw[..., None]], -1)

            extent = TensorUtils.to_numpy(parsed_batch["agent_hist_extent"][:, :, -1])
            hist_mask = TensorUtils.to_numpy(parsed_batch["hist_mask"])
            # pool = mp.Pool(len(indices))

            # def plot_scene(bi):
            for bi in indices:
                if isinstance(logger, str) or isinstance(logger, Path):
                    # directly save file
                    html_file_name = (
                        Path(logger) / f"CTT_visualization_{batch_idx}_{bi}.html"
                    )
                else:
                    html_file_name = f"CTT_visualization_{bi}.html"
                if os.path.exists(html_file_name):
                    os.remove(html_file_name)

                # plot agent
                for mode in range(min(NS, 3)):
                    output_file(html_file_name)
                    graph = figure(title="Bokeh graph", width=900, height=900)
                    graph.xgrid.grid_line_color = None
                    graph.ygrid.grid_line_color = None

                    graph.x_range = Range1d(
                        center_xy[bi, 0] - 40, center_xy[bi, 0] + 40
                    )
                    graph.y_range = Range1d(
                        center_xy[bi, 1] - 30, center_xy[bi, 1] + 50
                    )
                    plot_scene_open_loop(
                        graph,
                        world_traj[bi, mode],
                        extent[bi],
                        parsed_batch["vector_maps"][bi],
                        np.eye(3),
                        bbox=[
                            center_xy[bi, 0] - 40,
                            center_xy[bi, 0] + 40,
                            center_xy[bi, 1] - 30,
                            center_xy[bi, 1] + 50,
                        ],
                        mask=hist_mask[bi].any(-1),
                        color_scheme="palette",
                    )
                    if isinstance(logger, WandbLogger):
                        save(graph)
                        wandb_html = wandb.Html(open(html_file_name))
                        logger.experiment.log(
                            {f"val_pred_image_mode{mode}": wandb_html}
                        )

                    elif isinstance(logger, str) or isinstance(logger, Path):
                        html_file_name = (
                            Path(logger)
                            / f"CTT_visualization_{batch_idx}_{bi}_mode{mode}.html"
                        )
                        output_file(html_file_name)
                        save(graph)
                        png_name = (
                            str(html_file_name).removesuffix(".html")
                            + f"_mode{mode}.png"
                        )
                        export_png(graph, filename=png_name)
                    del graph
                    if savegif:
                        graph = figure(title="Bokeh graph", width=900, height=900)
                        graph.xgrid.grid_line_color = None
                        graph.ygrid.grid_line_color = None

                        graph.x_range = Range1d(
                            center_xy[bi, 0] - 40, center_xy[bi, 0] + 40
                        )
                        graph.y_range = Range1d(
                            center_xy[bi, 1] - 30, center_xy[bi, 1] + 50
                        )
                        gif_name = (
                            str(html_file_name).removesuffix(".html")
                            + f"_mode{mode}.gif"
                        )
                        animate_scene_open_loop(
                            graph,
                            world_traj[bi, mode],
                            extent[bi],
                            parsed_batch["vector_maps"][bi],
                            np.eye(3),
                            bbox=[
                                center_xy[bi, 0] - 40,
                                center_xy[bi, 0] + 40,
                                center_xy[bi, 1] - 30,
                                center_xy[bi, 1] + 50,
                            ],
                            mask=hist_mask[bi].any(-1),
                            color_scheme="palette",
                            dt=parsed_batch["dt"][0].item(),
                            gif_name=gif_name,
                            # tmp_dir=str(html_file_name).removesuffix(".html"),
                        )
                        del graph

            # pool.map(plot_scene, indices)
        except Exception as e:
            print(e)
            pass
