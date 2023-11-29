import enum
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from trajdata.data_structures import AgentType

import diffstack.utils.geometry_utils as GeoUtils
import diffstack.utils.lane_utils as LaneUtils
import diffstack.utils.model_utils as ModelUtils
import diffstack.utils.tensor_utils as TensorUtils
from diffstack.dynamics.unicycle import Unicycle, Unicycle_xyvsc
from diffstack.models.base_models import MLP
from diffstack.models.RPE_simple import sAuxRPEAttention, sAuxRPECrossAttention
from diffstack.models.TypeTransformer import *
from diffstack.utils.diffusion_utils import zero_module
from diffstack.utils.dist_utils import categorical_psample_wor
from diffstack.utils.geometry_utils import ratan2
from diffstack.utils.homotopy import HOMOTOPY_THRESHOLD, HomotopyType


class FeatureAxes(enum.Enum):
    """
    Axes of the features
    """

    B = enum.auto()  # batch
    A = enum.auto()  # agent
    T = enum.auto()  # time
    L = enum.auto()  # lanes
    F = enum.auto()  # features


class TFvars(enum.Enum):
    """
    Variables of the transformer
    """

    Agent_hist = enum.auto()  # Agent history trajectories
    Agent_future = enum.auto()  # Agent future trajectories
    Lane = enum.auto()  # Lanes


class GNNedges(enum.Enum):
    """edges of the GNN"""

    Agenthist2Lane = enum.auto()
    Agentfuture2Lane = enum.auto()
    Agenthist2Agenthist = enum.auto()
    Agentfuture2Agentfuture = enum.auto()
    Lane2Lane = enum.auto()


class FactorizedAttentionBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        PE_mode: str,
        use_rpe_net: bool,
        attn_attributes: OrderedDict,
        var_axes: dict,
        attn_pdrop: float = 0,
        resid_pdrop: float = 0,
        nominal_axes_order: list = None,
        MAX_T=50,
    ):
        """a factorized attention block

        Args:
            n_embd (int): embedding dimension
            n_head (int): number of heads
            PE_mode (str): Positional embedding mode, "RPE" or "PE"
            use_rpe_net (bool): For RPE, whether use RPE (True) or ProdPENet (False)
            attn_attributes (OrderedDict): recipe for attention blocks
            var_axes (dict): variables and their axes
            attn_pdrop (float, optional): dropout rate for attention. Defaults to 0.
            resid_pdrop (float, optional): dropout rate for residue connection. Defaults to 0.
            nominal_axes_order (list, optional): axes' order when arranging tensors. Defaults to None.
            MAX_T (int, optional): maximum teporal axis length for PE. Defaults to 50.

        """
        super().__init__()

        self.vars = list(var_axes.keys())
        self.var_axes = dict()
        for var in self.vars:
            self.var_axes[var] = [
                FeatureAxes[var_axis] for var_axis in var_axes[var].split(",")
            ]
        if nominal_axes_order is None:
            nominal_axes_order = list([var for var in FeatureAxes])
        self.nominal_axes_order = nominal_axes_order
        self.attn_attributes = attn_attributes
        self.attn = nn.ModuleDict()
        self.mlp = nn.ModuleDict()
        self.bn_1 = nn.ModuleDict()
        self.bn_2 = nn.ModuleDict()
        # key of attn_attributes are the two variables and the attention axis,
        # value of attn_attributes are the edge dimension, edge function (if defined), ntype, and whether to normalize the embedding
        for (var1, var2, axis), attributes in attn_attributes.items():
            edge_dim, edge_func, ntype, normalization = attributes

            if var1 == var2:
                # self attention
                attn_name = var1.name + "_" + var2.name + "_" + axis.name
                if axis in [FeatureAxes.A, FeatureAxes.L]:
                    # agent/lane axis attention
                    attn_net = TypeSelfAttention(
                        ntype=ntype,
                        n_embd=n_embd,
                        n_head=n_head,
                        edge_dim=edge_dim,
                        aux_edge_func=edge_func,
                        attn_pdrop=attn_pdrop,
                        resid_pdrop=resid_pdrop,
                    )
                elif axis == FeatureAxes.T:
                    # time axis attention
                    if edge_func is None:
                        if PE_mode == "RPE":
                            attn_net = sAuxRPEAttention(
                                n_embd=n_embd,
                                num_heads=n_head,
                                aux_vardim=edge_dim,
                                use_checkpoint=False,
                                use_rpe_net=use_rpe_net,
                            )
                        elif PE_mode == "PE":
                            attn_net = AuxSelfAttention(
                                n_embd=n_embd,
                                n_head=n_head,
                                edge_dim=edge_dim,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop,
                                PE_len=MAX_T,
                            )

                    else:
                        if PE_mode == "RPE":
                            attn_net = sAuxRPECrossAttention(
                                n_embd,
                                n_head,
                                edge_dim,
                                edge_func,
                                use_checkpoint=False,
                                use_rpe_net=use_rpe_net,
                            )
                        elif PE_mode == "PE":
                            attn_net = AuxCrossAttention(
                                n_embd,
                                n_head,
                                edge_dim,
                                edge_func,
                                attn_pdrop,
                                resid_pdrop,
                                PE_len=MAX_T,
                            )

                else:
                    raise NotImplementedError
                if normalization:
                    self.bn_1[attn_name + "_" + var1.name] = nn.BatchNorm1d(n_embd)
                    self.bn_2[attn_name] = nn.BatchNorm1d(n_embd)
            else:
                # cross attention
                attn_name = (
                    var1.name
                    + "_"
                    + var2.name
                    + "_"
                    + axis[0].name
                    + "->"
                    + axis[1].name
                )
                if axis == (FeatureAxes.A, FeatureAxes.L):
                    # cross attention between agent and lane
                    attn_net = TypeCrossAttention(
                        ntype=ntype,
                        n_embd=n_embd,
                        n_head=n_head,
                        edge_dim=edge_dim,
                        aux_edge_func=edge_func,
                        attn_pdrop=attn_pdrop,
                        resid_pdrop=resid_pdrop,
                    )
                elif var1 == TFvars.Agent_future and var2 == TFvars.Agent_hist:
                    assert axis == (FeatureAxes.T, FeatureAxes.T)
                    if PE_mode == "RPE":
                        attn_net = sAuxRPECrossAttention(
                            n_embd,
                            n_head,
                            edge_dim,
                            edge_func,
                            use_checkpoint=False,
                            use_rpe_net=use_rpe_net,
                        )
                    elif PE_mode == "PE":
                        attn_net = AuxCrossAttention(
                            n_embd,
                            n_head,
                            edge_dim,
                            edge_func,
                            attn_pdrop,
                            resid_pdrop,
                            PE_len=MAX_T,
                        )
                else:
                    raise NotImplementedError
                if normalization:
                    self.bn_1[attn_name + "_" + var1.name] = nn.BatchNorm1d(n_embd)
                    self.bn_1[attn_name + "_" + var2.name] = nn.BatchNorm1d(n_embd)
                    self.bn_2[attn_name] = nn.BatchNorm1d(n_embd)

            self.attn[attn_name] = attn_net
            self.mlp[attn_name] = tfmlp(n_embd, 4 * n_embd, resid_pdrop)

    def generate_agent_attn_mask(self, agent_mask, ntype):
        # return 2 attention masks for self, and neightbors, respectively, or only return neighbor mask
        # agent_mask: [B, N]
        assert ntype in [1, 2]
        B, N = agent_mask.shape
        cross_mask = agent_mask[:, :, None] * agent_mask[:, None]  # [B,N,N]
        # mask to attend to self
        I = torch.eye(N).to(agent_mask.device)[None]
        notI = torch.logical_not(I)

        self_mask = cross_mask * I
        # mask to attend to others
        neighbor_mask = cross_mask * notI
        if ntype == 1:
            return neighbor_mask
        else:
            return self_mask, neighbor_mask

    def get_cross_mask(
        self, var1, var2, axis, x1, x2, var_masks, cross_masks, ignore_var1_mask=True
    ):
        """get the attention mask for cross attention"""
        if (var1, var2, axis) in cross_masks:
            mask = cross_masks[(var1, var2, axis)]
        else:
            a1 = [a.name for a in self.var_axes[var1] if a != FeatureAxes.F]
            a2 = [a.name for a in self.var_axes[var2] if a != FeatureAxes.F]
            ag = [
                a.name
                for a in self.nominal_axes_order
                if a != FeatureAxes.F and ((a.name in a1) or (a.name in a2))
            ]

            # swap the attention axes to the last two
            ag = [a for a in ag if a != axis[0].name and a != axis[1].name] + [
                axis[0].name,
                axis[1].name,
            ]
            if axis[0] == axis[1]:
                # deal with repeated axes
                assert axis[1].name.swapcase() not in a1
                assert axis[0].name.swapcase() not in a2
                a2[a2.index(axis[1].name)] = axis[1].name.swapcase()
                ag[-1] = ag[-1].swapcase()
            cmd = "".join(a1) + "," + "".join(a2) + "->" + "".join(ag)
            if var1 in var_masks and not ignore_var1_mask:
                mask1 = var_masks[var1]
            else:
                mask1 = torch.ones_like(x1[..., 0])
            if var2 in var_masks:
                mask2 = var_masks[var2]
            else:
                mask2 = torch.ones_like(x2[..., 0])
            mask = torch.einsum(cmd, mask1, mask2)
            mask = mask.reshape(-1, *mask.shape[-2:])
        return mask

    def forward(
        self,
        vars,
        aux_xs=None,
        var_masks=dict(),
        cross_masks=dict(),
        frame_indices=dict(),
        edges=dict(),
    ):
        if TFvars.Agent_hist in vars:
            B, N, Th, D = vars[TFvars.Agent_hist].shape
        if TFvars.Lane in vars:
            B, L = vars[TFvars.Lane].shape[:2]
        if TFvars.Agent_future in vars:
            Tf = vars[TFvars.Agent_future].shape[2]
            if TFvars.Agent_hist in vars:
                assert N == vars[TFvars.Agent_future].shape[1]

        for (var1, var2, axis), attributes in self.attn_attributes.items():
            edge = edges[(var1, var2, axis)] if (var1, var2, axis) in edges else None
            edge_dim, edge_func, ntype, normalization = attributes
            if var1 == var2:
                attn_name = var1.name + "_" + var2.name + "_" + axis.name
                mlp = self.mlp[attn_name]
                # self attention

                attn_axis_idx = self.var_axes[var1].index(axis)
                x = vars[var1]
                aux_x = aux_xs[var1] if var1 in aux_xs else None
                if (var1, var2, axis) in cross_masks:
                    mask = cross_masks[(var1, var2, axis)]
                elif var1 in var_masks:
                    # provide the feature mask instead of attention mask
                    mask = var_masks[var1]
                    mask = mask.transpose(attn_axis_idx, -1)
                    mask = mask.reshape(-1, mask.size(-1))

                    if axis in [FeatureAxes.A, FeatureAxes.L]:
                        mask = self.generate_agent_attn_mask(mask, ntype)
                    else:
                        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                else:
                    mask = None
                if attn_axis_idx != len(self.var_axes[var1]) - 2:
                    # permute the attention axis to second last
                    x = x.transpose(attn_axis_idx, -2)
                    aux_x = (
                        aux_x.transpose(attn_axis_idx, -2)
                        if aux_x is not None
                        else None
                    )
                orig_shape = x.shape
                if axis in [FeatureAxes.A, FeatureAxes.L]:
                    x = x.reshape(-1, orig_shape[-2], orig_shape[-1])
                    aux_x = (
                        aux_x.reshape(-1, aux_x.shape[-2], aux_x.shape[-1])
                        if aux_x is not None
                        else None
                    )

                    if normalization:
                        bn_1 = self.bn_1[attn_name + "_" + var1.name]
                        bn_2 = self.bn_2[attn_name]
                        xn = bn_1(x.view(-1, x.shape[-1])).view(*x.shape)
                        resid = self.attn[attn_name](xn, aux_x, mask, edge=edge)
                        x = x + mlp(
                            bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                        )
                    else:
                        x = x + mlp(self.attn[attn_name](x, aux_x, mask, edge=edge))
                elif axis == FeatureAxes.T:
                    T = x.shape[-2]
                    x = x.reshape(-1, T, D)
                    aux_x = (
                        aux_x.reshape(-1, T, aux_x.shape[-1])
                        if aux_x is not None
                        else None
                    )
                    frame_index = frame_indices[var1]
                    frame_index = frame_index.reshape(-1, T)
                    if isinstance(self.attn[attn_name], sAuxRPEAttention) or isinstance(
                        self.attn[attn_name], AuxSelfAttention
                    ):
                        if normalization:
                            bn_1 = self.bn_1[attn_name + "_" + var1.name]
                            bn_2 = self.bn_2[attn_name]
                            xn = bn_1(x.view(-1, x.shape[-1])).view(*x.shape)
                            resid = self.attn[attn_name](
                                xn, aux_x, mask, frame_indices=frame_index, edge=edge
                            )
                            x = x + mlp(
                                bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                            )
                        else:
                            x = x + mlp(
                                self.attn[attn_name](
                                    x, aux_x, mask, frame_indices=frame_index, edge=edge
                                )
                            )

                    elif isinstance(
                        self.attn[attn_name], sAuxRPECrossAttention
                    ) or isinstance(self.attn[attn_name], AuxCrossAttention):
                        if normalization:
                            bn_1 = self.bn_1[attn_name + "_" + var1.name]
                            bn_2 = self.bn_2[attn_name]
                            xn = bn_1(x.view(-1, x.shape[-1])).view(*x.shape)
                            resid = self.attn[attn_name](
                                xn,
                                xn,
                                mask,
                                aux_x,
                                frame_index,
                                aux_x,
                                frame_index,
                                edge=edge,
                            )
                            x = x + mlp(
                                bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                            )
                        else:
                            x = x + mlp(
                                self.attn[attn_name](
                                    x,
                                    x,
                                    mask,
                                    aux_x,
                                    aux_x,
                                    frame_indices1=frame_index,
                                    frame_indices2=frame_index,
                                    edge=edge,
                                )
                            )
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                x = x.reshape(*orig_shape)
                if attn_axis_idx != len(self.var_axes[var1]) - 2:
                    # permute the attention axis back
                    x = x.transpose(attn_axis_idx, -2)
                vars[var1] = x
            else:
                # cross attention
                attn_name = (
                    var1.name
                    + "_"
                    + var2.name
                    + "_"
                    + axis[0].name
                    + "->"
                    + axis[1].name
                )
                mlp = self.mlp[attn_name]
                x1 = vars[var1]
                x2 = vars[var2]
                aux_x1 = aux_xs[var1] if var1 in aux_xs else None
                aux_x2 = aux_xs[var2] if var2 in aux_xs else None
                mask = self.get_cross_mask(
                    var1, var2, axis, x1, x2, var_masks, cross_masks
                )

                if (
                    var1 in [TFvars.Agent_hist, TFvars.Agent_future]
                    and var2 == TFvars.Lane
                ):
                    # cross attention between agent and lane
                    assert x1.ndim == 4  # B,N,T,D
                    assert x2.ndim == 3  # B,L,D
                    T = x1.shape[2]
                    x1 = x1.transpose(1, 2).reshape(-1, N, D)  # BT,N,D
                    x2 = x2.repeat_interleave(T, 0)  # BT,L,D
                    aux_x1 = aux_x1.transpose(1, 2).reshape(B * T, N, -1)
                    aux_x2 = aux_x2.repeat_interleave(T, 0)
                    if normalization:
                        bn_11 = self.bn_1[attn_name + "_" + var1.name]
                        bn_12 = self.bn_1[attn_name + "_" + var2.name]
                        bn_2 = self.bn_2[attn_name]
                        xn1 = bn_11(x1.view(-1, x1.shape[-1])).view(*x1.shape)
                        xn2 = bn_12(x2.view(-1, x2.shape[-1])).view(*x2.shape)
                        resid = self.attn[attn_name](
                            xn1, xn2, mask, aux_x1, aux_x2, edge=edge
                        )
                        x1 = x1 + mlp(
                            bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                        )
                    else:
                        x1 = x1 + mlp(
                            self.attn[attn_name](
                                x1, x2, mask, aux_x1, aux_x2, edge=edge
                            )
                        )
                    x1 = x1.reshape(B, T, N, D).transpose(1, 2)  # B,N,T,D
                elif var1 == TFvars.Lane and var2 in [
                    TFvars.Agent_hist,
                    TFvars.Agent_future,
                ]:
                    # cross attention between agent and lane
                    assert x2.ndim == 4  # B,N,T,D
                    assert x1.ndim == 3  # B,L,D
                    T = x2.shape[2]
                    L = x1.shape[1]
                    x2 = x2.transpose(1, 2).reshape(-1, N, D)  # BT,N,D
                    x1 = x1.repeat_interleave(T, 0)  # BT,L,D
                    aux_x2 = aux_x2.transpose(1, 2).reshape(-1, N, D)
                    aux_x1 = aux_x1.repeat_interleave(T, 0)
                    if normalization:
                        bn_11 = self.bn_1[attn_name + "_" + var1.name]
                        bn_12 = self.bn_1[attn_name + "_" + var2.name]
                        bn_2 = self.bn_2[attn_name]
                        xn1 = bn_11(x1.view(-1, x1.shape[-1])).view(*x1.shape)
                        xn2 = bn_12(x2.view(-1, x2.shape[-1])).view(*x2.shape)
                        resid = self.attn[attn_name](
                            xn1, xn2, mask, aux_x1, aux_x2, edge=edge
                        )
                        x1 = x1 + mlp(
                            bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                        )
                    else:
                        x1 = x1 + mlp(
                            self.attn[attn_name](
                                x1, x2, mask, aux_x1, aux_x2, edge=edge
                            )
                        )
                    x1 = x1.reshape(B, T, L, D).max(1)[0]  # B,L,D
                elif var1 == TFvars.Agent_future and var2 == TFvars.Agent_hist:
                    assert axis == (FeatureAxes.T, FeatureAxes.T)
                    x2 = vars[TFvars.Agent_hist].reshape(B * N, Th, D)
                    x1 = vars[TFvars.Agent_future].reshape(B * N, Tf, D)
                    aux_x2 = aux_xs[TFvars.Agent_hist].reshape(B * N, Th, -1)
                    aux_x1 = aux_xs[TFvars.Agent_future].reshape(B * N, Tf, -1)
                    frame_indices1 = frame_indices[TFvars.Agent_future].reshape(
                        B * N, Tf
                    )
                    frame_indices2 = frame_indices[TFvars.Agent_hist].reshape(B * N, Th)
                    if normalization:
                        bn_11 = self.bn_1[attn_name + "_" + var1.name]
                        bn_12 = self.bn_1[attn_name + "_" + var2.name]
                        bn_2 = self.bn_2[attn_name]
                        xn1 = bn_11(x1.view(-1, x1.shape[-1])).view(*x1.shape)
                        xn2 = bn_12(x2.view(-1, x2.shape[-1])).view(*x2.shape)
                        resid = self.attn[attn_name](
                            xn1,
                            xn2,
                            mask,
                            aux_x1,
                            aux_x2,
                            frame_indices1,
                            frame_indices2,
                            edge=edge,
                        ).reshape(B, N, Tf, D)
                        x1 = x1 + mlp(
                            bn_2(resid.view(-1, resid.shape[-1])).view(*resid.shape)
                        )
                    else:
                        x1 = mlp(
                            self.attn[attn_name](
                                x1,
                                x2,
                                mask,
                                aux_x1,
                                aux_x2,
                                frame_indices1,
                                frame_indices2,
                                edge=edge,
                            )
                        ).reshape(B, N, Tf, D)

                else:
                    raise NotImplementedError
                vars[var1] = x1
        return vars


class FactorizedGNN(nn.Module):
    def __init__(
        self,
        var_axes: dict,
        edge_var: dict,
        GNN_attributes: OrderedDict,
        node_n_embd: int,
        edge_n_embd: int,
        nominal_axes_order: list = None,
    ):
        """a factorized GNN

        Args:
            var_axes (dict): variables and their axes
            edge_var (dict): edges -> the variables it connects
            GNN_attributes (OrderedDict): recipe for GNN
            node_n_embd (int): embedding dimension for node variables
            edge_n_embd (int): embedding dimension for edge variables
            nominal_axes_order (list, optional): the order of axes when arranging tensors. Defaults to None.
        """
        super().__init__()
        self.vars = list(var_axes.keys())
        self.var_axes = dict()
        for var in self.vars:
            self.var_axes[var] = [
                FeatureAxes[var_axis] for var_axis in var_axes[var].split(",")
            ]
            self.edge_var = edge_var

        if nominal_axes_order is None:
            nominal_axes_order = list([var for var in FeatureAxes])
        self.nominal_axes_order = nominal_axes_order
        self.node_n_embd = node_n_embd
        self.edge_n_embd = edge_n_embd

        self.GNN_attributes = GNN_attributes
        self.GNN_nets = nn.ModuleDict()
        self.node_attn = nn.ParameterDict()
        self.pe_net = nn.ParameterDict()
        self.batch_norm = nn.ModuleDict()

        for (edge, gtype, var), attributes in self.GNN_attributes.items():
            hidden_dim, activation, pooling_method, Nmax = attributes

            if gtype == "edge":
                net_name = f"edge_{edge.name}"
                var1, var2 = self.edge_var[edge]
                nd1, nd2 = self.node_n_embd[var1], self.node_n_embd[var2]
                ed = self.edge_n_embd[edge]
                # message udpate of edge

                self.GNN_nets[net_name] = MLP(
                    input_dim=nd1 + nd2 + ed,
                    output_dim=ed,
                    layer_dims=hidden_dim,
                    layer_func=nn.Linear,
                    layer_func_kwargs=None,
                    activation=activation if activation is not None else nn.ReLU,
                    dropouts=None,
                    normalization=False,
                )
            elif gtype == "node":
                # message update of node
                net_name = f"node_{edge.name}_{var.name}"
                ed = self.edge_n_embd[edge]
                nd = self.node_n_embd[var]
                # message udpate of edge

                self.GNN_nets[net_name] = MLP(
                    input_dim=nd + ed,
                    output_dim=nd,
                    layer_dims=hidden_dim,
                    layer_func=nn.Linear,
                    layer_func_kwargs=None,
                    activation=activation if activation is not None else nn.ReLU,
                    dropouts=None,
                    normalization=False,
                )

                if pooling_method == "attn":
                    self.node_attn[net_name] = CrossAttention(nd, n_head=4)
                    self.pe_net[net_name] = (
                        nn.Parameter(torch.randn(Nmax, nd))
                        if Nmax is not None
                        else None
                    )

    def pooling(self, node_feat, node_feat_new, mask, net_name, pooling_method):
        """pooling the node features when aggregating the messages for node varaibles"""
        # node_feat: [B,N,node_dim]
        # node_feat_new: [B,N,N,node_dim]
        if pooling_method == "max":
            node_feat_new.masked_fill_(
                mask[..., None] == 0, node_feat_new.min().detach() - 1
            )
            node_feat_new = node_feat_new.max(2)[0]
        elif pooling_method == "mean":
            node_feat_new = (node_feat_new * mask[..., None]).sum(2) / mask.sum(2)[
                ..., None
            ].clip(min=1e-3)
        elif pooling_method == "attn":
            N, D = node_feat_new.shape[-2:]
            if self.pe_net[net_name] is not None:
                node_feat_new = node_feat_new + self.pe_net[net_name][None, None, :N]
            node_feat_new = self.node_attn[net_name](
                node_feat.reshape(-1, 1, D),
                node_feat_new.reshape(-1, N, D),
                mask.reshape(-1, 1, N),
            ).reshape(*node_feat.shape)

        return node_feat_new

    def get_cross_mask(
        self, var1, var2, edge, x1, x2, var_masks, cross_masks, axis=None
    ):
        """get the mask for message passing"""
        if edge in cross_masks:
            mask = cross_masks[edge]
        else:
            a1 = [a.name for a in self.var_axes[var1] if a != FeatureAxes.F]
            a2 = [a.name for a in self.var_axes[var2] if a != FeatureAxes.F]
            ag = [
                a.name
                for a in self.nominal_axes_order
                if a != FeatureAxes.F and ((a.name in a1) or (a.name in a2))
            ]
            if var1 == var2:
                # self mask
                # need to provide the axis on which the mask is applied
                assert axis is not None
                assert axis.name.swapcase() not in a1
                assert axis.name.swapcase() not in a2
                a2[a2.index(axis.name)] = axis.name.swapcase()
                ag.insert(ag.index(axis.name) + 1, axis.name.swapcase())

            cmd = "".join(a1) + "," + "".join(a2) + "->" + "".join(ag)
            if var1 in var_masks:
                mask1 = var_masks[var1]
            else:
                mask1 = torch.ones_like(x1[..., 0])
            if var2 in var_masks:
                mask2 = var_masks[var2]
            else:
                mask2 = torch.ones_like(x2[..., 0])
            mask = torch.einsum(cmd, mask1, mask2)
        return mask

    def forward(self, vars, var_masks, cross_masks):
        nodes = {var: x for var, x in vars.items() if var in TFvars}
        edges = {edge: x for edge, x in vars.items() if edge in GNNedges}
        for (edge, gtype, var), attributes in self.GNN_attributes.items():
            _, _, pooling_method, _ = attributes
            if gtype == "edge":
                net_name = f"edge_{edge.name}"
                var1, var2 = self.edge_var[edge]
                nx1, nx2 = nodes[var1], nodes[var2]
                ex = edges[edge]
                if edge in [GNNedges.Agenthist2Lane, GNNedges.Agentfuture2Lane]:
                    B, N, T = nx1.shape[:3]
                    M = nx2.size(1)
                    aggr_feature = torch.cat(
                        [
                            nx1.unsqueeze(3).expand(B, N, T, M, -1),
                            nx2.view(B, 1, 1, M, -1).expand(B, N, T, M, -1),
                            ex,
                        ],
                        dim=-1,
                    )
                    edges[edge] = edges[edge] + self.GNN_nets[net_name](aggr_feature)
                elif edge in [
                    GNNedges.Agenthist2Agenthist,
                    GNNedges.Agentfuture2Agentfuture,
                ]:
                    B, N, T = nx1.shape[:3]
                    aggr_feature = torch.cat(
                        [
                            nx1.unsqueeze(2).expand(B, N, N, T, -1),
                            nx1.unsqueeze(1).expand(B, N, N, T, -1),
                            ex,
                        ],
                        dim=-1,
                    )
                    edges[edge] = edges[edge] + self.GNN_nets[net_name](aggr_feature)
                elif edge == GNNedges.Lane2Lane:
                    B, M = nx1.shape[:2]
                    aggr_feature = torch.cat(
                        [
                            nx1.unsqueeze(2).expand(B, M, M, -1),
                            nx1.unsqueeze(3).expand(B, M, M, -1),
                            ex,
                        ],
                        dim=-1,
                    )
                    edges[edge] = edges[edge] + self.GNN_nets[net_name](aggr_feature)
                else:
                    raise NotImplementedError
            elif gtype == "node":
                net_name = f"node_{edge.name}_{var.name}"
                ex = edges[edge]
                var1, var2 = self.edge_var[edge]
                nx1, nx2 = nodes[var1], nodes[var2]
                nx = nodes[var]

                if edge in [GNNedges.Agenthist2Lane, GNNedges.Agentfuture2Lane]:
                    mask = self.get_cross_mask(
                        var1, var2, edge, nx1, nx2, var_masks, cross_masks
                    )
                    if var in [TFvars.Agent_future, TFvars.Agent_hist]:
                        B, N, T = nx.shape[:3]
                        aggr_feature = torch.cat(
                            [nx.unsqueeze(3).expand(B, N, T, M, -1), ex], dim=-1
                        )

                        new_nx = self.GNN_nets[net_name](aggr_feature)

                        nodes[var] = nodes[var] + self.pooling(
                            nx.reshape(B, N * T, -1),
                            new_nx.view(B, N * T, M, -1),
                            mask.view(B, N * T, M),
                            net_name,
                            pooling_method=pooling_method,
                        ).view(B, N, T, -1)
                    elif var == TFvars.Lane:
                        B, M = nx.shape[:2]
                        aggr_feature = torch.cat(
                            [nx[:, None, None].expand(B, N, T, M, -1), ex], dim=-1
                        )

                        new_nx = self.GNN_nets[net_name](aggr_feature)  # [B,N,T,M,D]
                        nodes[var] = nodes[var] + self.pooling(
                            nx,
                            new_nx.view(B, N * T, M, -1).transpose(1, 2),
                            mask.view(B, N * T, M).transpose(1, 2),
                            net_name,
                            pooling_method=pooling_method,
                        )

                elif edge in [
                    GNNedges.Agenthist2Agenthist,
                    GNNedges.Agentfuture2Agentfuture,
                ]:
                    mask = self.get_cross_mask(
                        var1,
                        var2,
                        edge,
                        nx1,
                        nx2,
                        var_masks,
                        cross_masks,
                        axis=FeatureAxes.A,
                    )
                    B, N, T = nx.shape[:3]
                    aggr_feature = torch.cat(
                        [nx.unsqueeze(2).expand(B, N, N, T, -1), ex], dim=-1
                    )
                    new_nx = self.GNN_nets[net_name](aggr_feature)  # [B,N,N,T,D]
                    nodes[var] = nodes[var] + self.pooling(
                        nx.transpose(1, 2).reshape(B * T, N, -1),
                        new_nx.permute(0, 3, 1, 2, 4).reshape(B * T, N, N, -1),
                        mask.permute(0, 3, 1, 2).reshape(B * T, N, N),
                        net_name,
                        pooling_method=pooling_method,
                    ).view(B, T, N, -1).transpose(1, 2)

                elif edge == GNNedges.Lane2Lane:
                    raise NotImplementedError

        return {**nodes, **edges}


class tfmlp(nn.Module):
    def __init__(self, n_embd, ff_dim, dropout):
        """An MLP with GELU activation and dropout for transformer residual connection

        Args:
            n_embd (int): embedding dimension
            ff_dim (int): feed forward dimension
            dropout (float): dropout rate
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ff_dim)
        self.c_proj = zero_module(nn.Linear(ff_dim, n_embd))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class CTTBlock(nn.Module):
    def __init__(self, TF_kwargs, GNN_kwargs):
        """a CTT block with a transformer block and a GNN block"""
        super().__init__()
        self.TFblock = (
            FactorizedAttentionBlock(**TF_kwargs) if TF_kwargs is not None else None
        )
        self.GNNblock = FactorizedGNN(**GNN_kwargs) if GNN_kwargs is not None else None

    def forward(
        self, vars, aux_xs, var_masks, cross_masks=None, frame_indices=None, edges={}
    ):
        if self.TFblock is not None:
            vars = self.TFblock(
                vars, aux_xs, var_masks, cross_masks, frame_indices, edges
            )
        if self.GNNblock is not None:
            vars = self.GNNblock(vars, var_masks, cross_masks)
        return vars


class CTTEncoder(nn.Module):
    def __init__(self, nblock, TF_kwargs, GNN_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CTTBlock(TF_kwargs, GNN_kwargs) for _ in range(nblock)]
        )

    def forward(self, vars, enc_kwargs):
        for block in self.blocks:
            vars = block(vars, **enc_kwargs)
        return vars


class CTT(nn.Module):
    def __init__(
        self,
        n_embd: int,
        embed_funcs: dict,
        enc_nblock: int,
        dec_nblock: int,
        enc_transformer_kwargs: dict,
        enc_GNN_kwargs: dict,
        dec_transformer_kwargs: dict,
        dec_GNN_kwargs: dict,
        enc_output_params: dict,
        dec_output_params: dict,
        hist_lane_relation,
        fut_lane_relation,
        max_joint_cardinality: int,
        classify_a2l_4all_lanes: bool = False,
        edge_func=dict(),
    ):
        """Categorical Traffic Transformer

        Args:
            n_embd (int): embedding dimensions
            embed_funcs (dict): embedding functions for variables
            enc_nblock (int): number of blocks in encoder
            dec_nblock (int): number of blocks in decoder
            enc_transformer_kwargs (dict): recipe for transformer in CTT encoder
            enc_GNN_kwargs (dict): recipe for GNN in CTT encoder
            dec_transformer_kwargs (dict): recipe for transformer in CTT decoder
            dec_GNN_kwargs (dict): recipe for GNN in CTT decoder
            enc_output_params (dict): parameters for mode prediction of CTT encoder
            dec_output_params (dict): parameters for trajectory prediction of CTT decoder
            hist_lane_relation: class of lane relation between agent history and lane segments
            fut_lane_relation: class of lane relation between agent future and lane segments
            max_joint_cardinality (int): maximum number of cardinality for each factor during importance sampling for joint scene mode
            classify_a2l_4all_lanes (bool, optional): whether to predict lane mode for every agent-lane pair. Defaults to False.
            edge_func (dict, optional): edge function between node variables. Defaults to dict().
        """
        super().__init__()
        # build transformer encoder and decoder
        self.embed_funcs = nn.ModuleDict({k.name: v for k, v in embed_funcs.items()})
        self.edge_var = {}
        if enc_GNN_kwargs is not None:
            self.edge_var.update(enc_GNN_kwargs["edge_var"])
        if dec_GNN_kwargs is not None:
            self.edge_var.update(dec_GNN_kwargs["edge_var"])
        self.encoder = CTTEncoder(enc_nblock, enc_transformer_kwargs, enc_GNN_kwargs)
        self.decoder = CTTEncoder(dec_nblock, dec_transformer_kwargs, dec_GNN_kwargs)
        self.n_embd = n_embd
        self.max_joint_cardinality = max_joint_cardinality
        self.hist_lane_relation = hist_lane_relation
        self.fut_lane_relation = fut_lane_relation
        self.classify_a2l_4all_lanes = classify_a2l_4all_lanes
        self.build_enc_pred_net(enc_output_params)
        self.build_dec_output_net(dec_output_params)
        self.use_hist_mode = False
        self.enc_output_params = enc_output_params
        self.dec_output_params = dec_output_params
        self.enc_vars = (
            [var for var in enc_transformer_kwargs["var_axes"].keys()]
            if enc_transformer_kwargs is not None
            else []
        )
        self.enc_edges = (
            [edge for edge in enc_GNN_kwargs["edge_var"].keys()]
            if enc_GNN_kwargs is not None
            else []
        )
        self.dec_vars = (
            [var for var in dec_transformer_kwargs["var_axes"].keys()]
            if dec_transformer_kwargs is not None
            else []
        )
        self.dec_edges = (
            [edge for edge in dec_GNN_kwargs["edge_var"].keys()]
            if dec_GNN_kwargs is not None
            else []
        )
        self.edge_func = edge_func
        self.dec_T = self.Tf

    def build_enc_pred_net(self, enc_output_params, use_hist_mode=False):
        # Nlr = len(self.fut_lane_relation)
        # Nhm = len(HomotopyType)
        self.Th = enc_output_params["Th"]
        self.pooling_T = enc_output_params["pooling_T"]
        self.null_lane_mode = enc_output_params["null_lane_mode"]
        mode_embed_dim = enc_output_params.get("mode_embed_dim", 64)
        self.lm_embed = nn.Embedding(len(self.fut_lane_relation), mode_embed_dim)
        self.homotopy_embed = nn.Embedding(len(HomotopyType), mode_embed_dim)
        hidden_dim = (
            enc_output_params["lane_mode"]["hidden_dim"]
            if "hidden_dim" in enc_output_params["lane_mode"]
            else [self.n_embd * 4, self.n_embd * 2]
        )
        # marginal probability of lane relation
        self.lane_mode_net = MLP(
            input_dim=self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
            output_dim=len(self.fut_lane_relation)
            - (
                1 - int(self.classify_a2l_4all_lanes)
            ),  # If loss over lanes, no NOTON relation
            layer_dims=hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
        )

        if self.pooling_T == "attn":
            n_head = enc_output_params["lane_mode"]["n_head"]
            if enc_output_params["PE_mode"] == "RPE":
                self.lane_mode_attn_T = sAuxRPECrossAttention(
                    self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    use_checkpoint=False,
                    use_rpe_net=False,
                )

                self.homotopy_attn_T = sAuxRPECrossAttention(
                    self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    use_checkpoint=False,
                    use_rpe_net=False,
                )

                self.agent_hist_attn_T = sAuxRPECrossAttention(
                    self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    use_checkpoint=False,
                    use_rpe_net=False,
                )
            elif enc_output_params["PE_mode"] == "PE":
                self.lane_mode_attn_T = AuxCrossAttention(
                    self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    PE_len=self.Th + 1,
                )
                self.homotopy_attn_T = AuxCrossAttention(
                    self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    PE_len=self.Th + 1,
                )
                self.agent_hist_attn_T = AuxCrossAttention(
                    self.n_embd,
                    n_head,
                    edge_dim=0,
                    aux_edge_func=None,
                    PE_len=self.Th + 1,
                )

        hidden_dim = (
            enc_output_params["homotopy"]["hidden_dim"]
            if "hidden_dim" in enc_output_params["homotopy"]
            else [self.n_embd * 4, self.n_embd * 2]
        )

        # marginal probability of homotopy
        self.homotopy_net = MLP(
            input_dim=self.n_embd + mode_embed_dim if use_hist_mode else self.n_embd,
            output_dim=len(HomotopyType),
            layer_dims=hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
        )

        GNN_kwargs = enc_output_params["joint_mode"]["GNN_kwargs"]
        jm_GNN_nblock = enc_output_params["joint_mode"].get("jm_GNN_nblock", 2)

        self.JM_GNN = nn.ModuleList(
            [FactorizedGNN(**GNN_kwargs) for i in range(jm_GNN_nblock)]
        )
        hidden_dim = enc_output_params["joint_mode"].get("hidden_dim", [256, 128])
        self.JM_lane_mode_factor = MLP(
            input_dim=self.n_embd + mode_embed_dim * 2
            if use_hist_mode
            else self.n_embd + mode_embed_dim,
            output_dim=1,
            layer_dims=hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
        )
        self.JM_homotopy_factor = MLP(
            input_dim=self.n_embd + mode_embed_dim * 2
            if use_hist_mode
            else self.n_embd + mode_embed_dim,
            output_dim=1,
            layer_dims=hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
        )
        self.num_joint_samples = enc_output_params["joint_mode"].get(
            "num_joint_samples", 30
        )
        self.num_joint_factors = enc_output_params["joint_mode"].get(
            "num_joint_factors", 6
        )

    def build_dec_output_net(self, dec_output_params):
        arch = dec_output_params["arch"]
        self.Tf = dec_output_params["Tf"]
        self.arch = arch
        dyn = dec_output_params.get("dyn", None)
        self.dyn = dyn
        self.dt = dec_output_params.get("dt", 0.1)
        traj_dim = dec_output_params["traj_dim"]
        self.decode_num_modes = dec_output_params.get("decode_num_modes", 1)
        self.AR_step_size = dec_output_params.get("AR_step_size", 1)
        self.AR_update_mode = dec_output_params.get("AR_update_mode", "step")
        self.LR_sample_hack = dec_output_params.get("LR_sample_hack", False)
        self.dec_rounds = dec_output_params.get("dec_rounds", 3)
        assert self.Tf % self.AR_step_size == 0
        if arch == "lstm":
            self.output_rnn = nn.ModuleDict()
            num_layers = dec_output_params.get("num_layers", 1)
            hidden_size = dec_output_params.get("lstm_hidden_size", self.n_embd)
            for k, v in self.dyn.items():
                if v is not None:
                    proj_size = v.udim
                else:
                    proj_size = traj_dim
                self.output_rnn[k.name] = nn.LSTM(
                    self.n_embd,
                    hidden_size,
                    batch_first=True,
                    num_layers=num_layers,
                    proj_size=proj_size,
                )
        elif arch == "mlp":
            self.output_mlp = nn.ModuleDict()
            hidden_dim = dec_output_params.get(
                "mlp_hidden_dims", [self.n_embd * 2, self.n_embd * 4]
            )
            for k, v in self.dyn.items():
                self.output_mlp[k.name] = MLP(
                    input_dim=self.n_embd,
                    output_dim=traj_dim if v is None else v.udim,
                    layer_dims=hidden_dim,
                    layer_func=nn.Linear,
                    layer_func_kwargs=None,
                    activation=nn.ReLU,
                    dropouts=None,
                    normalization=False,
                    output_activation=None,
                )

    def embed_raw(self, raw_vars, aux_xs, aux_edges={}):
        vars = dict()
        for kname, v in self.embed_funcs.items():
            if hasattr(TFvars, kname):
                # embed a TFvar
                k = getattr(TFvars, kname)
                if k in raw_vars:
                    vars[k] = self.embed_funcs[k.name](raw_vars[k])

            elif hasattr(GNNedges, kname):
                k = getattr(GNNedges, kname)
                var1, var2 = self.edge_var[k]
                if var1 in aux_xs and var2 in aux_xs:
                    aux_edge = aux_edges.get(k, None)
                    vars[k] = self.embed_funcs[k.name](
                        aux_xs[var1], aux_xs[var2], aux_edge
                    )
        return vars

    def predict_from_context(
        self,
        context_vars,
        aux_xs,
        frame_indices,
        var_masks,
        cross_masks,
        enc_edges,
        GT_lane_mode=None,
        GT_homotopy=None,
        prev_lm_pred_dict=defaultdict(lambda: None),
        prev_homo_pred_dict=defaultdict(lambda: None),
        num_samples=None,
        GT_batch_mask=None,
    ):
        # GT_batch_mask: identify which indices (among batch dimension) are given the GT modes

        device = context_vars[TFvars.Agent_hist].device
        if num_samples is None:
            num_samples = self.num_joint_samples
        # predict lane mode
        agent_feat = context_vars[TFvars.Agent_hist]
        lane_feat = context_vars[TFvars.Lane]
        a2l_edge = context_vars[GNNedges.Agenthist2Lane]
        a2a_edge = context_vars[GNNedges.Agenthist2Agenthist]
        B, N, Th = agent_feat.shape[:3]
        if GT_batch_mask is None and GT_lane_mode is not None:
            GT_batch_mask = torch.ones(B, device=device, dtype=bool)
        M = lane_feat.shape[1]
        if self.use_hist_mode:
            prev_lm_pred = [
                torch.zeros(B, N, M, dtype=torch.int, device=device) for t in range(Th)
            ]
            prev_homo_pred = [
                torch.zeros(B, N, N, dtype=torch.int, device=device) for t in range(Th)
            ]

            for t, v in prev_lm_pred_dict.items():
                prev_lm_pred[t] = v
            for t, v in prev_homo_pred_dict.items():
                if v is not None:
                    prev_homo_pred[t] = v
            prev_lm_embed = self.lm_embed(torch.stack(prev_lm_pred, 2))
            prev_homo_embed = self.homotopy_embed(torch.stack(prev_homo_pred, 3))
            a2l_edge = torch.cat([a2l_edge, prev_lm_embed], -1)
            a2a_edge = torch.cat([a2a_edge, prev_homo_embed], -1)
        if self.pooling_T == "max":
            a2l_edge_pool = a2l_edge.max(2)[0]
            a2a_edge_pool = a2a_edge.max(3)[0]
            agent_feat_pool = agent_feat.max(-2)[0]

        elif self.pooling_T == "attn":
            query = a2l_edge[:, :, -1].reshape(B * N * M, 1, -1)
            frame_indices2 = (
                frame_indices[TFvars.Agent_hist]
                .reshape(B * N, -1)
                .repeat_interleave(M, 0)
            )
            frame_indices1 = frame_indices2[:, -1:]
            hist_mask = var_masks[TFvars.Agent_hist]  # B,N,T
            mask = (
                (hist_mask[..., :1, None] * hist_mask.unsqueeze(2))
                .repeat_interleave(M, 1)
                .reshape(B * N * M, 1, Th)
            )
            a2l_edge_pool = self.lane_mode_attn_T(
                query,
                a2l_edge.permute(0, 1, 3, 2, 4).reshape(B * N * M, Th, -1),
                mask,
                None,
                None,
                frame_indices1=frame_indices1,
                frame_indices2=frame_indices2,
            ).reshape(B, N, M, -1)

            query = a2a_edge[:, :, :, -1].reshape(B * N * N, 1, -1)
            frame_indices2 = (
                frame_indices[TFvars.Agent_hist]
                .reshape(B * N, -1)
                .repeat_interleave(N, 0)
            )
            frame_indices1 = frame_indices2[:, -1:]
            mask = (
                (hist_mask[..., :1, None] * hist_mask.unsqueeze(2))
                .repeat_interleave(N, 1)
                .reshape(B * N * N, 1, Th)
            )
            a2a_edge_pool = self.homotopy_attn_T(
                query,
                a2a_edge.reshape(B * N * N, Th, -1),
                mask,
                None,
                None,
                frame_indices1,
                frame_indices2,
            ).reshape(B, N, N, -1)

            query = agent_feat[:, :, -1].reshape(B * N, 1, -1)
            frame_indices2 = frame_indices[TFvars.Agent_hist].reshape(B * N, -1)
            frame_indices1 = frame_indices2[:, -1:]
            mask = (hist_mask[..., :1, None] * hist_mask.unsqueeze(2)).reshape(
                B * N, 1, Th
            )
            agent_feat_pool = self.agent_hist_attn_T(
                query,
                agent_feat.reshape(B * N, Th, -1),
                mask,
                None,
                None,
                frame_indices1,
                frame_indices2,
            ).view(B, N, -1)

        lane_mode_pred = self.lane_mode_net(a2l_edge_pool)

        if not self.classify_a2l_4all_lanes:  # if self.per_mode
            lane_mode_pred = lane_mode_pred.swapaxes(
                -1, -2
            )  # [..., M, nbr_modes] -> [..., nbr_modes, M]
            if self.null_lane_mode:
                # add a null lane mode
                lane_mode_pred = torch.cat(
                    [lane_mode_pred, torch.zeros_like(lane_mode_pred[..., 0:1])], -1
                )

        lane_mode_prob = F.softmax(lane_mode_pred, dim=-1)

        homotopy_pred = self.homotopy_net(a2a_edge_pool)
        homotopy_asym = homotopy_pred.transpose(1, 2)
        homotopy_pred = (homotopy_pred + homotopy_asym) / 2

        homotopy_prob = F.softmax(homotopy_pred, dim=-1)

        ########### Compute joint mode probability ############
        D_lane = lane_mode_pred.shape[
            -1
        ]  # Either M (M+1) or len(self.fut_lane_relation)
        D_homo = len(HomotopyType)
        D = min([max([D_lane, D_homo]), self.max_joint_cardinality])
        M_lm = lane_mode_pred.shape[
            -2
        ]  # number of possible modes for each lane,agent pair

        # factor masks for joint mode prediction
        agent_mask = var_masks[TFvars.Agent_hist].any(-1).float()
        lane_mask = var_masks[TFvars.Lane]
        lm_factor_mask = agent_mask.unsqueeze(2) * lane_mask.unsqueeze(1)  # (B,N,M)
        if self.null_lane_mode and not self.classify_a2l_4all_lanes:
            lane_mask_ext = torch.cat(
                [lane_mask, torch.ones(*lane_mask.shape[:-1], 1, device=device)], -1
            )
            lm_factor_mask = torch.cat([lm_factor_mask, agent_mask.unsqueeze(-1)], -1)
        else:
            lane_mask_ext = lane_mask

        homo_factor_mask = agent_mask.unsqueeze(2) * agent_mask.unsqueeze(1)  # (B,N,N)
        if not self.classify_a2l_4all_lanes:
            mode_mask = torch.ones(
                [agent_mask.shape[0], M_lm], device=agent_mask.device
            )  # All modes active
            lm_factor_mask_per_mode = agent_mask.unsqueeze(2) * mode_mask.unsqueeze(1)
            combined_factor_mask = torch.cat(
                [
                    lm_factor_mask_per_mode.view(B, N * M_lm),
                    homo_factor_mask.view(B, N * N),
                ],
                -1,
            )
        else:
            combined_factor_mask = torch.cat(
                [lm_factor_mask.view(B, N * M_lm), homo_factor_mask.view(B, N * N)], -1
            )

        if D_homo < D:
            homotopy_pred_padded = torch.cat(
                [
                    homotopy_pred,
                    torch.ones(*homotopy_pred.shape[:-1], D - D_homo, device=device)
                    * -torch.inf,
                ],
                -1,
            )
        elif D_homo > D:
            raise NotImplementedError("We can only consider all homotopies")
        else:
            homotopy_pred_padded = homotopy_pred

        if (
            self.LR_sample_hack and GT_lane_mode is not None
        ):  # hack to favor sampling left and right lane of each agent
            if (
                self.hist_lane_relation == LaneUtils.LaneRelation
                and self.fut_lane_relation == LaneUtils.SimpleLaneRelation
            ):
                lane_mode_pred_m = lane_mode_pred.clone().detach()
                try:
                    hist_lane_flag = enc_edges[
                        (TFvars.Agent_hist, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L))
                    ][..., -len(LaneUtils.LaneRelation) :]
                    hist_lane_flag = hist_lane_flag.reshape(B, Th, N, M, -1)[
                        :, -1
                    ].type(torch.bool)
                    # pick out left and right lane and modify the probability
                    lm_logpi_max = lane_mode_pred_m.max(-1)[0]
                    left_lane_mask = hist_lane_flag[
                        ..., LaneUtils.LaneRelation.LEFTOF
                    ].unsqueeze(2)
                    right_lane_mask = hist_lane_flag[
                        ..., LaneUtils.LaneRelation.RIGHTOF
                    ].unsqueeze(2)
                    on_lane_mask = hist_lane_flag[
                        ..., LaneUtils.LaneRelation.ON
                    ].unsqueeze(2)
                    lane_mode_pred_m[..., :M] = (
                        lane_mode_pred_m[..., :M] * (~left_lane_mask)
                        + lm_logpi_max.unsqueeze(-1) * left_lane_mask
                    )
                    lane_mode_pred_m[..., :M] = (
                        lane_mode_pred_m[..., :M] * (~right_lane_mask)
                        + lm_logpi_max.unsqueeze(-1) * right_lane_mask
                    )
                    lane_mode_pred_m[..., :M] = (
                        lane_mode_pred_m[..., :M] * (~on_lane_mask)
                        + lm_logpi_max.unsqueeze(-1) * on_lane_mask
                    )
                except:
                    pass

            else:
                lane_mode_pred_m = None
        else:
            lane_mode_pred_m = None

        if D_lane < D:
            lane_mode_pred_padded = torch.cat(
                [
                    lane_mode_pred,
                    torch.ones(*lane_mode_pred.shape[:-1], D - D_lane, device=device)
                    * -torch.inf,
                ],
                -1,
            )
            indices = torch.arange(D_lane, device=device).expand(B, N, 1, D_lane)
        elif D_lane > D:
            # For each agent & mode, find the D likeliest lanes
            assert (
                not self.classify_a2l_4all_lanes
            ), "Currently only implemented for per_mode setting"

            lane_mode_pred_masked = lane_mode_pred.masked_fill(
                torch.logical_not(lane_mask_ext[:, None, None, :]), -torch.inf
            )

            sorted, indices = lane_mode_pred_masked.topk(
                D, dim=-1
            )  # Only consider most likely lanes!
            lane_mode_pred_padded = torch.gather(
                lane_mode_pred, -1, indices
            )  # (...,M) -> (...,D)
        else:
            lane_mode_pred_padded = lane_mode_pred
            indices = torch.arange(D_lane, device=device).expand(B, N, 1, D_lane)

        combined_logpi = torch.cat(
            [
                lane_mode_pred_padded.view(B, N * M_lm, -1),
                homotopy_pred_padded.view(B, N * N, -1),
            ],
            1,
        )
        # for all invalid factors, turn the logpi to [0,-inf,-inf,-inf...]
        combined_logpi[..., 1:].masked_fill_(
            ~(combined_factor_mask.bool().unsqueeze(-1)), -torch.inf
        )

        if lane_mode_pred_m is not None:
            if D_lane < D:
                lane_mode_pred_padded_m = torch.cat(
                    [
                        lane_mode_pred_m,
                        torch.ones(
                            *lane_mode_pred_m.shape[:-1], D - D_lane, device=device
                        )
                        * -torch.inf,
                    ],
                    -1,
                )
                indices_m = torch.arange(D_lane, device=device).expand(B, N, 1, D_lane)
            elif D_lane > D:
                # For each agent & mode, find the D likeliest lanes
                assert (
                    not self.classify_a2l_4all_lanes
                ), "Currently only implemented for per_mode setting"
                lane_mode_pred_masked_m = lane_mode_pred_m.masked_fill(
                    torch.logical_not(lane_mask_ext[:, None, None, :]), -torch.inf
                )
                sorted, indices_m = lane_mode_pred_masked_m.topk(
                    D, dim=-1
                )  # Only consider most likely lanes!
                lane_mode_pred_padded_m = torch.gather(
                    lane_mode_pred_m, -1, indices_m
                )  # (...,M) -> (...,D)

            else:
                lane_mode_pred_padded_m = lane_mode_pred_m
                indices_m = torch.arange(D_lane, device=device).expand(B, N, 1, D_lane)

            combined_logpi_m = torch.cat(
                [
                    lane_mode_pred_padded_m.view(B, N * M_lm, -1),
                    homotopy_pred_padded.clone().detach().view(B, N * N, -1),
                ],
                1,
            )
            # for all invalid factors, turn the logpi to [0,-inf,-inf,-inf...]
            combined_logpi_m[..., 1:].masked_fill_(
                ~(combined_factor_mask.bool().unsqueeze(-1)), -torch.inf
            )
        else:
            combined_logpi_m = None

        # relevance score
        # lane mode relevance score
        # first, factors with one dominant mode is less relevant (or one dominant lane)
        lm_dominance_score = -(
            lane_mode_prob.max(dim=-1)[0] - 1 / lane_mode_prob.shape[-1]
        )  # B,N,M_lm  
        homo_dominance_score = -(
            homotopy_prob.max(dim=-1)[0] - 1 / homotopy_prob.shape[-1]
        )  # B,N,N
        # second, lanes that are far away from the agents are less important
        a2ledge_raw = ModelUtils.agent2lane_edge_proj(
            aux_xs[TFvars.Agent_hist][:, :, -1], aux_xs[TFvars.Lane]
        )

        if not self.classify_a2l_4all_lanes:
            a2l_dis_score = 0  # For per mode, we don't consider lanes specifically so no a2l distance score

        else:
            a2l_dis = a2ledge_raw[..., :2].norm(dim=-1)  # B,N,M
            a2l_dis_score = 1 / (a2l_dis + 1)
        # lanes that are closer to ego is more important

        # third, agents far from ego is less relevant
        a2eedge_raw = ModelUtils.agent2agent_edge(
            aux_xs[TFvars.Agent_hist][:, 0:1, -1], aux_xs[TFvars.Agent_hist][:, :, -1]
        ).squeeze(1)
        a2e_dis = a2eedge_raw[..., :2].norm(dim=-1)
        a2e_dis.masked_fill_(torch.logical_not(agent_mask.bool()), torch.inf)
        a2e_dis_score = 1 / (a2e_dis.clip(min=0.5))
        a2e_dis_score_homo = a2e_dis_score.unsqueeze(1) * a2e_dis_score.unsqueeze(2)
        a2e_dis_score_homo.masked_fill_(
            torch.eye(N, device=a2e_dis_score.device, dtype=torch.bool).unsqueeze(0), 0
        )

        lm_factor_score = (
            lm_dominance_score * 0.1
            + a2l_dis_score
            + a2e_dis_score.unsqueeze(-1).repeat_interleave(M_lm, -1) * 2
        )
        homo_factor_score = homo_dominance_score + a2e_dis_score_homo

        # mask out half of homo_factor due to symmetry
        sym_mask = torch.tril(torch.ones(B, N, N, dtype=torch.bool, device=device))
        homo_factor_score.masked_fill_(sym_mask, homo_factor_score.min().detach() - 5)
        combined_factor_mask[:, -N * N :].masked_fill_(sym_mask.view(B, -1), 0)
        if not self.classify_a2l_4all_lanes:
            # Set the relevance score for ego agent (0) and the on lane index such that this is always a factor
            max_score = torch.max(lm_factor_score.max(), homo_factor_score.max())
            on_lane_idx = (
                self.fut_lane_relation.ON - 1
                if self.fut_lane_relation.ON > self.fut_lane_relation.NOTON
                else self.fut_lane_relation.ON
            )  # We removed lane_pred
            lm_factor_score[:, 0, on_lane_idx] = (
                1.1 * max_score
            )  # Temp, maybe make customizable

        combined_factor_score = torch.cat(
            [lm_factor_score.view(B, N * M_lm), homo_factor_score.view(B, N * N)], -1
        )

        num_factor = min(self.num_joint_factors, N * (M_lm + N))
        # modify the loglikelihood so that more important factors get more samples
        temperature = 2
        modfied_combined_logpi = combined_logpi / torch.exp(
            temperature
            * (combined_factor_score - combined_factor_score.min())
            / (combined_factor_score.max() - combined_factor_score.min())
        ).unsqueeze(-1)
        # Do importance sampling (sampling only the factors) the remaining marginals are chosen to be their argmax value
        joint_sample, factor_idx = categorical_psample_wor(
            modfied_combined_logpi,
            num_samples,
            num_factor,
            factor_mask=combined_factor_mask,
            relevance_score=combined_factor_score,
        )

        if combined_logpi_m is not None:
            modfied_combined_logpi_m = combined_logpi_m / torch.exp(
                temperature
                * (combined_factor_score - combined_factor_score.min())
                / (combined_factor_score.max() - combined_factor_score.min())
            ).unsqueeze(-1)
            joint_sample_m, factor_idx_m = categorical_psample_wor(
                modfied_combined_logpi_m,
                self.decode_num_modes,
                num_factor,
                factor_mask=combined_factor_mask,
                relevance_score=combined_factor_score,
            )
        else:
            joint_sample_m = None

        # Turn indices of factors of importance back to combined_logpi size, then sum over num factors to get actual one hot (assuming nonrepetitive factors)

        (
            lm_sample,
            homo_sample,
            factor_mask,
        ) = self.restore_lm_homotopy_from_joint_sample(
            joint_sample,
            indices,
            factor_idx,
            lm_factor_mask,
            N,
            lane_mask_ext.shape[-1],
            M_lm,
        )
        if joint_sample_m is not None:
            (
                lm_sample_m,
                homo_sample_m,
                factor_mask_m,
            ) = self.restore_lm_homotopy_from_joint_sample(
                joint_sample_m,
                indices_m,
                factor_idx_m,
                lm_factor_mask,
                N,
                lane_mask_ext.shape[-1],
                M_lm,
            )
            lm_sample = torch.cat([lm_sample, lm_sample_m], 1)
            homo_sample = torch.cat([homo_sample, homo_sample_m], 1)
            num_samples += lm_sample_m.size(1)
            factor_mask = factor_mask | factor_mask_m

        if self.null_lane_mode:
            # remove the factor_mask for the null lane
            factor_mask = factor_mask.view(B, N, -1)
            factor_mask = torch.cat(
                [factor_mask[..., :M], factor_mask[..., M + 1 :]], -1
            ).reshape(B, -1)

        if (
            GT_lane_mode is not None and GT_homotopy is not None
        ):  # If we have GT, add it as sample for training
            # for all the irrelevant entries, make them exactly the same as GT
            if self.classify_a2l_4all_lanes:
                lm_sample = lm_sample * lm_factor_mask.bool().unsqueeze(1) + (
                    GT_lane_mode * torch.logical_not(lm_factor_mask.bool())
                ).unsqueeze(1)

            homo_sample = homo_sample * homo_factor_mask.bool().unsqueeze(1) + (
                GT_homotopy * torch.logical_not(homo_factor_mask.bool())
            ).unsqueeze(1)
            # mask modes that are the same as GT

            lm_GT_flag = (lm_sample == GT_lane_mode.unsqueeze(1)).all(dim=3)
            lm_GT_flag = lm_GT_flag.masked_fill(
                torch.logical_not(agent_mask[:, None]), 1
            ).all(dim=2)
            homo_GT_flag = homo_sample == GT_homotopy.unsqueeze(1)
            homo_GT_flag = (
                homo_GT_flag.masked_fill(
                    torch.logical_not(homo_factor_mask)[:, None], 1
                )
                .all(-1)
                .all(-1)
            )
            GT_flag = (lm_GT_flag & homo_GT_flag) * GT_batch_mask[:, None]
            # make sure that at least one entry of GT_flag is true for all scene, we can then remove 1 sample from every scene
            num_samples = num_samples - GT_flag.sum(-1).max().item()

            lm_sample = torch.stack(
                [
                    lm_sample[i][torch.where(~GT_flag[i])[0]][:num_samples]
                    for i in range(B)
                ],
                0,
            )
            homo_sample = torch.stack(
                [
                    homo_sample[i][torch.where(~GT_flag[i])[0]][:num_samples]
                    for i in range(B)
                ],
                0,
            )
            if GT_batch_mask.all():
                lm_sample = torch.cat([GT_lane_mode.unsqueeze(1), lm_sample], 1)
                homo_sample = torch.cat([GT_homotopy.unsqueeze(1), homo_sample], 1)
            else:
                lm_sample_GT = torch.cat([GT_lane_mode.unsqueeze(1), lm_sample], 1)
                homo_sample_GT = torch.cat([GT_homotopy.unsqueeze(1), homo_sample], 1)
                # for instances where GT is not provided, simply pad the last index
                lm_sample_nonGT = torch.cat([lm_sample, lm_sample[:, :1]], 1)
                homo_sample_nonGT = torch.cat([homo_sample, homo_sample[:, :1]], 1)
                lm_sample = lm_sample_GT * GT_batch_mask[
                    :, None, None, None
                ] + lm_sample_nonGT * torch.logical_not(
                    GT_batch_mask[:, None, None, None]
                )
                homo_sample = homo_sample_GT * GT_batch_mask[
                    :, None, None, None
                ] + homo_sample_nonGT * torch.logical_not(
                    GT_batch_mask[:, None, None, None]
                )
            num_samples += 1

        lm_embedding = self.lm_embed(lm_sample[..., :M])

        homo_embedding = self.homotopy_embed(homo_sample)
        a2l_edge_with_mode = torch.cat(
            [
                a2l_edge_pool.repeat_interleave(num_samples, 0),
                lm_embedding.view(B * num_samples, N, M, -1),
            ],
            -1,
        )
        a2a_edge_with_mode = torch.cat(
            [
                a2a_edge_pool.repeat_interleave(num_samples, 0),
                homo_embedding.view(B * num_samples, N, N, -1),
            ],
            -1,
        )

        JM_GNN_vars = {
            TFvars.Agent_hist: agent_feat_pool.repeat_interleave(
                num_samples, 0
            ).unsqueeze(-2),
            TFvars.Lane: context_vars[TFvars.Lane].repeat_interleave(num_samples, 0),
            GNNedges.Agenthist2Lane: a2l_edge_with_mode.unsqueeze(2),
            GNNedges.Agenthist2Agenthist: a2a_edge_with_mode.unsqueeze(3),
        }
        JM_GNN_var_masks = {
            TFvars.Agent_hist: var_masks[TFvars.Agent_hist]
            .any(2, keepdim=True)
            .repeat_interleave(num_samples, 0),
            TFvars.Lane: var_masks[TFvars.Lane].repeat_interleave(num_samples, 0),
        }
        for i in range(len(self.JM_GNN)):
            JM_GNN_vars = self.JM_GNN[i](
                JM_GNN_vars, JM_GNN_var_masks, cross_masks=dict()
            )

        lm_factor = self.JM_lane_mode_factor(JM_GNN_vars[GNNedges.Agenthist2Lane]).view(
            B, num_samples, N, M
        )

        homotopy_factor = self.JM_homotopy_factor(
            JM_GNN_vars[GNNedges.Agenthist2Agenthist]
        ).view(B, num_samples, N, N)

        total_factor = torch.cat([lm_factor, homotopy_factor], -1).reshape(
            B, num_samples, N * (M + N)
        )
        total_factor_mask = (
            torch.cat([lm_factor_mask[..., :M], homo_factor_mask], -1).reshape(B, -1)
            * factor_mask
        )
        joint_logpi = (
            total_factor * (total_factor_mask * factor_mask).unsqueeze(1)
        ).sum(-1)

        if joint_sample_m is not None:
            num_sample_m = joint_sample_m.size(1)
            ml_idx = joint_logpi.argmax(dim=1, keepdim=True)
            # decode the GT, the most likely mode other than GT, and the modes sampled with modified probability
            if GT_lane_mode is not None:
                dec_idx = torch.cat(
                    [
                        torch.zeros(B, 1, device=device),
                        ml_idx,
                        torch.arange(
                            num_samples - num_sample_m, num_samples, device=device
                        )[None].repeat_interleave(B, 0),
                    ],
                    dim=1,
                ).type(torch.int64)
            else:
                dec_idx = torch.cat(
                    [
                        ml_idx,
                        torch.arange(
                            num_samples - num_sample_m, num_samples, device=device
                        )[None].repeat_interleave(B, 0),
                    ],
                    dim=1,
                ).type(torch.int64)
        else:
            dec_idx = joint_logpi.topk(
                min(self.decode_num_modes, joint_logpi.shape[1]), dim=1
            ).indices
            if GT_lane_mode is not None:
                dec_idx = torch.cat(
                    [torch.zeros(B, 1, device=device), dec_idx], 1
                ).type(torch.int64)
        if self.null_lane_mode:
            dec_cond_lm = torch.gather(
                lm_sample, 1, dec_idx.view(B, -1, 1, 1).repeat(1, 1, N, M + 1)
            )
        else:
            dec_cond_lm = torch.gather(
                lm_sample, 1, dec_idx.view(B, -1, 1, 1).repeat(1, 1, N, M)
            )
        dec_cond_homotopy = torch.gather(
            homo_sample, 1, dec_idx.view(B, -1, 1, 1).repeat(1, 1, N, N)
        )
        joint_mode_prob = torch.softmax(joint_logpi, -1)
        dec_cond_prob = torch.gather(joint_mode_prob, 1, dec_idx)
        # homotopy result test
        # dec_cond_homotopy[0,1,0,2]=2
        # dec_cond_homotopy[0,1,2,0]=2
        return dict(
            lane_mode_pred=lane_mode_pred,
            homotopy_pred=homotopy_pred,
            joint_pred=joint_logpi,
            lane_mode_sample=lm_sample,
            homotopy_sample=homo_sample,
            dec_cond_lm=dec_cond_lm,
            dec_cond_homotopy=dec_cond_homotopy,
            dec_cond_prob=dec_cond_prob,
        )

    def decode_trajectory_AR(
        self,
        enc_vars,
        raw_vars,
        aux_xs,
        var_masks,
        cross_masks,
        frame_indices,
        agent_mask,
        x0,
        u0,
        lane_mode,
        homotopy,
        center_from_agents,
    ):
        ARS = self.AR_step_size
        B, N, Th = enc_vars[TFvars.Agent_hist].shape[:3]
        M = aux_xs[TFvars.Lane].shape[1]
        device = enc_vars[TFvars.Agent_hist].device
        raw_agent_hist = raw_vars[TFvars.Agent_hist]  # [B,N,Th,F]
        raw_agent_fut = list(
            raw_agent_hist[:, :, -1:]
            .repeat_interleave(1 + self.dec_T, 2)
            .split(1, dim=2)
        )
        state_out = {k: list() for k in self.dyn}
        input_out = {k: list() for k in self.dyn}
        Tf = self.dec_T
        curr_yaw = ratan2(
            aux_xs[TFvars.Agent_future][:, :, 0, 3:4],
            aux_xs[TFvars.Agent_future][:, :, 0, 4:5],
        )
        curr_yaw = curr_yaw * var_masks[TFvars.Agent_future][:, :, 0:1]
        ar_mask = torch.zeros([1, 1, Tf + 1], dtype=bool, device=device)
        xt = x0
        dec_kwargs_ar = dict(
            aux_xs=aux_xs,
            cross_masks=cross_masks,
            frame_indices=frame_indices,
        )
        lane_xysc = aux_xs[TFvars.Lane].reshape(B, M, -1, 4)
        future_xyvsc = list(torch.split(aux_xs[TFvars.Agent_future][..., :5], 1, dim=2))
        for i in range(Tf // ARS):
            dec_raw_vars = {
                TFvars.Agent_future: torch.cat(raw_agent_fut, 2),
                TFvars.Lane: raw_vars[TFvars.Lane],
            }

            ar_mask[:, :, : i * ARS + 1] = 1
            x_mask = var_masks[TFvars.Agent_future] * ar_mask

            cross_masks = dict()
            Agent_fut_cross_mask = var_masks[TFvars.Agent_future].unsqueeze(
                -1
            ) * var_masks[TFvars.Agent_future].unsqueeze(-2)
            Agent_fut_cross_mask = torch.tril(Agent_fut_cross_mask) * ar_mask.unsqueeze(
                -2
            )

            cross_masks[
                TFvars.Agent_future, TFvars.Agent_future, FeatureAxes.T
            ] = Agent_fut_cross_mask

            dec_kwargs_ar["var_masks"] = var_masks
            dec_kwargs_ar["cross_masks"] = cross_masks
            future_xyvsc_tensor = torch.cat(future_xyvsc, dim=2)
            future_aux = torch.cat(
                [future_xyvsc_tensor, aux_xs[TFvars.Agent_future][..., 5:]], -1
            )
            aux_xs[TFvars.Agent_future] = future_aux
            future_xysc = future_xyvsc_tensor[..., [0, 1, 3, 4]]
            lane_margins = (
                self.fut_lane_relation.get_all_margins(
                    TensorUtils.join_dimensions(future_xysc, 0, 2),
                    lane_xysc.repeat_interleave(N, 0),
                )
                .reshape(B, N, M, 1 + Tf, -1)
                .clip(-20, 20)
            )
            lane_margins.masked_fill_(
                ~(
                    x_mask[:, :, None, :, None]
                    * var_masks[TFvars.Lane][:, None, :, None, None]
                ).bool(),
                0,
            )
            # mask out all unselected lanes
            lane_margins.masked_fill_(
                lane_mode[..., self.fut_lane_relation.NOTON, None, None].bool(), 0
            )
            agent_edge = future_xysc[..., :2].unsqueeze(2) - future_xysc[
                ..., :2
            ].unsqueeze(1)
            rel_angle = ratan2(agent_edge[..., [1]], agent_edge[..., [0]])
            rel_angle_offset = torch.cat(
                [rel_angle[..., :1, :], rel_angle[..., :-1, :]], -2
            )

            angle_diff = (rel_angle - rel_angle_offset).cumsum(-2)
            homotopy_margins = torch.cat(
                [
                    HOMOTOPY_THRESHOLD - angle_diff.abs(),
                    -angle_diff - HOMOTOPY_THRESHOLD,
                    angle_diff - HOMOTOPY_THRESHOLD,
                ],
                -1,
            )
            homotopy_margins.masked_fill_(
                torch.logical_not(x_mask.unsqueeze(1) * x_mask.unsqueeze(2)).unsqueeze(
                    -1
                ),
                0,
            )
            aux_dec_edges = {
                GNNedges.Agentfuture2Agentfuture: torch.cat(
                    [
                        homotopy.unsqueeze(3).repeat_interleave(1 + self.dec_T, 3),
                        homotopy_margins,
                    ],
                    -1,
                ),
                GNNedges.Agentfuture2Lane: torch.cat(
                    [
                        lane_mode.unsqueeze(2).repeat_interleave(1 + self.dec_T, 2),
                        lane_margins.transpose(2, 3),
                    ],
                    -1,
                ),
            }
            dec_vars = self.embed_raw(
                dec_raw_vars,
                {TFvars.Agent_future: future_aux, TFvars.Lane: aux_xs[TFvars.Lane]},
                aux_dec_edges,
            )
            vars = {**enc_vars, **dec_vars}
            # update edges due to change of aux_xs
            dec_agent_aux = future_aux.permute(0, 2, 1, 3).reshape(B * (Tf + 1), N, -1)

            a2a_edge = torch.cat(
                [
                    self.edge_func["a2a"](dec_agent_aux, dec_agent_aux),
                    homotopy.repeat_interleave(Tf + 1, 0),
                    TensorUtils.join_dimensions(
                        homotopy_margins.permute(0, 3, 1, 2, 4), 0, 2
                    ),
                ],
                -1,
            )
            a2l_edge = torch.cat(
                [
                    self.edge_func["a2l"](
                        dec_agent_aux, aux_xs[TFvars.Lane].repeat_interleave(Tf + 1, 0)
                    ),
                    lane_mode.repeat_interleave(Tf + 1, 0),
                    TensorUtils.join_dimensions(
                        lane_margins.permute(0, 3, 1, 2, 4), 0, 2
                    ),
                ],
                -1,
            )
            edges = {
                (TFvars.Agent_future, TFvars.Agent_future, FeatureAxes.A): a2a_edge,
                (TFvars.Agent_future, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L)): [
                    a2l_edge
                ],
            }
            dec_kwargs_ar["edges"] = edges
            vars = self.decoder(vars, dec_kwargs_ar)
            out = dict()

            if self.AR_update_mode == "step":
                if self.arch == "lstm":
                    raise ValueError("LSTM arch is not working!")

                elif self.arch == "mlp":
                    for k in self.dyn:
                        out[k] = self.output_mlp[k.name](
                            vars[TFvars.Agent_future].reshape(B * N, Tf + 1, -1)[
                                :, i * ARS : (i + 1) * ARS
                            ]
                        ).view(B, N, ARS, -1)
                else:
                    raise NotImplementedError

            elif self.AR_update_mode == "all":
                if self.arch == "lstm":
                    raise ValueError("LSTM arch is not working!")
                elif self.arch == "mlp":
                    for k in self.dyn:
                        out[k] = self.output_mlp[k.name](
                            vars[TFvars.Agent_future].reshape(B * N, Tf + 1, -1)[
                                :, : (i + 1) * ARS
                            ]
                        ).view(B, N, ARS * (i + 1), -1)
                else:
                    raise NotImplementedError

            agent_xyvsc_by_type = dict()
            agent_acce = dict()
            agent_r = dict()
            for k, dyn in self.dyn.items():
                if dyn is None:
                    # output assumed to take form [x,y,v,h]
                    if center_from_agents is None:
                        agent_xyvsc_by_type[k] = torch.cat(
                            [
                                out[k][..., :3],
                                torch.sin(out[k][..., 3:]),
                                torch.cos(out[k][..., 3:]),
                            ],
                            -1,
                        )

                    else:
                        # transform to global coordinate
                        xy_local = out[k][..., :2]
                        agent_v = out[k][..., 2:3]
                        h_local = out[k][..., 3:4]
                        xy_global = GeoUtils.batch_nd_transform_points(
                            xy_local, center_from_agents.unsqueeze(2)
                        )
                        h_global = h_local + curr_yaw.unsqueeze(2)
                        agent_xyvsc_by_type[k] = torch.cat(
                            [
                                xy_global,
                                agent_v,
                                torch.sin(h_global),
                                torch.cos(h_global),
                            ],
                            -1,
                        )

                    agent_v = out[k][..., 2:3]
                    if self.AR_update_mode == "step":
                        agent_v_pre = torch.cat(
                            future_xyvsc[i * ARS : (i + 1) * ARS], 2
                        )[..., 2:3]
                        # here we assume input is always 2 dimensional
                        input_out[k].append(torch.zeros_like(out[k][..., :2]))
                    elif self.AR_update_mode == "all":
                        agent_v_pre = torch.cat(future_xyvsc[: (i + 1) * ARS], 2)[
                            ..., 2:3
                        ]

                    agent_acce[k] = (agent_v - agent_v_pre) / self.dt
                    agent_r[k] = torch.zeros_like(agent_v)

                else:
                    if isinstance(dyn, Unicycle):
                        xseq = dyn.forward_dynamics(
                            xt[k], out[k], mode="chain", bound=False
                        )

                        if self.AR_update_mode == "step":
                            agent_xyvsc_by_type[k] = dyn.state2xyvsc(xseq)
                            state_out[k].append(xseq)
                            xt[k] = xseq[:, :, -1]
                        elif self.AR_update_mode == "all":
                            agent_xyvsc_by_type[k] = dyn.state2xyvsc(xseq)

                        agent_acce[k] = out[k][..., :1]
                        agent_r[k] = out[k][..., 1:2]

                    elif isinstance(dyn, Unicycle_xyvsc):
                        xseq = dyn.forward_dynamics(xt[k], out[k], bound=False)

                        agent_xyvsc_by_type[k] = xseq
                        if self.AR_update_mode == "step":
                            xt[k] = xseq[:, :, -1]
                            state_out[k].append(xseq)
                        agent_acce[k] = out[k][..., :1]
                        agent_r[k] = out[k][..., 1:2]
                    else:
                        raise NotImplementedError
                    input_out[k].append(out[k])
            agent_xyvsc_combined = sum(
                [
                    agent_mask[k][..., None, None] * agent_xyvsc_by_type[k]
                    for k in agent_xyvsc_by_type
                ]
            )

            acce_combined = sum(
                [agent_mask[k][..., None, None] * agent_acce[k] for k in agent_acce]
            )
            r_combined = sum(
                [agent_mask[k][..., None, None] * agent_r[k] for k in agent_r]
            )
            agent_v = agent_xyvsc_combined[..., 2:3]
            # update the agent future aux

            # update the agent future raw [v,acce,r,static_features]

            agent_vu = torch.cat([agent_v, acce_combined, r_combined], -1)
            if self.AR_update_mode == "step":
                future_xyvsc[i * ARS + 1 : (i + 1) * ARS + 1] = list(
                    torch.split(agent_xyvsc_combined, 1, dim=2)
                )
                for j in range(ARS):
                    raw_agent_fut[i * ARS + 1 + j] = torch.cat(
                        [
                            agent_vu[:, :, j : j + 1],
                            raw_agent_fut[i * ARS + 1 + j][..., 3:],
                        ],
                        -1,
                    )
            elif self.AR_update_mode == "all":
                future_xyvsc[: (i + 1) * ARS + 1] = list(
                    torch.split(agent_xyvsc_combined, 1, dim=2)
                )
                for j in range(ARS * (i + 1)):
                    raw_agent_fut[j] = torch.cat(
                        [agent_vu[:, :, j : j + 1], raw_agent_fut[j][..., 3:]], -1
                    )

        if self.AR_update_mode == "step":
            future_xyvsc = torch.cat(future_xyvsc, dim=2)
            input_out = {
                k: torch.cat(input_type, dim=2) if len(input_type) > 0 else None
                for k, input_type in input_out.items()
            }
            state_out = {
                k: torch.cat(state_type, dim=2)
                if len(state_type) > 0
                else future_xyvsc[..., 1:, :]
                for k, state_type in state_out.items()
            }
        elif self.AR_update_mode == "all":
            future_xyvsc = agent_xyvsc_combined
            for k, v in out.items():
                if v is not None:
                    input_out[k] = v
                else:
                    input_out[k] = torch.zeros_like(out[k][..., :2])

        trajectories = torch.cat(
            [
                future_xyvsc[:, :, 1:, :2],
                ratan2(future_xyvsc[:, :, 1:, 3:4], future_xyvsc[:, :, 1:, 4:5]),
            ],
            -1,
        )

        input_violation = dict()
        jerk = dict()
        for k, dyn in self.dyn.items():
            if dyn is not None:
                xl = torch.cat([x0[k].unsqueeze(-2), state_out[k][..., :-1, :]], -2)
                input_violation[k] = (
                    dyn.get_input_violation(xl, input_out[k])
                    * agent_mask[k][..., None, None]
                )
                inputs_extended = torch.cat([u0[k].unsqueeze(-2), input_out[k]], -2)
                jerk[k] = (
                    (inputs_extended[..., 1:, :] - inputs_extended[..., :-1, :])
                    / self.dt
                ) * agent_mask[k][..., None, None]
        return trajectories, input_out, state_out, input_violation, jerk

    def decode_trajectory_OS(
        self,
        enc_vars,
        raw_vars,
        aux_xs,
        var_masks,
        cross_masks,
        frame_indices,
        agent_mask,
        x0,
        u0,
        lane_mode,
        homotopy,
        center_from_agents,
    ):
        B, N, Th = enc_vars[TFvars.Agent_hist].shape[:3]
        M = aux_xs[TFvars.Lane].shape[1]
        device = enc_vars[TFvars.Agent_hist].device
        raw_agent_hist = raw_vars[TFvars.Agent_hist]  # [B,N,Th,F]
        raw_agent_fut = raw_agent_hist[:, :, -1:].repeat_interleave(1 + self.dec_T, 2)
        Tf = self.dec_T
        curr_yaw = ratan2(
            aux_xs[TFvars.Agent_future][:, :, 0, 3:4],
            aux_xs[TFvars.Agent_future][:, :, 0, 4:5],
        )
        curr_yaw = curr_yaw * var_masks[TFvars.Agent_future][:, :, 0:1]
        dec_kwargs = dict(
            aux_xs=aux_xs,
            cross_masks=cross_masks,
            frame_indices=frame_indices,
        )
        lane_xysc = aux_xs[TFvars.Lane].reshape(B, M, -1, 4)
        future_xyvsc = aux_xs[TFvars.Agent_future][..., :5]
        input_out = dict()
        state_out = dict()
        for i in range(self.dec_rounds):
            dec_raw_vars = {
                TFvars.Agent_future: raw_agent_fut,
                TFvars.Lane: raw_vars[TFvars.Lane],
            }

            x_mask = var_masks[TFvars.Agent_future]

            cross_masks = dict()
            Agent_fut_cross_mask = var_masks[TFvars.Agent_future].unsqueeze(
                -1
            ) * var_masks[TFvars.Agent_future].unsqueeze(-2)

            cross_masks[
                TFvars.Agent_future, TFvars.Agent_future, FeatureAxes.T
            ] = torch.tril(Agent_fut_cross_mask)

            dec_kwargs["var_masks"] = var_masks
            dec_kwargs["cross_masks"] = cross_masks
            future_aux = torch.cat(
                [future_xyvsc, aux_xs[TFvars.Agent_future][..., 5:]], -1
            )
            aux_xs[TFvars.Agent_future] = future_aux
            future_xysc = future_xyvsc[..., [0, 1, 3, 4]]
            lane_margins = (
                self.fut_lane_relation.get_all_margins(
                    TensorUtils.join_dimensions(future_xysc, 0, 2),
                    lane_xysc.repeat_interleave(N, 0),
                )
                .reshape(B, N, M, 1 + Tf, -1)
                .clip(-20, 20)
            )
            lane_margins.masked_fill_(
                ~(
                    x_mask[:, :, None, :, None]
                    * var_masks[TFvars.Lane][:, None, :, None, None]
                ).bool(),
                0,
            )
            # mask out all unselected lanes
            lane_margins.masked_fill_(
                lane_mode[..., self.fut_lane_relation.NOTON, None, None].bool(), 0
            )
            agent_edge = future_xysc[..., :2].unsqueeze(2) - future_xysc[
                ..., :2
            ].unsqueeze(1)
            rel_angle = ratan2(agent_edge[..., [1]], agent_edge[..., [0]])
            rel_angle_offset = torch.cat(
                [rel_angle[..., :1, :], rel_angle[..., :-1, :]], -2
            )

            angle_diff = (rel_angle - rel_angle_offset).cumsum(-2)
            homotopy_margins = torch.cat(
                [
                    HOMOTOPY_THRESHOLD - angle_diff.abs(),
                    -angle_diff - HOMOTOPY_THRESHOLD,
                    angle_diff - HOMOTOPY_THRESHOLD,
                ],
                -1,
            )
            homotopy_margins.masked_fill_(
                torch.logical_not(x_mask.unsqueeze(1) * x_mask.unsqueeze(2)).unsqueeze(
                    -1
                ),
                0,
            )
            aux_dec_edges = {
                GNNedges.Agentfuture2Agentfuture: torch.cat(
                    [
                        homotopy.unsqueeze(3).repeat_interleave(1 + self.dec_T, 3),
                        homotopy_margins,
                    ],
                    -1,
                ),
                GNNedges.Agentfuture2Lane: torch.cat(
                    [
                        lane_mode.unsqueeze(2).repeat_interleave(1 + self.dec_T, 2),
                        lane_margins.transpose(2, 3),
                    ],
                    -1,
                ),
            }
            dec_vars = self.embed_raw(
                dec_raw_vars,
                {TFvars.Agent_future: future_aux, TFvars.Lane: aux_xs[TFvars.Lane]},
                aux_dec_edges,
            )
            vars = {**enc_vars, **dec_vars}
            # update edges due to change of aux_xs
            dec_agent_aux = future_aux.permute(0, 2, 1, 3).reshape(B * (Tf + 1), N, -1)

            a2a_edge = torch.cat(
                [
                    self.edge_func["a2a"](dec_agent_aux, dec_agent_aux),
                    homotopy.repeat_interleave(Tf + 1, 0),
                    TensorUtils.join_dimensions(
                        homotopy_margins.permute(0, 3, 1, 2, 4), 0, 2
                    ),
                ],
                -1,
            )
            a2l_edge = torch.cat(
                [
                    self.edge_func["a2l"](
                        dec_agent_aux, aux_xs[TFvars.Lane].repeat_interleave(Tf + 1, 0)
                    ),
                    lane_mode.repeat_interleave(Tf + 1, 0),
                    TensorUtils.join_dimensions(
                        lane_margins.permute(0, 3, 1, 2, 4), 0, 2
                    ),
                ],
                -1,
            )
            edges = {
                (TFvars.Agent_future, TFvars.Agent_future, FeatureAxes.A): a2a_edge,
                (TFvars.Agent_future, TFvars.Lane, (FeatureAxes.A, FeatureAxes.L)): [
                    a2l_edge
                ],
            }
            dec_kwargs["edges"] = edges
            vars = self.decoder(vars, dec_kwargs)
            out = dict()

            if self.arch == "mlp":
                for k in self.dyn:
                    out[k] = self.output_mlp[k.name](
                        vars[TFvars.Agent_future].reshape(B, N, Tf + 1, -1)
                    )
            else:
                raise NotImplementedError

            agent_xyvsc_by_type = dict()
            agent_acce = dict()
            agent_r = dict()
            for k, dyn in self.dyn.items():
                if dyn is None:
                    # output assumed to take form [x,y,v,h]
                    if center_from_agents is None:
                        agent_xyvsc_by_type[k] = torch.cat(
                            [
                                out[k][..., :3],
                                torch.sin(out[k][..., 3:]),
                                torch.cos(out[k][..., 3:]),
                            ],
                            -1,
                        )[..., 1:, :]

                    else:
                        # transform to global coordinate
                        xy_local = out[k][..., :2]
                        agent_v = out[k][..., 2:3]
                        h_local = out[k][..., 3:4]
                        xy_global = GeoUtils.batch_nd_transform_points(
                            xy_local, center_from_agents.unsqueeze(2)
                        )
                        h_global = h_local + curr_yaw.unsqueeze(2)
                        agent_xyvsc_by_type[k] = torch.cat(
                            [
                                xy_global,
                                agent_v,
                                torch.sin(h_global),
                                torch.cos(h_global),
                            ],
                            -1,
                        )[..., 1:, :]

                    agent_v = out[k][..., 2:3]

                    agent_acce[k] = (
                        out[k][..., 1:, 2:3] - out[k][..., :Tf, 2:3]
                    ) / self.dt
                    agent_acce[k] = torch.cat(
                        [agent_acce[k], agent_acce[k][..., -1:, :]], -2
                    )
                    agent_r[k] = torch.zeros_like(agent_v)

                else:
                    if isinstance(dyn, Unicycle):
                        xseq = dyn.forward_dynamics(
                            x0[k], out[k][..., :Tf, :], mode="chain", bound=False
                        )

                        agent_xyvsc_by_type[k] = dyn.state2xyvsc(xseq)

                        agent_acce[k] = out[k][..., :1]
                        agent_r[k] = out[k][..., 1:2]

                    elif isinstance(dyn, Unicycle_xyvsc):
                        xseq = dyn.forward_dynamics(
                            x0[k], out[k][..., :Tf, :], bound=False
                        )

                        agent_xyvsc_by_type[k] = xseq
                        agent_acce[k] = out[k][..., :1]
                        agent_r[k] = out[k][..., 1:2]
                    else:
                        raise NotImplementedError
                    state_out[k] = xseq
            future_xyvsc[..., 1:, :] = sum(
                [
                    agent_mask[k][..., None, None] * agent_xyvsc_by_type[k]
                    for k in agent_xyvsc_by_type
                ]
            )

            acce_combined = sum(
                [agent_mask[k][..., None, None] * agent_acce[k] for k in agent_acce]
            )
            r_combined = sum(
                [agent_mask[k][..., None, None] * agent_r[k] for k in agent_r]
            )
            agent_v = future_xyvsc[..., 2:3]
            # update the agent future aux

            # update the agent future raw [v,acce,r,static_features]

            agent_vu = torch.cat([agent_v, acce_combined, r_combined], -1)
            raw_agent_fut = torch.cat([agent_vu, raw_agent_fut[..., 3:]], -1)
        for k, v in out.items():
            if v is not None:
                input_out[k] = v[..., :Tf, :]
            else:
                input_out[k] = torch.zeros_like(out[k][..., :Tf, :2])

        trajectories = torch.cat(
            [
                future_xyvsc[:, :, 1:, :2],
                ratan2(future_xyvsc[:, :, 1:, 3:4], future_xyvsc[:, :, 1:, 4:5]),
            ],
            -1,
        )

        input_violation = dict()
        jerk = dict()
        for k, dyn in self.dyn.items():
            if dyn is not None:
                xl = torch.cat([x0[k].unsqueeze(-2), state_out[k][..., :-1, :]], -2)
                input_violation[k] = (
                    dyn.get_input_violation(xl, input_out[k][..., :Tf, :])
                    * agent_mask[k][..., None, None]
                )
                inputs_extended = torch.cat([u0[k].unsqueeze(-2), input_out[k]], -2)
                jerk[k] = (
                    (inputs_extended[..., 1:, :] - inputs_extended[..., :-1, :])
                    / self.dt
                ) * agent_mask[k][..., None, None]
        return trajectories, input_out, state_out, input_violation, jerk

    def decode_trajectory(
        self,
        enc_vars,
        raw_vars,
        aux_xs,
        var_masks,
        cross_masks,
        frame_indices,
        agent_type,
        lane_mode,
        homotopy,
        x0,
        u0,
        center_from_agents,
        num_samples=None,
    ):
        device = enc_vars[TFvars.Agent_hist].device
        B, N, Th = enc_vars[TFvars.Agent_hist].shape[:3]
        assert AgentType.VEHICLE in self.dyn
        agent_mask = dict()
        for k, v in self.dyn.items():
            if k == AgentType.VEHICLE:
                # this is the default mode
                other_types = [k.value for k in self.dyn if k != AgentType.VEHICLE]

                agent_mask[k] = torch.logical_not(
                    agent_type[..., other_types].any(-1)
                ) * agent_type.any(-1)
            else:
                agent_mask[k] = agent_type[..., k.value]
        if lane_mode.ndim == 4:
            # sample mode
            sample_mode = True
            num_samples = homotopy.shape[1]
            lane_mode, homotopy = TensorUtils.join_dimensions(
                (lane_mode, homotopy), 0, 2
            )
            B0 = B
            B = B0 * num_samples
            enc_vars = {
                k: v.repeat_interleave(num_samples, 0) for k, v in enc_vars.items()
            }
            raw_vars = {
                k: v.repeat_interleave(num_samples, 0) for k, v in raw_vars.items()
            }
            aux_xs = {k: v.repeat_interleave(num_samples, 0) for k, v in aux_xs.items()}
            var_masks = {
                k: v.repeat_interleave(num_samples, 0) for k, v in var_masks.items()
            }
            cross_masks = {
                k: v.repeat_interleave(num_samples, 0) for k, v in cross_masks.items()
            }
            frame_indices = {
                k: v.repeat_interleave(num_samples, 0) for k, v in frame_indices.items()
            }
            for k, v in x0.items():
                if v is not None:
                    x0[k] = v.repeat_interleave(num_samples, 0)
            for k, v in u0.items():
                if v is not None:
                    u0[k] = v.repeat_interleave(num_samples, 0)
            center_from_agents = center_from_agents.repeat_interleave(num_samples, 0)
            for k, v in agent_mask.items():
                agent_mask[k] = v.repeat_interleave(num_samples, 0)

        else:
            sample_mode = False

        lane_mode = F.one_hot(lane_mode, len(self.fut_lane_relation)).float()
        if self.null_lane_mode:
            lane_mode = lane_mode[..., :-1, :]
        homotopy = F.one_hot(homotopy, len(HomotopyType)).float()

        Tf = self.dec_T
        # create agent_future from agent_hist for auto-regressive decoding

        var_masks[TFvars.Agent_future] = (
            var_masks[TFvars.Agent_hist]
            .any(2, keepdim=True)
            .repeat_interleave(1 + Tf, 2)
        )
        aux_xs[TFvars.Agent_future] = aux_xs[TFvars.Agent_hist][
            :, :, -1:
        ].repeat_interleave(1 + Tf, 2)

        frame_indices[TFvars.Agent_future] = Th + torch.arange(
            1 + self.dec_T, device=device
        )[None, None, :].repeat(B, N, 1)

        future_xyvsc = list(torch.split(aux_xs[TFvars.Agent_future][..., :5], 1, dim=2))

        if self.AR_update_mode is not None:
            (
                trajectories,
                input_out,
                state_out,
                input_violation,
                jerk,
            ) = self.decode_trajectory_AR(
                enc_vars,
                raw_vars,
                aux_xs,
                var_masks,
                cross_masks,
                frame_indices,
                agent_mask,
                x0,
                u0,
                lane_mode,
                homotopy,
                center_from_agents,
            )
        else:
            (
                trajectories,
                input_out,
                state_out,
                input_violation,
                jerk,
            ) = self.decode_trajectory_OS(
                enc_vars,
                raw_vars,
                aux_xs,
                var_masks,
                cross_masks,
                frame_indices,
                agent_mask,
                x0,
                u0,
                lane_mode,
                homotopy,
                center_from_agents,
            )

        if sample_mode:
            trajectories = trajectories.view(B0, num_samples, N, Tf, -1)
            state_out = {
                k: v.view(B0, num_samples, N, Tf, -1) if v is not None else None
                for k, v in state_out.items()
            }
            input_out = {
                k: v.view(B0, num_samples, N, Tf, -1) if v is not None else None
                for k, v in input_out.items()
            }
            agent_mask = {k: v.view(B0, num_samples, N) for k, v in agent_mask.items()}
            input_violation = {
                k: v.view(B0, num_samples, N, Tf, -1)
                for k, v in input_violation.items()
            }
            jerk = {k: v.view(B0, num_samples, N, Tf, -1) for k, v in jerk.items()}

        return dict(
            trajectories=trajectories,
            states=state_out,
            inputs=input_out,
            input_violation=input_violation,
            jerk=jerk,
            type_mask=agent_mask,
            future_xyvsc=future_xyvsc,
        )

    def restore_lm_homotopy_from_joint_sample(
        self, joint_sample, indices, factor_idx, lm_factor_mask, N, M, M_lm
    ):
        device = joint_sample.device
        B, num_samples = joint_sample.shape[:2]
        factor_mask = F.one_hot(factor_idx, N * (N + M_lm)).sum(
            -2
        )  # (B, |combined_log_pi|)
        if not self.classify_a2l_4all_lanes:
            factor_mask = torch.cat(
                [
                    torch.ones(B, N * M, device=device, dtype=torch.int),
                    factor_mask[:, N * M_lm :],
                ],
                -1,
            )
            candidate_flag = torch.zeros(B, N, M, dtype=torch.bool, device=device)
            for i in range(M_lm):
                candidate_flag.scatter_(-1, indices[:, :, i], 1)  # Back to lanes
            lm_factor_mask = lm_factor_mask * candidate_flag

        if not self.classify_a2l_4all_lanes:
            lm_sample_per_mode = joint_sample[..., : N * M_lm].view(
                B, -1, N, M_lm
            )  # First N*M_lm are mode contributions
            lm_sample = torch.zeros(
                [B, joint_sample.size(1), N, M], dtype=torch.long, device=device
            )

            i = 0
            lm_sample_per_mode = lm_sample_per_mode.clip(0, indices.shape[-1] - 1)
            for mode in self.fut_lane_relation:
                if mode == self.fut_lane_relation.NOTON:
                    continue
                # restore from the important sampling of of lane segments
                restored_lm_samples_i = torch.gather(
                    indices[:, :, i].unsqueeze(1).repeat_interleave(num_samples, 1),
                    -1,
                    lm_sample_per_mode[..., i : i + 1],
                ).squeeze(-1)
                # value of the lane mode
                mode_enum = i + 1 if mode > self.fut_lane_relation.NOTON else i
                lm_sample = lm_sample + (lm_sample == 0) * (
                    torch.scatter(
                        lm_sample, -1, restored_lm_samples_i.unsqueeze(-1), mode_enum
                    )
                )
                i += 1
        else:
            lm_sample = joint_sample[..., : N * M].view(
                B, -1, N, M
            )  # .clip(0,len(self.fut_lane_relation)-1)
        lm_sample.masked_fill_(~lm_factor_mask.bool().unsqueeze(1), 0)
        homo_sample = joint_sample[..., N * M_lm :].view(B, -1, N, N)
        homo_sample = HomotopyType.enforce_symmetry(homo_sample)
        return lm_sample, homo_sample, factor_mask

    def forward(
        self,
        raw_vars,
        aux_xs,
        var_masks,
        cross_masks,
        frame_indices,
        agent_type,
        enc_edges=dict(),
        dec_edges=dict(),
        GT_lane_mode=None,
        GT_homotopy=None,
        center_from_agents=None,
        num_samples=None,
    ):
        enc_raw_vars = {
            k: v
            for k, v in raw_vars.items()
            if k in self.enc_vars or k in self.enc_edges
        }
        enc_vars = self.embed_raw(enc_raw_vars, aux_xs)

        x0 = dict()
        u0 = dict()
        for k, dyn in self.dyn.items():
            if dyn is not None:
                # get x0 and u0 from aux_xs
                if isinstance(dyn, Unicycle_xyvsc):
                    x0[k] = aux_xs[TFvars.Agent_hist][:, :, -1, :5]
                    xprev = aux_xs[TFvars.Agent_hist][:, :, -2, :5]
                    u0[k] = dyn.inverse_dyn(x0[k], xprev, self.dt)
                elif isinstance(dyn, Unicycle):
                    xyvsc = aux_xs[TFvars.Agent_hist][:, :, -2:, :5]
                    h = ratan2(xyvsc[..., 3:4], xyvsc[..., 4:5])
                    x0[k] = torch.cat([xyvsc[..., -1, :3], h[..., -1, :]], -1)
                    xprev = torch.cat([xyvsc[..., -2, :3], h[..., -2, :]], -1)
                    u0[k] = dyn.inverse_dyn(x0[k], xprev, self.dt)
                else:
                    raise NotImplementedError
            else:
                x0[k] = None
                u0[k] = None

        enc_kwargs = dict(
            aux_xs=aux_xs,
            var_masks=var_masks,
            cross_masks=cross_masks,
            frame_indices=frame_indices,
            edges=enc_edges,
        )
        enc_vars = self.encoder(enc_vars, enc_kwargs)
        mode_pred = self.predict_from_context(
            enc_vars,
            aux_xs,
            frame_indices,
            var_masks,
            cross_masks,
            enc_edges,
            GT_lane_mode,
            GT_homotopy,
            num_samples=num_samples,
        )

        # decode trajectory predictions
        # prepare modes

        if GT_lane_mode is not None and GT_homotopy is not None:
            # train mode
            if self.decode_num_modes == 1:
                lane_mode = GT_lane_mode
                homotopy = GT_homotopy
                mode_pred["dec_cond_lm"] = GT_lane_mode.unsqueeze(1)
                mode_pred["dec_cond_homotopy"] = GT_homotopy.unsqueeze(1)
                mode_pred["dec_cond_prob"] = torch.ones(
                    [GT_lane_mode.shape[0], 1], device=GT_lane_mode.device
                )
            else:
                lane_mode = mode_pred["dec_cond_lm"]
                homotopy = mode_pred["dec_cond_homotopy"]

        else:
            # infer mode
            lane_mode = mode_pred["lane_mode_sample"]
            homotopy = mode_pred["homotopy_sample"]

        dec_vars = self.decode_trajectory(
            enc_vars,
            raw_vars,
            aux_xs,
            var_masks,
            cross_masks,
            frame_indices,
            agent_type,
            lane_mode,
            homotopy,
            x0=x0,
            u0=u0,
            center_from_agents=center_from_agents,
        )
        return dec_vars, mode_pred
