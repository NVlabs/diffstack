"""
Transformer that accepts multiple types with separate attention
Adopted from the minGPT project https://github.com/karpathy/minGPT


"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0, resid_pdrop=0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, mask=None):
        # mask: (B,T,T)
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, None] == 0, float("-inf"))
            att = att.masked_fill((mask == 0).all(-1)[:, None, :, None], 0.0)
        finfo = torch.finfo(att.dtype)
        att = att.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AuxSelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        edge_dim,
        aux_edge_func=None,
        attn_pdrop=0,
        resid_pdrop=0,
        PE_len=None,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # assert edge_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_net = nn.Linear(n_embd, n_embd)
        self.k_net = nn.Linear(n_embd + edge_dim, n_embd)
        self.v_net = nn.Linear(n_embd + edge_dim, n_embd)
        self.aux_edge_func = aux_edge_func

        self.PE_len = PE_len
        if PE_len is not None:
            self.PE_q = nn.Embedding(PE_len, n_embd)
            self.PE_k = nn.Embedding(PE_len, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, aux_x, mask=None, edge=None, frame_indices=None):
        # mask: (B,T,T)
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_net(x)
        if edge is not None or self.aux_edge_func is not None:
            T = aux_x.size(1)
            if edge is None:
                edge = self.aux_edge_func(aux_x, aux_x)  # (B,T,T,aux_vardim)
            if self.k_net.weight.dtype == torch.float16:
                edge = edge.half()
            aug_x = torch.cat([x.unsqueeze(2).repeat_interleave(T, 2), edge], dim=-1)
            k = self.k_net(aug_x)
            v = self.v_net(aug_x)
            k = k.view(B, T, T, self.n_head, C // self.n_head).permute(
                0, 3, 1, 2, 4
            )  # (B, nh, T, T, hs)
            q = (
                q.view(B, T, self.n_head, 1, C // self.n_head)
                .transpose(1, 2)
                .repeat_interleave(T, 3)
            )  # (B, nh, T, T, hs)
            v = v.view(B, T, T, self.n_head, C // self.n_head).permute(
                0, 3, 1, 2, 4
            )  # (B, nh, T, T, hs)
            if self.PE_len is not None:
                q = q + self.PE_q(frame_indices).view(
                    B, T, self.n_head, 1, C // self.n_head
                ).transpose(1, 2).repeat_interleave(T, 3)
                k = k + self.PE_k(frame_indices).view(
                    B, T, self.n_head, C // self.n_head
                ).transpose(1, 2).unsqueeze(2).repeat_interleave(T, 2)
            # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q * k).sum(-1) * (1.0 / math.sqrt(k.size(-1)))
        else:
            aug_x = torch.cat([x, aux_x], dim=-1)
            k = self.k_net(aug_x)
            v = self.v_net(aug_x)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(
                1, 2
            )  # (B, nh, T, hs)
            if self.PE_len is not None:
                q = q + self.PE_q(frame_indices).view(
                    B, T, self.n_head, C // self.n_head
                ).transpose(1, 2)
                k = k + self.PE_k(frame_indices).view(
                    B, T, self.n_head, C // self.n_head
                ).transpose(1, 2)
            # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            att = att.masked_fill(mask[:, None] == 0, float("-inf"))
            att = att.masked_fill((mask == 0).all(-1)[:, None, :, None], 0.0)
        finfo = torch.finfo(att.dtype)
        att = att.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # if v.ndim==4, (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) elif v.ndim==5, (B, nh, T, T) x (B, nh, T, T, hs) -> (B, nh, T, hs)
        y = att @ v if v.ndim == 4 else (att.unsqueeze(-1) * v).sum(-2)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AuxCrossAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        edge_dim,
        aux_edge_func=None,
        attn_pdrop=0,
        resid_pdrop=0,
        PE_len=None,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # assert edge_dim % n_head == 0
        self.edge_dim = edge_dim
        # key, query, value projections for all heads, but in a batch
        self.q_net = nn.Linear(n_embd, n_embd)
        self.k_net = nn.Linear(n_embd + edge_dim, n_embd)
        self.v_net = nn.Linear(n_embd + edge_dim, n_embd)
        self.PE_len = PE_len
        if PE_len is not None:
            self.PE_q = nn.Embedding(PE_len, n_embd)
            self.PE_k = nn.Embedding(PE_len, n_embd)
        self.aux_edge_func = aux_edge_func
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(
        self,
        x1,
        x2,
        mask,
        aux_x1,
        aux_x2,
        frame_indices1=None,
        frame_indices2=None,
        edge=None,
    ):
        # mask: (B,T,T)
        (
            B,
            T1,
            C,
        ) = x1.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        T2 = x2.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_net(x1)
        if self.edge_dim > 0:
            if edge is None:
                if self.aux_edge_func is None:
                    # default to concatenation
                    edge = torch.cat(
                        [
                            aux_x1.unsqueeze(2).repeat_interleave(T2, 2),
                            aux_x2.unsqueeze(1).repeat_interleave(T1, 1),
                        ],
                        -1,
                    )  # (B,T1,T2,auxdim1+auxdim2)
                else:
                    edge = self.aux_edge_func(aux_x1, aux_x2)  # (B,T1,T2,aux_vardim)
            if self.k_net.weight.dtype == torch.float16:
                edge = edge.half()
            aug_x2 = torch.cat([x2.unsqueeze(1).repeat_interleave(T1, 1), edge], dim=-1)
        else:
            aug_x2 = x2.unsqueeze(1).repeat_interleave(T1, 1)

        k = self.k_net(aug_x2)
        v = self.v_net(aug_x2)
        k = k.view(B, T1, T2, self.n_head, C // self.n_head).permute(
            0, 3, 1, 2, 4
        )  # (B, nh, T1, T2, hs)
        q = (
            q.view(B, T1, self.n_head, 1, C // self.n_head)
            .transpose(1, 2)
            .repeat_interleave(T2, 3)
        )  # (B, nh, T1, T2, hs)
        v = v.view(B, T1, T2, self.n_head, C // self.n_head).permute(
            0, 3, 1, 2, 4
        )  # (B, nh, T1, T2, hs)
        if self.PE_len is not None:
            q = q + self.PE_q(frame_indices1).view(
                B, T1, self.n_head, 1, C // self.n_head
            ).transpose(1, 2).repeat_interleave(T2, 3)
            k = k + self.PE_k(frame_indices2).view(
                B, T2, self.n_head, C // self.n_head
            ).transpose(1, 2).unsqueeze(2).repeat_interleave(T1, 2)
        #     # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * k).sum(-1) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, None] == 0, float("-inf"))
            att = att.masked_fill((mask == 0).all(-1)[:, None, :, None], 0.0)
        finfo = torch.finfo(att.dtype)
        att = att.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # if v.ndim==4, (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) elif v.ndim==5, (B, nh, T, T) x (B, nh, T, T, hs) -> (B, nh, T, hs)
        y = att @ v if v.ndim == 4 else (att.unsqueeze(-1) * v).sum(-2)
        y = (
            y.transpose(1, 2).contiguous().view(B, T1, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    """
    A vanilla multi-head cross-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0, resid_pdrop=0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_q = nn.Linear(n_embd, n_embd)
        self.c_k = nn.Linear(n_embd, n_embd)
        self.c_v = nn.Linear(n_embd, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_pdrop = attn_pdrop
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, z, mask=None):
        # x: (B, T_in, C), query
        # z: (B, T_m, C), memory
        # mask: (B, T_in, T_m), mask for memory
        (
            B,
            T_in,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        T_m = z.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_q(x)
        k, v = self.c_k(z), self.c_v(z)

        k = k.view(B, T_m, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_m, hs)
        q = q.view(B, T_in, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_in, hs)
        v = v.view(B, T_m, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_m, hs)

        # cross-attention; Cross-attend: (B, nh, T_in, hs) x (B, nh, hs, T_m) -> (B, nh, T_in, T_m)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, None] == 0, float("-inf"))
            att = att.masked_fill((mask == 0).all(-1)[:, None, :, None], 0.0)
        finfo = torch.finfo(att.dtype)
        att = att.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T_in, T_m) x (B, nh, T_m, hs) -> (B, nh, T_in, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T_in, C)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))

        return y


class TypeSelfAttention(nn.Module):
    """a composite attention block supporting multiple attention types"""

    def __init__(
        self,
        ntype,
        n_embd,
        n_head,
        edge_dim=0,
        aux_edge_func=None,
        attn_pdrop=0,
        resid_pdrop=0,
    ):
        super().__init__()
        self.ntype = ntype
        self.edge_dim = edge_dim
        if edge_dim > 0:
            self.attn = nn.ModuleList(
                [
                    AuxSelfAttention(
                        n_embd, n_head, edge_dim, aux_edge_func, attn_pdrop, resid_pdrop
                    )
                    for _ in range(self.ntype)
                ]
            )
        else:
            self.attn = nn.ModuleList(
                [
                    SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
                    for _ in range(self.ntype)
                ]
            )

    def forward(self, x, aux_x, masks, edge=None):
        if self.ntype == 1 and isinstance(masks, torch.Tensor):
            masks = [masks]
        assert len(masks) == self.ntype
        if edge is not None:
            if isinstance(edge, list):
                assert len(edge) == self.ntype
            elif isinstance(edge, torch.Tensor):
                edge = [edge] * self.ntype
        else:
            edge = [None] * self.ntype
        resid = torch.zeros_like(x)
        if self.edge_dim > 0:
            assert aux_x is not None
            for i, mask in enumerate(masks):
                resid = resid + self.attn[i](x, aux_x, mask, edge=edge[i])
        else:
            for i, mask in enumerate(masks):
                resid = resid + self.attn[i](x, mask, edge=edge[i])
        return resid


class TypeCrossAttention(nn.Module):
    """a composite attention block supporting multiple attention types"""

    def __init__(
        self,
        ntype,
        n_embd,
        n_head,
        edge_dim,
        aux_edge_func=None,
        attn_pdrop=0,
        resid_pdrop=0,
    ):
        super().__init__()
        self.ntype = ntype
        self.edge_dim = edge_dim
        if edge_dim > 0:
            self.attn = nn.ModuleList(
                [
                    AuxCrossAttention(
                        n_embd, n_head, edge_dim, aux_edge_func, attn_pdrop, resid_pdrop
                    )
                    for _ in range(self.ntype)
                ]
            )
        else:
            self.attn = nn.ModuleList(
                [
                    CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
                    for _ in range(self.ntype)
                ]
            )

    def forward(self, x1, x2, masks, aux_x1=None, aux_x2=None, edge=None):
        if self.ntype == 1 and isinstance(masks, torch.Tensor):
            masks = [masks]
        assert len(masks) == self.ntype
        if edge is not None:
            if isinstance(edge, list):
                assert len(edge) == self.ntype
            elif isinstance(edge, torch.Tensor):
                edge = [edge] * self.ntype
        else:
            edge = [None] * self.ntype
        resid = torch.zeros_like(x1)
        if self.edge_dim > 0:
            for i, mask in enumerate(masks):
                resid = resid + self.attn[i](x1, x2, mask, aux_x1, aux_x2, edge=edge[i])
        else:
            for i, mask in enumerate(masks):
                resid = resid + self.attn[i](x1, x2, mask, edge=edge[i])
        return resid


class EncBlock(nn.Module):
    """an unassuming Transformer encoder block"""

    def __init__(self, n_embd, n_head, attn_pdrop=0, resid_pdrop=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class DecBlock(nn.Module):
    """an unassuming Transformer decoder block"""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, z, mask=None):
        x = x + self.attn(self.ln_1(x), z, mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class TypeEncBlock(nn.Module):
    """an unassuming Transformer encoder block supporting multiple attention types"""

    def __init__(self, n_embd, ntype, n_head, attn_pdrop=0, resid_pdrop=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ntype = ntype
        self.attn = nn.ModuleList(
            [
                SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
                for _ in range(self.ntype)
            ]
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, masks):
        assert len(masks) == self.ntype
        for i, mask in enumerate(masks):
            x = x + self.attn[i](self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class TypeDecBlock(nn.Module):
    """an unassuming Transformer decoder block supporting multiple attention types"""

    def __init__(self, n_embd, ntype, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ntype = ntype
        self.attn = nn.ModuleList(
            [
                CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
                for _ in range(self.ntype)
            ]
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, z, masks):
        assert len(masks) == self.ntype
        for i, mask in enumerate(masks):
            x = x + self.attn[i](self.ln_1(x), z, mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_embd,
        ntype,
        n_head,
        attn_pdrop,
        resid_pdrop,
        nblock,
        indim=None,
        outdim=None,
    ):
        super().__init__()
        self.nblock = nblock
        self.indim = indim
        self.outdim = outdim
        self.in_embd = (
            nn.Linear(self.indim, n_embd)
            if self.indim is not None
            else torch.nn.Identity()
        )
        self.out_proj = (
            nn.Linear(n_embd, self.outdim)
            if self.outdim is not None
            else torch.nn.Identity()
        )
        self.ntype = ntype
        if self.ntype == 1:
            self.blocks = nn.ModuleDict(
                {
                    "block_{}".format(i): EncBlock(
                        n_embd, n_head, attn_pdrop, resid_pdrop
                    )
                    for i in range(self.nblock)
                }
            )
        else:
            self.blocks = nn.ModuleDict(
                {
                    "block_{}".format(i): TypeEncBlock(
                        n_embd, ntype, n_head, attn_pdrop, resid_pdrop
                    )
                    for i in range(self.nblock)
                }
            )

    def forward(self, x, mask=None):
        x = self.in_embd(x)
        for i in range(self.nblock):
            x = self.blocks["block_{}".format(i)](x, mask)
        x = self.out_proj(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_embd,
        ntype,
        n_head,
        attn_pdrop,
        resid_pdrop,
        nblock,
        indim=None,
        outdim=None,
    ):
        super().__init__()
        self.nblock = nblock
        self.indim = indim
        self.outdim = outdim
        self.in_embd = (
            nn.Linear(self.indim, n_embd)
            if self.indim is not None
            else torch.nn.Identity()
        )
        self.out_proj = (
            nn.Linear(n_embd, self.outdim)
            if self.outdim is not None
            else torch.nn.Identity()
        )
        self.ntype = ntype
        if self.ntype == 1:
            self.blocks = nn.ModuleDict(
                {
                    "block_{}".format(i): DecBlock(
                        n_embd, n_head, attn_pdrop, resid_pdrop
                    )
                    for i in range(self.nblock)
                }
            )
        else:
            self.blocks = nn.ModuleDict(
                {
                    "block_{}".format(i): TypeDecBlock(
                        n_embd, ntype, n_head, attn_pdrop, resid_pdrop
                    )
                    for i in range(self.nblock)
                }
            )

    def forward(self, x, z, mask=None):
        x = self.in_embd(x)
        for i in range(self.nblock):
            x = self.blocks["block_{}".format(i)](x, z, mask)
        x = self.out_proj(x)
        return x


def test():
    n_embd = 256
    ntype = 3
    n_head = 8
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    enc = TransformerEncoder(
        n_embd, ntype, n_head, attn_pdrop, resid_pdrop, nblock=2, indim=15
    ).cuda()
    dec = TransformerDecoder(
        n_embd, ntype, n_head, attn_pdrop, resid_pdrop, nblock=2, indim=15, outdim=40
    ).cuda()
    b = 2
    T_in = 10
    T_m = 20
    xin = torch.randn(b, T_in, 15).cuda()

    type = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2]).cuda()
    type_flag = [(type == i)[None].repeat_interleave(b, 0) for i in range(3)]
    enc_masks = [fg[..., None] * fg[..., None, :] for fg in type_flag]

    mask1 = torch.ones(2, T_in, T_m).bool().cuda()
    mask2 = torch.tril(mask1)
    mask3 = torch.triu(mask1)
    dec_masks = [mask1, mask2, mask3]
    mem = enc(xin, mask=enc_masks)
    mem = torch.repeat_interleave(mem, 2, 1)
    x1 = dec(xin, mem, [mask1, mask2, mask3])
    print("done")


def test_edge_func():
    n_embd = 256
    n_head = 8
    aux_vardim = 16
    aux_edge_func = lambda x, y: x.unsqueeze(2) * y.unsqueeze(1)

    attn = AuxCrossAttention(n_embd, n_head, aux_vardim, aux_edge_func)
    b = 2
    N1 = 3
    N2 = 4
    x1 = torch.randn([b, N1, n_embd])
    aux_x1 = torch.randn([b, N1, aux_vardim])

    x2 = torch.randn([b, N2, n_embd])
    aux_x2 = torch.randn([b, N2, aux_vardim])
    xx = attn(x1, x2, None, aux_x1, aux_x2)
    print("123")


if __name__ == "__main__":
    test_edge_func()
