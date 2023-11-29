import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from diffstack.configs.config import Dict
from diffstack.models.TypeTransformer import NewGELU, zero_module
from diffstack.utils.model_utils import *

"""
modified from RPE.py with the following changes:
1. removed diffusion time embedding
2. removed the D dimension
3. reversed the order of T and C
4. changed attn_mask to be [B,T,T] instead of [B,T]
5. removed the residual connection, instead, output the residual itself
6. removed the batchnorm
"""
class sRPENet(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.embed_distances = nn.Linear(3, n_embd)

        self.gelu = NewGELU()
        self.out = nn.Linear(n_embd, n_embd)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.n_embd = n_embd
        self.num_heads = num_heads

    def forward(self, relative_distances):
        distance_embs = torch.stack(
            [torch.log(1+(relative_distances).clamp(min=0)),
             torch.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        if self.embed_distances.weight.dtype==torch.float16:
            distance_embs = distance_embs.half()
        emb = self.embed_distances(distance_embs)
        return self.out(self.gelu(emb)).view(*relative_distances.shape, self.num_heads, self.n_embd//self.num_heads)

class ProdPENet(nn.Module):
    # embed the tuple of two positions [P1,P2] into positional embedding
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.embed_distances = nn.Linear(5, n_embd)

        self.gelu = NewGELU()
        self.out = nn.Linear(n_embd, n_embd)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.n_embd = n_embd
        self.num_heads = num_heads

    def forward(self, relative_distances,P1,P2):
        distance_embs = torch.stack(
            [torch.log(1+(relative_distances).clamp(min=0)),
             torch.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float(),
             torch.log(1+P1),
             torch.log(1+P2)],
            dim=-1
        )  # BxTxTx3
        if self.embed_distances.weight.dtype==torch.float16:
            distance_embs = distance_embs.half()
        emb = self.embed_distances(distance_embs)
        return self.out(self.gelu(emb)).view(*relative_distances.shape, self.num_heads, self.n_embd//self.num_heads)

class sRPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, n_embd, num_heads, use_rpe_net=False):
        """ This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = sRPENet(n_embd, num_heads)
        else:
            self.prod_pe_net = ProdPENet(n_embd, num_heads)
            # raise NotImplementedError
            # self.lookup_table_weight = nn.Parameter(
            #     torch.zeros(2 * self.beta + 1,
            #              self.num_heads,
            #              self.head_dim))

    def get_R(self, pairwise_distances,P1=None,P2=None):
        if self.use_rpe_net:
            return self.rpe_net(pairwise_distances)
        else:
            return self.prod_pe_net(pairwise_distances,P1,P2)
            # return self.lookup_table_weight[pairwise_distances]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, mode,P1=None,P2=None):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances,P1=P1,P2=P2)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances,P1=P1,P2=P2)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances,P1=None,P2=None):
        # qv is either of q or k and has shape BxHxTx(C/H)
        # Output shape should be # BxHxTxT
        R = self.get_R(pairwise_distances,P1=P1,P2=P2)
        if qk.ndim==4:
            return torch.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
                "bhtf,btshf->bhts", qk, R  # BxHxTxT
            )
        elif qk.ndim==5:
            return torch.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
                "bhtsf,btshf->bhts", qk, R  # BxHxTxT
            )

    def forward_v(self, attn, pairwise_distances,P1=None,P2=None):
        # attn has shape BxHxT1xT2
        # Output shape should be # BxHxT1x(C/H)
        R = self.get_R(pairwise_distances,P1=P1,P2=P2)
        return torch.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
                "bhts,btshf->bhtf", attn, R  # BxHxTxT
            )

    def forward_safe_qk(self, x, pairwise_distances):
        R = self.get_R(pairwise_distances)
        B, T, _, H, F = R.shape
        res = x.new_zeros(B, H, T, T) # attn shape
        for b in range(B):
            for h in range(H):
                for i in range(T):
                    for j in range(T):
                        res[b, h, i, j] = x[b, h, i].dot(R[b, i, j, h])
        return res
    
class sRPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, n_embd, num_heads, use_checkpoint=False,use_rpe_net=None,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = n_embd // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = nn.Linear(n_embd, n_embd * 3)
        self.proj_out = zero_module(nn.Linear(n_embd, n_embd))

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return sRPE(
                n_embd=n_embd, num_heads=num_heads, use_rpe_net=use_rpe_net,
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, attn_mask, frame_indices, attn_weights_list=None):
        out, attn = checkpoint(self._forward, (x, attn_mask, frame_indices), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, T, C = x.shape
            attn_weights_list.append(attn.detach().view(B, -1, T, T).mean(dim=1).abs())  # logging attn weights
        return out

    def _forward(self, x, attn_mask, frame_indices):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = torch.einsum("BTtHF -> tBHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q, k, v shapes: BxHxTx(C/H)
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) # BxDxHxTxT
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2)) # BxTxT
        # relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, mode="qk")
        # relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, pairwise_distances, mode="qk")

        # softmax where all elements with mask==0 can attend to eachother and all with mask==1
        # can attend to eachother (but elements with mask==0 can't attend to elements with mask==1)
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = torch.inf
                w = w - inf_mask.view(B, 1, 1, T, T)  # BxDxHxTxT
                w.masked_fill_((attn_mask==0).all(-1).view(B,1,1,T,1),0)
            finfo = torch.finfo(w.dtype)
            w = w.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
            return torch.softmax(w.float(), dim=-1).type(w.dtype)
        if attn_mask is None:
            attn_mask = torch.ones_like(attn[:,0])
        attn = softmax(attn, attn_mask.type(attn.dtype))
        out = attn @ v
        # relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, mode="v")
        out = torch.einsum("BHTF -> BTHF", out).reshape(B, T, C)
        out = self.proj_out(out)
        return out, attn
    
class AuxPEAttention(nn.Module):
    def __init__(self, n_embd, num_heads, max_len = 100,attn_pdrop=0.0,resid_pdrop = 0.0,aux_vardim=0):
        super().__init__()
        
        self.num_heads = num_heads
        self.PE_q = nn.Embedding(max_len, n_embd)
        self.PE_k = nn.Embedding(max_len, n_embd)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)        

        self.aux_vardim = aux_vardim
        self.qnet = nn.Linear(n_embd, n_embd)
        self.knet = nn.Linear(n_embd+aux_vardim, n_embd)
        self.vnet = nn.Linear(n_embd+aux_vardim, n_embd)
        self.proj_out = zero_module(nn.Linear(n_embd, n_embd))

    def forward(self, x, aux_x, attn_mask, frame_indices):
        
        B, T, C = x.shape
        if aux_x is not None:
            x_aug = torch.cat([x, aux_x], dim=2)
        else:
            x_aug = x
        q,k,v = self.qnet(x).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2),\
                self.knet(x_aug).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2),\
                self.vnet(x_aug).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2)
        
        # q, k, v shapes: BxHxTx(C/H)
        q = q + self.PE_q(frame_indices).view(B,T,self.num_heads,C//self.num_heads)
        k = k + self.PE_k(frame_indices).view(B,T,self.num_heads,C//self.num_heads)
        
        out = F.scaled_dot_product_attention(q,k,v,attn_mask[:,None].repeat_interleave(self.num_heads,1),dropout=0.0)
        out = out.nan_to_num(0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        out = self.resid_dropout(self.proj_out(out))
        return out
    


class sAuxRPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, n_embd, num_heads, aux_vardim, use_checkpoint=False,
                 use_rpe_net=None, use_rpe_q=True, use_rpe_k=True, use_rpe_v=True
                 ):
        super().__init__()
        assert aux_vardim>0
        self.aux_vardim = aux_vardim
        self.num_heads = num_heads
        head_dim = n_embd // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qnet = nn.Linear(n_embd, n_embd)
        self.knet = nn.Linear(n_embd+aux_vardim, n_embd)
        self.vnet = nn.Linear(n_embd+aux_vardim, n_embd)
        self.proj_out = zero_module(nn.Linear(n_embd, n_embd))

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return sRPE(
                n_embd=n_embd, num_heads=num_heads, use_rpe_net=use_rpe_net,
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, aux_x, attn_mask, frame_indices, attn_weights_list=None,**kwargs):
        out, attn = checkpoint(self._forward, (x, aux_x, attn_mask, frame_indices), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, T, C = x.shape
            attn_weights_list.append(attn.detach().view(B, -1, T, T).mean(dim=1).abs())  # logging attn weights
        return out

    def _forward(self, x, aux_x, attn_mask, frame_indices):
        B, T, C = x.shape
        x_aug = torch.cat([x, aux_x], dim=2)

        q = self.qnet(x).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2)
        k = self.knet(x_aug).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2)
        v = self.vnet(x_aug).reshape(B,T,self.num_heads,C//self.num_heads).transpose(1,2)
        # q, k, v shapes: BxHxTx(C/H)
        q *= self.scale
        attn = (q @ k.transpose(-2, -1)) # BxHxTxT
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2)) # BxTxT
        # relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, mode="qk")
        # relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, pairwise_distances, mode="qk")

        # softmax where all elements with mask==0 can attend to eachother and all with mask==1
        # can attend to eachother (but elements with mask==0 can't attend to elements with mask==1)
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = torch.inf
                w = w - inf_mask.view(B, 1, T, T)  # BxHxTxT
                w.masked_fill_((attn_mask[:,None]==0).all(-1).unsqueeze(-1),0)
            finfo = torch.finfo(w.dtype)
            w = w.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
            return torch.softmax(w.float(), dim=-1).type(w.dtype)
        if attn_mask is None:
            attn_mask = torch.ones_like(attn[:,0])
        attn = softmax(attn, attn_mask.type(attn.dtype))
        out = attn @ v
        # relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, mode="v")
        out = torch.einsum("BHTF -> BTHF", out).reshape(B, T, C)
        out = self.proj_out(out)
        return out, attn
    
class sAuxRPECrossAttention(nn.Module):
    """ RPE cross attention with auxillary variable

    """
    def __init__(self, n_embd, num_heads, edge_dim,aux_edge_func=None, use_checkpoint=False,
                 use_rpe_net=None, use_rpe_k=True, use_rpe_v=True
                 ):
        super().__init__()
        # assert edge_dim% num_heads==0
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        head_dim = n_embd // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qnet = nn.Linear(n_embd, n_embd)
        self.knet = nn.Linear(n_embd+edge_dim, n_embd)
        self.vnet = nn.Linear(n_embd+edge_dim, n_embd)
        self.proj_out = zero_module(nn.Linear(n_embd, n_embd))
        self.aux_edge_func = aux_edge_func

        if use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None
        def make_rpe_func():
            return sRPE(
                n_embd=n_embd, num_heads=num_heads, use_rpe_net=use_rpe_net,
            )
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x1, x2, attn_mask, aux_x1,aux_x2, frame_indices1, frame_indices2, edge = None,attn_weights_list=None):
        out, attn = checkpoint(self._forward, (x1, x2, attn_mask, aux_x1, aux_x2, frame_indices1, frame_indices2,edge), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, T1, C = x1.shape
            T2 = x2.shape[1]
            attn_weights_list.append(attn.detach().view(B, -1, T1, T2).mean(dim=1).abs())  # logging attn weights
        return out

    def _forward(self, x1, x2, attn_mask, aux_x1, aux_x2, frame_indices1, frame_indices2,edge=None):
        B, T1, C = x1.shape
        T2 = x2.shape[1]
        q = self.qnet(x1)
        if self.edge_dim>0:
            if edge is None:
                if self.aux_edge_func is None:
                    # default to concatenation
                    edge = torch.cat([aux_x1.unsqueeze(2).repeat_interleave(T2,2),
                                    aux_x2.unsqueeze(1).repeat_interleave(T1,1)],-1) # (B,T1,T2,auxdim1+auxdim2)
                else:
                    edge = self.aux_edge_func(aux_x1,aux_x2) # (B,T1,T2,aux_vardim)
            if self.knet.weight.dtype==torch.float16:
                edge = edge.half()
            aug_x2 = torch.cat([x2.unsqueeze(1).repeat_interleave(T1,1), edge], dim=-1)
        else:
            aug_x2 = x2.unsqueeze(1).repeat_interleave(T1,1)
        
        k = self.knet(aug_x2).reshape(B,T1, T2, self.num_heads,C//self.num_heads).permute(0, 3, 1, 2,4) #B,nh,T1,T2,C//nh
        v = self.vnet(aug_x2).reshape(B,T1,T2,self.num_heads,C//self.num_heads).permute(0, 3, 1, 2,4) #B,nh,T1,T2,C//nh

        q = (q*self.scale).view(B, T1, self.num_heads, 1, C // self.num_heads).transpose(1, 2).repeat_interleave(T2,3) # (B, nh, T1, T2, hs)
        attn = (q * k).sum(-1) * (1.0 / math.sqrt(k.size(-1)))
        if self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices1.unsqueeze(-1) - frame_indices2.unsqueeze(-2)) # BxT1xT2
        # relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, mode="qk",P1=frame_indices1.unsqueeze(-1).expand(*pairwise_distances.shape),\
                P2=frame_indices2.unsqueeze(-2).expand(*pairwise_distances.shape))

        # softmax where all elements with mask==0 can attend to eachother and all with mask==1
        # can attend to eachother (but elements with mask==0 can't attend to elements with mask==1)
        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask
                inf_mask = (1-allowed_interactions)
                inf_mask[inf_mask == 1] = torch.inf
                w = w - inf_mask.view(B, 1, T1, T2)  # BxHxTxT
                w.masked_fill_((attn_mask[:,None]==0).all(-1).unsqueeze(-1),0)
            finfo = torch.finfo(w.dtype)
            w = w.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
            return torch.softmax(w.float(), dim=-1).type(w.dtype)
        if attn_mask is None:
            attn_mask = torch.ones_like(attn[:,0])
        attn = softmax(attn, attn_mask.type(attn.dtype))
        out = attn @ v if v.ndim==4 else (attn.unsqueeze(-1)*v).sum(-2)
        # relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, mode="v",P1=frame_indices1.unsqueeze(-1).expand(*pairwise_distances.shape),\
                P2=frame_indices2.unsqueeze(-2).expand(*pairwise_distances.shape))
        out = torch.einsum("BHTF -> BTHF", out).reshape(B, T1, C)
        out = self.proj_out(out)
        return out, attn
    
def testAux():

    net = sAuxRPEAttention(n_embd=32, num_heads=4, aux_vardim=4,use_rpe_net=True)
    B,C,T = 3,32,10
    x = torch.randn(B,T,C)
    aux_x = torch.randn(B,T,4)
    frame_indices = torch.arange(T).unsqueeze(0).repeat(B,1)
    out = net(x, aux_x, None, frame_indices)

def testAuxCross():
    net = sAuxRPECrossAttention(n_embd=32, num_heads=4, edge_dim=4,use_rpe_net=True)
    B,C,T1,T2 = 3,32,10,5
    x1 = torch.randn(B,T1,C)
    x2 = torch.randn(B,T2,C)
    aux_x1 = torch.randn(B,T1,4)
    aux_x2 = torch.randn(B,T2,4)
    frame_indices1 = torch.arange(T1).unsqueeze(0).repeat(B,1)
    frame_indices2 = torch.arange(T2).unsqueeze(0).repeat(B,1)+T1
    out = net(x1,x2, None, aux_x1, aux_x2, frame_indices1, frame_indices2)
    print("123")
    
if __name__ == "__main__":
    testAuxCross()