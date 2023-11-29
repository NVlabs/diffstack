import torch
import torch.nn.functional as F
import numpy as np
from diffstack.models.base_models import MLP
import torch.nn as nn
from diffstack.utils.geometry_utils import round_2pi, batch_proj_xysc, rel_xysc
import community as community_louvain
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import torch.distributions as td
import diffstack.utils.tensor_utils as TensorUtils
import math
from diffstack.models.TypeTransformer import CrossAttention

class PED_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(PED_PED_encode, self).__init__()
        self.FC = MLP(10, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        input = torch.cat((deltax, x1[..., 2:4], x2[..., 2:4], size1, size2), dim=-1)
        return self.FC(input)


class PED_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(PED_VEH_encode, self).__init__()
        self.FC = MLP(10, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        veh_vel = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(x2[..., 3]), dim=-1),
                torch.unsqueeze(x2[..., 2] * torch.sin(x2[..., 3]), dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat((deltax, x1[..., 2:4], veh_vel, size1, size2), dim=-1)
        return self.FC(input)


class VEH_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(VEH_PED_encode, self).__init__()
        self.FC = MLP(9, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dv = torch.cat(
            (
                torch.unsqueeze(
                    x2[..., 2] * torch.cos(theta)
                    + torch.sin(theta) * x2[..., 3]
                    - x1[..., 2],
                    dim=-1,
                ),
                torch.unsqueeze(
                    x2[..., 3] * torch.cos(theta) - torch.sin(theta) * x2[..., 2],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        input = torch.cat(
            (dx, torch.unsqueeze(x1[..., 2], dim=-1), dv, size1, size2), dim=-1
        )
        return self.FC(input)


class VEH_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(VEH_VEH_encode, self).__init__()
        self.FC = MLP(11, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dtheta = x2[..., 3] - x1[..., 3]
        dv = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(dtheta) - x1[..., 2], dim=-1),
                torch.unsqueeze(torch.sin(dtheta) * x2[..., 2], dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat(
            (
                dx,
                torch.unsqueeze(x1[..., 2], dim=-1),
                dv,
                torch.unsqueeze(torch.cos(dtheta), dim=-1),
                torch.unsqueeze(torch.sin(dtheta), dim=-1),
                size1,
                size2,
            ),
            dim=-1,
        )
        return self.FC(input)


def PED_rel_state(x, x0):
    rel_x = torch.clone(x)
    rel_x[..., 0:2] -= x0[..., 0:2]
    return rel_x


def VEH_rel_state(x, x0):
    rel_XY = x[..., 0:2] - x0[..., 0:2]
    theta = x0[..., 3]
    rel_x = torch.stack(
        [
            rel_XY[..., 0] * torch.cos(theta) + rel_XY[..., 1] * torch.sin(theta),
            rel_XY[..., 1] * torch.cos(theta) - rel_XY[..., 0] * torch.sin(theta),
            x[..., 2],
            x[..., 3] - x0[..., 3],
        ],
        dim=-1,
    )
    rel_x[..., 3] = round_2pi(rel_x[..., 3])
    return rel_x


class PED_pre_encode(nn.Module):
    def __init__(self, enc_dim, hidden_dim=[64], use_lane_info=False):
        super(PED_pre_encode, self).__init__()
        self.FC = MLP(4, enc_dim, hidden_dim)

    def forward(self, x):
        return self.FC(x)


class VEH_pre_encode(nn.Module):
    def __init__(self, enc_dim, hidden_dim=[64], use_lane_info=False):
        super(VEH_pre_encode, self).__init__()
        self.use_lane_info = use_lane_info
        if use_lane_info:
            self.FC = MLP(8, enc_dim, hidden_dim)
        else:
            self.FC = MLP(5, enc_dim, hidden_dim)

    def forward(self, x):
        if self.use_lane_info:
            input = torch.cat(
                (
                    x[..., 0:3],
                    torch.cos(x[..., 3:4]),
                    torch.sin(x[..., 3:4]),
                    x[..., 4:5],
                    torch.cos(x[..., 5:6]),
                    torch.sin(x[..., 5:6]),
                ),
                dim=-1,
            )
        else:
            input = torch.cat(
                (x[..., 0:3], torch.cos(x[..., 3:]), torch.sin(x[..., 3:])), dim=-1
            )
        return self.FC(input)

def break_graph(M, resol=1.0):
    if isinstance(M, np.ndarray):
        resol = resol * np.max(M)
        G = nx.Graph()
        for i in range(M.shape[0]):
            G.add_node(i)
        for i in range(M.shape[0]):
            for j in range(i + 1, M.shape[0]):
                if M[i, j] > 0:
                    G.add_edge(i, j, weight=M[i, j])
        partition = community_louvain.best_partition(G, resolution=resol)
    elif isinstance(M, nx.classes.graph.Graph):
        G = M
        partition = community_louvain.best_partition(G, resolution=resol)

    while max(partition.values()) == 0 and resol >= 0.1:
        resol = resol * 0.9
        partition = community_louvain.best_partition(G, resolution=resol)
    return partition


def break_graph_recur(M, max_num):
    n_components, labels = connected_components(
        csgraph=csr_matrix(M), directed=False, return_labels=True
    )
    idx = 0

    while idx < n_components:
        subset = np.where(labels == idx)[0]
        if subset.shape[0] <= max_num:
            idx += 1
        else:
            partition = break_graph(M[np.ix_(subset, subset)])
            added_partition = 0
            for i in range(subset.shape[0]):
                if partition[i] > 0:
                    labels[subset[i]] = n_components + partition[i] - 1
                    added_partition = max(added_partition, partition[i])

            n_components += added_partition
            if added_partition == 0:
                idx += 1

    return n_components, labels

def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))

class Normal:

    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu
    def pseudo_sample(self,n_sample):
        sigma_points = torch.stack((self.mu,self.mu-self.sigma,self.mu+self.sigma),1)
        if n_sample<=1:
            return sigma_points[:,:n_sample]
        else:
            remain_n = n_sample-3
            sigma_tiled = self.sigma.unsqueeze(1).repeat_interleave(remain_n,1)
            mu_tiled = self.mu.unsqueeze(1).repeat_interleave(remain_n,1)
            sample = torch.randn_like(sigma_tiled)*sigma_tiled+mu_tiled
            return torch.cat([sigma_points,sample],1)

class Categorical:

    def __init__(self, probs=None, logits=None, temp=0.01):
        super().__init__()
        self.logits = logits
        self.temp = temp
        if probs is not None:
            self.probs = probs
        else:
            assert logits is not None
            self.probs = torch.softmax(logits, dim=-1)
        self.dist = td.OneHotCategorical(self.probs)

    def rsample(self,n_sample=1):
        relatex_dist = td.RelaxedOneHotCategorical(self.temp, self.probs)
        return relatex_dist.rsample((n_sample,)).transpose(0,-2)

    def sample(self):
        return self.dist.sample()
    
    def pseudo_sample(self,n_sample):
        D = self.probs.shape[-1]
        idx = self.probs.argsort(-1,descending=True)
        assert n_sample<=D
        return TensorUtils.to_one_hot(idx[...,:n_sample],num_class=D)



    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            p = Categorical(logits=torch.zeros_like(self.probs))
        kl = td.kl_divergence(self.dist, p.dist)
        return kl

    def mode(self):
        argmax = self.probs.argmax(dim=-1)
        one_hot = torch.zeros_like(self.probs)
        one_hot.scatter_(1, argmax.unsqueeze(1), 1)
        return one_hot

        

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

class AFMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())        

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

def rotation_2d_torch(x, theta, origin=None):
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x


class ExpParamAnnealer(nn.Module):

    def __init__(self, start, finish, rate, cur_epoch=0):
        super().__init__()
        self.register_buffer('start', torch.tensor(start))
        self.register_buffer('finish', torch.tensor(finish))
        self.register_buffer('rate', torch.tensor(rate))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.finish - (self.finish - self.start) * (self.rate ** self.cur_epoch)
    
class IntegerParamAnnealer(nn.Module):

    def __init__(self, start, finish, length, cur_epoch=0):
        super().__init__()
        self.register_buffer('start', torch.tensor(start))
        self.register_buffer('finish', torch.tensor(finish))
        self.register_buffer('length', torch.tensor(length))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.finish if self.cur_epoch>=self.length else self.start+int((self.finish-self.start)*self.cur_epoch/self.length)



class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)



def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(8, channels)

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def Gaussian_RBF_conv(sigma,radius,device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.range(-radius,radius)
    Y = torch.range(-radius,radius)
    XY = torch.meshgrid(X,Y)
    dis_sq = XY[0]**2+XY[1]**2
    RBF = torch.exp(-dis_sq/(2*sigma**2)).to(device)
    RBF = torch.nn.Parameter(RBF[None,None]/RBF.sum(),requires_grad=False)
    net = nn.Conv2d(1,1,2*radius+1,bias=False,padding=radius,device=device)
    net.weight = RBF
    return net
    
def agent2agent_edge(x1,x2,padto=None,scale=1,clip=[-np.inf,np.inf]):
    # assumming x1 and x2 are state of the form [x,y,v,s,c,w,l,type_one_hot]
    N1 = x1.shape[1]
    N2 = x2.shape[1]
    x1_flag = torch.logical_not((x1==0).all(-1)).type(x1.dtype)
    x2_flag = torch.logical_not((x2==0).all(-1)).type(x2.dtype)
    x1 = x1.unsqueeze(2).repeat_interleave(N2,2)
    x2 = x2.unsqueeze(1).repeat_interleave(N1,1)
    x1_xysc = x1[...,[0,1,3,4]]
    x2_xysc = x2[...,[0,1,3,4]]
    dx_xysc = rel_xysc(x1_xysc,x2_xysc)*(x1_flag.unsqueeze(2)*x2_flag.unsqueeze(1)).unsqueeze(-1)
    dx_xy = (dx_xysc[...,:2]/scale).clip(min=clip[0],max=clip[1])

    dx_xysc = torch.cat([dx_xy,dx_xysc[...,2:]],-1)
    
    v1 = x1[...,2:3]
    v2 = x2[...,2:3]
    edge = torch.cat([dx_xysc,v2*dx_xysc[...,2:3],v2*dx_xysc[...,3:4],x2[...,5:]],-1)
    if padto is not None:
        edge = torch.cat((edge,torch.zeros(*edge.shape[:-1],padto-edge.shape[-1],device=edge.device)),-1)
    return edge

def agent2lane_edge_proj(x1,x2,padto=None,scale=1,clip=[-np.inf,np.inf]):
    # x1: [B,N,d], x2:[B,M,L*4]
    x2 = x2.reshape(*x2.shape[:-1],-1,4)
    N = x1.shape[1]
    M = x2.shape[1]
    x1_flag = torch.logical_not((x1==0).all(-1)).type(x1.dtype)
    x2_flag = torch.logical_not((x2==0).all(-1)).type(x2.dtype)
    dx = batch_proj_xysc(x1[:,:,None,[0,1,3,4]].repeat_interleave(M,2),x2[:,None].repeat_interleave(N,1))
    dx_xy = (dx[...,:2]/scale).clip(min=clip[0],max=clip[1])
    dx = torch.cat([dx_xy,dx[...,2:]],-1)
    dx = (dx * x1_flag[:,:,None,None,None]*x2_flag[:,None,:,:,None])
    
    min_idx = dx[...,0].abs().argmin(3)
    min_pts = torch.gather(dx,3,min_idx[...,None,None].repeat(1,1,1,1,4)).squeeze(-2)
    edge = torch.cat([min_pts,dx[:,:,:,0],dx[:,:,:,-1]],-1)
    if padto is not None:
        edge = torch.cat((edge,torch.zeros(*edge.shape[:-1],padto-edge.shape[-1],device=edge.device)),-1)
    return edge

def agent2lane_edge_per_pts(x1,x2,padto=None,scale=1,clip=[-np.inf,np.inf]):
    # x1: [B,N,4], x2:[B,M,4]
    B,N = x1.shape[:2]
    M = x2.shape[1]
    rel_coord = rel_xysc(x1.repeat_interleave(M,1),x2.repeat_interleave(N,0).reshape(B,N*M,4)).reshape(B,N,M,4)
    
    x1_flag = torch.logical_not((x1==0).all(-1)).type(x1.dtype)
    x2_flag = torch.logical_not((x2==0).all(-1)).type(x2.dtype)
    rel_xy = (rel_coord[...,:2]/scale).clip(min=clip[0],max=clip[1])
    rel_coord = torch.cat([rel_xy,rel_coord[...,2:]],-1)
    rel_coord = rel_coord * x1_flag[:,:,None,None]*x2_flag[:,None,:,None]

    return rel_coord

def lane2lane_edge(x1,x2,padto=None,scale=1,clip=[-np.inf,np.inf]):
    # x1: [B,M,L*4], x2:[B,M,L*4]
    x1 = x1.reshape(*x1.shape[:-1],-1,4).unsqueeze(2).repeat_interleave(x2.shape[1],2)
    x2 = x2.reshape(*x2.shape[:-1],-1,4).unsqueeze(1).repeat_interleave(x1.shape[1],1)
    
    x1s = x1[...,0,:]
    x1e = x1[...,-1,:]
    x2s = x2[...,0,:]
    x2e = x2[...,-1,:]
    dx1 = rel_xysc(x1s,x2s)
    dx2 = rel_xysc(x1s,x2e)
    dx3 = rel_xysc(x1e,x2s)
    dx4 = rel_xysc(x1e,x2e)
    dx1 = torch.cat([(dx1[...,:2]/scale).clip(min=clip[0],max=clip[1]),dx1[...,2:]],-1)
    dx2 = torch.cat([(dx2[...,:2]/scale).clip(min=clip[0],max=clip[1]),dx2[...,2:]],-1)
    dx3 = torch.cat([(dx3[...,:2]/scale).clip(min=clip[0],max=clip[1]),dx3[...,2:]],-1)
    dx4 = torch.cat([(dx4[...,:2]/scale).clip(min=clip[0],max=clip[1]),dx4[...,2:]],-1)
    edge = torch.cat([dx1,dx2,dx3,dx4],-1)
    if padto is not None:
        edge = torch.cat((edge,torch.zeros(*edge.shape[:-1],padto-edge.shape[-1],device=edge.device)),-1)
    return edge

def edge_as_aux1(x1,x2):
    # when the edge encoding is passed in via x1
    # x1: [B,N1,N2*d], x2:[B,N2,_]
    B,N1 = x1.shape[:2]
    N2 = x2.shape[1]
    return x1.reshape(B,N1,N2,-1)

def edge_as_aux2(x1,x2):
    # when the edge encoding is passed in via x1
    # x1: [B,N1,_], x2:[B,N2,N1*d]
    B,N1 = x1.shape[:2]
    N2 = x2.shape[1]
    return x2.reshape(B,N2,N1,-1).transpose(1,2)

class Agent_emb(nn.Module):
    def __init__(self, raw_dim,n_embd):
        super(Agent_emb, self).__init__()
        self.raw_dim = raw_dim
        self.n_embd = n_embd
        self.FC = nn.Linear(raw_dim, n_embd)

    def forward(self, input):
        if input.size(-1)<self.raw_dim:
            torch.cat((input,torch.zeros(*input.shape[:-1],self.raw_dim-input.shape[-1],device=input.device)),-1)
        if self.FC.weight.dtype==torch.float16:
            input = input.half()
        return self.FC(input)
    
class Lane_emb(nn.Module):
    def __init__(self, raw_dim,n_embd,xy_scale=10,xy_clip = [-np.inf,np.inf]):
        super(Lane_emb, self).__init__()
        self.raw_dim = raw_dim
        self.n_embd = n_embd
        self.xy_scale = xy_scale
        self.xy_clip = xy_clip
        self.FC = nn.Linear(raw_dim, n_embd)

    def forward(self, lane):
        # lane: [B,M,L*4]
        lane = lane.reshape(*lane.shape[:-1],-1,4)
        B,M,L = lane.shape[:3]
        lane_rel = rel_xysc(lane[...,0:1,:].repeat_interleave(L,-2),lane).reshape(B,M,L,-1)
        lane_rel = torch.cat([(lane_rel[...,:2]/self.xy_scale).clip(min=self.xy_clip[0],max=self.xy_clip[1]),lane_rel[...,2:]],-1)
        lane_rel = lane_rel.reshape(B,M,-1)
        
        if lane_rel.size(-1)<self.raw_dim:
            torch.cat((lane_rel,torch.zeros(*lane_rel.shape[:-1],self.raw_dim-lane_rel.shape[-1],device=lane_rel.device)),-1)
        if self.FC.weight.dtype==torch.float16:
            lane_rel = lane_rel.half()
        return self.FC(lane_rel)

class Lane2Lane_emb(nn.Module):
    def __init__(self, edge_dim,n_embd,xy_scale=10,xy_clip = [-np.inf,np.inf]):
        super(Lane2Lane_emb, self).__init__()
        self.edge_dim = edge_dim
        self.n_embd = n_embd
        self.xy_scale = xy_scale
        self.xy_clip = xy_clip
        self.FC = nn.Linear(edge_dim, n_embd)

    def forward(self, lane1,lane2,aux_edge=None):
        # lane: [B,M,L*4]
        edge = lane2lane_edge(lane1,lane2,scale = self.xy_scale,clip = self.xy_clip)
        if aux_edge is not None:
            edge = torch.cat((edge,aux_edge),-1)
        if self.FC.weight.dtype==torch.float16:
            edge = edge.half()
        return self.FC(edge)

class Agent2Lane_emb_proj(nn.Module):
    def __init__(self, edge_dim,n_embd,xy_scale=10,xy_clip = [-np.inf,np.inf]):
        super(Agent2Lane_emb_proj, self).__init__()
        self.edge_dim = edge_dim
        self.n_embd = n_embd
        self.xy_scale = xy_scale
        self.xy_clip = xy_clip
        self.FC = nn.Linear(edge_dim, n_embd)

    def forward(self, agent,lane,aux_edge=None):
        # lane: [B,M,L*4]
        B,N,T = agent.shape[:3]
        M = lane.shape[1]
        edge = agent2lane_edge_proj(agent.view(B,N*T,-1),lane,scale = self.xy_scale,clip = self.xy_clip)
        
        if aux_edge is not None:
            edge = torch.cat((edge,aux_edge.view(B,N*T,M,-1)),-1)
        if self.FC.weight.dtype==torch.float16:
            edge = edge.half()
        return self.FC(edge).view(B,N,T,M,-1)
    
class Agent2Lane_emb_attn(nn.Module):
    def __init__(self, edge_dim,n_embd,agent_feat_dim,output_dim=None,aux_edge_dim=0,xy_scale=10,xy_clip = [-np.inf,np.inf],Lmax=30):
        super(Agent2Lane_emb_attn, self).__init__()
        self.edge_dim = edge_dim
        self.n_embd = n_embd
        self.xy_scale = xy_scale
        self.xy_clip = xy_clip
        self.edge_embed_net = nn.Linear(edge_dim, n_embd)
        self.agent_embed_net = nn.Linear(agent_feat_dim, n_embd)
        self.attn = CrossAttention(n_embd,n_head = 4)
        self.pe = nn.Parameter(torch.randn(Lmax,n_embd))
        self.proj = nn.Linear(n_embd+aux_edge_dim,output_dim) if output_dim is not None else nn.Identity()
        

    def forward(self, agent,lane,aux_edge=None):
        # agent: [B,N,d] (d starts with x,y,v,s,c, then the static features)
        # lane: [B,M,L*4]
        orig_shape = agent.shape
        if agent.ndim==4:
            agent = TensorUtils.join_dimensions(agent,1,3)
            
            
        agent_xysc = agent[...,[0,1,3,4]]
        agent_feat = torch.cat([agent[...,[2]],agent[...,5:]],-1)
        B,N = agent.shape[:2]

        M = lane.shape[1]
        L = int(lane.size(2)/4)
        edge = agent2lane_edge_per_pts(agent_xysc.view(B,N,-1),lane.reshape([B,M*L,4]),scale = self.xy_scale,clip = self.xy_clip).reshape(B,N,M,L,-1)
        edge_emb = self.edge_embed_net(edge)
        edge_emb = edge_emb + self.pe[:L][None,None,None]
        agent_emb = self.agent_embed_net(agent_feat)
        edge_emb = self.attn(agent_emb.repeat_interleave(M,1).reshape(B*N*M,-1,self.n_embd),edge_emb.reshape(B*N*M,L,self.n_embd)).reshape(*orig_shape[:-1],M,-1)
        if aux_edge is not None:
            edge_emb = torch.cat((edge_emb,aux_edge),-1)
        return self.proj(edge_emb)

class Agent2Agent_emb(nn.Module):
    def __init__(self, edge_dim,n_embd,xy_scale=10,xy_clip = [-np.inf,np.inf]):
        super(Agent2Agent_emb, self).__init__()
        self.edge_dim = edge_dim
        self.n_embd = n_embd
        self.xy_scale = xy_scale
        self.xy_clip = xy_clip
        self.FC = nn.Linear(edge_dim, n_embd)

    def forward(self, agent1,agent2,aux_edge=None):
        edge = agent2agent_edge(agent1,agent2,scale = self.xy_scale,clip = self.xy_clip)
        if aux_edge is not None:
            edge = torch.cat((edge,aux_edge),-1)
        if self.FC.weight.dtype==torch.float16:
            edge = edge.half()
        return self.FC(edge)


def test_RBF():
    func = Gaussian_RBF_conv(1.6,6,"cuda")
    x = torch.zeros(1,15,15)
    x[:,8,8]=1
    y = func(x)
    img = y[0].detach().cpu().numpy()
    
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

def test_edge():
    import pickle
    with open("sf_test.pkl","rb") as f:
        data = pickle.load(f)
    lane_xyh = data["lane_xyh"]    
    lane_feat = torch.cat([lane_xyh[...,:2],torch.cos(lane_xyh[...,2:3]),torch.sin(lane_xyh[...,2:3])],-1)
    lane_feat = lane_feat.reshape(*lane_feat.shape[:-2],-1)
    ego_xycs = data["agent_hist"][:,:,-1,[0,1,6,7]]
    ego_v = data["agent_hist"][:,:,-1,2:4].norm(dim=-1,keepdim=True)
    ego_wl = data["extent"][...,:2]
    
    ego_xyvcswl = torch.cat([ego_xycs[...,:2],ego_v,ego_xycs[...,-2:],ego_wl],-1)
    ego_xyvcswl = ego_xyvcswl.repeat_interleave(2,1)
    B,M = lane_feat.shape[:2]
    edge_obj = Agent2Lane_emb_attn(edge_dim=4,n_embd=64,agent_feat_dim=3).cuda()
    f2 = agent2agent_edge(ego_xyvcswl,ego_xyvcswl)
    f1 = agent2lane_edge_proj(ego_xyvcswl,lane_feat)
    
    f3 = lane2lane_edge(lane_feat,lane_feat)
    f4 = edge_obj(ego_xyvcswl,lane_feat)
    
    print("123")
    
def test_emb():
    import pickle
    with open("sf_test.pkl","rb") as f:
        data = pickle.load(f)
    lane_xyh = data["lane_xyh"]    
    lane_feat = torch.cat([lane_xyh[...,:2],torch.sin(lane_xyh[...,2:3]),torch.cos(lane_xyh[...,2:3])],-1)
    lane_feat = lane_feat.reshape(*lane_feat.shape[:-2],-1)
    ego_xycs = data["agent_hist"][:,:,-1,[0,1,6,7]]
    ego_v = data["agent_hist"][:,:,-1,2:4].norm(dim=-1,keepdim=True)
    ego_wl = data["extent"][...,:2]
    ego_type = nn.functional.one_hot(data["type"],4).float()
    veh_feat = torch.cat([ego_v,ego_wl,ego_type],-1)
    n_emb = 256
    veh_emb = Agent_emb(7,n_emb).to(veh_feat.device)
    lane_emb = Lane_emb(4*30,n_emb).to(veh_feat.device)
    
    xv = veh_emb(veh_feat)
    xl = lane_emb(lane_feat)
    

    
    print("123")
if __name__ == '__main__':
    test_edge()