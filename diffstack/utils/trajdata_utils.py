from logging import raiseExceptions
from signal import raise_signal
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict

from trajdata.data_structures.batch import AgentBatch, SceneBatch, StateTensor, PadDirection

import diffstack.utils.tensor_utils as TensorUtils
import diffstack.utils.geometry_utils as GeoUtils
from diffstack.utils.geometry_utils import transform_points_tensor
from diffstack.configs.base import ExperimentConfig
from trajdata import AgentType


def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        # state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
        yaw = state[...,6:7]
    else:
        assert state.shape[-1] == 8
        yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    pos = state[..., :2]
    
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents_scene(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_hist_pos.shape
    _, _, _, h, w = maps.shape
    maps = maps.clone()
    agent_hist_pos = TensorUtils.unsqueeze_expand_at(agent_hist_pos,a,1)
    agent_mask_tiled = TensorUtils.unsqueeze_expand_at(agent_mask,a,1)*TensorUtils.unsqueeze_expand_at(agent_mask,a,2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos.reshape(b*a,-1,2), raster_from_agent.reshape(b*a,3,3)).reshape(b,a,a,t,2)
    raster_hist_pos = raster_hist_pos * agent_mask_tiled.unsqueeze(-1)  # Set invalid positions to 0.0 Will correct below
    
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels [B, A, A, T, 2]
    raster_hist_pos = raster_hist_pos.transpose(2,3)
    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, A, T, A]
    hist_image = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    ego_mask = torch.zeros_like(raster_hist_pos_flat,dtype=torch.bool)
    ego_mask[:,range(a),:,range(a)]=1
    agent_mask = torch.logical_not(ego_mask)


    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*agent_mask, src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*ego_mask, src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[..., 0] = 0  # correct the 0th index from invalid positions
    hist_image[..., -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, a, t, h, w)

    maps = torch.cat((hist_image, maps), dim=2)  # treat time as extra channels
    return maps


def rasterize_agents_sc(
        maps: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_yaw: torch.Tensor,
        agent_speed: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_pos.shape
    _, _, _, h, w = maps.shape
    
    # take the first agent as the center agent
    raster_pos = transform_points_tensor(agent_pos.reshape(b,-1,2), raster_from_agent[:,0]).reshape(b,a,t,2)
    raster_pos = raster_pos * agent_mask.unsqueeze(-1)  # Set invalid positions to 0.0 Will correct below

    raster_pos[..., 0].clip_(0, (w - 1))
    raster_pos[..., 1].clip_(0, (h - 1))
    raster_pos_round = torch.round(raster_pos).long()  # round pixels [B, A, T, 2]
    raster_dxy = raster_pos - raster_pos_round.float()

    raster_pos_flat = raster_pos_round[..., 1] * w + raster_pos_round[..., 0]  # [B, A, T]
    prob = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    # dx = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    # dy = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    # heading = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    # vel = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    raster_pos_flat = raster_pos_flat.unsqueeze(-1)
    prob.scatter_(dim=3, index=raster_pos_flat, src=torch.ones_like(prob))
    # dx.scatter_(dim=3, index=raster_pos_flat, src=raster_dxy[...,0:1].clone().repeat_interleave(h*w,-1)) 
    # dy.scatter_(dim=3, index=raster_pos_flat, src=raster_dxy[...,1:2].clone().repeat_interleave(h*w,-1))
    # heading.scatter_(dim=3, index=raster_pos_flat, src=agent_yaw.repeat_interleave(h*w,-1))
    # vel.scatter_(dim=3, index=raster_pos_flat, src=agent_speed.unsqueeze(-1).repeat_interleave(h*w,-1))
    
    # feature = torch.stack((prob, dx, dy, heading), dim=3) # [B, A, T, 5, H * W]
    feature = prob
    feature[..., 0] = 0  # correct the 0th index from invalid positions
    feature[..., -1] = 0  # correct the maximum index caused by out of bound locations

    feature = feature.reshape(b, a, t, -1, h, w)

    return feature



def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        filter=None,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)
    if filter=="0.5-1-0.5":
        kernel = torch.tensor([[0.5, 0.5, 0.5],
                    [0.5, 1., 0.5],
                    [0.5, 0.5, 0.5]]).to(hist_image.device)

        kernel = kernel.view(1, 1, 3, 3).repeat(t, t, 1, 1)
        hist_image = F.conv2d(hist_image, kernel,padding=1)
    if cat:
        maps = maps.clone()
        maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
        return maps
    else:
        return hist_image

def rasterize_agents_rec(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        ego_neg = False,
        parallel_raster=True,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    with torch.no_grad():
        b, a, t, _ = agent_hist_pos.shape
        _, _, h, w = maps.shape
        
        coord_tensor = torch.cat((torch.arange(w).view(w,1,1).repeat_interleave(h,1),
                                torch.arange(h).view(1,h,1).repeat_interleave(w,0),),-1).to(maps.device)

        agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
        raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
        

        raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below

        raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
        
        raster_hist_pos_yx = torch.cat((raster_hist_pos[...,1:],raster_hist_pos[...,0:1]),-1)

        if parallel_raster:
        # vectorized version, uses much more memory
            coord_tensor_tiled = coord_tensor.view(1,1,1,h,w,-1).repeat(b,t,a,1,1,1)
            dyx = raster_hist_pos_yx[...,None,None,:]-coord_tensor_tiled
            cos_yaw = torch.cos(-agent_hist_yaw)
            sin_yaw = torch.sin(-agent_hist_yaw)

            rotM = torch.stack(
                [
                    torch.stack([cos_yaw, sin_yaw], dim=-1),
                    torch.stack([-sin_yaw, cos_yaw], dim=-1),
                ],dim=-2,
            )
            rotM = rotM.transpose(1,2)
            rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            extent_tiled = agent_extent_yx[:,None,:,None,None]
            
            flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)

            agent_mask_tiled = agent_mask.transpose(1,2).type(torch.int)
            if ego_neg:
                # flip the value for ego
                agent_mask_tiled[:,:,0] = -agent_mask_tiled[:,:,0]
            hist_img = flag*agent_mask_tiled.view(b,t,a,1,1)
            
            if hist_img.shape[2]>1:
            # aggregate along the agent dimension
                hist_img = hist_img[:,:,0] + hist_img[:,:,1:].max(2)[0]*(hist_img[:,:,0]==0)
            else:
                hist_img = hist_img.squeeze(2)
        else:

        # loop through all agents, slow but memory efficient
            coord_tensor_tiled = coord_tensor.view(1,1,h,w,-1).repeat(b,t,1,1,1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            hist_img_ego = torch.zeros([b,t,h,w],device=maps.device)
            hist_img_nb = torch.zeros([b,t,h,w],device=maps.device)
            for i in range(raster_hist_pos_yx.shape[-2]):
                dyx = raster_hist_pos_yx[...,i,None,None,:]-coord_tensor_tiled
                yaw_i = agent_hist_yaw[:,i]
                cos_yaw = torch.cos(-yaw_i)
                sin_yaw = torch.sin(-yaw_i)

                rotM = torch.stack(
                    [
                        torch.stack([cos_yaw, sin_yaw], dim=-1),
                        torch.stack([-sin_yaw, cos_yaw], dim=-1),
                    ],dim=-2,
                )
                
                rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
                extent_tiled = agent_extent_yx[:,None,i,None,None]
                
                flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)
                if i==0:
                    if ego_neg:
                        hist_img_ego = -flag*agent_mask[:,0,:,None,None]
                    else:
                        hist_img_ego = flag*agent_mask[:,0,:,None,None]
                else:
                    hist_img_nb = torch.maximum(hist_img_nb,agent_mask[:,0,:,None,None]*flag)
                
            if a>1:
                hist_img = hist_img_ego + hist_img_nb*(hist_img_ego==0)
            else:
                hist_img = hist_img_ego

        if cat:
            maps = maps.clone()
            maps = torch.cat((hist_img, maps), dim=1)  # treat time as extra channels
            return maps
        else:
            return hist_img



def get_drivable_region_map(maps):
    if isinstance(maps, torch.Tensor):
        if maps.shape[-3]>=7:
            drivable = torch.amax(maps[..., -7:-4, :, :], dim=-3).bool()
        else:
            drivable = torch.amax(maps[..., -3:, :, :], dim=-3).bool()
    else:
        if maps.shape[-3]>=7:
            drivable = np.amax(maps[..., -7:-4, :, :], axis=-3).astype(bool)
        else:
            drivable = np.amax(maps[..., -3:, :, :], dim=-3).astype(bool)
    return drivable


def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    empty_neighbor = a == 0
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1],device=batch["agent_hist"].device) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1],device=batch["agent_hist"].device) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1,device=batch["agent_hist"].device)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1],device=batch["agent_hist"].device)
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1],device=batch["agent_hist"].device)
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1],device=batch["agent_hist"].device) * torch.nan
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1],device=batch["agent_hist"].device)
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)

def parse_scene_centric(batch: dict, rasterize_mode:str):
    fut_pos, fut_yaw, fut_speed, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])

    curr_pos = hist_pos[:,:,-1]
    curr_yaw = hist_yaw[:,:,-1]
    if batch["centered_agent_state"].shape[-1]==7:
        world_yaw = batch["centered_agent_state"][...,6]
    else:
        assert batch["centered_agent_state"].shape[-1]==8
        world_yaw = torch.atan2(batch["centered_agent_state"][...,6],batch["centered_agent_state"][...,7])
    curr_speed = hist_speed[..., -1]
    centered_state = batch["centered_agent_state"]
    centered_yaw = centered_state[:, -1]
    centered_pos = centered_state[:, :2]
    old_type = batch["agent_type"]
    agent_type = torch.zeros_like(old_type)
    agent_type[old_type < 0] = 0
    agent_type[old_type ==[3,4]] = 2
    agent_type[old_type ==1] = 3
    agent_type[old_type ==2] = 2
    agent_type[old_type ==5] = 4

    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    if not batch["centered_agent_from_world_tf"].isnan().any():
        centered_world_from_agent = torch.inverse(batch["centered_agent_from_world_tf"])
    else:
        centered_world_from_agent = None
    b,a = curr_yaw.shape[:2]
    agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
    center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
    # map-related
    if batch["maps"] is not None:
        map_res = batch["maps_resolution"][0,0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        
        centered_raster_from_agent = torch.Tensor([
            [map_res, 0, 0.125 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).type_as(agents_from_center)
        
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        
        raster_from_center = centered_raster_from_agent @ agents_from_center
        center_from_raster = center_from_agents @ centered_agent_from_raster

        raster_from_world = batch["rasters_from_world_tf"]
        world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
        raster_from_world[torch.isnan(raster_from_world)] = 0.
        world_from_raster[torch.isnan(world_from_raster)] = 0.
        
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
            maps = rasterize_agents_scene(
                batch["maps"],
                hist_pos,
                hist_yaw,
                None,
                hist_mask,
                raster_from_center,
                map_res
            )
        elif rasterize_mode=="square":
            #TODO: add the square rasterization function for scene-centric data
            raise NotImplementedError
        elif rasterize_mode=="point_sc":
            hist_hm = rasterize_agents_sc(
                batch["maps"],
                hist_pos,
                hist_yaw,
                hist_speed,
                hist_mask,
                raster_from_center,
            )
            fut_hm = rasterize_agents_sc(
                batch["maps"],
                fut_pos,
                fut_yaw,
                fut_speed,
                fut_mask,
                raster_from_center,
            )
        maps = batch["maps"]
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_center = None
        center_from_raster = None
        raster_from_world = None
        centered_agent_from_raster = None
        centered_raster_from_agent = None

    extent_scale = 1.0


    d = dict(
        image=maps,
        drivable_map=drivable_map,
        fut_pos=fut_pos,
        fut_yaw=fut_yaw,
        fut_mask=fut_mask,
        hist_pos=hist_pos,
        hist_yaw=hist_yaw,
        hist_mask=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        world_yaw=world_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=centered_raster_from_agent,
        agent_from_raster=centered_agent_from_raster,
        raster_from_center=raster_from_center,
        center_from_raster=center_from_raster,
        agents_from_center = agents_from_center,
        center_from_agents = center_from_agents,
        raster_from_world=raster_from_world,
        agent_from_world=batch["centered_agent_from_world_tf"],
        world_from_agent=centered_world_from_agent,
    )
    if rasterize_mode=="point_sc":
        d["hist_hm"] = hist_hm
        d["fut_hm"] = fut_hm

    return d 

def parse_node_centric(batch: dict,rasterize_mode:str,):
    maybe_pad_neighbor(batch)
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]
    curr_yaw = curr_state[:, -1]
    curr_pos = curr_state[:, :2]

    # convert nuscenes types to l5kit types
    # old_type = batch["agent_type"]
    # agent_type = torch.zeros_like(old_type)
    # agent_type[old_type < 0] = 0
    # agent_type[old_type ==[3,4]] = 2
    # agent_type[old_type ==1] = 3
    # agent_type[old_type ==2] = 2
    # agent_type[old_type ==5] = 4
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"])
    neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"])
    if neigh_hist_speed.nelement() > 0:
        neigh_curr_speed = neigh_hist_speed[..., -1]
    else:
        neigh_curr_speed = neigh_hist_speed.unsqueeze(-1)
    # old_neigh_types = batch["neigh_types"]
    # # convert nuscenes types to l5kit types
    # neigh_types = torch.zeros_like(old_neigh_types)
    # # neigh_types = torch.zeros_like(old_type)
    # neigh_types[old_neigh_types < 0] = 0
    # neigh_types[old_neigh_types ==[3,4]] = 2
    # neigh_types[old_neigh_types ==1] = 3
    # neigh_types[old_neigh_types ==2] = 2
    # neigh_types[old_neigh_types ==5] = 4

    # mask out invalid extents
    neigh_hist_extents = batch["neigh_hist_extents"]
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch["agents_from_world_tf"])
    if batch["curr_agent_state"].shape[-1]==7:
        world_yaw = batch["curr_agent_state"][...,6]
    else:
        assert batch["curr_agent_state"].shape[-1]==8
        world_yaw = torch.atan2(batch["curr_agent_state"][...,6],batch["curr_agent_state"][...,7])

    # map-related
    if batch["maps"] is not None:
        map_res = batch["maps_resolution"][0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        raster_from_agent = torch.tensor([
            [map_res, 0, 0.125 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ],device=curr_pos.device)
        agent_from_raster = torch.inverse(raster_from_agent)
        raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=batch["maps"].shape[0], dim=0)
        agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=batch["maps"].shape[0], dim=0)
        raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])
        if neigh_hist_pos.nelement()>0:
            all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos), dim=1)
            all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw), dim=1)

            all_extents = torch.cat((batch["agent_hist_extent"].unsqueeze(1),batch["neigh_hist_extents"]),1).max(dim=2)[0][...,:2]
            all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask), dim=1)
        else:
            all_hist_pos = hist_pos[:, None]
            all_hist_yaw = hist_yaw[:, None]
            all_extents = batch["agent_hist_extent"].unsqueeze(1)[...,:2]
            all_hist_mask = hist_mask[:, None]
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
                maps = rasterize_agents(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        elif rasterize_mode=="square":
            maps = rasterize_agents_rec(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        else:
            raise Exception("unknown rasterization mode")
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0
    d = dict(
        image=maps,
        drivable_map=drivable_map,
        fut_pos=fut_pos,
        fut_yaw=fut_yaw,
        fut_mask=fut_mask,
        hist_pos=hist_pos,
        hist_yaw=hist_yaw,
        hist_mask=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        world_yaw=world_yaw,
        type=batch["agent_type"],
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=raster_from_agent,
        agent_from_raster=agent_from_raster,
        raster_from_world=raster_from_world,
        agent_from_world=batch["agents_from_world_tf"],
        world_from_agent=world_from_agents,
        neigh_hist_pos=neigh_hist_pos,
        neigh_hist_yaw=neigh_hist_yaw,
        neigh_hist_mask=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
        neigh_curr_speed=neigh_curr_speed,
        neigh_fut_pos=neigh_fut_pos,
        neigh_fut_yaw=neigh_fut_yaw,
        neigh_fut_mask=neigh_fut_mask,
        neigh_types=batch["neigh_types"],
        neigh_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale if neigh_hist_extents.nelement()>0 else None,
        
    )
    # if "agent_lanes" in batch:
    #     d["ego_lanes"] = batch["agent_lanes"]

    return d

@torch.no_grad()
def parse_trajdata_batch(batch, rasterize_mode="point"):
    if isinstance(batch,AgentBatch):
        # Be careful here, without making a copy of vars(batch) we would modify the fields of AgentBatch.
        batch = dict(vars(batch))
        d = parse_node_centric(batch,rasterize_mode)
    elif isinstance(batch,SceneBatch):
        batch = dict(vars(batch))
        d = parse_scene_centric(batch,rasterize_mode)
    elif isinstance(batch,dict):
        batch = dict(batch)
        if "num_agents" in batch:
            # scene centric
            d = parse_scene_centric(batch,rasterize_mode)
            
        else:
            # agent centric
            d = parse_node_centric(batch,rasterize_mode)

    batch.update(d)
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            batch[k]=v.nan_to_num(0)
    batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    return batch


def get_modality_shapes(cfg: ExperimentConfig, rasterize_mode: str = "point"):
    h = cfg.env.rasterizer.raster_size
    if rasterize_mode=="none":
        return dict(static=(3,h,h),dynamic=(0,h,h),image=(3,h,h))
    else:
        num_channels = (cfg.history_num_frames + 1) + 3
        return dict(static=(3,h,h),dynamic=(cfg.history_num_frames + 1,h,h),image=(num_channels, h, h))
        

def gen_ego_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types):
    """generate edges between ego trajectory samples and agent trajectories

    Args:
        ego_trajectories (torch.Tensor): [B,N,T,3]
        agent_trajectories (torch.Tensor): [B,A,T,3] or [B,N,A,T,3]
        ego_extents (torch.Tensor): [B,2]
        agent_extents (torch.Tensor): [B,A,2]
        raw_types (torch.Tensor): [B,A]
    Returns:
        edges (torch.Tensor): [B,N,A,T,10]
        type_mask (dict)
    """
    B,N,T = ego_trajectories.shape[:3]
    A = agent_trajectories.shape[-3]

    veh_mask = raw_types == int(AgentType["VEHICLE"])
    ped_mask = raw_types == int(AgentType["PEDESTRIAN"])

    edges = torch.zeros([B,N,A,T,10],device=ego_trajectories.device)
    edges[...,:3] = ego_trajectories.unsqueeze(2).repeat(1,1,A,1,1)
    if agent_trajectories.ndim==4:
        edges[...,3:6] = agent_trajectories.unsqueeze(1).repeat(1,N,1,1,1)
    else:
        edges[...,3:6] = agent_trajectories
    edges[...,6:8] = ego_extents.reshape(B,1,1,1,2).repeat(1,N,A,T,1)
    edges[...,8:] = agent_extents.reshape(B,1,A,1,2).repeat(1,N,1,T,1)
    type_mask = {"VV":veh_mask,"VP":ped_mask}
    return edges,type_mask


        
    print("abc")

def gen_EC_edges(ego_trajectories,agent_trajectories,ego_extents, agent_extents, raw_types,mask=None):
    """generate edges between ego trajectory samples and agent trajectories

    Args:
        ego_trajectories (torch.Tensor): [B,A,T,3]
        agent_trajectories (torch.Tensor): [B,A,T,3]
        ego_extents (torch.Tensor): [B,2]
        agent_extents (torch.Tensor): [B,A,2]
        raw_types (torch.Tensor): [B,A]
        mask (optional, torch.Tensor): [B,A]
    Returns:
        edges (torch.Tensor): [B,N,A,T,10]
        type_mask (dict)
    """

    B,A = ego_trajectories.shape[:2]
    T = ego_trajectories.shape[-2]

    veh_mask = raw_types == int(AgentType["VEHICLE"])
    ped_mask = raw_types == int(AgentType["PEDESTRIAN"])

    
    if ego_trajectories.ndim==4:
        edges = torch.zeros([B,A,T,10],device=ego_trajectories.device)
        edges[...,:3] = ego_trajectories
        edges[...,3:6] = agent_trajectories
        edges[...,6:8] = ego_extents.reshape(B,1,1,2).repeat(1,A,T,1)
        edges[...,8:] = agent_extents.unsqueeze(2).repeat(1,1,T,1)
    elif ego_trajectories.ndim==5:
        
        K = ego_trajectories.shape[2]
        edges = torch.zeros([B,A*K,T,10],device=ego_trajectories.device)
        edges[...,:3] = TensorUtils.join_dimensions(ego_trajectories,1,3)
        edges[...,3:6] = agent_trajectories.repeat(1,K,1,1)
        edges[...,6:8] = ego_extents.reshape(B,1,1,2).repeat(1,A*K,T,1)
        edges[...,8:] = agent_extents.unsqueeze(2).repeat(1,K,T,1)
        veh_mask = veh_mask.tile(1,K)
        ped_mask = ped_mask.tile(1,K)
    if mask is not None:
        veh_mask = veh_mask*mask
        ped_mask = ped_mask*mask
    type_mask = {"VV":veh_mask,"VP":ped_mask}
    return edges,type_mask


def generate_edges(
        raw_type,
        extents,
        pos_pred,
        yaw_pred,
        batch_first = False,
):
    veh_mask = raw_type == int(AgentType["VEHICLE"])
    ped_mask = raw_type == int(AgentType["PEDESTRIAN"])

    agent_mask = veh_mask | ped_mask
    edge_types = ["VV", "VP", "PV", "PP"]
    edges = {et: list() for et in edge_types}
    for i in range(agent_mask.shape[0]):
        agent_idx = torch.where(agent_mask[i] != 0)[0]
        edge_idx = torch.combinations(agent_idx, r=2)
        VV_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        VP_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        PV_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        PP_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        if pos_pred.ndim == 4:
            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, edge_idx[:, 0], :],
                    yaw_pred[i, edge_idx[:, 0], :],
                    pos_pred[i, edge_idx[:, 1], :],
                    yaw_pred[i, edge_idx[:, 1], :],
                    extents[i, edge_idx[:, 0]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                    extents[i, edge_idx[:, 1]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[VV_idx])
            edges["VP"].append(edges_of_all_types[VP_idx])
            edges["PV"].append(edges_of_all_types[PV_idx])
            edges["PP"].append(edges_of_all_types[PP_idx])
        elif pos_pred.ndim == 5:

            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, :, edge_idx[:, 0], :],
                    yaw_pred[i, :, edge_idx[:, 0], :],
                    pos_pred[i, :, edge_idx[:, 1], :],
                    yaw_pred[i, :, edge_idx[:, 1], :],
                    extents[i, None, edge_idx[:, 0], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                    extents[i, None, edge_idx[:, 1], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[:, VV_idx])
            edges["VP"].append(edges_of_all_types[:, VP_idx])
            edges["PV"].append(edges_of_all_types[:, PV_idx])
            edges["PP"].append(edges_of_all_types[:, PP_idx])
    if batch_first:
        for et, v in edges.items():
            edges[et] = pad_sequence(v, batch_first=True,padding_value=torch.nan)
    else:
        if pos_pred.ndim == 4:
            for et, v in edges.items():
                edges[et] = torch.cat(v, dim=0)
        elif pos_pred.ndim == 5:
            for et, v in edges.items():
                edges[et] = torch.cat(v, dim=1)
    return edges


def merge_scene_batches(scene_batches: List[SceneBatch], dt: float) -> SceneBatch:
    assert scene_batches[0].history_pad_dir == PadDirection.BEFORE
    assert all([b.agent_hist.shape[0] == 1 for b in scene_batches]), "only batch_size=1 is supported"
    
    # Convert everything to world coordinates
    scene_batches = [b.apply_transform(b.centered_world_from_agent_tf) for b in scene_batches]
    state_format = scene_batches[0].agent_hist._format

    # Not all agent might be present at all time steps, so we match them by name.
    # Get unique names, use np.unique return_index and sort to preserve ordering.
    agent_names = [np.array(b.agent_names[0]) for b in scene_batches]
    all_agent_names = np.concatenate(agent_names)
    _, idx = np.unique(all_agent_names, return_index=True)
    unique_agent_names = all_agent_names[np.sort(idx)]

    num_agents = len(unique_agent_names)
    hist_len = len(scene_batches)
    fut_len = scene_batches[-1].agent_fut.shape[-2]

    # Create full history with nans, then replace them for each time step
    agent_hist = torch.full((1, num_agents, hist_len, scene_batches[0].agent_hist.shape[-1]), dtype=scene_batches[0].agent_hist.dtype, fill_value=torch.nan)
    agent_hist_extent = torch.full((1, num_agents, hist_len, 2), dtype=scene_batches[0].agent_hist_extent.dtype, fill_value=torch.nan)
    agent_type = torch.full((1, num_agents), dtype=scene_batches[0].agent_type.dtype, fill_value=-1)

    for t, scene_batch in enumerate(scene_batches):
        match_inds = np.argwhere(np.array(scene_batch.agent_names[0])[:, None] == unique_agent_names[None, :])  # n_current, n_all -> n_current, 2
        assert match_inds.shape[0] == len(scene_batch.agent_names[0]), "there should be only 1 unique match"
        agent_hist[0, match_inds[:, 1], t, :] = scene_batch.agent_hist[0, match_inds[:, 0], -1, :]
        agent_hist_extent[0, match_inds[:, 1], t, :] = scene_batch.agent_hist_extent[0, match_inds[:, 0], -1, :]
        agent_type[0, match_inds[:, 1]] = scene_batch.agent_type[0, match_inds[:, 0]]

    # Dummy future, repeat last state
    agent_fut = agent_hist[:, :, -1:, :].repeat_interleave(fut_len, dim=-2)
    agent_fut_extent = agent_hist_extent[:, :, -1:, :].repeat_interleave(fut_len, dim=-2)

    # Create trajdata batch
    merged_batch = SceneBatch(
        data_idx=torch.tensor([torch.nan]),
        scene_ts=scene_batches[0].scene_ts,
        scene_ids=scene_batches[0].scene_ids,
        dt=torch.tensor([dt]),
        num_agents=torch.tensor([num_agents]),
        agent_type=agent_type,
        centered_agent_state=scene_batches[0].centered_agent_state,
        agent_names=[list(unique_agent_names)],
        agent_hist=StateTensor.from_array(agent_hist, state_format),
        agent_hist_extent=agent_hist_extent,
        agent_hist_len=torch.tensor([[hist_len] * num_agents]),  # len includes current state
        agent_fut=StateTensor.from_array(agent_fut, state_format),
        agent_fut_extent=agent_fut_extent,
        agent_fut_len=torch.from_numpy(np.array([[fut_len] * num_agents])),
        robot_fut=None,
        robot_fut_len=None,
        map_names=scene_batches[0].map_names,
        maps=scene_batches[0].map_names,
        maps_resolution=scene_batches[0].maps_resolution,
        vector_maps=scene_batches[0].vector_maps,
        rasters_from_world_tf=scene_batches[0].rasters_from_world_tf,
        centered_agent_from_world_tf=scene_batches[0].centered_agent_from_world_tf,
        centered_world_from_agent_tf=scene_batches[0].centered_world_from_agent_tf,
        history_pad_dir=PadDirection.BEFORE,
        extras=dict(scene_batches[0].extras),
    )  

    return merged_batch
