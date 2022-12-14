import torch
import numpy as np
import os
import dill
import json

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from typing import Dict, Iterable, Union
from time import time

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trajdata import AgentBatch, UnifiedDataset, AgentType
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.agent import AgentType
from trajdata.augmentation import NoiseHistories

from diffstack.data import agent_batch_extras, scene_batch_extras
from diffstack.utils.utils import all_gather


def prepare_avdata(rank, hyperparams, scene_centric=True, use_cache=False):

    # Load cached data or process and cache data if cache file does not exists.
    cache_params = ".".join([str(hyperparams[k]) for k in 
                            ["prediction_sec", "history_sec"]]) + ".v6"
    cached_train_data_path = os.path.join(hyperparams['trajdata_cache_dir'], f"{hyperparams['train_data']}.{cache_params}.cached.trajdata.pkl")
    cached_eval_data_path = os.path.join(hyperparams['trajdata_cache_dir'], f"{hyperparams['eval_data']}.{cache_params}.cached.trajdata.pkl")

    # Load training and evaluation environments and scenes
    attention_radius = defaultdict(lambda: 20.0) # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    data_dirs: Dict[str, str] = json.loads(hyperparams['data_loc_dict'])

    augmentations = list()
    if hyperparams['augment_input_noise'] > 0.0:
        augmentations.append(NoiseHistories(stddev=hyperparams['augment_input_noise']))

    map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

    if hyperparams["plan_agent_choice"] == "most_relevant":
        pass
    elif hyperparams["plan_agent_choice"] == "ego":
        raise NotImplementedError
    else:
        raise ValueError("Unknown plan_agent_choice: {}".format(hyperparams["plan_agent_choice"]))

    # # Debug with NuScenes map API 
    # nusc_maps = {data_name: 
    #     {map_name: NuScenesMap(os.path.expanduser(data_path), map_name) for map_name in nusc_map_names} 
    #     for data_name, data_path in data_dirs.items()}
    # transforms = (
    #     lambda el: remove_parked(el),
    #     lambda el: robot_selector(el), 
    #     lambda el: make_robot_the_first(el), 
    #     lambda el: augment_with_goal(el),
    #     # lambda el: augment_with_lanes_nusc(el, nusc_maps), 
    #     # lambda el: augment_with_lanes_compare(el, nusc_maps), 
    #     lambda el: augment_with_lanes(el), 
    #     # lambda el: augment_with_goal_lanes_nusc(el, nusc_maps), 
    #     lambda el: augment_with_goal_lanes(el), 
    #     )

    if scene_centric:
        # Use actual ego as our planning agent, pick the closest other agent to predict 
        # for agent-centric prediction models (like T++). 
        # We only consider vehicles for prediction agent for now.
        # Scene-centric prediction models can do prediction for all agents.
        pre_filter_transforms = (
            lambda el: scene_batch_extras.remove_parked(el),
            lambda el: scene_batch_extras.role_selector(el, pred_agent_types=[AgentType.VEHICLE]), 
        )
        transforms = pre_filter_transforms + (
            lambda el: scene_batch_extras.make_robot_the_first(el), 
            lambda el: scene_batch_extras.augment_with_point_goal(el),
            lambda el: scene_batch_extras.augment_with_lanes(el, make_missing_lane_invalid=True), 
            lambda el: scene_batch_extras.augment_with_goal_lanes(el), 
            )
        get_filter_func = scene_batch_extras.get_filter_func
    else:
        pre_filter_transforms = (
            lambda el: agent_batch_extras.remove_parked(el),
            lambda el: agent_batch_extras.robot_selector(el), 
        )
        transforms = pre_filter_transforms + (
            lambda el: agent_batch_extras.make_robot_the_first(el), 
            lambda el: agent_batch_extras.augment_with_point_goal(el),
            lambda el: agent_batch_extras.augment_with_lanes(el, make_missing_lane_invalid=True), 
            lambda el: agent_batch_extras.augment_with_goal_lanes(el), 
            )
        get_filter_func = agent_batch_extras.get_filter_func            

    eval_dataset = UnifiedDataset(desired_data=[hyperparams['eval_data']],
                                   desired_dt=hyperparams['dt'],
                                   centric="scene" if scene_centric else "agent",
                                   history_sec=(hyperparams['history_sec'], hyperparams['history_sec']),
                                   future_sec=(hyperparams['prediction_sec'], hyperparams['prediction_sec']),
                                   agent_interaction_distances=attention_radius,
                                   incl_robot_future=hyperparams['incl_robot_node'],
                                   incl_raster_map=hyperparams['map_encoding'],
                                   incl_vector_map=True,
                                   raster_map_params=map_params,
                                   only_predict=[node_type for node_type in AgentType if node_type.name in hyperparams['pred_state']],
                                   no_types=[AgentType.UNKNOWN, AgentType.BICYCLE, AgentType.MOTORCYCLE],
                                   num_workers=hyperparams['preprocess_workers'],
                                   cache_location=hyperparams['trajdata_cache_dir'],
                                   data_dirs=data_dirs,
                                   # extras=OrderedDict(robot_ind=robot_selector),
                                   transforms=pre_filter_transforms,                                
                                   verbose=True,
                                   rank=rank,
                                   rebuild_cache=hyperparams['rebuild_cache'])

    train_dataset = UnifiedDataset(desired_data=[hyperparams['train_data']],
                                   desired_dt=hyperparams['dt'], 
                                   centric="scene" if scene_centric else "agent",
                                   history_sec=(0.1, hyperparams['history_sec']),
                                   # future_sec=(0.1, hyperparams['prediction_sec']),  # TODO support planning with partial predictions
                                   future_sec=(hyperparams['prediction_sec'], hyperparams['prediction_sec']),
                                   agent_interaction_distances=attention_radius,
                                   incl_robot_future=hyperparams['incl_robot_node'],
                                   incl_raster_map=hyperparams['map_encoding'],
                                   incl_vector_map=True,
                                   raster_map_params=map_params,
                                   only_predict=[node_type for node_type in AgentType if node_type.name in hyperparams['pred_state']],
                                   no_types=[AgentType.UNKNOWN, AgentType.BICYCLE, AgentType.MOTORCYCLE],
                                   augmentations=augmentations if len(augmentations) > 0 else None,
                                   num_workers=hyperparams['preprocess_workers'],
                                   cache_location=hyperparams['trajdata_cache_dir'],
                                   data_dirs=data_dirs,
                                   # extras=OrderedDict(robot_ind=robot_selector),
                                   transforms=pre_filter_transforms,                               
                                   verbose=True,
                                   rank=rank,
                                   rebuild_cache=hyperparams['rebuild_cache'])


    # Filter / cache dataset.
    # TODO(pkarkus) Filtering is quite slow currently and inefficient for multi-GPU setup 
    #   because each process has to scan through the entire dataset.
    filter_fn_train = get_filter_func(
        ego_valid=hyperparams['filter_plan_valid'], 
        pred_not_parked=hyperparams['filter_pred_not_parked'], 
        pred_near_ego=hyperparams['filter_pred_near_ego'],
        )
    filter_fn_eval = filter_fn_train
    
    # # TODO(pkarkus) Temporarily disable filtering for agentcetric input to make debugging faster.
    # if not scene_centric:
    #     filter_fn_train = None
    #     filter_fn_eval = None

    if use_cache:        
        eval_dataset.load_or_create_cache(cached_eval_data_path, num_workers=hyperparams['preprocess_workers'],
                                        filter_fn=filter_fn_eval)
        train_dataset.load_or_create_cache(cached_train_data_path, num_workers=hyperparams['preprocess_workers'], 
                                           filter_fn=filter_fn_train)                                        
    else:
        eval_dataset.apply_filter(filter_fn=filter_fn_eval, num_workers=hyperparams['preprocess_workers'], max_count=64000, all_gather=all_gather)
        train_dataset.apply_filter(filter_fn=filter_fn_train, num_workers=hyperparams['preprocess_workers'], max_count=512000, all_gather=all_gather)                                     

    eval_dataset.transforms = transforms
    train_dataset.transforms = transforms


    # Create samplers
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=hyperparams["world_size"],
        rank=rank
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=hyperparams["world_size"],
        rank=rank
    )


    # Wrap in a dataloader that samples datapoints and constructs batches
    eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=eval_dataset.get_collate_fn(pad_format="right"),
                                    pin_memory=False if hyperparams['device'] == 'cpu' else True,
                                    batch_size=hyperparams['eval_batch_size'],
                                    shuffle=False,
                                    num_workers=hyperparams['preprocess_workers'],
                                    sampler=eval_sampler)
    train_dataloader = DataLoader(train_dataset,
                                        collate_fn=train_dataset.get_collate_fn(pad_format="right"),
                                        pin_memory=False if hyperparams['device'] == 'cpu' else True,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=False,
                                        num_workers=hyperparams['preprocess_workers'],
                                        sampler=train_sampler)

    input_wrapper = lambda batch: {"batch": batch}

    return train_dataloader, train_sampler, train_dataset, eval_dataloader, eval_sampler, eval_dataset, input_wrapper

