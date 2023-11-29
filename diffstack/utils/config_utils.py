import json
import hydra
import os

import diffstack
from diffstack.configs.registry import get_registered_experiment_config
from diffstack.configs.base import AlgoConfig, ExperimentConfig
from diffstack.configs.config import Dict
from omegaconf import DictConfig
from pathlib import Path


def load_hydra_config(file_name: str) -> DictConfig:
    fullpath = Path(file_name)
    config_path = Path(diffstack.__path__[0]).parent / "config"
    config_name = (
        fullpath.absolute().relative_to(config_path.absolute()).with_suffix("")
    )
    # Need it to be relative to this file, not the working directory.
    rel_config_path = os.path.relpath(config_path, Path(__file__).parent)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=str(rel_config_path))
    cfg = hydra.compose(config_name=str(config_name), overrides=[])

    return cfg


####### Functions below are related to tbsim style config classes


def recursive_update(config: Dict, external_dict: dict):
    leftover = list()
    for k, v in external_dict.items():
        if k in config and isinstance(config[k], dict):
            config[k] = recursive_update(config[k], v)
        else:
            leftover.append(k)

    leftover_dict = {k: external_dict[k] for k in leftover}
    config.update(**leftover_dict)
    return config


def recursive_update_flat(config: Dict, external_dict: dict):
    left_over = dict()
    for k, v in external_dict.items():
        assert not isinstance(v, dict)
        if k in config:
            assert not isinstance(config[k], dict)
            config[k] = v
        else:
            left_over[k] = v
    if len(left_over) > 0:
        for k, v in config.items():
            if isinstance(v, dict):
                config[k], leftover = recursive_update_flat(v, left_over)
    return config, left_over


def get_experiment_config_from_file(file_path, locked=False):
    ext_cfg = json.load(open(file_path, "r"))
    cfg = get_registered_experiment_config(ext_cfg["registered_name"])
    cfg = recursive_update(cfg, ext_cfg)
    cfg.lock(locked)
    return cfg


def translate_trajdata_cfg(cfg: ExperimentConfig):
    rcfg = Dict()
    # assert cfg.stack.step_time == 0.5  # TODO: support interpolation

    if (
        "predictor" in cfg.stack
        and "scene_centric" in cfg.stack["predictor"]
        and cfg.stack["predictor"].scene_centric
    ):
        rcfg.centric = "scene"
    else:
        rcfg.centric = "agent"
    if "standardize_data" in cfg.env.data_generation_params:
        rcfg.standardize_data = cfg.env.data_generation_params.standardize_data
    else:
        rcfg.standardize_data = True
    if "predictor" in cfg.stack:
        rcfg.step_time = cfg.stack["predictor"].step_time
        rcfg.history_num_frames = cfg.stack["predictor"].history_num_frames
        rcfg.future_num_frames = cfg.stack["predictor"].future_num_frames
    elif "planner" in cfg.stack:
        rcfg.step_time = cfg.stack["planner"].step_time
        rcfg.history_num_frames = cfg.stack["planner"].history_num_frames
        rcfg.future_num_frames = cfg.stack["planner"].future_num_frames
    if "remove_parked" in cfg.env:
        rcfg.remove_parked = cfg.env.remove_parked
    rcfg.trajdata_source_root = cfg.train.trajdata_source_root
    rcfg.trajdata_val_source_root = cfg.train.trajdata_val_source_root
    rcfg.trajdata_source_train = cfg.train.trajdata_source_train
    rcfg.trajdata_source_valid = cfg.train.trajdata_source_valid
    rcfg.trajdata_source_test = cfg.train.trajdata_source_test
    rcfg.trajdata_test_source_root = cfg.train.trajdata_test_source_root
    rcfg.dataset_path = cfg.train.dataset_path

    rcfg.max_agents_distance = cfg.env.data_generation_params.max_agents_distance
    rcfg.num_other_agents = cfg.env.data_generation_params.other_agents_num
    rcfg.max_agents_distance_simulation = cfg.env.simulation.distance_th_close
    rcfg.pixel_size = cfg.env.rasterizer.pixel_size
    rcfg.raster_size = int(cfg.env.rasterizer.raster_size)
    rcfg.raster_center = cfg.env.rasterizer.ego_center
    rcfg.yaw_correction_speed = cfg.env.data_generation_params.yaw_correction_speed
    rcfg.incl_neighbor_map = cfg.env.incl_neighbor_map
    rcfg.incl_vector_map = cfg.env.incl_vector_map
    rcfg.incl_raster_map = cfg.env.incl_raster_map
    rcfg.calc_lane_graph = cfg.env.calc_lane_graph
    rcfg.other_agents_num = cfg.env.data_generation_params.other_agents_num
    rcfg.max_num_lanes = cfg.env.get("max_num_lanes", 15)
    rcfg.remove_single_successor = cfg.env.get("remove_single_successor", False)
    rcfg.num_lane_pts = cfg.env.get("num_lane_pts", 20)
    if "vectorize_lane" in cfg.env.data_generation_params:
        rcfg.vectorize_lane = cfg.env.data_generation_params.vectorize_lane

    else:
        rcfg.vectorize_lane = "None"

    rcfg.lock()
    return rcfg


def boolean_string(s):
    if s not in {"False", "True", "0", "1", "false", "true", "on", "off"}:
        raise ValueError("Not a valid boolean string")
    return s in ["True", "1", "true", "on"]
