import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffstack.configs.base import TrainConfig
from diffstack.data import scene_batch_extras, agent_batch_extras

from trajdata import AgentBatch, AgentType, UnifiedDataset


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config: TrainConfig):
        super(UnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_batch_size = self._train_config.training.batch_size
        self.val_batch_size = self._train_config.validation.batch_size
        try:
            self.test_batch_size = self._train_config.test["batch_size"]
            self.test_num_workers = self._train_config.validation.num_data_workers
        except:
            self.test_batch_size = self.val_batch_size
            self.test_num_workers = self._train_config.validation.num_data_workers
        self._is_setup = False

    def setup(self, stage=None):
        if self._is_setup:
            return
        data_cfg = self._data_config
        future_sec = (
            data_cfg.future_num_frames * data_cfg.step_time
        )  # python float precision bug
        history_sec = (
            data_cfg.history_num_frames * data_cfg.step_time
        )  # python float precision bug
        neighbor_distance = data_cfg.max_agents_distance
        attention_radius = defaultdict(
            lambda: 20.0
        )  # Default range is 20m unless otherwise specified.
        attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
        attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

        if self._data_config.centric == "scene":
            # Use actual ego as our planning agent, pick the closest other agent to predict
            # for agent-centric prediction models (like T++).
            # We only consider vehicles for prediction agent for now.
            # Scene-centric prediction models can do prediction for all agents.
            pre_filter_transforms = (
                # lambda el: scene_batch_extras.remove_parked(el, keep_agent_ind=0),
                lambda el: scene_batch_extras.role_selector(
                    el, pred_agent_types=[AgentType.VEHICLE]
                ),
            )
            if self._data_config.get("remove_parked", False):
                pre_filter_transforms = pre_filter_transforms + (
                    lambda el: scene_batch_extras.remove_parked(el, keep_agent_ind=0),
                )
            transforms = pre_filter_transforms + (
                lambda el: scene_batch_extras.make_robot_the_first(el),
                # lambda el: scene_batch_extras.augment_with_point_goal(el),
                # lambda el: scene_batch_extras.augment_with_lanes(el, make_missing_lane_invalid=True),
                # lambda el: scene_batch_extras.augment_with_goal_lanes(el),
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
                lambda el: agent_batch_extras.augment_with_lanes(
                    el, make_missing_lane_invalid=True
                ),
                lambda el: agent_batch_extras.augment_with_goal_lanes(el),
            )
            get_filter_func = agent_batch_extras.get_filter_func
        try:
            cache_dir = data_cfg.get("cache_dir", os.environ["TRAJDATA_CACHE_DIR"])
        except:
            cache_dir = "~/.unified_avdata_cache"
        kwargs = dict(
            centric=data_cfg.centric,
            desired_data=[data_cfg.trajdata_source_train],
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                data_cfg.trajdata_source_root: data_cfg.dataset_path,
            },
            # only_types=[AgentType.VEHICLE,AgentType.PEDESTRIAN],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=data_cfg.get("incl_raster_map", False),
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            state_format="x,y,xd,yd,xdd,ydd,s,c",
            obs_format="x,y,xd,yd,xdd,ydd,s,c",
            incl_vector_map=data_cfg.get("incl_vector_map", False),
            verbose=True,
            max_agent_num=1 + data_cfg.other_agents_num,
            # max_neighbor_num = data_cfg.other_agents_num,
            num_workers=min(os.cpu_count(), 64),
            # num_workers = 0,
            ego_only=self._train_config.ego_only,
            transforms=transforms,
            rebuild_cache=self._train_config.rebuild_cache,
            cache_location=cache_dir,
            save_index=False,
        )
        if kwargs["incl_vector_map"]:
            kwargs["vector_map_params"] = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                "max_num_lanes": data_cfg.max_num_lanes,
                "num_lane_pts": data_cfg.num_lane_pts,
                "remove_single_successor": data_cfg.remove_single_successor,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "collate": True,
                "calc_lane_graph": data_cfg.get("calc_lane_graph", False),
            }
            if "waymo" in data_cfg.trajdata_source_root:
                kwargs["vector_map_params"]["keep_in_memory"] = False
                kwargs["vector_map_params"]["radius"] = 300
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        # prepare validation dataset
        kwargs["desired_data"] = [data_cfg.trajdata_source_valid]
        if data_cfg.trajdata_val_source_root is not None:
            kwargs["data_dirs"] = {
                data_cfg.trajdata_val_source_root: data_cfg.dataset_path,
            }
        self.valid_dataset = UnifiedDataset(**kwargs)

        # prepare test dataset if specified
        if data_cfg.trajdata_source_test is not None:
            kwargs["desired_data"] = [data_cfg.trajdata_source_test]
            if data_cfg.trajdata_test_source_root is not None:
                kwargs["data_dirs"] = {
                    data_cfg.trajdata_test_source_root: data_cfg.dataset_path,
                }
            self.test_dataset = UnifiedDataset(**kwargs)
        self._is_setup = True

    def train_dataloader(self):
        batch_name = (
            "scene_batch" if self._data_config.centric == "scene" else "agent_batch"
        )
        collate_fn = lambda *args, **kwargs: {
            batch_name: self.train_dataset.get_collate_fn(return_dict=False)(
                *args, **kwargs
            )
        }

        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=True
            if self._train_config.training.num_data_workers
            else False,
        )

    def val_dataloader(self):
        batch_name = (
            "scene_batch" if self._data_config.centric == "scene" else "agent_batch"
        )
        collate_fn = lambda *args, **kwargs: {
            batch_name: self.valid_dataset.get_collate_fn(return_dict=False)(
                *args, **kwargs
            )
        }

        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=False,
            batch_size=self.val_batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=True
            if self._train_config.validation.num_data_workers > 0
            else False,
        )

    def test_dataloader(self):
        batch_name = (
            "scene_batch" if self._data_config.centric == "scene" else "agent_batch"
        )
        collate_fn = lambda *args, **kwargs: {
            batch_name: self.test_dataset.get_collate_fn(return_dict=False)(
                *args, **kwargs
            )
        }
        return (
            self.val_dataloader()
            if self.test_dataset is None
            else DataLoader(
                dataset=self.test_dataset,
                shuffle=False,
                batch_size=self.test_batch_size,
                num_workers=self.test_num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                persistent_workers=True if self.test_num_workers > 0 else False,
            )
        )

    def predict_dataloader(self):
        return self.val_dataloader
