import torch

import diffstack.utils.trajdata_utils as av_utils
from diffstack import dynamics as dynamics
from diffstack.configs.base import ExperimentConfig


global BATCH_TYPE

BATCH_TYPE = "trajdata"
# def set_global_batch_type(batch_type):
#     global BATCH_TYPE
#     assert batch_type in ["trajdata", "l5kit"]
#     BATCH_TYPE = batch_type


def batch_utils(**kwargs):
    if BATCH_TYPE == "trajdata":
        return trajdataBatchUtils(**kwargs)
    else:
        raise NotImplementedError(
            "Please set BATCH_TYPE in batch_utils.py to {trajdata, l5kit}"
        )


class BatchUtils(object):
    """A base class for processing environment-independent batches"""

    def __init__(self, **kwargs):
        if "parse" in kwargs:
            self.parse = kwargs["parse"]
        else:
            self.parse = True
        if "rasterize_mode" in kwargs:
            self.rasterize_mode = kwargs["rasterize_mode"]
        else:
            self.rasterize_mode = "point"

    @staticmethod
    def get_last_available_index(avails):
        """
        Args:
            avails (torch.Tensor): target availabilities [B, (A), T]

        Returns:
            last_indices (torch.Tensor): index of the last available frame
        """
        num_frames = avails.shape[-1]
        inds = torch.arange(0, num_frames).to(avails.device)  # [T]
        inds = (
            avails > 0
        ).float() * inds  # [B, (A), T] arange indices with unavailable indices set to 0
        last_inds = inds.max(dim=-1)[
            1
        ]  # [B, (A)] calculate the index of the last availale frame
        return last_inds

    @staticmethod
    def get_current_states(batch: dict, dyn_type: dynamics.DynType) -> torch.Tensor:
        """Get the dynamic states of the current timestep"""
        bs = batch["curr_speed"].shape[0]
        if dyn_type == dynamics.DynType.BICYCLE:
            current_states = torch.zeros(bs, 6).to(
                batch["curr_speed"].device
            )  # [x, y, yaw, vel, dh, veh_len]
            current_states[:, 3] = batch["curr_speed"].abs()
            current_states[:, [4]] = (
                batch["hist_yaw"][:, 0] - batch["hist_yaw"][:, 1]
            ).abs()
            current_states[:, 5] = batch["extent"][:, 0]  # [veh_len]
        else:
            current_states = torch.zeros(bs, 4).to(
                batch["curr_speed"].device
            )  # [x, y, vel, yaw]
            current_states[:, 2] = batch["curr_speed"]
        return current_states

    @classmethod
    def get_current_states_all_agents(
        cls, batch: dict, step_time, dyn_type: dynamics.DynType
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def parse_batch(data_batch):
        raise NotImplementedError

    @staticmethod
    def batch_to_raw_all_agents(data_batch, step_time):
        raise NotImplementedError

    @staticmethod
    def batch_to_target_all_agents(data_batch):
        raise NotImplementedError

    @staticmethod
    def get_edges_from_batch(data_batch, ego_predictions=None, all_predictions=None):
        raise NotImplementedError

    @staticmethod
    def generate_edges(raw_type, extents, pos_pred, yaw_pred):
        raise NotImplementedError

    @staticmethod
    def gen_ego_edges(
        ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types
    ):
        raise NotImplementedError

    @staticmethod
    def gen_EC_edges(
        ego_trajectories,
        agent_trajectories,
        ego_extents,
        agent_extents,
        raw_types,
        mask=None,
    ):
        raise NotImplementedError

    @staticmethod
    def get_drivable_region_map(rasterized_map):
        raise NotImplementedError

    @staticmethod
    def get_modality_shapes(cfg: ExperimentConfig):
        raise NotImplementedError


class trajdataBatchUtils(BatchUtils):
    """Batch utils for trajdata"""

    def parse_batch(self, data_batch):
        if self.parse:
            return av_utils.parse_trajdata_batch(data_batch, self.rasterize_mode)
        else:
            return data_batch

    @staticmethod
    def batch_to_raw_all_agents(data_batch, step_time):
        raw_type = torch.cat(
            (data_batch["agent_type"].unsqueeze(1), data_batch["neigh_types"]),
            dim=1,
        ).type(torch.int64)
        if data_batch["num_neigh"].max() > 0:
            src_pos = torch.cat(
                (
                    data_batch["hist_pos"].unsqueeze(1),
                    data_batch["neigh_hist_pos"],
                ),
                dim=1,
            )
            src_yaw = torch.cat(
                (
                    data_batch["hist_yaw"].unsqueeze(1),
                    data_batch["neigh_hist_yaw"],
                ),
                dim=1,
            )
            src_mask = torch.cat(
                (
                    data_batch["hist_mask"].unsqueeze(1),
                    data_batch["neigh_hist_mask"],
                ),
                dim=1,
            ).bool()

            extents = torch.cat(
                (
                    data_batch["extent"][..., :2].unsqueeze(1),
                    data_batch["neigh_extents"][..., :2],
                ),
                dim=1,
            )

            curr_speed = torch.cat(
                (data_batch["curr_speed"].unsqueeze(1), data_batch["neigh_curr_speed"]),
                dim=1,
            )
        else:
            src_pos = data_batch["hist_pos"].unsqueeze(1)
            src_yaw = data_batch["hist_yaw"].unsqueeze(1)
            src_mask = data_batch["hist_mask"].unsqueeze(1)
            extents = data_batch["extent"][..., :2].unsqueeze(1)
            curr_speed = data_batch["curr_speed"].unsqueeze(1)

        return {
            "hist_pos": src_pos,
            "hist_yaw": src_yaw,
            "curr_speed": curr_speed,
            "raw_types": raw_type,
            "hist_mask": src_mask,
            "extents": extents,
        }

    @staticmethod
    def batch_to_target_all_agents(data_batch):
        pos = torch.cat(
            (
                data_batch["fut_pos"].unsqueeze(1),
                data_batch["neigh_fut_pos"],
            ),
            dim=1,
        )
        yaw = torch.cat(
            (
                data_batch["fut_yaw"].unsqueeze(1),
                data_batch["neigh_fut_yaw"],
            ),
            dim=1,
        )
        avails = torch.cat(
            (
                data_batch["fut_mask"].unsqueeze(1),
                data_batch["neigh_fut_mask"],
            ),
            dim=1,
        )

        extents = torch.cat(
            (
                data_batch["extent"][..., :2].unsqueeze(1),
                data_batch["neigh_extents"][..., :2],
            ),
            dim=1,
        )

        return {"fut_pos": pos, "fut_yaw": yaw, "fut_mask": avails, "extents": extents}

    @staticmethod
    def get_current_states_all_agents(
        batch: dict, step_time, dyn_type: dynamics.DynType
    ) -> torch.Tensor:
        if batch["hist_pos"].ndim == 3:
            state_all = trajdataBatchUtils.batch_to_raw_all_agents(batch, step_time)
        else:
            state_all = batch
        bs, na = state_all["curr_speed"].shape[:2]
        if dyn_type == dynamics.DynType.BICYCLE:
            current_states = torch.zeros(bs, na, 6).to(
                state_all["curr_speed"].device
            )  # [x, y, yaw, vel, dh, veh_len]
            current_states[:, :, :2] = state_all["hist_pos"][:, :, -1]
            current_states[:, :, 3] = state_all["curr_speed"].abs()
            current_states[:, :, [4]] = (
                state_all["hist_yaw"][:, :, -1] - state_all["hist_yaw"][:, :, 1]
            ).abs()
            current_states[:, :, 5] = state_all["extent"][:, :, -1]  # [veh_len]
        else:
            current_states = torch.zeros(bs, na, 4).to(
                state_all["curr_speed"].device
            )  # [x, y, vel, yaw]
            current_states[:, :, :2] = state_all["hist_pos"][:, :, -1]
            current_states[:, :, 2] = state_all["curr_speed"]
            current_states[:, :, 3:] = state_all["hist_yaw"][:, :, -1]
        return current_states

    @staticmethod
    def get_edges_from_batch(data_batch, ego_predictions=None, all_predictions=None):
        raise NotImplementedError

    @staticmethod
    def generate_edges(raw_type, extents, pos_pred, yaw_pred, batch_first=False):
        return av_utils.generate_edges(
            raw_type, extents, pos_pred, yaw_pred, batch_first=batch_first
        )

    @staticmethod
    def gen_ego_edges(
        ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types
    ):
        return av_utils.gen_ego_edges(
            ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types
        )

    @staticmethod
    def gen_EC_edges(
        ego_trajectories,
        agent_trajectories,
        ego_extents,
        agent_extents,
        raw_types,
        mask=None,
    ):
        return av_utils.gen_EC_edges(
            ego_trajectories,
            agent_trajectories,
            ego_extents,
            agent_extents,
            raw_types,
            mask,
        )

    @staticmethod
    def get_drivable_region_map(rasterized_map):
        return av_utils.get_drivable_region_map(rasterized_map)

    def get_modality_shapes(self, cfg: ExperimentConfig):
        return av_utils.get_modality_shapes(cfg, rasterize_mode=self.rasterize_mode)
