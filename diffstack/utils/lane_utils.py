import numpy as np
import torch
import diffstack.utils.geometry_utils as GeoUtils
from diffstack.utils.geometry_utils import ratan2
import enum
from dataclasses import dataclass
import scipy.interpolate as spint


class LaneModeConst:
    X_ahead_thresh = 5.0
    X_rear_thresh = 0.0
    Y_near_thresh = 1.8
    Y_far_thresh = 5.0
    psi_thresh = np.pi / 4
    longitudinal_scale = 30
    lateral_scale = 1
    heading_scale = 0.5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_edge(lane, dir, W=2.0, num_pts=None):
    if dir == "L":
        if lane.left_edge is not None:
            if num_pts is not None:
                lane.left_edge = lane.left_edge.interpolate(num_pts)
            xy = lane.left_edge.xy
            if lane.left_edge.has_heading:
                h = lane.left_edge.h
            else:
                # check if the points are reversed
                edge_angle = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
                center_angle = np.arctan2(
                    lane.center.xy[-1, 1] - lane.center.xy[0, 1],
                    lane.center.xy[-1, 0] - lane.center.xy[0, 0],
                )
                if np.abs(GeoUtils.round_2pi(edge_angle - center_angle)) > np.pi / 2:
                    xy = np.flip(xy, 0)
                dxy = xy[1:] - xy[:-1]
                h = GeoUtils.round_2pi(np.arctan2(dxy[:, 1], dxy[:, 0]))
                h = np.hstack((h, h[-1]))
        else:
            if num_pts is not None:
                lane.center = lane.center.interpolate(num_pts)
            angle = lane.center.h + np.pi / 2
            offset = np.stack([W * np.cos(angle), W * np.sin(angle)], -1)
            xy = lane.center.xy + offset
            h = lane.center.h
    elif dir == "R":
        if lane.right_edge is not None:
            if num_pts is not None:
                lane.right_edge = lane.right_edge.interpolate(num_pts)
            xy = lane.right_edge.xy
            if lane.right_edge.has_heading:
                h = lane.right_edge.h
            else:
                # check if the points are reversed
                edge_angle = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
                center_angle = np.arctan2(
                    lane.center.xy[-1, 1] - lane.center.xy[0, 1],
                    lane.center.xy[-1, 0] - lane.center.xy[0, 0],
                )
                if np.abs(GeoUtils.round_2pi(edge_angle - center_angle)) > np.pi / 2:
                    xy = np.flip(xy, 0)
                dxy = xy[1:] - xy[:-1]
                h = GeoUtils.round_2pi(np.arctan2(dxy[:, 1], dxy[:, 0]))
                h = np.hstack((h, h[-1]))
        else:
            if num_pts is not None:
                lane.center = lane.center.interpolate(num_pts)
            angle = lane.center.h - np.pi / 2
            offset = np.stack([W * np.cos(angle), W * np.sin(angle)], -1)
            xy = lane.center.xy + offset
            h = lane.center.h
    elif dir == "C":
        if num_pts is not None:
            lane.center = lane.center.interpolate(num_pts)
        xy = lane.center.xy
        if lane.center.has_heading:
            h = lane.center.h
        else:
            dxy = xy[1:] - xy[:-1]
            h = GeoUtils.round_2pi(np.arctan2(dxy[:, 1], dxy[:, 0]))
            h = np.hstack((h, h[-1]))
    return xy, h


def get_bdry_xyh(lane1, lane2=None, dir="L", W=3.6, num_pts=25):
    if lane2 is None:
        xy, h = get_edge(lane1, dir, W, num_pts)
    else:
        xy1, h1 = get_edge(lane1, dir, W, num_pts)
        xy2, h2 = get_edge(lane2, dir, W, num_pts)
        xy = np.concatenate((xy1, xy2), 0)
        h = np.concatenate((h1, h2), 0)
    return xy, h


def LaneRelationFromCfg(lane_relation):
    if lane_relation == "SimpleLaneRelation":
        return SimpleLaneRelation
    elif lane_relation == "LRLaneRelation":
        return LRLaneRelation
    elif lane_relation == "LaneRelation":
        return LaneRelation
    else:
        raise ValueError("Invalid lane relation type")


class SimpleLaneRelation(enum.IntEnum):
    """
    Categorical token describing the relationship between an agent and a Lane, unitary lane mode that only considers which lane the agent is one
    """

    NOTON = 0  # (0, 2, 3, 4, 5, 6)
    ON = 1

    @staticmethod
    def get_all_margins(agent_xysc, lane_xysc, t_range=None, const_override={}):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        return torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        )

    @staticmethod
    def categorize_lane_relation_pts(
        agent_xysc,
        lane_xysc,
        agent_mask=None,
        lane_mask=None,
        t_range=None,
        force_select=True,
        force_unique=True,
        const_override={},
        return_all_margins=False,
    ):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        margin = torch.zeros(
            *x_ahead_margin.shape, len(SimpleLaneRelation), device=x_ahead_margin.device
        )  # margin > 0, then mode is active
        margin[..., SimpleLaneRelation.ON] = torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        ).min(-1)[0]
        margin = fill_margin(
            margin,
            noton_idx=SimpleLaneRelation.NOTON,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )
        if force_select:
            # offset the margin to make sure that at least one is positive
            margin_max = margin[..., SimpleLaneRelation.ON].max(dim=1)[0]
            margin_offset = -margin_max.clip(max=0).detach() + 1e-6
            margin[..., SimpleLaneRelation.ON] = (
                margin[..., SimpleLaneRelation.ON] + margin_offset[:, None, :]
            )

        if force_unique:
            second_largest_margin = margin[..., 1].topk(2, 1, sorted=True)[0][:, 1]
            margin_offset = -second_largest_margin.clip(min=0).detach()
            margin[..., 1] = margin[..., 1] + margin_offset[:, None, :]
        margin[..., SimpleLaneRelation.NOTON] = -margin[..., SimpleLaneRelation.ON]

        flag = get_flag(margin)
        return flag, margin


class LRLaneRelation(enum.IntEnum):
    """Unitary lane mode that considers which lane the agent is on, which lane the ego is on the left of, and which lane the ego is on the right of"""

    NOTON = 0  # (0, 2, 3, 4)
    ON = 1
    LEFTOF = 2  # (5)
    RIGHTOF = 3  # (6)

    @staticmethod
    def get_all_margins(agent_xysc, lane_xysc, t_range=None, const_override={}):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        return torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        )

    @staticmethod
    def categorize_lane_relation_pts(
        agent_xysc,
        lane_xysc,
        agent_mask=None,
        lane_mask=None,
        t_range=None,
        force_select=True,
        force_unique=True,
        const_override={},
    ):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        margin = torch.zeros(
            *x_ahead_margin.shape, len(LRLaneRelation), device=x_ahead_margin.device
        )  # margin > 0, then mode is active
        margin[..., LRLaneRelation.ON] = torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        ).min(-1)[0]
        margin[..., LRLaneRelation.LEFTOF] = torch.stack(
            [x_ahead_margin, x_behind_margin, -y_left_near, y_left_far, psi_margin], -1
        ).min(-1)[
            0
        ]  # further than left near, closer than left far
        margin[..., LRLaneRelation.RIGHTOF] = torch.stack(
            [x_ahead_margin, x_behind_margin, -y_right_near, y_right_far, psi_margin],
            -1,
        ).min(-1)[0]

        margin = fill_margin(
            margin,
            noton_idx=LRLaneRelation.NOTON,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )
        if force_select:
            # offset the margin to make sure that at least one is positive
            margin_max = margin[..., LRLaneRelation.ON].max(dim=1)[0]
            margin_offset = -margin_max.clip(max=0).detach() + 1e-6
            margin[..., LRLaneRelation.ON] = (
                margin[..., LRLaneRelation.ON] + margin_offset[:, None, :]
            )

        if force_unique:
            second_largest_margin = margin[..., LRLaneRelation.ON].topk(
                2, 1, sorted=True
            )[0][:, 1]
            margin_offset = -second_largest_margin.clip(min=0).detach()
            margin[..., LRLaneRelation.ON] = (
                margin[..., LRLaneRelation.ON] + margin_offset[:, None, :]
            )
        idx_excl_noton = torch.arange(margin.shape[-1])
        idx_excl_noton = idx_excl_noton[idx_excl_noton != LRLaneRelation.NOTON]
        margin[..., LRLaneRelation.NOTON] = -margin[..., idx_excl_noton].max(-1)[0]
        flag = get_flag(margin)
        return flag, margin


class LaneRelation(enum.IntEnum):
    """
    pairwise lane mode that gives each agent-lane pair a categorical token describing the relationship between the agent and the lane
    """

    NOTON = 0
    ON = 1
    AHEAD = 2
    BEHIND = 3
    MISALIGN = 4
    LEFTOF = 5
    RIGHTOF = 6

    @staticmethod
    def get_all_margins(agent_xysc, lane_xysc, t_range=None, const_override={}):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        return torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        )

    @staticmethod
    def categorize_lane_relation_pts(
        agent_xysc,
        lane_xysc,
        agent_mask=None,
        lane_mask=None,
        t_range=None,
        force_select=True,
        force_unique=True,
        const_override={},
    ):
        (
            x_ahead_margin,
            x_behind_margin,
            y_left_near,
            y_left_far,
            y_right_near,
            y_right_far,
            psi_margin,
        ) = get_l2a_geometry(
            agent_xysc, lane_xysc, t_range, const_override=const_override
        )
        margin = torch.zeros(
            *x_ahead_margin.shape, len(LaneRelation), device=x_ahead_margin.device
        )  # margin > 0, then mode is active
        margin[..., LaneRelation.ON] = torch.stack(
            [x_ahead_margin, x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        ).min(-1)[0]
        margin[..., LaneRelation.AHEAD] = torch.stack(
            [-x_ahead_margin, y_left_near, y_right_near, psi_margin], -1
        ).min(-1)[0]
        margin[..., LaneRelation.BEHIND] = torch.stack(
            [-x_behind_margin, y_left_near, y_right_near, psi_margin], -1
        ).min(-1)[0]
        margin[..., LaneRelation.MISALIGN] = torch.stack(
            [y_left_near, y_right_near, -psi_margin], -1
        ).min(-1)[0]
        margin[..., LaneRelation.LEFTOF] = torch.stack(
            [x_ahead_margin, x_behind_margin, -y_left_near, y_left_far, psi_margin], -1
        ).min(-1)[
            0
        ]  # further than left near, closer than left far
        margin[..., LaneRelation.RIGHTOF] = torch.stack(
            [x_ahead_margin, x_behind_margin, -y_right_near, y_right_far, psi_margin],
            -1,
        ).min(-1)[0]

        margin = fill_margin(
            margin,
            noton_idx=LaneRelation.NOTON,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )
        flag = get_flag(margin)
        return flag, margin


def get_l2a_geometry(agent_xysc, lane_xysc, t_range=None, const_override={}):
    const = LaneModeConst(**const_override)
    # agent_xysc:[B,T,4], lane_xysc:[B,M,L,4]
    # lane_mask: [B,M], agent_mask: [B,T]
    B, T = agent_xysc.shape[:2]
    M, L = lane_xysc.shape[1:3]

    # idx1 = max(int(T*0.3),1)
    # idx2 = min(T-idx1,T-1)

    dx = GeoUtils.batch_proj_xysc(
        agent_xysc.repeat_interleave(M, 0).reshape(-1, 4),
        lane_xysc.repeat_interleave(T, 1).reshape(-1, L, 4),
    ).reshape(
        B, M, T, L, -1
    )  # [B,M,T,L,xdim]
    close_idx = (
        dx[..., 0].abs().argmin(-1)
    )  # Take first element (x-pos), and find closest index (of L) within each lane segment
    proj_pts = dx.gather(-2, close_idx[..., None, None].repeat(1, 1, 1, 1, 4)).squeeze(
        -2
    )  # Get projection points using the closest point for each lane seg [B,M,T,4]
    psi = ratan2(proj_pts[..., 2], proj_pts[..., 3]).detach()
    y_dev = proj_pts[..., 1]
    # Hausdorff-like distance
    x_ahead_margin = (
        const.X_ahead_thresh + dx[..., 0].max(-1)[0]
    ) / const.longitudinal_scale  # We only have to check the minimal value
    x_behind_margin = (
        -dx[..., 0].min(-1)[0] + const.X_rear_thresh
    ) / const.longitudinal_scale
    y_left_near = (const.Y_near_thresh + y_dev) / const.lateral_scale
    y_left_far = (const.Y_far_thresh + y_dev) / const.lateral_scale
    y_right_near = (const.Y_near_thresh - y_dev) / const.lateral_scale
    y_right_far = (const.Y_far_thresh - y_dev) / const.lateral_scale
    psi_margin = (const.psi_thresh - psi.abs()) / const.heading_scale
    if t_range is not None:
        t0, t1 = t_range
        x_ahead_margin = x_ahead_margin[:, :, t0:t1].mean(dim=2)
        x_behind_margin = x_behind_margin[:, :, t0:t1].mean(dim=2)
        y_left_near = y_left_near[:, :, t0:t1].mean(dim=2)
        y_left_far = y_left_far[:, :, t0:t1].mean(dim=2)
        y_right_near = y_right_near[:, :, t0:t1].mean(dim=2)
        y_right_far = y_right_far[:, :, t0:t1].mean(dim=2)
        psi_margin = psi_margin[:, :, t0:t1].mean(dim=2)
    return (
        x_ahead_margin,
        x_behind_margin,
        y_left_near,
        y_left_far,
        y_right_near,
        y_right_far,
        psi_margin,
    )


def get_ypsi_dev(agent_xysc, lane_xysc):
    # agent_xysc:[B,T,4], lane_xysc:[B,M,L,4]
    # lane_mask: [B,M], agent_mask: [B,T]
    B, T = agent_xysc.shape[:2]
    M, L = lane_xysc.shape[1:3]

    # idx1 = max(int(T*0.3),1)
    # idx2 = min(T-idx1,T-1)

    dx = GeoUtils.batch_proj_xysc(
        agent_xysc.repeat_interleave(M, 0).reshape(-1, 4),
        lane_xysc.repeat_interleave(T, 1).reshape(-1, L, 4),
    ).reshape(
        B, M, T, L, -1
    )  # [B,M,T,L,xdim]
    close_idx = (
        dx[..., 0].abs().argmin(-1)
    )  # Take first element (x-pos), and find closest index (of L) within each lane segment
    proj_pts = dx.gather(-2, close_idx[..., None, None].repeat(1, 1, 1, 1, 4)).squeeze(
        -2
    )  # Get projection points using the closest point for each lane seg [B,M,T,4]
    psi = ratan2(proj_pts[..., 2], proj_pts[..., 3]).detach()
    y_dev = proj_pts[..., 1]
    return y_dev, psi


def fill_margin(margin, noton_idx, agent_mask=None, lane_mask=None):
    idx_excl_noton = torch.arange(margin.shape[-1])
    idx_excl_noton = idx_excl_noton[idx_excl_noton != noton_idx]
    # put anything that does not belong to all the classes above to NOTON
    margin[..., noton_idx] = -margin[..., idx_excl_noton].max(-1)[
        0
    ]  # Negation of the max of the rest

    # Put anything that is masked out to NOTON
    margin[..., noton_idx] = margin[..., noton_idx].masked_fill(
        torch.logical_not(agent_mask).unsqueeze(1), 10
    )  # agents we're not considering set to noton
    margin[..., idx_excl_noton] = margin[..., idx_excl_noton].masked_fill(
        torch.logical_not(agent_mask)[:, None, :, None], -10
    )
    margin[..., noton_idx] = margin[..., noton_idx].masked_fill(
        torch.logical_not(lane_mask).unsqueeze(2), 10
    )
    margin[..., idx_excl_noton] = margin[..., idx_excl_noton].masked_fill(
        torch.logical_not(lane_mask)[:, :, None, None], -10
    )
    return margin


def get_flag(margin):
    flag = (margin >= 0).float()
    # HACK: suppress multiple on flags
    # if flag.shape[-1] > 2:
    #    flag[..., 2:] = flag[..., 2:].masked_fill(
    #        (flag[..., 1:2] > 0).repeat_interleave(flag.shape[-1] - 2, -1), 0
    #    )
    # assert (flag.sum(-1) == 1).all()
    return flag


def get_ref_traj(agent_xyh, lane_xyh, des_vel, dt, T):
    """calculate reference trajectory along the lane center

    Args:
        agent_xyvsc (np.ndarray): B,3
        lane_xysc (np.ndarray): B,L,3
        des_vel (np.ndarray): B
    """
    B = agent_xyh.shape[0]
    delta_x, _, _ = GeoUtils.batch_proj(agent_xyh, lane_xyh)
    indices = np.abs(delta_x).argmin(-1)
    xrefs = list()
    for agent, lane, idx, delta_x_i, vel in zip(
        agent_xyh, lane_xyh, indices, delta_x, des_vel
    ):
        idx = min(idx, lane.shape[0] - 2)
        s = np.linalg.norm(
            lane[idx + 1 :, :2] - lane[idx:-1, :2], axis=-1, keepdims=True
        ).cumsum()
        s = np.insert(s, 0, 0.0) - delta_x_i[idx]
        f = spint.interp1d(
            s,
            lane[idx:],
            axis=0,
            assume_sorted=True,
            bounds_error=False,
            fill_value="extrapolate",
        )
        xref = f(vel * np.arange(1, T + 1) * dt)
        if np.isnan(xref).any():
            xref = np.zeros((T, 3))
        xrefs.append(xref)
    return np.stack(xrefs, 0)


def get_closest_lane_pts(xyh, lane):
    delta_x, _, _ = GeoUtils.batch_proj(xyh, lane.center.xyh)
    if isinstance(xyh, np.ndarray):
        return np.abs(delta_x).argmin()
    elif isinstance(xyh, torch.Tensor):
        return delta_x.abs().argmin()


def test_edge():
    import pickle
    import torch

    with open("sf_test.pkl", "rb") as f:
        data = pickle.load(f)
    lane_xyh = data["lane_xyh"]
    lane_feat = torch.cat(
        [
            lane_xyh[..., :2],
            torch.sin(lane_xyh[..., 2:3]),
            torch.cos(lane_xyh[..., 2:3]),
        ],
        -1,
    )
    ego_xycs = data["agent_hist"][:, 0, :, [0, 1, 6, 7]]

    get_l2a_geometry(ego_xycs, lane_feat, [0, 4])

    print("123")


if __name__ == "__main__":
    test_edge()
