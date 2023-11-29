import numpy as np
import torch

from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Union, Any, Tuple

from nuscenes.map_expansion.map_api import NuScenesMap

from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.maps import RasterizedMap

from diffstack.utils.utils import subsample_future
from diffstack.utils.metrics import compute_ade_pt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes


def legend_unique_labels(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    labels, legend_ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in legend_ids]
    plt.legend(handles, labels, **kwargs)


def plot_plan_input_batch(
    batch: Union[AgentBatch, SceneBatch],
    batch_idx: int,
    ax: Optional[Axes] = None,
    legend: bool = True,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    # For now we just convert SceneBatch to AgentBatch
    if isinstance(batch, SceneBatch):
        batch = batch.to_agent_batch(batch.extras["pred_agent_ind"])

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    ax.set_title(f"{str(agent_type)}/{agent_name}")

    pred_agent_history_xy: torch.Tensor = batch.agent_hist[batch_idx].cpu()
    pred_agent_future_xy: torch.Tensor = batch.agent_fut[batch_idx, :, :2].cpu()
    neighbor_hist = batch.neigh_hist[batch_idx].cpu()
    neighbor_fut = batch.neigh_fut[batch_idx].cpu()
    # The index of the current time step depends on the padding direction when the history is incomplete.
    if batch.history_pad_dir == batch.history_pad_dir.AFTER:
        pred_agent_cur_ind = batch.agent_hist_len[batch_idx].cpu() - 1
        neighbor_cur_ind = batch.neigh_hist_len[batch_idx].cpu() - 1
    else:
        pred_agent_cur_ind = -1
        neighbor_cur_ind = [-1 for _ in range(neighbor_hist.shape[0])]

    robot_ind: torch.Tensor = batch.extras['robot_ind'][batch_idx].cpu()
    lane_projection_points: torch.Tensor = batch.extras['lane_projection_points'][batch_idx].cpu()
    goal: torch.Tensor = batch.extras['goal'][batch_idx].cpu()

    # Map
    if batch.maps is not None:
        agent_from_world_tf: torch.Tensor = batch.agents_from_world_tf[batch_idx].cpu()
        world_from_raster_tf: torch.Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx].cpu()
        )

        agent_from_raster_tf: torch.Tensor = agent_from_world_tf @ world_from_raster_tf

        patch_size: int = batch.maps[batch_idx].shape[-1]

        left_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            0
        ].item()
        right_extent: float = (
            agent_from_raster_tf @ torch.tensor([patch_size, 0.0, 1.0])
        )[0].item()
        bottom_extent: float = (
            agent_from_raster_tf @ torch.tensor([0.0, patch_size, 1.0])
        )[1].item()
        top_extent: float = (agent_from_raster_tf @ torch.tensor([0.0, 0.0, 1.0]))[
            1
        ].item()

        ax.imshow(
            RasterizedMap.to_img(
                batch.maps[batch_idx].cpu(),
                # [[0], [1], [2]]
                # [[0, 1, 2], [3, 4], [5, 6]],
            ),
            extent=(
                left_extent,
                right_extent,
                bottom_extent,
                top_extent,
            ),
            alpha=0.3,
        )

    # Lanes
    if robot_ind >= 0:
        ax.scatter(
            lane_projection_points[:, 0],
            lane_projection_points[:, 1],
            s=15,
            c="black",
            label="Lane projections",
        )
        if 'lanes_near_goal' in batch.extras:
            lanes_near_goal = batch.extras['lanes_near_goal'][batch_idx]
            ax.plot([], [], c="grey", ls="--", label="Goal lanes")
            for lane_near_goal in lanes_near_goal:
                ax.plot(lane_near_goal[:, 0], lane_near_goal[:, 1], c="grey", ls="--")

    # Pred agent
    ax.plot(pred_agent_history_xy[..., 0], pred_agent_history_xy[..., 1], c="orange", ls="--", label="Agent History")
    ax.quiver(
        pred_agent_history_xy[..., 0],
        pred_agent_history_xy[..., 1],
        pred_agent_history_xy[..., -1],
        pred_agent_history_xy[..., -2],
        color="k",
        # scale=50,
        width=2e-3,
    )
    ax.plot(pred_agent_future_xy[..., 0], pred_agent_future_xy[..., 1], c="violet", label="Agent Future")
    ax.scatter(pred_agent_history_xy[pred_agent_cur_ind, 0], pred_agent_history_xy[pred_agent_cur_ind, 1], s=20, c="orangered", label="Agent Current")

    # Ego + goal
    if robot_ind >= 0:
        ego_hist = neighbor_hist[robot_ind]
        ego_fut = neighbor_fut[robot_ind]
        ax.plot(ego_hist[:, 0], ego_hist[:, 1], c="olive", ls="--", label="Ego History")
        ax.plot(ego_fut[:, 0], ego_fut[:, 1], c="darkgreen", label="Ego Future")
        ax.scatter(
            ego_hist[None, neighbor_cur_ind[robot_ind], 0],
            ego_hist[None, neighbor_cur_ind[robot_ind], 1],
            s=20,
            c="red",
            label="Ego Current",
        )
        ax.scatter(goal[None, 0], goal[None, 1], s=15, c="purple", label="Goal")

    # Neighbors
    neighbors_idx = [i for i in range(batch.num_neigh[batch_idx]) if i != robot_ind]
    if len(neighbors_idx) > 0:
        ax.plot([], [], c="olive", ls="--", label="Neighbor History")
        for n in neighbors_idx:
            ax.plot(neighbor_hist[n][:, 0], neighbor_hist[n, :, 1], c="olive", ls="--")

        ax.plot([], [], c="darkgreen", label="Neighbor Future")
        for n in neighbors_idx:
            ax.plot(neighbor_fut[n][:, 0], neighbor_fut[n, :, 1], c="darkgreen")

        ax.scatter(
            torch.stack([neighbor_hist[n][neighbor_cur_ind[n], 0] for n in neighbors_idx], dim=0),
            torch.stack([neighbor_hist[n][neighbor_cur_ind[n], 1] for n in neighbors_idx], dim=0),
            s=20,
            c="gold",
            label="Neighbor Current",
        )

    # Ego conditioning in prediction
    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        raise NotImplementedError()

    # Formatting
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    ax.axis("equal")

    if legend:
        ax.legend(loc="best", frameon=True)

    if show:
        plt.show()

    if close:
        plt.close()

    return ax



def plot_plan_result(
    plan_x: np.ndarray,
    control_x: np.ndarray = None,
    ax: Optional[Axes] = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(plan_x[..., 0], plan_x[..., 1], c="blue", ls="-", label="Plan")
    if control_x is not None:
        ax.plot(control_x[..., 0], control_x[..., 1], c="purple", ls="-", label="Control")
    return ax


def plot_plan_candidates(
    plan_candidates_x: np.ndarray,
    ax: Optional[Axes] = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(plan_candidates_x[..., 0].transpose(1, 0), plan_candidates_x[..., 1].transpose(1, 0), c="lightsteelblue", ls="-", label="Plan candidates")

    return ax




def plot_map(dataroot: Optional[str], map_name: Optional[str], map_patch: Tuple[float, float, float, float], nusc_map: Optional[NuScenesMap], figsize=(24, 24)) -> Tuple[Figure, plt.Axes]:
    if nusc_map is None:
        assert dataroot is not None and map_name is not None, "Must provide map_obj or path to map file."
        nusc_map = NuScenesMap(dataroot, map_name)
    bitmap = None #BitMap(dataroot, map_name, 'basemap')

    return nusc_map.render_map_patch(map_patch, 
                                     ['drivable_area',
                                      'road_segment',
                                      # 'road_block',
                                      'lane',
                                      'ped_crossing',
                                      'walkway',
                                      'stop_line',
                                      'carpark_area',
                                      'road_divider',
                                      'lane_divider'], 
                                     alpha=0.05,
                                     render_egoposes_range=False, 
                                     bitmap=bitmap,
                                     figsize=figsize,
                                     render_legend=False,
                                     )


def visualize_plan_batch(nusc_maps, scenes, x_t, y_t, plan_data, plot_data, titles, plot_styles, num_plots, planner, ph, planh):

    allowed_plot_styles = ["anim_iters", "compare_futures", "fan_with_pred", "compare_fan_vs_mpc"]
    if not all([x in allowed_plot_styles for x in plot_styles]):
        raise ValueError(f"Not all requested plot styles are known.\n  Requested: {plot_styles}.\n  Known: {allowed_plot_styles}")

    plot_margin_m = 60
    plan_color = 'coral'
    ego_gt_color = 'royalblue'
    gt_color = 'black'
    pred_color = 'orange'
    pred_gt_color = 'yellow'
    plan_gt_color = 'red'
    plan_nof_color = 'purple'
    plan_nopred_color = 'grey'
    plan_tree_color = 'brown'
    ego_gthcost_color = 'green'
    mpc_plan_color = 'green'

    plan_metrics, plan_iters = plot_data['plan']
    nopred_plan_metrics, nopred_plan_iters = plot_data['nopred_plan']
    # nof_plan_metrics, nof_plan_iters = plot_data['nof_plan']
    gt_plan_metrics, gt_plan_iters = plot_data['gt_plan']

    output = defaultdict(list)
    plotted_inds = []

    def index_plan_iters(plan_iters, batch_i):
        skip_list = ['x', 'u', 'cost', 'all_gt_neighbors', 'gt_neighbors', 'cost_components', 'hcost_components']
        res = {}
        for k, v in plan_iters.items():
            if k in skip_list:
                continue
            if v is None:
                res[k] = None
            elif isinstance(v, list) or isinstance(v, tuple):
                res[k] = v[batch_i].cpu().numpy()
            elif v.ndim == 1:
                res[k] = v[batch_i].cpu().numpy()
            else:
                res[k] = v[:, batch_i].cpu().numpy()
        return res

    def apply_offset(xy, offset_x):
        if offset_x is not None:
            while offset_x.ndim < xy.ndim:
                offset_x = offset_x[None]
            xy = xy[..., :2] + offset_x[..., :2]
        return xy

    def plot_traj(ax, xy, offset_x, label=None, c=None, plot_dot=True, linewidth=1.5, **kwargs):
        xy = apply_offset(xy, offset_x)
        if plot_dot:
            ax.scatter(xy[[0], ..., 0], xy[[0], ..., 1], c=c, s=80.0)
        ax_plots = ax.plot(xy[..., 0], xy[..., 1], label=label, c=c, linewidth=linewidth, **kwargs)
        return ax_plots
    
    def plot_preds(ax, xy, offset_x, probs=None, label=None, c=None, linewidth=0.75, alphas=None, **kwargs):
        xy = apply_offset(xy, offset_x)
        if probs is None:
            ax_plots = ax.plot(xy[..., 0], xy[..., 1], label=label, c=c, linewidth=linewidth, **kwargs)
        else:
            # Normalize probs to 0.2...1
            if alphas is None:
                alphas = probs / probs.max() * 0.6 + 0.4
            # Unfortunately there is no support for multiple alpha values so we need to loop
            ax_plots = []
            for i in range(xy.shape[1]):
                ax_plots.extend(ax.plot(xy[:, i, 0], xy[:, i, 1], label=label, c=c, linewidth=linewidth, alpha=alphas[i], **kwargs))
        return ax_plots
        
    def plot_lane(ax, lanes, offset_x, label="lane", c='black', marker='x', s=4, **kwargs):
        lanes = apply_offset(lanes, offset_x)
        ax.scatter(lanes[..., 0], lanes[..., 1], label=label, c=c, marker=marker, s=s, linewidths=1, **kwargs)

    plan_batch_filter = plan_iters['plan_batch_filter']
    pred_mus_batch = plot_data['y_dists'].mus[0, plan_batch_filter].cpu().numpy()  # (b, t, N, 2)
    pred_probs_batch = torch.exp(plot_data['y_dists'].log_pis[0, plan_batch_filter, 0]).cpu().numpy()  # (b, N)
    gt_pred_xy_batch = y_t[plan_batch_filter].cpu().numpy()
    gt_pred_target_xy_batch = plot_data['y_for_pred'][plan_batch_filter].cpu().numpy()  # differs from g
    x_t_batch = x_t[plan_batch_filter, -1:, :2].cpu().numpy()
    pred_ade_unbiased = compute_ade_pt(plot_data['predictions'], y_t)[plan_batch_filter]
    pred_ade_biased = compute_ade_pt(plot_data['predictions'], plot_data['y_for_pred'])[plan_batch_filter]
    cost_components_batch = plan_iters['cost_components'].mean(0).cpu().numpy()
    hcost_components_batch = plan_iters['hcost_components'].mean(0).cpu().numpy()

    assert len(gt_plan_metrics['hcost']) == len(plan_metrics['hcost'])
    assert len(pred_ade_unbiased) == len(plan_metrics['hcost'])
    assert cost_components_batch.shape[0] == gt_pred_xy_batch.shape[0]
    assert hcost_components_batch.shape[0] == gt_pred_xy_batch.shape[0]

    for batch_i in range(gt_pred_xy_batch.shape[0]):            
        # Limit to number of requested plots
        if num_plots > 0 and len(plotted_inds) >= num_plots:
            break

        scene = scenes[batch_i]
        offset_x = np.array([scene.x_min, scene.y_min, 0., 0.])
        plan_iters_i = index_plan_iters(plan_iters, batch_i)
        gt_neighbors_xy = plan_iters['gt_neighbors'][:, :, batch_i].cpu().numpy()

        gt_pred_xy = gt_pred_xy_batch[batch_i]
        gt_pred_target_xy = gt_pred_target_xy_batch[batch_i]
        pred_mus = pred_mus_batch[batch_i]
        lanes_xy = plan_iters_i['lanes']
        lane_points_xy = plan_iters_i['lane_points']
        gt_neighbors_xy = gt_neighbors_xy[np.logical_not(np.isnan(gt_neighbors_xy[:, 0, 0]))]

        pred_t_xy = x_t_batch[batch_i]
        gt_pred_xy = np.concatenate([pred_t_xy, gt_pred_xy], axis=0)
        pred_mus = np.concatenate([pred_t_xy[:, None].repeat(pred_mus.shape[1], axis=1), pred_mus], axis=0)

        if planner in ['mpc', 'fan_mpc']:
            last_plan_x = plan_iters['x'][-1][:, batch_i].cpu().numpy() 
            last_nopred_plan_x = nopred_plan_iters['x'][-1][:, batch_i].cpu().numpy()
            # last_nof_plan_x = nof_plan_iters['x'][-1][:, batch_i].cpu().numpy()
            last_gt_plan_x = gt_plan_iters['x'][-1][:, batch_i].cpu().numpy()
            plan_gt_xy = plan_iters_i['x_gt'][:, :2]
            plan_gt_xy = np.concatenate([last_plan_x[:1, :2], plan_gt_xy], axis=0)  # append t0
            plan_cost_i = plan_iters['cost'][-1][batch_i].cpu().numpy()/plan_gt_xy.shape[0]
                    # # skip boring examples
                    # if len(plan_iters_i['gt_neighbors']) < 1:
                    #     continue
                    # is_interesting = False
                    # if (nof_plan_metrics['hcost'][batch_i] < gt_plan_metrics['hcost'][batch_i] - 0.01) and gt_plan_converged[batch_i]:
                    #     is_interesting = True
                    # if nopred_plan_metrics['hcost'][batch_i] > gt_plan_metrics['hcost'][batch_i] + 0.1:
                    #     is_interesting = True
                    # if not is_interesting:
                    #     continue

                    # add present

            # Filter
            # TODO
            plotted_inds.append(batch_i)

            label = (
                        f"{titles[batch_i]} \n" +
                        f"#{batch_i} n={len(gt_neighbors_xy)} conv={str(plan_iters_i['converged'])} " + 
                        f"pred_ade={pred_ade_biased[batch_i]:.3f} {pred_ade_unbiased[batch_i]:.3f} plan_mse={plan_metrics['mse'][batch_i]:.3f} plan_mcost={plan_cost_i:.3f} plan_hcost={plan_metrics['hcost'][batch_i]:.3f} \n" + 
                        # f"nof_plan_hcost={nof_plan_metrics['hcost'][batch_i]:.3f} nopred_plan_hcost={nopred_plan_metrics['hcost'][batch_i]:.3f}  gtplan_hcost={gt_plan_metrics['hcost'][batch_i]:.3f} \n" + 
                        f"nopred_plan_hcost={nopred_plan_metrics['hcost'][batch_i]:.3f}  gtplan_hcost={gt_plan_metrics['hcost'][batch_i]:.3f} \n" + 
                        f"nopred_regret_now={nopred_plan_metrics['hcost'][batch_i]-gt_plan_metrics['hcost'][batch_i]:.4f} " + 
                        f"nopred_regret_cache={plan_data['nopred_plan_hcost'][batch_i]-plan_data['gt_plan_hcost'][batch_i]:.4f}")
            print (label)

            # # Debug cost change
            # print (f"gt plan internal cost: {torch.stack([c[batch_i].cpu() for c in gt_plan_iters['cost']])}")
            # print (f"nof plan internal cost: {torch.stack([c[batch_i].cpu() for c in nof_plan_iters['cost']])}")
            # print (f"nopred plan internal cost: {torch.stack([c[batch_i].cpu() for c in nopred_plan_iters['cost']])}")
            # print (f"pred plan internal cost: {torch.stack([c[batch_i].cpu() for c in plan_iters['cost']])}")

            # Animate planning process
            if "anim_iters" in plot_styles:
                ref_agent_x = last_plan_x + offset_x[None]
                def anim_background():
                    fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_maps[plan_data['map_name'][batch_i]],
                                            # map_name=env_helper.get_map_name_from_sample_token(), 
                                            map_patch=(ref_agent_x[:, 0].min() - plot_margin_m, 
                                                        ref_agent_x[:, 1].min() - plot_margin_m, 
                                                        ref_agent_x[:, 0].max() + plot_margin_m, 
                                                        ref_agent_x[:, 1].max() + plot_margin_m))
                    plot_lane(ax, lanes_xy, offset_x, c="pink")
                    if lane_points_xy is not None:
                        plot_lane(ax, lane_points_xy, offset_x, c="black")
                    plot_preds(ax, pred_mus[:, :, :2], offset_x, label='pred', c=pred_color, probs=pred_probs_batch[batch_i])
                    plot_traj(ax, gt_pred_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=True)
                    # In the case of prediction target offset, this will be different. 
                    plot_traj(ax, gt_pred_target_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=False)                        
                    plt.suptitle(label, fontsize=10)
                    return fig, ax

                def animate_plan_iters(fig, ax, plan_iters, *args, **kwargs):
                    xy = plan_iters['x'][0][:, batch_i].cpu().numpy()
                    ax_plot, = plot_traj(ax, xy, offset_x, *args, **kwargs)
                    ax_text = plt.text(0.05, 0.05, "Iter 0", transform=ax.transAxes)
                    def init():
                        return ax_plot,
                    def animate(i):
                        i = min(i, len(plan_iters['x'])-1)
                        xy = plan_iters['x'][i][:, batch_i].cpu().numpy()
                        xy = apply_offset(xy, offset_x)
                        ax_plot.set_data(xy[..., 0], xy[..., 1])
                        ax_text.set_text(f"Iter {i}; cost={plan_iters['cost'][i][batch_i].cpu().numpy():.3f}")
                        return ax_plot,
                    anim = FuncAnimation(fig, animate, init_func=init,
                                                frames=len(plan_iters['x'])+2, interval=60, blit=False)
                    plt.show()
                    plt.pause(1.0)
                    print("anim done")
                    return anim      
                                        
                # fig, ax = anim_background()
                # anim1 = animate_plan_iters(fig, ax, nopred_plan_iters, label='nopred plan', c=plan_nopred_color)
                # output['anim_iters_nopred'].append(anim1)

                # fig, ax = anim_background()
                # anim2 = animate_plan_iters(fig, ax, gt_plan_iters, label='gt plan', c=plan_gt_color)
                # output['anim_iters_gt'].append(anim2)

                fig, ax = anim_background()
                anim2 = animate_plan_iters(fig, ax, plan_iters, label='mpc plan', c=mpc_plan_color)
                output['anim_iters_mpc'].append(anim2)

                # anim1.save(save_plot_paths[batch_i] + '-nopred_plan.gif',writer='imagemagick', fps=2) 
                # anim2.save(save_plot_paths[batch_i] + '-gt_plan.gif',writer='imagemagick', fps=2) 
                # plt.show()
                # plt.pause(1.0)
                        
            # Plot compare plans
            if "compare_futures" in plot_styles:
                ref_agent_x = last_plan_x + offset_x[None]
                fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_maps[plan_data['map_name'][batch_i]],
                                        map_patch=(ref_agent_x[:, 0].min() - plot_margin_m, 
                                                    ref_agent_x[:, 1].min() - plot_margin_m, 
                                                    ref_agent_x[:, 0].max() + plot_margin_m, 
                                                    ref_agent_x[:, 1].max() + plot_margin_m))

                plot_lane(ax, lanes_xy, offset_x, c="pink")
                if lane_points_xy is not None:
                    plot_lane(ax, lane_points_xy, offset_x, c="black")

                plot_preds(ax, pred_mus[:, :, :2], offset_x, label='pred', c=pred_color, probs=pred_probs_batch[batch_i])
                plot_traj(ax, gt_pred_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=True)
                # In the case of prediction target offset, this will be different. 
                plot_traj(ax, gt_pred_target_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=False)                        
                plot_traj(ax, gt_neighbors_xy.transpose((1, 0, 2)), offset_x, label='gt', c=gt_color)
                # plot_traj(ax, last_nof_plan_x[:, :2], offset_x, label='nofuture plan', c=plan_nof_color, plot_dot=False)
                plot_traj(ax, plan_gt_xy, offset_x, label='ego gt', c=ego_gt_color, plot_dot=True, linewidth=2.0)
                plot_traj(ax, last_gt_plan_x[:, :2], offset_x, label='gt plan', c=plan_gt_color, plot_dot=False)
                plot_traj(ax, last_nopred_plan_x[:, :2], offset_x, label='nopred plan', c=plan_nopred_color, plot_dot=False)
                plot_traj(ax, last_plan_x[:, :2], offset_x, label='mpc plan', c=mpc_plan_color, plot_dot=False)
                        
                # ax.scatter(ref_agent_x[[0], 0], ref_agent_x[[0], 1], label='Ego', c=ego_color)
                # ax.plot(ref_agent_x[:, 0], ref_agent_x[:, 1], label='Ego Motion Plan', c=ego_color)

                plt.suptitle(label, fontsize=10)

                legend_unique_labels(plt.gca())

                output['compare_futures'].append(fig)
                # plt.savefig(save_plot_paths[batch_i] + '-plan.png')
                # plt.show()

        if planner in ['fan', 'fan_mpc']:
            # Plot fan planner
            traj_xy = plan_iters_i['traj_xu'][..., :2]  # N, T+1, 6
            traj_costs = plan_iters_i['traj_cost']
            plan_i = traj_costs.argmin()
            fan_plan_xy = traj_xy[plan_i]
            label_goaldist_xy = traj_xy[plan_iters_i['label_goaldist']]
            label_hcost_xy = traj_xy[plan_iters_i['label_hcost']]
            plan_gt_xy = plan_iters_i['x_gt'][:, :2]                
            plan_gt_xy = np.concatenate([fan_plan_xy[:1, :2], plan_gt_xy], axis=0)  # append t0

            # Filter
            # if not plan_iters_i['converged']:
            #     continue
            plotted_inds.append(batch_i)

            # TODO mse and hcost are wrong for fan_mpc
            label = (
                f"{titles[batch_i]} \n" + 
                f"#{batch_i} n={len(gt_neighbors_xy)} conv={str(plan_iters_i['converged'])} " + 
                f"pred_ade={pred_ade_biased[batch_i]:.3f} {pred_ade_unbiased[batch_i]:.3f} plan_mse={plan_metrics['mse'][batch_i]:.3f} plan_cost={traj_costs[plan_i]:.3f} plan_hcost={plan_metrics['hcost'][batch_i]:.3f} \n" + 
                f"#candid={traj_xy.shape[0]} label_goaldist={plan_iters_i['label_goaldist']} label_hcost={plan_iters_i['label_hcost']} lowest_cost={plan_i} plan_loss={plan_iters_i['plan_loss']:.3f} \n" + 
                f"cost " + " ".join([f"{c:.2f}" for c in cost_components_batch[batch_i]]) + "\n" + 
                f"hcost " + " ".join([f"{c:.2f}" for c in hcost_components_batch[batch_i]])
                )
            print (label)

            if "fan_with_pred" in plot_styles:
                ref_agent_x = plan_gt_xy + offset_x[None, :2]
                fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_maps[plan_data['map_name'][batch_i]],
                                        map_patch=np.array((ref_agent_x[:, 0].min() - plot_margin_m, 
                                                    ref_agent_x[:, 1].min() - plot_margin_m, 
                                                    ref_agent_x[:, 0].max() + plot_margin_m, 
                                                    ref_agent_x[:, 1].max() + plot_margin_m)))

                plot_lane(ax, lanes_xy, offset_x, c="pink")
                if lane_points_xy is not None:
                    plot_lane(ax, lane_points_xy, offset_x, c="black")

                plot_preds(ax, pred_mus[:, :, :2], offset_x, label='pred', c=pred_color, probs=pred_probs_batch[batch_i])
                plot_traj(ax, gt_pred_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=True)
                # In the case of prediction target offset, this will be different. 
                plot_traj(ax, gt_pred_target_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=False)

                plot_traj(ax, gt_neighbors_xy.transpose((1, 0, 2)), offset_x, label='gt', c=gt_color)

                plot_traj(ax, plan_gt_xy, offset_x, label='ego gt', c=ego_gt_color, plot_dot=True)

                # Plot candidate targets only
                plot_lane(ax, traj_xy.transpose((1, 0, 2)), offset_x, label='trajectory fan', c=plan_tree_color)
                # Plot candidate trajectories
                # plot_traj(ax, traj_xy.transpose((1, 0, 2)), offset_x, label='tree', c=plan_tree_color)
                # plot_traj(ax, gt_plan_x[:, :2], offset_x, label='gt plan', c=plan_color, plot_dot=False)
                plot_traj(ax, label_goaldist_xy, offset_x, label='label goaldist', c=ego_gt_color, plot_dot=False)
                plot_traj(ax, label_hcost_xy, offset_x, label='label hcost', c=ego_gthcost_color, plot_dot=False)
                plot_traj(ax, fan_plan_xy, offset_x, label='ego plan', c=plan_color, plot_dot=False)

                        
                        # ax.scatter(ref_agent_x[[0], 0], ref_agent_x[[0], 1], label='Ego', c=ego_color)
                        # ax.plot(ref_agent_x[:, 0], ref_agent_x[:, 1], label='Ego Motion Plan', c=ego_color)

                plt.suptitle(label, fontsize=10)

                legend_unique_labels(plt.gca(), loc='lower left')

                output['fan_with_pred'].append(fig)
                # plt.show()

        if "compare_fan_vs_mpc" in plot_styles:
            # Plot fan_mpc planner
            assert planner == 'fan_mpc'

            traj_xy = plan_iters_i['traj_xu'][..., :2]  # N, T+1, 6
            traj_costs = plan_iters_i['traj_cost']
            plan_i = traj_costs.argmin()
            fan_plan_xy = traj_xy[plan_i]
            label_goaldist_xy = traj_xy[plan_iters_i['label_goaldist']]
            label_hcost_xy = traj_xy[plan_iters_i['label_hcost']]
            plan_gt_xy = plan_iters_i['x_gt'][:, :2]
            plan_gt_xy = np.concatenate([fan_plan_xy[:1, :2], plan_gt_xy], axis=0)  # append t0
            fan_mse = np.square(subsample_future(fan_plan_xy, ph, planh)[1:] - plan_gt_xy[1:, :2]).sum(axis=-1).mean()
            mpc_plan_cost = plan_iters['cost'][-1][batch_i].cpu().numpy() / plan_gt_xy.shape[0] # from sum over time to mean over time

            # Filter
            # if not plan_iters_i['converged']:
            #     continue
            plotted_inds.append(batch_i)

            label = (
                f"{titles[batch_i]} \n" + 
                f"#{batch_i} n={len(gt_neighbors_xy)} conv={str(plan_iters_i['fan_converged'])} {str(plan_iters_i['mpc_converged'])} " + 
                f"#candid={traj_xy.shape[0]} label_goaldist={plan_iters_i['label_goaldist']} label_hcost={plan_iters_i['label_hcost']} lowest_cost={plan_i} plan_loss={plan_iters_i['plan_loss']:.3f} \n" + 
                f"fan_mse={fan_mse:.3f} fan_mcost={traj_costs[plan_i]/plan_gt_xy.shape[0]:.3f} fan_hcost={plan_iters_i['traj_hcost'][plan_i]/plan_gt_xy.shape[0]:.3f} \n" + 
                f"mpc_mse={plan_metrics['mse'][batch_i]:.3f} mpc_mcost={mpc_plan_cost:.3f} mpc_hcost={plan_metrics['hcost'][batch_i]:.3f} \n" +
                f"fan cost " + " ".join([f"{c:.2f}" for c in plan_iters['fan_cost_components'][:, batch_i].mean(0).cpu().numpy()]) + "\n" + 
                f"mpc cost " + " ".join([f"{c:.2f}" for c in cost_components_batch[batch_i]])
                )
            print (label)

            dist = plot_data['y_dists'] 
            # TODO requires batch=1
            # original_batch_i = torch.arange(dist.mus.shape[1], device=dist.mus.device)[plan_batch_filter][batch_i]
            dist.mus = dist.mus.detach().clone()
            dist.mus[..., 0] += offset_x[0]
            dist.mus[..., 1] += offset_x[1]
            ml_k = np.argmax(pred_probs_batch[batch_i])
            pred_ml = pred_mus[:, ml_k]

            for rep_i in range(4):
                ref_agent_x = fan_plan_xy + offset_x[None, :2]
                fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_maps[plan_data['map_name'][batch_i]],
                                        map_patch=np.array((ref_agent_x[:, 0].min() - plot_margin_m/2, 
                                                    ref_agent_x[:, 1].min() - plot_margin_m/2,
                                                    ref_agent_x[:, 0].max() + plot_margin_m/2,
                                                    ref_agent_x[:, 1].max() + plot_margin_m/2)),
                                        # figsize=(8,8),
                                        figsize=(24,24),
                                        )

                dist.log_pis = plot_data['y_dists'].log_pis.detach().clone()
                if rep_i == 0:
                    probs = pred_probs_batch[batch_i]
                    alphas = probs / probs.max() # *  0.6 + 0.4 
                    plot_preds(ax, pred_mus[:, :, :2], offset_x, label='predictions', c='orange', probs=pred_probs_batch[batch_i], alphas=alphas, linewidth=2.0)
                    
                    # visualize_distribution2(plt.gca(), dist, pi_threshold=0.05, color=pred_color, pi_alpha=0.1, topn=1)
                elif rep_i == 1:
                    probs = pred_probs_batch[batch_i]
                    alphas = probs / probs.max() *  0.8 + 0.2
                    plot_preds(ax, pred_mus[:, :, :2], offset_x, label='predictions', c='orange', probs=pred_probs_batch[batch_i], alphas=alphas, linewidth=2.0)
                                            # visualize_distribution2(plt.gca(), dist, pi_threshold=0.05, color=pred_color, pi_alpha=0.1, topn=3)
                elif rep_i == 2:
                    probs = pred_probs_batch[batch_i]
                    alphas = probs / probs.max() *  0.9 + 0.1
                    plot_preds(ax, pred_mus[:, :, :2], offset_x, label='predictions', c='orange', probs=pred_probs_batch[batch_i], alphas=alphas, linewidth=2.0)
                                            # visualize_distribution2(plt.gca(), dist, pi_threshold=0.05, color=pred_color, pi_alpha=0.1)
                else:
                    probs = pred_probs_batch[batch_i]
                    alphas = probs / probs.max() *  0.95 + 0.05
                    plot_preds(ax, pred_mus[:, :, :2], offset_x, label='predictions', c='orange', probs=pred_probs_batch[batch_i], alphas=alphas, linewidth=2.0)
                                            # visualize_distribution2(plt.gca(), dist, pi_threshold=0.05, color=pred_color, pi_alpha=0.1)

                    # visualize_distribution2(plt.gca(), dist, pi_threshold=0.05, color=pred_color, pi_alpha=0.1)

                # plot_preds(ax, pred_mus[:, :, :2], offset_x, label='predictions', c='orange', probs=pred_probs_batch[batch_i], linewidth=2.0)
                plot_traj(ax, gt_pred_xy[:, :2], offset_x, label='gt_future', c=gt_color, linewidth=2.5, plot_dot=True)
                plot_traj(ax, gt_neighbors_xy.transpose((1, 0, 2)), offset_x, label='gt_future', linewidth=2.5, c=gt_color)
                plot_traj(ax, plan_gt_xy, offset_x, label='gt_future', c=gt_color, linewidth=2.5, plot_dot=True)

                # plot_traj(ax, fan_plan_xy, offset_x, label='fan plan', c=plan_color, linewidth=2.0, plot_dot=False)
                plot_traj(ax, last_plan_x[:, :2], offset_x, label='ego_plan', c='red', linewidth=2.5, plot_dot=False)
                # TODO use plot_lane to add markers (at subsampled timesteps) to indicate velocity



                plot_traj(ax, pred_ml[:, :2], offset_x, label='dist_prediction', linewidth=2.5, c=pred_color, plot_dot=False)  # only to make it appear in legend
                plot_traj(ax, pred_ml[:, :2], offset_x, label='ml_prediction', linewidth=2.5, c='yellow', plot_dot=False)


                # Hide grid lines
                ax.grid(False)
                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                
                if rep_i == 0:
                    handles, labels = plt.gca().get_legend_handles_labels()
                    labels, legend_ids = np.unique(labels, return_index=True)
                    # handles = [handles[i] for i in legend_ids]
                    oredered_handles = []
                    oredered_labels =  ['gt_future', 'ego_plan', 'ml_prediction', 'dist_prediction',  'drivable_area', 'lane',
                                'lane_divider', 'walkway', 'ped_crossing', 'stop_line', 'road_divider', 'road_segment',  'carpark_area',
                                ]
                    for lb in oredered_labels:
                        finds = np.where(labels==lb)[0]
                        if len(finds) < 1:
                            continue
                        i = finds[0]
                        oredered_handles.append(handles[legend_ids[i]])
                    plt.legend(oredered_handles, oredered_labels, loc='lower left', framealpha=1.)
                    output['compare_fan_vs_mpc_label'].append(fig)
                elif rep_i == 1:
                    output['compare_fan_vs_mpc'].append(fig)
                else:
                    output['compare_fan_vs_mpc'+str(rep_i)].append(fig)


        if "OLD_compare_fan_vs_mpc" in plot_styles:
            # Plot fan_mpc planner
            assert planner == 'fan_mpc'

            traj_xy = plan_iters_i['traj_xu'][..., :2]  # N, T+1, 6
            traj_costs = plan_iters_i['traj_cost']
            plan_i = traj_costs.argmin()
            fan_plan_xy = traj_xy[plan_i]
            label_goaldist_xy = traj_xy[plan_iters_i['label_goaldist']]
            label_hcost_xy = traj_xy[plan_iters_i['label_hcost']]
            plan_gt_xy = plan_iters_i['x_gt'][:, :2]
            plan_gt_xy = np.concatenate([fan_plan_xy[:1, :2], plan_gt_xy], axis=0)  # append t0
            fan_mse = np.square(subsample_future(fan_plan_xy, ph, planh)[1:] - plan_gt_xy[1:, :2]).sum(axis=-1).mean()
            mpc_plan_cost = plan_iters['cost'][-1][batch_i].cpu().numpy() / plan_gt_xy.shape[0] # from sum over time to mean over time

            # Filter
            # if not plan_iters_i['converged']:
            #     continue
            plotted_inds.append(batch_i)

            label = (
                f"{titles[batch_i]} \n" + 
                f"#{batch_i} n={len(gt_neighbors_xy)} conv={str(plan_iters_i['fan_converged'])} {str(plan_iters_i['mpc_converged'])} " + 
                f"#candid={traj_xy.shape[0]} label_goaldist={plan_iters_i['label_goaldist']} label_hcost={plan_iters_i['label_hcost']} lowest_cost={plan_i} plan_loss={plan_iters_i['plan_loss']:.3f} \n" + 
                f"fan_mse={fan_mse:.3f} fan_mcost={traj_costs[plan_i]/plan_gt_xy.shape[0]:.3f} fan_hcost={plan_iters_i['traj_hcost'][plan_i]/plan_gt_xy.shape[0]:.3f} \n" + 
                f"mpc_mse={plan_metrics['mse'][batch_i]:.3f} mpc_mcost={mpc_plan_cost:.3f} mpc_hcost={plan_metrics['hcost'][batch_i]:.3f} \n" +
                f"fan cost " + " ".join([f"{c:.2f}" for c in plan_iters['fan_cost_components'][:, batch_i].mean(0).cpu().numpy()]) + "\n" + 
                f"mpc cost " + " ".join([f"{c:.2f}" for c in cost_components_batch[batch_i]])
                )
            print (label)

            ref_agent_x = fan_plan_xy + offset_x[None, :2]
            fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_maps[plan_data['map_name'][batch_i]],
                                    map_patch=np.array((ref_agent_x[:, 0].min() - plot_margin_m/2, 
                                                ref_agent_x[:, 1].min() - plot_margin_m/2,
                                                ref_agent_x[:, 0].max() + plot_margin_m/2,
                                                ref_agent_x[:, 1].max() + plot_margin_m/2)))

            plot_lane(ax, lanes_xy, offset_x, c="pink")
            if lane_points_xy is not None:
                plot_lane(ax, lane_points_xy, offset_x, c="black")

            plot_preds(ax, pred_mus[:, :, :2], offset_x, label='pred', c=pred_color, probs=pred_probs_batch[batch_i])
            plot_traj(ax, gt_pred_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=True)
            # In the case of prediction target offset, this will be different. 
            plot_traj(ax, gt_pred_target_xy[:, :2], offset_x, label='pred gt', c=pred_gt_color, plot_dot=False)

            plot_traj(ax, gt_neighbors_xy.transpose((1, 0, 2)), offset_x, label='gt', c=gt_color)

            plot_traj(ax, plan_gt_xy, offset_x, label='ego gt', c=ego_gt_color, linewidth=2.0, plot_dot=True)

            plot_traj(ax, fan_plan_xy, offset_x, label='fan plan', c=plan_color, linewidth=2.0, plot_dot=False)
            plot_traj(ax, last_plan_x[:, :2], offset_x, label='mpc plan', c=mpc_plan_color, linewidth=1.5, plot_dot=False)
            # TODO use plot_lane to add markers (at subsampled timesteps) to indicate velocity

            plt.suptitle(label, fontsize=10)

            legend_unique_labels(plt.gca(), loc='lower left')

            output['compare_fan_vs_mpc'].append(fig)

    # Remove duplicates
    plotted_inds = list(OrderedDict.fromkeys(plotted_inds))
    return output, plotted_inds


def visualize_closed_loop(sim_hist, scenario_metrics, scene, nusc_maps, animate=True):
    """
    all_sim_hist[t][node_type] --> [N][state_dim]
    """
    plot_margin_m = 30
    ego_log_color = 'royalblue'
    ego_sim_color = 'blue'
    gt_color = 'black'
    pred_color = 'orange'
    pred_gt_color = 'yellow'
    plan_color = 'red'

    output = defaultdict(list)

    def apply_offset(xy, offset_x):
        if offset_x is not None and xy is not None:
            while offset_x.ndim < xy.ndim:
                offset_x = offset_x[None]
            xy = xy[..., :2] + offset_x[..., :2]
        return xy

    def plot_traj(ax, xy, offset_x, label=None, c=None, plot_dot=True, dot_last=False, linewidth=1.5, **kwargs):
        xy = apply_offset(xy, offset_x)
        if plot_dot:
            dot_ind = -1 if dot_last else 0
            ax_scatter = ax.scatter(xy[[dot_ind], ..., 0], xy[[dot_ind], ..., 1], c=c, s=80.0)
        else:
            ax_scatter = None
        ax_plots = ax.plot(xy[..., 0], xy[..., 1], label=label, c=c, linewidth=linewidth, **kwargs)
        return ax_plots, ax_scatter
    
    def plot_preds(ax, xy, offset_x, probs=None, label=None, c=None, linewidth=0.75, alphas=None, **kwargs):
        xy = apply_offset(xy, offset_x)
        if probs is None:
            ax_plots = ax.plot(xy[..., 0], xy[..., 1], label=label, c=c, linewidth=linewidth, **kwargs)
        else:
            # Normalize probs to 0.2...1
            if alphas is None:
                alphas = probs / probs.max() * 0.6 + 0.4
            # Unfortunately there is no support for multiple alpha values so we need to loop
            ax_plots = []
            for i in range(xy.shape[1]):
                ax_plots.extend(ax.plot(xy[:, i, 0], xy[:, i, 1], label=label, c=c, linewidth=linewidth, alpha=alphas[i], **kwargs))
        return ax_plots
        
    def plot_lane(ax, lanes, offset_x, label="lane", c='black', marker='x', s=4, **kwargs):
        lanes = apply_offset(lanes, offset_x)
        return ax.scatter(lanes[..., 0], lanes[..., 1], label=label, c=c, marker=marker, s=s, linewidths=1, **kwargs)

    offset_x = np.array([scene.x_min, scene.y_min, 0., 0.])
    nusc_map = nusc_maps[scene.map_name]

    # label = (
    #             f"{titles[batch_i]} \n" +
    #             f"#{batch_i} n={len(gt_neighbors_xy)} conv={str(plan_iters_i['converged'])} " + 
    #             f"pred_ade={pred_ade_biased[batch_i]:.3f} {pred_ade_unbiased[batch_i]:.3f} plan_mse={plan_metrics['mse'][batch_i]:.3f} plan_mcost={plan_cost_i:.3f} plan_hcost={plan_metrics['hcost'][batch_i]:.3f} \n" + 
    #             # f"nof_plan_hcost={nof_plan_metrics['hcost'][batch_i]:.3f} nopred_plan_hcost={nopred_plan_metrics['hcost'][batch_i]:.3f}  gtplan_hcost={gt_plan_metrics['hcost'][batch_i]:.3f} \n" + 
    #             f"nopred_plan_hcost={nopred_plan_metrics['hcost'][batch_i]:.3f}  gtplan_hcost={gt_plan_metrics['hcost'][batch_i]:.3f} \n" + 
    #             f"nopred_regret_now={nopred_plan_metrics['hcost'][batch_i]-gt_plan_metrics['hcost'][batch_i]:.4f} " + 
    #             f"nopred_regret_cache={plan_data['nopred_plan_hcost'][batch_i]-plan_data['gt_plan_hcost'][batch_i]:.4f}")
    # print (label)
    label = f"Closed-loop"


    def subsample_history(sim_hist, i, offset_x):
        plan_xu = np.concatenate([np.concatenate(sim_hist['plan_x'][:i+1], axis=0), np.concatenate(sim_hist['plan_u'][:i+1], axis=0)], axis=-1)  # T+1, b
        gt_ego = plan_xu[..., :2]
        log_ego = np.stack(sim_hist['logged_x'][:i+1], axis=0)[..., :2]

        plan_all_gt_neighbors = np.concatenate(sim_hist['plan_all_gt_neighbors_batch'][i:i+1], axis=1)  # N, 1 -- only last step, otherwise need to deal with nans and association
        goal_batch = sim_hist['logged_x'][-1][..., :2]
        lanes = np.concatenate(sim_hist['lanes'][i], axis=0)   # T+1,
        # TODO support plotting predictions. For now return empty prediction structure.
        empty_mus_batch = np.zeros((0, plan_all_gt_neighbors.shape[1], 1, 1, 2), dtype=np.float32)
        empty_logp_batch = np.zeros((0, 1, 1), dtype=np.float32)
        lane_points = None

        # Separate predicted agent and gt neighbors
        neighbor_invalid = np.isnan(plan_all_gt_neighbors).any(axis=2).any(axis=1)
        plan_all_gt_neighbors = plan_all_gt_neighbors[neighbor_invalid == False]
        gt_neighbors = plan_all_gt_neighbors[:-1]
        gt_pred = plan_all_gt_neighbors[-1:]

        return (
            None, apply_offset(gt_ego, offset_x), apply_offset(log_ego, offset_x),
            apply_offset(gt_neighbors, offset_x), apply_offset(gt_pred, offset_x), 
            apply_offset(empty_mus_batch, offset_x), apply_offset(empty_logp_batch, offset_x), apply_offset(goal_batch, offset_x), 
            apply_offset(lanes, offset_x), apply_offset(lane_points, offset_x))

    # Concatenate all planned trajectory into a dummy collection of reference points
    ref_agent_x = []
    for plan_x in sim_hist['plan_x']:
        ref_agent_x.append(plan_x[:, :4])
    ref_agent_x = np.concatenate(ref_agent_x, axis=0) + offset_x[None]

    def anim_background():
        fig, ax = plot_map(dataroot=None, map_name=None, nusc_map=nusc_map,
                                # map_name=env_helper.get_map_name_from_sample_token(), 
                                map_patch=(ref_agent_x[:, 0].min() - plot_margin_m, 
                                            ref_agent_x[:, 1].min() - plot_margin_m, 
                                            ref_agent_x[:, 0].max() + plot_margin_m, 
                                            ref_agent_x[:, 1].max() + plot_margin_m), figsize=(6, 6))                      
        plt.suptitle(label, fontsize=10)
        ax.set_aspect('equal')
        return fig, ax

    def plot_sim_state(fig, ax, plan_xu, gt_ego, log_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points):
        ax_lane = plot_lane(ax, lanes, None, c="pink")
        ax_gt_agent_traj, ax_gt_agent_dots = plot_traj(ax, gt_neighbors.transpose((1, 0, 2)), None, dot_last=True, label='gt', c=gt_color)
        ax_gt_pred_traj, ax_gt_pred_dots = plot_traj(ax, gt_pred.transpose((1, 0, 2)), None, dot_last=True, label='gt pred', c=pred_gt_color)
        # plot_preds(ax, pred_mus[:, :, :2], offset_x, label='pred', c=pred_color, probs=pred_probs_batch[batch_i])
        ax_ego_traj, ax_ego_dots = plot_traj(ax, gt_ego, None, dot_last=True, label='ego sim', c=ego_sim_color)
        ax_ego_log_traj, ax_ego_log_dots = plot_traj(ax, log_ego, None, plot_dot=False, dot_last=True, label='ego sim', c=ego_log_color)
        ax_text = plt.text(0.05, 0.05, "Step 0", transform=ax.transAxes)
        return ax_lane, ax_gt_agent_traj, ax_gt_agent_dots, ax_gt_pred_traj, ax_gt_pred_dots, ax_ego_traj, ax_ego_dots, ax_ego_log_traj, ax_ego_log_dots, ax_text

    def animate_plan_iters(fig, ax, sim_hist):

        plan_xu, gt_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points = subsample_history(sim_hist, 0, offset_x)            
        ax_lane, ax_gt_agent_traj, ax_gt_agent_dots, ax_gt_pred_traj, ax_gt_pred_dots, ax_ego_traj, ax_ego_dots, ax_ego_log_traj, ax_ego_log_dots, ax_text = plot_sim_state(fig, ax, plan_xu, gt_ego, log_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points)
        def init():
            return ax_lane, ax_gt_agent_traj, ax_gt_agent_dots, ax_gt_pred_traj, ax_gt_pred_dots, ax_ego_traj, ax_ego_dots, ax_ego_log_traj, ax_ego_log_dots, ax_text
        def animate(i):
            i = min(i, len(sim_hist['plan_x'])-1)
            plan_xu, gt_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points = subsample_history(sim_hist, i, offset_x)            

            ax_lane.set_offsets(lanes[-1])
            ax_gt_agent_traj, ax_gt_agent_dots = plot_traj(ax, gt_neighbors.transpose((1, 0, 2)), None, label='gt', c=gt_color)
            ax_gt_pred_traj, ax_gt_pred_dots = plot_traj(ax, gt_pred.transpose((1, 0, 2)), None, label='gt pred', c=gt_color)
            ax_ego_traj, ax_ego_dots = plot_traj(ax, gt_ego, None, label='ego', c=gt_color)                                
            # ax_gt_agent_traj.set_data(gt_neighbors.transpose((1, 0, 2))[..., 0], gt_neighbors.transpose(1, 0, 2)[..., 1])
            # ax_gt_agent_dots.set_offsets(gt_neighbors.transpose((1, 0, 2))[-1])
            # ax_gt_pred_traj.set_data(gt_pred.transpose((1, 0, 2))[..., 0], gt_neighbors.transpose(1, 0, 2)[..., 1])
            # ax_gt_pred_dots.set_offsets(gt_pred.transpose((1, 0, 2))[-1])
            # ax_ego_traj.set_data(gt_ego[..., 0], gt_ego[..., 1])
            # ax_ego_dots.set_offsets(gt_ego[-1])

            ax_text.set_text(f"Step {i}")
            return ax_lane, ax_gt_agent_traj, ax_gt_agent_dots, ax_gt_pred_traj, ax_gt_pred_dots, ax_ego_traj, ax_ego_dots, ax_ego_log_traj, ax_ego_log_dots, ax_text

        anim = FuncAnimation(fig, animate, init_func=init,
                                frames=len(sim_hist['plan_x'])+2, interval=60, blit=False)
        plt.show()
        plt.pause(1.0)
        print("anim done")
        return anim      


    
    for i in range(len(sim_hist['plan_x'])):
        fig, ax = anim_background()
        plan_xu, gt_ego,  log_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points = subsample_history(sim_hist, i, offset_x)            
        plot_sim_state(fig, ax, plan_xu, gt_ego, log_ego, gt_neighbors, gt_pred,  mus, logp, goal, lanes, lane_points)
    plt.show()

    # Animate planning process
    if animate:

        # fig, ax = anim_background()
        # anim1 = animate_plan_iters(fig, ax, nopred_plan_iters, label='nopred plan', c=plan_nopred_color)
        # output['anim_iters_nopred'].append(anim1)

        # fig, ax = anim_background()
        # anim2 = animate_plan_iters(fig, ax, gt_plan_iters, label='gt plan', c=plan_gt_color)
        # output['anim_iters_gt'].append(anim2)

        fig, ax = anim_background()
        anim2 = animate_plan_iters(fig, ax, sim_hist)
        output['anim_closed_loop'].append(anim2)
        anim2.save('./cache/closed-loop' + '.gif',writer='imagemagick', fps=2) 

        # anim2.save(save_plot_paths[batch_i] + '-gt_plan.gif',writer='imagemagick', fps=2) 
        # plt.show()
        # plt.pause(1.0)
 
    return output
