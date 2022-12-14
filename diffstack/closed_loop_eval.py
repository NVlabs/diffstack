import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm

# Dataset related
from trajectron.trajectron.model.dataset import collate  # collate needs to be imported like this for cached files to recognize the class. If changed caching has to be redone.

# Model related

from trajectron.trajectron.model.dataset.preprocessing import get_node_timestep_data, get_node_closest_to_robot, pred_state_to_plan_state, plan_state_to_pred_state  # need to import like this to match class instance comparison
from trajectron.trajectron.model.dataset import restore  # import from here for legacy reasons, if changed isinstance will fail
from mpc.util import get_traj


def simulate_scenarios_in_scene(trajectron_module, nusc_maps, env, scene, node_type, hyperparams, replan_every_ns=1, all_scenarios_in_scene=False):
    max_hl = hyperparams['maximum_history_length']
    # Unroll once but calculate metrics at different unroll lengths defined by scene_eval_trajlens
    scene_eval_trajlens = [hyperparams["cl_trajlen"], hyperparams["prediction_horizon"]]      
    # scene_eval_trajlens = [hyperparams["cl_trajlen"]]

    eval_loss = defaultdict(list)

    # Get valid timesteps and nodes for the scene
    timestep = np.arange(scene.timesteps)
    nodes_per_ts = scene.present_nodes(timestep,
                                    type=node_type,
                                    min_history_timesteps=max_hl,
                                    min_future_timesteps=1,
                                    return_robot=not hyperparams['incl_robot_node'])
    valid_timesteps = np.array(sorted(nodes_per_ts.keys()))

    # Construct a set of closed-loop scenarios defined by the start timestep
    if all_scenarios_in_scene:
        valid_start_timesteps = valid_timesteps[:-(hyperparams["cl_trajlen"])]
    else:
        valid_start_timesteps = valid_timesteps[:1]

    # Iterate over scenarios in scene
    for t_sim_start in valid_start_timesteps:
        
        # iterate over replan frequencies
        for replan_every in replan_every_ns:
            scenario_metrics, sim_hist, ego_sim_hist = simulate_scenario(trajectron_module, nusc_maps, env, scene, node_type, nodes_per_ts, t_sim_start, hyperparams, replan_every=replan_every)
            if not scenario_metrics:
                print ("Skip scene, no valid segment for planning")
                continue

            # # # Visualize
            # visualize_closed_loop(sim_hist, scenario_metrics, scene, nusc_maps)

            # Done with unroll, average metrics through time
            for metric, values_through_time in scenario_metrics.items():
                # Mean through time and dummy batch dimension
                for scene_eval_trajlen in scene_eval_trajlens:
                    eval_loss[f'cl{scene_eval_trajlen}r{replan_every}_' + metric].append(
                        np.array(values_through_time[:scene_eval_trajlen]).mean(axis=0)[None])
            eval_loss[f"clr{replan_every}_unroll_len"].append(np.array([ego_sim_hist.shape[0]], np.float32))
            eval_loss[f"clr{replan_every}_active_len"].append(np.array([len(scenario_metrics['mse_t0'])], np.float32))

    return eval_loss


def simulate_scenario(trajectron_module, nusc_maps, env, scene, node_type, nodes_per_ts, t_sim_start, hyperparams, replan_every=1):
    max_hl = hyperparams['maximum_history_length']

    ego_sim_hist = None   # "position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"], "heading": ["°", "d°"],
    all_sim_hist = defaultdict(lambda: defaultdict(list))  # all_sim_hist[t][node_type] --> [N][state_dim]
    plan_hist = defaultdict(list)  # all_sim_hist[var] --> [t]
    scenario_metrics = defaultdict(list)

    steps_since_plan = replan_every
    last_plan = None

    # Return if too short
    if t_sim_start + hyperparams["cl_trajlen"] not in nodes_per_ts:
        return scenario_metrics, all_sim_hist, ego_sim_hist

    # Unroll time
    for t_sim in range(t_sim_start, t_sim_start + hyperparams["cl_trajlen"]):
        # In closed-loop eval we first pick ego, then the predicted agent, which is the opposite of open-loop.
        node = get_node_closest_to_robot(scene, t_sim, node_type, nodes=nodes_per_ts[t_sim])
        # TODO support closest nodes without complete history. Currently present_nodes() filters out these to avoid assert in predict_and_evaluate_batch().
        if node is None:
            print ("No closest node found!")
            return scenario_metrics, all_sim_hist, ego_sim_hist

        # Run preprocessing steps to construct a dummy batch
        sample, (neighbors_data_not_st, logged_robot_data, robot_idx) = get_node_timestep_data(
            env, scene, t_sim, node, trajectron_module.state, trajectron_module.pred_obj.pred_state,
            env.get_edge_types(), max_hl, hyperparams["prediction_horizon"], hyperparams, nusc_maps=nusc_maps,
            is_closed_loop=True, 
            closed_loop_ego_hist=(None if ego_sim_hist is None else ego_sim_hist[-(max_hl+1):]))
        sample = trajectron_module.planner_obj.augment_sample_with_dummy_plan_info(sample)
        batch = collate([sample])
        if logged_robot_data is not None:
            logged_robot_data = logged_robot_data.to(trajectron_module.device)
        batch_i = 0

        # Log
        for log_node_type in env.NodeType:
            if (node_type, log_node_type) in neighbors_data_not_st:
                for node_idx, node_hist in enumerate(neighbors_data_not_st[(node_type, log_node_type)]):
                    if node_idx == robot_idx:
                        all_sim_hist[t_sim]['ROBOT_SIM'].append(node_hist[-1].cpu().numpy())
                    else:
                        all_sim_hist[t_sim][log_node_type].append(node_hist[-1].cpu().numpy())
        if logged_robot_data is not None:
            all_sim_hist[t_sim]['ROBOT_LOG'].append(logged_robot_data[-1].cpu().numpy())
        x_t = sample[1]
        all_sim_hist[t_sim]['PRED'].append(x_t[-1].cpu().numpy())

        if robot_idx < 0:
            # When history is incomplete we cannot plan, so we will just follow the log
            continue 

        ego_log_hist = neighbors_data_not_st[(node_type, 'VEHICLE')][robot_idx]
        if ego_sim_hist is None:
            ego_sim_hist = ego_log_hist.cpu().numpy()

        # Make sure past steps match
        assert np.isclose(ego_log_hist.cpu().numpy(), ego_sim_hist[-ego_log_hist.shape[0]:]).all()
        # # Track deviation
        # print (np.isclose(logged_robot_data.cpu().numpy(), ego_sim_hist[-ego_log_hist.shape[0]:]).all(axis=1))
        # print (np.linalg.norm(logged_robot_data.cpu().numpy()[..., :2] - ego_sim_hist[-ego_log_hist.shape[0]:, ..., :2], axis=-1))
        plan_x0 = pred_state_to_plan_state(ego_sim_hist[-1])[..., :4]
        plan_x0 = torch.tensor(plan_x0, dtype=torch.float, device=trajectron_module.device)  # x, y, yaw, vel

        # Inference. Choose between replan or follow last plan.
        if last_plan is None or steps_since_plan >= replan_every:
            # Replan
            eval_loss_node_type, plot_data = trajectron_module.predict_and_evaluate_batch(batch, node_type, max_hl, return_plot_data=True)
            # Recover plan inputs
            plan_metrics, plan_iters = plot_data['plan']
            # plan_batch_filter = plan_iters['plan_batch_filter']
            plan_x = plan_iters['x'][-1][:, batch_i]
            plan_u = plan_iters['u'][-1][:, batch_i]
            assert torch.isclose(plan_x0, plan_x[0]).all()
            last_plan = (plan_x, plan_u)
            steps_since_plan = 1

            # Metrics for planning
            scenario_metrics['plan_hcost'].append(eval_loss_node_type["plan_hcost"][0].cpu().numpy())
            scenario_metrics['ade_unbiased'].append(eval_loss_node_type["ade_unbiased"][0].cpu().numpy())
            scenario_metrics['nll_mean_unbiased'].append(eval_loss_node_type["nll_mean_unbiased"][0].cpu().numpy())
            # It's not meaningful to log class_hcost_valid, because whether plan is valid will be different depending on ego state.
            # if 'class_hcost_valid' in eval_loss_node_type and eval_loss_node_type["class_hcost_valid"].shape[0] > 0:
            #     scenario_metrics['class_hcost_valid'].append(eval_loss_node_type["class_hcost_valid"][0].cpu().numpy())

            # Log for final cost
            plan_hist['plan_all_gt_neighbors_batch'].append(plan_iters['all_gt_neighbors'][:, :replan_every].squeeze(2))
            plan_hist['lanes'].append(plan_iters['lanes'][:replan_every].squeeze(1))
            plan_hist['plan_x'].append(plan_x[:replan_every])
            plan_hist['plan_u'].append(plan_u[:replan_every]) 

        else:
            # Use the last plan
            plan_x, plan_u = last_plan
            plan_x = plan_x[steps_since_plan:]
            plan_u = plan_u[steps_since_plan:]
            steps_since_plan += 1

        # Unroll unnormalized state with first control step
        plan_x_unroll = get_traj(2, plan_u[:2].unsqueeze(1), plan_x0.unsqueeze(0), trajectron_module.planner_obj.dyn_obj).squeeze(1)
        new_ego_state = torch.concat([plan_x_unroll[1], plan_u[0]], dim=-1).cpu().numpy()
        
        # Convert from planner's state to predictor's state:
        # (x, y, h, v), (acc, dh) --> x, y, vx, vy, ax, ay, heading, delta_heading
        # TODO not sure if prediction state represent acceleration as v[t]-v[t-1] or v[t+1]-v[t] \
        # but it shouldnt matter much
        new_ego_state = plan_state_to_pred_state(new_ego_state)
        # Update sim state
        ego_sim_hist = np.append(ego_sim_hist, new_ego_state[None], axis=0)
        plan_hist['logged_x'].append(logged_robot_data[-1])
        
        # Metrics
        t_metric = 1
        hcost_t = plan_iters['hcost_components'][t_metric, batch_i]
        for i in range(hcost_t.shape[0]):
            scenario_metrics[f'hcost_t{t_metric}_comp{i}'].append(hcost_t[i].cpu().numpy())
        scenario_metrics[f'hcost_t{t_metric}'].append(hcost_t[:].sum(dim=0).cpu().numpy())
        
        icost_t = plan_iters['icost_components'][t_metric, batch_i]
        for i in range(icost_t.shape[0]):
            scenario_metrics[f'icost_t{t_metric}_comp{i}'].append(icost_t[i].cpu().numpy())
        
        mse_t0 = torch.linalg.norm(plan_x0[:2] - logged_robot_data[-1][:2], dim=-1)
        scenario_metrics['mse_t0'].append(mse_t0.cpu().numpy())

    # Eval cost for full trajectory  
    if len(plan_hist['plan_x']) >= 2:
        plan_xu = torch.concat([torch.concat(plan_hist['plan_x'], dim=0), torch.concat(plan_hist['plan_u'], dim=0)], dim=-1).unsqueeze(1)  # T+1, b
        plan_all_gt_neighbors_batch = torch.concat(plan_hist['plan_all_gt_neighbors_batch'], dim=1).unsqueeze(2)  # N, T+1, b
        plan_all_gt_neighbors_batch = plan_all_gt_neighbors_batch[:, 1:]  # normall we dont have t0, N, T, b
        goal_batch = plan_hist['logged_x'][-1][..., :2].unsqueeze(0)  # b
        lanes = torch.concat(plan_hist['lanes'], dim=0).unsqueeze(1)  # T+1, b
        empty_mus_batch = torch.zeros((0, plan_all_gt_neighbors_batch.shape[1], 1, 1, 2), dtype=torch.float, device=trajectron_module.device)
        empty_logp_batch = torch.zeros((0, 1, 1), dtype=torch.float, device=trajectron_module.device)
        lane_points = None

        hcost_components = trajectron_module.planner_obj.cost_obj(
            plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points),
            keep_components=True)  # (T, b, c)        
        hcost_components = hcost_components.squeeze(1)

        icost_components = trajectron_module.planner_obj.interpretable_cost_obj(
            plan_xu, cost_inputs=(plan_all_gt_neighbors_batch, empty_mus_batch, empty_logp_batch, goal_batch, lanes, lane_points),
            keep_components=True)  # (T, b, c)    
        icost_components = icost_components.squeeze(1)

        for i in range(hcost_components.shape[1]):
            scenario_metrics[f'hcost_traj_comp{i}'] = hcost_components[:, i].cpu().numpy()
        scenario_metrics[f'hcost_traj'] = hcost_components.sum(dim=1).cpu().numpy()
        
        for i in range(icost_components.shape[1]):
            scenario_metrics[f'icost_traj_comp{i}'] = icost_components[:, i].cpu().numpy()

    sim_hist = {k: [el.cpu().numpy() for el in v] for k, v in plan_hist.items()}

    return scenario_metrics, sim_hist, ego_sim_hist



# def run_closed_loop_eval(eval_data_loader, trajectron_module, nusc_maps, log_writer, env, hyperparams, epoch, rank, all_scenarios_in_scene=False):
