import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill

from nuscenes.map_expansion import arcline_path_utils
from trajdata.utils.arr_utils import batch_proj

container_abcs = collections.abc


# Wrapper around dict to identify batchable dict data.
class batchable_dict(dict):
    pass


class batchable_list(list):
    pass


class batchable_nonuniform_tensor(torch.Tensor):
    pass


def np_unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif (
        isinstance(elem, str)
        or isinstance(elem, batchable_list)
        or isinstance(elem, batchable_nonuniform_tensor)
    ):
        return dill.dumps(batch) if torch.utils.data.get_worker_info() else batch
    elif isinstance(elem, container_abcs.Sequence):
        if (
            len(elem) == 4
        ):  # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map,
                scene_pts=torch.Tensor(scene_pts),
                patch_size=patch_size[0],
                rotation=heading_angle,
            )
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, batchable_dict):
        # We dill the dictionary for the same reason as the neighbors structure (see below).
        # Unlike for neighbors where we keep a list, here we collate elements recursively
        data_dict = {key: collate([d[key] for d in batch]) for key in elem}
        return (
            dill.dumps(data_dict) if torch.utils.data.get_worker_info() else data_dict
        )
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return (
            dill.dumps(neighbor_dict)
            if torch.utils.data.get_worker_info()
            else neighbor_dict
        )
    try:
        return default_collate(batch)
    except RuntimeError:
        # This happens when tensors are not of the same shape.
        return dill.dumps(batch) if torch.utils.data.get_worker_info() else batch


def get_relevant_lanes_np(nusc_map, scene_offset, position, yaw, vel, dt):
    # - max deceleration
    # - [constant speed]
    # - max acceleration
    # Normally we should do this along the current lane -- but if that's hard to
    # determin reliably, an alternative is to consider both ego-centric and lane centric ways.

    # Simple alternative: use goal state for lane selection.
    # This is biased e.g. when stopped at a traffic light, and we will miss on the
    # lanes across the intersection when gt was stopped, and we will miss on the current
    # nearby lanes when gt accelerates rapidly.

    interp_vel = max(abs(vel), 0.2)
    x = position[0] + scene_offset[0]
    y = position[1] + scene_offset[1]
    state = np.hstack((position, yaw))
    # t = time.time()
    lanes = nusc_map.get_records_in_radius(x, y, 20.0, ["lane", "lane_connector"])
    lanes = lanes["lane"] + lanes["lane_connector"]
    # t_near = time.time() - t
    # t = time.time()
    relevant_lanes = list()
    relevant_lane_tokens = list()
    relevant_lane_arclines = list()
    for lane in lanes:
        lane_arcline = nusc_map.get_arcline_path(lane)
        poses = arcline_path_utils.discretize_lane(
            lane_arcline, resolution_meters=interp_vel * dt
        )
        poses = np.array(poses)
        poses[:, 0:2] -= scene_offset
        delta_x, delta_y, dpsi = batch_proj(state, poses)
        if abs(dpsi[0]) < np.pi / 4 and np.min(np.abs(delta_y)) < 4.5:
            relevant_lanes.append(poses)
            relevant_lane_tokens.append(str(lane))
            relevant_lane_arclines.append(lane_arcline)
    # print (f"Time near: {t_near:.4f} filt: {time.time()-t:.4f}")  # Time near: 0.1969 filt: 0.016
    return relevant_lanes, relevant_lane_tokens, relevant_lane_arclines


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(
        robot_traj, state[robot_type], node_type=robot_type, mean=node_traj, std=std
    )
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def pred_state_to_plan_state(pred_state):
    """
    input: x, y, vx, vy, ax, ay, heading, delta_heading
    output: x, y, heading, v, acc, delta_heading
    """
    x, y, vx, vy, ax, ay, h, dh = np_unstack(pred_state, -1)
    v = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=False)
    a = np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1, keepdims=False)
    # a = np.divide(ax * vx + ay * vy, v, out=np.zeros_like(ax), where=(v > 1.))
    plan_state = np.stack([x, y, h, v, a, dh], axis=-1)

    # assert np.isclose(pred_state, plan_state_to_pred_state(plan_state)).all()  # accelerations mismatch, the calculation of ax and ay cannot be recovered from a_norm and heading
    return plan_state


def plan_state_to_pred_state(plan_state):
    """
    input: x, y, h, v, a, delta_heading
    output: x, y, vx, vy, ax, ay, heading, delta_heading
    """
    x, y, h, v, a, dh = np_unstack(plan_state, -1)
    # Assume vehicle can only move forwards
    vx = v * np.cos(h)
    vy = v * np.sin(h)
    ax = a * np.cos(h)
    ay = a * np.sin(h)
    pred_state = np.stack([x, y, vx, vy, ax, ay, h, dh], axis=-1)
    return pred_state


def get_node_closest_to_robot(scene, t, node_type=None, nodes=None):
    """

    :param scene: Scene
    :param t: Timestep in scene
    """
    get_pose = lambda n: n.get(np.array([t, t]), {"position": ["x", "y"]}, padding=0.0)
    node_dist = lambda a, b: np.linalg.norm(get_pose(a) - get_pose(b))

    robot_node = scene.robot
    closest_node = None
    closest_dist = None

    if nodes is None:
        nodes = scene.nodes

    for node in nodes:
        if node == robot_node:
            continue
        if node_type is not None and node.type != node_type:
            continue
        dist = node_dist(node, robot_node)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_node = node

    return closest_node


def get_node_timestep_data(
    env,
    scene,
    t,
    node,
    state,
    pred_state,
    edge_types,
    max_ht,
    max_ft,
    hyperparams,
    nusc_maps,
    scene_graph=None,
    is_closed_loop=False,
    closed_loop_ego_hist=None,
):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])
    timestep_range_plan = np.array([t, t + max_ft])

    plan_vehicle_state_dict = {
        "position": ["x", "y"],
        "heading": ["°"],
        "velocity": ["norm"],
    }
    if hyperparams["dataset_version"] == "v2":
        plan_vehicle_features_dict = {
            "heading": ["d°"],
            "acceleration": ["norm2"],  # control
            "lane": ["x", "y", "°"],
            "projected": ["x", "y"],  # lane and lane-projected states
            "control_dh": [
                "t" + str(i) for i in range(max_ft + 1)
            ],  # future fitted controls
            "control_a": ["t" + str(i) for i in range(max_ft + 1)],
        }
    else:
        plan_vehicle_features_dict = {
            "heading": ["d°"],
            "acceleration": ["norm2"],  # control
            "lane": ["x", "y", "°"],
            "projected": ["x", "y"],  # lane and lane-projected states
            "control_traj_dh": [
                "t" + str(i) for i in range(max_ft + 1)
            ],  # future fitted controls
            "control_traj_a": ["t" + str(i) for i in range(max_ft + 1)],
            # 'control_goal_dh': ["t"+str(i) for i in range(max_ft+1)],  # future fitted controls
            # 'control_goal_a': ["t"+str(i) for i in range(max_ft+1)],
        }
    plan_vehicle_features_dict_old = {
        "heading": ["d°"],
        "acceleration": ["norm2"],  # control
        "lane": ["x", "y", "°"],
        "projected": ["x", "y"],  # lane and lane-projected states
        "control_dh": [i for i in range(max_ft + 1)],  # future fitted controls
        "control_a": [i for i in range(max_ft + 1)],
    }
    plan_pedestrian_state_dict = {
        "position": ["x", "y"],
    }

    # Filter fields not in data
    state = {
        nk: {k: v for k, v in ndict.items() if k != "augment"}
        for nk, ndict in state.items()
    }

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    # Origin
    x_origin = np.array(x)[-1]

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = x_origin[0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if (
        list(pred_state[node.type].keys())[0] == "position"
    ):  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    scene_offset_np = np.array([scene.x_min, scene.y_min], dtype=np.float32)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    neighbors_future_data = None
    plan_data = None
    if hyperparams["edge_encoding"]:
        # Scene Graph
        scene_graph = (
            scene.get_scene_graph(
                t,
                env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            if scene_graph is None
            else scene_graph
        )

        neighbors_data_not_st = dict()  # closed loop
        logged_robot_data = None  # closed loop
        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        neighbors_future_data = dict()
        is_neighbor_parked = dict()
        for edge_type in edge_types:
            neighbors_data_not_st[edge_type] = list()
            neighbors_data_st[edge_type] = list()
            neighbors_future_data[edge_type] = list()
            is_neighbor_parked[edge_type] = list()
            robot_neighbor = -1

            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams["dynamic_edges"] == "yes":
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(
                    scene_graph.get_edge_scaling(node), dtype=torch.float
                )
                neighbors_edge_value[edge_type] = edge_masks

            for n_i, connected_node in enumerate(connected_nodes):
                neighbor_state_np = connected_node.get(
                    timestep_range_x, state[connected_node.type], padding=0.0
                )

                # Closed loop, replace ego trajectory
                if is_closed_loop and connected_node.is_robot:
                    # assert logged_robot_data is None, "We can only replace robot trajectory once, and we already did that."
                    # We expect to get here twice, once for (VEHICLE, ROBOT) and once for (PEDESTRIAN, ROBOT)
                    logged_robot_data = torch.tensor(
                        neighbor_state_np, dtype=torch.float
                    )
                    if closed_loop_ego_hist is not None:
                        assert (
                            neighbor_state_np.shape[0] <= closed_loop_ego_hist.shape[0]
                        )
                        assert (
                            neighbor_state_np.shape[1] == closed_loop_ego_hist.shape[1]
                        )
                        neighbor_state_np = closed_loop_ego_hist.copy()

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(
                    state[connected_node.type], node_type=connected_node.type
                )
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x_origin[..., :equal_dims]
                neighbor_state_np_st = env.standardize(
                    neighbor_state_np,
                    state[connected_node.type],
                    node_type=connected_node.type,
                    mean=rel_state,
                    std=std,
                )

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)
                if is_closed_loop:
                    neighbors_data_not_st[edge_type].append(
                        torch.tensor(neighbor_state_np, dtype=torch.float)
                    )

                # Add future states for all neighbors. Standardize with same origin and std.
                if edge_type[1] == "VEHICLE":
                    assert connected_node.type == "VEHICLE"

                    try:
                        neighbor_future_features_np = np.concatenate(
                            (
                                # x, y, orient, vel
                                connected_node.get(
                                    timestep_range_plan,
                                    plan_vehicle_state_dict,
                                    padding=np.nan,
                                ),
                                # d_orient, acc_norm  | lane x y heading | projected x y | future_controls steer*ph + acc*ph
                                connected_node.get(
                                    timestep_range_plan,
                                    plan_vehicle_features_dict,
                                    padding=np.nan,
                                ),
                            ),
                            axis=-1,
                        )
                    except KeyError:
                        neighbor_future_features_np = np.concatenate(
                            (
                                # x, y, orient, vel
                                connected_node.get(
                                    timestep_range_plan,
                                    plan_vehicle_state_dict,
                                    padding=np.nan,
                                ),
                                # d_orient, acc_norm  | lane x y heading | projected x y | future_controls steer*ph + acc*ph
                                connected_node.get(
                                    timestep_range_plan,
                                    plan_vehicle_features_dict_old,
                                    padding=np.nan,
                                ),
                            ),
                            axis=-1,
                        )

                    if (
                        is_closed_loop
                        and closed_loop_ego_hist is not None
                        and connected_node.is_robot
                    ):
                        neighbor_future_features_np[0, :6] = pred_state_to_plan_state(
                            closed_loop_ego_hist[-1]
                        )

                    # Add lane points
                    if hyperparams["dataset_version"] == "v2":
                        pass
                    else:
                        lane_ref_points = connected_node.get_lane_points(
                            timestep_range_plan, padding=np.nan, num_lane_points=16
                        )
                        neighbor_future_features_np = np.concatenate(
                            (
                                neighbor_future_features_np,
                                lane_ref_points.reshape(
                                    (lane_ref_points.shape[0], 16 * 3)
                                ),
                            ),
                            axis=1,
                        )

                    # # Assert accelearation norm is correct
                    # acc = connected_node.get(timestep_range_plan, {'acceleration': ['x', 'y']})
                    # acc = np.linalg.norm(acc, axis=-1)
                    # if not np.logical_or(np.isclose(acc, neighbor_future_features_np[:, 5]), np.isnan(acc)).all():
                    #     print (acc, neighbor_future_features_np[:, 5])
                    #     pass
                    # dheading = connected_node.get(timestep_range_plan, {'heading': ['d°']})[:, 0]
                    # if not np.logical_or(np.isclose(dheading, neighbor_future_features_np[:, 4]), np.isnan(dheading)).all():
                    #     print(dheading, neighbor_future_features_np[:, 4])
                    #     pass

                    # if np.isclose(neighbor_future_features_np[:, :2], 0.).any():
                    #     print("issue")
                    #     pass

                    # lane_dist = np.linalg.norm(neighbor_future_features_np[:, :2]-neighbor_future_features_np[:, 6:8], axis=-1)
                    # if np.any(np.nan_to_num(lane_dist, 0.) >= 50.):
                    #     print (lane_dist)
                    #     pass

                    # is_robot_vect = np.ones((neighbor_future_features_np.shape[0], 1)) * float(connected_node.is_robot)
                    # neighbor_future_features_np = np.concatenate((neighbor_future_features_np, is_robot_vect), axis=-1)
                    # is_robot = torch.tensor([float(connected_node.is_robot)], dtype=torch.float)
                    if connected_node.is_robot:
                        robot_neighbor = torch.Tensor([n_i]).int()

                    # Check if car is parked, i.e., it hasnt moved more more than 1m
                    xy_first = connected_node.data[0, {"position": ["x", "y"]}].astype(
                        np.float32
                    )
                    xy_last = connected_node.data[
                        connected_node.last_timestep - connected_node.first_timestep,
                        {"position": ["x", "y"]},
                    ].astype(np.float32)
                    is_neighbor_parked[edge_type].append(
                        np.square(xy_first - xy_last).sum(0) < 1.0**2
                    )

                elif edge_type[1] == "PEDESTRIAN":
                    neighbor_future_features_np = connected_node.get(
                        timestep_range_plan, plan_pedestrian_state_dict, padding=0.0
                    )  # x, y | lane x y heading | projected x y
                else:
                    raise ValueError("Unknown type {}".format(edge_type[1]))
                neighbor_future_features = torch.tensor(
                    neighbor_future_features_np, dtype=torch.float
                )
                neighbors_future_data[edge_type].append(neighbor_future_features)

                assert (
                    hyperparams["dt"] == scene.dt
                )  # we will rely on this hyperparam for delta_t, make sure its correct

            # Find closest neighbor for planning
            # if 'planner' in hyperparams and hyperparams['planner'] and edge_type[1] == 'VEHICLE':
            if edge_type[1] == "VEHICLE":
                vehicle_future_f = neighbors_future_data[(node.type, "VEHICLE")]
                is_parked = is_neighbor_parked[(node.type, "VEHICLE")]

                # Find most relevant agent
                dists = []
                inds = []
                for n_i in range(len(vehicle_future_f)):
                    # Filter incomplete future states, controls or missing lanes
                    if torch.isnan(vehicle_future_f[n_i][:, : (4 + 2 + 3)]).any():
                        continue

                    # Filter parked cars for v7 only.
                    if hyperparams["dataset_version"] not in ["v7"]:
                        if is_parked[n_i]:
                            continue

                    dist = torch.square(
                        y_t - vehicle_future_f[n_i][1:, :2]
                    )  # [1:] exclude current state for vehicle future
                    dist = dist.sum(dim=-1).amin(
                        dim=-1
                    )  # sum over states, min over time
                    inds.append(n_i)
                    dists.append(dist)

                if dists:
                    plan_i = inds[
                        torch.argmin(torch.stack(dists))
                    ]  # neighbor that gets closest to current node
                else:
                    # No neighbors or all futures are incomplete
                    plan_i = -1

                # Robot index, filter incomplete futures, controls or missing lanes
                if (
                    robot_neighbor >= 0
                    and torch.isnan(
                        vehicle_future_f[robot_neighbor][:, : (4 + 2 + 3)]
                    ).any()
                ):
                    robot_i = -1
                else:
                    robot_i = robot_neighbor

                # Pretend robot is the most relevant agent for closed loop
                if is_closed_loop:
                    plan_i = robot_i

                # Get nearby lanes for most_relevant neighbor (used for trajectroy fan planner)
                if plan_i >= 0:
                    nusc_map = nusc_maps[scene.map_name]

                    # Relevant lanes that are near the goal state
                    x_goal_np = vehicle_future_f[plan_i][-1, :4].numpy()  # x,y,h,v
                    pos_xy_goal, yaw_goal, vel_goal = np.split(
                        x_goal_np, (2, 3), axis=-1
                    )
                    (
                        relevant_lanes,
                        relevant_lane_tokens,
                        relevant_lane_arclines,
                    ) = get_relevant_lanes_np(
                        nusc_map,
                        scene_offset_np,
                        pos_xy_goal,
                        yaw_goal,
                        vel_goal,
                        hyperparams["dt"],
                    )
                else:
                    relevant_lanes = []
                    relevant_lane_tokens = []

                # plan_data = torch.Tensor([float(plan_i), float(robot_neighbor)])
                plan_data = batchable_dict(
                    most_relevant_idx=torch.Tensor([int(plan_i)]).int().squeeze(0),
                    robot_idx=torch.Tensor([int(robot_i)]).int().squeeze(0),
                    most_relevant_nearby_lanes=batchable_list(
                        [torch.from_numpy(pts).float() for pts in relevant_lanes]
                    ),
                    most_relevant_nearby_lane_tokens=batchable_list(
                        relevant_lane_tokens
                    ),
                    map_name=str(scene.map_name),
                    scene_offset=torch.from_numpy(scene_offset_np),
                )
    else:
        assert hyperparams["planner"] in ["", "none"]
        plan_data = None

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    if hyperparams["incl_robot_node"]:
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=np.nan)
        robot_traj_st_t = get_relative_robot_traj(
            env, state, x_node, robot_traj, node.type, robot_type
        )
        robot_traj_st_t[torch.isnan(robot_traj_st_t)] = 0.0

    # Map
    map_tuple = None
    if hyperparams["use_map_encoding"]:
        if node.type in hyperparams["map_encoder"]:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams["map_encoder"][node.type]
            if "heading_state_index" in me_hyp:
                heading_state_index = me_hyp["heading_state_index"]
                # We have to rotate the map in the opposit direction of the agent to match them
                if (
                    type(heading_state_index) is list
                ):  # infer from velocity or heading vector
                    heading_angle = (
                        -np.arctan2(
                            x[-1, heading_state_index[1]], x[-1, heading_state_index[0]]
                        )
                        * 180
                        / np.pi
                    )
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams["map_encoder"][node.type]["patch_size"]
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    data_tuple = (
        first_history_index,
        x_t,
        y_t,
        x_st_t,
        y_st_t,
        neighbors_data_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map_tuple,
        neighbors_future_data,
        plan_data,
    )
    if is_closed_loop:
        return (data_tuple, (neighbors_data_not_st, logged_robot_data, robot_i))
    return data_tuple


def get_timesteps_data(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(
        t,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams["incl_robot_node"],
    )
    # Filter fields not in data
    state = {
        nk: {k: v for k, v in ndict.items() if k != "augment"}
        for nk, ndict in state.items()
    }

    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
        scene_graph = scene.get_scene_graph(
            timestep,
            env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )
        present_nodes = nodes_per_ts[timestep]
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(
                get_node_timestep_data(
                    env,
                    scene,
                    timestep,
                    node,
                    state,
                    pred_state,
                    edge_types,
                    max_ht,
                    max_ft,
                    hyperparams,
                    nusc_maps=env.nusc_maps,
                    scene_graph=scene_graph,
                )
            )
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps
