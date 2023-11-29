import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
from typing import List, Optional, Tuple, Dict
from trajdata.maps.vec_map import VectorMap

from l5kit.geometry import transform_points
from l5kit.rasterization.render_context import RenderContext
from l5kit.configs.config import load_metadata
from trajdata.maps import RasterizedMap

from diffstack.utils.tensor_utils import map_ndarray
from diffstack.utils.geometry_utils import get_box_world_coords_np
import diffstack.utils.tensor_utils as TensorUtils
import os
import glob
from bokeh.models import ColumnDataSource, GlyphRenderer
from bokeh.plotting import figure, curdoc
import bokeh
from bokeh.io import export_png
from trajdata.utils.arr_utils import (
    transform_coords_2d_np,
    batch_nd_transform_points_pt,
    batch_nd_transform_points_np,
)
from trajdata.utils.vis_utils import draw_map_elems

COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
    "ego_contour": "#911A12",
    "ego_fill": "#FE5F55",
}


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def draw_actions(
    state_image,
    trans_mat,
    pred_action=None,
    pred_plan=None,
    pred_plan_info=None,
    ego_action_samples=None,
    plan_samples=None,
    action_marker_size=3,
    plan_marker_size=8,
):
    im = Image.fromarray((state_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    if pred_action is not None:
        raster_traj = agent_to_raster_np(
            pred_action["positions"].reshape(-1, 2), trans_mat
        )
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#FE5F55", outline="#911A12")
    if ego_action_samples is not None:
        raster_traj = agent_to_raster_np(
            ego_action_samples["positions"].reshape(-1, 2), trans_mat
        )
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#808080", outline="#911A12")

    if pred_plan is not None:
        pos_raster = agent_to_raster_np(pred_plan["positions"][:, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    if plan_samples is not None:
        pos_raster = agent_to_raster_np(plan_samples["positions"][0, :, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    im = np.asarray(im)
    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap("jet")
        heatmap = pred_plan_info["location_map"][0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        heatmap = cm(heatmap)

        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(size=(im.shape[1], im.shape[0]))
        heatmap = np.asarray(heatmap)[..., :3]
        padding = np.ones((im.shape[0], 200, 3), dtype=np.uint8) * 255

        composite = heatmap.astype(np.float32) * 0.3 + im.astype(np.float32) * 0.7
        composite = composite.astype(np.uint8)
        im = np.concatenate((im, padding, heatmap, padding, composite), axis=1)

    return im


def draw_agent_boxes(
    image, pos, yaw, extent, raster_from_agent, outline_color, fill_color
):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2)).astype(np.int)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    for b in boxes_raster:
        im_draw.polygon(
            xy=b.reshape(-1).tolist(), outline=outline_color, fill=fill_color
        )

    im = np.asarray(im).astype(np.float32) / 255.0
    return im


def render_state_trajdata(
    batch: dict,
    batch_idx: int,
    action,
) -> np.ndarray:
    pos = batch["hist_pos"][batch_idx, -1]
    yaw = batch["hist_yaw"][batch_idx, -1]
    extent = batch["extent"][batch_idx, :2]

    image = RasterizedMap.to_img(
        TensorUtils.to_tensor(batch["maps"][batch_idx]),
        [[0, 1, 2], [3, 4], [5, 6]],
    )

    image = draw_agent_boxes(
        image,
        pos=pos[None, :],
        yaw=yaw[None, :],
        extent=extent[None, :],
        raster_from_agent=batch["raster_from_agent"][batch_idx],
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"],
    )

    scene_index = batch["scene_index"][batch_idx]
    agent_scene_index = scene_index == batch["scene_index"]
    agent_scene_index[batch_idx] = 0  # don't plot ego

    neigh_pos = batch["centroid"][agent_scene_index]
    neigh_yaw = batch["world_yaw"][agent_scene_index]
    neigh_extent = batch["extent"][agent_scene_index, :2]

    if neigh_pos.shape[0] > 0:
        image = draw_agent_boxes(
            image,
            pos=neigh_pos,
            yaw=neigh_yaw[:, None],
            extent=neigh_extent,
            raster_from_agent=batch["raster_from_world"][batch_idx],
            outline_color=COLORS["agent_contour"],
            fill_color=COLORS["agent_fill"],
        )

    plan_info = None
    plan_samples = None
    action_samples = None
    if "plan_info" in action.agents_info:
        plan_info = TensorUtils.map_ndarray(
            action.agents_info["plan_info"], lambda x: x[[batch_idx]]
        )
    if "plan_samples" in action.agents_info:
        plan_samples = TensorUtils.map_ndarray(
            action.agents_info["plan_samples"], lambda x: x[[batch_idx]]
        )
    if "action_samples" in action.agents_info:
        action_samples = TensorUtils.map_ndarray(
            action.agents_info["action_samples"], lambda x: x[[batch_idx]]
        )

    vis_action = TensorUtils.map_ndarray(
        action.agents.to_dict(), lambda x: x[batch_idx]
    )
    image = draw_actions(
        image,
        trans_mat=batch["raster_from_agent"][batch_idx],
        pred_action=vis_action,
        pred_plan_info=plan_info,
        ego_action_samples=action_samples,
        plan_samples=plan_samples,
        action_marker_size=2,
        plan_marker_size=3,
    )
    return image


def get_state_image_with_boxes_l5kit(ego_obs, agents_obs, rasterizer):
    yaw = ego_obs["world_yaw"]  # set to 0 to fix the video
    state_im = rasterizer.rasterize(ego_obs["centroid"], yaw)

    raster_from_world = rasterizer.render_context.raster_from_world(
        ego_obs["centroid"], yaw
    )
    raster_from_agent = raster_from_world @ ego_obs["world_from_agent"]

    state_im = draw_agent_boxes(
        state_im,
        agents_obs["centroid"],
        agents_obs["world_yaw"][:, None],
        agents_obs["extent"][:, :2],
        raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"],
    )

    state_im = draw_agent_boxes(
        state_im,
        ego_obs["centroid"][None],
        ego_obs["world_yaw"][None, None],
        ego_obs["extent"][None, :2],
        raster_from_world,
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"],
    )

    return state_im, raster_from_agent, raster_from_world


def get_agent_edge(xy, h, extent):
    edges = (
        np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])
        * extent[np.newaxis, :2]
    )
    rotM = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    edges = (rotM @ edges[..., np.newaxis]).squeeze(-1) + xy[np.newaxis, :]
    return edges


def plot_scene_open_loop(
    fig: figure,
    traj: np.ndarray,
    extent: np.ndarray,
    vec_map: VectorMap,
    map_from_world_tf: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    color_scheme="blue_red",
    mask=None,
):
    Na = traj.shape[0]
    static_glyphs = draw_map_elems(fig, vec_map, map_from_world_tf, bbox)
    agent_edge = np.stack(
        [
            get_agent_edge(traj[i, 0, :2], traj[i, 0, 2], extent[i, :2])
            for i in range(Na)
        ],
        0,
    )
    agent_edge = batch_nd_transform_points_np(agent_edge, map_from_world_tf[None])
    agent_patches = defaultdict(lambda: None)
    traj_lines = defaultdict(lambda: None)
    traj_xy = batch_nd_transform_points_np(traj[..., :2], map_from_world_tf[None])

    if color_scheme == "blue_red":
        agent_color = ["red"] + ["blue"] * (Na - 1)
    elif color_scheme == "palette":
        palette = bokeh.palettes.Category20[20]
        agent_color = ["blueviolet"] + [palette[i % 20] for i in range(Na - 1)]
    for i in range(Na):
        if mask is None or mask[i]:
            agent_patches[i] = fig.patch(
                x=agent_edge[i, :, 0],
                y=agent_edge[i, :, 1],
                color=agent_color[i],
            )
            traj_lines[i] = fig.line(
                x=traj_xy[i, :, 0],
                y=traj_xy[i, :, 1],
                color=agent_color[i],
                line_width=2,
            )
    return static_glyphs, agent_patches, traj_lines


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")


def make_gif(frame_folder, gif_name, duration=100, loop=0):
    frames = [
        Image.open(image)
        for image in sorted(glob.glob(f"{frame_folder}/*.png"), key=os.path.getmtime)
    ]
    frame_one = frames[0]
    frame_one.save(
        gif_name,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
    )


def animate_scene_open_loop(
    fig: figure,
    traj: np.ndarray,
    extent: np.ndarray,
    vec_map: VectorMap,
    map_from_world_tf: np.ndarray,
    # rel_bbox: Optional[Tuple[float, float, float, float]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    color_scheme="blue_red",
    mask=None,
    dt=0.1,
    tmp_dir="tmp",
    gif_name="diffstack_anim.gif",
):
    Na, T = traj.shape[:2]
    # traj_xy = batch_nd_transform_points_np(traj[..., :2], map_from_world_tf[None])
    # bbox = (
    #     [
    #         rel_bbox[0] + traj_xy[0, 0, 0],
    #         rel_bbox[1] + traj_xy[0, 0, 0],
    #         rel_bbox[2] + traj_xy[0, 0, 1],
    #         rel_bbox[3] + traj_xy[0, 0, 1],
    #     ]
    #     if rel_bbox is not None
    #     else None
    # )
    static_glyphs = draw_map_elems(fig, vec_map, map_from_world_tf, bbox)
    agent_edge = np.stack(
        [
            get_agent_edge(traj[i, 0, :2], traj[i, 0, 2], extent[i, :2])
            for i in range(Na)
        ],
        0,
    )
    agent_edge = batch_nd_transform_points_np(agent_edge, map_from_world_tf[None])
    agent_patches = defaultdict(lambda: None)
    traj_lines = defaultdict(lambda: None)

    agent_xy_source = defaultdict(lambda: None)
    if color_scheme == "blue_red":
        agent_color = ["red"] + ["blue"] * (Na - 1)
    elif color_scheme == "palette":
        palette = bokeh.palettes.Category20[20]
        agent_color = ["blueviolet"] + [palette[i % 20] for i in range(Na - 1)]
    for i in range(Na):
        if mask is None or mask[i]:
            agent_xy_source[i] = ColumnDataSource(
                data=dict(x=agent_edge[i, :, 0], y=agent_edge[i, :, 1])
            )

            agent_patches[i] = fig.patch(
                x="x",
                y="y",
                color=agent_color[i],
                source=agent_xy_source[i],
                name=f"patch_{i}",
            )
    if os.path.exists(tmp_dir):
        if os.path.isfile(tmp_dir):
            os.remove(tmp_dir)
    else:
        os.mkdir(tmp_dir)
    delete_files_in_directory(tmp_dir)

    for t in range(T):
        agent_edge = np.stack(
            [
                get_agent_edge(traj[i, t, :2], traj[i, t, 2], extent[i, :2])
                for i in range(Na)
            ],
            0,
        )
        agent_edge = batch_nd_transform_points_np(agent_edge, map_from_world_tf[None])
        for i in range(Na):
            if mask is None or mask[i]:
                new_source_data = dict(x=agent_edge[i, :, 0], y=agent_edge[i, :, 1])
                patch = fig.select_one({"name": f"patch_{i}"})
                patch.data_source.data = new_source_data
        export_png(fig, filename=tmp_dir + "/plot_" + str(t) + ".png")

    make_gif(tmp_dir, gif_name, duration=dt * 1000)
    delete_files_in_directory(tmp_dir)
    return static_glyphs, agent_patches, traj_lines
