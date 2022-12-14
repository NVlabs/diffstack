import os
from torch.utils import data
import time
import torch
import numpy as np

try:
    from math import prod
except ImportError:
    from functools import reduce  # Required in Python 3
    import operator
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)

from .preprocessing import get_node_timestep_data
from tqdm import tqdm
from diffstack.modules.predictors.trajectron_utils.environment import EnvironmentMetadata
from functools import partial
from pathos.multiprocessing import ProcessPool as Pool


class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


def parallel_process_scene(scene, env_metadata, node_type, 
                           state, pred_state, edge_types, 
                           max_ht, max_ft, 
                           node_freq_mult, scene_freq_mult, 
                           hyperparams, augment, nusc_maps, kwargs):
    results = list()
    indexing_info = list()

    tsteps = np.arange(0, scene.timesteps)
    present_node_dict = scene.present_nodes(tsteps, type=node_type, **kwargs)

    for t, nodes in present_node_dict.items():
        for node in nodes:
            if augment:
                scene_aug = scene.augment()
                node_aug = scene.get_node_by_id(node.id)

                scene_data = get_node_timestep_data(env_metadata, scene_aug, t, node_aug, state, pred_state, 
                                                    edge_types, max_ht, max_ft, hyperparams, nusc_maps)
            else:
                scene_data = get_node_timestep_data(env_metadata, scene, t, node, state, pred_state, 
                                                    edge_types, max_ht, max_ft, hyperparams, nusc_maps)

            results += [(
                scene_data, 
                (scene, t, node)
            )]

            indexing_info += [(
                scene.frequency_multiplier if scene_freq_mult else 1, 
                node.frequency_multiplier if node_freq_mult else 1
            )]

    return (results, indexing_info)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.env_metadata = EnvironmentMetadata(env)
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.index, self.data, self.data_origin = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        num_cpus = kwargs['num_workers']
        del kwargs['num_workers']

        rank = kwargs['rank']
        del kwargs['rank']

        if num_cpus > 0:
            with Pool(num_cpus) as pool:
                indexed_scenes = list(
                    tqdm(
                        pool.imap(
                            partial(parallel_process_scene,
                                    env_metadata=self.env_metadata,
                                    node_type=self.node_type,
                                    state=self.state,
                                    pred_state=self.pred_state,
                                    edge_types=self.edge_types,
                                    max_ht=self.max_ht,
                                    max_ft=self.max_ft,
                                    node_freq_mult=node_freq_mult,
                                    scene_freq_mult=scene_freq_mult,
                                    hyperparams=self.hyperparams,
                                    augment=self.augment,
                                    nusc_maps=self.env.nusc_maps,
                                    kwargs=kwargs),
                            self.env.scenes
                        ),
                        desc=f'Indexing {self.node_type}s ({num_cpus} CPUs)',
                        total=len(self.env.scenes),
                        disable=(rank > 0)
                    )
                )
        else:
            indexed_scenes = [parallel_process_scene(scene, 
                                    env_metadata=self.env_metadata,
                                    node_type=self.node_type,
                                    state=self.state,
                                    pred_state=self.pred_state,
                                    edge_types=self.edge_types,
                                    max_ht=self.max_ht,
                                    max_ft=self.max_ft,
                                    node_freq_mult=node_freq_mult,
                                    scene_freq_mult=scene_freq_mult,
                                    hyperparams=self.hyperparams,
                                    augment=self.augment,
                                    nusc_maps=self.env.nusc_maps,
                                    kwargs=kwargs) for scene in self.env.scenes]

        results = list()
        indexing_info = list()
        for res in indexed_scenes:
            results.extend(res[0])
            indexing_info.extend(res[1])

        index = list()
        for i, counts in enumerate(indexing_info):
            total = prod(counts)

            index += [i]*total

        data, data_origin = zip(*results)

        return np.asarray(index, dtype=int), list(data), list(data_origin)

    def __len__(self):
        return self.index.shape[0]

    def preprocess_online(self, ind):
        batch = self.data[ind]
        (first_history_index, x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st, neighbors_edge_value, robot_traj_st_t,
         map_input, neighbors_future_data, plan_data) = batch

        if 'map_name' not in plan_data:
            scene, _, _ = self.data_origin[ind]
            plan_data['map_name'] = str(scene.map_name)
            plan_data['scene_offset'] = torch.Tensor([scene.x_min, scene.y_min], device='cpu').float()
        if 'most_relevant_nearby_lane_tokens' not in plan_data:
            plan_data['most_relevant_nearby_lane_tokens'] = None

        return (first_history_index, x_t, y_t, x_st_t, y_st_t,
                neighbors_data_st, neighbors_edge_value, robot_traj_st_t,
                map_input, neighbors_future_data, plan_data)

    def __getitem__(self, i):
        # TODO (pkarkus) this seems to lead to memory leak
        # https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662 

        # return self.data[self.index[i]]
        return self.preprocess_online(self.index[i])

    def filter(self, filter_fn, verbose=False):
        tstart = time.time()
        len_start = len(self)

        if filter_fn is not None:
            self.index = np.fromiter((i for i in self.index if filter_fn(self.data[i])), dtype=self.index.dtype)

        if verbose:
            print ("Filter: kept %d/%d (%.1f%%) of samples. Filtering took %.1fs."%(len(self), len_start, len(self)/len_start*100., time.time() - tstart))
