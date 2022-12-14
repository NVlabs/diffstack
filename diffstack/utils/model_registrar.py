import os
import torch
import torch.nn as nn


def get_model_device(model):
    return next(model.parameters()).device


class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        super(ModelRegistrar, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_dir = model_dir
        self.device = device

    def forward(self):
        raise NotImplementedError('Although ModelRegistrar is a nn.Module, it is only to store parameters.')

    def get_model(self, name, model_if_absent=None):
        # 4 cases: name in self.model_dict and model_if_absent is None         (OK)
        #          name in self.model_dict and model_if_absent is not None     (OK)
        #          name not in self.model_dict and model_if_absent is not None (OK)
        #          name not in self.model_dict and model_if_absent is None     (NOT OK)

        if name in self.model_dict:
            return self.model_dict[name]

        elif model_if_absent is not None:
            self.model_dict[name] = model_if_absent.to(self.device)
            return self.model_dict[name]

        else:
            raise ValueError(f'{name} was never initialized in this Registrar!')

    def get_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def get_all_but_name_match(self, names):
        if not isinstance(names, list) and not isinstance(names, tuple):
            names = [names]
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if all([name not in key for name in names]):
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def print_model_names(self):
        print(self.model_dict.keys())

    def save_models(self, curr_iter):
        # Create the model directiory if it's not present.
        save_path = os.path.join(self.model_dir,
                                 'model_registrar-%d.pt' % curr_iter)

        torch.save(self.model_dict, save_path)

    def load_models(self, iter_num):        
        save_path = os.path.join(self.model_dir,
                                 'model_registrar-%d.pt' % iter_num)
        self.load_model_from_file(save_path)

    def load_model_from_file(self, file_path, except_contains=()):
        print('\nLoading from ' + file_path)

        # Import error can happen here is trying to load checkpoint with old planner_cost object.
        # To resolve it, one can remove the planner_cost from the checkpoint file using cleanup_checkpoint.py
        # Alternatively, the old pred_metrics folder needs to be copied with environment/nuScenes_data/cost_functions.py 
        # Same can happend with `model` which reqires the original trajectron++ code to be in the path.
        # sys.path.append('./trajectron/trajectron')
        file_path = os.path.expanduser(file_path)
        new_model_dict = torch.load(file_path, map_location=self.device)
        
        # Selectively update parameters
        for k in new_model_dict:
            if any([(substr in k) for substr in except_contains]):
                print(f"Skipping {k}")
            else:
                self.model_dict[k] = new_model_dict[k]
            
            # self.model_dict = {k: v for k, v in self.model_dict.items() if substr not in k}
        print('Loaded!')
        print('')
