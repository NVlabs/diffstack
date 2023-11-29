"""Factory methods for creating models"""
from diffstack.modules.module import ModuleSequence
from diffstack.modules.predictors.factory import (
    predictor_factory,
)
from diffstack.configs.config import Dict
import torch
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List

from diffstack.stacks.pred_stack import PredStack
from omegaconf import DictConfig, OmegaConf, open_dict


def get_checkpoint_dict(stack_cfg: DictConfig, module_names: List, device: str):
    checkpoint_dict = defaultdict(lambda: None)
    for module_name in module_names:
        if (
            "load_checkpoint" in stack_cfg[module_name]
            and stack_cfg[module_name].load_checkpoint
        ):
            ckpt_path = Path(stack_cfg[module_name].load_checkpoint).expanduser()
            checkpoint_dict[module_name] = torch.load(ckpt_path, map_location=device)
    return checkpoint_dict


def stack_factory(cfg: DictConfig, model_registrar=None, log_writer=None, device=None):
    """
    A factory for creating training stacks

    Args:
        cfg (ExperimentConfig): an ExperimentConfig object,
    Returns:
        stack: pl.LightningModule
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.stack.stack_type == "pred":
        predictor_config = cfg.stack["predictor"]

        checkpoint_dict = get_checkpoint_dict(cfg.stack, ["predictor"], device)
        if "env" in cfg:
            if isinstance(predictor_config, Dict):
                predictor_config.unlock()
                predictor_config.env = cfg.env
                predictor_config.lock()
            elif isinstance(predictor_config, DictConfig):
                OmegaConf.set_struct(predictor_config, True)
                with open_dict(predictor_config):
                    predictor_config.env = cfg.env
        predictor = predictor_factory(
            model_registrar,
            predictor_config,
            log_writer,
            device,
            checkpoint=checkpoint_dict["predictor"],
        )

        modules = ModuleSequence(
            OrderedDict(predictor=predictor),
            model_registrar,
            predictor_config,
            log_writer,
            device,
        )
        if (
            "monitor_key" in predictor_config
            and predictor_config["monitor_key"] is not None
        ):
            avstack = PredStack(
                modules, cfg, monitor_key=predictor_config["monitor_key"]
            )
        else:
            avstack = PredStack(
                modules, cfg, monitor_key=predictor.checkpoint_monitor_keys
            )
        return avstack

    else:
        raise NotImplementedError("The type of stack structure is not recognized!")
