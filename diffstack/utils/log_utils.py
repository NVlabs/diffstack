"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import sys
import os
import time
import pathlib
import json
import wandb
from omegaconf import OmegaConf, DictConfig


def prepare_logging(cfg: DictConfig, use_logging=True, use_wandb=None, verbose=True, wandb_init_retries=10):
    # Logging
    log_writer = None
    model_dir = None
    if not use_logging:
        return log_writer, model_dir
    if use_wandb is None:
        use_wandb = cfg.run.wandb_logging

    # Create the log and model directory if they're not present.
    model_dir = os.path.join(cfg.run.log_dir,
                                cfg.run.log_tag + time.strftime('-%d_%b_%Y_%H_%M_%S', time.localtime()))
    cfg.logdir = model_dir

    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Save config to model directory
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
        json.dump(cfg_dict, conf_json)

    # wandb.tensorboard.patch(root_logdir=model_dir, pytorch=True)

    # WandB init. Put it in a loop because it can fail on ngc.
    if use_wandb:
        log_writer = init_wandb(cfg, wandb_init_retries=wandb_init_retries)

    if verbose:
        print (f"Log path: {model_dir}")

    return log_writer, model_dir


def init_wandb(cfg, wandb_init_retries=10):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for _ in range(wandb_init_retries):
        try:
            log_writer = wandb.init(
                    project=cfg.run.wandb_project,
                    name=f"{cfg.run['log_tag']}",
                    config=cfg_dict, 
                    mode="offline" if cfg.run["interactive"] else "online",  # sync_tensorboard=True, 
                    settings=wandb.Settings(start_method="thread"),
                )
        except:
            continue
        break
    else:
        raise ValueError("Could not connect to wandb")
    return log_writer


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
