"""
This file contains several utility functions used to define the main training loop. It
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import socket
import shutil
import pytorch_lightning as pl
from pytorch_lightning.loops.utilities import _reset_progress

def infinite_iter(data_loader):
    """
    Get an infinite generator
    Args:
        data_loader (DataLoader): data loader to iterate through

    """
    c_iter = iter(data_loader)
    while True:
        try:
            data = next(c_iter)
        except StopIteration:
            c_iter = iter(data_loader)
            data = next(c_iter)
        yield data


def get_exp_dir(exp_name, output_dir, save_checkpoints=True, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        exp_name (str): name of the experiment
        output_dir (str): output directory of the experiment
        save_checkpoints (bool): if save checkpoints
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.abspath(base_output_dir)
    base_output_dir = os.path.join(base_output_dir, exp_name)
    if os.path.exists(base_output_dir):
        # if not auto_remove_exp_dir:
        #     ans = input(
        #         "WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(
        #             base_output_dir
        #         )
        #     )
        # else:
        #     ans = "y"
        # if ans == "y":
        #     print("REMOVING")
        #     shutil.rmtree(base_output_dir)        
        if auto_remove_exp_dir:
            print(f"REMOVING {base_output_dir}")
            shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    # version the run
    existing_runs = [
        a
        for a in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, a))
    ]
    run_counts = [-1]
    for ep in existing_runs:
        m = ep.split("run")
        if len(m) == 2 and m[0] == "":
            run_counts.append(int(m[1]))
    version_str = "run{}".format(max(run_counts) + 1)

    # only make model directory if model saving is enabled
    ckpt_dir = None
    if save_checkpoints:
        ckpt_dir = os.path.join(base_output_dir, version_str, "checkpoints")
        os.makedirs(ckpt_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, version_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, version_str, "videos")
    os.makedirs(video_dir)
    return base_output_dir, log_dir, ckpt_dir, video_dir, version_str


def trajdata_auto_set_batch_size(trainer: "pl.Trainer", model: "pl.LightningModule",datamodule:"pl.LightningDataModule",bs_min = 2,bs_max = None,conservative_reduction=True) -> int:
    if bs_max == None:
        # power search to find the bs_max
        bs_trial = bs_min

        while True:
            
            datamodule.train_batch_size = bs_trial
            datamodule.val_batch_size = bs_trial
            try:
                trainer.fit(model=model,datamodule=datamodule)
                _reset_progress(trainer.fit_loop)
                print(f"batch size {bs_trial} succeeded, trying {bs_trial*2}")
                bs_min = bs_trial
                bs_trial *= 2
                
                
            except:
                print(f"batch size {bs_trial} failed, setting max batch size to {bs_trial}")
                bs_max = bs_trial
                break
            
            # the maximum batch size is dataset size divided by validation interval (there needs to be at least 1 validation per epoch)
            if bs_trial >=len(datamodule.train_dataset)/getattr(trainer,"val_check_interval",100):
                bs_max = int(len(datamodule.train_dataset)/getattr(trainer,"val_check_interval",100))-1
                break
    else:
        bs_max = min(bs_max,int(len(datamodule.train_dataset)/getattr(trainer,"val_check_interval",100))-1)
    # binary search to find the optimal batch size
    print(f" starting binary search with minimum batch size {bs_min}, maximum batch size {bs_max}")
    while bs_max - bs_min > 1:
        bs_trial = (bs_min + bs_max) // 2
        print(f"trying batch size {bs_trial}")
        datamodule.train_batch_size = bs_trial
        datamodule.val_batch_size = bs_trial
        try:
            trainer.fit(model=model,datamodule=datamodule)
            _reset_progress(trainer.fit_loop)
            print(f"batch size {bs_trial} succeeded")
            bs_min = bs_trial
        except:
            bs_max = bs_trial
            print(f"batch size {bs_trial} failed")
        if bs_max-bs_min<min([bs_min*0.1,5]):
            break
    print(f"Binary search terminated with optimal batch size {bs_min}")
    # return bs_min for safety
    if conservative_reduction:
        bs_min = max(bs_min-2,int(bs_min*0.8))
    return bs_min