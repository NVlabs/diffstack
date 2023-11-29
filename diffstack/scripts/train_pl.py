import argparse
import sys
import os
import socket
import torch

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler

from diffstack.utils.log_utils import PrintLogger
from diffstack.utils.experiment_utils import get_checkpoint
import diffstack.utils.train_utils as TrainUtils
from diffstack.data.trajdata_datamodules import UnifiedDataModule
from diffstack.configs.registry import get_registered_experiment_config
from diffstack.utils.config_utils import (
    get_experiment_config_from_file,
    recursive_update_flat,
    translate_trajdata_cfg,
)
from diffstack.stacks.stack_factory import stack_factory
from pathlib import Path
import json
import multiprocessing as mp
from diffstack.configs.config import Dict

test = False


def main(cfg, auto_remove_exp_dir=False, debug=False, evaluate=False, **extra_kwargs):
    pl.seed_everything(cfg.seed)
    # Dataset
    trajdata_config = translate_trajdata_cfg(cfg)

    model = stack_factory(
        cfg=cfg,
    )
    # if test:
    #     model.set_eval()
    #     import pickle
    #     with open("homo_test_case.pkl","rb") as f:
    #         batch = pickle.load(f)
    #     from diffstack.modules.module import Module, DataFormat, RunMode
    #     with torch.no_grad():
    #         res = model.components["predictor"]._run_forward(dict(parsed_batch=batch),RunMode.VALIDATE)
    #         model.components["predictor"].log_pred_image(batch,res,501,"curated_result")

    datamodule = UnifiedDataModule(data_config=trajdata_config, train_config=cfg.train)

    datamodule.setup()

    if not evaluate:
        print("\n============= New Training Run with Config =============")
        print(cfg)
        print("")
        root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
            exp_name=cfg.name,
            output_dir=cfg.root_dir,
            save_checkpoints=cfg.train.save.enabled,
            auto_remove_exp_dir=auto_remove_exp_dir,
        )

        # Save experiment config to the training dir
        cfg.dump(os.path.join(root_dir, version_key, "config.json"))

        # if cfg.train.logging.terminal_output_to_txt and not debug:
        #     # log stdout and stderr to a text file
        #     logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        #     sys.stdout = logger
        #     sys.stderr = logger

        train_callbacks = []

        # Training Parallelism
        assert cfg.train.parallel_strategy in [
            "ddp",
            "ddp_spawn",
            # "ddp",  # TODO for ddp we need to look at NODE_RANK and disable logging in config
            None,
        ]  # TODO: look into other strategies
        if not cfg.devices.num_gpus > 1:
            # Override strategy when training on a single GPU
            with cfg.train.unlocked():
                cfg.train.parallel_strategy = None
        if cfg.train.parallel_strategy in ["ddp_spawn", "ddp"]:
            with cfg.train.training.unlocked():
                cfg.train.training.batch_size = int(
                    cfg.train.training.batch_size / cfg.devices.num_gpus
                )
            with cfg.train.validation.unlocked():
                cfg.train.validation.batch_size = int(
                    cfg.train.validation.batch_size / cfg.devices.num_gpus
                )

        # # Environment for close-loop evaluation
        # if cfg.train.rollout.enabled:
        #     # Run rollout at regular intervals
        #     rollout_callback = RolloutCallback(
        #         exp_config=cfg,
        #         every_n_steps=cfg.train.rollout.every_n_steps,
        #         warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
        #         verbose=True,
        #         save_video=cfg.train.rollout.save_video,
        #         video_dir=video_dir
        #     )
        #     train_callbacks.append(rollout_callback)

        # Model

        # Checkpointing
        if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
            assert (
                cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
            ), "checkpointing frequency needs to be greater than validation frequency"
            for metric_name, metric_key in model.checkpoint_monitor_keys.items():
                print(
                    "Monitoring metrics {} under alias {}".format(
                        metric_key, metric_name
                    )
                )
                ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="iter{step}_ep{epoch}_%s{%s:.2f}"
                    % (metric_name, metric_key),
                    # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                    auto_insert_metric_name=False,
                    save_top_k=cfg.train.save.best_k,  # save the best k models
                    monitor=metric_key,
                    mode="min",
                    every_n_train_steps=cfg.train.save.every_n_steps,
                    verbose=True,
                )
                train_callbacks.append(ckpt_valid_callback)

        if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
            assert (
                cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
            ), "checkpointing frequency needs to be greater than rollout frequency"
            ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor="rollout/metrics_ego_ADE",
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_rollout_callback)

        # a ckpt monitor to save at fixed interval
        ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            monitor=None,
            every_n_train_steps=10000,
            verbose=True,
        )
        train_callbacks.append(ckpt_fixed_callback)

        # def wandb_login(i,return_dict):
        #     apikey = os.environ["WANDB_APIKEY"]
        #     wandb.login(key=apikey,host="https://api.wandb.ai")
        #     logger = WandbLogger(
        #             name=cfg.name, project=cfg.train.logging.wandb_project_name,
        #         )
        #     return_dict[i] = logger
        # manager = mp.Manager()

        logger = None
        if debug:
            print("Debugging mode, suppress logging.")
        elif cfg.train.logging.log_tb:
            logger = TensorBoardLogger(
                save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
            )
            print("Tensorboard event will be saved at {}".format(logger.log_dir))
        elif cfg.train.logging.log_wandb:
            assert (
                "WANDB_APIKEY" in os.environ
            ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
            try:
                apikey = os.environ["WANDB_APIKEY"]
                wandb.login(key=apikey, host="https://api.wandb.ai")
                logger = WandbLogger(
                    name=cfg.name,
                    project=cfg.train.logging.wandb_project_name,
                )
            except:
                logger = None
            # return_dict = manager.dict()
            # p1 = mp.Process(target=wandb_login,args=(0,return_dict))
            # p1.start()
            # p1.join(timeout=30)
            # p1.terminate()
            # if 0 in return_dict:
            #     logger = return_dict[0]

        else:
            print("WARNING: not logging training stats")

        # Train
        kwargs = dict(
            default_root_dir=root_dir,
            # checkpointing
            enable_checkpointing=cfg.train.save.enabled,
            # logging
            logger=logger,
            # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
            log_every_n_steps=cfg.train.logging.log_every_n_steps,
            # training
            max_steps=cfg.train.training.num_steps,
            # validation
            val_check_interval=cfg.train.validation.every_n_steps,
            limit_val_batches=cfg.train.validation.num_steps_per_epoch,
            # all callbacks
            callbacks=train_callbacks,
            # device & distributed training setup
            # accelerator='cpu',
            # accelerator=('gpu' if cfg.devices.num_gpus > 0 else 'cpu'),
            devices=(cfg.devices.num_gpus if cfg.devices.num_gpus > 0 else None),
            # strategy=cfg.train.parallel_strategy if cfg.train.parallel_strategy is not None else DDPStrategy(find_unused_parameters=True),
            strategy="ddp_find_unused_parameters_true",
            accelerator="gpu",
            gradient_clip_val=cfg.train.gradient_clip_val,
            # detect_anomaly=True,
            # setting for overfit debugging
            # limit_val_batches=0,
            # overfit_batches=2
        )
        # if debug:
        #     profiler = PyTorchProfiler(
        #         output_filename=None,
        #         enabled=True,
        #         use_cuda=cfg.devices.num_gpus > 0,
        #         record_shapes=False,
        #         profile_memory=True,
        #         group_by_input_shapes=False,
        #         with_stack=False,
        #         use_kineto=False,
        #         use_cpu=cfg.devices.num_gpus == 0,
        #         emit_nvtx=False,
        #         export_to_chrome=False,
        #         path_to_export_trace=None,
        #         row_limit=200,
        #         sort_by_key=None,
        #         profiled_functions=None,
        #         local_rank=None,
        #     )
        #     kwargs["profiler"] = profiler
        #     kwargs["max_steps"] = extra_kwargs["profile_steps"]

        if cfg.train.get("amp", False):
            kwargs["precision"] = 16
            # kwargs["amp_backend"] = "apex"
            # kwargs["amp_level"] = "O2"
        # if cfg.train.get("auto_batch_size",False):
        #     kwargs["auto_scale_batch_size"] = "binsearch"

        if cfg.train.get("auto_batch_size", False):
            from diffstack.utils.train_utils import trajdata_auto_set_batch_size

            kwargs_tune = kwargs.copy()
            kwargs_tune["max_steps"] = 3
            trial_trainer = pl.Trainer(**kwargs_tune)
            trial_model = stack_factory(
                cfg=cfg,
            )
            bs_max = cfg.train.get("max_batch_size", None)
            batch_size = trajdata_auto_set_batch_size(
                trial_trainer,
                trial_model,
                datamodule,
                bs_min=cfg.train.training.batch_size,
                bs_max=bs_max,
            )
            # batch_size = trajdata_auto_set_batch_size(trial_trainer,trial_model,datamodule,bs_min = 50,bs_max = 58)
            datamodule.train_batch_size = batch_size
            datamodule.val_batch_size = batch_size
            del trial_trainer, trial_model
        if cfg.devices.num_gpus > 0:
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer = pl.Trainer(**kwargs)
        # Logging
        assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)

        if isinstance(logger, WandbLogger):
            # record the entire config on wandb
            if trainer.global_rank == 0:
                logger.experiment.config.update(cfg.to_dict())
                logger.watch(model=model)
        # kwargs_tune = kwargs.copy()
        # kwargs_tune["max_steps"] = 30
        # trial_trainer = pl.Trainer(**kwargs_tune)
        # trial_trainer.fit(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)

    else:
        kwargs = dict(
            devices=(1 if cfg.devices.num_gpus > 0 else None),
            # strategy=cfg.train.parallel_strategy if cfg.train.parallel_strategy is not None else DDPStrategy(find_unused_parameters=True),
            accelerator="auto",
        )
        trainer = pl.Trainer(**kwargs)
        # Evaluation

        model.set_eval()

        metrics = trainer.test(model, datamodule=datamodule)
        if len(metrics) == 1:
            flattened_metrics = metrics[0]
        else:
            flattened_metrics = dict()
            for k, v in metrics[0].items():
                flattened_metrics[k] = sum([m[k] for m in metrics]) / len(metrics)
        result_path = extra_kwargs.get(
            "eval_output_dir", os.path.join(os.getcwd(), "eval_result")
        )
        file_name = f"{cfg.registered_name}_{cfg.train.trajdata_test_source_root}_{cfg.train.trajdata_source_test}_eval_result.json"

        with open(os.path.join(result_path, file_name), "w") as f:
            json.dump(flattened_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

    parser.add_argument(
        "--on_ngc",
        action="store_true",
        help="whether running the script on ngc (this will change some behaviors like avoid writing into dataset)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    parser.add_argument(
        "--profile_steps", type=int, default=500, help="number of steps to run profiler"
    )
    # evaluation mode
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate mode, suppress training."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to the checkpoint to be evaluated",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=None,
        help="path to store evaluation result",
    )
    parser.add_argument(
        "--ckpt_root_dir", type=str, default=None, help="path of ngc checkpoint folder"
    )
    parser.add_argument("--ngc_job_id", type=str, default=None, help="ngc job id")
    parser.add_argument("--ckpt_key", type=str, default=None, help="ngc checkpoint key")
    parser.add_argument(
        "--test_data", type=str, default=None, help="trajdata_source_test in config"
    )
    parser.add_argument(
        "--test_data_root",
        type=str,
        default=None,
        help="trajdata_test_source_root in config",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=None, help="batch size of test run"
    )
    parser.add_argument(
        "--log_image_frequency", type=int, default=None, help="image logging frequency"
    )

    parser.add_argument(
        "--log_all_image",
        action="store_true",
        help="log images for all scenes instead of only the first one for every batch",
    )

    parser.add_argument(
        "--remove_parked",
        action="store_true",
        help="remove parked agents from the scene",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
    elif args.evaluate:
        ckpt_path = None
        if args.ckpt_path is not None:
            config_path = str(Path(args.ckpt_path).parents[1] / "config.json")
            ckpt_path = args.ckpt_path
        elif (
            args.ngc_job_id is not None
            and args.ckpt_key is not None
            and args.ckpt_root_dir is not None
        ):
            ckpt_path, config_path = get_checkpoint(
                ngc_job_id=args.ngc_job_id,
                ckpt_key=args.ckpt_key,
                ckpt_root_dir=args.ckpt_root_dir,
            )
        default_config = get_experiment_config_from_file(config_path)

        if default_config.stack.stack_type == "pred":
            default_config.stack.predictor.load_checkpoint = ckpt_path
        elif default_config.stack.stack_type == "plan":
            default_config.stack.planner.load_checkpoint = ckpt_path
        else:
            raise NotImplementedError

        if "test" not in default_config.train:
            default_config.train.test = Dict(
                enabled=True,
                batch_size=32,
                num_data_workers=6,
                every_n_steps=500,
                num_steps_per_epoch=50,
            )
        if args.remove_parked:
            default_config.env.remove_parked = True
        if args.test_data is not None:
            default_config.train.trajdata_source_test = args.test_data
        if args.test_data_root is not None:
            default_config.train.trajdata_test_source_root = args.test_data_root
        # modify dataset path to reduce loading time
        if default_config.train.trajdata_source_test is None:
            default_config.train.trajdata_source_test = (
                default_config.train.trajdata_source_valid
            )
        if default_config.train.trajdata_test_source_root is None:
            default_config.train.trajdata_test_source_root = (
                default_config.train.trajdata_val_source_root
            )

        default_config.train.trajdata_val_source_root = None
        default_config.train.trajdata_source_valid = (
            default_config.train.trajdata_source_test
        )
        default_config.train.trajdata_source_root = (
            default_config.train.trajdata_test_source_root
        )
        default_config.train.trajdata_source_train = (
            default_config.train.trajdata_source_test
        )
        if args.test_batch_size is not None:
            default_config.train.test["batch_size"] = args.test_batch_size
        if (
            "predictor" in default_config.stack
            and default_config.stack.predictor.name == "CTT"
        ):
            default_config.env.max_num_lanes = 64
            default_config.stack.predictor.decoder.decode_num_modes = 8
            default_config.stack.predictor.LR_sample_hack = False
            # default_config.env.remove_single_successor = False
        default_config.eval.results_dir = args.eval_output_dir
        if args.ngc_job_id is not None:
            default_config.registered_name = (
                default_config.registered_name + "_" + args.ngc_job_id
            )
        default_config.eval.log_image_frequency = args.log_image_frequency
        if args.log_all_image:
            default_config.eval.log_all_image = True
        else:
            default_config.eval.log_all_image = False

    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    if args.on_ngc:
        ngc_job_id = socket.gethostname()
        default_config.name = default_config.name + "_" + ngc_job_id

    default_config.train.on_ngc = args.on_ngc
    args_dict = vars(args)
    args_dict = {
        k: v
        for k, v in args_dict.items()
        if k
        not in ["name", "dataset_path", "output_dir", "wandb_project_name", "on_ngc"]
    }
    default_config, leftover = recursive_update_flat(default_config, args_dict)
    if len(leftover) > 0:
        Warning(f"Arguments {list(leftover.keys())} are not found in the config")

    if args.debug:
        # Test policy rollout
        default_config.train.rollout.every_n_steps = 10
        default_config.train.rollout.num_episodes = 1

    # make rollout evaluation config consistent with the rest of the config

    default_config.lock()  # Make config read-only
    main(
        default_config,
        auto_remove_exp_dir=args.remove_exp_dir,
        debug=args.debug,
        profile_steps=args.profile_steps,
        evaluate=args.evaluate,
        eval_output_dir=args.eval_output_dir,
    )
