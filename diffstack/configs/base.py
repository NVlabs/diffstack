from diffstack.configs.config import Dict
from copy import deepcopy
from diffstack.configs.eval_config import EvaluationConfig


class TrainConfig(Dict):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "diffstack"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100

        ## save config - if and when to save model checkpoints ##
        self.save.enabled = True  # whether model saving should be enabled or disabled
        self.save.every_n_steps = 100  # save model every n epochs
        self.save.best_k = 5
        self.save.save_best_rollout = False
        self.save.save_best_validation = True

        ## evaluation rollout config ##
        self.rollout.save_video = True
        self.rollout.enabled = False  # enable evaluation rollouts
        self.rollout.every_n_steps = 1000  # do rollouts every @rate epochs
        self.rollout.warm_start_n_steps = (
            1  # number of steps to wait before starting rollouts
        )

        ## training config
        self.training.batch_size = 100
        self.training.num_steps = 200000
        self.training.num_data_workers = 0

        ## validation config
        self.validation.enabled = True
        self.validation.batch_size = 100
        self.validation.num_data_workers = 0
        self.validation.every_n_steps = 1000
        self.validation.num_steps_per_epoch = 100

        ## Training parallelism (e.g., multi-GPU)
        self.parallel_strategy = "ddp"

        self.rebuild_cache = False

        self.on_ngc = False

        # AMP
        self.amp = False

        # auto batch size
        self.auto_batch_size = False
        self.max_batch_size = 1000
        # graidient clipping
        self.gradient_clip_val = 0.5


class EnvConfig(Dict):
    def __init__(self):
        super(EnvConfig, self).__init__()
        self.name = "my_env"


class AlgoConfig(Dict):
    def __init__(self):
        super(AlgoConfig, self).__init__()
        self.name = "my_algo"


class ExperimentConfig(Dict):
    def __init__(
        self,
        train_config: TrainConfig,
        env_config: EnvConfig,
        module_configs: dict,
        eval_config: EvaluationConfig = None,
        registered_name: str = None,
        stack_type: str = None,
        name: str = None,
        root_dir=None,
        seed=None,
        devices=None,
    ):
        """

        Args:
            train_config (TrainConfig): training config
            env_config (EnvConfig): environment config
            module_configs dict(AlgoConfig): algorithm configs for all modules
            registered_name (str): name of the experiment config object in the global config registry
        """
        super(ExperimentConfig, self).__init__()
        self.registered_name = registered_name

        self.train = train_config
        self.env = env_config
        self.stack = module_configs
        self.stack.stack_type = stack_type
        self.eval = EvaluationConfig() if eval_config is None else eval_config

        # Write all results to this directory. A new folder with the timestamp will be created
        # in this directory, and it will contain three subfolders - "log", "models", and "videos".
        # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
        # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
        # videos.
        self.name = (
            "test"  # name of the experiment (creates a subdirectory under root_dir)
        )
        stack_name = ""
        for key, config in self.stack.items():
            if isinstance(config, dict):
                stack_name += key + "_" + config["name"]

        self.root_dir = (
            "{}_trained_models/".format(stack_name) if root_dir is None else root_dir
        )
        self.seed = (
            1 if seed is None else seed
        )  # seed for everything (for reproducibility)

        self.devices = (
            Dict(num_gpus=1) if devices is None else devices
        )  # Set to 0 to use CPU

    def clone(self):
        return self.__class__(
            train_config=deepcopy(self.train),
            env_config=deepcopy(self.env),
            module_configs=deepcopy(self.stack),
            eval_config=deepcopy(self.eval),
            registered_name=self.registered_name,
            stack_type=self.stack.stack_type,
            name=self.name,
            root_dir=self.root_dir,
            seed=self.seed,
            devices=self.devices,
        )
