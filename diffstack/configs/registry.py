"""A global registry for looking up named experiment configs"""
from diffstack.configs.base import ExperimentConfig
from diffstack.configs.config import Dict


from diffstack.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig

from diffstack.configs.algo_config import (
    AgentFormerConfig,
    CTTConfig,
)


EXP_CONFIG_REGISTRY = dict()


EXP_CONFIG_REGISTRY["AFPredStack"] = ExperimentConfig(
    train_config=TrajdataTrainConfig(),
    env_config=TrajdataEnvConfig(),
    module_configs=Dict(predictor=AgentFormerConfig()),
    registered_name="AFPredStack",
    stack_type="pred",
)


EXP_CONFIG_REGISTRY["CTTPredStack"] = ExperimentConfig(
    train_config=TrajdataTrainConfig(),
    env_config=TrajdataEnvConfig(),
    module_configs=Dict(predictor=CTTConfig()),
    registered_name="CTTPredStack",
    stack_type="pred",
)


def get_registered_experiment_config(registered_name):
    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()
