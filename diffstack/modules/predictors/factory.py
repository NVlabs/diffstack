"""Factory methods for creating planner"""
from diffstack.configs.base import AlgoConfig

from diffstack.utils.utils import removeprefix

from diffstack.modules.predictors.kinematic_predictor import KinematicTreeModel
from diffstack.modules.predictors.tbsim_predictors import (
    AgentFormerTrafficModel,
)

from diffstack.modules.predictors.CTT import CTTTrafficModel


def predictor_factory(
    model_registrar,
    config: AlgoConfig,
    logger,
    device,
    input_mappings={},
    checkpoint=None,
):
    """
    A factory for creating predictor modules

    Args:
        config (AlgoConfig): an AlgoConfig object,

    Returns:
        predictor: predictor module
    """
    algo_name = config.name

    if algo_name == "kinematic":
        predictor = KinematicTreeModel(
            model_registrar, config, logger, device, input_mappings=input_mappings
        )

    elif algo_name in [
        "agentformer_multistage",
        "agentformer_singlestage",
        "agentformer",
    ]:
        predictor = AgentFormerTrafficModel(
            model_registrar, config, logger, device, input_mappings=input_mappings
        )
    elif algo_name == "CTT":
        predictor = CTTTrafficModel(
            model_registrar, config, logger, device, input_mappings=input_mappings
        )
    else:
        raise NotImplementedError(f"{algo_name} is not implemented")

    if checkpoint is not None:
        if "state_dict" in checkpoint:
            predictor_dict = {
                removeprefix(k, "components.predictor."): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("components.predictor.")
            }
        elif "model_dict" in checkpoint:
            predictor_dict = {
                "model." + k: v for k, v in checkpoint["model_dict"].items()
            }
            # Are we loading all model parameters?
            assert all([k in predictor_dict for k in predictor.state_dict().keys()])
        else:
            raise ValueError("Unknown checkpoint format")
        predictor.load_state_dict(predictor_dict)
    return predictor
