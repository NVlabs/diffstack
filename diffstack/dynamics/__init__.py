from typing import Union

from diffstack.dynamics.single_integrator import SingleIntegrator
from diffstack.dynamics.unicycle import Unicycle
from diffstack.dynamics.bicycle import Bicycle
from diffstack.dynamics.double_integrator import DoubleIntegrator
from diffstack.dynamics.base import Dynamics, DynType
from diffstack.dynamics.unicycle import Unicycle


def get_dynamics_model(dyn_type: Union[str, DynType]):
    if dyn_type in ["Unicycle", DynType.UNICYCLE]:
        return Unicycle
    elif dyn_type == ["SingleIntegrator", DynType.SI]:
        return SingleIntegrator
    elif dyn_type == ["DoubleIntegrator", DynType.DI]:
        return DoubleIntegrator
    else:
        raise NotImplementedError(
            "Dynamics model {} is not implemented".format(dyn_type)
        )
