import dill
import numpy as np
import torch

from collections import OrderedDict
from enum import IntEnum
from typing import Dict, Optional, Union, Any, List, Set

from diffstack.utils.model_registrar import ModelRegistrar


class RunMode(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    INFER = 2


class DataFormat(object):
    def __init__(self, required_elements: Set[str]) -> None:
        self.required_elements = required_elements

    def satisfied_by(self, data_dict: Dict[str, Any]) -> bool:
        return all(x in data_dict for x in self.required_elements)

    def __iter__(self) -> str:
        for x in self.required_elements:
            yield x

    def for_run_mode(self, run_mode: RunMode):
        elements = []
        for k in self.required_elements:
            ksplit = k.split(":")
            if len(ksplit) == 1:
                elements.append(k)
            elif len(ksplit) == 2:
                if ksplit[1].upper() == run_mode.name:
                    elements.append(ksplit[0])
        return DataFormat(elements)


class Module(torch.nn.Module):
    """Abstract module in a differentiable stack.

    Inheriting classes need to implement:
    - input_format
    - output_format
    - train_step()
    - validate_step()
    - infer_step()

    - train/validate/infer methods.
    """

    @property
    def name(self) -> str:
        self.__class__.__name__

    @property
    def input_format(self) -> DataFormat:
        """Required input keys specified as a set of strings wrapped as a DataFormat.

        The naming convention `my_input:run_mode` is used to identify a key
        `my_input` that is only required for run mode `run_mode`.

        Example:
            return DataFormat(["rgb_image", "pointcloud", "label:train"])
        """
        return None

    @property
    def output_format(self) -> DataFormat:
        """Output keys specified as a set of strings wrapped as a DataFormat.

        The naming convention `my_output:run_mode` is used to identify a key
        `my_output` that is only provided for run mode `run_mode`.

        Example:
            return DataFormat(["prediction", "loss:train", "ml_prediction:infer"])
        """
        return None

    def __init__(
        self,
        model_registrar: Optional[ModelRegistrar] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        log_writer: Optional[Any] = None,
        device: Optional[str] = None,
        input_mappings: Dict[str, str] = {},
    ) -> None:
        """
        Args:
            model_registrar (ModelRegistrar): handles the registration of trainable parameters
            hyperparams (dict): config parameters
            log_writer (wandb.Run): `wandb.Run` object for logging or None.
            device (str): torch device
            input_mappings (dict): a remapping of input names of the format {target_name: source_name}.
                This is used when connecting multiple modules and we need to rename the outputs of the
                previous module to the inputs of this module.
        """
        super().__init__()
        self.model_registrar = model_registrar
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.input_mappings = input_mappings

        # Initialize epoch counter
        self.curr_iter = 0
        self.curr_epoch = 0

    def apply_input_mappings(
        self, inputs: Dict, run_mode: RunMode, allow_partial: bool = True
    ):
        mapped_inputs = {}
        for k in self.input_format.for_run_mode(run_mode):
            if k in self.input_mappings:
                if self.input_mappings[k] in inputs:
                    mapped_inputs[k] = inputs[self.input_mappings[k]]
                elif not allow_partial:
                    raise ValueError(
                        f"Key `{k}` is remapped `{k}`<--`{self.input_mappings[k]}` but "
                        + f"there is no key `{self.input_mappings[k]}` in inputs.\n  "
                        + f"inputs={list(inputs.keys())};\n  "
                        + f"input_mappings={self.input_mappings}"
                    )
            elif k in inputs:
                mapped_inputs[k] = inputs[k]
            elif "input." + k in inputs:
                mapped_inputs[k] = inputs["input." + k]
            elif not allow_partial:
                raise ValueError(
                    f"Key `{k}` is not found in inputs and input_mappings.\n  "
                    + f"inputs={list(inputs.keys())};\n  "
                    + f"input_mappings={self.input_mappings}"
                )
        return mapped_inputs

    # Optional functions for tracking training iteration/epoch and annealers.
    # This logic is inherited from Trajectron++.
    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def set_curr_epoch(self, curr_epoch):
        self.curr_epoch = curr_epoch

    def set_annealing_params(self):
        pass

    def step_annealers(self, node_type=None):
        pass

    def forward(self, inputs: Dict, **kwargs) -> Dict:
        return self.train_step(inputs, **kwargs)

    # Abstract methods that children classes should implement.
    def train_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.TRAIN, **kwargs)

    def validate_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.VALIDATE, **kwargs)

    def infer_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.INFER, **kwargs)

    def _run_forward(self, inputs: Dict, run_mode: RunMode, **kwargs) -> Dict:
        raise NotImplementedError

    def set_eval():
        pass

    def set_train():
        pass

    def reset(self):
        pass


class ModuleSequence(Module):
    """A sequence of stack modules defined by an ordered dict of {name: Module}.

    The output of component N will be fed to the input of component N+1.
    When the output names of module N does not match the input names of module N+1
    we can specify the name mapping with the `input_mappings` argument of module N+1.
    The outputs of all previous modules can be referenced by `modulename.outputname`.
    The overall inputs can be referenced by `input.inputname`.

    The output will the sequence will contain the outputs of the last component, as
    well as the outputs of all components in the `modulename.outputname` format.

    Example:
        We have two modules with the following inputs and outputs:
            MyPredictor: {'agent_history'} -> {'most_likely_pred'}
            MyPlanner: {'prediction', 'ego_state', 'goal'} -> {'plan_x'}

        We can sequence them in the followin way.
        stack = ModuleSequence(OrderedDict(
            pred=MyPredictor(),
            plan=MyPlanner(input_mappings={
                'prediction': 'pred.most_likely_pred'
                'ego_state': 'input.ego_state',
                'goal': 'input.ego_goal',
                })))

        input_dict = {'agent_history', 'ego_state', 'ego_goal'}
        output_dict = stack.train_step(input_dict)

        Now `output_dict.keys()` will contain [
            'input.agent_history',
            'input.ego_state',
            'input.ego_goal',
            'pred.most_likely_pred',
            'plan.plan_x',
            'plan_x'
        ]
    """

    @property
    def input_format(self) -> DataFormat:
        list(self.components.values())[0].input_format

    @property
    def output_format(self) -> DataFormat:
        list(self.components.values())[-1].output_format

    def __init__(
        self,
        components: OrderedDict,
        model_registrar,
        hyperparams,
        log_writer,
        device,
        input_mappings: Dict[str, str] = {},
    ) -> None:
        super().__init__(
            model_registrar,
            hyperparams,
            log_writer,
            device,
            input_mappings=input_mappings,
        )
        self.components = components

        if "input" in self.components:
            raise ValueError(
                'Name "input" is reserved for the overall input to the module sequence.'
            )

    def validate_interfaces(self, desired_input: DataFormat) -> bool:
        data_dict = {k: None for k in desired_input}
        for component in self.components.values():
            if component.input_format.satisfied_by(data_dict):
                data_dict.update({k: None for k in component.output_format})
            else:
                missing_inputs = [
                    x
                    for x in component.input_format.required_elements
                    if x not in data_dict
                ]
                raise ValueError(
                    f"Component '{component.name}' missing input(s): {missing_inputs}"
                )

        return data_dict

    def sequence_components(
        self,
        inputs: Dict,
        run_mode: RunMode,
        pre_module_cb: Optional[callable] = None,
        post_module_cb: Optional[callable] = None,
        **kwargs,
    ):
        """Sequence the ordered dict of components by connecting outputs to inputs."""
        all_outputs = {"input." + k: v for k, v in inputs.items()}
        next_inputs = {**all_outputs, **inputs, **kwargs}

        for comp_i, (name, component) in enumerate(self.components.items()):
            component: Module

            if pre_module_cb is not None:
                pre_module_cb(name, component)

            if component.input_format is None:
                inputs = next_inputs
            else:
                inputs = component.apply_input_mappings(next_inputs, run_mode)

            if run_mode == RunMode.TRAIN:
                output = component.train_step(inputs, **kwargs)
            elif run_mode == RunMode.VALIDATE:
                output = component.validate_step(inputs, **kwargs)
            elif run_mode == RunMode.INFER:
                output = component.infer_step(inputs, **kwargs)
            else:
                raise ValueError(f"Unknown mode {run_mode}")

            if post_module_cb is not None:
                post_module_cb(name, component)

            # Add to all_outputs
            all_outputs.update({f"{name}.{k}": v for k, v in output.items()})

            # Construct input from all_outputs and the previous outputs.
            next_inputs = {**all_outputs, **output}

        return next_inputs

    def dry_run(
        self,
        input_keys: List[str] = None,
        run_mode: Optional[RunMode] = None,
        check_output: bool = True,
        raise_error: bool = True,
    ) -> List[str]:
        """Checks that all inputs are defined in module sequence.

        We only verify input and output based on module.input_format and
        module.output_format. We do not check if the modules correctly
        define their respective input_format and output_format.

        Args:
            input_keys: list of input keys we will feed to the module. If None we
                will use `self.input_format`.
            run_mode: run mode. If not specified we will check for all possible run modes.
            check_output: check that `self.output_format` is satisfied by last component output.
            raise_error: will raise error for an issues if True
        Returns:
            list of found issues represented as strings
        Raises:
            ValueError if some inputs are not defined.
        """
        issues = []
        if run_mode is None:
            for run_mode in RunMode:
                issues.extend(
                    self.dry_run(input_keys, run_mode, check_output, raise_error=False)
                )
                if issues:
                    break

        else:
            if input_keys is None:
                input_keys = list(self.input_format)
            inputs = {k: None for k in input_keys}
            all_outputs = {"input." + k: v for k, v in inputs.items()}
            next_inputs = {**all_outputs, **inputs}

            component: Module
            for name, component in self.components.items():
                inputs = component.apply_input_mappings(
                    next_inputs, run_mode, allow_partial=True
                )
                input_format = component.input_format.for_run_mode(run_mode)
                if not input_format.satisfied_by(inputs):
                    issues.append(
                        f"Component {name}({run_mode.name}): \n  Required inputs: {list(input_format)}\n  Provided inputs: {list(inputs.keys())}"
                    )

                output = {
                    k: None for k in component.output_format.for_run_mode(run_mode)
                }
                # Add to all_outputs
                all_outputs.update({f"{name}.{k}": v for k, v in output.items()})
                # Construct input from all_outputs and the previous outputs.
                next_inputs = {**all_outputs, **output}

            if check_output:
                output_format = self.output_format.for_run_mode(run_mode)
                if not output_format.satisfied_by(next_inputs):
                    issues.append(
                        f"Sequence {self.name}({run_mode.name}): \n  Required outputs: {list(output_format)}\n  Provided outputs: {list(next_inputs.keys())}"
                    )

        if raise_error and issues:
            print("\n".join(issues))
            raise ValueError("\n".join(issues))

        return issues

    def set_curr_iter(self, curr_iter):
        super().set_curr_iter(curr_iter)
        for comp in self.components.values():
            comp.set_curr_iter(curr_iter)

    def set_curr_epoch(self, curr_epoch):
        super().set_curr_epoch(curr_epoch)
        for comp in self.components.values():
            comp.set_curr_epoch(curr_epoch)

    def set_annealing_params(self):
        super().set_annealing_params()
        for comp in self.components.values():
            comp.set_annealing_params()

    def step_annealers(self, node_type=None):
        super().step_annealers(node_type)
        for comp in self.components.values():
            comp.step_annealers(node_type)

    def train_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.TRAIN, **kwargs)

    def validate_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.VALIDATE, **kwargs)

    def infer_step(self, inputs: Dict, **kwargs) -> Dict:
        return self._run_forward(inputs, RunMode.INFER, **kwargs)

    def _run_forward(self, inputs: Dict, run_mode: RunMode, **kwargs) -> Dict:
        return self.sequence_components(inputs, run_mode=run_mode, **kwargs)

    def __getstate__(self):
        # Custom getstate for pickle that allows for lambda functions.
        import dill  # reimport in case not available in forked function

        return dill.dumps(self.__dict__)

    def __setstate__(self, state):
        # Custom setstate for pickle that allows for lambda functions.
        state = dill.loads(state)
        self.__dict__.update(state)

    def __str__(self):
        return f"Module sequence (device={self.device}) \n   " + "\n   ".join(
            self.components.keys()
        )

    def set_eval(self):
        for k, v in self.components.items():
            v.set_eval()

    def set_train(self):
        for k, v in self.components.items():
            v.set_train()

    def reset(self):
        for k, v in self.components.items():
            v.reset()
