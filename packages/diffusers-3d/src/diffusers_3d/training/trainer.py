from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.loaders import PeftAdapterMixin
from diffusers.optimization import get_scheduler
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from ..data import Object3DDataset
from .exceptions import (
    TrainableParameterError,
    TrainingCheckpointError,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingPolicyError,
    TrainingTargetError,
)
from .manifest import TrainingManifest3D
from .recipe import TRAINING_ADAPTER_NAME, TrainingRecipe3D
from .registry import _TRAINING_RECIPE_REGISTRY, TrainingRecipeRegistry
from .types import (
    ComponentPolicy,
    FineTuneStrategy3D,
    FullFineTune,
    LoRAFineTune,
    TrainingConfig3D,
    TrainingStep3DOutput,
    TrainingSummary3D,
)

ACCELERATOR_STATE_DIRECTORY = "accelerator_state"
TRAINER_STATE_NAME = "diffusers_3d_trainer_state.json"
TRAINER_STATE_SCHEMA_VERSION = 1


def _resolve_path(root: object, path: str) -> object:
    value = root
    if not path:
        return value
    for part in path.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as error:
            raise TrainingTargetError(f"Target has no exact component path {path!r}") from error
    return value


def _replace_path(root: object, path: str, value: object) -> object:
    if not path:
        return value
    parent_path, _, attribute = path.rpartition(".")
    parent = _resolve_path(root, parent_path)
    setattr(parent, attribute, value)
    return root


def _iter_nested_modules(value: object, prefix: str) -> Iterator[tuple[str, nn.Module]]:
    if isinstance(value, nn.Module):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            if isinstance(name, str):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from _iter_nested_modules(item, child_prefix)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_nested_modules(item, child_prefix)


def _reachable_named_parameters(target: object) -> dict[int, tuple[nn.Parameter, tuple[str, ...]]]:
    modules: list[tuple[str, nn.Module]]
    if isinstance(target, nn.Module):
        modules = [("", target)]
    else:
        modules = []
        for attribute, value in vars(target).items():
            for path, module in _iter_nested_modules(value, attribute):
                modules.append((path, module))

    parameters: dict[int, tuple[nn.Parameter, set[str]]] = {}
    for prefix, module in modules:
        for local_name, parameter in module.named_parameters(recurse=True, remove_duplicate=False):
            name = f"{prefix}.{local_name}" if prefix and local_name else prefix or local_name
            parameter_id = id(parameter)
            if parameter_id not in parameters:
                parameters[parameter_id] = (parameter, {name})
            else:
                parameters[parameter_id][1].add(name)
    return {parameter_id: (parameter, tuple(sorted(names))) for parameter_id, (parameter, names) in parameters.items()}


class Object3DTrainer:
    """Strict trainer whose only executable objective is a reviewed recipe instance."""

    _registry: ClassVar[TrainingRecipeRegistry] = _TRAINING_RECIPE_REGISTRY

    def __init__(
        self,
        recipe: TrainingRecipe3D,
        dataset: Object3DDataset[object],
        strategy: FineTuneStrategy3D,
        config: TrainingConfig3D,
    ) -> None:
        if not isinstance(recipe, TrainingRecipe3D):
            raise TypeError("recipe must be a TrainingRecipe3D instance")
        if type(strategy) not in (LoRAFineTune, FullFineTune):
            raise TypeError("strategy must be an exact LoRAFineTune or FullFineTune")
        if type(config) is not TrainingConfig3D:
            raise TypeError("config must be an exact TrainingConfig3D")
        self.recipe = recipe
        self.dataset = dataset
        self.strategy = strategy
        self.config = config
        self._prepared = False
        self._accelerator: Accelerator | None = None
        self._optimizer: Optimizer | None = None
        self._lr_scheduler: Any = None
        self._dataloader: DataLoader | None = None
        self._components: Mapping[str, nn.Module] = MappingProxyType({})
        self._frozen_components: Mapping[str, nn.Module] = MappingProxyType({})
        self._selected_policies: Mapping[str, ComponentPolicy] = MappingProxyType({})
        self._trainable_parameters: tuple[nn.Parameter, ...] = ()
        self._trainable_parameter_names: tuple[str, ...] = ()
        self._manifest: TrainingManifest3D | None = None
        self._micro_steps = 0
        self._optimizer_steps = 0
        self._dataloader_generator: torch.Generator | None = None
        self._dataloader_sampler: DistributedSampler | None = None
        self._dataloader_iterator: Iterator[object] | None = None
        self._dataloader_epoch = 0
        self._dataloader_position = 0
        self._dataloader_epoch_generator_state: torch.Tensor | None = None

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def accelerator(self) -> Accelerator:
        if self._accelerator is None:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._accelerator

    @property
    def optimizer(self) -> Optimizer:
        if self._optimizer is None:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._optimizer

    @property
    def lr_scheduler(self) -> Any:
        if self._lr_scheduler is None:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._lr_scheduler

    @property
    def dataloader(self) -> DataLoader:
        if self._dataloader is None:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._dataloader

    @property
    def components(self) -> Mapping[str, nn.Module]:
        if not self._prepared:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._components

    @property
    def frozen_components(self) -> Mapping[str, nn.Module]:
        if not self._prepared:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._frozen_components

    @property
    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        if not self._prepared:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._trainable_parameters

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        if not self._prepared:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._trainable_parameter_names

    @property
    def manifest(self) -> TrainingManifest3D:
        if self._manifest is None:
            raise TrainingConfigurationError("trainer is not prepared")
        return self._manifest

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    @property
    def micro_steps(self) -> int:
        return self._micro_steps

    def prepare(self) -> Object3DTrainer:
        if self._prepared:
            return self

        # Registration and exact target identity are always the first gate.
        registration = self._registry.validate(self.recipe)
        target = self.recipe.target

        policies = {policy.key: policy for policy in registration.component_policies}
        unknown_keys = set(self.strategy.components).difference(policies)
        if unknown_keys:
            raise TrainingPolicyError(f"Unknown recipe component keys: {', '.join(sorted(unknown_keys))}")

        selected_components: dict[str, nn.Module] = {}
        selected_policies: dict[str, ComponentPolicy] = {}
        for key in self.strategy.components:
            policy = policies[key]
            if self.strategy.kind not in policy.supported_strategies:
                raise TrainingPolicyError(f"Component {key!r} does not support {self.strategy.kind.value} fine-tuning")
            component = _resolve_path(target, policy.component_path)
            if type(component) not in policy.expected_types:
                expected = ", ".join(
                    f"{expected_type.__module__}.{expected_type.__qualname__}"
                    for expected_type in policy.expected_types
                )
                raise TrainingTargetError(
                    f"Component {key!r} at {policy.component_path!r} must have exact type ({expected}); "
                    f"got {type(component).__module__}.{type(component).__qualname__}"
                )
            selected_components[key] = component
            selected_policies[key] = policy

            if type(self.strategy) is LoRAFineTune:
                if not isinstance(component, PeftAdapterMixin):
                    raise TrainingPolicyError(
                        f"LoRA component {key!r} must expose the public PeftAdapterMixin.add_adapter API"
                    )
                named_modules = dict(component.named_modules())
                missing_targets = set(policy.lora_target_modules).difference(named_modules)
                if missing_targets:
                    raise TrainingTargetError(
                        f"Component {key!r} is missing exact recipe-owned LoRA modules: "
                        f"{', '.join(sorted(missing_targets))}"
                    )

        frozen_components: dict[str, nn.Module] = {}
        for policy in registration.frozen_component_policies:
            component = _resolve_path(target, policy.component_path)
            if type(component) not in policy.expected_types:
                expected = ", ".join(
                    f"{expected_type.__module__}.{expected_type.__qualname__}"
                    for expected_type in policy.expected_types
                )
                raise TrainingTargetError(
                    f"Frozen component at {policy.component_path!r} must have exact type ({expected}); "
                    f"got {type(component).__module__}.{type(component).__qualname__}"
                )
            frozen_components[policy.component_path] = component

        if len({id(component) for component in selected_components.values()}) != len(selected_components):
            raise TrainingTargetError("Selected component keys must resolve to distinct exact component objects")
        all_managed_components = [*selected_components.values(), *frozen_components.values()]
        if len({id(component) for component in all_managed_components}) != len(all_managed_components):
            raise TrainingTargetError("Trainable and frozen component policies must resolve to distinct objects")
        self.recipe.validate_target()
        if not isinstance(self.dataset, Object3DDataset):
            raise TrainingConfigurationError("dataset must implement the runtime-checkable Object3DDataset protocol")
        try:
            dataset_length = len(self.dataset)
        except Exception as error:
            raise TrainingConfigurationError("dataset length could not be read") from error
        if not isinstance(dataset_length, int) or isinstance(dataset_length, bool) or dataset_length <= 0:
            raise TrainingConfigurationError("dataset must contain at least one example")

        original_target = target
        original_components = dict(selected_components)
        original_frozen_modes = {path: component.training for path, component in frozen_components.items()}
        original_frozen_devices = {}
        for path, component in frozen_components.items():
            tensors = tuple(component.parameters()) + tuple(component.buffers())
            original_frozen_devices[path] = tensors[0].device if tensors else None
        before_parameters = _reachable_named_parameters(target)
        for key, component in {**selected_components, **frozen_components}.items():
            component_parameter_ids = {
                id(parameter)
                for _, parameter in nn.Module.named_parameters(
                    component,
                    recurse=True,
                    remove_duplicate=False,
                )
            }
            if not component_parameter_ids.issubset(before_parameters):
                raise TrainingTargetError(f"Component {key!r} is not reachable from the exact recipe target")
        requires_grad_snapshot = {
            parameter_id: parameter.requires_grad for parameter_id, (parameter, _) in before_parameters.items()
        }
        injected_components: list[nn.Module] = []
        replaced_paths: list[tuple[str, nn.Module]] = []

        try:
            for parameter, _ in before_parameters.values():
                parameter.requires_grad_(False)

            expected_parameter_ids: set[int] = set()
            if type(self.strategy) is FullFineTune:
                for key in self.strategy.components:
                    component = selected_components[key]
                    policy = selected_policies[key]
                    local_parameters = dict(component.named_parameters(recurse=True))
                    if policy.full_parameter_names is None:
                        approved_names = tuple(local_parameters)
                    else:
                        missing_names = set(policy.full_parameter_names).difference(local_parameters)
                        if missing_names:
                            raise TrainingTargetError(
                                f"Component {key!r} is missing exact recipe-owned parameters: "
                                f"{', '.join(sorted(missing_names))}"
                            )
                        approved_names = policy.full_parameter_names
                    for name in approved_names:
                        parameter = local_parameters[name]
                        parameter.requires_grad_(True)
                        expected_parameter_ids.add(id(parameter))
            else:
                try:
                    from peft import LoraConfig
                except ImportError as error:
                    raise TrainingDependencyError(
                        "LoRA fine-tuning requires the optional 'peft' dependency"
                    ) from error

                for key in self.strategy.components:
                    component = selected_components[key]
                    policy = selected_policies[key]
                    adapter_config = LoraConfig(
                        r=self.strategy.rank,
                        lora_alpha=self.strategy.alpha,
                        lora_dropout=self.strategy.dropout,
                        bias="none",
                        target_modules=list(policy.lora_target_modules),
                    )
                    component.add_adapter(adapter_config, adapter_name=TRAINING_ADAPTER_NAME)
                    injected_components.append(component)

                after_injection = _reachable_named_parameters(target)
                expected_parameter_ids = set(after_injection).difference(before_parameters)
                if expected_parameter_ids:
                    allowed_prefixes = []
                    for key in self.strategy.components:
                        policy = selected_policies[key]
                        component_prefix = f"{policy.component_path}." if policy.component_path else ""
                        allowed_prefixes.extend(
                            f"{component_prefix}{target_module}." for target_module in policy.lora_target_modules
                        )
                    for parameter_id in expected_parameter_ids:
                        names = after_injection[parameter_id][1]
                        if not any(
                            name.startswith(tuple(allowed_prefixes))
                            and TRAINING_ADAPTER_NAME in name
                            and (".lora_A." in name or ".lora_B." in name)
                            for name in names
                        ):
                            raise TrainableParameterError(f"LoRA injected an unapproved parameter: {', '.join(names)}")

            audited_parameters = _reachable_named_parameters(target)
            actual_parameter_ids = {
                parameter_id for parameter_id, (parameter, _) in audited_parameters.items() if parameter.requires_grad
            }
            if not expected_parameter_ids:
                raise TrainableParameterError("The approved strategy produced zero trainable parameters")
            if actual_parameter_ids != expected_parameter_ids:
                unexpected = actual_parameter_ids.difference(expected_parameter_ids)
                missing = expected_parameter_ids.difference(actual_parameter_ids)
                details = []
                if unexpected:
                    details.append(
                        "unexpected="
                        + ",".join(name for parameter_id in unexpected for name in audited_parameters[parameter_id][1])
                    )
                if missing:
                    details.append(
                        "missing="
                        + ",".join(
                            name
                            for parameter_id in missing
                            for name in audited_parameters.get(parameter_id, (None, ("<removed>",)))[1]
                        )
                    )
                raise TrainableParameterError(
                    "Actual trainable parameters do not exactly match the recipe policy"
                    + (f": {'; '.join(details)}" if details else "")
                )

            trainable_parameter_names = tuple(
                sorted(name for parameter_id in expected_parameter_ids for name in audited_parameters[parameter_id][1])
            )
            trainable_parameters = tuple(
                audited_parameters[parameter_id][0]
                for parameter_id in sorted(
                    expected_parameter_ids,
                    key=lambda item: audited_parameters[item][1][0],
                )
            )
            optimizer = AdamW(
                trainable_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_epsilon,
            )
            optimizer_parameter_ids = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            if optimizer_parameter_ids != expected_parameter_ids:
                raise TrainableParameterError("Optimizer parameters do not exactly match the audited parameter set")

            set_seed(self.config.seed)
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                cpu=self.config.cpu,
            )
            generator = torch.Generator()
            generator.manual_seed(self.config.seed)
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=self.config.shuffle,
                seed=self.config.seed,
            )

            def collate_examples(examples: Sequence[object]) -> object:
                if any(type(example) is not registration.example_type for example in examples):
                    raise TrainingConfigurationError(
                        f"dataset items must be exact {registration.example_type.__name__} values"
                    )
                for example in examples:
                    example.validate()
                batch = self.recipe.collate(tuple(examples))
                if type(batch) is not registration.batch_type:
                    raise TrainingConfigurationError(
                        f"Recipe collate must return exact {registration.batch_type.__name__}, "
                        f"got {type(batch).__name__}"
                    )
                validate = getattr(batch, "validate", None)
                if not callable(validate):
                    raise TrainingConfigurationError("the exact recipe batch type must expose validate()")
                validate()
                return batch

            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.train_batch_size,
                sampler=sampler,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=collate_examples,
                generator=generator,
            )
            lr_scheduler = get_scheduler(
                self.config.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=self.config.max_train_steps,
            )
            for component in frozen_components.values():
                component.to(accelerator.device)
                component.requires_grad_(False)
                component.eval()

            unique_components: list[nn.Module] = []
            component_indices: dict[str, int] = {}
            component_id_to_index: dict[int, int] = {}
            for key in self.strategy.components:
                component = selected_components[key]
                if id(component) not in component_id_to_index:
                    component_id_to_index[id(component)] = len(unique_components)
                    unique_components.append(component)
                component_indices[key] = component_id_to_index[id(component)]

            prepared = accelerator.prepare(*unique_components, optimizer, lr_scheduler)
            prepared_values = prepared if isinstance(prepared, tuple) else (prepared,)
            wrapped_components = prepared_values[: len(unique_components)]
            optimizer = prepared_values[-2]
            lr_scheduler = prepared_values[-1]

            for key in self.strategy.components:
                policy = selected_policies[key]
                wrapped = wrapped_components[component_indices[key]]
                if wrapped is not selected_components[key]:
                    replaced_paths.append((policy.component_path, selected_components[key]))
                    target = _replace_path(target, policy.component_path, wrapped)
                selected_components[key] = wrapped
            if target is not original_target:
                self.recipe._target = target

            self._accelerator = accelerator
            self._optimizer = optimizer
            self._lr_scheduler = lr_scheduler
            self._dataloader = dataloader
            self._components = MappingProxyType(dict(selected_components))
            self._frozen_components = MappingProxyType(dict(frozen_components))
            self._selected_policies = MappingProxyType(dict(selected_policies))
            self._trainable_parameters = trainable_parameters
            self._trainable_parameter_names = trainable_parameter_names
            self._manifest = TrainingManifest3D.create(
                target_type=registration.target_type,
                example_type=registration.example_type,
                family_id=registration.family_id,
                recipe_id=registration.recipe_id,
                recipe_version=registration.recipe_version,
                strategy=self.strategy,
                base_model=self.config.base_model,
                revision=self.config.revision,
                trainable_parameter_names=trainable_parameter_names,
                objective_config=self.recipe.objective_config(),
                training_config={
                    **self.config.resume_config(),
                    "distributed_type": accelerator.distributed_type.value,
                    "num_processes": accelerator.num_processes,
                },
            )
            self._dataloader_generator = generator
            self._dataloader_sampler = sampler
            self._dataloader_epoch_generator_state = generator.get_state().clone()
            self._optimizer.zero_grad(set_to_none=True)
            for component in self._components.values():
                component.train()
            self._prepared = True
            return self
        except Exception:
            if target is not original_target:
                self.recipe._target = original_target
            else:
                for path, component in reversed(replaced_paths):
                    _replace_path(original_target, path, component)
            for component in reversed(injected_components):
                try:
                    component.delete_adapters(TRAINING_ADAPTER_NAME)
                except Exception:
                    pass
            restored_parameters = _reachable_named_parameters(original_target)
            for parameter_id, requires_grad in requires_grad_snapshot.items():
                if parameter_id in restored_parameters:
                    restored_parameters[parameter_id][0].requires_grad_(requires_grad)
            for parameter_id, (parameter, _) in restored_parameters.items():
                if parameter_id not in requires_grad_snapshot:
                    parameter.requires_grad_(False)
            self._components = MappingProxyType(original_components)
            for path, component in frozen_components.items():
                original_device = original_frozen_devices[path]
                if original_device is not None:
                    component.to(original_device)
                component.train(original_frozen_modes[path])
            self._frozen_components = MappingProxyType({})
            self._selected_policies = MappingProxyType({})
            raise

    def _move_batch(self, batch: object) -> object:
        if type(batch) is not type(self.recipe).batch_type:
            raise TrainingConfigurationError(
                f"train_step requires exact {type(self.recipe).batch_type.__name__}, got {type(batch).__name__}"
            )
        validate = getattr(batch, "validate", None)
        move = getattr(batch, "to", None)
        if not callable(validate) or not callable(move):
            raise TrainingConfigurationError("the exact recipe batch type must expose validate() and to()")
        validate()
        moved = move(device=self.accelerator.device)
        if type(moved) is not type(batch):
            raise TrainingConfigurationError("batch.to() must preserve the exact batch type")
        moved.validate()
        return moved

    def train_step(self, batch: object) -> TrainingStep3DOutput:
        if not self._prepared:
            self.prepare()
        moved_batch = self._move_batch(batch)
        models = tuple(dict.fromkeys(self._components.values()))
        accumulation_context = self.accelerator.accumulate(*models) if models else nullcontext()
        with accumulation_context:
            output = self.recipe.compute_loss(moved_batch)
            if type(output) is not TrainingStep3DOutput:
                raise TrainingConfigurationError("recipe compute_loss must return exact TrainingStep3DOutput")
            if not output.loss.requires_grad:
                raise TrainingConfigurationError("training loss must require gradients")
            self.accelerator.backward(output.loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self._trainable_parameters, self.config.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        self._micro_steps += 1
        if self.accelerator.sync_gradients:
            self._optimizer_steps += 1
        detached_metrics = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in output.metrics.items()
        }
        return TrainingStep3DOutput(loss=output.loss.detach(), metrics=detached_metrics)

    def _start_dataloader_epoch(
        self,
        *,
        skip_batches: int = 0,
        generator_state: torch.Tensor | None = None,
    ) -> None:
        if self._dataloader_generator is None:
            raise TrainingConfigurationError("trainer dataloader generator is unavailable")
        if self._dataloader_sampler is None:
            raise TrainingConfigurationError("trainer distributed sampler is unavailable")
        if generator_state is not None:
            self._dataloader_generator.set_state(generator_state)
        self._dataloader_sampler.set_epoch(self._dataloader_epoch)
        self._dataloader_epoch_generator_state = self._dataloader_generator.get_state().clone()
        dataloader = self.dataloader
        if skip_batches:
            dataloader = self.accelerator.skip_first_batches(dataloader, skip_batches)
        self._dataloader_iterator = iter(dataloader)

    def _next_batch(self) -> object:
        if self._dataloader_iterator is None:
            self._start_dataloader_epoch()
        assert self._dataloader_iterator is not None
        try:
            batch = next(self._dataloader_iterator)
        except StopIteration:
            self._dataloader_epoch += 1
            self._dataloader_position = 0
            self._start_dataloader_epoch()
            assert self._dataloader_iterator is not None
            batch = next(self._dataloader_iterator)
        self._dataloader_position += 1
        return batch

    def train(self, max_optimizer_steps: int | None = None) -> TrainingSummary3D:
        if not self._prepared:
            self.prepare()
        if max_optimizer_steps is not None and (
            not isinstance(max_optimizer_steps, int)
            or isinstance(max_optimizer_steps, bool)
            or max_optimizer_steps <= 0
        ):
            raise TrainingConfigurationError("max_optimizer_steps must be a positive integer or None")
        target_optimizer_steps = self.config.max_train_steps
        if max_optimizer_steps is not None:
            target_optimizer_steps = min(
                target_optimizer_steps,
                self._optimizer_steps + max_optimizer_steps,
            )
        final_output = None
        while self._optimizer_steps < target_optimizer_steps:
            final_output = self.train_step(self._next_batch())
        final_metrics = {}
        final_loss = None
        if final_output is not None:
            final_loss = float(final_output.loss.float().cpu())
            final_metrics = {
                name: float(value.float().cpu()) if isinstance(value, torch.Tensor) else float(value)
                for name, value in final_output.metrics.items()
            }
        return TrainingSummary3D(
            final_loss=final_loss,
            final_metrics=final_metrics,
            micro_steps=self._micro_steps,
            optimizer_steps=self._optimizer_steps,
        )

    def _checkpoint_components(self) -> Mapping[str, nn.Module]:
        components = {}
        for key, component in self._components.items():
            unwrapped = self.accelerator.unwrap_model(component)
            policy = self._selected_policies[key]
            if type(unwrapped) not in policy.expected_types:
                raise TrainingCheckpointError(
                    f"Checkpoint component {key!r} must unwrap to an exact reviewed component type"
                )
            components[key] = unwrapped
        return MappingProxyType(components)

    def _trainer_state_dict(self) -> dict[str, object]:
        if self._dataloader_epoch_generator_state is None:
            raise TrainingCheckpointError("dataloader generator state is unavailable")
        return {
            "dataloader_epoch": self._dataloader_epoch,
            "dataloader_epoch_generator_state": self._dataloader_epoch_generator_state.tolist(),
            "dataloader_position": self._dataloader_position,
            "micro_steps": self._micro_steps,
            "optimizer_steps": self._optimizer_steps,
            "schema_version": TRAINER_STATE_SCHEMA_VERSION,
        }

    def _save_trainer_state(self, state_directory: Path) -> None:
        destination = state_directory / TRAINER_STATE_NAME
        payload = json.dumps(self._trainer_state_dict(), indent=2, sort_keys=True) + "\n"
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=state_directory,
            prefix=f".{TRAINER_STATE_NAME}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            os.chmod(destination, 0o644)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise TrainingCheckpointError(f"Could not atomically save trainer state to {destination}") from error

    def _save_accelerator_state(self, directory: Path) -> None:
        destination = directory / ACCELERATOR_STATE_DIRECTORY
        temporary = directory / f".{ACCELERATOR_STATE_DIRECTORY}.tmp"
        backup = directory / f".{ACCELERATOR_STATE_DIRECTORY}.backup"
        if self.accelerator.is_main_process:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            temporary.mkdir(parents=True)
        self.accelerator.wait_for_everyone()
        try:
            self.accelerator.save_state(temporary, safe_serialization=True)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_trainer_state(temporary)
                if destination.exists():
                    os.replace(destination, backup)
                os.replace(temporary, destination)
                shutil.rmtree(backup, ignore_errors=True)
            self.accelerator.wait_for_everyone()
        except Exception as error:
            if self.accelerator.is_main_process:
                shutil.rmtree(temporary, ignore_errors=True)
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
            raise TrainingCheckpointError(f"Could not save Accelerator state to {destination}") from error

    def _load_trainer_state(self, state_directory: Path) -> None:
        path = state_directory / TRAINER_STATE_NAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingCheckpointError(f"Could not read trainer state from {path}") from error
        expected_fields = {
            "dataloader_epoch",
            "dataloader_epoch_generator_state",
            "dataloader_position",
            "micro_steps",
            "optimizer_steps",
            "schema_version",
        }
        if not isinstance(data, dict) or set(data) != expected_fields:
            raise TrainingCheckpointError("trainer state has invalid fields")
        if data["schema_version"] != TRAINER_STATE_SCHEMA_VERSION:
            raise TrainingCheckpointError("trainer state has an unsupported schema version")
        for name in ("dataloader_epoch", "dataloader_position", "micro_steps", "optimizer_steps"):
            value = data[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TrainingCheckpointError(f"trainer state {name} must be a non-negative integer")
        generator_values = data["dataloader_epoch_generator_state"]
        if not isinstance(generator_values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
            for value in generator_values
        ):
            raise TrainingCheckpointError("trainer state generator state must be a JSON byte array")
        if data["optimizer_steps"] > self.config.max_train_steps:
            raise TrainingCheckpointError("trainer state optimizer_steps exceeds configured max_train_steps")
        self._dataloader_epoch = data["dataloader_epoch"]
        self._dataloader_position = data["dataloader_position"]
        self._micro_steps = data["micro_steps"]
        self._optimizer_steps = data["optimizer_steps"]
        generator_state = torch.tensor(generator_values, dtype=torch.uint8)
        self._dataloader_iterator = None
        self._start_dataloader_epoch(
            skip_batches=self._dataloader_position,
            generator_state=generator_state,
        )

    def save_checkpoint(self, checkpoint_directory: str | Path | None = None) -> Path:
        if not self._prepared:
            self.prepare()
        if self.config.dataloader_num_workers != 0:
            raise TrainingCheckpointError("Exact checkpoint continuation requires dataloader_num_workers=0")
        directory = Path(checkpoint_directory) if checkpoint_directory is not None else self.config.output_dir
        if directory is None:
            raise TrainingCheckpointError("checkpoint_directory or config.output_dir is required")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            directory.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self._checkpoint_components()
        self._save_accelerator_state(directory)
        if self.accelerator.is_main_process:
            self.recipe.save_weights(directory, self.strategy, self._checkpoint_components())
            self.manifest.save(directory)
        self.accelerator.wait_for_everyone()
        return directory / "diffusers_3d_training.json"

    def validate_resume(self, checkpoint_directory: str | Path) -> TrainingManifest3D:
        if not self._prepared:
            self.prepare()
        loaded = TrainingManifest3D.load(checkpoint_directory)
        loaded.validate_resume(self.manifest)
        return loaded

    def load_checkpoint(self, checkpoint_directory: str | Path) -> None:
        if self.config.dataloader_num_workers != 0:
            raise TrainingCheckpointError("Exact checkpoint continuation requires dataloader_num_workers=0")
        self.validate_resume(checkpoint_directory)
        state_directory = Path(checkpoint_directory) / ACCELERATOR_STATE_DIRECTORY
        if not state_directory.is_dir():
            raise TrainingCheckpointError(f"Accelerator state directory {state_directory} does not exist")
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(state_directory)
        self._checkpoint_components()
        self._load_trainer_state(state_directory)
        self.accelerator.wait_for_everyone()


__all__ = [
    "ACCELERATOR_STATE_DIRECTORY",
    "TRAINER_STATE_NAME",
    "TRAINER_STATE_SCHEMA_VERSION",
    "Object3DTrainer",
]
