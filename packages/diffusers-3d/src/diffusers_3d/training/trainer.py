from __future__ import annotations

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
from torch.utils.data import DataLoader

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
)


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
        self._trainable_parameters: tuple[nn.Parameter, ...] = ()
        self._trainable_parameter_names: tuple[str, ...] = ()
        self._manifest: TrainingManifest3D | None = None
        self._micro_steps = 0
        self._optimizer_steps = 0

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

        if len({id(component) for component in selected_components.values()}) != len(selected_components):
            raise TrainingTargetError("Selected component keys must resolve to distinct exact component objects")
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
        before_parameters = _reachable_named_parameters(target)
        for key, component in selected_components.items():
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

            generator = torch.Generator()
            generator.manual_seed(self.config.seed)

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
                shuffle=self.config.shuffle,
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
            set_seed(self.config.seed)
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                cpu=self.config.cpu,
            )

            unique_components: list[nn.Module] = []
            component_indices: dict[str, int] = {}
            component_id_to_index: dict[int, int] = {}
            for key in self.strategy.components:
                component = selected_components[key]
                if id(component) not in component_id_to_index:
                    component_id_to_index[id(component)] = len(unique_components)
                    unique_components.append(component)
                component_indices[key] = component_id_to_index[id(component)]

            prepared = accelerator.prepare(*unique_components, optimizer, dataloader, lr_scheduler)
            prepared_values = prepared if isinstance(prepared, tuple) else (prepared,)
            wrapped_components = prepared_values[: len(unique_components)]
            optimizer = prepared_values[-3]
            dataloader = prepared_values[-2]
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
            )
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

    def train(self) -> tuple[TrainingStep3DOutput, ...]:
        if not self._prepared:
            self.prepare()
        outputs = []
        iterator = iter(self.dataloader)
        while self._optimizer_steps < self.config.max_train_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                batch = next(iterator)
            outputs.append(self.train_step(batch))
        return tuple(outputs)

    def _checkpoint_components(self) -> Mapping[str, nn.Module]:
        return MappingProxyType(
            {key: self.accelerator.unwrap_model(component) for key, component in self._components.items()}
        )

    def save_checkpoint(self, checkpoint_directory: str | Path | None = None) -> Path:
        if not self._prepared:
            self.prepare()
        directory = Path(checkpoint_directory) if checkpoint_directory is not None else self.config.output_dir
        if directory is None:
            raise TrainingCheckpointError("checkpoint_directory or config.output_dir is required")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            directory.mkdir(parents=True, exist_ok=True)
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
        self.validate_resume(checkpoint_directory)
        self.recipe.load_weights(checkpoint_directory, self.strategy, self._checkpoint_components())


__all__ = ["Object3DTrainer"]
