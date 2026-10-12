__version__ = "0.41.0.dev0"

from typing import TYPE_CHECKING

from .utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_accelerate_available,
    is_auto_round_available,
    is_bitsandbytes_available,
    is_gguf_available,
    is_nvidia_modelopt_available,
    is_onnx_available,
    is_optimum_quanto_available,
    is_scipy_available,
    is_sdnq_available,
    is_torch_available,
    is_torchao_available,
    is_torchsde_available,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "guiders": [],
    "hooks": [],
    "loaders": ["PeftAdapterMixin"],
    "models": [],
    "modular_pipelines": [],
    "pipelines": [],
    "quantizers.pipe_quant_config": ["PipelineQuantizationConfig"],
    "quantizers.quantization_config": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_inflect_available",
        "is_invisible_watermark_available",
        "is_librosa_available",
        "is_note_seq_available",
        "is_onnx_available",
        "is_scipy_available",
        "is_torch_available",
        "is_torchsde_available",
        "is_transformers_available",
        "is_transformers_version",
        "is_unidecode_available",
        "logging",
    ],
}

try:
    if not is_torch_available() and not is_accelerate_available() and not is_bitsandbytes_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_bitsandbytes_objects

    _import_structure["utils.dummy_bitsandbytes_objects"] = [
        name for name in dir(dummy_bitsandbytes_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("BitsAndBytesConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_gguf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_gguf_objects

    _import_structure["utils.dummy_gguf_objects"] = [
        name for name in dir(dummy_gguf_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("GGUFQuantizationConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_torchao_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torchao_objects

    _import_structure["utils.dummy_torchao_objects"] = [
        name for name in dir(dummy_torchao_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("TorchAoConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_optimum_quanto_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_optimum_quanto_objects

    _import_structure["utils.dummy_optimum_quanto_objects"] = [
        name for name in dir(dummy_optimum_quanto_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("QuantoConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_nvidia_modelopt_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_nvidia_modelopt_objects

    _import_structure["utils.dummy_nvidia_modelopt_objects"] = [
        name for name in dir(dummy_nvidia_modelopt_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("NVIDIAModelOptConfig")

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_nunchaku_lite_objects

    _import_structure["utils.dummy_nunchaku_lite_objects"] = [
        name for name in dir(dummy_nunchaku_lite_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("NunchakuLiteQuantizationConfig")

try:
    if not is_auto_round_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_auto_round_objects

    _import_structure["utils.dummy_auto_round_objects"] = [
        name for name in dir(dummy_auto_round_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("AutoRoundConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_sdnq_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sdnq_objects

    _import_structure["utils.dummy_sdnq_objects"] = [
        name for name in dir(dummy_sdnq_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("SDNQConfig")

try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_onnx_objects  # noqa F403

    _import_structure["utils.dummy_onnx_objects"] = [
        name for name in dir(dummy_onnx_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["OnnxRuntimeModel"])

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects  # noqa F403

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:
    _import_structure["guiders"].extend(
        [
            "AdaptiveProjectedGuidance",
            "AdaptiveProjectedMixGuidance",
            "AutoGuidance",
            "BaseGuidance",
            "ClassifierFreeGuidance",
            "ClassifierFreeZeroStarGuidance",
            "FrequencyDecoupledGuidance",
            "MagnitudeAwareGuidance",
            "PerturbedAttentionGuidance",
            "SkipLayerGuidance",
            "SmoothedEnergyGuidance",
            "TangentialClassifierFreeGuidance",
        ]
    )
    _import_structure["hooks"].extend(
        [
            "FasterCacheConfig",
            "FirstBlockCacheConfig",
            "HookRegistry",
            "LayerSkipConfig",
            "MagCacheConfig",
            "PyramidAttentionBroadcastConfig",
            "SmoothedEnergyGuidanceConfig",
            "TaylorSeerCacheConfig",
            "apply_faster_cache",
            "apply_first_block_cache",
            "apply_layer_skip",
            "apply_mag_cache",
            "apply_pyramid_attention_broadcast",
            "apply_taylorseer_cache",
        ]
    )
    _import_structure["image_processor"] = [
        "InpaintProcessor",
        "IPAdapterMaskProcessor",
        "PixArtImageProcessor",
        "VaeImageProcessor",
        "VaeImageProcessorLDM3D",
    ]
    _import_structure["loaders"].extend(["AttnProcsLayers", "LoraBaseMixin"])
    _import_structure["models"].extend(
        [
            "AttentionBackendName",
            "AutoModel",
            "CacheMixin",
            "ContextParallelConfig",
            "ModelMixin",
            "ParallelConfig",
            "TensorParallelConfig",
            "attention_backend",
        ]
    )
    _import_structure["modular_pipelines"].extend(
        [
            "AutoPipelineBlocks",
            "BlockState",
            "ComponentsManager",
            "ComponentSpec",
            "ConditionalPipelineBlocks",
            "ConfigSpec",
            "InputParam",
            "InsertableDict",
            "LoopSequentialPipelineBlocks",
            "ModularPipeline",
            "ModularPipelineBlocks",
            "OutputParam",
            "PipelineState",
            "SequentialPipelineBlocks",
        ]
    )
    _import_structure["optimization"] = [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["pipelines"].extend(
        [
            "AudioPipelineOutput",
            "DiffusionPipeline",
            "ImagePipelineOutput",
        ]
    )
    _import_structure["quantizers"] = ["DiffusersQuantizer"]
    _import_structure["schedulers"].extend(
        [
            "AmusedScheduler",
            "BlockRefinementScheduler",
            "BlockRefinementSchedulerOutput",
            "CMStochasticIterativeScheduler",
            "CogVideoXDDIMScheduler",
            "CogVideoXDPMScheduler",
            "DDIMInverseScheduler",
            "DDIMParallelScheduler",
            "DDIMScheduler",
            "DDPMParallelScheduler",
            "DDPMScheduler",
            "DDPMWuerstchenScheduler",
            "DEISMultistepScheduler",
            "DiscreteDDIMScheduler",
            "DiscreteDDIMSchedulerOutput",
            "DPMSolverMultistepInverseScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "EDMDPMSolverMultistepScheduler",
            "EDMEulerScheduler",
            "EntropyBoundScheduler",
            "EntropyBoundSchedulerOutput",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "FlowMapEulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchHeunDiscreteScheduler",
            "FlowMatchLCMScheduler",
            "HeliosDMDScheduler",
            "HeliosScheduler",
            "HeunDiscreteScheduler",
            "IPNDMScheduler",
            "KarrasVeScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "LCMScheduler",
            "LTXEulerAncestralRFScheduler",
            "MiniMaxH3Scheduler",
            "PNDMScheduler",
            "RePaintScheduler",
            "SASolverScheduler",
            "SchedulerMixin",
            "SCMScheduler",
            "ScoreSdeVeScheduler",
            "TCDScheduler",
            "UnCLIPScheduler",
            "UniPCMultistepScheduler",
            "VQDiffusionScheduler",
        ]
    )
    _import_structure["training_utils"] = ["EMAModel"]
    _import_structure["video_processor"] = ["VideoProcessor"]

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_scipy_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_scipy_objects"] = [
        name for name in dir(dummy_torch_and_scipy_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["LMSDiscreteScheduler"])

try:
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_torchsde_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_torchsde_objects"] = [
        name for name in dir(dummy_torch_and_torchsde_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["CosineDPMSolverMultistepScheduler", "DPMSolverSDEScheduler"])


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin
    from .loaders import PeftAdapterMixin
    from .quantizers import PipelineQuantizationConfig

    try:
        if not is_bitsandbytes_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_bitsandbytes_objects import *
    else:
        from .quantizers.quantization_config import BitsAndBytesConfig

    try:
        if not is_gguf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_gguf_objects import *
    else:
        from .quantizers.quantization_config import GGUFQuantizationConfig

    try:
        if not is_torchao_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torchao_objects import *
    else:
        from .quantizers.quantization_config import TorchAoConfig

    try:
        if not is_optimum_quanto_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_optimum_quanto_objects import *
    else:
        from .quantizers.quantization_config import QuantoConfig

    try:
        if not is_nvidia_modelopt_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_nvidia_modelopt_objects import *
    else:
        from .quantizers.quantization_config import NVIDIAModelOptConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_nunchaku_lite_objects import *
    else:
        from .quantizers.quantization_config import NunchakuLiteQuantizationConfig

    try:
        if not is_auto_round_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_auto_round_objects import *
    else:
        from .quantizers.quantization_config import AutoRoundConfig

    try:
        if not is_sdnq_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_sdnq_objects import *
    else:
        from .quantizers.quantization_config import SDNQConfig

    try:
        if not is_onnx_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_onnx_objects import *  # noqa F403
    else:
        from .pipelines import OnnxRuntimeModel

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *  # noqa F403
    else:
        from .guiders import (
            AdaptiveProjectedGuidance,
            AdaptiveProjectedMixGuidance,
            AutoGuidance,
            BaseGuidance,
            ClassifierFreeGuidance,
            ClassifierFreeZeroStarGuidance,
            FrequencyDecoupledGuidance,
            MagnitudeAwareGuidance,
            PerturbedAttentionGuidance,
            SkipLayerGuidance,
            SmoothedEnergyGuidance,
            TangentialClassifierFreeGuidance,
        )
        from .hooks import (
            FasterCacheConfig,
            FirstBlockCacheConfig,
            HookRegistry,
            LayerSkipConfig,
            MagCacheConfig,
            PyramidAttentionBroadcastConfig,
            SmoothedEnergyGuidanceConfig,
            TaylorSeerCacheConfig,
            apply_faster_cache,
            apply_first_block_cache,
            apply_layer_skip,
            apply_mag_cache,
            apply_pyramid_attention_broadcast,
            apply_taylorseer_cache,
        )
        from .image_processor import (
            InpaintProcessor,
            IPAdapterMaskProcessor,
            PixArtImageProcessor,
            VaeImageProcessor,
            VaeImageProcessorLDM3D,
        )
        from .loaders import AttnProcsLayers, LoraBaseMixin
        from .models import (
            AttentionBackendName,
            AutoModel,
            CacheMixin,
            ContextParallelConfig,
            ModelMixin,
            ParallelConfig,
            TensorParallelConfig,
            attention_backend,
        )
        from .modular_pipelines import (
            AutoPipelineBlocks,
            BlockState,
            ComponentsManager,
            ComponentSpec,
            ConditionalPipelineBlocks,
            ConfigSpec,
            InputParam,
            InsertableDict,
            LoopSequentialPipelineBlocks,
            ModularPipeline,
            ModularPipelineBlocks,
            OutputParam,
            PipelineState,
            SequentialPipelineBlocks,
        )
        from .optimization import (
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )
        from .pipelines import (
            AudioPipelineOutput,
            DiffusionPipeline,
            ImagePipelineOutput,
        )
        from .quantizers import DiffusersQuantizer
        from .schedulers import (
            AmusedScheduler,
            BlockRefinementScheduler,
            BlockRefinementSchedulerOutput,
            CMStochasticIterativeScheduler,
            CogVideoXDDIMScheduler,
            CogVideoXDPMScheduler,
            DDIMInverseScheduler,
            DDIMParallelScheduler,
            DDIMScheduler,
            DDPMParallelScheduler,
            DDPMScheduler,
            DDPMWuerstchenScheduler,
            DEISMultistepScheduler,
            DiscreteDDIMScheduler,
            DiscreteDDIMSchedulerOutput,
            DPMSolverMultistepInverseScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EDMDPMSolverMultistepScheduler,
            EDMEulerScheduler,
            EntropyBoundScheduler,
            EntropyBoundSchedulerOutput,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            FlowMapEulerDiscreteScheduler,
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            FlowMatchLCMScheduler,
            HeliosDMDScheduler,
            HeliosScheduler,
            HeunDiscreteScheduler,
            IPNDMScheduler,
            KarrasVeScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LCMScheduler,
            LTXEulerAncestralRFScheduler,
            MiniMaxH3Scheduler,
            PNDMScheduler,
            RePaintScheduler,
            SASolverScheduler,
            SchedulerMixin,
            SCMScheduler,
            ScoreSdeVeScheduler,
            TCDScheduler,
            UnCLIPScheduler,
            UniPCMultistepScheduler,
            VQDiffusionScheduler,
        )
        from .training_utils import EMAModel
        from .video_processor import VideoProcessor

    try:
        if not (is_torch_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
    else:
        from .schedulers import LMSDiscreteScheduler

    try:
        if not (is_torch_available() and is_torchsde_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_torchsde_objects import *  # noqa F403
    else:
        from .schedulers import CosineDPMSolverMultistepScheduler, DPMSolverSDEScheduler

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
