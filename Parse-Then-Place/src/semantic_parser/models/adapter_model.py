# coding=utf8

import re
from dataclasses import dataclass, field
from typing import Optional, Union

import torch.nn as nn
from transformers.adapters import (CompacterConfig, HoulsbyConfig,
                                   ParallelConfig, PfeifferConfig,
                                   PrefixTuningConfig)
from transformers.optimization import Adafactor, AdamW
from transformers.trainer_pt_utils import get_parameter_names

from .pretrained_model import PretrainedLM


@dataclass
class AdapterArguments:
    adapter_name: Optional[str] = field(
        default="parsing",
        metadata={
            "help": "name in model",
        }
    )
    adapter_type: Optional[str] = field(
        default="prefix_tuning",
        metadata={
            "help": "Adapter type such as bottleneck, prefix_tuning",
            "choices": ["Houlsby", "Pfeiffer", "Parallel", "prefix_tuning"]
        },
    )

    # bottlenect adapter
    adapter_non_linearity: Optional[str] = field(
        default="relu", metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[float] = field(
        default=16, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    adapter_scaling: Optional[float] = field(
        default=4.0, metadata={"help": "Override the scaling factor of the adapter configuration, applicable in LoRA and Parallel."}
    )

    # Prefix tuning
    prefix_length: Optional[int] = field(
        default=30,
        metadata={"help": "prefix length in prefix_tuning"}
    )


class AdapterPretrainedLM(PretrainedLM):

    def __init__(self, model_name: str, adapter_args: AdapterArguments):
        super().__init__(model_name)

        if adapter_args.adapter_type == "Houlsby":
            adapter_config = HoulsbyConfig(
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
        elif adapter_args.adapter_type == "Pfeiffer":
            adapter_config = PfeifferConfig(
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
        elif adapter_args.adapter_type == "Parallel":
            adapter_config = ParallelConfig(
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                scaling=adapter_args.adapter_scaling
            )
        elif adapter_args.adapter_type == "compacter":
            adapter_config = CompacterConfig(
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor
            )
        elif adapter_args.adapter_type ==  'prefix_tuning':
            adapter_config = PrefixTuningConfig(
                flat=False,
                prefix_length=adapter_args.prefix_length
            )
        else:
            raise NotImplementedError(f"Adapter: {adapter_args.adapter_type} is not supported yet")
        # Add adapter
        self.model.add_adapter(adapter_args.adapter_name, config=adapter_config)

        # Freeze all model weights except of those of this adapter
        self.model.train_adapter([adapter_args.adapter_name])

        self.model.set_active_adapters(adapter_args.adapter_name)


def create_adapter_optimizer(model: AdapterPretrainedLM, training_args) -> Union[AdamW, Adafactor]:
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if hasattr(model.model, "config") and hasattr(model.model.config, "adapters"):
        match_str = r"adapter_fusion_layer\..*\.value"
        decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    if training_args.use_adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }
    optimizer_kwargs["lr"] = training_args.learning_rate
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
