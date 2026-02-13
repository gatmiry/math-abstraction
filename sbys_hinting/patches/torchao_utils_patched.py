"""
Common utilities for torchao - patched for torchao 0.16+ compatibility.
"""

import logging
import os
import pwd
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def get_gemlite_cache_path() -> str:
    return f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"


def save_gemlite_cache(print_error: bool = False) -> bool:
    try:
        from gemlite.core import GemLiteLinearTriton
        GemLiteLinearTriton.cache_config(get_gemlite_cache_path())
    except Exception:
        if print_error:
            logger.error("Failed to save the GemLite cache.")
        return False
    return True


def proj_filter(module: torch.nn.Module, fqn: str):
    return "proj" in fqn


def apply_torchao_config_to_model(
    model: torch.nn.Module,
    torchao_config: str,
    filter_fn: Optional[Callable] = proj_filter,
):
    from torchao.quantization import quantize_
    
    # New Config-based API (torchao >= 0.16.0)
    from torchao.quantization import (
        Float8WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        PerRow, PerTensor,
    )

    if torchao_config == "" or torchao_config is None:
        return model
    elif "int8wo" in torchao_config:
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=filter_fn)
    elif "int8dq" in torchao_config:
        quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=filter_fn)
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [32, 64, 128, 256]
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size), filter_fn=filter_fn)
    elif "gemlite" in torchao_config:
        from gemlite.core import GemLiteLinearTriton
        from torchao.quantization import gemlite_uintx_weight_only
        _quant_args = torchao_config.split("-")
        bit_width = int(_quant_args[-2])
        group_size = None if _quant_args[-1] == "None" else int(_quant_args[-1])
        try:
            packing_bitwidth = int(_quant_args[-3])
        except (ValueError, IndexError):
            packing_bitwidth = 32
        quantize_(model, gemlite_uintx_weight_only(group_size, bit_width, packing_bitwidth))
        GemLiteLinearTriton.load_config(get_gemlite_cache_path())
    elif "fp8wo" in torchao_config:
        quantize_(model, Float8WeightOnlyConfig(), filter_fn=filter_fn)
    elif "fp8dq" in torchao_config:
        granularity = torchao_config.split("-")[-1]
        GRANULARITY_MAP = {"per_row": PerRow(), "per_tensor": PerTensor()}
        assert granularity in GRANULARITY_MAP
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=GRANULARITY_MAP[granularity]), filter_fn=filter_fn)
    else:
        raise ValueError(f"Unexpected config: {torchao_config}")

    return model
