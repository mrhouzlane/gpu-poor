"""Read config and optional safetensors headers. Never instantiate a model."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig

SUPPORTED = {"llama", "mistral", "qwen2"}


def positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    parameters: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    parameter_source: str = "config formula"
    revision: str | None = None
    sliding_window: int | None = None

    def kv_bytes_per_token(self, dtype: str = "bf16") -> int:
        sizes = {"bf16": 2, "fp16": 2, "fp8": 1}
        if dtype not in sizes:
            raise ValueError("KV dtype must be bf16, fp16, or fp8; int4 KV is not modeled")
        return 2 * self.layers * self.kv_heads * self.head_dim * sizes[dtype]


def parse_config(raw: dict[str, Any], model_id: str = "local") -> ModelSpec:
    """Dense Llama/Mistral/Qwen2 only; reject unsupported layouts rather than guess."""
    if not isinstance(raw, dict):
        raise ValueError("config.json must contain an object")
    model_type = raw.get("model_type")
    if not isinstance(model_type, str) or model_type not in SUPPORTED:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Supported: {sorted(SUPPORTED)}. "
            "MoE, multimodal, encoder-decoder, MLA and state-space are not modeled."
        )
    if raw.get("quantization_config"):
        raise ValueError("Use the original dense model id, not a pre-quantized config")
    if raw.get("num_local_experts") or raw.get("num_experts") or raw.get("text_config"):
        raise ValueError("Expert and nested text layouts are not supported")
    # Dimensions must exist in the actual file, not be invented by registry defaults.
    hidden = positive_int(raw, "hidden_size")
    layers = positive_int(raw, "num_hidden_layers")
    heads = positive_int(raw, "num_attention_heads")
    intermediate = positive_int(raw, "intermediate_size")
    vocab = positive_int(raw, "vocab_size")
    max_context = positive_int(raw, "max_position_embeddings")
    kv_heads = positive_int({"kv": raw.get("num_key_value_heads", heads)}, "kv")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if "head_dim" not in raw and hidden % heads:
        raise ValueError("hidden_size must divide attention heads, or specify head_dim")
    head_dim = positive_int({"dim": raw.get("head_dim", hidden // heads)}, "dim")
    # AutoConfig uses the installed Transformers registry; never execute Hub remote code.
    config = AutoConfig.for_model(
        model_type, **{k: v for k, v in raw.items() if k != "model_type"}
    ).to_dict()
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    # q, k, v, o matrices; gated MLP gate/up/down; two RMSNorms per layer.
    per_layer = hidden * (2 * q_dim + 2 * kv_dim) + 3 * hidden * intermediate + 2 * hidden
    # Qwen2 always has Q/K/V biases, but no O bias.
    if model_type == "qwen2":
        per_layer += q_dim + 2 * kv_dim
    elif config.get("attention_bias", False):
        per_layer += q_dim + 2 * kv_dim + hidden
    if config.get("mlp_bias", False):
        per_layer += 2 * intermediate + hidden
    embeddings = vocab * hidden * (1 if config.get("tie_word_embeddings", False) else 2)
    parameters = embeddings + layers * per_layer + hidden  # final RMSNorm
    sliding = raw.get("sliding_window")
    if model_type == "qwen2" and not raw.get("use_sliding_window", False):
        sliding = None
    return ModelSpec(
        model_id,
        str(model_type),
        parameters,
        layers,
        heads,
        kv_heads,
        head_dim,
        hidden,
        max_context,
        sliding_window=sliding,
    )


def load_model(
    model_id: str,
    *,
    revision: str = "main",
    parameter_source: Literal["config", "safetensors"] = "config",
    local_files_only: bool = False,
    config_path: str | Path | None = None,
) -> ModelSpec:
    """Download config.json only; optional API fetch reads safetensors headers via ranges.

    HF_TOKEN is consumed by huggingface_hub normally. No credential handling here.
    A local config is useful for completely offline reproduction.
    """
    if parameter_source not in {"config", "safetensors"}:
        raise ValueError("parameter_source must be config or safetensors")
    if parameter_source == "safetensors" and (local_files_only or config_path is not None):
        raise ValueError("safetensors metadata needs Hub access; use config for offline mode")
    path = (
        Path(config_path)
        if config_path
        else Path(
            hf_hub_download(
                model_id,
                "config.json",
                revision=revision,
                local_files_only=local_files_only,
                library_name="gpu-poor",
            )
        )
    )
    spec = parse_config(json.loads(path.read_text()), model_id)
    # hf_hub_download cache paths include the immutable commit under snapshots/.
    resolved = path.parent.name if path.parent.parent.name == "snapshots" else revision
    spec = replace(spec, revision=resolved)
    if parameter_source == "safetensors":
        api = HfApi(library_name="gpu-poor")
        if not hasattr(api, "get_safetensors_metadata"):
            warnings.warn(
                "Hub client lacks safetensors metadata; using config formula", stacklevel=2
            )
            return spec
        # Public API reads tensor headers, not tensor payloads. Errors are not hidden.
        metadata = api.get_safetensors_metadata(model_id, revision=resolved)
        count = sum(metadata.parameter_count.values())
        if count <= 0:
            raise ValueError("Empty safetensors parameter metadata")
        spec = replace(spec, parameters=count, parameter_source="safetensors headers")
    return spec
