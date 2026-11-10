"""A simple single-GPU decode roofline, not an engine benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ModelSpec
from .prices import PriceRange
from .specs import GPU

WEIGHT_BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}


@dataclass(frozen=True)
class Estimate:
    batch: int
    context: int
    fits: bool
    weight_bytes: int
    kv_bytes_per_token: int
    kv_bytes: int
    max_context_per_sequence: int
    max_total_context_tokens: int
    memory_tokens_s: float | None
    compute_tokens_s: float | None
    aggregate_tokens_s: float | None
    tokens_s_per_sequence: float | None
    arithmetic_intensity: float
    ridge_flops_per_byte: float
    crossover_batch: float | None
    bottleneck: str
    usd_per_million_low: float | None
    usd_per_million_median: float | None
    usd_per_million_high: float | None


def cost_per_million(price_per_hour: float, tokens_s: float) -> float:
    if not math.isfinite(price_per_hour) or price_per_hour < 0:
        raise ValueError("price must be finite and nonnegative")
    if not math.isfinite(tokens_s) or tokens_s <= 0:
        raise ValueError("tokens/s must be finite and positive")
    return price_per_hour * 1_000_000 / (tokens_s * 3600)


def estimate(
    model: ModelSpec,
    gpu: GPU,
    *,
    dtype: str = "bf16",
    kv_dtype: str = "bf16",
    batch: int = 1,
    context: int = 4096,
    memory_fraction: float = 0.9,
    prices: PriceRange | None = None,
) -> Estimate:
    """An optimistic synchronous decode step: read weights once, KV once per sequence.

    Quantization changes storage only. All compute uses dense BF16 peak as an explicit
    simplifying assumption (weight-only dequantization); no FP8/INT4 kernel claim.
    """
    if dtype not in WEIGHT_BYTES:
        raise ValueError("weight dtype must be bf16, fp8, or int4")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise ValueError("context must be a positive integer")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")
    if context > model.max_position_embeddings:
        raise ValueError("context exceeds config.max_position_embeddings")
    weight_bytes = math.ceil(model.parameters * WEIGHT_BYTES[dtype])
    kv_per_token = model.kv_bytes_per_token(kv_dtype)
    capacity = max(0, math.floor(gpu.memory_bytes * memory_fraction) - weight_bytes)
    total_tokens = capacity // kv_per_token
    max_context = min(total_tokens // batch, model.max_position_embeddings)
    kv_bytes = batch * context * kv_per_token
    fits = context <= max_context
    # KV writes, activations, launches, and normalization are omitted from ideal traffic.
    traffic = weight_bytes + kv_bytes
    # Approximate linear layers by 2P; QK and AV add 4*L*Hq*D*C FLOPs/token.
    flops_per_token = 2 * model.parameters + (
        4 * model.layers * model.attention_heads * model.head_dim * context
    )
    flops_step = batch * flops_per_token
    bandwidth = gpu.bandwidth_bytes_s
    peak = gpu.bf16_tflops * 10**12
    mem_rate = batch * bandwidth / traffic
    compute_rate = peak / flops_per_token
    # Solve B*F/(W+B*K*C) = peak/bandwidth. May have no finite crossover.
    denominator = flops_per_token * bandwidth - peak * context * kv_per_token
    crossover = peak * weight_bytes / denominator if denominator > 0 else None
    throughput = min(mem_rate, compute_rate) if fits else None
    costs = [
        cost_per_million(v, throughput) if throughput is not None and prices else None
        for v in ([prices.low, prices.median, prices.high] if prices else [0, 0, 0])
    ]
    return Estimate(
        batch,
        context,
        fits,
        weight_bytes,
        kv_per_token,
        kv_bytes,
        max_context,
        min(total_tokens, batch * model.max_position_embeddings),
        mem_rate if fits else None,
        compute_rate if fits else None,
        throughput,
        throughput / batch if throughput is not None else None,
        flops_step / traffic,
        peak / bandwidth,
        crossover,
        ("memory" if mem_rate <= compute_rate else "compute") if fits else "does not fit",
        *costs,
    )
