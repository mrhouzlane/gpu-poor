import math
from dataclasses import replace

import pytest

from gpu_poor.estimate import cost_per_million, estimate
from gpu_poor.prices import PriceRange
from gpu_poor.specs import GPU, GPUS, get_gpu


def test_hand_checked_memory_and_traffic(llama):
    result = estimate(llama, GPUS["H100"], batch=8, context=4096)
    assert result.weight_bytes == 16_060_522_496
    assert result.kv_bytes == 8 * 4096 * 131_072
    capacity = 72_000_000_000 - 16_060_522_496
    assert result.max_context_per_sequence == capacity // (8 * 131_072)
    assert result.memory_tokens_s == pytest.approx(
        8 * 3_350_000_000_000 / (16_060_522_496 + 8 * 4096 * 131_072)
    )
    assert result.tokens_s_per_sequence == pytest.approx(result.aggregate_tokens_s / 8)


def test_quantization_weight_only(llama):
    a = estimate(llama, GPUS["H100"], dtype="bf16")
    b = estimate(llama, GPUS["H100"], dtype="fp8")
    c = estimate(llama, GPUS["H100"], dtype="int4")
    assert a.weight_bytes == 2 * b.weight_bytes == 4 * c.weight_bytes
    assert a.kv_bytes_per_token == b.kv_bytes_per_token == c.kv_bytes_per_token
    assert a.compute_tokens_s == b.compute_tokens_s == c.compute_tokens_s


def test_fp8_kv_independent(llama):
    a = estimate(llama, GPUS["H100"])
    b = estimate(llama, GPUS["H100"], kv_dtype="fp8")
    assert a.kv_bytes == 2 * b.kv_bytes
    assert a.weight_bytes == b.weight_bytes


def test_no_fit_no_costs(llama):
    r = estimate(llama, GPUS["RTX4090"], batch=32, context=4096)
    assert not r.fits
    assert r.aggregate_tokens_s is None
    assert r.usd_per_million_median is None
    assert r.bottleneck == "does not fit"


def test_weights_do_not_fit(llama):
    tiny = GPU("tiny", 1, 100, 100, "fixture")
    r = estimate(llama, tiny)
    assert not r.fits
    assert r.max_context_per_sequence == 0
    assert r.max_total_context_tokens == 0


def test_exact_capacity_boundary(llama):
    needed = 2 * llama.parameters + 100 * llama.kv_bytes_per_token()
    gpu = GPU("fixture", needed / 1e9, 1000, 1000, "fixture")
    assert estimate(llama, gpu, context=100, memory_fraction=1).fits
    assert not estimate(llama, gpu, context=101, memory_fraction=1).fits


def test_config_context_cap(llama):
    tiny_model = replace(llama, max_position_embeddings=100)
    r = estimate(tiny_model, GPUS["MI300X"], context=100)
    assert r.max_context_per_sequence == 100
    with pytest.raises(ValueError, match="exceeds"):
        estimate(tiny_model, GPUS["H100"], context=101)


def test_roofline_compute_cap_and_crossover(llama):
    slow_compute = GPU("fixture", 1000, 3350, 1, "fixture")
    r = estimate(llama, slow_compute, batch=32)
    assert r.bottleneck == "compute"
    assert r.aggregate_tokens_s == r.compute_tokens_s
    assert r.crossover_batch is not None and r.crossover_batch < 32
    # Evaluate the algebraic crossover: arithmetic intensity equals the ridge.
    b = r.crossover_batch
    flops = 2 * llama.parameters + 4 * 32 * 32 * 128 * 4096
    intensity = b * flops / (r.weight_bytes + b * 4096 * r.kv_bytes_per_token)
    assert intensity == pytest.approx(r.ridge_flops_per_byte)


def test_no_finite_roofline_crossover(llama):
    r = estimate(llama, GPUS["H100"], context=32768)
    assert r.crossover_batch is None


def test_cost_conversion_and_range(llama):
    assert cost_per_million(3.6, 1000) == 1
    price = PriceRange(
        "H100", "2026-11-03", 1, 2, 4, "USD", "on demand", "fixture", True, "min/max"
    )
    r = estimate(llama, GPUS["H100"], prices=price)
    assert r.usd_per_million_low < r.usd_per_million_median < r.usd_per_million_high
    assert r.usd_per_million_median == pytest.approx(2e6 / (r.aggregate_tokens_s * 3600))


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch", 0),
        ("batch", -1),
        ("batch", True),
        ("batch", 1.5),
        ("context", 0),
        ("context", True),
        ("context", 2.5),
        ("memory_fraction", 0),
        ("memory_fraction", 1.1),
        ("memory_fraction", math.nan),
        ("dtype", "float"),
        ("kv_dtype", "int4"),
    ],
)
def test_invalid_inputs(llama, field, value):
    with pytest.raises(ValueError):
        estimate(llama, GPUS["H100"], **{field: value})


@pytest.mark.parametrize("price,rate", [(-1, 1), (1, 0), (1, -1), (math.nan, 1), (1, math.inf)])
def test_bad_cost_input(price, rate):
    with pytest.raises(ValueError):
        cost_per_million(price, rate)


@pytest.mark.parametrize(
    "name,key",
    [("H100 SXM", "H100"), ("A100 80GB", "A100"), ("RTX 4090", "RTX4090"), ("mi300x", "MI300X")],
)
def test_gpu_aliases(name, key):
    assert get_gpu(name) == GPUS[key]


def test_unknown_gpu():
    with pytest.raises(ValueError, match="Unknown GPU"):
        get_gpu("Imaginary GPU")
