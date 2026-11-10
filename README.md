---
title: gpu-poor
emoji: 🧮
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: Model configs to decode rooflines and rental costs
license: mit
---

# gpu-poor

**Try it in your browser:** [huggingface.co/spaces/newway14/gpu-poor](https://huggingface.co/spaces/newway14/gpu-poor)

**GPU-poor and proud.** What does serving this model cost before downloading its weights?

A small Python library, CLI and CPU-only Gradio demo connecting Hugging Face model
configs to single-GPU memory, decode rooflines and dated GPU rental price ranges.
For Hourly research. By Mehdi Rhouzlane ([GitHub](https://github.com/mrhouzlane)).

**All performance numbers are theoretical upper bounds within an idealized model,
not measured tokens/s. Costs at those rates are optimistic floors, not quotes.**
There is no claim that a particular engine reaches them. No model weights, GPU,
PyTorch install, Hub write access, or remote-code execution are needed.

## Quick start

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,app,data]'
gpu-poor Qwen/Qwen2.5-7B-Instruct --gpu H100 --dtype bf16 --batch 1,8,32
# Smaller weights, independently selected KV precision:
gpu-poor HuggingFaceTB/SmolLM2-1.7B-Instruct --gpu RTX4090 --dtype int4 --no-prices
# Optional safetensors headers (HTTP range reads, not tensor payloads):
gpu-poor Qwen/Qwen2.5-7B-Instruct --parameter-source safetensors --json
# Explicit price file and historical date:
gpu-poor Qwen/Qwen2.5-7B-Instruct --prices src/gpu_poor/data/hourly-prices.csv --price-date 2026-11-03
# Fully offline, including the model config:
gpu-poor fixture/llama3-8b --config tests/fixtures/llama3-8b.json --json
python app.py
```

The example Llama 3 fixture is a hand-checkable config, not a model download.
A gated Hub model still needs access to its config; use a public ungated model or
a local config. The library uses the Hub client's normal optional `HF_TOKEN`
handling; no token is required for public examples. Never paste tokens into code.

Default CLI: context 4,096 resident tokens per sequence, 90% GPU memory budget,
BF16 KV cache, single GPU. Prices are a **3 November 2026 historical research
snapshot**, not current prices and **not the held/published Hourly index**. Missing
prices fail explicitly; use `--no-prices` for memory/speed-only output.

## Scope

v0.1 supports dense causal **Llama, Mistral and Qwen2** configs, including GQA and
MQA. Qwen2.5 uses the Qwen2 config layout. Other ids fail clearly; accepting any
Hub id does not mean that every architecture is supported. No MoE, MLA,
multimodal, encoder-decoder, recurrent, hybrid, or pre-quantized layouts.

`huggingface_hub.hf_hub_download` fetches **config.json only**. `AutoConfig.for_model`
uses the installed Transformers registry, never `trust_remote_code=True`.
Parameter counts come from an explicit dense architecture formula; selecting
`--parameter-source safetensors` uses `HfApi.get_safetensors_metadata`, which reads
headers, not tensor data. All tensors in those headers are counted; unusual
buffers/layouts need separate validation. Header access failures surface as errors.
A Hub client without that method emits a warning and keeps the config formula.
Resolved Hub config revisions are recorded for reproducibility.
`requirements-lock.txt` records the tested Linux/Python 3.13 environment, not a
cross-platform resolver lock.

## Method

Use SI GB = 10^9 bytes, GB/s = 10^9 bytes/s. Memory shown by GPU runtime tools may
use GiB; this first version treats the vendor-labelled capacity as SI GB.

Let P be dense parameter count, L layers, Hq query heads, Hkv KV heads, D head
dimension, C resident context per sequence, B simultaneous sequences, W weight
bytes, K KV bytes per token, BW peak memory bandwidth, and F peak dense BF16 FLOPs/s.

### 1. Parameters and memory

For Llama/Mistral gated MLPs with hidden size H, intermediate size I, vocab V:

```text
per_layer = H*(2*Hq*D + 2*Hkv*D) + 3*H*I + 2*H
P = L*per_layer + V*H*(1 if tied_embeddings else 2) + H
W = ceil(P * bytes_per_weight)
K = 2 * L * Hkv * D * bytes_per_KV_element
KV_total = B * C * K
max_context_per_sequence = min(config_context_limit, floor((f*GPU_bytes-W)/(B*K)))
```

Bias parameters are included when configured; Qwen2 Q/K/V biases are included.
Weight storage is BF16=2, FP8=1, INT4=0.5 bytes/parameter. KV precision is separate
(BF16/FP16=2, FP8=1), **not automatically INT4 because weights are INT4**.
INT4 nibble packing rounds total storage up. Scales/zero points, higher-precision
embeddings and quantization metadata are omitted, so quantized memory is optimistic.
`f=0.9` reserves 10% as a user-adjustable allowance, not a validated engine budget.
`max_total_context_tokens` is the total available across the selected batch after
config context caps; it is not the maximum sequence count at an arbitrary context.
If weights or requested KV do not fit, throughput and cost are `null`/`n/a`.

Hand check: Llama 3 8B has L=32, Hkv=8, D=128. BF16 KV:
`2 * 32 * 8 * 128 * 2 = 131,072 bytes/token` (128 KiB).
Full MHA at 32 KV heads would use four times as much. GQA shares K/V across groups
of query heads, reducing cache capacity and attention traffic.

### 2. Decode roofline

Each synchronous decode step generates B output tokens. The idealized step reads
the weights once, plus every sequence's current KV once:

```text
traffic_per_step = W + B*C*K
memory_ceiling_aggregate = B*BW / (W+B*C*K)
memory_ceiling_per_sequence = BW / (W+B*C*K)
FLOPs_per_output_token ~= 2*P + 4*L*Hq*D*C
compute_ceiling_aggregate = F / FLOPs_per_output_token
aggregate_tokens_per_second = min(memory_ceiling_aggregate, compute_ceiling_aggregate)
per_sequence_tokens_per_second = aggregate_tokens_per_second / B
```

Arithmetic intensity is `B*FLOPs_per_output_token/(W+B*C*K)`; the hardware ridge is
`F/BW`. The algebraic batch crossover is `F*W/(FLOPs_per_output_token*BW-F*C*K)`
when the denominator is positive; otherwise there is no finite crossover. It may
lie outside the batch sizes that fit in memory. Attention FLOPs use Hq, whereas
KV bytes use Hkv. The `2P` term is a standard coarse dense-linear approximation.

**Quantization changes storage, not the compute peak in this version.** We use
BF16 dense compute for every storage choice, as if ideal weight dequantization
preceded matmul. This is not an FP8 or INT4 kernel simulator, and does not imply
that every GPU/engine supports such a path. FP8 KV is also an explicit idealized
storage choice, not a compatibility claim. Dense rates never use sparse TFLOPS.

### 3. Price and cost ranges

```text
USD_per_million_output_tokens = USD_per_GPU_hour * 1,000,000 / (3600 * total_tokens_s)
```

Compute this for **low, median and high rental prices**. These are price ranges,
not performance confidence intervals. The bundled range is min/median/max of
provider medians on one date, not p10/p90 and not an official Hourly fixing.
`is_example` is always explicit in the schema; example inputs are labelled EXAMPLE.
See [data provenance and schema](data/README.md). Parquet input uses HF Datasets.

## Example output

See `examples/` for three actual config-only CLI runs and machine-readable results.
Qwen2.5-7B-Instruct, H100 SXM, BF16 weights and KV, context 4,096, 90% memory budget. Historical rental range $2.25 / $3.345 / $4.841 per GPU hour.

| Batch | Max context/sequence | Per-sequence tok/s ceiling | Aggregate tok/s ceiling | USD/M output floor: low / median / high |
|---:|---:|---:|---:|---|
| 1 | 32,768 | 216.60 | 216.60 | 2.8855 / 4.2897 / 6.2082 |
| 8 | 32,768 | 195.79 | 1566.31 | 0.3990 / 0.5932 / 0.8585 |
| 32 | 30,936 | 147.27 | 4712.62 | 0.1326 / 0.1972 / 0.2853 |


## GPU sources

| GPU variant | Memory GB | Bandwidth GB/s | Dense BF16 TFLOPS | Vendor source |
|---|---:|---:|---:|---|
| H100 SXM | 80 | 3350 | 989.5 | [NVIDIA datasheet](https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306) |
| H200 SXM | 141 | 4800 | 989.5 | [NVIDIA](https://www.nvidia.com/en-us/data-center/h200/) |
| A100 80GB SXM | 80 | 2039 | 312 | [NVIDIA datasheet](https://resources.nvidia.com/en-us-gpu/nvidia-a100-datashee-1) |
| L40S | 48 | 864 | 362.05 | [NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/) |
| RTX 4090 | 24 | 1008 | 165.2 | [NVIDIA Ada whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) |
| MI300X | 192 | 5300 | 1307.4 | [AMD datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |

Sources checked 10 November 2026. H100/H200 sparse 1,979 TFLOPS divided by two.
RTX uses BF16 with FP32 accumulation, not marketing INT8 TOPS. L40S uses the
explicit dense BF16 figure on NVIDIA's table. Source URLs also live in code.

## Limits: read before using the numbers

- No hardware benchmark has been run. No inference or generated text is executed.
- Prefill, TTFT, prompt cost, KV writes, activations, normalization, softmax,
  sampling, kernel launches, memory fragmentation and the scheduler are omitted.
- Full weight reads and `2P` include embeddings for simplicity; embedding lookup
  traffic differs in an engine. The analytic ceiling is conditional on these
  assumptions, not a strict hardware bound for every implementation.
- Full-context attention/cache is assumed even for Mistral sliding attention;
  a note appears in results. No cache eviction, prefix sharing or paged layout.
- No tensor/pipeline parallelism, CPU offload, interconnect costs or multi-node math.
- No measured utilization, continuous batching, varying request lengths, idle time,
  quality impact, quantization scale overhead or real dequantization costs.
- Rental prices are archived list prices, not booked capacity. Host resources,
  storage, egress, taxes, availability, contracts and service levels are excluded.
- Exact H100 SXM/A100 SXM SKUs are used rather than ambiguous H100/A100 families.
  H200 input says H200 without an explicit SXM/NVL form factor; that match remains
  approximate. Inspect original rows before relying on any comparison.

## Local validation and dataset export

```sh
ruff check .
ruff format --check .
mypy src/gpu_poor
pytest --cov=gpu_poor --cov-report=term-missing
python scripts/build_prices.py data/raw/2026-11-03.json /tmp/hourly-prices.csv
python scripts/export_dataset.py src/gpu_poor/data/hourly-prices.csv data/dataset
```

Tests block sockets and HTTP and use small cached config fixtures. No external service is
needed. Dataset export creates only local Parquet plus a dataset card.
`requirements.txt` and the YAML above prepare a future CPU Gradio Space;
**nothing is deployed**. Local launch binds to 127.0.0.1 with `share=False`.

## Authorship and licensing

Initial local prototype is AI-assisted, prepared for Mehdi's review and hands-on
learning. It is not yet public work, a merged contribution, or a production-tested
engine. See [CONTRIBUTING.md](CONTRIBUTING.md) and [the study guide](docs/STUDY.md).
Code: MIT. Bundled GPU rental price data: CC BY 4.0 with separate attribution.
