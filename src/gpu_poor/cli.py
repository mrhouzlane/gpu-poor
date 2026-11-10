"""Command-line interface; JSON output is suitable for further analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from huggingface_hub.errors import HfHubHTTPError

from .estimate import estimate
from .models import load_model
from .prices import load_prices
from .specs import GPUS, get_gpu

POLICY = (
    "THEORETICAL UPPER BOUNDS under idealized assumptions, NOT measured throughput. "
    "Costs are optimistic floors at those rates, not a serving quote. "
    "Quantization is weight storage only; BF16 compute roofline, BF16 KV by default."
)


def batch_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
        if not values or any(v < 1 for v in values):
            raise ValueError
        return values
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch must be positive integers, e.g. 1,8,32") from exc


def format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "batch | fits | max ctx/seq | tok/s/seq ceiling | total tok/s ceiling | "
        "$/M out low / median / high floor"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        per, total = r["tokens_s_per_sequence"], r["aggregate_tokens_s"]
        costs = [r[f"usd_per_million_{x}"] for x in ("low", "median", "high")]
        lines.append(
            f"{r['batch']:5} | {'yes' if r['fits'] else 'NO ':4} | "
            f"{r['max_context_per_sequence']:11,} | "
            f"{f'{per:.2f}' if per is not None else 'n/a':>17} | "
            f"{f'{total:.2f}' if total is not None else 'n/a':>19} | "
            + " / ".join(f"{v:.4f}" if v is not None else "n/a" for v in costs)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Config-only decode rooflines and rental costs")
    parser.add_argument("model", help="HF Hub model id; config.json only, no model weights")
    parser.add_argument("--gpu", default="H100", help=f"GPU: {', '.join(GPUS)}")
    parser.add_argument("--dtype", choices=["bf16", "fp8", "int4"], default="bf16")
    parser.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8"], default="bf16")
    parser.add_argument("--batch", type=batch_list, default=[1, 8, 32])
    parser.add_argument("--context", type=int, default=4096, help="Resident tokens per sequence")
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument(
        "--prices", help="CSV or Parquet; default is bundled dated research snapshot"
    )
    parser.add_argument("--price-date", help="Exact YYYY-MM-DD; default latest row in file")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--parameter-source", choices=["config", "safetensors"], default="config")
    parser.add_argument("--config", help="Local config.json instead of a Hub request")
    parser.add_argument("--offline", action="store_true", help="Only use locally cached config")
    parser.add_argument("--no-prices", action="store_true", help="Estimate memory/speed only")
    parser.add_argument("--json", action="store_true", help="Print structured output")
    args = parser.parse_args(argv)
    try:
        gpu = get_gpu(args.gpu)
        model = load_model(
            args.model,
            revision=args.revision,
            config_path=args.config,
            parameter_source=args.parameter_source,
            local_files_only=args.offline,
        )
        price = None if args.no_prices else load_prices(args.prices, args.gpu, day=args.price_date)
        rows = [
            asdict(
                estimate(
                    model,
                    gpu,
                    dtype=args.dtype,
                    kv_dtype=args.kv_dtype,
                    batch=b,
                    context=args.context,
                    memory_fraction=args.memory_fraction,
                    prices=price,
                )
            )
            for b in args.batch
        ]
        payload = {
            "notice": POLICY,
            "model": asdict(model),
            "gpu": asdict(gpu),
            "weight_dtype": args.dtype,
            "kv_dtype": args.kv_dtype,
            "memory_fraction": args.memory_fraction,
            "prices": asdict(price) if price else None,
            "estimates": rows,
        }
        if args.json:
            print(json.dumps(payload, indent=2, allow_nan=False))
        else:
            print(POLICY)
            print(f"{model.model_id}: {model.parameters:,} parameters ({model.parameter_source})")
            print(
                f"{gpu.name}; {args.dtype} weights; {args.kv_dtype} KV; "
                f"{rows[0]['kv_bytes_per_token']:,} KV bytes/token; context={args.context:,}"
            )
            if price:
                label = (
                    "EXAMPLE" if price.is_example else "archived research data, NOT a live index"
                )
                print(
                    f"Prices: {label}; {price.date}; {price.kind}; USD/GPU hour "
                    f"{price.low:g} / {price.median:g} / {price.high:g}"
                )
            if model.sliding_window:
                print("Note: full-context KV used; sliding attention eviction is not modeled.")
            print(format_table(rows))
        return 0
    except (ValueError, KeyError, OSError, HfHubHTTPError) as exc:
        print(f"gpu-poor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
