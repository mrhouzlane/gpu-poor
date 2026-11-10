"""CSV/Parquet price ranges, with explicit dates, units, and provenance."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

from .specs import get_gpu


@dataclass(frozen=True)
class PriceRange:
    gpu: str
    date: str
    low: float
    median: float
    high: float
    currency: str
    kind: str
    source: str
    is_example: bool
    range_method: str
    n_providers: int = 0

    def __post_init__(self) -> None:
        date.fromisoformat(self.date)
        if self.currency != "USD":
            raise ValueError("Only USD prices are supported; convert explicitly before loading")
        if not all(math.isfinite(v) for v in (self.low, self.median, self.high)):
            raise ValueError("Prices must be finite")
        if not 0 <= self.low <= self.median <= self.high:
            raise ValueError("Prices must satisfy 0 <= low <= median <= high")
        if not self.source or not self.range_method:
            raise ValueError("Price source and range_method are required")


def default_price_path() -> Path:
    return Path(str(files("gpu_poor").joinpath("data/hourly-prices.csv")))


def _example(value: Any) -> bool:
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    raise ValueError("is_example must be true or false, never inferred")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ValueError("Parquet needs pip install 'gpu-poor[data]'") from exc
        return list(Dataset.from_parquet(str(path)))
    raise ValueError("Prices must be .csv or .parquet")


def load_prices(path: str | Path | None, gpu: str, *, day: str | None = None) -> PriceRange:
    """Load latest row for exact GPU key, or exact day. Never interpolate missing days.

    Canonical adapter schema is documented in data/README.md. Legacy Hourly research
    bands (chip,p10,median,p90) are intentionally not silently treated as min/max.
    """
    path = Path(path) if path is not None else default_price_path()
    target = get_gpu(gpu).name
    if day is not None:
        date.fromisoformat(day)
    rows = _read_rows(path)
    matches = [
        r
        for r in rows
        if get_gpu(str(r["gpu"])).name == target and (day is None or str(r["date"]) == day)
    ]
    if not matches:
        raise ValueError(f"No price range for {target}" + (f" on {day}" if day else ""))
    latest = max(str(r["date"]) for r in matches)
    matches = [r for r in matches if str(r["date"]) == latest]
    if len(matches) != 1:
        raise ValueError("Multiple price rows for the same GPU/day; select a pricing tier first")
    row = matches[0]
    return PriceRange(
        str(row["gpu"]),
        str(row["date"]),
        float(row["low"]),
        float(row["median"]),
        float(row["high"]),
        str(row["currency"]),
        str(row["kind"]),
        str(row["source"]),
        _example(row["is_example"]),
        str(row["range_method"]),
        int(row.get("n_providers", 0)),
    )


def range_from_snapshot(snapshot: dict[str, Any], gpu_key: str) -> PriceRange | None:
    """Adapter for Hourly's archived gpurentalprices JSON observation format.

    Exact SKU only; fresh/ok providers; on demand or secure; single-GPU offers only.
    Take each provider's median, then min/median/max across providers.
    """
    from statistics import median

    skus = {
        "H100": "h100-sxm",
        "H200": "h200",
        "A100": "a100-sxm-80",
        "L40S": "l40s",
        "RTX4090": "rtx-4090",
        "MI300X": "mi300x",
    }
    if gpu_key not in skus:
        raise ValueError(f"No snapshot mapping for {gpu_key}")
    groups: dict[str, list[float]] = {}
    sources: set[str] = set()
    for row in snapshot["offers"]:
        provider = str(row["provider"])
        status = snapshot.get("providers", {}).get(provider, {})
        if row["gpu"] != skus[gpu_key] or row["kind"] not in {"on-demand", "secure"}:
            continue
        if not status.get("ok") or status.get("stale", True) or row.get("min_gpus", 1) > 1:
            continue
        if str(row.get("fetched_at", ""))[:10] != snapshot["date"]:
            continue
        value = float(row["usd_hr"])
        if not math.isfinite(value) or value <= 0 or not row.get("source_url"):
            continue
        groups.setdefault(provider, []).append(value)
        sources.add(str(row["source_url"]))
    if not groups:
        return None
    values = [float(median(v)) for v in groups.values()]
    return PriceRange(
        gpu_key,
        snapshot["date"],
        min(values),
        float(median(values)),
        max(values),
        "USD",
        "on demand",
        json.dumps(sorted(sources)),
        False,
        "min/median/max of provider medians; exact SKU; single GPU; fresh only",
        len(values),
    )
