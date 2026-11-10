import csv
import json
from dataclasses import asdict, replace

import pytest

from gpu_poor.prices import PriceRange, load_prices, range_from_snapshot


@pytest.fixture
def price():
    return PriceRange(
        "H100",
        "2026-11-03",
        1,
        2,
        3,
        "USD",
        "on demand",
        "https://example.org",
        True,
        "min/median/max",
        3,
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_csv_dates_no_interpolation(tmp_path, price):
    path = tmp_path / "prices.csv"
    write_csv(path, [asdict(replace(price, date="2026-11-02", median=1.5)), asdict(price)])
    assert load_prices(path, "H100 SXM").date == "2026-11-03"
    assert load_prices(path, "H100", day="2026-11-02").median == 1.5
    with pytest.raises(ValueError, match="No price"):
        load_prices(path, "H100", day="2026-11-01")


def test_parquet_roundtrip(tmp_path, price):
    datasets = pytest.importorskip("datasets")
    path = tmp_path / "prices.parquet"
    datasets.Dataset.from_list([asdict(price)]).to_parquet(str(path))
    assert load_prices(path, "H100") == price


def test_duplicate_rejected(tmp_path, price):
    path = tmp_path / "prices.csv"
    write_csv(path, [asdict(price), asdict(price)])
    with pytest.raises(ValueError, match="Multiple"):
        load_prices(path, "H100")


@pytest.mark.parametrize(
    "change",
    [
        {"low": -1},
        {"low": 3},
        {"high": 1},
        {"median": float("nan")},
        {"currency": "EUR"},
        {"source": ""},
        {"date": "today"},
    ],
)
def test_bad_prices(price, change):
    with pytest.raises(ValueError):
        replace(price, **change)


def test_example_flag_must_be_explicit(tmp_path, price):
    path = tmp_path / "prices.csv"
    write_csv(path, [{**asdict(price), "is_example": "maybe"}])
    with pytest.raises(ValueError, match="is_example"):
        load_prices(path, "H100")


def test_wrong_format(tmp_path):
    with pytest.raises(ValueError, match="csv or .parquet"):
        load_prices(tmp_path / "p.json", "H100")


def test_real_bundled_data_has_dates_sources_and_labels():
    p = load_prices(None, "H100")
    assert p.date == "2026-11-03"
    assert not p.is_example
    assert p.n_providers > 0
    assert all(s.startswith("https://") for s in json.loads(p.source))


def test_snapshot_adapter_filters():
    def offer(provider, price, **kw):
        return dict(
            provider=provider,
            usd_hr=price,
            gpu="h100-sxm",
            kind="on-demand",
            fetched_at="2026-11-03T08:00:00Z",
            source_url="https://example.org",
            **kw,
        )

    snapshot = {
        "date": "2026-11-03",
        "providers": {
            "a": {"ok": True, "stale": False},
            "b": {"ok": True, "stale": False},
            "stale": {"ok": True, "stale": True},
        },
        "offers": [
            offer("a", 1),
            offer("a", 3),
            offer("b", 6),
            offer("stale", 100),
            offer("missing", 100),
            offer("a", 100, min_gpus=8),
        ],
    }
    snapshot["offers"] += [
        {**offer("a", 0.1), "kind": "spot"},
        {**offer("a", 100), "gpu": "h100-pcie"},
        {**offer("a", 100), "fetched_at": "2026-11-02T08:00:00Z"},
        {**offer("a", 0), "usd_hr": float("nan")},
        {**offer("a", 100), "source_url": ""},
    ]
    band = range_from_snapshot(snapshot, "H100")
    assert (band.low, band.median, band.high, band.n_providers) == (2, 4, 6, 2)
    assert range_from_snapshot(snapshot, "L40S") is None
    with pytest.raises(ValueError):
        range_from_snapshot(snapshot, "B200")
