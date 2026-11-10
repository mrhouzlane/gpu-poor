# Price provenance and schema

**Real archived research data, not EXAMPLE prices. Not the published Hourly index.**

Source: a 3 November 2026 snapshot of the public
[gpurentalprices/gpu-rental-prices](https://huggingface.co/datasets/gpurentalprices/gpu-rental-prices) dataset (CC BY 4.0).
The older aggregated bands mix H100/A100 variants and p10/p90; this project does
**not** import those bands as exact-SKU min/max.

The observation snapshot is copied unchanged to `data/raw/2026-11-03.json`.
SHA256: `ed3ee8522109abf0f4ec2036de5227bb01a736db53db5ff7ef6a5859f4ca97d9`.
Snapshot generated **3 November 2026, 09:48:29 Europe/Paris**. Prices are historical,
not refreshed by this tool; original row timestamps and URLs remain in the JSON.

Upstream: [gpurentalprices/gpu-rental-prices](https://huggingface.co/datasets/gpurentalprices/gpu-rental-prices),
`data/snapshots/2026-11-03.json`.
The upstream dataset README was checked 10 November 2026 and declares CC BY 4.0.
Its current rolling archive may omit old days; the local original is retained.

**Attribution:** GPU rental price data by [gpurentalprices.com](https://gpurentalprices.com),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The MIT code license does not relicense these data. This project made the
aggregation and filtering below; that is a modification of the raw observations.

## Conservative adapter

- Match `h100-sxm`, `a100-sxm-80`, `h200`, `l40s`, `rtx-4090`, `mi300x` exactly.
  Do not silently combine ambiguous `h100`/`a100` family labels, PCIe or NVL.
- H200 is not explicitly SXM in the input. Using the H200 SXM spec is an
  approximation; it is disclosed rather than silently inferring a form factor.
- Include only provider `ok=true`, `stale=false` rows fetched on the snapshot day.
- Include raw tier `on-demand` or `secure`; label combined tier `on demand`.
- Exclude rows with explicit `min_gpus > 1`; absent `min_gpus` is assumed one.
  A missing minimum is not proof that a provider sells a single GPU.
- Require a finite positive USD/GPU hour and a source URL.
- Take median offer within each provider, then min / median / max across providers.
  Each provider gets one observation. These are **not** p10/p90.

Bundled canonical table: `src/gpu_poor/data/hourly-prices.csv`, six rows.
It is a research snapshot derived anew from a public dataset already used in
Hourly research. It is **not** a production index fixing, live website value,
transaction, rental availability check, or a verified current provider quote.
The held website response contains no prices and is not used. No holds or data
pipelines were changed. No remote data were written.

## CSV / Parquet contract

| Field | Meaning |
|---|---|
| gpu | H100, H200, A100, L40S, RTX4090, or MI300X |
| date | YYYY-MM-DD; latest row per GPU chosen unless --price-date is supplied |
| low, median, high | finite ascending USD per GPU hour |
| currency | USD only; no implicit conversion |
| kind | human-readable tier, e.g. on demand |
| source | provenance text; bundled table uses JSON list of provider URLs |
| is_example | explicit true or false; never inferred |
| range_method | exact meaning of low/median/high |
| n_providers | optional count |

A duplicate GPU/day row fails to avoid silently combining spot, reserved and
on demand. Missing dates are not filled. To adapt another Hourly export, map
its fields explicitly into this contract and preserve provenance; never treat
a historical index level as an available rental quote. CSV uses the standard
library; Parquet input uses `datasets.Dataset.from_parquet`.

Regenerate locally:
```sh
python scripts/build_prices.py data/raw/2026-11-03.json src/gpu_poor/data/hourly-prices.csv
python scripts/export_dataset.py src/gpu_poor/data/hourly-prices.csv data/dataset
```

Host resources, region, storage, tax, uptime and availability vary across offers.
The source collector's normalization has not been independently audited here.
