---
license: cc-by-4.0
language:
- en
size_categories:
- n<1K
pretty_name: GPU rental research price ranges
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
---
# GPU rental research price ranges

Historical research inputs, not the Hourly published index. Not current quotes.
GPU rental price data by [gpurentalprices.com](https://gpurentalprices.com), CC BY 4.0.
Derived from the archived 3 November 2026 snapshot. Exact SKU; fresh/ok providers;
on demand or secure; excludes offers requiring multiple GPUs. Take the median
within each provider, then min/median/max across providers. These are not p10/p90.
The `source` column is a JSON list of original provider URLs. See the code project's
data/README.md for original archive path, snapshot SHA256, and limitations.

## Fields
`gpu`, `date`, `low`, `median`, `high`, `currency` (USD per GPU hour), `kind`,
`source`, `is_example`, `range_method`, `n_providers`.

No availability, performance, service-level, or hardware-form-factor equivalence
is implied. H200 rows do not explicitly identify SXM versus NVL; the app uses SXM.
No storage, CPU, RAM, network, egress, tax, or utilization adjustment is included.

## Local use
```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files="train.parquet", split="train")
```
