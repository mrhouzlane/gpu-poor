"""Convert an archived Hourly research observation JSON into exact-SKU price bands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from gpu_poor.prices import range_from_snapshot
from gpu_poor.specs import GPUS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    rows = [
        asdict(band) for key in GPUS if (band := range_from_snapshot(snapshot, key)) is not None
    ]
    if not rows:
        raise SystemExit("No eligible prices; no file written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} exact-SKU price ranges to {args.output}")
    print("snapshot sha256:", hashlib.sha256(args.snapshot.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
