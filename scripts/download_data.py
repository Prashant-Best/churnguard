"""Download the IBM Telco Customer Churn dataset (7,043 customers) into data/raw/.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    url = config["data"]["url"]
    out_path = ROOT / config["data"]["raw_path"]

    if out_path.exists():
        print(f"Dataset already present at {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    rows = response.text.count("\n") - 1
    print(f"Saved {rows:,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
