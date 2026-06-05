#!/usr/bin/env python3
"""
Normalize a LongMemEval export into ARM's thin format.

Usage (after cloning longmemeval and obtaining data):
  python scripts/convert_longmemeval.py \
    --source /path/to/longmemeval/data \
    --out examples/longmemeval_subset \
    --limit 116
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LongMemEval to ARM format")
    parser.add_argument("--source", type=Path, required=True, help="Upstream LongMemEval data dir")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for ARM loader")
    parser.add_argument("--limit", type=int, default=116, help="Max questions (paper subset)")
    args = parser.parse_args()

    # Hook point: map upstream schema → ARM jsonl + per-session corpus files.
    # This script is intentionally a stub until you point it at your copy of the dataset.
    raise SystemExit(
        "Implement mapping for your LongMemEval checkout.\n"
        "Expected ARM output:\n"
        f"  {args.out}/questions.jsonl\n"
        f"  {args.out}/sessions/<session_id>.json\n"
        "\nSee README section 'Connecting LongMemEval'."
    )


if __name__ == "__main__":
    main()