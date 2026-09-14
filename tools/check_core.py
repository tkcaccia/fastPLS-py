#!/usr/bin/env python3
"""Check the vendored core against a local fastPLS source checkout."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fastpls", type=Path, help="path to the fastPLS R repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "UPSTREAM_CORE.json").read_text())
    source = args.fastpls / metadata["source_path"]
    vendored = root / "vendor" / "fastpls" / "include" / "fastpls"
    comparison = filecmp.dircmp(source, vendored)
    differences = comparison.left_only + comparison.right_only + comparison.diff_files
    for child in comparison.common_dirs:
        nested = filecmp.dircmp(source / child, vendored / child)
        differences.extend(
            f"{child}/{name}"
            for name in nested.left_only + nested.right_only + nested.diff_files
        )
    if differences:
        print("Core mismatch:")
        for path in sorted(differences):
            print(f"  {path}")
        return 1
    print(f"Core matches fastPLS {metadata['package_version']} at {metadata['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

