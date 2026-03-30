"""Download and organize registered datasets for deterministic model training."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List


def load_registry(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets(registry: Dict) -> None:
    for ds in registry.get("datasets", []):
        mode = "manual" if ds.get("manual", True) else "download"
        print(f"{ds.get('id')}: {mode} | {ds.get('type')} | {ds.get('description')}")


def _download(url: str, out_path: Path, chunk_size: int = 1024 * 1024, insecure: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"skip existing: {out_path}")
        return

    print(f"downloading: {url}")
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(url, timeout=30, context=context) as resp, open(out_path, "wb") as f:
        total = resp.length or 0
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100.0
                print(f"  {out_path.name}: {pct:.2f}% ({downloaded}/{total})", end="\r")
    print(f"\ncompleted: {out_path}")


def fetch_dataset(ds: Dict, target_root: Path, include_manual: bool, insecure: bool) -> None:
    ds_id = ds.get("id", "unknown")
    ds_dir = target_root / ds_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    if ds.get("manual", True):
        note = ds.get("landing_page", "")
        if include_manual:
            print(f"manual dataset ({ds_id}): complete registration and download from {note}")
        else:
            print(f"manual dataset skipped ({ds_id}): {note}")
        return

    files: List[Dict] = ds.get("files", [])
    if not files:
        print(f"no files listed for dataset: {ds_id}")
        return

    for item in files:
        url = item.get("url", "")
        name = item.get("name", "")
        if not url or not name:
            print(f"invalid file entry in {ds_id}: {item}")
            continue
        out_path = ds_dir / name
        try:
            _download(url, out_path, insecure=insecure)
        except Exception as exc:  # noqa: BLE001
            print(f"download failed for {url}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch datasets listed in datasets/registry.json")
    parser.add_argument("--registry", default="datasets/registry.json")
    parser.add_argument("--target-dir", default="datasets/raw")
    parser.add_argument("--dataset", default="", help="Specific dataset id to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all listed datasets")
    parser.add_argument("--include-manual", action="store_true", help="Print manual dataset instructions")
    parser.add_argument("--list", action="store_true", help="List dataset registry entries")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for hosts with cert-chain issues")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    datasets = registry.get("datasets", [])

    if args.list:
        list_datasets(registry)
        return

    selected: List[Dict] = []
    if args.dataset:
        selected = [ds for ds in datasets if ds.get("id") == args.dataset]
        if not selected:
            print(f"dataset not found: {args.dataset}")
            sys.exit(1)
    elif args.all:
        selected = datasets
    else:
        print("Choose --dataset <id> or --all. Use --list to view ids.")
        sys.exit(1)

    target_root = Path(args.target_dir)
    for ds in selected:
        fetch_dataset(ds, target_root=target_root, include_manual=args.include_manual, insecure=args.insecure)


if __name__ == "__main__":
    main()
