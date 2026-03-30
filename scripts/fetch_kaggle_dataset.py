"""Download a Kaggle dataset archive (or unzip) into datasets/raw/kaggle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Kaggle dataset by slug (owner/dataset-name).")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. uciml/human-activity-recognition-with-smartphones")
    parser.add_argument("--target-dir", default="datasets/raw/kaggle")
    parser.add_argument("--unzip", action="store_true", help="Unzip archive after download")
    args = parser.parse_args()

    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        print("Missing Kaggle token: ~/.kaggle/kaggle.json")
        print("Create API token from https://www.kaggle.com/settings and place it there with chmod 600.")
        sys.exit(1)

    os.chmod(token_path, 0o600)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kaggle_bin = Path(".venv/bin/kaggle")
    if kaggle_bin.exists():
        cmd = [str(kaggle_bin), "datasets", "download", "-d", args.dataset, "-p", str(target_dir)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(target_dir)]

    if args.unzip:
        cmd.append("--unzip")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
