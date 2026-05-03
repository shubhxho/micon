#!/usr/bin/env python3
"""Upload the full output/ folder to Hugging Face as a dataset.

Usage:
  python3 upload_hf.py                           # auto-detect, public
  python3 upload_hf.py --repo shubhxho/micom-mri  # specific repo
  python3 upload_hf.py --dir ./output --private   # custom dir, private

Requires: pip install huggingface_hub
Auth: run `huggingface-cli login` first, or set HF_TOKEN env var.
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument("--dir", default="output", help="Directory to upload (default: output/)")
    parser.add_argument("--repo", default="", help="HF repo id (default: auto from username)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--study", default="", help="Study name for the dataset card")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    if not out_dir.exists():
        print(f"Error: {out_dir} not found")
        sys.exit(1)

    # Count what we're uploading
    all_files = [f for f in out_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
    total_bytes = sum(f.stat().st_size for f in all_files)
    by_ext: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower() or "(none)"
        by_ext[ext] = by_ext.get(ext, 0) + 1

    print(f"Dataset: {out_dir}")
    print(f"  {len(all_files)} files, {total_bytes / 1024**3:.1f} GB")
    print(
        f"  Types: {', '.join(f'{ext}:{n}' for ext, n in sorted(by_ext.items(), key=lambda x: -x[1])[:10])}"
    )
    print()

    # Auth
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or ""
        except Exception:
            pass
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("Error: No HF token found. Run: huggingface-cli login")
        sys.exit(1)

    api = HfApi(token=token)
    user = api.whoami()["name"]

    repo_id = args.repo
    if not repo_id:
        study = args.study or out_dir.name
        safe = study.lower().replace(" ", "-").replace("/", "-")
        repo_id = f"{user}/{safe}"

    print(f"Uploading to: https://huggingface.co/datasets/{repo_id}")
    print(f"  Visibility: {'private' if args.private else 'public'}")
    print()

    # Create repo
    create_repo(repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)

    # Upload entire folder in one commit
    t0 = time.time()
    print(f"Uploading {len(all_files)} files ({total_bytes / 1024**3:.1f} GB)...")
    print("  This may take a while for large datasets. Progress shown by HF client.")
    print()

    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {args.study or out_dir.name} dataset ({len(all_files)} files, {total_bytes / 1024**3:.1f} GB)",
    )

    elapsed = time.time() - t0
    rate = total_bytes / max(elapsed, 1) / 1024**2
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nDone in {elapsed:.0f}s ({rate:.1f} MB/s)")
    print(f"Dataset: {url}")


if __name__ == "__main__":
    main()
