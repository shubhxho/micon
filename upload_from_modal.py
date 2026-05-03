"""Upload dataset from Modal Volume directly to Hugging Face.

No local download needed — reads from Modal Volume, uploads to HF.
Run with: modal run upload_from_modal.py --study redacted

This runs on a Modal container with the volume mounted, so the upload
uses Modal's bandwidth (fast) instead of your local connection.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

VOLUME_NAME = "micom-data"
MOUNT_POINT = PurePosixPath("/vol")
OUTPUT_DIR = MOUNT_POINT / "output"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.25", "polars>=1.17"
)

app = modal.App("micom-hf-upload", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,  # 2 hours for large uploads
    memory=4096,
    cpu=2.0,
)
def upload_to_hf(study: str, repo_id: str = "", private: bool = False):
    """Upload a study's output from Modal Volume to Hugging Face.

    Runs on Modal's infrastructure — uses cloud bandwidth, not yours.
    For 200GB+ datasets this is much faster than downloading locally first.
    """
    from huggingface_hub import HfApi, create_repo

    volume.reload()

    out_dir = Path(str(OUTPUT_DIR)) / study
    if not out_dir.exists():
        print(f"Error: {out_dir} not found on volume")
        print("Available studies:")
        try:
            for entry in volume.listdir("output"):
                print(f"  {PurePosixPath(entry.path).name}")
        except Exception:
            print("  (none)")
        return {"error": f"Study '{study}' not found"}

    # Count files — skip HTML reports and oversized files
    MAX_FILE_MB = 500
    SKIP_EXTENSIONS = {".html", ".htm"}
    all_files = []
    skipped = []
    for f in out_dir.rglob("*"):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() in SKIP_EXTENSIONS:
            skipped.append((f.name, f.stat().st_size / 1024**3, "html"))
            continue
        sz = f.stat().st_size
        if sz > MAX_FILE_MB * 1024 * 1024:
            skipped.append((f.name, sz / 1024**3, "oversized"))
        else:
            all_files.append(f)
    if skipped:
        print(f"Skipping {len(skipped)} files:")
        for item in skipped[:10]:
            print(f"  {item[0]}: {item[1]:.1f}GB ({item[2]})")
    total_bytes = sum(f.stat().st_size for f in all_files)
    by_ext: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower() or "(none)"
        by_ext[ext] = by_ext.get(ext, 0) + 1

    print(f"Study: {study}")
    print(f"  {len(all_files)} files, {total_bytes / 1024**3:.1f} GB")
    print(
        f"  Types: {', '.join(f'{ext}:{n}' for ext, n in sorted(by_ext.items(), key=lambda x: -x[1])[:10])}"
    )

    # Auth — uses HF_TOKEN env var (set via Modal secret or env)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or ""
        except Exception:
            pass
    if not token:
        return {
            "error": "No HF_TOKEN. Create Modal secret: modal secret create hf-secret HF_TOKEN=hf_..."
        }

    api = HfApi(token=token)

    if not repo_id:
        user = api.whoami()["name"]
        safe = study.lower().replace(" ", "-").replace("/", "-")
        repo_id = f"{user}/{safe}"

    print(f"\nUploading to: https://huggingface.co/datasets/{repo_id}")
    print(f"  Visibility: {'private' if private else 'public'}")

    # Create repo
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    # Upload using upload_large_folder — handles 14K+ files, auto-resumes on failure
    t0 = time.time()
    print(f"\nUploading {len(all_files)} files ({total_bytes / 1024**3:.1f} GB)...")

    # Copy to /tmp first — volume may be full and HF needs to write cache files
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp())
    print("  Copying to /tmp staging area...")
    for f in all_files:
        rel = f.relative_to(out_dir)
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    print(f"  Staged {len(all_files)} files in {staging}")

    # Set HF cache to /tmp too
    os.environ["HF_HOME"] = str(staging / ".hf_cache")

    print("  Uploading via upload_folder...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {study} ({len(all_files)} files, {total_bytes / 1024**3:.1f} GB)",
    )

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)

    elapsed = time.time() - t0
    rate = total_bytes / max(elapsed, 1) / 1024**2
    url = f"https://huggingface.co/datasets/{repo_id}"

    print(f"\nDone in {elapsed:.0f}s ({rate:.1f} MB/s)")
    print(f"Dataset: {url}")

    return {
        "url": url,
        "files": len(all_files),
        "size_gb": round(total_bytes / 1024**3, 1),
        "time_s": round(elapsed),
    }


@app.local_entrypoint()
def main(
    study: str = "redacted",
    repo: str = "",
    private: bool = False,
):
    """Upload study output from Modal Volume to Hugging Face.

    Uses .spawn() so the upload keeps running even if your laptop disconnects.
    Check progress at the Modal dashboard URL printed above.
    """
    call = upload_to_hf.spawn(study, repo_id=repo, private=private)
    print("Upload spawned — runs on Modal even if you disconnect.")
    print("Check progress at: https://modal.com/apps/shubhxho/main")
    print(f"Call ID: {call.object_id}")

    # Wait for result if we're still connected
    try:
        result = call.get(timeout=7200)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Disconnected locally — upload continues on Modal.")
