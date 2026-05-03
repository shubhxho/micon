"""SHA-256 file-integrity manifest: build, write, and verify.

CLI usage:
    python -m src.integrity.checksums build --root <path> --manifest <path>
    python -m src.integrity.checksums verify --root <path> --manifest <path>
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_CHUNK = 1 << 20  # 1 MB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_checksum_manifest(
    root: Path,
    exclude_patterns: Sequence[str] = ("*.tar", "*.dcm"),
) -> dict:
    """Walk *root* and return a SHA-256 manifest dict.

    Returns:
        {
            "version": 1,
            "generated_at": "<ISO-8601>",
            "algorithm": "sha256",
            "files": {"rel/path": "<hex-digest>", ...},
        }
    """
    root = root.resolve()
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude_patterns):
            continue
        files[rel] = _sha256(path)
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "files": files,
    }


def write_checksum_manifest(root: Path, out_path: Path) -> None:
    """Build manifest for *root* and write JSON to *out_path*."""
    manifest = build_checksum_manifest(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_checksum_manifest(
    manifest_path: Path, root: Path
) -> tuple[int, int, list[str]]:
    """Verify files in *root* against a manifest.

    Returns:
        (n_ok, n_mismatched, mismatched_paths)
        Missing files are included in mismatched_paths with suffix " <missing>".
    """
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded: dict[str, str] = manifest["files"]
    n_ok = 0
    bad: list[str] = []
    for rel, expected in recorded.items():
        path = root / rel
        if not path.exists():
            bad.append(f"{rel} <missing>")
            continue
        actual = _sha256(path)
        if actual == expected:
            n_ok += 1
        else:
            bad.append(rel)
    return n_ok, len(bad), bad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.integrity.checksums")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--root", required=True, type=Path)
    b.add_argument("--manifest", required=True, type=Path)

    v = sub.add_parser("verify")
    v.add_argument("--root", required=True, type=Path)
    v.add_argument("--manifest", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "build":
        write_checksum_manifest(args.root, args.manifest)
        print(f"Manifest written to {args.manifest}")
    else:
        n_ok, n_bad, bad = verify_checksum_manifest(args.manifest, args.root)
        if bad:
            for p in bad:
                print(f"MISMATCH: {p}", file=sys.stderr)
            print(f"{n_ok} OK, {n_bad} failed")
            sys.exit(1)
        else:
            print(f"All {n_ok} files OK")


if __name__ == "__main__":
    _main()
