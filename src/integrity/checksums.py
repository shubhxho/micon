"""SHA-256 file-integrity manifest: build, write, and verify.

CLI usage:
    python -m src.integrity.checksums build --root <path> --manifest <path>
    python -m src.integrity.checksums verify --root <path> --manifest <path>
"""

from __future__ import annotations

import fnmatch
import hashlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.io.msgspec_io import dumps_bytes, loads

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
        files[rel] = sha256_file(path)
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "algorithm": "sha256",
        "files": files,
    }


def write_checksum_manifest(root: Path, out_path: Path) -> None:
    """Build manifest for *root* and write JSON to *out_path*."""
    manifest = build_checksum_manifest(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(dumps_bytes(manifest, indent=2))


def verify_checksum_manifest(manifest_path: Path, root: Path) -> tuple[int, int, list[str]]:
    """Verify files in *root* against a manifest.

    Returns:
        (n_ok, n_mismatched, mismatched_paths)
        Missing files are included in mismatched_paths with suffix " <missing>".
    """
    root = root.resolve()
    manifest = loads(manifest_path.read_bytes())
    recorded: dict[str, str] = manifest["files"]
    n_ok = 0
    bad: list[str] = []
    for rel, expected in recorded.items():
        path = root / rel
        if not path.exists():
            bad.append(f"{rel} <missing>")
            continue
        actual = sha256_file(path)
        if actual == expected:
            n_ok += 1
        else:
            bad.append(rel)
    return n_ok, len(bad), bad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path* read in 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# Backwards-compatible private alias (kept for any internal callers).
_sha256 = sha256_file


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_cmd(
    root: Path,
    manifest: Path,
) -> None:  # pragma: no cover
    """Build a manifest of SHA-256 hashes for every file under ROOT."""
    write_checksum_manifest(root, manifest)
    print(f"Manifest written to {manifest}")


def _verify_cmd(
    root: Path,
    manifest: Path,
) -> None:  # pragma: no cover
    """Verify ROOT against an existing checksum manifest."""
    n_ok, n_bad, bad = verify_checksum_manifest(manifest, root)
    if bad:
        for p in bad:
            print(f"MISMATCH: {p}", file=sys.stderr)
        print(f"{n_ok} OK, {n_bad} failed")
        sys.exit(1)
    else:
        print(f"All {n_ok} files OK")


def _main() -> None:  # pragma: no cover
    from cyclopts import App

    cli = App(name="checksums", help="SHA-256 file-integrity manifests.")
    cli.command(_build_cmd, name="build")
    cli.command(_verify_cmd, name="verify")
    cli()


if __name__ == "__main__":  # pragma: no cover
    _main()
