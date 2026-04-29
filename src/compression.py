"""Compression utilities — gzip, zlib, lzma (stdlib only).

Three algorithms with different tradeoffs:
  gzip  — universal compatibility, good ratio, moderate speed
  zlib  — same deflate algorithm as gzip, no file headers (smaller)
  lzma  — best ratio but slower; use preset=1 for large files
"""

from __future__ import annotations

import gzip
import lzma
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def gzip_bytes(data: bytes, level: int = 6) -> bytes:
    """Compress bytes with gzip. Default level 6 balances speed/ratio."""
    return gzip.compress(data, compresslevel=level)


def zlib_bytes(data: bytes, level: int = 6) -> bytes:
    """Compress bytes with zlib (deflate). Same algorithm as gzip, no headers."""
    return zlib.compress(data, level=level)


def lzma_bytes(data: bytes, preset: int = 1) -> bytes:
    """Compress bytes with LZMA/XZ. Best ratio. Preset 1 for speed on large data."""
    return lzma.compress(data, preset=preset)


def write_gzip(path: Path, text: str) -> Path:
    """Write text to a .gz file. Returns the .gz path."""
    gz_path = path.parent / (path.name + ".gz")
    with gzip.open(gz_path, "wt", compresslevel=6, encoding="utf-8") as f:
        f.write(text)
    return gz_path


def write_lzma(path: Path, text: str) -> Path:
    """Write text to a .xz file. Returns the .xz path."""
    xz_path = path.parent / (path.name + ".xz")
    with lzma.open(xz_path, "wt", preset=1, encoding="utf-8") as f:
        f.write(text)
    return xz_path


def compress_file_multi(path: Path) -> dict[str, tuple[Path, int]]:
    """Compress a file with gzip, zlib, and lzma in parallel."""
    data = path.read_bytes()

    with ThreadPoolExecutor(max_workers=3) as pool:
        gz_fut = pool.submit(_write_compressed, path, data, "gz")
        zl_fut = pool.submit(_write_compressed, path, data, "zlib")
        xz_fut = pool.submit(_write_compressed, path, data, "xz")

        return {
            "gz": gz_fut.result(),
            "zlib": zl_fut.result(),
            "xz": xz_fut.result(),
        }


def _write_compressed(orig_path: Path, data: bytes, fmt: str) -> tuple[Path, int]:
    """Write compressed version of data to a new file."""
    if fmt == "gz":
        out = orig_path.parent / (orig_path.name + ".gz")
        compressed = gzip.compress(data, compresslevel=6)
    elif fmt == "zlib":
        out = orig_path.parent / (orig_path.name + ".zlib")
        compressed = zlib.compress(data, level=6)
    elif fmt == "xz":
        out = orig_path.parent / (orig_path.name + ".xz")
        compressed = lzma.compress(data, preset=1)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    out.write_bytes(compressed)
    return out, len(compressed)


def format_size(nbytes: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def compression_ratio(original: int, compressed: int) -> str:
    """Return compression ratio as a string like '5.2x'."""
    if compressed == 0:
        return "∞x"
    return f"{original / compressed:.1f}x"
