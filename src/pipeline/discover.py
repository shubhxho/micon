"""Stage 1 — Discover DICOM files via recursive walk + threaded subdirectory scanning.

Complexity
----------
- Directory traversal: O(N) where N = total filesystem entries (files + dirs).
  Uses os.scandir() which avoids extra stat() calls — each entry's type is
  resolved from the directory buffer itself (DT_REG / DT_DIR on most FSes).
  Shallow subdirectories (depth < 2) are scanned in parallel threads for
  latency hiding on network/spinning-disk filesystems.

- Sorting: O(F log F) where F = number of .dcm files found.
  Uses Python's built-in sorted() (Timsort, implemented in C) which is
  adaptive — O(F) on already-sorted input, O(F log F) worst-case.
  The previous hand-rolled merge-sort had the same asymptotic bound but
  ~3-5× higher constant factor (Python-level comparisons, repeated list
  allocations at every merge step).

Overall: O(N + F log F)  — dominated by the sort when F is large,
         by the walk when the tree is deep / wide with many non-.dcm files.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

console = Console()


def _recursive_scan(directory: Path, depth: int = 0) -> list[Path]:
    """Recursively walk a directory tree collecting .dcm files.

    Uses os.scandir() instead of Path.iterdir() — scandir reads directory
    entries in bulk and exposes d_type from the kernel, so is_file()/is_dir()
    don't require a separate stat() syscall on most platforms.  O(N) total
    where N = entries in this subtree.

    At depth < 2, subdirectories fan out across threads for I/O parallelism.
    """
    found: list[Path] = []
    subdirs: list[Path] = []

    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".dcm"):
                    found.append(Path(entry.path))
                elif entry.is_dir(follow_symlinks=False):
                    subdirs.append(Path(entry.path))
    except PermissionError:
        return found

    if depth < 2 and len(subdirs) > 1:
        with ThreadPoolExecutor(max_workers=min(len(subdirs), 8)) as pool:
            futures = {pool.submit(_recursive_scan, sd, depth + 1): sd for sd in subdirs}
            for fut in as_completed(futures):
                found.extend(fut.result())
    else:
        for sd in subdirs:
            found.extend(_recursive_scan(sd, depth + 1))

    return found


def discover_files(folder: Path, recursive: bool = True) -> list[Path]:
    """Discover .dcm files using recursive walk + threaded scanning.

    When *recursive* is False, only looks in *folder* directly (no subdirs).

    Sorting uses Python's built-in Timsort — O(F log F) worst-case,
    O(F) when input is already partially ordered (common for sequential
    DICOM filenames like IM-0001.dcm … IM-0500.dcm).
    """
    if recursive:
        raw = _recursive_scan(folder)
    else:
        raw = [
            Path(e.path)
            for e in os.scandir(folder)
            if e.is_file(follow_symlinks=False) and e.name.lower().endswith(".dcm")
        ]
    dcm_files = sorted(raw)

    if not dcm_files:
        console.print(f"[red]No .dcm files found in {folder}.[/red]")
        raise SystemExit(1)

    n_subdirs = len({f.parent for f in dcm_files}) - (
        1 if any(f.parent == folder for f in dcm_files) else 0
    )
    loc_msg = f"in [dim]{folder}[/dim]"
    if n_subdirs > 0:
        loc_msg += f" [dim](+ {n_subdirs} subfolder{'s' if n_subdirs > 1 else ''})[/dim]"
    console.print(f"Found [bold]{len(dcm_files)}[/bold] DICOM files {loc_msg}\n")
    return dcm_files


def discover_dcm_folders(root: Path) -> list[Path]:
    """Walk *root* and return every directory that directly contains .dcm files.

    Uses os.scandir for efficient directory traversal. Returns folders sorted
    by path for deterministic ordering.
    """
    folders: list[Path] = []

    def _has_dcm(directory: Path) -> bool:
        try:
            with os.scandir(directory) as it:
                return any(
                    e.is_file(follow_symlinks=False) and e.name.lower().endswith(".dcm") for e in it
                )
        except PermissionError:
            return False

    # Check root itself
    if _has_dcm(root):
        folders.append(root)

    # Walk all subdirectories
    for dirpath, _, _ in os.walk(root):
        dp = Path(dirpath)
        if dp != root and _has_dcm(dp):
            folders.append(dp)

    folders.sort()

    if not folders:
        console.print(f"[red]No folders with .dcm files found under {root}.[/red]")
        raise SystemExit(1)

    console.print(
        f"Found [bold]{len(folders)}[/bold] folder(s) with DICOM files under [dim]{root}[/dim]\n"
    )
    return folders
