"""Pure helper functions — no state, safe for multiprocessing."""

from __future__ import annotations

import numpy as np
import pydicom


def safe_value(elem) -> str | int | float | list | None:
    if elem is None:
        return None
    val = elem.value
    if isinstance(val, pydicom.sequence.Sequence):
        return f"[Sequence: {len(val)} items]"
    if isinstance(val, pydicom.valuerep.PersonName):
        return str(val)
    if isinstance(val, pydicom.uid.UID):
        return str(val)
    if isinstance(val, bytes):
        return val.hex() if len(val) <= 64 else f"[bytes: {len(val)}]"
    if isinstance(val, pydicom.multival.MultiValue):
        return [
            float(v)
            if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal))
            else str(v)
            for v in val
        ]
    if isinstance(val, (int, float)):
        return val
    return str(val)


def to_json(v):
    """Recursively convert a value to JSON-safe types.

    Handles nested dicts, lists, numpy scalars/arrays by recursing
    into each container element.
    """
    if isinstance(v, dict):
        return {k: to_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_json(item) for item in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return to_json(v.tolist())
    return v


def entropy(arr: np.ndarray) -> float:
    """Shannon entropy in bits.  O(V) — one histogram pass + O(B) log."""
    hist, _ = np.histogram(arr.ravel(), bins=256)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def skewness(arr: np.ndarray) -> float:
    """Sample skewness.  O(V) — two passes (mean, then cubed deviation)."""
    m, s = arr.mean(), arr.std()
    return float(((arr - m) ** 3).mean() / s**3) if s > 1e-10 else 0.0


def kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis.  O(V) — two passes (mean, then 4th-power deviation)."""
    m, s = arr.mean(), arr.std()
    return float(((arr - m) ** 4).mean() / s**4 - 3.0) if s > 1e-10 else 0.0


def skewness_kurtosis(arr: np.ndarray) -> tuple[float, float]:
    """Combined skewness + kurtosis in a single pass over deviations.

    Instead of computing mean+std twice (once for skewness, once for kurtosis),
    this shares the work:  O(V) for mean/std, then one pass for both m3 and m4.
    Total: 2 passes over V instead of 4.
    """
    m = arr.mean()
    s = arr.std()
    if s < 1e-10:
        return 0.0, 0.0
    d = arr - m  # O(V) — deviation array, reused for both moments
    d2 = d * d  # O(V)
    m3 = (d2 * d).mean()  # O(V) — third central moment
    m4 = (d2 * d2).mean()  # O(V) — fourth central moment (reuses d²)
    s3 = s**3
    s4 = s**4
    return float(m3 / s3), float(m4 / s4 - 3.0)


def safe_squeeze(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr


def get_2d_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    slc = vol[idx]
    while slc.ndim > 2:
        slc = slc[0]
    return slc


def safe_getfloat(ds, attr: str) -> float | None:
    """Safely extract a float attribute from a pydicom Dataset."""
    try:
        v = getattr(ds, attr, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
