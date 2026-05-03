"""PHI / path-leak sweep gate for pre-upload validation.

Walks all *.json under a root directory and flags:
  - Absolute paths in string values (/vol/, /Users/, /Volumes/, /home/, /root/)
  - MRN-shaped values: 7+ consecutive digits in a string value, not inside a UID
  - PHI key names as JSON keys: PatientName, patient_name, etc.
  - DOB-shaped values (YYYYMMDD) under known birth-date keys

CLI:
  python -m src.preflight_phi_sweep --root <path>
  Exits 0 with no findings, exits 1 with findings printed to stdout.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TypedDict

# ── Types ────────────────────────────────────────────────────────────────────

class Finding(TypedDict):
    path: str
    line_number: int
    kind: str
    snippet: str

# ── Patterns ─────────────────────────────────────────────────────────────────

_ABS_PATH_RE = re.compile(r'(/vol/|/Users/|/Volumes/|/home/|/root/)')
_MRN_RE = re.compile(r'(?<!\d)\d{7,}(?!\d)')

_PHI_KEY_TOKENS = frozenset([
    "PatientName", "patient_name", "PATIENT NAME",
    "PatientID", "patient_id",
    "PatientBirthDate",
])

_BIRTH_KEY_RE = re.compile(r'birth|dob|birthdate|patient_birth_date', re.IGNORECASE)
_DOB_RE = re.compile(r'^\d{8}$')

# Keys whose values may legitimately contain long digit runs (DICOM UIDs, stats)
_UID_KEY_RE = re.compile(r'uid|UID|volume_voxel_count|direction_cosines', re.IGNORECASE)


# ── Core walkers ─────────────────────────────────────────────────────────────

def _line_number_of(raw: str, char_offset: int) -> int:
    """Return 1-based line number for a character offset in raw text."""
    return raw.count('\n', 0, char_offset) + 1


def _scan_lines(raw: str, file_path: str) -> list[Finding]:
    """Line-based regex pass: absolute paths."""
    findings: list[Finding] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        m = _ABS_PATH_RE.search(line)
        if m:
            findings.append({
                "path": file_path,
                "line_number": lineno,
                "kind": "absolute_path",
                "snippet": line.strip()[:120],
            })
    return findings


def _walk_json(
    node: object,
    key_path: list[str],
    raw: str,
    file_path: str,
    findings: list[Finding],
) -> None:
    """Recursively walk parsed JSON; emit key-conditioned findings."""
    if isinstance(node, dict):
        for k, v in node.items():
            current_path = key_path + [str(k)]

            # Flag PHI key names that exist as actual JSON keys
            if k in _PHI_KEY_TOKENS and v not in ("", None, []):
                snippet = f'"{k}": {json.dumps(v, default=str)}'[:120]
                lineno = _find_key_line(raw, k)
                findings.append({
                    "path": file_path,
                    "line_number": lineno,
                    "kind": "phi_key",
                    "snippet": snippet,
                })

            # DOB check: string value under a birth-related key
            if _BIRTH_KEY_RE.search(k) and isinstance(v, str) and _DOB_RE.match(v.strip()):
                snippet = f'"{k}": "{v}"'
                lineno = _find_key_line(raw, k)
                findings.append({
                    "path": file_path,
                    "line_number": lineno,
                    "kind": "dob_value",
                    "snippet": snippet,
                })

            # MRN check: 7+ digits in a string value, skip UID/stats keys
            if (
                isinstance(v, str)
                and not _UID_KEY_RE.search(k)
                and _MRN_RE.search(v)
            ):
                snippet = f'"{k}": "{v}"'[:120]
                lineno = _find_key_line(raw, k)
                findings.append({
                    "path": file_path,
                    "line_number": lineno,
                    "kind": "mrn_shaped",
                    "snippet": snippet,
                })

            _walk_json(v, current_path, raw, file_path, findings)

    elif isinstance(node, list):
        for item in node:
            _walk_json(item, key_path, raw, file_path, findings)


def _find_key_line(raw: str, key: str) -> int:
    """Best-effort: find first line containing quoted key."""
    pattern = re.compile(r'"' + re.escape(key) + r'"')
    m = pattern.search(raw)
    if m:
        return _line_number_of(raw, m.start())
    return 0


# ── Public API ────────────────────────────────────────────────────────────────

def sweep(root: Path) -> list[Finding]:
    """Walk all *.json under root; return list of PHI/path-leak findings."""
    findings: list[Finding] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            raw = json_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_str = str(json_path)

        # Line-level pass: absolute paths
        findings.extend(_scan_lines(raw, file_str))

        # JSON walk: key-conditioned PHI patterns
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            continue
        _walk_json(doc, [], raw, file_str, findings)

    # Deduplicate: same file + line + kind
    seen: set[tuple[str, int, str]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f["path"], f["line_number"], f["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PHI / path-leak sweep")
    parser.add_argument("--root", required=True, help="Directory to sweep")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    findings = sweep(root)
    if not findings:
        print(f"OK — no PHI/path leaks found under {root}")
        sys.exit(0)

    print(f"FINDINGS: {len(findings)} issue(s) found under {root}\n")
    for f in findings:
        print(f"  [{f['kind']}] {f['path']}:{f['line_number']}")
        print(f"    {f['snippet']}")
    sys.exit(1)


if __name__ == "__main__":
    _main()
