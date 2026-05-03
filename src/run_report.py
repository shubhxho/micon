"""Per-run report: persists stage timings, counts, errors, and cost estimates.

Usage (integration sketch -- actual wiring is a follow-up commit)::

    report = RunReport(output_dir)
    report.record_stage("quality", ok=100, skipped=20, failed=3,
                        elapsed_s=45.2, errors=["IOError: ...", ...])
    report.set_actual_cost(modal_dollars=0.12, openrouter_dollars=0.04)
    path = report.write()
    print(report.format_text())

Schema written to ``output_dir/runs/<run_id>.json``::

    {
      "run_id": "20260430T120000Z",
      "started_at": "2026-04-30T12:00:00+00:00",
      "finished_at": "2026-04-30T12:05:12+00:00",
      "stages": [...],
      "cost_estimated": {"modal_dollars": null, "openrouter_dollars": null},
      "cost_actual":    {"modal_dollars": null, "openrouter_dollars": null},
      "modal_app": null,
      "git_sha": "abc123"
    }

No Modal imports -- this module runs fine outside a container.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str:
    """Return current HEAD sha, or 'unknown' on any error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _cost_block(modal_dollars: float | None, openrouter_dollars: float | None) -> dict:
    return {"modal_dollars": modal_dollars, "openrouter_dollars": openrouter_dollars}


class RunReport:
    """Collect stage results and write a diffable JSON summary."""

    def __init__(self, output_dir: Path, run_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id: str = run_id or _make_run_id()
        self.started_at: str = _now_iso()
        self.finished_at: str | None = None
        self.stages: list[dict[str, Any]] = []
        self.cost_estimated: dict = _cost_block(None, None)
        self.cost_actual: dict = _cost_block(None, None)
        self.modal_app: str | None = None
        self.git_sha: str = _git_sha()

    def record_stage(
        self,
        stage: str,
        ok: int,
        skipped: int,
        failed: int,
        elapsed_s: float,
        errors: list[str] | None = None,
        **extras: Any,
    ) -> None:
        """Append one stage result.  ``errors`` is capped at 20 samples."""
        entry: dict[str, Any] = {
            "stage": stage,
            "ok": ok,
            "skipped": skipped,
            "failed": failed,
            "elapsed_s": round(elapsed_s, 2),
            "errors": (errors or [])[:20],
        }
        entry.update(extras)
        self.stages.append(entry)

    def set_estimated_cost(
        self,
        modal_dollars: float | None = None,
        openrouter_dollars: float | None = None,
    ) -> None:
        self.cost_estimated = _cost_block(modal_dollars, openrouter_dollars)

    def set_actual_cost(
        self,
        modal_dollars: float | None = None,
        openrouter_dollars: float | None = None,
    ) -> None:
        self.cost_actual = _cost_block(modal_dollars, openrouter_dollars)

    def _to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": self.stages,
            "cost_estimated": self.cost_estimated,
            "cost_actual": self.cost_actual,
            "modal_app": self.modal_app,
            "git_sha": self.git_sha,
        }

    def write(self) -> Path:
        """Stamp ``finished_at``, write JSON, return path."""
        self.finished_at = _now_iso()
        runs_dir = self.output_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = runs_dir / f"{self.run_id}.json"
        out_path.write_text(json.dumps(self._to_dict(), indent=2, default=str))
        return out_path

    def format_text(self) -> str:
        """Return a multi-line human-readable summary for stdout."""
        lines: list[str] = [
            f"Run {self.run_id}",
            f"  started  : {self.started_at}",
            f"  finished : {self.finished_at or '(not yet)'}",
            f"  git_sha  : {self.git_sha}",
        ]
        if self.modal_app:
            lines.append(f"  modal_app: {self.modal_app}")

        if self.stages:
            lines.append("  stages:")
            for s in self.stages:
                lines.append(
                    f"    {s['stage']:<20} ok={s['ok']:>6}  "
                    f"skip={s['skipped']:>6}  fail={s['failed']:>4}  "
                    f"{s['elapsed_s']:.1f}s"
                )
                for err in s.get("errors", []):
                    lines.append(f"      ! {err}")

        def _fmt_cost(label: str, block: dict) -> str:
            m = block.get("modal_dollars")
            o = block.get("openrouter_dollars")
            m_str = f"${m:.4f}" if m is not None else "n/a"
            o_str = f"${o:.4f}" if o is not None else "n/a"
            total = (m or 0.0) + (o or 0.0)
            t_str = f"${total:.4f}" if (m is not None or o is not None) else "n/a"
            return f"  {label:<18} modal={m_str}  openrouter={o_str}  total={t_str}"

        lines.append(_fmt_cost("cost_estimated:", self.cost_estimated))
        lines.append(_fmt_cost("cost_actual:", self.cost_actual))
        return "\n".join(lines)
