"""Per-study QC HTML report generator.

Reads study_full_series_stats.json (or series_stats.json) plus individual
series JSON files under a study directory, then renders a self-contained
single-file HTML report with:

  - Study header (manufacturer, model, field strength, …)
  - Series summary table (color-coded grades)
  - Per-series cards with embedded multiplane PNG and key metrics
  - Conformance issues section
  - Anomaly flags section
  - Footer with pipeline_version + git SHA

CLI usage:
    python -m src.qc.per_study --study Speall_MRI_Samples --out qc.html
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

# ---------------------------------------------------------------------------
# Grade styling
# ---------------------------------------------------------------------------

_GRADE_CSS_CLASS: dict[str, str] = {
    "A": "grade-a",
    "B": "grade-b",
    "C": "grade-c",
    "D": "grade-d",
    "F": "grade-f",
}

# ---------------------------------------------------------------------------
# Inline CSS + JS (no external deps)
# ---------------------------------------------------------------------------

_INLINE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       background: #f0f4f8; color: #1a202c; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 18px; font-weight: 600; margin: 24px 0 12px; }
h3 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
.header-bar { background: #1a365d; color: #fff; padding: 20px 24px; }
.header-bar p { font-size: 13px; opacity: .8; margin-top: 4px; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 12px; margin: 16px 0; }
.meta-card { background: #fff; border-radius: 8px; padding: 12px 16px;
             box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.meta-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
              color: #718096; margin-bottom: 4px; }
.meta-value { font-size: 16px; font-weight: 700; color: #2d3748; }
.section { background: #fff; border-radius: 10px; padding: 20px 24px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; }
th { background: #2d3748; color: #fff; padding: 8px 12px; text-align: left;
     font-size: 12px; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }
tr:hover td { background: #f7fafc; }
.grade-pill { display: inline-block; border-radius: 999px; padding: 2px 10px;
              font-weight: 700; font-size: 13px; }
.grade-a { background: #c6f6d5; color: #276749; }
.grade-b { background: #bee3f8; color: #2b6cb0; }
.grade-c { background: #fefcbf; color: #975a16; }
.grade-d { background: #feebc8; color: #c05621; }
.grade-f { background: #fed7d7; color: #9b2c2c; }
.series-card { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 16px;
               overflow: hidden; }
.card-header { background: #2d3748; color: #fff; padding: 10px 16px;
               display: flex; align-items: center; gap: 12px; }
.card-body { padding: 16px; display: grid; grid-template-columns: auto 1fr; gap: 16px; }
.card-image img { max-width: 320px; max-height: 200px; border-radius: 6px;
                  display: block; object-fit: contain; }
.no-image { width: 320px; height: 140px; background: #e2e8f0; border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            color: #a0aec0; font-size: 13px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
dt { font-weight: 600; color: #718096; font-size: 12px; }
dd { font-size: 13px; color: #2d3748; }
.flag { display: inline-block; background: #fed7d7; color: #9b2c2c; border-radius: 4px;
        padding: 2px 8px; font-size: 12px; margin: 2px; }
.conformance-file { background: #fffbeb; border-left: 3px solid #f6ad55;
                    padding: 6px 10px; margin-bottom: 6px; border-radius: 0 4px 4px 0; }
.conformance-tags { font-size: 11px; color: #718096; }
.anomaly-row { background: #fff5f5; border-left: 3px solid #fc8181;
               padding: 8px 12px; margin-bottom: 6px; border-radius: 0 4px 4px 0; }
footer { text-align: center; padding: 20px; color: #a0aec0; font-size: 12px; }
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.kpi { flex: 1; min-width: 100px; background: #ebf8ff; border-radius: 8px;
       padding: 10px 16px; text-align: center; }
.kpi-val { font-size: 22px; font-weight: 800; color: #2b6cb0; }
.kpi-lbl { font-size: 11px; color: #718096; }
@media (max-width: 700px) {
  .card-body { grid-template-columns: 1fr; }
  .card-image img, .no-image { max-width: 100%; width: 100%; }
}
"""

# ---------------------------------------------------------------------------
# Jinja2 template
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QC Report — {{ study.study_description | default("Study", true) }}</title>
<style>{{ css }}</style>
</head>
<body>

<div class="header-bar">
  <div class="container">
    <h1>QC Report &mdash; {{ study.study_description | default("Unknown Study", true) }}</h1>
    <p>{{ study.manufacturer | default("") }} {{ study.model | default("") }} &bull;
       {{ study.field_strength_T | default("?") }}T &bull;
       Study date: {{ study.study_date | default("Unknown") }}</p>
  </div>
</div>

<div class="container">

  {# ── Study header ── #}
  <div class="meta-grid">
    <div class="meta-card"><div class="meta-label">Manufacturer</div><div class="meta-value">{{ study.manufacturer | default("—") }}</div></div>
    <div class="meta-card"><div class="meta-label">Model</div><div class="meta-value">{{ study.model | default("—") }}</div></div>
    <div class="meta-card"><div class="meta-label">Field Strength</div><div class="meta-value">{{ study.field_strength_T | default("?") }} T</div></div>
    <div class="meta-card"><div class="meta-label">Software</div><div class="meta-value" style="font-size:12px;">{{ study.software | default("—") | truncate(40) }}</div></div>
    <div class="meta-card"><div class="meta-label">Station</div><div class="meta-value">{{ study.station_name | default("—") }}</div></div>
    <div class="meta-card"><div class="meta-label">Series Count</div><div class="meta-value">{{ kpis.n_series }}</div></div>
    <div class="meta-card"><div class="meta-label">DICOM Files</div><div class="meta-value">{{ kpis.n_files }}</div></div>
    <div class="meta-card"><div class="meta-label">Study Date</div><div class="meta-value">{{ study.study_date | default("—") }}</div></div>
  </div>

  {# ── Grade KPIs ── #}
  <div class="kpi-row">
    {% for g in ["A","B","C","D","F"] %}
    <div class="kpi">
      <div class="kpi-val">{{ grade_counts.get(g, 0) }}</div>
      <div class="kpi-lbl">Grade {{ g }}</div>
    </div>
    {% endfor %}
  </div>

  {# ── Series summary table ── #}
  <div class="section">
    <h2>Series Summary</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Description</th><th>Sequence Type</th>
          <th>Files</th><th>TR / TE (ms)</th><th>Grade</th><th>ML Score</th>
        </tr>
      </thead>
      <tbody>
        {% for s in series %}
        <tr>
          <td>{{ s.series_number }}</td>
          <td>{{ s.series_description | default("—") }}</td>
          <td>{{ s.sequence_type | default("—") }}</td>
          <td>{{ s.file_count }}</td>
          <td>{{ "%.0f" | format(s.tr_ms) if s.tr_ms else "—" }} / {{ "%.1f" | format(s.te_ms) if s.te_ms else "—" }}</td>
          <td><span class="grade-pill {{ grade_css(s.quality_grade) }}">{{ s.quality_grade | default("?") }}</span></td>
          <td>{{ "%.1f" | format(s.quality_score) if s.quality_score is not none else "—" }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {# ── Per-series cards ── #}
  <h2>Series Detail</h2>
  {% for card in series_cards %}
  <div class="series-card">
    <div class="card-header">
      <span style="font-weight:700; font-size:15px;">{{ card.series_number }}: {{ card.series_description | default("—") }}</span>
      <span class="grade-pill {{ grade_css(card.quality_grade) }}">{{ card.quality_grade | default("?") }}</span>
      <span style="font-size:12px; opacity:.7;">{{ card.sequence_type | default("") }}</span>
    </div>
    <div class="card-body">
      <div class="card-image">
        {% if card.image_b64 %}
        <img src="data:image/png;base64,{{ card.image_b64 }}" alt="Multiplane montage for {{ card.series_description }}">
        {% else %}
        <div class="no-image">No montage available</div>
        {% endif %}
      </div>
      <div>
        <dl>
          <dt>TR / TE</dt><dd>{{ "%.0f" | format(card.tr_ms) if card.tr_ms else "—" }} / {{ "%.1f" | format(card.te_ms) if card.te_ms else "—" }} ms</dd>
          <dt>Files</dt><dd>{{ card.file_count }}</dd>
          <dt>Volume Shape</dt><dd>{{ card.volume_shape | join(" × ") if card.volume_shape else "—" }}</dd>
          <dt>SNR</dt><dd>{{ "%.2f" | format(card.snr) if card.snr else "—" }}</dd>
          <dt>CNR</dt><dd>{{ "%.1f" | format(card.cnr) if card.cnr else "—" }}</dd>
          <dt>Quality Score</dt><dd>{{ "%.1f" | format(card.quality_score) if card.quality_score is not none else "—" }}</dd>
          {% if card.b_value is not none %}<dt>b-value</dt><dd>{{ card.b_value }}</dd>{% endif %}
          {% if card.motion_severity is not none %}<dt>Motion Severity</dt><dd>{{ "%.1f" | format(card.motion_severity) }}</dd>{% endif %}
          {% if card.symmetry_index is not none %}<dt>Symmetry Index</dt><dd>{{ "%.3f" | format(card.symmetry_index) }}</dd>{% endif %}
          {% if card.n_anomalous is not none %}<dt>Anomalous Slices</dt><dd>{{ card.n_anomalous }}</dd>{% endif %}
        </dl>
        {% if card.quality_flags %}
        <div style="margin-top:10px;">
          {% for flag in card.quality_flags %}
          <span class="flag">{{ flag }}</span>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </div>
  </div>
  {% endfor %}

  {# ── Conformance issues ── #}
  <div class="section">
    <h2>Conformance Issues
      <span style="font-size:13px; font-weight:400; color:#718096;">
        ({{ conformance_issues_count }} files with missing tags &bull; pass rate: {{ "%.1f" | format(conformance_pass_pct) }}%)
      </span>
    </h2>
    {% if conformance_issues %}
    {% for issue in conformance_issues[:50] %}
    <div class="conformance-file">
      <strong>{{ issue.filename }}</strong>
      <span class="conformance-tags">Missing: {{ issue.missing_tags | join(", ") }}</span>
      &nbsp;— completeness {{ "%.1f" | format(issue.completeness_pct) }}%
    </div>
    {% endfor %}
    {% if conformance_issues | length > 50 %}
    <p style="color:#718096; font-size:12px; margin-top:8px;">… and {{ (conformance_issues | length) - 50 }} more files.</p>
    {% endif %}
    {% else %}
    <p style="color:#48bb78;">All files pass conformance checks.</p>
    {% endif %}
  </div>

  {# ── Anomaly flags ── #}
  <div class="section">
    <h2>Anomaly Flags
      <span style="font-size:13px; font-weight:400; color:#718096;">({{ anomalies_count }} series flagged)</span>
    </h2>
    {% if anomalies %}
    {% for a in anomalies %}
    <div class="anomaly-row">
      <strong>{{ a.series_description }}</strong>
      {% for reason in a.reasons %}<span class="flag">{{ reason }}</span>{% endfor %}
    </div>
    {% endfor %}
    {% else %}
    <p style="color:#48bb78;">No anomalies detected.</p>
    {% endif %}
  </div>

</div><!-- /container -->

<footer>
  Pipeline version: {{ pipeline_version }} &bull;
  Generated: {{ generated_at }} &bull;
  Git SHA: {{ git_sha }}
</footer>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _encode_png(path: Path, max_px: int = 800) -> str:
    """Return base64-encoded PNG, downsampling if larger than max_px wide."""
    if not path.exists():
        return ""
    try:
        from PIL import Image
        import io
        with Image.open(path) as img:
            w, h = img.size
            if w > max_px:
                ratio = max_px / w
                img = img.resize((max_px, int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return base64.b64encode(path.read_bytes()).decode()


def _load_series_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _extract_series_row(s: dict) -> dict:
    """Flatten a series dict from study_full_series_stats.json into a row."""
    seq_cls = s.get("sequence_classification", {})
    vs = s.get("volume_stats", {})
    qa = s.get("quality_analysis", {})
    qg = qa.get("quality_grade", {})
    motion = qa.get("motion_analysis", {})
    symmetry = qa.get("symmetry_analysis", {})
    anomaly = qa.get("anomaly_detection", {})
    sp = s.get("sequence_params", {})
    return {
        "series_number": s.get("series_number"),
        "series_description": s.get("series_description", ""),
        "sequence_type": seq_cls.get("sequence_type", ""),
        "file_count": s.get("file_count", 0),
        "tr_ms": sp.get("tr"),
        "te_ms": sp.get("te"),
        "b_value": sp.get("b_value"),
        "volume_shape": vs.get("volume_shape"),
        "quality_grade": (qg.get("grade") or vs.get("quality_grade")),
        "quality_score": (qg.get("score") or vs.get("quality_score")),
        "snr": vs.get("volume_snr_estimate"),
        "cnr": vs.get("volume_cnr"),
        "motion_severity": motion.get("motion_severity_score"),
        "symmetry_index": symmetry.get("symmetry_index"),
        "n_anomalous": anomaly.get("n_anomalous"),
        "conformance_issues": s.get("conformance_issues", []),
    }


def _extract_series_from_summary(s: dict) -> dict:
    """Flatten a series dict from study_summary.json series array."""
    return {
        "series_number": s.get("series_number"),
        "series_description": s.get("series_description", ""),
        "sequence_type": s.get("sequence_type", ""),
        "file_count": s.get("file_count", 0),
        "tr_ms": s.get("tr_ms"),
        "te_ms": s.get("te_ms"),
        "b_value": s.get("b_value"),
        "volume_shape": s.get("volume_shape"),
        "quality_grade": s.get("quality_grade"),
        "quality_score": s.get("quality_score"),
        "snr": s.get("volume_snr"),
        "cnr": s.get("volume_cnr"),
        "motion_severity": None,
        "symmetry_index": None,
        "n_anomalous": None,
        "conformance_issues": [],
    }


def _flag_anomalies(row: dict) -> list[str]:
    flags = []
    ms = row.get("motion_severity")
    if ms is not None and ms > 50:
        flags.append(f"Motion severity {ms:.1f} > 50")
    si = row.get("symmetry_index")
    if si is not None and si < 0.85:
        flags.append(f"Symmetry index {si:.3f} < 0.85")
    na = row.get("n_anomalous")
    if na is not None and na > 0:
        flags.append(f"{na} anomalous slice(s)")
    return flags


def _find_multiplane_png(study_dir: Path, series_row: dict) -> Path | None:
    """Search for a multiplane PNG for this series.  Returns None if not found."""
    snum = series_row.get("series_number")
    sdesc = (series_row.get("series_description") or "").strip()

    candidates: list[Path] = []

    # Pattern: s{snum:04d}_*_multiplane.png
    if snum is not None:
        candidates += list(study_dir.glob(f"s{int(snum):04d}_*multiplane*.png"))
        candidates += list(study_dir.glob(f"s{int(snum):04d}_*.png"))

    # Fallback: fuzzy match on description
    if sdesc:
        safe = sdesc.replace(" ", "_").replace("/", "_")
        candidates += list(study_dir.glob(f"*{safe}*multiplane*.png"))

    # Also look in series/ subdirectory
    series_subdir = study_dir / "series"
    if series_subdir.exists():
        candidates += list(series_subdir.glob("*.png"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_study_qc_report(
    study_dir: Path,
    out_path: Path,
    template_str: str | None = None,
    pipeline_version: str = "5.0.0",
) -> dict[str, Any]:
    """Build a self-contained HTML QC report for a single study.

    Args:
        study_dir: Directory containing study_full_series_stats.json (or
                   study_summary.json) and series/ sub-dir with per-series
                   JSON files.
        out_path:  Destination path for the generated HTML file.
        template_str: Optional Jinja2 template string.  Falls back to the
                      built-in default template.
        pipeline_version: Version string embedded in the footer.

    Returns:
        dict with keys: n_series, grade_counts, anomalies_count,
        conformance_issues_count.
    """
    study_dir = Path(study_dir)
    out_path = Path(out_path)

    # ------------------------------------------------------------------
    # Load study metadata
    # ------------------------------------------------------------------
    full_stats_path = study_dir / "study_full_series_stats.json"
    summary_path = study_dir / "study_summary.json"

    if full_stats_path.exists():
        raw = json.loads(full_stats_path.read_text())
        patient = raw.get("patient", {})
        raw_series_list = raw.get("series", [])
        series_rows = [_extract_series_row(s) for s in raw_series_list]
    elif summary_path.exists():
        raw = json.loads(summary_path.read_text())
        patient = raw.get("study", {})
        raw_series_list = raw.get("series", [])
        series_rows = [_extract_series_from_summary(s) for s in raw_series_list]
    else:
        # Fall back to scanning individual series JSONs
        patient = {}
        series_json_paths = sorted((study_dir / "series").glob("s*.json"))
        series_rows = []
        for p in series_json_paths:
            try:
                d = _load_series_json(p)
                series_rows.append(_extract_series_row(d))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Study header dict
    # ------------------------------------------------------------------
    study = {
        "study_date": patient.get("study_date", ""),
        "study_description": patient.get("study_description", ""),
        "manufacturer": patient.get("manufacturer", ""),
        "model": patient.get("model", ""),
        "field_strength_T": patient.get("field_strength", patient.get("field_strength_T", "")),
        "software": str(patient.get("software_versions", patient.get("software", ""))),
        "station_name": patient.get("station_name", patient.get("station", "")),
    }

    # ------------------------------------------------------------------
    # Grade counts
    # ------------------------------------------------------------------
    grade_counts: dict[str, int] = {g: 0 for g in "ABCDF"}
    for row in series_rows:
        g = row.get("quality_grade") or ""
        if g in grade_counts:
            grade_counts[g] += 1

    # ------------------------------------------------------------------
    # Conformance issues aggregation
    # ------------------------------------------------------------------
    all_conformance: list[dict] = []
    for row in series_rows:
        all_conformance.extend(row.get("conformance_issues", []))

    n_files = sum(r.get("file_count", 0) for r in series_rows)
    files_with_issues = len({i.get("filename") for i in all_conformance})
    conformance_pass_pct = (1.0 - files_with_issues / max(n_files, 1)) * 100.0

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    anomalies = []
    for row in series_rows:
        flags = _flag_anomalies(row)
        if flags:
            anomalies.append({
                "series_description": row.get("series_description", "?"),
                "reasons": flags,
            })

    # ------------------------------------------------------------------
    # Series cards (with embedded images)
    # ------------------------------------------------------------------
    series_cards = []
    for row in series_rows:
        png_path = _find_multiplane_png(study_dir, row)
        image_b64 = _encode_png(png_path) if png_path else ""
        card = dict(row)
        card["image_b64"] = image_b64
        card["quality_flags"] = _flag_anomalies(row)
        series_cards.append(card)

    # ------------------------------------------------------------------
    # Render template
    # ------------------------------------------------------------------
    env = Environment(loader=BaseLoader())
    env.globals["grade_css"] = lambda g: _GRADE_CSS_CLASS.get(str(g).upper(), "grade-f")

    tmpl = env.from_string(template_str or _DEFAULT_TEMPLATE)

    kpis = {"n_series": len(series_rows), "n_files": n_files}

    html = tmpl.render(
        css=_INLINE_CSS,
        study=study,
        kpis=kpis,
        grade_counts=grade_counts,
        series=[
            {
                "series_number": r.get("series_number"),
                "series_description": r.get("series_description"),
                "sequence_type": r.get("sequence_type"),
                "file_count": r.get("file_count"),
                "tr_ms": r.get("tr_ms"),
                "te_ms": r.get("te_ms"),
                "quality_grade": r.get("quality_grade"),
                "quality_score": r.get("quality_score"),
            }
            for r in series_rows
        ],
        series_cards=series_cards,
        conformance_issues=all_conformance,
        conformance_issues_count=files_with_issues,
        conformance_pass_pct=conformance_pass_pct,
        anomalies=anomalies,
        anomalies_count=len(anomalies),
        pipeline_version=pipeline_version,
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        git_sha=_get_git_sha(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {
        "n_series": len(series_rows),
        "grade_counts": grade_counts,
        "anomalies_count": len(anomalies),
        "conformance_issues_count": files_with_issues,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a per-study QC HTML report.",
    )
    parser.add_argument("--study", required=True, type=Path,
                        help="Path to the study directory.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output HTML file path.")
    parser.add_argument("--pipeline-version", default="5.0.0")
    args = parser.parse_args()

    summary = build_study_qc_report(
        args.study, args.out, pipeline_version=args.pipeline_version,
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"Written {args.out}  ({size_kb:.1f} KB)")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    _cli()
