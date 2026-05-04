"""HTML report generation — threaded image encoding + concurrent section building.

Generates an interactive, modern dashboard-style report with:
- Sticky navigation bar with section links
- Study-level quality dashboard with grade distribution
- Collapsible, tabbed series cards (params / quality / images)
- Click-to-zoom lightbox for all images
- Quality score progress bars with breakdowns
- Search/filter by series type or quality grade
- Smooth CSS animations and responsive layout
- Print-friendly styles
"""

from __future__ import annotations

import base64
import datetime
import html as html_mod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .quality import grade_study
from .report_templates import CSS as _CSS
from .report_templates import JS as _JS


def _img_to_b64(path: str | Path | None) -> str:
    if path:
        p = Path(path)
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""


def _build_series_card(s: dict, b64_cache: dict, idx: int) -> str:
    """Build the HTML card for a single series."""
    if not s.get("has_pixels"):
        return ""

    snum = s.get("series_number", "?")
    desc = s.get("series_description", "?")
    key = f"{snum}_{desc}"
    seq_class = s.get("sequence_classification", {})
    seq_type = seq_class.get("sequence_type", "Unknown")
    confidence = seq_class.get("confidence", "low")

    badge_cls = "seq-other"
    for k, cls in [
        ("T1", "seq-T1"),
        ("T2", "seq-T2"),
        ("FLAIR", "seq-FLAIR"),
        ("DWI", "seq-DWI"),
        ("ADC", "seq-ADC"),
        ("GRE", "seq-GRE"),
    ]:
        if k in seq_type.upper():
            badge_cls = cls
            break
    conf_cls = {"high": "conf-high", "medium": "conf-med", "low": "conf-low"}.get(
        confidence, "conf-low"
    )

    # Quality data
    qa = s.get("quality_analysis", {})
    qg = qa.get("quality_grade", {})
    grade = qg.get("grade", "?")
    score = qg.get("score", 0)
    breakdown = qg.get("breakdown", {})

    grade_cls = f"grade-{grade}" if grade in "ABCDF" else "grade-F"

    # Sequence params
    params = s.get("sequence_params", {})
    vs = s.get("volume_stats", {})

    # Build param rows
    param_html = ""
    param_items = [(k, v) for k, v in params.items() if v not in (None, "", "None")]
    if param_items:
        rows = "".join(
            f"<tr><td>{html_mod.escape(str(k))}</td><td>{html_mod.escape(str(v))}</td></tr>"
            for k, v in param_items
        )
        param_html = f'<table class="param-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>'

    # Build metrics rows
    metric_items = [
        ("Volume Shape", vs.get("volume_shape", "")),
        ("Spacing (mm)", vs.get("spacing_mm", "")),
        ("SNR Estimate", f"{vs.get('volume_snr_estimate', 0):.2f}"),
        ("Entropy", f"{vs.get('volume_entropy', 0):.1f} bits"),
        ("Tissue %", f"{vs.get('volume_tissue_pct', 0):.1f}%"),
        ("Dynamic Range", f"{vs.get('volume_dynamic_range', 0):.0f}"),
        ("Uniformity", f"{vs.get('slice_intensity_uniformity', 0):.3f}"),
        ("Volume (cc)", f"{vs.get('total_volume_cc', 0):.1f}"),
        ("Non-zero %", f"{vs.get('volume_nonzero_pct', 0):.1f}%"),
    ]
    metric_rows = "".join(f"<tr><td>{m}</td><td>{v}</td></tr>" for m, v in metric_items if v)
    metric_html = (
        f'<table class="param-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>'
        if metric_rows
        else ""
    )

    # Quality breakdown bars
    quality_bars = ""
    if breakdown:
        max_pts = {
            "snr": 30,
            "uniformity": 20,
            "tissue_coverage": 15,
            "dynamic_range": 15,
            "entropy": 10,
            "nonzero_coverage": 10,
        }
        bar_items = []
        for metric_name, pts in breakdown.items():
            mx = max_pts.get(metric_name, 30)
            pct = min(pts / mx * 100, 100) if mx > 0 else 0
            label = metric_name.replace("_", " ").title()
            color = "#3fb950" if pct >= 70 else "#d29922" if pct >= 40 else "#f85149"
            bar_items.append(
                f'<div class="qbar-row">'
                f'<span class="qbar-label">{label}</span>'
                f'<div class="qbar-track"><div class="qbar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'
                f'<span class="qbar-pts">{pts:.0f}/{mx}</span>'
                f"</div>"
            )
        quality_bars = '<div class="qbar-container">' + "".join(bar_items) + "</div>"

    # Quality badges
    quality_badges = ""
    if qa:
        badges = []
        motion = qa.get("motion_analysis", {})
        interp = motion.get("interpretation", "")
        if motion.get("motion_detected"):
            badges.append('<span class="qa-badge qa-warn">Motion Detected</span>')
        elif interp and interp != "N/A":
            badges.append(f'<span class="qa-badge qa-ok">{html_mod.escape(interp)}</span>')

        anomaly = qa.get("anomaly_detection", {})
        n_anom = anomaly.get("n_anomalous", 0)
        if n_anom > 0:
            badges.append(
                f'<span class="qa-badge qa-danger">{n_anom} outlier slice{"s" if n_anom > 1 else ""}</span>'
            )

        sym = qa.get("symmetry_analysis", {})
        si = sym.get("symmetry_index", 1)
        sym_interp = sym.get("interpretation", "")
        if sym_interp and sym_interp not in (
            "N/A",
            "N/A (2D)",
            "N/A (non-axial)",
            "N/A (single slice)",
            "N/A (small)",
        ):
            sym_cls = "qa-ok" if si > 0.90 else "qa-warn" if si > 0.80 else "qa-danger"
            badges.append(f'<span class="qa-badge {sym_cls}">Symmetry: {si:.2f}</span>')

        sharpness = qa.get("sharpness_analysis", {})
        sharp_interp = sharpness.get("interpretation", "")
        if sharp_interp and sharp_interp != "N/A":
            sharp_cls = (
                "qa-ok"
                if "sharp" in sharp_interp.lower()
                else "qa-warn"
                if "moderate" in sharp_interp.lower()
                else "qa-danger"
            )
            badges.append(
                f'<span class="qa-badge {sharp_cls}">{html_mod.escape(sharp_interp)}</span>'
            )

        quality_badges = '<div class="qa-badges">' + "".join(badges) + "</div>" if badges else ""

    # Reasoning
    reasoning = seq_class.get("reasoning", [])
    reasoning_html = ""
    if reasoning:
        reasoning_html = (
            f'<div class="reasoning">{" &bull; ".join(html_mod.escape(r) for r in reasoning)}</div>'
        )

    # Images
    mb64 = b64_cache.get(f"montage_{key}", "")
    hb64 = b64_cache.get(f"hist_{key}", "")
    montage_img = (
        f'<img src="data:image/png;base64,{mb64}" class="zoomable" alt="Montage for series {snum}" loading="lazy">'
        if mb64
        else '<div class="no-image">No montage available</div>'
    )
    hist_img = (
        f'<img src="data:image/png;base64,{hb64}" class="zoomable hist-img" alt="Histogram for series {snum}" loading="lazy">'
        if hb64
        else ""
    )

    card_id = f"series-{snum}"

    return f"""<div class="series-card" id="{card_id}" data-type="{html_mod.escape(seq_type.upper())}" data-grade="{grade}" data-desc="{html_mod.escape(desc.lower())}">
  <div class="card-header" onclick="toggleCard(this)">
    <div class="card-title-row">
      <span class="card-expand-icon">&#9654;</span>
      <h3>Series {snum} &mdash; {html_mod.escape(desc)}</h3>
      <span class="seq-badge {badge_cls}">{html_mod.escape(seq_type)}</span>
      <span class="conf-badge {conf_cls}">{html_mod.escape(confidence)}</span>
      <div class="card-grade {grade_cls}">{grade}<span class="card-grade-score">{score:.0f}</span></div>
    </div>
    {reasoning_html}
  </div>
  <div class="card-body">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab(event, 'images-{idx}')">Images</button>
      <button class="tab-btn" onclick="switchTab(event, 'quality-{idx}')">Quality</button>
      <button class="tab-btn" onclick="switchTab(event, 'params-{idx}')">Parameters</button>
      <button class="tab-btn" onclick="switchTab(event, 'metrics-{idx}')">Metrics</button>
    </div>
    <div id="images-{idx}" class="tab-panel active">
      <div class="image-grid">
        <div class="image-cell">{montage_img}</div>
        <div class="image-cell">{hist_img}</div>
      </div>
    </div>
    <div id="quality-{idx}" class="tab-panel">
      {quality_bars}
      {quality_badges}
    </div>
    <div id="params-{idx}" class="tab-panel">
      {param_html or '<div class="empty-state">No sequence parameters available</div>'}
    </div>
    <div id="metrics-{idx}" class="tab-panel">
      {metric_html or '<div class="empty-state">No volume metrics available</div>'}
    </div>
  </div>
</div>"""


def _build_series_cards(series_list: list[dict], b64_cache: dict) -> str:
    """Build HTML for all series cards in parallel threads."""
    image_series = [(i, s) for i, s in enumerate(series_list) if s.get("has_pixels")]
    if len(image_series) <= 2:
        return "\n".join(_build_series_card(s, b64_cache, i) for i, s in enumerate(series_list))
    with ThreadPoolExecutor(max_workers=min(len(image_series), 8)) as pool:
        futures = [
            pool.submit(_build_series_card, s, b64_cache, i) for i, s in enumerate(series_list)
        ]
        return "\n".join(f.result() for f in futures)


def _build_conformance_section(conformance_issues: list[dict]) -> str:
    """Build the conformance check HTML section."""
    parts = ['<div class="section" id="conformance"><h2>DICOM Conformance</h2>']
    if conformance_issues:
        worst = min(i["completeness_pct"] for i in conformance_issues)
        best = max(i["completeness_pct"] for i in conformance_issues)
        parts.append(
            f'<div class="alert alert-warn">Tag completeness ranges from {worst}% to {best}% across files.</div>'
        )
        parts.append(
            '<div class="card"><table class="data-table"><thead><tr><th>File</th><th>Missing Tags</th><th>Completeness</th></tr></thead><tbody>'
        )
        for issue in sorted(conformance_issues, key=lambda x: x["completeness_pct"])[:20]:
            ms = ", ".join(issue["missing_tags"][:8])
            if len(issue["missing_tags"]) > 8:
                ms += f" (+{len(issue['missing_tags']) - 8})"
            pct = issue["completeness_pct"]
            pct_cls = "pct-good" if pct >= 90 else "pct-warn" if pct >= 70 else "pct-bad"
            parts.append(
                f'<tr><td class="mono">{html_mod.escape(issue["filename"])}</td><td class="small">{ms}</td><td class="{pct_cls}">{pct}%</td></tr>'
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append(
            '<div class="alert alert-ok">All files pass DICOM MR conformance checks.</div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_study_dashboard(series_info: list[dict], n_files: int) -> str:
    """Build the study-level quality dashboard."""
    image_series = [s for s in series_info if s.get("has_pixels")]
    grades = [
        s.get("quality_analysis", {}).get("quality_grade", {})
        for s in image_series
        if s.get("quality_analysis")
    ]

    study_grade = grade_study(grades)
    sg = study_grade.get("grade", "N/A")
    ss = study_grade.get("score", 0)
    dist = study_grade.get("grade_distribution", {})

    grade_cls = f"grade-{sg}" if sg in "ABCDF" else ""

    # Grade distribution mini-chart (pure CSS)
    max_count = max(dist.values()) if dist else 1
    dist_bars = ""
    for g in ["A", "B", "C", "D", "F"]:
        count = dist.get(g, 0)
        height = (count / max_count * 100) if max_count > 0 else 0
        dist_bars += f'<div class="dist-col"><div class="dist-bar grade-{g}-bg" style="height:{max(height, 4):.0f}%"></div><span class="dist-label">{g}</span><span class="dist-count">{count}</span></div>'

    # Count issues
    motion_count = sum(
        1
        for s in image_series
        if s.get("quality_analysis", {}).get("motion_analysis", {}).get("motion_detected")
    )
    anomaly_count = sum(
        s.get("quality_analysis", {}).get("anomaly_detection", {}).get("n_anomalous", 0)
        for s in image_series
    )

    return f"""<div class="section" id="dashboard">
  <h2>Study Quality Dashboard</h2>
  <div class="dashboard-grid">
    <div class="dash-card dash-grade">
      <div class="dash-grade-circle {grade_cls}">
        <span class="dash-grade-letter">{sg}</span>
        <span class="dash-grade-score">{ss:.0f}/100</span>
      </div>
      <div class="dash-label">Overall Grade</div>
    </div>
    <div class="dash-card">
      <div class="dash-dist">{dist_bars}</div>
      <div class="dash-label">Grade Distribution</div>
    </div>
    <div class="dash-card">
      <div class="dash-metrics-grid">
        <div class="dash-metric"><span class="dash-val">{n_files}</span><span class="dash-sub">Files</span></div>
        <div class="dash-metric"><span class="dash-val">{len(series_info)}</span><span class="dash-sub">Series</span></div>
        <div class="dash-metric"><span class="dash-val">{len(image_series)}</span><span class="dash-sub">Image Series</span></div>
        <div class="dash-metric"><span class="dash-val">{len(series_info) - len(image_series)}</span><span class="dash-sub">Non-Image</span></div>
      </div>
    </div>
    <div class="dash-card">
      <div class="dash-metrics-grid">
        <div class="dash-metric"><span class="dash-val {"text-danger" if motion_count > 0 else "text-ok"}">{motion_count}</span><span class="dash-sub">Motion Issues</span></div>
        <div class="dash-metric"><span class="dash-val {"text-danger" if anomaly_count > 0 else "text-ok"}">{anomaly_count}</span><span class="dash-sub">Outlier Slices</span></div>
        <div class="dash-metric"><span class="dash-val">{study_grade.get("best_score", 0):.0f}</span><span class="dash-sub">Best Score</span></div>
        <div class="dash-metric"><span class="dash-val">{study_grade.get("worst_score", 0):.0f}</span><span class="dash-sub">Worst Score</span></div>
      </div>
    </div>
  </div>
</div>"""


def generate_html_report(
    patient_info: dict,
    series_info: list[dict],
    conformance_issues: list[dict],
    image_paths: dict[str, dict],
    cross_series_path: str | None,
    out_dir: Path,
    thread_pool: ThreadPoolExecutor,
) -> Path:
    """Generate HTML report with threaded image encoding + concurrent section building."""

    # Pre-encode all images in parallel
    b64_futures: dict[str, Future] = {}
    if cross_series_path:
        b64_futures["cross"] = thread_pool.submit(_img_to_b64, cross_series_path)
    for key, paths in image_paths.items():
        if paths.get("montage"):
            b64_futures[f"montage_{key}"] = thread_pool.submit(_img_to_b64, paths["montage"])
        if paths.get("histogram"):
            b64_futures[f"hist_{key}"] = thread_pool.submit(_img_to_b64, paths["histogram"])

    # Build conformance section concurrently while images encode
    conformance_fut = thread_pool.submit(_build_conformance_section, conformance_issues)

    b64_cache = {k: fut.result() for k, fut in b64_futures.items()}

    # Build all series cards
    series_cards = _build_series_cards(series_info, b64_cache)
    conformance_html = conformance_fut.result()

    p = patient_info
    n_files = sum(s.get("file_count", 0) for s in series_info)
    sum(1 for s in series_info if s.get("has_pixels"))

    # Study dashboard
    dashboard_html = _build_study_dashboard(series_info, n_files)

    # Collect unique sequence types and grades for filter buttons
    seq_types = sorted(
        {
            s.get("sequence_classification", {}).get("sequence_type", "").upper()
            for s in series_info
            if s.get("has_pixels") and s.get("sequence_classification")
        }
    )
    seq_types = [t for t in seq_types if t and t != "UNKNOWN"]

    grades_present = sorted(
        {
            s.get("quality_analysis", {}).get("quality_grade", {}).get("grade", "")
            for s in series_info
            if s.get("has_pixels") and s.get("quality_analysis")
        }
    )
    grades_present = [g for g in grades_present if g]

    type_btns = "".join(
        f'<button class="filter-btn" data-filter-type="type" data-seqtype="{t}" onclick="filterByType(\'{t}\')">{t}</button>'
        for t in seq_types
    )
    grade_btns = "".join(
        f'<button class="filter-btn" data-filter-type="grade" data-grade="{g}" onclick="filterByGrade(\'{g}\')">Grade {g}</button>'
        for g in grades_present
    )

    # Cross-series section
    cross_html = ""
    if b64_cache.get("cross"):
        cross_html = f"""<div class="section" id="cross-series">
  <h2>Cross-Series Comparison</h2>
  <div class="card"><img src="data:image/png;base64,{b64_cache["cross"]}" class="cross-series-img" alt="Cross-series comparison"></div>
</div>"""

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DICOM Study &mdash; {html_mod.escape(p.get("patient_name", "Report"))}</title>
<style>{_CSS}</style>
</head>
<body>

<nav class="navbar">
  <span class="navbar-brand">DICOM Report</span>
  <div class="nav-links">
    <a href="#patient">Patient</a>
    <a href="#dashboard">Quality</a>
    <a href="#cross-series">Comparison</a>
    <a href="#series-detail">Series</a>
    <a href="#conformance">Conformance</a>
  </div>
  <div class="nav-search">
    <input type="text" id="search-input" placeholder="Search series..." autocomplete="off">
  </div>
</nav>

<div class="container">

<div class="report-header">
  <h1>DICOM Study Report</h1>
  <div class="subtitle">{html_mod.escape(p.get("patient_name", "?"))} &bull; {html_mod.escape(p.get("study_date", "?"))} &bull; {html_mod.escape(p.get("manufacturer", "?"))} {html_mod.escape(p.get("model", ""))} &bull; {html_mod.escape(p.get("field_strength", "?"))}T</div>
</div>

<div class="section" id="patient">
  <h2>Patient &amp; Scanner</h2>
  <div class="patient-grid">
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("patient_name", "?"))}</div><div class="label">Patient</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("patient_sex", "?"))}</div><div class="label">Sex</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("study_date", "?"))}</div><div class="label">Study Date</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("manufacturer", "?"))}</div><div class="label">Manufacturer</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("model", "?"))}</div><div class="label">Scanner Model</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("field_strength", "?"))}T</div><div class="label">Field Strength</div></div>
    <div class="patient-cell"><div class="val">{html_mod.escape(p.get("institution", "?"))}</div><div class="label">Institution</div></div>
  </div>
</div>

{dashboard_html}

{cross_html}

<div class="section" id="series-detail">
  <h2>Series Detail</h2>
  <div class="filter-bar">
    <label>Filter:</label>
    {type_btns}
    {grade_btns}
    <button class="filter-btn" onclick="expandAll()" style="margin-left:auto">Expand All</button>
    <button class="filter-btn" onclick="collapseAll()">Collapse All</button>
  </div>
  {series_cards}
</div>

{conformance_html}

<div class="footer">Generated {now} by micom &mdash; DICOM Extractor v5</div>

</div>

<div class="lightbox" id="lightbox">
  <img id="lightbox-img" src="" alt="Zoomed image">
</div>

<script>{_JS}</script>
</body>
</html>"""

    html_path = out_dir / "report.html"
    html_path.write_text(html)
    return html_path
