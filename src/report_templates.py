# ruff: noqa: E501
"""Static HTML/CSS/JS templates for the DICOM HTML report.

Pure data — no logic, no imports. Extracted from ``report.py`` to keep that
module focused on orchestration and HTML composition.
"""

from __future__ import annotations

CSS = """
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #21262d;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #484f58;
  --blue: #58a6ff;
  --blue-dim: #1f6feb;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --purple: #d2a8ff;
  --cyan: #79c0ff;
  --radius: 10px;
  --shadow: 0 2px 8px rgba(0,0,0,0.3);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  overflow-x: hidden;
}

/* ── Navigation ──────────────────────────────────────────── */
.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(13, 17, 23, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  height: 52px;
}
.navbar-brand {
  font-weight: 700;
  font-size: 1.1em;
  color: var(--blue);
  white-space: nowrap;
  letter-spacing: -0.02em;
}
.nav-links {
  display: flex;
  gap: 4px;
  overflow-x: auto;
  scrollbar-width: none;
}
.nav-links::-webkit-scrollbar { display: none; }
.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 0.85em;
  padding: 6px 12px;
  border-radius: 6px;
  white-space: nowrap;
  transition: all 0.15s;
}
.nav-links a:hover { background: var(--bg-tertiary); color: var(--text-primary); }
.nav-search {
  margin-left: auto;
  position: relative;
}
.nav-search input {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-primary);
  padding: 6px 12px 6px 32px;
  font-size: 0.85em;
  width: 220px;
  outline: none;
  transition: border-color 0.15s, width 0.2s;
}
.nav-search input:focus { border-color: var(--blue-dim); width: 280px; }
.nav-search::before {
  content: "\\1F50D";
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.8em;
  opacity: 0.5;
}

/* ── Layout ──────────────────────────────────────────────── */
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.section { margin-bottom: 36px; animation: fadeIn 0.3s ease; }
.section h2 {
  color: var(--cyan);
  font-size: 1.3em;
  font-weight: 600;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ── Header ──────────────────────────────────────────────── */
.report-header {
  text-align: center;
  padding: 36px 24px 24px;
  margin-bottom: 12px;
}
.report-header h1 {
  font-size: 2em;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.03em;
  margin-bottom: 4px;
}
.report-header .subtitle {
  color: var(--text-secondary);
  font-size: 0.95em;
}

/* ── Patient Info ────────────────────────────────────────── */
.patient-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}
.patient-cell {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  text-align: center;
}
.patient-cell .val {
  font-size: 1.3em;
  font-weight: 600;
  color: var(--blue);
  word-break: break-word;
}
.patient-cell .label {
  font-size: 0.78em;
  color: var(--text-secondary);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Dashboard ───────────────────────────────────────────── */
.dashboard-grid {
  display: grid;
  grid-template-columns: auto 1fr 1fr 1fr;
  gap: 16px;
  align-items: stretch;
}
@media (max-width: 900px) { .dashboard-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 550px) { .dashboard-grid { grid-template-columns: 1fr; } }

.dash-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.dash-label {
  font-size: 0.78em;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.dash-grade-circle {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 3px solid var(--border);
}
.dash-grade-letter { font-size: 2.2em; font-weight: 800; line-height: 1; }
.dash-grade-score { font-size: 0.75em; color: var(--text-secondary); }

.dash-dist {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  height: 80px;
  padding: 0 8px;
}
.dist-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  height: 100%;
  justify-content: flex-end;
}
.dist-bar {
  width: 100%;
  min-width: 24px;
  border-radius: 4px 4px 0 0;
  transition: height 0.5s ease;
}
.dist-label { font-size: 0.75em; color: var(--text-secondary); margin-top: 4px; }
.dist-count { font-size: 0.7em; color: var(--text-muted); }

.dash-metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  width: 100%;
}
.dash-metric { text-align: center; }
.dash-val { display: block; font-size: 1.6em; font-weight: 700; color: var(--blue); }
.dash-sub { font-size: 0.72em; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.03em; }

/* ── Grade colors ────────────────────────────────────────── */
.grade-A { color: var(--green); border-color: var(--green) !important; }
.grade-B { color: var(--blue); border-color: var(--blue) !important; }
.grade-C { color: var(--yellow); border-color: var(--yellow) !important; }
.grade-D { color: var(--red); border-color: var(--red) !important; }
.grade-F { color: #f0453a; border-color: #f0453a !important; }
.grade-A-bg { background: var(--green); }
.grade-B-bg { background: var(--blue); }
.grade-C-bg { background: var(--yellow); }
.grade-D-bg { background: var(--red); }
.grade-F-bg { background: #f0453a; }

.text-ok { color: var(--green) !important; }
.text-danger { color: var(--red) !important; }

/* ── Filter bar ──────────────────────────────────────────── */
.filter-bar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  align-items: center;
}
.filter-bar label {
  font-size: 0.82em;
  color: var(--text-secondary);
  margin-right: 4px;
}
.filter-btn {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8em;
  cursor: pointer;
  transition: all 0.15s;
}
.filter-btn:hover { border-color: var(--blue-dim); color: var(--text-primary); }
.filter-btn.active { background: var(--blue-dim); border-color: var(--blue); color: #fff; }

/* ── Series Cards ────────────────────────────────────────── */
.series-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  overflow: hidden;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.series-card:hover { border-color: var(--bg-tertiary); box-shadow: var(--shadow); }
.series-card.hidden { display: none; }

.card-header {
  padding: 14px 18px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
}
.card-header:hover { background: rgba(255,255,255,0.02); }

.card-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.card-expand-icon {
  font-size: 0.7em;
  color: var(--text-muted);
  transition: transform 0.2s;
  flex-shrink: 0;
}
.series-card.expanded .card-expand-icon { transform: rotate(90deg); }

.card-title-row h3 {
  font-size: 1em;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.seq-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.75em;
  font-weight: 600;
  letter-spacing: 0.03em;
}
.seq-T1 { background: #0d3320; color: var(--green); }
.seq-T2 { background: #0d2744; color: var(--blue); }
.seq-FLAIR { background: #2d1a00; color: var(--yellow); }
.seq-DWI { background: #2d1a2d; color: var(--purple); }
.seq-ADC { background: #1a1a3d; color: var(--cyan); }
.seq-GRE { background: #2d1010; color: var(--red); }
.seq-other { background: var(--bg-tertiary); color: var(--text-secondary); }

.conf-badge {
  font-size: 0.7em;
  padding: 2px 8px;
  border-radius: 20px;
  font-weight: 500;
}
.conf-high { background: #0d3320; color: var(--green); }
.conf-med { background: #2d1a00; color: var(--yellow); }
.conf-low { background: #2d1010; color: var(--red); }

.card-grade {
  margin-left: auto;
  font-size: 1.4em;
  font-weight: 800;
  display: flex;
  align-items: baseline;
  gap: 4px;
  flex-shrink: 0;
}
.card-grade-score {
  font-size: 0.45em;
  font-weight: 400;
  color: var(--text-secondary);
}

.reasoning {
  font-size: 0.78em;
  color: var(--text-muted);
  margin-top: 6px;
  padding-left: 22px;
}

.card-body {
  display: none;
  border-top: 1px solid var(--border);
  padding: 0;
}
.series-card.expanded .card-body { display: block; animation: slideDown 0.2s ease; }

@keyframes slideDown { from { opacity: 0; } to { opacity: 1; } }

/* ── Tabs ────────────────────────────────────────────────── */
.tab-bar {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--bg-primary);
  overflow-x: auto;
}
.tab-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  padding: 10px 18px;
  font-size: 0.85em;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  white-space: nowrap;
}
.tab-btn:hover { color: var(--text-primary); }
.tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }

.tab-panel { display: none; padding: 18px; }
.tab-panel.active { display: block; }

/* ── Images ──────────────────────────────────────────────── */
.image-grid {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: start;
}
@media (max-width: 700px) { .image-grid { grid-template-columns: 1fr; } }

.image-cell img {
  max-width: 100%;
  border-radius: 8px;
  cursor: zoom-in;
  transition: transform 0.15s;
}
.image-cell img:hover { transform: scale(1.01); }
.hist-img { max-height: 260px; }
.no-image {
  color: var(--text-muted);
  font-size: 0.85em;
  padding: 32px;
  text-align: center;
  background: var(--bg-primary);
  border-radius: 8px;
}

/* ── Quality bars ────────────────────────────────────────── */
.qbar-container { display: flex; flex-direction: column; gap: 8px; }
.qbar-row { display: flex; align-items: center; gap: 10px; }
.qbar-label { width: 130px; font-size: 0.82em; color: var(--text-secondary); text-align: right; flex-shrink: 0; }
.qbar-track {
  flex: 1;
  height: 8px;
  background: var(--bg-primary);
  border-radius: 4px;
  overflow: hidden;
}
.qbar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}
.qbar-pts { width: 50px; font-size: 0.78em; color: var(--text-muted); }

/* ── QA badges ───────────────────────────────────────────── */
.qa-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.qa-badge {
  font-size: 0.78em;
  padding: 3px 10px;
  border-radius: 20px;
  font-weight: 500;
}
.qa-ok { background: #0d3320; color: var(--green); }
.qa-warn { background: #2d1a00; color: var(--yellow); }
.qa-danger { background: #2d1010; color: var(--red); }

/* ── Tables ──────────────────────────────────────────────── */
.param-table, .data-table {
  width: 100%;
  border-collapse: collapse;
}
.param-table th, .data-table th {
  background: var(--bg-primary);
  color: var(--blue);
  padding: 8px 14px;
  text-align: left;
  font-size: 0.8em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border);
}
.param-table td, .data-table td {
  padding: 6px 14px;
  border-bottom: 1px solid rgba(48, 54, 61, 0.4);
  font-size: 0.85em;
}
.param-table tr:hover, .data-table tr:hover { background: rgba(255,255,255,0.02); }
.mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82em; }
.small { font-size: 0.78em; }
.pct-good { color: var(--green); font-weight: 600; }
.pct-warn { color: var(--yellow); font-weight: 600; }
.pct-bad { color: var(--red); font-weight: 600; }
.empty-state { color: var(--text-muted); font-size: 0.85em; padding: 20px; text-align: center; }

/* ── Alerts ──────────────────────────────────────────────── */
.alert {
  padding: 10px 16px;
  border-radius: var(--radius);
  margin-bottom: 12px;
  font-size: 0.9em;
  border-left: 3px solid;
}
.alert-ok { background: #0d2818; border-color: var(--green); color: var(--green); }
.alert-warn { background: #2d1a00; border-color: var(--yellow); color: var(--yellow); }

/* ── Cross-series ────────────────────────────────────────── */
.cross-series-img {
  max-width: 100%;
  border-radius: 8px;
  cursor: zoom-in;
}

/* ── Lightbox ────────────────────────────────────────────── */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 200;
  background: rgba(0,0,0,0.9);
  backdrop-filter: blur(4px);
  align-items: center;
  justify-content: center;
  cursor: zoom-out;
}
.lightbox.active { display: flex; }
.lightbox img {
  max-width: 95vw;
  max-height: 95vh;
  border-radius: 8px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}

/* ── Footer ──────────────────────────────────────────────── */
.footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 0.78em;
  padding: 32px 0 16px;
  border-top: 1px solid var(--border);
  margin-top: 24px;
}

/* ── Card inner helper ───────────────────────────────────── */
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

/* ── Print ───────────────────────────────────────────────── */
@media print {
  .navbar, .nav-search, .filter-bar, .lightbox { display: none !important; }
  body { background: #fff; color: #000; }
  .series-card, .card, .dash-card, .patient-cell { border-color: #ccc; background: #fff; }
  .card-body { display: block !important; }
  .series-card { page-break-inside: avoid; }
}
"""

JS = """
// Toggle card expand/collapse
function toggleCard(header) {
  const card = header.closest('.series-card');
  card.classList.toggle('expanded');
}

// Tab switching
function switchTab(e, panelId) {
  e.stopPropagation();
  const card = e.target.closest('.series-card');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  card.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(panelId).classList.add('active');
}

// Search/filter
function initSearch() {
  const input = document.getElementById('search-input');
  if (!input) return;
  input.addEventListener('input', function() {
    const q = this.value.toLowerCase().trim();
    document.querySelectorAll('.series-card').forEach(card => {
      const desc = card.dataset.desc || '';
      const type = (card.dataset.type || '').toLowerCase();
      const grade = (card.dataset.grade || '').toLowerCase();
      const match = !q || desc.includes(q) || type.includes(q) || grade.includes(q);
      card.classList.toggle('hidden', !match);
    });
  });
}

// Grade filter buttons
function filterByGrade(grade) {
  const btns = document.querySelectorAll('.filter-btn[data-filter-type="grade"]');
  btns.forEach(b => {
    if (b.dataset.grade === grade && b.classList.contains('active')) {
      b.classList.remove('active');
      grade = null;
    } else {
      b.classList.toggle('active', b.dataset.grade === grade);
    }
  });
  document.querySelectorAll('.series-card').forEach(card => {
    if (!grade) { card.classList.remove('hidden'); return; }
    card.classList.toggle('hidden', card.dataset.grade !== grade);
  });
}

// Type filter buttons
function filterByType(type) {
  const btns = document.querySelectorAll('.filter-btn[data-filter-type="type"]');
  btns.forEach(b => {
    if (b.dataset.seqtype === type && b.classList.contains('active')) {
      b.classList.remove('active');
      type = null;
    } else {
      b.classList.toggle('active', b.dataset.seqtype === type);
    }
  });
  document.querySelectorAll('.series-card').forEach(card => {
    if (!type) { card.classList.remove('hidden'); return; }
    card.classList.toggle('hidden', !card.dataset.type.includes(type));
  });
}

// Lightbox
function initLightbox() {
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lightbox-img');
  if (!lb) return;

  document.querySelectorAll('.zoomable, .cross-series-img').forEach(img => {
    img.addEventListener('click', function(e) {
      e.stopPropagation();
      lbImg.src = this.src;
      lb.classList.add('active');
    });
  });
  lb.addEventListener('click', () => lb.classList.remove('active'));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') lb.classList.remove('active');
  });
}

// Expand all / collapse all
function expandAll() {
  document.querySelectorAll('.series-card').forEach(c => c.classList.add('expanded'));
}
function collapseAll() {
  document.querySelectorAll('.series-card').forEach(c => c.classList.remove('expanded'));
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  initSearch();
  initLightbox();
});
"""
