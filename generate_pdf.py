#!/usr/bin/env python3
"""Speall MRI prospectus -- minimal edition.

Pulls totals from Speall_MRI_Dataset_Info.json and the example study
breakdown from Speall_MRI_Samples/study_summary.json. Designed to be
sent as-is to data buyers.
"""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).parent
PDF_PATH = ROOT / "Speall_MRI_Dataset_Prospectus.pdf"
INFO_PATH = ROOT / "Speall_MRI_Dataset_Info.json"
SAMPLES = ROOT / "Speall_MRI_Samples"

# Palette -- editorial, restrained
INK = (18, 22, 32)
SLATE = (70, 76, 88)
GREY = (135, 140, 150)
RULE = (220, 222, 228)
WASH = (248, 249, 251)
WHITE = (255, 255, 255)
ACCENT = (28, 70, 140)


def safe(s) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "—": "--",
        "–": "-",
        "•": "-",
        "→": "->",
        "←": "<-",
        "≤": "<=",
        "≥": ">=",
        "²": "^2",
        "³": "^3",
        "°": "deg",
        "±": "+/-",
        "×": "x",
        "·": ".",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class Doc(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*GREY)
            self.cell(95, 5, "SPEALL MRI", align="L")
            self.cell(95, 5, f"{self.page_no():02d}", align="R")
            self.ln(8)

    def footer(self):
        pass

    # -- type styles -------------------------------------------------
    def section(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*INK)
        self.cell(0, 8, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 25, self.get_y())
        self.ln(5)

    def h2(self, t):
        self.ln(1)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*INK)
        self.cell(0, 6, safe(t), new_x="LMARGIN", new_y="NEXT")

    def lead(self, t):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*SLATE)
        self.multi_cell(0, 5.8, safe(t))
        self.ln(2)

    def p(self, t):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*SLATE)
        self.multi_cell(0, 4.8, safe(t))
        self.ln(2)

    def kvrow(self, k, v, label_w=58):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*GREY)
        self.cell(label_w, 5.5, safe(k))
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*INK)
        self.cell(0, 5.5, safe(v), new_x="LMARGIN", new_y="NEXT")

    def kpi_tile(self, x, y, w, h, val, label):
        self.set_xy(x, y)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*INK)
        self.cell(w, 12, safe(val), align="L")
        self.set_xy(x, y + 12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        self.cell(w, 4, safe(label.upper()), align="L")

    def table(self, heads, rows, widths=None, aligns=None):
        if not widths:
            widths = [190 / len(heads)] * len(heads)
        aligns = aligns or ["L"] * len(heads)

        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*GREY)
        for i, h in enumerate(heads):
            self.cell(widths[i], 5, safe(h.upper()), align=aligns[i])
        self.ln()
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(0.5)

        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*INK)
        for _ri, r in enumerate(rows):
            for i, c in enumerate(r):
                self.cell(widths[i], 5.5, safe(c), align=aligns[i])
            self.ln()
        self.ln(2)

    def img(self, path, cap=None, w=180):
        p = Path(path)
        if not p.exists():
            return
        if self.get_y() > 215:
            self.add_page()
        self.image(str(p), w=w)
        if cap:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GREY)
            self.cell(0, 4, safe(cap), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)


def fmt_int(n) -> str:
    if n is None:
        return "-"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build():
    info = load_json(INFO_PATH) or {}
    study = load_json(SAMPLES / "study_summary.json")

    totals = info.get("totals", {})
    period = info.get("acquisition_period", {})
    primary = info.get("primary_protocol_subset", {})
    sex = info.get("patient_demographics", {}).get("sex_share", {})

    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 10, 10)

    # ── COVER ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 56)
    pdf.set_text_color(*INK)
    pdf.cell(0, 22, "Speall MRI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*SLATE)
    pdf.cell(0, 7, "A clinical brain MRI dataset", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.6)
    pdf.line(95, pdf.get_y(), 115, pdf.get_y())
    pdf.ln(10)

    # KPI band
    y = pdf.get_y()
    pdf.kpi_tile(15, y, 44, 18, fmt_int(totals.get("studies")), "Studies")
    pdf.kpi_tile(63, y, 44, 18, "355K", "DICOM Files")
    pdf.kpi_tile(111, y, 44, 18, "34.5K", "Series")
    pdf.kpi_tile(159, y, 44, 18, "66 GB", "Raw")
    pdf.ln(28)

    # Hero image
    pdf.img(
        str(ROOT / "output/000_s0007_Ax_T2_FLAIR/s0007_Ax_T2_FLAIR_multiplane.png"), None, w=178
    )

    pdf.set_y(-22)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(
        0, 4, safe(f"{period.get('start', '')} to {period.get('end', '')}  -  India"), align="C"
    )

    # ── PAGE 2 -- WHAT'S IN IT ────────────────────────────────────
    pdf.add_page()
    pdf.section("What's in the dataset")
    pdf.lead(
        f"{fmt_int(totals.get('studies'))} de-identified brain MRI studies acquired at one "
        f"hospital network in India between {period.get('start', '')} and {period.get('end', '')}. "
        f"Each study is a complete neurology workup -- 14 to 30 series across diffusion, "
        f"perfusion, structural and vascular imaging."
    )
    pdf.p(
        "The core of the dataset is a single GE SIGNA Pioneer 3.0T system with a 32-channel "
        "head coil -- about three quarters of the studies. The rest comes from Philips, Siemens, "
        "and other GE platforms, across both 3.0T and 1.5T. So you get a clean training core "
        "and a built-in distribution-shift validation set in one package."
    )

    pdf.ln(4)
    y = pdf.get_y()
    pdf.kpi_tile(10, y, 44, 18, fmt_int(totals.get("studies")), "Studies")
    pdf.kpi_tile(58, y, 44, 18, fmt_int(totals.get("series")), "Series")
    pdf.kpi_tile(106, y, 44, 18, fmt_int(totals.get("dicom_files")), "DICOM Files")
    pdf.kpi_tile(154, y, 44, 18, f"{period.get('span_years', '?')} yr", "Time span")
    pdf.ln(26)

    pdf.h2("At a glance")
    pdf.kvrow("Modality", "Brain MRI (multi-sequence + cerebrovascular)")
    pdf.kvrow("Time span", f"{period.get('start', '')} to {period.get('end', '')}")
    pdf.kvrow("Geography", info.get("dataset", {}).get("geography", "-"))
    pdf.kvrow("Patients (M / F)", f"{sex.get('M', 0) * 100:.1f}% / {sex.get('F', 0) * 100:.1f}%")
    pdf.kvrow("Primary scanner", primary.get("scanner", "GE SIGNA Pioneer 3.0T"))
    pdf.kvrow("Vendors", "GE, Philips, Siemens, Toshiba")
    pdf.kvrow("Field strengths", "3.0T (80%) and 1.5T (20%)")
    pdf.kvrow("Series per study", "14 to 30+")
    pdf.kvrow("De-identification", "PHI removed before delivery")

    # ── PAGE 3 -- WHY ──────────────────────────────────────────────
    pdf.add_page()
    pdf.section("Why this dataset")

    pillars = [
        (
            "Big, clean training core",
            "262K files come from a single 3.0T scanner with one coil class. One scanner, one "
            "protocol family. Hard to assemble a training set this consistent at this size.",
        ),
        (
            "Built-in distribution shift",
            "The other 26% spans Philips, Siemens, and three more GE platforms across 3.0T and "
            "1.5T -- ready-made validation for whatever you train on the core.",
        ),
        (
            "Dense per-study coverage",
            "Most studies carry 14 to 30+ series: diffusion, perfusion, FLAIR, susceptibility, "
            "TOF angiography, 3D T1 / T2 isotropic, MRV, and processed maps.",
        ),
        (
            "Demographics that fill a gap",
            "Single hospital network in India -- a population that's structurally under-represented "
            "in public neuroimaging datasets. Sex distribution is near-balanced.",
        ),
        (
            "Quality you can audit",
            "Every series carries an automated A-F quality grade. Filter to Grade A, or use the "
            "full distribution to train robust quality models.",
        ),
        (
            "Delivery-ready",
            "DICOM headers extracted to CSV / Parquet (240+ columns). Multiplane montages, "
            "histograms and per-series records ship alongside the source DICOM.",
        ),
    ]
    for title, body in pillars:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE)
        pdf.multi_cell(0, 4.8, safe(body))
        pdf.ln(3)

    # ── PAGE 4 -- SCANNER MIX + COVERAGE ───────────────────────────
    pdf.add_page()
    pdf.section("Scanner mix")
    pdf.p("Distribution of source scanners by file count.")
    rows = []
    for s in info.get("scanner_mix_by_files", []):
        rows.append(
            [
                s.get("vendor", ""),
                s.get("model", ""),
                f"{s.get('field_T', '')} T",
                fmt_int(s.get("files")),
                f"{s.get('share', 0) * 100:.1f}%",
            ]
        )
    pdf.table(
        ["Vendor", "Model", "Field", "Files", "Share"],
        rows,
        [22, 60, 18, 45, 45],
        aligns=["L", "L", "C", "R", "R"],
    )

    pdf.ln(2)
    pdf.section("Top sequences")
    pdf.p("Most common series descriptions across the corpus.")
    seq_rows = []
    for s in (info.get("series_descriptions_top") or [])[:14]:
        seq_rows.append([s.get("description", ""), fmt_int(s.get("series"))])
    pdf.table(["Series description", "Count"], seq_rows, [140, 50], aligns=["L", "R"])

    # ── PAGE 5 -- CLINICAL MIX ────────────────────────────────────
    pdf.add_page()
    pdf.section("What patients came in for")
    pdf.p("Top study indications recorded by the referring radiologist.")
    case_rows = []
    for s in (info.get("study_descriptions_top") or [])[:14]:
        case_rows.append([s.get("description", ""), fmt_int(s.get("files"))])
    pdf.table(["Indication", "Files"], case_rows, [140, 50], aligns=["L", "R"])

    pdf.section("Patient profile")
    pdf.kvrow("Sex (M / F)", f"{sex.get('M', 0) * 100:.1f}% / {sex.get('F', 0) * 100:.1f}%")
    pdf.kvrow("Age", "Paediatric to >85, adult-skewed (typical neuroradiology mix)")
    pdf.kvrow("Patients", f"~{fmt_int(totals.get('studies'))} (one study per patient)")
    pdf.kvrow("Source", "Single hospital network, India")

    # ── PAGE 6 -- ONE STUDY DEEP DIVE ────────────────────────────
    if study:
        pdf.add_page()
        pdf.section("Inside a single study")
        pst = study.get("study", {})
        pdf.p(
            f"Full per-series breakdown of one study from the 3.0T core "
            f"(date {pst.get('study_date', '-')}, {pst.get('study_description', '-')}). "
            f"{study.get('totals', {}).get('series_count', '-')} series, "
            f"{fmt_int(study.get('totals', {}).get('dicom_files'))} DICOM files."
        )

        rows = []
        for s in study.get("series", []):
            sn = s.get("series_number")
            rows.append(
                [
                    f"s{sn:04d}" if isinstance(sn, int) else f"s{sn}",
                    s.get("series_description", "")[:36],
                    s.get("sequence_type", "")[:18],
                    fmt_int(s.get("file_count")),
                    f"{s.get('tr_ms', '-')}/{s.get('te_ms', '-')}",
                    s.get("quality_grade", "-"),
                ]
            )
        pdf.table(
            ["#", "Description", "Type", "Files", "TR/TE", "Grade"],
            rows,
            [16, 64, 38, 22, 30, 20],
            aligns=["L", "L", "L", "R", "R", "C"],
        )

    # ── PAGE 7 -- IMAGERY ─────────────────────────────────────────
    pdf.add_page()
    pdf.section("Sample imagery")
    pdf.p("Every series ships with a three-plane montage (axial, coronal, sagittal).")

    montages = [
        ("output/000_s0005_Ax_DWI/s0005_Ax_DWI_multiplane.png", "Axial DWI (b=1000)"),
        ("output/000_s0011_AxT1_MEMP/s0011_AxT1_MEMP_multiplane.png", "Axial T1 MEMP"),
        ("output/000_s0013_BRAIN_ANGIO/s0013_BRAIN_ANGIO_multiplane.png", "Brain MR Angiography"),
        ("output/000_s0009_3D_Ax_SWAN/s0009_3D_Ax_SWAN_multiplane.png", "3D SWAN"),
        (
            "output/000_s0010_Ax_T2_PROPELLER/s0010_Ax_T2_PROPELLER_multiplane.png",
            "Axial T2 PROPELLER",
        ),
        ("output/000_s0012__SAG_T2/s0012__SAG_T2_multiplane.png", "Sagittal T2"),
        ("output/000_s0017_3D_Ax_TOF_NECK/s0017_3D_Ax_TOF_NECK_multiplane.png", "3D TOF Neck"),
        (
            "output/000_s0018_3D_Ax_T2_Cube_HyperCube./s0018_3D_Ax_T2_Cube_HyperCube._multiplane.png",
            "3D T2 Cube",
        ),
        ("output/000_s1001_B_FFE/s1001_B_FFE_multiplane.png", "B-FFE (CISS-equivalent)"),
    ]
    for path, cap in montages:
        if (ROOT / path).exists():
            pdf.img(str(ROOT / path), cap, w=178)

    # ── PAGE 8 -- QUALITY ─────────────────────────────────────────
    pdf.add_page()
    pdf.section("Quality grades")
    pdf.p(
        "Every series gets an automatic A-F grade so you can pick the tier you need without "
        "manually reviewing the corpus."
    )
    pdf.table(
        ["Grade", "Score", "Use"],
        [
            ["A", ">= 80", "Premium training and validation"],
            ["B", "65-79", "Standard training set"],
            ["C", "50-64", "Pre-training and quality models"],
            ["D", "35-49", "Artifact and motion training"],
            ["F", "< 35", "Excluded from premium tiers"],
        ],
        [22, 38, 130],
        aligns=["C", "L", "L"],
    )

    pdf.ln(2)
    pdf.h2("Cross-series quality, single study")
    if (ROOT / "output/cross_series_comparison.png").exists():
        pdf.img(str(ROOT / "output/cross_series_comparison.png"), None, w=178)

    # ── PAGE 9 -- DELIVERY ────────────────────────────────────────
    pdf.add_page()
    pdf.section("What you get")

    pdf.h2("Per series")
    pdf.table(
        ["Item", "Description"],
        [
            ["DICOM", "Original files, PHI-redacted"],
            ["Montage", "Axial / coronal / sagittal PNG"],
            ["Histogram", "Intensity distribution PNG"],
            ["Enhanced view", "Contrast-enhanced reference PNG"],
            ["Detail JSON", "Volume statistics, geometry, quality grade"],
        ],
        [44, 146],
    )

    pdf.h2("Per study")
    pdf.table(
        ["Item", "Description"],
        [
            ["Quality chart", "Multi-metric quality profile"],
            ["Study record", "Aggregated per-study summary"],
        ],
        [44, 146],
    )

    pdf.h2("Across the corpus")
    pdf.table(
        ["Item", "Description"],
        [
            ["Tabular metadata", "DICOM headers in CSV / Parquet (240+ columns)"],
            ["HTML report", "Browseable dashboard"],
            ["Series statistics", "Aggregated stats across the corpus"],
        ],
        [44, 146],
    )

    pdf.section("De-identification")
    pdf.kvrow("Identifiers", "Patient name, MRN, DOB, accession, institution -- removed")
    pdf.kvrow("Pixel data", "Screened for burned-in PHI")
    pdf.kvrow("Audit", "Per-study redaction report retained")

    pdf.section("Transfer")
    pdf.kvrow("Default", "Object storage handoff (S3 / GCS)")
    pdf.kvrow("Subsets", "Filtered by sequence, grade, vendor or date range")
    pdf.kvrow("Custom", "Per-buyer manifests on request")

    # ── PAGE 10 -- USE CASES ──────────────────────────────────────
    pdf.add_page()
    pdf.section("What teams build with this")

    cases = [
        ("Stroke and acute care", "DWI / ADC pairs at scale with paired structural sequences."),
        (
            "Tumour and lesion detection",
            "FLAIR, SWAN, 3D T1 / T2 Cube and post-contrast cover the standard work-up.",
        ),
        (
            "Vascular models",
            "TOF intracranial and neck angiography plus MRV -- a layer most public datasets lack.",
        ),
        ("Quality assessment", "Pre-graded so quality models can train against reference labels."),
        (
            "Cross-vendor robustness",
            "Train on the GE 3.0T core, validate on Philips / Siemens / 1.5T.",
        ),
        (
            "Foundation pre-training",
            "Multi-contrast, multi-vendor, 34K+ series for self-supervised pre-training.",
        ),
    ]
    for title, body in cases:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE)
        pdf.multi_cell(0, 4.8, safe(body))
        pdf.ln(2.5)

    pdf.output(str(PDF_PATH))
    sz = PDF_PATH.stat().st_size
    print(f"{PDF_PATH}  |  {sz / 1024 / 1024:.1f} MB  |  {pdf.page_no()} pages")


if __name__ == "__main__":
    build()
