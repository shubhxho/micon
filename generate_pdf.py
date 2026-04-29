#!/usr/bin/env python3
"""Minimal corporate dataset prospectus.

Numbers and facts in this document are derived from a full sweep of the source
DICOM corpus on the T7 Shield (355,133 files, 0 read errors) cross-checked
against the modal pipeline output (`micom-data` and `micom-v2` volumes).
"""

from pathlib import Path
from fpdf import FPDF

PDF_PATH = Path("Speall_MRI_Dataset_Prospectus.pdf")

# Minimal palette
K = (20, 20, 24)        # near-black
D = (50, 52, 58)        # dark gray
M = (120, 124, 132)     # mid gray
L = (190, 193, 198)     # light gray
F = (245, 246, 248)     # faint bg
W = (255, 255, 255)


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "", 6)
            self.set_text_color(*L)
            self.cell(95, 5, "SPEALL MRI  |  DATASET OVERVIEW", align="L")
            self.cell(95, 5, f"{self.page_no()}", align="R")
            self.ln(6)

    def footer(self):
        pass

    def h1(self, t):
        self.ln(8)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*K)
        self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*L)
        self.set_line_width(0.3)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(5)

    def h2(self, t):
        self.ln(3)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*D)
        self.cell(0, 6, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def p(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*D)
        self.multi_cell(0, 4.6, t)
        self.ln(2)

    def row(self, k, v):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*M)
        self.cell(48, 5, k)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*K)
        self.cell(0, 5, str(v), new_x="LMARGIN", new_y="NEXT")

    def tbl(self, heads, rows, ws=None):
        if not ws:
            w = 190 / len(heads)
            ws = [w] * len(heads)
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(*K)
        self.set_text_color(*W)
        for i, h in enumerate(heads):
            self.cell(ws[i], 6, h, fill=True, align="C" if i == 0 else "L")
        self.ln()
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*D)
        for ri, r in enumerate(rows):
            self.set_fill_color(*(F if ri % 2 else W))
            for i, c in enumerate(r):
                self.cell(ws[i], 5, str(c), fill=True, align="C" if i == 0 else "L")
            self.ln()
        self.ln(3)

    def img(self, path, cap, w=180):
        p = Path(path)
        if not p.exists():
            return
        if self.get_y() > 212:
            self.add_page()
        self.image(str(p), w=w)
        self.set_font("Helvetica", "", 6.5)
        self.set_text_color(*M)
        self.cell(0, 4, cap, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def stat(self, x, y, val, label):
        self.set_xy(x, y)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*K)
        self.cell(44, 12, val, align="C")
        self.set_xy(x, y + 12)
        self.set_font("Helvetica", "", 6.5)
        self.set_text_color(*M)
        self.cell(44, 5, label, align="C")


def build():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)

    # ── COVER ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(45)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*M)
    pdf.cell(0, 5, "DATASET OVERVIEW", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 38)
    pdf.set_text_color(*K)
    pdf.cell(0, 18, "Speall MRI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*D)
    pdf.cell(0, 8, "Clinical Brain Imaging Collection", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_draw_color(*L)
    pdf.set_line_width(0.3)
    pdf.line(80, pdf.get_y(), 130, pdf.get_y())
    pdf.ln(12)

    # Stats — verified from T7 Shield sweep (355,133 files, 0 errors)
    y = pdf.get_y()
    pdf.stat(10, y, "1,105", "STUDIES")
    pdf.stat(58, y, "355K", "DICOM FILES")
    pdf.stat(106, y, "34.5K", "SERIES")
    pdf.stat(154, y, "66 GB", "RAW VOLUME")
    pdf.ln(28)

    # Teaser
    pdf.img("output/000_s0007_Ax_T2_FLAIR/s0007_Ax_T2_FLAIR_multiplane.png",
            "Axial T2 FLAIR  --  Multiplane montage", w=175)

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*L)
    pdf.cell(0, 5, "April 2026", align="C")

    # ── OVERVIEW ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Overview")
    pdf.p(
        "A clinical brain MRI collection of 1,105 patient studies acquired between "
        "December 2021 and October 2024. The corpus is dominated by a GE SIGNA "
        "Pioneer 3.0T protocol but also includes paired Philips Achieva 1.5T, "
        "Siemens MAGNETOM, and additional GE platforms -- providing a realistic "
        "multi-vendor, multi-field-strength sample of routine neuroradiology practice."
    )
    pdf.p(
        "Each study delivers a multi-sequence brain protocol (typically 14 to 30+ series) "
        "spanning diffusion, perfusion, structural, vascular, and susceptibility imaging. "
        "Every series is enriched with full DICOM tag extraction, volumetric statistics, "
        "automated quality grading, and a composite ML training score."
    )
    pdf.p(
        "Multiplane montages, intensity histograms, contrast-enhanced views, and "
        "structured per-series JSON detail records are produced for every series."
    )

    # ── SPECIFICATIONS ──────────────────────────────────────────────
    pdf.h1("Specifications")
    pdf.row("Studies", "1,105 patient studies")
    pdf.row("Series", "34,574")
    pdf.row("DICOM Files", "355,133")
    pdf.row("Raw Volume", "~66 GB")
    pdf.row("Acquisition Period", "December 2021 -- October 2024")
    pdf.row("Modality", "MR (brain)")
    pdf.row("Vendors", "GE 86%, Philips 12%, Siemens 2%, Toshiba <1%")
    pdf.row("Field Strengths", "3.0T (80%), 1.5T (20%)")
    pdf.row("Primary Scanner", "GE SIGNA Pioneer 3.0T")
    pdf.row("Primary Software", "PX26.1_R03_2128.b (56%)")
    pdf.row("Primary Coil", "Head 32-channel (50%)")
    pdf.row("Patient Sex", "M 51.6%, F 48.3%")
    pdf.row("Series Per Study", "14 -- 30+")
    pdf.row("Metadata Fields", "240+ DICOM tags per file (incl. vendor-private)")

    # ── VENDOR / SCANNER MIX ────────────────────────────────────────
    pdf.h1("Scanner Mix")
    pdf.p(
        "Distribution of source scanners by file count -- the SIGNA Pioneer is the "
        "majority platform, with the remaining studies providing realistic "
        "cross-vendor and cross-field-strength variation:"
    )
    pdf.tbl(
        ["Vendor", "Model", "Field", "Files", "Share"],
        [
            ["GE",       "SIGNA Pioneer",      "3.0T", "262,192", "73.8%"],
            ["Philips",  "Achieva",            "1.5T",  "38,713", "10.9%"],
            ["GE",       "SIGNA Explorer",     "1.5T",  "20,161",  "5.7%"],
            ["GE",       "Signa HDxt",         "1.5T",   "9,154",  "2.6%"],
            ["GE",       "SIGNA Creator",      "1.5T",   "8,378",  "2.4%"],
            ["Siemens",  "MAGNETOM Essenza",   "1.5T",   "5,504",  "1.6%"],
            ["GE",       "Optima MR360",       "1.5T",   "4,804",  "1.4%"],
            ["Philips",  "Achieva dStream",    "1.5T",   "2,588",  "0.7%"],
            ["Philips",  "Spectra",            "1.5T",   "2,010",  "0.6%"],
            ["Philips",  "Ingenia CX",         "3.0T",     "983",  "0.3%"],
            ["Toshiba",  "FILMER 6.0 / MRT200SP3", "1.5T", "619",  "0.2%"],
        ],
        [22, 50, 18, 50, 50],
    )

    # ── PROTOCOL ────────────────────────────────────────────────────
    pdf.h1("Top Sequences (Series Counts)")
    rows = [
        ["1",  "Ax DSC Perfusion",       "47,593", "Dynamic susceptibility contrast"],
        ["2",  "3D Ax SWAN (+FILT_PHA)", "42,974", "Susceptibility-weighted angiography"],
        ["3",  "3D Ax T1 SPGR FS",       "11,296", "3D T1 fat-suppressed"],
        ["4",  "3D SAG T1 SPGR FS",      "10,756", "3D sagittal T1 fat-suppressed"],
        ["5",  "Ax DWI",                  "9,217", "Diffusion-Weighted Imaging"],
        ["6",  "Ax T2 FLAIR",             "6,912", "Fluid-Attenuated Inversion Recovery"],
        ["7",  "Ax T2 PROPELLER",         "5,603", "T2 with motion correction"],
        ["8",  "3D Sag T1 BRAVO",         "4,484", "3D sagittal T1"],
        ["9",  "3D Ax T2 Cube HyperCube", "4,460", "3D isotropic T2"],
        ["10", "3D Ax TOF NECK",          "4,236", "Neck angiography"],
        ["11", "BRAIN ANGIO",             "3,814", "MR Angiography (TOF)"],
        ["12", "ADC (10^-6 mm^2/s)",      "3,621", "ADC map"],
        ["13", "AxT1 MEMP",               "3,152", "T1 Multi-Echo Multi-Phase"],
        ["14", "Ax DWI ALL B-1000",       "3,126", "Diffusion isotropic"],
        ["15", "3D Sag T2 Cube",          "3,068", "3D isotropic T2"],
    ]
    pdf.tbl(["#", "Sequence", "Series", "Description"],
            rows, [10, 60, 30, 90])
    pdf.p(
        "Additional sequences include eADC, 3D Sag MRV, NECK ANGIO, COR T2 FSE, "
        "DTI 27-direction tensor, B-FFE / CISS, GRE, T2W AXIAL, T1W AXIAL, and "
        "vendor-derived processed maps (Reg-DWI, isoB1000, dADC, MIP / projection)."
    )

    # ── CLINICAL CASE MIX ──────────────────────────────────────────
    pdf.h1("Clinical Case Mix")
    pdf.p("Top study descriptions across the corpus:")
    pdf.tbl(
        ["Study Description", "Studies (file count)"],
        [
            ["BRAIN",                       "37,133"],
            ["BRAIN P+C",                   "23,723"],
            ["BRAIN P + C",                 "15,833"],
            ["BRAIN CE",                    "14,879"],
            ["BRAIN ANGIO",                 "13,010"],
            ["BRAIN P+CE",                  "11,354"],
            ["BRAIN P",                      "7,634"],
            ["BRAIN PLAIN",                  "5,160"],
            ["BRAIN P+C DTI+CSF FLOW",       "5,125"],
            ["BRAIN CONTRAST",               "5,007"],
            ["BRAIN CE TRACTOGRAM",          "4,797"],
            ["BRAIN EPILEPSY",               "4,788"],
            ["BRAIN +ANGIO",                 "8,906"],
        ],
        [120, 70],
    )

    # ── IMAGERY ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Sample Imagery")

    montages = [
        ("output/000_s0005_Ax_DWI/s0005_Ax_DWI_multiplane.png", "Axial DWI (b=1000)"),
        ("output/000_s0011_AxT1_MEMP/s0011_AxT1_MEMP_multiplane.png", "Axial T1 MEMP"),
        ("output/000_s0013_BRAIN_ANGIO/s0013_BRAIN_ANGIO_multiplane.png", "Brain MR Angiography"),
        ("output/000_s0009_3D_Ax_SWAN/s0009_3D_Ax_SWAN_multiplane.png", "3D SWAN"),
        ("output/000_s0010_Ax_T2_PROPELLER/s0010_Ax_T2_PROPELLER_multiplane.png", "Axial T2 PROPELLER"),
        ("output/000_s0012__SAG_T2/s0012__SAG_T2_multiplane.png", "Sagittal T2"),
        ("output/000_s0015_COR_DWI/s0015_COR_DWI_multiplane.png", "Coronal DWI"),
        ("output/000_s0017_3D_Ax_TOF_NECK/s0017_3D_Ax_TOF_NECK_multiplane.png", "3D TOF Neck"),
        ("output/000_s0018_3D_Ax_T2_Cube_HyperCube./s0018_3D_Ax_T2_Cube_HyperCube._multiplane.png", "3D T2 Cube"),
        ("output/000_s1001_B_FFE/s1001_B_FFE_multiplane.png", "B-FFE (CISS)"),
    ]
    for path, cap in montages:
        if Path(path).exists():
            pdf.img(path, cap, w=178)

    # Quality chart
    pdf.add_page()
    pdf.h1("Quality Analytics")
    pdf.p("Cross-series comparison of SNR, entropy, tissue coverage, and dynamic range:")
    pdf.img("output/cross_series_comparison.png", "Cross-series quality profile", w=178)

    hists = [
        ("output/000_s0005_Ax_DWI/s0005_Ax_DWI_histogram.png", "DWI"),
        ("output/000_s0007_Ax_T2_FLAIR/s0007_Ax_T2_FLAIR_histogram.png", "T2 FLAIR"),
        ("output/000_s0011_AxT1_MEMP/s0011_AxT1_MEMP_histogram.png", "T1 MEMP"),
    ]
    for path, cap in hists:
        if Path(path).exists():
            pdf.img(path, f"{cap} intensity distribution", w=130)

    # ── DATA ENRICHMENT ─────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Data Enrichment")

    pdf.h2("Metadata")
    pdf.p(
        "240+ DICOM tag columns extracted per file including acquisition parameters "
        "(TR, TE, TI, FA, b-value, bandwidth), spatial geometry (spacing, "
        "orientation, position), demographics, and vendor-private tags (GE groups "
        "0009, 0019, 0021, 0023, 0025, 0027, 0029, 0043, 0051; Philips and "
        "Siemens equivalents preserved where present)."
    )

    pdf.h2("Volumetric Statistics")
    pdf.tbl(["Category", "Metrics"],
            [["Geometry", "Shape, spacing, origin, direction cosines, voxel volume, FOV"],
             ["Intensity", "Min / max / mean / std / median / percentiles (p1--p99) / IQR / dynamic range"],
             ["Signal", "SNR, CNR, background noise, Otsu threshold"],
             ["Distribution", "Entropy, skewness, kurtosis, tissue coverage"],
             ["Per-Slice", "Slice SNR (mean / std / min / max), uniformity"]],
            [28, 162])

    pdf.h2("Quality Assessment")
    pdf.tbl(["Metric", "Description"],
            [["Grade (A--F)", "Composite from SNR, CNR, uniformity, tissue coverage, dynamic range, entropy, nonzero coverage"],
             ["Anomaly Detection", "Per-slice z-scores identifying artifact or dropout"],
             ["Symmetry", "Hemispheric symmetry index"],
             ["Sharpness", "Laplacian edge variance (mean / std / p95)"],
             ["Motion", "Ghosting ratio, directional energy, slice correlation"]],
            [30, 160])

    pdf.h2("ML Training Score")
    pdf.p("Composite 0--100 score from six weighted quality metrics:")
    pdf.tbl(["Metric", "Weight", "Measures"],
            [["CNR", "25", "Tissue-to-background contrast-to-noise ratio"],
             ["Noise Floor", "20", "Rician noise model estimation"],
             ["Bias Field", "15", "B1 inhomogeneity severity"],
             ["Sharpness", "15", "Laplacian variance"],
             ["Histogram Sep.", "10", "Tissue class peak separation"],
             ["Consistency", "15", "Adjacent-slice correlation"]],
            [30, 14, 146])

    pdf.tbl(["Grade", "Score", "Tier"],
            [["A", "80--100", "Premium"],
             ["B", "65--79", "Standard"],
             ["C", "50--64", "Usable"],
             ["D", "35--49", "Limited"],
             ["F", "< 35", "Exclude"]],
            [20, 30, 140])

    # ── DELIVERABLES ────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Deliverables")

    pdf.h2("Per Series")
    pdf.tbl(["File", "Contents"],
            [["{name}_detail.json", "Metadata, quality grades, volume statistics, ML training score"],
             ["{name}_multiplane.png", "Three-plane montage (axial + coronal + sagittal)"],
             ["{name}_histogram.png", "Intensity distribution"],
             ["{name}_enhanced.png", "Contrast-enhanced view"],
             ["{name}.mcap", "MCAP container of the series + metadata"]],
            [50, 140])

    pdf.h2("Per Study")
    pdf.tbl(["File", "Contents"],
            [["cross_series_comparison.png", "Multi-metric quality chart"],
             ["dicom_study.mcap", "Full study MCAP record"]],
            [52, 138])

    pdf.h2("Dataset Level")
    pdf.tbl(["File", "Contents"],
            [["dicom_metadata.csv / .parquet", "Per-file tabular metadata (240+ columns)"],
             ["dicom_full_dump.json", "Complete DICOM tag dump (~4.2 GB)"],
             ["series_stats.json", "Series-level aggregate statistics"],
             ["report.html", "Interactive HTML dashboard"]],
            [50, 140])

    # ── DIFFERENTIATORS ─────────────────────────────────────────────
    pdf.h1("Differentiators")

    diffs = [
        ("Multi-Vendor, Multi-Field Realism",
         "GE, Philips, Siemens, and Toshiba scanners across 3.0T and 1.5T -- representative of routine neuroradiology variability rather than a sanitized single-scanner sample."),
        ("Dominant 3.0T Backbone",
         "262,192 files (74%) come from a single GE SIGNA Pioneer 3.0T with a 32-channel head coil, providing a homogeneous core for controlled experiments."),
        ("Dense Multi-Sequence Coverage",
         "14--30+ series per study including DWI, perfusion (DSC), FLAIR, SWAN, TOF, multi-Cube 3D T1/T2, DTI, MRV, and vendor processed maps."),
        ("Complete Metadata",
         "240+ DICOM tags per file with vendor-private tags preserved -- full provenance chain intact for reproducibility."),
        ("Pre-Scored for ML",
         "Every series graded A--F and tiered. Immediate data selection without manual review."),
        ("South Asian Representation",
         "Addresses demographic gaps in neuroimaging datasets dominated by Western cohorts. Sex distribution near-balanced (M 51.6% / F 48.3%)."),
    ]
    for title, desc in diffs:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*K)
        pdf.cell(0, 5.5, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*D)
        pdf.multi_cell(0, 4.5, desc)
        pdf.ln(2)

    # ── APPLICATIONS ────────────────────────────────────────────────
    pdf.h1("Applications")
    apps = [
        ("Segmentation", "Gray/white matter, CSF, lesion segmentation"),
        ("Pathology Detection", "Stroke, hemorrhage, neoplasm, white matter disease, epilepsy"),
        ("Sequence Classification", "30+ sequence types for classification benchmarks"),
        ("Quality Assessment", "IQA development against pre-computed metrics"),
        ("Diffusion / Perfusion", "DWI/ADC + DSC perfusion with paired structural imaging"),
        ("Vascular Analysis", "Intracranial and cervical MRA + MRV data"),
        ("Cross-Vendor Robustness", "Train and evaluate on GE / Philips / Siemens distribution shift"),
        ("Multi-Contrast Learning", "Cross-sequence contrastive and self-supervised learning"),
    ]
    for title, desc in apps:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*M)
        pdf.cell(48, 5, title)
        pdf.set_text_color(*D)
        pdf.cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")

    # ── SAMPLE RECORD ───────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Sample Record (Ax DWI from a SIGNA Pioneer study)")
    pdf.set_font("Courier", "", 6.5)
    pdf.set_text_color(*D)
    lines = [
        '{',
        '  "series_description": "Ax DWI",',
        '  "series_number": 5,',
        '  "modality": "MR",',
        '  "file_count": 50,',
        '  "sequence_classification": {',
        '    "sequence_type": "DWI",',
        '    "confidence": "high",',
        '    "reasoning": ["Name matches \'\\\\bDWI\\\\b\'", "b-value=1000.0 confirms diffusion"]',
        '  },',
        '  "sequence_params": {',
        '    "tr": 6034.0, "te": 74.5, "fa": 90.0, "b_value": 1000.0',
        '  },',
        '  "volume_stats": {',
        '    "volume_shape": [50, 256, 256],',
        '    "spacing_mm": [1.094, 1.094, 2.939],',
        '    "fov_mm": [146.9, 280.0, 280.0],',
        '    "volume_snr_estimate": 1.90,',
        '    "volume_cnr": 1769.0,',
        '    "volume_entropy": 1.67,',
        '    "volume_tissue_pct": 5.81',
        '  },',
        '  "quality_analysis": {',
        '    "quality_grade": { "grade": "D", "score": 44.5,',
        '      "breakdown": { "snr": 5.9, "cnr": 10.0, "uniformity": 3.8,',
        '                     "tissue_coverage": 1.5, "dynamic_range": 12.0,',
        '                     "entropy": 3.3, "nonzero_coverage": 8.0 } },',
        '    "anomaly_detection": { "n_anomalous": 0 },',
        '    "sharpness_analysis": { "interpretation": "very sharp" },',
        '    "motion_analysis": { "motion_severity_score": 35.4 }',
        '  },',
        '  "ml_training_score": {',
        '    "score": 72.5, "grade": "B", "commercial_tier": "standard"',
        '  }',
        '}',
    ]
    for line in lines:
        pdf.cell(0, 3.3, line, new_x="LMARGIN", new_y="NEXT")

    # ── PROVENANCE ─────────────────────────────────────────────────
    pdf.h1("Provenance")
    pdf.p(
        "Numbers in this prospectus are derived from a complete sweep of the source "
        "DICOM corpus on 2026-04-30 (355,133 files, 0 read errors) and cross-checked "
        "against the modal pipeline output. PHI has been redacted; institution names "
        "are stripped from the source headers."
    )
    pdf.p(
        "Acquisition window: earliest StudyDate 2021-12-16, latest 2024-10-31 "
        "(450 distinct study dates spanning ~2.9 years)."
    )

    # Save
    pdf.output(str(PDF_PATH))
    sz = PDF_PATH.stat().st_size
    print(f"{PDF_PATH}  |  {sz / 1024 / 1024:.1f} MB  |  {pdf.page_no()} pages")


if __name__ == "__main__":
    build()
