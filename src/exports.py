"""Image export functions — montages, histograms, cross-series comparison.

Montages and histograms use PIL (lock-free, fully parallel across threads).
Only cross-series comparison uses matplotlib (runs once, not per-series).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

from ._logging import get_logger
from .helpers import get_2d_slice, safe_squeeze

logger = get_logger(__name__)

_mpl_lock = threading.Lock()
compress_images = False

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch
]


def _load_fonts(sizes: dict[str, int]) -> dict[str, ImageFont.FreeTypeFont]:
    """Load fonts at given sizes, trying multiple paths. Falls back to default."""
    for path in _FONT_PATHS:
        try:
            return {name: ImageFont.truetype(path, sz) for name, sz in sizes.items()}
        except OSError:
            continue
    # Fallback to PIL's bundled bitmap font; cast to FreeTypeFont so callers can
    # treat the return type uniformly (PIL accepts either at draw time).
    default = ImageFont.load_default()
    return dict.fromkeys(sizes, default)  # type: ignore[arg-type]


# ── PIL montage (lock-free — fully parallel) ─────────────────────────────────


def _slice_to_pil(slc: np.ndarray, cell_w: int, cell_h: int) -> Image.Image:
    """Convert a 2D numpy slice to a resized PIL grayscale image."""
    while slc.ndim > 2:
        slc = slc[0]
    img = Image.fromarray((np.clip(slc, 0, 1) * 255).astype(np.uint8), mode="L")
    return img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)


def _extract_plane_slices(
    vol_norm: np.ndarray, plane: str, n_per_plane: int
) -> tuple[str, int, list[np.ndarray], np.ndarray]:
    """Extract slices for a single plane — numpy only, GIL-releasing."""
    nz, ny, nx = vol_norm.shape[:3]
    if plane == "Axial":
        n = nz
        indices = (
            np.linspace(max(n // 8, 0), min(n - n // 8, n - 1), n_per_plane, dtype=int)
            if n > 1
            else np.array([0])
        )
        slices = [get_2d_slice(vol_norm, i) if i < nz else np.zeros((ny, nx)) for i in indices]
    elif plane == "Coronal":
        n = ny
        indices = (
            np.linspace(max(n // 8, 0), min(n - n // 8, n - 1), n_per_plane, dtype=int)
            if n > 1
            else np.array([0])
        )
        slices = [
            vol_norm[:, min(i, ny - 1), :] if vol_norm.ndim >= 3 else vol_norm[0] for i in indices
        ]
    else:  # Sagittal
        n = nx
        indices = (
            np.linspace(max(n // 8, 0), min(n - n // 8, n - 1), n_per_plane, dtype=int)
            if n > 1
            else np.array([0])
        )
        slices = [
            vol_norm[:, :, min(i, nx - 1)] if vol_norm.ndim >= 3 else vol_norm[0] for i in indices
        ]
    return plane, n, slices, indices


def export_multiplane_montage(
    vol: np.ndarray,
    series_name: str,
    out_dir: str,
    n_per_plane: int = 6,
    vol_stats: dict | None = None,
) -> str:
    """Export multiplane montage using PIL — no matplotlib lock, fully parallel."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    vol = safe_squeeze(vol)

    from .metal import gpu_available, gpu_normalize

    if gpu_available() and vol.size >= 16_384:
        vol_norm = gpu_normalize(vol, 1, 99)
    else:
        vmin, vmax = np.percentile(vol, [1, 99])
        vol_norm = np.clip((vol - vmin) / (vmax - vmin + 1e-6), 0, 1).astype(np.float32)
    if vol_norm.ndim < 3:
        vol_norm = vol_norm[np.newaxis, ...]

    nz, ny, nx = vol_norm.shape[:3]

    # Extract planes in parallel threads (numpy releases GIL)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_extract_plane_slices, vol_norm, p, n_per_plane)
            for p in ("Axial", "Coronal", "Sagittal")
        ]
        plane_defs = [f.result() for f in futs]

    # Layout constants — higher resolution cells
    cell_w, cell_h = 300, 300
    pad = 3
    label_w = 80
    header_h = 65
    total_w = label_w + n_per_plane * (cell_w + pad) + pad
    total_h = header_h + 3 * (cell_h + pad + 18) + pad

    canvas = Image.new("RGB", (total_w, total_h), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)

    fonts = _load_fonts({"title": 20, "label": 13, "small": 11, "grade": 28})
    font_title, font_label, font_small = fonts["title"], fonts["label"], fonts["small"]
    font_grade = fonts["grade"]

    # Header: series name + grade badge + stats
    display_name = series_name.replace("_", " ")
    grade = vol_stats.get("quality_grade", "") if vol_stats else ""
    grade_colors = {
        "A": (76, 175, 80),
        "B": (139, 195, 74),
        "C": (255, 193, 7),
        "D": (255, 87, 34),
        "F": (244, 67, 54),
    }

    # Grade badge (right side)
    if grade and grade in grade_colors:
        gc = grade_colors[grade]
        badge_x = total_w - 55
        draw.rounded_rectangle([badge_x, 8, badge_x + 44, 52], radius=8, fill=gc)
        draw.text((badge_x + 22, 30), grade, fill=(255, 255, 255), font=font_grade, anchor="mm")

    draw.text((total_w // 2, 10), display_name, fill=(0, 220, 240), font=font_title, anchor="mt")
    if vol_stats:
        snr = vol_stats.get("volume_snr_estimate", 0)
        snr_color = (76, 175, 80) if snr >= 20 else (255, 193, 7) if snr >= 5 else (244, 67, 54)
        stats = (
            f"Volume: {nz}x{ny}x{nx}  |  "
            f"Tissue: {vol_stats.get('volume_tissue_pct', 0):.0f}%  |  "
            f"Entropy: {vol_stats.get('volume_entropy', 0):.1f} bits"
        )
        draw.text((total_w // 2, 34), stats, fill=(139, 148, 158), font=font_small, anchor="mt")
        draw.text(
            (total_w // 2, 50), f"SNR: {snr:.1f}", fill=snr_color, font=font_small, anchor="mt"
        )

    # Collect all slices across all planes, convert in a single flat pool
    # (avoids nested pool oversubscription: 18 series × 6 workers = 108 threads)
    all_slices = []
    slice_meta = []  # (row, col, plane_name, n_total, idx)
    for row, (plane_name, n_total, slices, indices) in enumerate(plane_defs):
        for col, slc in enumerate(slices[:n_per_plane]):
            all_slices.append(slc)
            slice_meta.append(
                (row, col, plane_name, n_total, indices[col] if col < len(indices) else 0)
            )

    with ThreadPoolExecutor(max_workers=min(len(all_slices), 6)) as pool:
        pil_imgs = list(pool.map(lambda s: _slice_to_pil(s, cell_w, cell_h), all_slices))

    for (row, col, plane_name, n_total, idx), pil_img in zip(slice_meta, pil_imgs, strict=False):
        y_base = header_h + row * (cell_h + pad + 16)
        if col == 0:
            draw.text(
                (4, y_base + cell_h // 2),
                plane_name,
                fill=(0, 255, 255),
                font=font_label,
                anchor="lm",
            )
        x = label_w + col * (cell_w + pad)
        canvas.paste(pil_img.convert("RGB"), (x, y_base))
        draw.text(
            (x + cell_w // 2, y_base + cell_h + 2),
            f"{idx}/{n_total}",
            fill=(170, 170, 170),
            font=font_small,
            anchor="mt",
        )

    p = str(out_path / f"{series_name}_multiplane.png")
    canvas.save(p)
    if compress_images:
        _png_to_webp(p)
    return p


# ── Histogram (PIL — lock-free, fully parallel) ──────────────────────────────


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    counts: np.ndarray,
    edges: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    bar_color: tuple[int, int, int],
    log_scale: bool,
    pcts: np.ndarray | None,
    title: str,
    font: ImageFont.FreeTypeFont,
    font_sm: ImageFont.FreeTypeFont,
    stats_text: str | None = None,
) -> None:
    """Draw a single histogram panel on a PIL canvas."""
    # Background with subtle rounded border
    draw.rounded_rectangle(
        [x0, y0, x0 + w, y0 + h], radius=6, fill=(13, 17, 23), outline=(30, 36, 44), width=1
    )

    pad_l, pad_r, pad_t, pad_b = 55, 15, 28, 30
    plot_x = x0 + pad_l
    plot_y = y0 + pad_t
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    # Title
    draw.text((x0 + w // 2, y0 + 6), title, fill=(201, 209, 217), font=font, anchor="mt")

    vals = counts.astype(np.float64)
    if log_scale:
        with np.errstate(divide="ignore"):
            vals = np.where(vals > 0, np.log10(np.maximum(vals, 1)), 0)
    max_val = vals.max() if vals.max() > 0 else 1

    # Bars
    n = len(vals)
    bar_w = max(plot_w / n, 1)
    for i, v in enumerate(vals):
        if v <= 0:
            continue
        bx = plot_x + int(i * plot_w / n)
        bh = int(v / max_val * plot_h)
        by = plot_y + plot_h - bh
        draw.rectangle([bx, by, bx + max(int(bar_w), 1), plot_y + plot_h], fill=bar_color)

    # Axes
    draw.line([plot_x, plot_y, plot_x, plot_y + plot_h], fill=(48, 54, 61), width=1)
    draw.line(
        [plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h], fill=(48, 54, 61), width=1
    )

    # Y-axis labels
    ylabel = "Log Count" if log_scale else "Count"
    draw.text(
        (x0 + 4, plot_y + plot_h // 2), ylabel, fill=(139, 148, 158), font=font_sm, anchor="lm"
    )

    # Percentile markers
    if pcts is not None:
        emin, emax = float(edges[0]), float(edges[-1])
        rng = emax - emin if emax > emin else 1
        for pval, lbl, clr in [
            (pcts[0], "P5", (244, 67, 54)),
            (pcts[2], "P50", (76, 175, 80)),
            (pcts[4], "P95", (244, 67, 54)),
        ]:
            px = plot_x + int((pval - emin) / rng * plot_w)
            if plot_x <= px <= plot_x + plot_w:
                draw.line([px, plot_y, px, plot_y + plot_h], fill=clr, width=1)
                draw.text((px, plot_y - 2), lbl, fill=clr, font=font_sm, anchor="mb")

    # Stats box (multiline — draw line by line, no anchor)
    if stats_text:
        sy = plot_y + 4
        for line in stats_text.split("\n"):
            draw.text((x0 + w - pad_r - 4, sy), line, fill=(139, 148, 158), font=font_sm)
            sy += 12


def export_histogram(
    vol: np.ndarray, series_name: str, out_dir: str, vol_stats: dict | None = None
) -> str:
    """Export intensity histogram using PIL — no matplotlib lock, fully parallel."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    display_name = series_name.replace("_", " ")

    from .metal import gpu_histogram_and_percentiles

    flat = vol.ravel()
    hist_counts, hist_edges, pcts = gpu_histogram_and_percentiles(
        flat,
        bins=200,
        pcts=[5, 25, 50, 75, 95],
    )

    # Layout — wider panels
    panel_w, panel_h = 640, 360
    pad = 10
    header_h = 50
    total_w = panel_w * 2 + pad * 3
    total_h = header_h + panel_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)

    fonts = _load_fonts({"font": 12, "title": 16, "sm": 10})
    font, font_title, font_sm = fonts["font"], fonts["title"], fonts["sm"]

    # Header with grade
    grade = vol_stats.get("quality_grade", "") if vol_stats else ""
    grade_colors = {
        "A": (76, 175, 80),
        "B": (139, 195, 74),
        "C": (255, 193, 7),
        "D": (255, 87, 34),
        "F": (244, 67, 54),
    }

    draw.text((total_w // 2, 10), display_name, fill=(0, 220, 240), font=font_title, anchor="mt")

    if grade and grade in grade_colors:
        gc = grade_colors[grade]
        draw.text((total_w - 40, 10), grade, fill=gc, font=font_title, anchor="mt")

    # Stats line under header
    if vol_stats and pcts is not None:
        stats_line = (
            f"Mean: {vol_stats.get('volume_mean', 0):.1f}  |  "
            f"Std: {vol_stats.get('volume_std', 0):.1f}  |  "
            f"IQR: {pcts[3] - pcts[1]:.1f}  |  "
            f"Range: {vol_stats.get('volume_dynamic_range', 0):.0f}"
        )
        draw.text((total_w // 2, 32), stats_line, fill=(100, 110, 120), font=font_sm, anchor="mt")

    # Stats text for log panel
    stats_text = None
    if vol_stats:
        stats_text = (
            f"SNR: {vol_stats.get('volume_snr_estimate', 0):.1f}\n"
            f"Tissue: {vol_stats.get('volume_tissue_pct', 0):.0f}%\n"
            f"Entropy: {vol_stats.get('volume_entropy', 0):.1f} bits"
        )

    # Draw two panels: linear + log
    _draw_bar_chart(
        draw,
        hist_counts,
        hist_edges,
        pad,
        header_h,
        panel_w,
        panel_h,
        (33, 150, 243),
        False,
        pcts,
        "Linear Scale",
        font,
        font_sm,
    )
    _draw_bar_chart(
        draw,
        hist_counts,
        hist_edges,
        panel_w + pad * 2,
        header_h,
        panel_w,
        panel_h,
        (255, 152, 0),
        True,
        pcts,
        "Log Scale",
        font,
        font_sm,
        stats_text,
    )

    p = str(out_dir_p / f"{series_name}_histogram.png")
    canvas.save(p)
    if compress_images:
        _png_to_webp(p)
    return p


# ── Cross-series comparison (matplotlib) ─────────────────────────────────────


def export_cross_series_comparison(series_data: dict[str, dict], out_dir: str) -> str | None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    names, snrs, entropies, tissue_pcts, dyn_ranges = [], [], [], [], []

    for name, info in sorted(series_data.items()):
        vs = info.get("vstats")
        if not vs:
            continue
        names.append(info.get("label", name[:20]))
        snrs.append(vs.get("volume_snr_estimate", 0))
        entropies.append(vs.get("volume_entropy", 0))
        tissue_pcts.append(vs.get("volume_tissue_pct", 0))
        dyn_ranges.append(vs.get("volume_dynamic_range", 0))

    if not names:
        return None

    with _mpl_lock:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        x = np.arange(len(names))
        for ax, vals, title, color in [
            (axes[0, 0], snrs, "SNR Estimate", "#4CAF50"),
            (axes[0, 1], entropies, "Entropy (bits)", "#2196F3"),
            (axes[1, 0], tissue_pcts, "Tissue Coverage %", "#FF9800"),
            (axes[1, 1], dyn_ranges, "Dynamic Range", "#9C27B0"),
        ]:
            ax.barh(x, vals, 0.6, color=color, alpha=0.8)
            ax.set_yticks(x)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title(title, fontweight="bold")
            ax.invert_yaxis()
            for i, v in enumerate(vals):
                ax.text(v + max(vals) * 0.01, i, f"{v:.1f}", va="center", fontsize=7)

        fig.suptitle("Cross-Series Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        p = str(out_path / "cross_series_comparison.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        if compress_images:
            _png_to_webp(p)
        plt.close(fig)
    return p


# ── GPU-rendered enhanced views for VLM analysis ─────────────────────────────


def export_enhanced_views(
    vol: np.ndarray, series_name: str, out_dir: str, vol_stats: dict | None = None
) -> str | None:
    """Export GPU-rendered enhanced composite for VLM analysis.

    Generates a single image with:
      Row 1: MIP projections (Axial, Coronal, Sagittal) — reveals vessels, bright lesions
      Row 2: MinIP projections — reveals CSF spaces, dark lesions, cysts
      Row 3: Tissue mask overlaid on mid-slices — shows segmentation quality

    All rendering done on Metal GPU when available.
    Returns path to the composite PNG, or None if volume is too small.
    """
    from .helpers import safe_squeeze
    from .metal import gpu_batch_enhanced

    vol = safe_squeeze(vol)
    if vol.ndim < 3 or vol.shape[0] < 3:
        return None

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = vol.shape[:3]

    # Single GPU upload + lock for all 4 operations (was 4 separate uploads)
    (mip_ax, mip_cor, mip_sag), (minip_ax, minip_cor, minip_sag), tissue_mask, vol_norm = (
        gpu_batch_enhanced(vol, 1, 99, 15, 99)
    )

    def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
        mn, mx_val = arr.min(), arr.max()
        if mx_val - mn < 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) / (mx_val - mn) * 255).astype(np.uint8)

    # Layout: 3 rows × 3 columns — larger cells
    cell_w, cell_h = 320, 320
    pad = 4
    label_w = 80
    row_label_h = 20
    header_h = 55
    n_cols = 3
    n_rows = 3
    total_w = label_w + n_cols * (cell_w + pad) + pad
    total_h = header_h + n_rows * (cell_h + pad + row_label_h) + pad

    canvas = Image.new("RGB", (total_w, total_h), (10, 12, 16))
    draw = ImageDraw.Draw(canvas)

    fonts = _load_fonts({"title": 18, "label": 12, "small": 10, "grade": 24})
    font_title, font_label, font_small = fonts["title"], fonts["label"], fonts["small"]

    display_name = series_name.replace("_", " ")
    draw.text(
        (total_w // 2, 8),
        f"{display_name} — Enhanced Views",
        fill=(0, 220, 240),
        font=font_title,
        anchor="mt",
    )

    # Grade badge
    grade = vol_stats.get("quality_grade", "") if vol_stats else ""
    grade_colors = {
        "A": (76, 175, 80),
        "B": (139, 195, 74),
        "C": (255, 193, 7),
        "D": (255, 87, 34),
        "F": (244, 67, 54),
    }
    if grade and grade in grade_colors:
        gc = grade_colors[grade]
        badge_x = total_w - 50
        draw.rounded_rectangle([badge_x, 6, badge_x + 40, 42], radius=6, fill=gc)
        draw.text((badge_x + 20, 24), grade, fill=(255, 255, 255), font=fonts["grade"], anchor="mm")

    if vol_stats:
        info_text = (
            f"Volume: {nz}x{ny}x{nx}  |  "
            f"SNR: {vol_stats.get('volume_snr_estimate', 0):.1f}  |  "
            f"Tissue: {vol_stats.get('volume_tissue_pct', 0):.0f}%"
        )
        draw.text((total_w // 2, 32), info_text, fill=(139, 148, 158), font=font_small, anchor="mt")

    rows_data = [
        ("MIP", [mip_ax, mip_cor, mip_sag], ["Axial", "Coronal", "Sagittal"]),
        ("MinIP", [minip_ax, minip_cor, minip_sag], ["Axial", "Coronal", "Sagittal"]),
        ("Tissue", None, ["Axial", "Coronal", "Sagittal"]),  # special handling
    ]

    # Prepare tissue overlay mid-slices
    mid_slices = [
        vol_norm[nz // 2, :, :],
        vol_norm[:, ny // 2, :],
        vol_norm[:, :, nx // 2],
    ]
    mask_slices = [
        tissue_mask[nz // 2, :, :],
        tissue_mask[:, ny // 2, :],
        tissue_mask[:, :, nx // 2],
    ]

    for row_idx, (row_label, projections, col_labels) in enumerate(rows_data):
        y_base = header_h + row_idx * (cell_h + pad + row_label_h)

        # Row label
        draw.text(
            (4, y_base + cell_h // 2), row_label, fill=(0, 255, 255), font=font_label, anchor="lm"
        )

        for col_idx in range(3):
            x = label_w + col_idx * (cell_w + pad)

            if row_label == "Tissue":
                # Tissue overlay: grayscale base + green mask overlay
                base = _norm_to_uint8(mid_slices[col_idx])
                mask_u8 = (mask_slices[col_idx] * 180).astype(np.uint8)
                while base.ndim > 2:
                    base = base[0]
                while mask_u8.ndim > 2:
                    mask_u8 = mask_u8[0]
                rgb = np.stack(
                    [base, np.clip(base.astype(np.int16) + mask_u8, 0, 255).astype(np.uint8), base],
                    axis=-1,
                )
                pil_img = Image.fromarray(rgb, mode="RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            else:
                proj = projections[col_idx]
                while proj.ndim > 2:
                    proj = proj[0]
                pil_img = (
                    Image.fromarray(_norm_to_uint8(proj), mode="L")
                    .resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    .convert("RGB")
                )

            canvas.paste(pil_img, (x, y_base))
            draw.text(
                (x + cell_w // 2, y_base + cell_h + 2),
                col_labels[col_idx],
                fill=(170, 170, 170),
                font=font_small,
                anchor="mt",
            )

    p = str(out_dir_p / f"{series_name}_enhanced.png")
    canvas.save(p)
    if compress_images:
        _png_to_webp(p)
    return p


# ── NIfTI export ─────────────────────────────────────────────────────────────


def export_nifti(dcm_files: list[str], out_path: str) -> str:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dcm_files)
    reader.SetGlobalWarningDisplay(False)
    image = reader.Execute()
    reader.SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, out_path)
    return out_path


# ── Helpers ──────────────────────────────────────────────────────────────────


def _png_to_webp(png_path: str, quality: int = 80) -> str:
    webp_path = png_path.rsplit(".", 1)[0] + ".webp"
    img = Image.open(png_path)
    img.save(webp_path, "WEBP", quality=quality, method=4)
    return webp_path
