"""Stage-level chain-of-custody graph renderer.

The full PROV-JSON document produced by :mod:`src.integrity.provenance` may
contain thousands of per-DICOM entities and is unsuitable for a buyer-facing
prospectus. This module exposes a *stage graph* -- a small, fixed left-to-right
flowchart of the pipeline stages every series passes through -- and renders it
to either Mermaid source (text) or a PNG (image embedded in the PDF).

Public API:
    build_stage_graph()           -> dict[str, list]
    to_mermaid(graph)             -> str
    to_png(graph, out, *, ...)    -> Path
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TypedDict


class StageGraph(TypedDict):
    """Lightweight directed graph of pipeline stages.

    nodes: list of (id, human label) tuples in left-to-right order.
    edges: list of (from_id, to_id) tuples describing flow.
    """

    nodes: list[tuple[str, str]]
    edges: list[tuple[str, str]]


# Canonical pipeline stages (matches BRAINSTORM / pipeline modules).
_STAGES: list[tuple[str, str]] = [
    ("INGEST", "Raw DICOM\ningest"),
    ("DEFACE", "De-identify\n& deface"),
    ("METADATA", "Metadata\nextract"),
    ("VALIDATE", "Schema\nvalidate"),
    ("QUALITY", "Quality\ngrade"),
    ("MANIFEST", "Manifest\n& checksums"),
    ("EXPORT", "Package\n& upload"),
]


def build_stage_graph() -> StageGraph:
    """Return the canonical buyer-facing pipeline stage graph."""
    nodes = list(_STAGES)
    edges = [(a[0], b[0]) for a, b in pairwise(nodes)]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Mermaid emitter
# ---------------------------------------------------------------------------


def to_mermaid(graph: StageGraph) -> str:
    """Emit Mermaid ``flowchart LR`` source for *graph*.

    The output is deterministic and human-readable -- safe to embed in markdown
    docs, the PDF as a code block, or piped to a Mermaid renderer.
    """
    labels = {nid: label.replace("\n", " ") for nid, label in graph["nodes"]}
    lines = ["flowchart LR"]
    for nid, _label in graph["nodes"]:
        # Use one-line labels in mermaid (no embedded newlines)
        lines.append(f"    {nid}[{labels[nid]}]")
    for src, dst in graph["edges"]:
        lines.append(f"    {src} --> {dst}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PNG renderer (matplotlib, headless)
# ---------------------------------------------------------------------------


def to_png(
    graph: StageGraph,
    out: Path,
    *,
    width_in: float = 7.5,
    dpi: int = 200,
) -> Path:
    """Render *graph* to a deterministic PNG using matplotlib.

    Nodes are placed left-to-right at fixed positions so successive runs
    produce byte-stable output (modulo matplotlib font hinting).
    """
    # Headless backend BEFORE pyplot import to avoid display init in CI.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    nodes = graph["nodes"]
    edges = graph["edges"]
    n = len(nodes)
    if n == 0:
        raise ValueError("Cannot render an empty stage graph")

    height_in = max(1.4, width_in * 0.22)
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Deterministic positions: node i centered at x = i + 0.5, y = 0.5.
    box_w = 0.78
    box_h = 0.55
    positions: dict[str, tuple[float, float]] = {}

    ink = "#121620"
    accent = "#1c468c"
    wash = "#f8f9fb"
    rule = "#dcdee4"

    for i, (nid, label) in enumerate(nodes):
        cx = i + 0.5
        cy = 0.5
        positions[nid] = (cx, cy)
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor=rule,
            facecolor=wash,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=ink,
            family="sans-serif",
        )

    for src, dst in edges:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate(
            "",
            xy=(dx - box_w / 2, dy),
            xytext=(sx + box_w / 2, sy),
            arrowprops={
                "arrowstyle": "->",
                "color": accent,
                "lw": 1.2,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
