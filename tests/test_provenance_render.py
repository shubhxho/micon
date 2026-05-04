"""Tests for the buyer-facing provenance stage-graph renderer."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from PIL import Image

from src.integrity.provenance_render import (
    build_stage_graph,
    to_mermaid,
    to_png,
)


def test_build_stage_graph_shape() -> None:
    g = build_stage_graph()
    assert len(g["nodes"]) == 7
    assert len(g["edges"]) == 6
    ids = [nid for nid, _ in g["nodes"]]
    assert ids == [
        "INGEST",
        "DEFACE",
        "METADATA",
        "VALIDATE",
        "QUALITY",
        "MANIFEST",
        "EXPORT",
    ]
    # Edges form a chain
    expected = list(pairwise(ids))
    assert g["edges"] == expected


def test_to_mermaid_syntax() -> None:
    g = build_stage_graph()
    src = to_mermaid(g)
    assert src.startswith("flowchart LR"), "Mermaid output must start with 'flowchart'"
    # Every stage id appears
    for nid, _ in g["nodes"]:
        assert nid in src
    # At least one arrow
    assert "-->" in src


def test_to_png_writes_valid_png(tmp_path: Path) -> None:
    g = build_stage_graph()
    out = tmp_path / "prov.png"
    returned = to_png(g, out)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.format == "PNG"
        # Width should respect the requested inches roughly.
        assert im.width > 100
        assert im.height > 50


def test_to_png_empty_graph_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        to_png({"nodes": [], "edges": []}, tmp_path / "x.png")
