r"""Tests for the _safe_name helper in resume_pipeline.py and src/dry_run.py.

Property test: for any string label, _safe_name(label) contains only
characters that match [a-zA-Z0-9_-] (the regex character class [\w\-]).

Concrete test: the em-dash variant used in series labels collapses to three
underscores so that existing on-disk annotation filenames remain stable
across any future formatting edits.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import from src.dry_run (which is the standalone copy used in tests without
# Modal). resume_pipeline.py defines an identical _safe_name but imports modal
# at module level, making it unsuitable for unit tests.
from src.dry_run import _safe_name

# The expected output character class: word chars (\w) plus hyphens.
_ALLOWED_RE = re.compile(r"^[\w\-]*$")


# ---------------------------------------------------------------------------
# Hypothesis property test
# ---------------------------------------------------------------------------


@given(label=st.text(max_size=200))
@settings(max_examples=500)
def test_safe_name_output_only_allowed_chars(label: str) -> None:
    """_safe_name(label) must only contain [a-zA-Z0-9_-] for ALL inputs."""
    result = _safe_name(label)
    assert _ALLOWED_RE.match(result) is not None, (
        f"_safe_name({label!r}) => {result!r} contains disallowed characters"
    )


# ---------------------------------------------------------------------------
# Concrete em-dash collapse tests (filename stability)
# ---------------------------------------------------------------------------


def test_em_dash_space_collapses_to_three_underscores() -> None:
    """' \u2014 ' (space + em-dash + space) becomes '___' (three underscores).

    This pins the on-disk filename encoding so changing whitespace around the
    em-dash in the label-building code does not silently orphan existing
    annotation JSON files.
    """
    label = "Series 5 \u2014 Ax DWI"
    result = _safe_name(label)
    assert result == "Series_5___Ax_DWI", (
        f"em-dash collapse changed -- existing annotation files would be orphaned. Got: {result!r}"
    )


def test_em_dash_without_spaces_collapses_to_one_underscore() -> None:
    """'\u2014' (em-dash alone, no surrounding spaces) -> single underscore."""
    label = "Series5\u2014AxDWI"
    result = _safe_name(label)
    assert result == "Series5_AxDWI"


@pytest.mark.parametrize(
    "label,expected",
    [
        # The canonical pipeline format: space + em-dash + space
        ("Series 10 \u2014 T1", "Series_10___T1"),
        # Another real label from the annotation path
        ("Series 9 \u2014 MIP", "Series_9___MIP"),
        # Plain spaces only
        ("Series 3 T1 MPRAGE", "Series_3_T1_MPRAGE"),
        # Already safe -- no change
        ("Series_3_T1", "Series_3_T1"),
        # Hyphens are allowed and preserved
        ("Series-3-T1", "Series-3-T1"),
    ],
)
def test_safe_name_concrete_cases(label: str, expected: str) -> None:
    assert _safe_name(label) == expected
