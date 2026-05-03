"""SVG quality badge generator for dataset cards.

Produces self-contained SVG shields for embedding in Hugging Face cards,
README files, or any HTML context.  No external dependencies.

CLI usage:
    python -m src.qc.badge --grade A > badge.svg
"""

from __future__ import annotations

import argparse

_GRADE_COLORS: dict[str, tuple[str, str]] = {
    "A": ("#276749", "#48bb78"),  # dark green / green
    "B": ("#2b6cb0", "#63b3ed"),  # dark blue / blue
    "C": ("#975a16", "#f6e05e"),  # dark yellow / yellow
    "D": ("#c05621", "#f6ad55"),  # dark orange / orange
    "F": ("#9b2c2c", "#fc8181"),  # dark red / red
}

_LABEL = "Quality"
_LABEL_WIDTH = 60
_VALUE_WIDTH = 24


def build_quality_badge(grade: str) -> str:
    """Return a self-contained SVG badge string for the given quality grade.

    Args:
        grade: One of "A", "B", "C", "D", "F".  Unknown grades render grey.

    Returns:
        SVG markup as a string (valid XML, no external deps).
    """
    grade = grade.upper().strip()
    dark, light = _GRADE_COLORS.get(grade, ("#4a5568", "#a0aec0"))
    total_w = _LABEL_WIDTH + _VALUE_WIDTH

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" aria-label="{_LABEL}: {grade}">
  <title>{_LABEL}: {grade}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{_LABEL_WIDTH}" height="20" fill="#555"/>
    <rect x="{_LABEL_WIDTH}" width="{_VALUE_WIDTH}" height="20" fill="{dark}"/>
    <rect width="{total_w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
    <text x="{_LABEL_WIDTH // 2 * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(_LABEL_WIDTH - 10) * 10}" lengthAdjust="spacing">{_LABEL}</text>
    <text x="{_LABEL_WIDTH // 2 * 10}" y="140" transform="scale(.1)" textLength="{(_LABEL_WIDTH - 10) * 10}" lengthAdjust="spacing">{_LABEL}</text>
    <text x="{(_LABEL_WIDTH + _VALUE_WIDTH // 2) * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(_VALUE_WIDTH - 6) * 10}" lengthAdjust="spacing">{grade}</text>
    <text x="{(_LABEL_WIDTH + _VALUE_WIDTH // 2) * 10}" y="140" transform="scale(.1)" textLength="{(_VALUE_WIDTH - 6) * 10}" lengthAdjust="spacing">{grade}</text>
  </g>
</svg>"""


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate an SVG quality badge.")
    parser.add_argument("--grade", required=True, choices=list(_GRADE_COLORS) + [g.lower() for g in _GRADE_COLORS],
                        help="Quality grade: A B C D F")
    args = parser.parse_args()
    print(build_quality_badge(args.grade))


if __name__ == "__main__":
    _cli()
