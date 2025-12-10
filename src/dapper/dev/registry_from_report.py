#!/usr/bin/env python
"""
Generate registry entries from report.rst.

Parses the list-table in report.rst and writes a Python module that
exposes:

    REGISTRY_FROM_REPORT: Dict[str, VarDef]

You can then import that module in dapper.surf.schema and do:

    from .registry_from_report import REGISTRY_FROM_REPORT
    REGISTRY.update(REGISTRY_FROM_REPORT)

Usage
-----
    python gen_registry_from_report.py \
        --input path/to/report.rst \
        --output path/to/registry_from_report.py

If no args are given, defaults to:
    input  = ./report.rst
    output = ./registry_from_report.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any


LIST_TABLE_MARKER = ".. list-table:: Variables in the ELM surface data file"


def clean_text(cell: str) -> str:
    """Strip Sphinx-style citations and collapse whitespace."""
    # remove 【...】 citation markers
    s = re.sub(r"【[^】]+】", "", cell)
    # collapse whitespace (including newlines, NBSP, etc.)
    s = " ".join(s.split())
    return s


def parse_dims_cell(raw_dims: str) -> List[str]:
    """
    Parse the dims cell like ``(grlnd, topounit)`` -> ["grlnd", "topounit"].
    """
    s = raw_dims.strip()
    s = s.replace("``", "").strip()
    s = re.sub(r"【[^】]+】", "", s).strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
    else:
        inner = s
    dims = [d.strip() for d in inner.split(",") if d.strip()]
    return dims


def map_dims_elms_to_dapper(dims: List[str]) -> str:
    """
    Map ELM/CLM Fortran dims from surfrdMod (grlnd, topounit, natpft, ...)
    into Dapper's surface-file convention (non-spatial dims first, then
    lsmlat,lsmlon).

    Examples:
        ["grlnd"]                        -> "lsmlat,lsmlon"
        ["grlnd","topounit"]            -> "topounit,lsmlat,lsmlon"
        ["grlnd","topounit","natpft"]   -> "topounit,natpft,lsmlat,lsmlon"
        ["grlnd","topounit","nglcec"]   -> "topounit,nglcec,lsmlat,lsmlon"
    """
    if not dims:
        return ""

    # Pure grid dimension
    if dims == ["grlnd"]:
        return "lsmlat,lsmlon"

    # Remove grlnd, keep the others in their original order
    others = [d for d in dims if d != "grlnd"]
    if not others:
        return "lsmlat,lsmlon"

    return ",".join(others + ["lsmlat", "lsmlon"])


def classify_required(req_text: str) -> str:
    """
    Map the full 'Required/optional' prose into a compact level:

        "Required" -> "required"
        "Optional" / "Optional (...)" -> "optional"
        "Required when X ...; optional otherwise" -> "conditional"
        "Optional (not required, ...)" -> "optional"
    """
    s = clean_text(req_text).lower()
    if not s:
        return ""

    if s == "required":
        return "required"
    if s == "optional":
        return "optional"

    # "not required" usually implies optional even if the word "required" appears
    if "not required" in s:
        return "optional"

    has_req = "required" in s
    has_opt = "optional" in s

    if has_req and has_opt:
        return "conditional"
    if has_req:
        return "conditional"
    if has_opt:
        return "optional"

    return ""


def parse_report(text: str) -> List[Dict[str, Any]]:
    """
    Parse the list-table in report.rst into rows:

        {
          "var_names": [...],
          "dims_elms": [...],
          "dims_dapper": "topounit,lsmlat,lsmlon",
          "description": "...",
          "requirement_text": "...",
          "required_level": "required|optional|conditional|''",
        }
    """
    start = text.find(LIST_TABLE_MARKER)
    if start == -1:
        raise RuntimeError(f"Could not find list-table marker: {LIST_TABLE_MARKER}")
    table = text[start:]

    # Each row is 4 bullets: var / dims / desc / required
    pattern = re.compile(
        r"^\s*\*\s*-\s*(?P<var>.+?)\n"
        r"\s*-\s*(?P<dims>.+?)\n"
        r"\s*-\s*(?P<desc>.+?)\n"
        r"\s*-\s*(?P<req>.+?)(?=\n\s*\*\s*-|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    rows: List[Dict[str, Any]] = []

    for m in pattern.finditer(table):
        var_cell = m.group("var").strip()
        dims_cell = m.group("dims").strip()
        desc_cell = m.group("desc").strip()
        req_cell = m.group("req").strip()

        # Skip header row
        if var_cell.startswith("**") and "Variable name" in var_cell:
            continue

        var_names = re.findall(r"``([^`]+)``", var_cell)
        dims_elms = parse_dims_cell(dims_cell)
        dims_dapper = map_dims_elms_to_dapper(dims_elms)
        desc = clean_text(desc_cell)
        req_text = clean_text(req_cell)
        level = classify_required(req_cell)

        rows.append(
            {
                "var_names": var_names,
                "dims_elms": dims_elms,
                "dims_dapper": dims_dapper,
                "description": desc,
                "requirement_text": req_text,
                "required_level": level,
            }
        )

    return rows


def render_registry_module(rows: List[Dict[str, Any]]) -> str:
    """
    Turn parsed rows into a Python module that defines REGISTRY_FROM_REPORT.
    Deduplicates variable names (first occurrence wins).
    """
    lines: List[str] = []
    lines.append('"""')
    lines.append("Auto-generated surface variable registry entries from report.rst.")
    lines.append("")
    lines.append("This module is produced by gen_registry_from_report.py;")
    lines.append("do not edit by hand.")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dapper.surf.schema import vdef, VarDef")
    lines.append("")
    lines.append("REGISTRY_FROM_REPORT = {")
    seen: set[str] = set()

    for row in rows:
        dims_str = row["dims_dapper"]
        desc = row["description"]
        req_level = row["required_level"]
        req_text = row["requirement_text"]

        for name in row["var_names"]:
            if name in seen:
                continue
            seen.add(name)

            lines.append(f'    "{name}": vdef(')
            if dims_str:
                lines.append(f'        "{dims_str}",')
            else:
                lines.append('        "",  # TODO: no dims parsed')
            lines.append(f"        doc={desc!r},")
            lines.append(f'        required_level="{req_level}",')
            # Preserve the full requirement prose as an attribute when it's
            # more than just "Required" or "Optional".
            if req_text and req_text not in ("Required", "Optional"):
                lines.append(f"        requirement={req_text!r},")
            lines.append("    ),")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate registry_from_report.py from report.rst")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="report.rst",
        help="Path to report.rst (ChatGPT-generated surface variable table).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="registry_from_report.py",
        help="Path to write the generated registry module.",
    )
    args = parser.parse_args()

    in_path = Path(r"C:\Users\318596\Downloads\report.rst")
    out_path = Path(r'X:\Research\NGEE Arctic\dapper\src\dapper\dev\blah.txt')

    text = in_path.read_text(encoding="utf-8")
    rows = parse_report(text)
    code = render_registry_module(rows)
    out_path.write_text(code, encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {out_path}")




