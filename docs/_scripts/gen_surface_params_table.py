from __future__ import annotations

"""Generate the surface variable/parameter table used by the docs.

This writes ``docs/_generated/surface_variables_tables.rst`` from
``dapper.surf.surface_var_specs.SURFACE_VAR_SPECS``.
"""

from pathlib import Path
import sys

DOCS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DOCS_DIR.parent
SRC_DIR = REPO_DIR / "src"
# Allow generation without installing the package
sys.path.insert(0, str(SRC_DIR))


def _rst_inline_code_list(items: list[str]) -> str:
    if not items:
        return ""
    return ", ".join(f"``{c}``" for c in items)


def main() -> None:
    from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS

    out_dir = DOCS_DIR / "_generated"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_variables_tables.rst"

    specs: dict[str, dict] = dict(SURFACE_VAR_SPECS)

    def write_table(f, title: str, items: dict[str, dict]) -> None:
        if not items:
            return

        f.write(f"{title}\n")
        f.write(f"{'-' * len(title)}\n\n")

        # Wrap in a container so wide tables can scroll horizontally.
        f.write(".. container:: scroll-x\n\n")
        f.write("   .. list-table::\n")
        f.write("      :header-rows: 1\n\n")

        f.write("      * - **Variable**\n")
        f.write("        - **Dimensions**\n")
        f.write("        - **Units**\n")
        f.write("        - **Required level**\n")
        f.write("        - **Contexts**\n")
        f.write("        - **Description**\n\n")

        for name in sorted(items):
            spec = items[name] or {}
            dims = spec.get("dims", "") or ""
            units = spec.get("units", "") or ""
            req_level = spec.get("required_level", "") or ""
            ctxs = sorted(spec.get("contexts", []) or [])
            ctx_str = _rst_inline_code_list(ctxs)

            doc = (spec.get("doc", "") or "").replace("\n", " ").strip()
            req_attr = (spec.get("attrs", {}) or {}).get("requirement", "") or ""
            if req_attr:
                doc = f"{doc} (Requirement: {req_attr})" if doc else f"Requirement: {req_attr}"

            f.write(f"      * - ``{name}``\n")
            f.write(f"        - ``{dims}``\n")
            f.write(f"        - ``{units}``\n")
            f.write(f"        - ``{req_level}``\n")
            f.write(f"        - {ctx_str}\n")
            f.write(f"        - {doc}\n\n")

    all_contexts = sorted({c for spec in specs.values() for c in (spec.get("contexts", []) or [])})

    with out_path.open("w", encoding="utf-8") as f:
        write_table(f, "All surface variables", specs)
        f.write("\n")

        if all_contexts:
            f.write("Variables by context\n")
            f.write("--------------------\n\n")
            for ctx in all_contexts:
                subset = {
                    name: spec
                    for name, spec in specs.items()
                    if ctx in (spec.get("contexts", []) or [])
                }
                if subset:
                    write_table(f, f"Context: {ctx}", subset)
                    f.write("\n")


if __name__ == "__main__":
    main()
