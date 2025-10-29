from __future__ import annotations
import re
import subprocess
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

def find_surface_vars_v3(
    repo_root: str | Path,
    search_root: str | Path = "components/elm",
    file_globs: Iterable[str] = ("**/*.F90","**/*.f90","**/*.F95","**/*.f95","**/*.F","**/*.f"),
    reader_allow: Iterable[str] = (
        # Common ELM/E3SM NetCDF/PIO wrappers (expand as needed)
        "ncd_io","ncd_inqdlen","ncd_inqvar","ncd_getvar",
        "ncd_pio_get_var","ncd_pio_inq_varid","ncd_pio_inq_dimid",
        "pio_inq_varid","pio_get_var","pio_inq_dimid",
    ),
    named_var_keys: Iterable[str] = ("fldname","varname","name"),
    # signals
    abort_keywords: Iterable[str] = ("endrun","shr_sys_abort","error stop","ELM_abort","shr_sys_stop"),
    optional_keywords: Iterable[str] = ("optional","allow_missing","missing_ok","use_default","default"),
    status_token_hints: Iterable[str] = ("ierr","rc","status","found","exists","ios","iostat"),
    context_lines: int = 14,
    include_snippet: bool = True,
) -> List[Dict[str, Optional[str]]]:
    """
    Static extractor for ELM/E3SM surface-file variables (v3 heuristics).
    Returns list of dicts:
      var_name, object_kind, op, file, line, fn, status_sym, required_inferred, reason, commit_hash, snippet
    """
    repo_root = Path(repo_root)
    search_root = repo_root / search_root
    allowed = {x.lower() for x in reader_allow}

    try:
        commit_hash = subprocess.run(
            ["git","-C",str(repo_root),"rev-parse","HEAD"],
            capture_output=True,text=True,check=True
        ).stdout.strip()
    except Exception:
        commit_hash = None

    # --------- Patterns (DOTALL for multiline arg lists) ----------
    call_pat = re.compile(r"(?i)\bcall\s+(?P<fn>\w+)\s*\((?P<args>.*?)\)", re.DOTALL)
    func_pat = re.compile(r"(?i)\b(?P<lhs>\w+)\s*=\s*(?P<fn>\w+)\s*\((?P<args>.*?)\)", re.DOTALL)

    named_pat = re.compile(
        rf"(?i)\b(?P<key>{'|'.join(map(re.escape, named_var_keys))})\s*=\s*(?P<q>['\"])(?P<val>.*?)(?P=q)",
        re.DOTALL
    )
    any_str_pat = re.compile(r"(['\"])(?P<val>.*?)(\1)", re.DOTALL)

    # IF variants
    if_then_line_pat  = re.compile(r"(?i)^\s*if\s*\((?P<cond>.*)\)\s*then\s*$")
    if_inline_pat      = re.compile(r"(?i)^\s*if\s*\((?P<cond>.*)\)\s*(?!then\b)(?P<body>.*\S.*)$")
    else_pat           = re.compile(r"(?i)^\s*else\s*$")
    endif_pat          = re.compile(r"(?i)^\s*end\s*if\s*$")

    abort_pat   = re.compile("|".join(re.escape(k) for k in abort_keywords), re.I)
    optional_pat= re.compile("|".join(re.escape(k) for k in optional_keywords), re.I)
    plausible_var = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    # explicit required=.<bool>.
    explicit_required_pat = re.compile(r"(?i)\brequired\s*=\s*\.(true|false)\.")

    def collect_files() -> List[Path]:
        files: List[Path] = []
        for g in file_globs:
            files.extend(search_root.glob(g))
        return sorted(set(files))

    def extract_var_names(fn: str, args: str) -> Tuple[List[str], str, str]:
        named_hits = [m.group("val").strip() for m in named_pat.finditer(args)]
        cand = [v for v in named_hits if plausible_var.match(v)]
        if not cand and "inq" in fn.lower():
            cand = [m.group("val").strip() for m in any_str_pat.finditer(args)]
            cand = [v for v in cand if plausible_var.match(v)]
        kind = "variable"
        if "inqdlen" in fn.lower() or ("inq" in fn.lower() and "dim" in fn.lower()):
            kind = "dimension"
        op = "inq" if "inq" in fn.lower() else "read"
        return cand, kind, op

    def find_nearest_if_block(lines, start_idx: int, status_sym: str) -> Optional[dict]:
        """
        Find the first IF construct *after* start_idx whose condition references status_sym.
        Supports both block IF (THEN/END IF) and inline IF (single line).
        Returns dict with keys: kind ('block'|'inline'), cond, then, else, span.
        """
        n = len(lines)
        max_look = min(n, start_idx + context_lines + 60)
        status_l = status_sym.lower()

        # 1) Scan for inline IF first (common in E3SM)
        for i in range(start_idx, max_look):
            m = if_inline_pat.match(lines[i])
            if m and status_l in m.group("cond").lower():
                return {"kind":"inline", "cond": m.group("cond"), "then":[m.group("body")], "else":[], "span":(i,i)}

        # 2) Scan for block IF
        for i in range(start_idx, max_look):
            m = if_then_line_pat.match(lines[i])
            if not m:
                continue
            cond = m.group("cond")
            if status_l not in cond.lower():
                continue
            then_body, else_body, cur = [], [], "then"
            for j in range(i+1, min(n, i + context_lines + 120)):
                line = lines[j]
                if endif_pat.match(line):
                    return {"kind":"block","cond":cond,"then":then_body,"else":else_body,"span":(i,j)}
                if else_pat.match(line):
                    cur = "else"; continue
                (then_body if cur=="then" else else_body).append(line)
        return None

    def cond_is_error_branch(cond: str, status_sym: str) -> Optional[bool]:
        c = cond.lower(); s = status_sym.lower()
        if s not in c: return None
        # error if: ierr /= 0, .ne. 0, > 0, or .not. okflag
        if re.search(rf"\b{s}\b\s*(?:/=?|\.ne\.)\s*0", c) or \
           re.search(rf"\b{s}\b\s*>\s*0", c) or \
           re.search(rf"\.not\.\s*\b{s}\b", c):
            return True
        # success if: == 0 or compares to *_NOERR
        if re.search(rf"\b{s}\b\s*(?:==|\.eq\.)\s*0", c) or \
           "pio_noerr" in c or "nf90_noerr" in c:
            return False
        return None

    def classify_required_from_if(ifblk: dict, error_is_then: Optional[bool]) -> Tuple[str,str]:
        if ifblk is None:
            return "unknown","no status-guard IF"
        if ifblk["kind"] == "inline":
            body_txt = " ".join(ifblk["then"])
            if abort_pat.search(body_txt): return "required","inline abort"
            if optional_pat.search(body_txt): return "optional","inline optional hint"
            return "unknown","inline no-abort"
        # block:
        err_body = ifblk["then"] if error_is_then is True else (
                   ifblk["else"] if error_is_then is False else (ifblk["then"] + ifblk["else"]))
        body_txt = "\n".join(err_body)
        if abort_pat.search(body_txt): return "required","abort in error branch"
        if optional_pat.search(body_txt): return "optional","optional hint in error branch"
        if body_txt.strip(): return "optional","handled error branch"
        return "unknown","empty error branch"

    def fallback_classify(context_txt: str, var_name: str) -> Tuple[str,str]:
        # Nearby abort that mentions var or looks like missing-var message
        if abort_pat.search(context_txt):
            if var_name and re.search(rf"{re.escape(var_name)}|missing|not\s+found|absent", context_txt, re.I):
                return "required","abort near call (var/missing mentioned)"
        if optional_pat.search(context_txt):
            return "optional","optional/default hint near call"
        return "unknown","no clear signal"

    results: List[Dict[str, Optional[str]]] = []

    # --------- Main scan ----------
    files = []
    for g in file_globs:
        files.extend(search_root.glob(g))
    files = sorted(set(files))

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()

        # Collect I/O events (call- and func-style)
        events = []
        for m in call_pat.finditer(text):
            fn = m.group("fn")
            if fn.lower() in allowed:
                events.append(("call", m.start(), {"fn": fn, "args": m.group("args"), "lhs": None}))
        for m in func_pat.finditer(text):
            fn = m.group("fn")
            if fn.lower() in allowed:
                events.append(("func", m.start(), {"fn": fn, "args": m.group("args"), "lhs": m.group("lhs")}))
        events.sort(key=lambda x: x[1])

        for kind_evt, start_pos, info in events:
            fn, args, lhs = info["fn"], info["args"], info["lhs"]
            line_no = text.count("\n", 0, start_pos) + 1

            cand_vars, obj_kind, op = extract_var_names(fn, args)
            if not cand_vars:
                continue

            # 1) explicit required=
            explicit = explicit_required_pat.search(args)
            explicit_required = None
            if explicit:
                explicit_required = explicit.group(1).lower() == "true"

            # status symbol
            status_sym = None
            if kind_evt == "func":
                status_sym = lhs
            else:
                mstat = re.search(r"(?i)\b(?:ierr|rc|status|ios|iostat)\s*=\s*(\w+)", args)
                if mstat:
                    status_sym = mstat.group(1)
                else:
                    for tok in status_token_hints:
                        if re.search(rf"\b{re.escape(tok)}\b", args, flags=re.I):
                            status_sym = tok; break

            i0 = max(0, line_no - 1 - context_lines)
            i1 = min(len(lines), line_no - 1 + context_lines + 1)
            ctx_lines = lines[i0:i1]
            ctx = "\n".join(ctx_lines)

            # 2) IF-driven classification
            required_inferred, reason = "unknown","unknown"
            if explicit_required is True:
                required_inferred, reason = "required","required=true (arg)"
            elif explicit_required is False:
                required_inferred, reason = "optional","required=false (arg)"
            elif status_sym:
                ifblk = find_nearest_if_block(lines, line_no-1, status_sym)
                pol = cond_is_error_branch(ifblk["cond"], status_sym) if ifblk else None
                required_inferred, reason = classify_required_from_if(ifblk, pol)

            # 3) Fallback: nearby cues
            if required_inferred == "unknown":
                required_inferred, reason = fallback_classify(ctx, cand_vars[0] if cand_vars else "")

            for vname in cand_vars:
                results.append({
                    "var_name": vname,
                    "object_kind": obj_kind,
                    "op": op,
                    "file": str(f.relative_to(repo_root)),
                    "line": line_no,
                    "fn": fn,
                    "status_sym": status_sym,
                    "required_inferred": required_inferred,
                    "reason": reason,
                    "commit_hash": commit_hash,
                    "snippet": ctx if include_snippet else None,
                })

    # de-dupe
    dedup, seen = [], set()
    for r in results:
        key = (r["var_name"], r["file"], r["line"], r["fn"])
        if key not in seen:
            seen.add(key); dedup.append(r)
    return dedup


rows = pd.DataFrame(find_surface_vars_v3(r"X:\Research\NGEE Arctic\E3SM\E3SM"))
rows.query("object_kind == 'variable' and required_inferred == 'required'").shape[0]
