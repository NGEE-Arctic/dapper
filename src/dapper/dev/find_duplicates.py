import ast, difflib, hashlib, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root-ish
PY = [p for p in ROOT.rglob("*.py") if "__pycache__" not in str(p)]

def norm_source(src: str) -> str:
    # strip comments/blank lines & collapse spaces
    lines = []
    for line in src.splitlines():
        if re.match(r'^\s*#', line): continue
        if line.strip() == "": continue
        lines.append(re.sub(r'\s+', ' ', line.strip()))
    return "\n".join(lines)

def func_blocks(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # grab original source lines
            try:
                # Python 3.8+: ast.get_source_segment
                src = ast.get_source_segment(text, node) or ""
            except Exception:
                src = ""
            out.append((node.name, norm_source(src), path))
    return out

# index by hash of normalized body
buckets = {}
funcs = []
for p in PY:
    for name, body, path in func_blocks(p):
        h = hashlib.md5(body.encode("utf-8")).hexdigest()
        buckets.setdefault(h, []).append((name, path))
        funcs.append((name, body, path))

print("=== EXACT DUPLICATES ===")
for h, group in buckets.items():
    if len(group) > 1:
        names = ", ".join([f"{n} @ {str(p)}" for n,p in group])
        print("•", names)

# near-duplicates by name using similarity
print("\n=== NEAR DUPLICATES (same name, >0.92 sim) ===")
by_name = {}
for name, body, path in funcs:
    by_name.setdefault(name, []).append((body, path))
for name, items in by_name.items():
    if len(items) < 2: continue
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            b1, p1 = items[i]; b2, p2 = items[j]
            s = difflib.SequenceMatcher(None, b1, b2).ratio()
            if s > 0.92 and str(p1) != str(p2):
                print(f"• {name}: {p1}  ~  {p2}  (sim={s:.2f})")
