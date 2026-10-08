"""Tables, figure environments and Supplementary Materials of the manuscript, and the map of its materials.

    python scripts/paper_assets.py                       # into paper/v2, the manuscript (from any directory)
    python scripts/paper_assets.py --out paper_assets    # the same files into another directory
    python scripts/paper_assets.py --manifest            # also docs/paper_materials.md

Reads the full-precision CSVs of runs/_report/tables/ (and runs/_tables/m4, m5, m8, m9 where a table is
taken from there) and the figures of runs/_report/figures and runs/_report/supplement/figures, and
writes, under the output directory (--out, default paper/v2; a relative path is taken from the repository
root; the directories are created when missing):

  tables/<name>.tex                          the ten generated tables of the main text, label tab:<name>
  supplement/tables/S<nn>_<name>.tex         Tables S1-S41 of the Supplementary Materials, label tab:S<name>
  figures/<name>.pdf, <name>_env.tex         the seven figures of the main text, label fig:<name>
  supplement/figures/<name>.pdf, S<nn>_<name>_env.tex   Figures S1-S22, label fig:S<name>
  supplementary.tex                          the Supplementary Materials (inputs the files above, by section;
                                             \\setcounter keeps the binding numbers where the order differs)
  supplement/items.txt                       the list "Table S1: title; ..." for \\supplementary of main.tex

--manifest [PATH] also writes the map of the materials (default docs/paper_materials.md): one row per Table
1-14, Figure 1-7, Table S1-S41 and Figure S1-S22 of the manuscript with its label, the generated files, the
files it is built from (every CSV and report its builder reads, the figure it copies) and the commands that
write them. The numbering of the main text is MAIN_FLOATS (checked against paper/v2/main.tex when the
manuscript is present); four tables of the main text are written by hand in the manuscript.

Until 8 October 2026 the generator was paper/v2/build_assets_v2.py, next to the manuscript; the headers of
the generated files name this script (GENERATED_BY) and the rest of every file is unchanged.

Tables S32-S37 and S39-S41 are generic: md_tab() turns a CSV of runs/_tables/<m>/ into a
table with the columns, heads, rounding and rows of the markdown report of the same stem (every cell checked
against that report).

Every number is formatted here from the CSVs, with the rounding of scripts/make_report.py: spacing
RMSE 2 decimals, shares 2 (main text) or as in the generated report (supplement), macro errors and
correlations 3, percentages 1, p-values 3 decimals or <0.001.

Nothing overflows by construction: every table is a tabularx (or, when it is longer than a page, an
xltabular = longtable + tabularx) of a fixed total width -- \\textwidth in the text column (393 pt),
\\fulllength inside adjustwidth (523 pt), \\textwidth on a landscape page (770 pt). The row labels and
the other text columns are X columns that wrap; the numbers are in natural-width columns whose widths
are estimated from the Palatino metrics of the class (Adobe AFM widths, 1/1000 em) at the chosen font
size. For every table the script searches the font size (\\footnotesize, then \\scriptsize), the column
separation and the set of columns whose intervals go under their estimate (\\cellstack) for the layout
of least height whose estimated natural width (+3 % margin) leaves every X column at least its longest
word; for a table of the Supplementary Materials it also chooses the context (text column, full width,
landscape page) and the kind (float or xltabular). No \\resizebox anywhere.

Deterministic: the same CSVs give the same files. Rerun after scripts/make_report.py; hand edits of the
generated files are overwritten (main.tex and the sections are not touched).
"""

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import math
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "paper" / "v2"  # the manuscript: the default output directory
REPORT = ROOT / "runs" / "_report"
SRC = REPORT / "tables"
M4 = ROOT / "runs" / "_tables" / "m4"
M5 = ROOT / "runs" / "_tables" / "m5"
M8 = ROOT / "runs" / "_tables" / "m8"
M9 = ROOT / "runs" / "_tables" / "m9"
MANIFEST = "docs/paper_materials.md"  # --manifest without a path
GENERATED_BY = "scripts/paper_assets.py"  # named in the headers of the generated files (module docstring)

# subdirectories of the output directory
OUT_MAIN = Path("tables")
OUT_SUPP = Path("supplement") / "tables"
OUT_FIG = Path("figures")
OUT_SUPP_FIG = Path("supplement") / "figures"

# ------------------------------------------------------------------------------------------------
# geometry of the MDPI class (Definitions/mdpi.cls of 23 June 2026), in pt
TARGET = {"text": 393.0, "wide": 523.0, "landscape": 770.0}
WIDTH_EXPR = {"text": r"\textwidth", "wide": r"\fulllength", "landscape": r"\textwidth"}
HEIGHT_LIMIT = {"text": 640.0, "wide": 640.0, "landscape": 420.0}  # text height about 700 and 455 pt
CAPTION_WIDTH = {"text": 393.0, "wide": 393.0, "landscape": 770.0 - 130.4}  # landscape: caption margin 4.6 cm
SIZES = {"footnotesize": 8.0, "scriptsize": 7.0}
LINE = {"footnotesize": 9.5 * 1.165, "scriptsize": 8.0 * 1.165}  # \baselineskip x \linespread{1.165}
CAPTION_LINE = 11.0 * 1.17  # \small captions, stretch 1.17
SAFETY = 1.03  # margin on the estimated natural widths
SEPS = (6.0, 5.0, 4.0, 3.0)  # \tabcolsep candidates, pt
CI_NOTE = r"Brackets: 95\% percentile bootstrap intervals (1000 resamples)."

# ------------------------------------------------------------------------------------------------
# Palatino Roman widths (Adobe AFM, = URW Palladio L), 1/1000 em
AFM = {" ": 250, "!": 278, '"': 371, "#": 500, "$": 500, "%": 840, "&": 778, "'": 278, "(": 333, ")": 333,
       "*": 389, "+": 606, ",": 250, "-": 333, ".": 250, "/": 606, ":": 250, ";": 250, "<": 606, "=": 606,
       ">": 606, "?": 444, "@": 747, "[": 333, "\\": 606, "]": 333, "^": 606, "_": 500, "`": 278, "{": 333,
       "|": 606, "}": 333, "~": 606, "\u2212": 606, "\u2192": 1000, "\u2265": 606, "\u2264": 606,
       "\u00b1": 606, "\u00d7": 606, "\u2013": 500, "\u2014": 1000, "\u2009": 167, "\u03c9": 620,
       "\u0394": 700, "\u2032": 300}
AFM.update({c: 500 for c in "0123456789"})
AFM.update(dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [778, 611, 709, 774, 611, 556, 763, 832, 337, 333, 726, 611,
                                                   946, 831, 786, 604, 786, 668, 525, 613, 778, 722, 1000, 667,
                                                   667, 667])))
AFM.update(dict(zip("abcdefghijklmnopqrstuvwxyz", [500, 553, 444, 611, 479, 333, 556, 582, 291, 234, 556, 291,
                                                   883, 582, 546, 601, 560, 395, 424, 326, 603, 565, 834, 516,
                                                   556, 500])))
UNDERSCORE = "\u2017"  # placeholder of a text underscore while math sub/superscripts are removed


def plain(s: str) -> str:
    """Rough rendering of a LaTeX cell as the characters it prints (for the width estimate)."""
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    for a, b in ((r"$-$", "\u2212"), (r"$+$", "+"), (r"$<$", "<"), (r"$>$", ">"), (r"\leq", "\u2264"),
                 (r"\geq", "\u2265"), (r"\to", "\u2192"), (r"\rightarrow", "\u2192"), (r"\pm", "\u00b1"),
                 (r"\times", "\u00d7"), (r"\%", "%"), (r"\_", UNDERSCORE), (r"\&", "&"), (r"\#", "#"),
                 (r"\,", "\u2009"), (r"\omega", "\u03c9"), (r"\Delta", "\u0394"), (r"\max", "max"),
                 (r"\min", "min"), (r"\mathrm", ""), (r"\allowbreak{}", ""), (r"\allowbreak", ""),
                 (r"\newline", " "), ("~", " "), (r"\prime", "\u2032")):
        s = s.replace(a, b)
    s = re.sub(r"\\(textit|textbf|emph|text|textsuperscript|textsubscript|mbox|textrm)\{", "{", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("^", "").replace("_", "")
    return s.replace(UNDERSCORE, "_")


@functools.lru_cache(maxsize=None)  # the width functions are pure; the layout search calls them very often
def text_w(s: str, size: float) -> float:
    return sum(AFM.get(ch, 600) for ch in plain(s)) / 1000.0 * size


def split_words(text: str) -> list[str]:
    return list(_split_words(text))


@functools.lru_cache(maxsize=None)
def _split_words(text: str) -> tuple[str, ...]:
    """Words of a wrapping cell: spaces split outside braces and math; \\allowbreak{} and \\newline too."""
    text = text.replace(r"\allowbreak{}", "\x00").replace(r"\newline", " ")
    words, cur, depth, math_mode, i = [], "", 0, False, 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            m = re.match(r"\\([a-zA-Z]+|.)", text[i:])
            cur += m.group(0)
            i += len(m.group(0))
            continue
        if ch == "$":
            math_mode = not math_mode
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "\x00" or (ch == " " and depth == 0 and not math_mode):
            if cur:
                words.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    if cur:
        words.append(cur)
    return tuple(words)


def wrap_lines(text: str, limit: float, size: float) -> list[str]:
    """Greedy line breaking between words at the width limit (explicit breaks given as '\\\\')."""
    if r"\\" in text:
        return [t.strip() for t in text.split(r"\\")]
    lines: list[str] = []
    for w in split_words(text):
        if lines and text_w(lines[-1] + " " + w, size) <= limit:
            lines[-1] += " " + w
        else:
            lines.append(w)
    return lines or [""]


@functools.lru_cache(maxsize=None)
def n_lines(text: str, width: float, size: float) -> int:
    """Lines of a wrapping (X) cell at the column width (forced breaks at \\newline counted)."""
    if not text:
        return 1
    total = 0
    for part in text.split(r"\newline"):
        lines, cur = 0, 0.0
        for w in split_words(part):
            ww = text_w(w, size)
            if cur and cur + text_w(" ", size) + ww <= width:
                cur += text_w(" ", size) + ww
            else:
                lines += 1
                cur = ww
        total += max(lines, 1)
    return total


@functools.lru_cache(maxsize=None)
def longest_word(text: str, size: float) -> float:
    return max((text_w(w, size) for w in split_words(text)), default=0.0)


# ------------------------------------------------------------------------------------------------
# CSV and number formatting (the rounding of scripts/make_report.py, cf_stability.eval.tables._cell)
READ: list[Path] = []  # the files read by the builder that runs: the sources of its table (the manifest)


def used(path: Path) -> Path:
    if path not in READ:
        READ.append(path)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with used(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x: str | None) -> float | None:
    if x is None:
        return None
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return None
    return float(x)


def minus(s: str) -> str:
    """Minus signs, not hyphens: '-0.5' -> '$-$0.5'."""
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)


def same(a: str, b: str) -> bool:
    """Two CSV numbers equal up to the float formatting of the writer."""
    x, y = fnum(a), fnum(b)
    return x == y if x is None or y is None else abs(x - y) <= 1e-12 * max(1.0, abs(x))


def f_num(x: float | None, d: int = 2) -> str:
    return "" if x is None else minus(f"{x:.{d}f}")


def f_pct(x: float | None, d: int = 1, unit: bool = True) -> str:
    if x is None:
        return ""
    return minus(f"{100.0 * x:+.{d}f}") + (r"\%" if unit else "")


def f_p(x: float | None) -> str:
    if x is None:
        return ""
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def f_int(x: float | None) -> str:
    return "" if x is None else str(int(x))


def f_w(x: float | None) -> str:
    return "" if x is None else f"{x:g}"


def f_coll(x: float | None) -> str:
    """Collisions per 1000 vehicle-km in the compact tables of the main text: 0, 2 decimals below 1, else integer."""
    if x is None:
        return ""
    if x == 0:
        return "0"
    return f"{x:.2f}" if x < 1 else f"{x:.0f}"


@dataclass
class V:
    """A number with its interval: on one line, or the interval under the estimate (\\cellstack)."""
    est: str
    ci: str = ""


def num_ci(row: dict, key: str, d: int = 2, kind: str = "num", unit: bool = True) -> V | str:
    fmt = {"num": lambda v: f_num(v, d), "pct": lambda v: f_pct(v, d, unit)}[kind]
    est = fnum(row.get(key))
    if est is None:
        return ""
    lo, hi = fnum(row.get(f"{key}_low")), fnum(row.get(f"{key}_high"))
    if lo is None or hi is None:
        return V(fmt(est))
    return V(fmt(est), f"[{fmt(lo)}, {fmt(hi)}]")


ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
          "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
DCODE = re.compile(r"\s*\((?:D\d+(?:,\s*D\d+)*(?: part a)?)\)")


def tex(s: str) -> str:
    """Plain text of a CSV cell as LaTeX: special characters escaped, relations in math, no decision codes."""
    s = DCODE.sub("", s.strip())
    s = s.replace(" %", "%")
    out = "".join(ESCAPE.get(ch, ch) for ch in s)
    out = out.replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace("+-", r"$\pm$")
    out = re.sub(r"(?<![$\\])>", "$>$", out)
    out = re.sub(r"(?<![$\\])<", "$<$", out)
    return minus(out)


def ident(s: str, breakable: bool = False) -> str:
    """A law or run identifier; in a wrapping column it may break after every underscore."""
    return s.replace("_", r"\_\allowbreak{}" if breakable else r"\_")


ARCH = {"mlp": "MLP", "pidl": "PIDL", "gru": "GRU", "lstm": "LSTM", "perl": "PERL", "residual_idm": "ResidualIDM",
        "idm": "IDM", "knn": "$k$-NN", "ovm": "OVM", "newell": "Newell", "persistence": "persistence",
        "pooled": "pooled"}
DATA = {"follownet_highd": "highD", "ngsim_i80": "NGSIM I-80", "ngsim_us101": "NGSIM US-101", "waymo": "Waymo",
        "openacc_acc": "OpenACC, ACC", "openacc_human": "OpenACC, human"}
TARGET_DATA = {"ngsim_i80": "I-80", "ngsim_us101": "US-101", "waymo": "Waymo"}
PENALTY = {"jacobian": "Jacobian", "gain": "rollout"}
METHOD = {"spearman": "Spearman", "pearson": "Pearson"}
SCENARIOS = {"all": "all", "i80_p1, i80_p2": "periods 1, 2"}
CORRIDOR_TITLE = {"I-80": "Corridor I-80", "US-101": "Corridor US-101"}
DESIGN_LAWS = ["idm_global", "idm_heterogeneous", "knn", "mlp", "pidl", "gru", "lstm", "perl", "residual_idm",
               "residual_idm_certified", "mlp_penalty", "gru_penalty", "lstm_penalty"]


# ------------------------------------------------------------------------------------------------
# table model
@dataclass
class Col:
    head: str  # LaTeX; '\\' marks explicit head line breaks
    cells: list  # str | V | list[str] (explicit lines)
    align: str = "r"  # l, c, r: natural width; L, C, R: wrapping X column
    weight: float = 1.0  # relative width of an X column
    stack: str = "auto"  # auto | never | always: interval under the estimate

    @property
    def x(self) -> bool:
        return self.align in "LCR"


@dataclass
class Tab:
    name: str  # file stem
    label: str
    caption: str
    cols: list[Col]
    groups: dict[int, str] = field(default_factory=dict)  # row index -> title of the group row before it
    top: list[tuple[str, int, int]] = field(default_factory=list)  # spanning heads: (text, first, last column)
    contexts: tuple[str, ...] = ("text",)
    sizes: tuple[str, ...] = ("footnotesize", "scriptsize")
    long_ok: bool = False  # may become an xltabular (supplement)
    panels: list["Tab"] = field(default_factory=list)
    title: str = ""  # panel title
    note: str = ""  # header comment: source
    footer: str = ""  # note under the table (MDPI table footer)
    layout: dict = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)  # the files its builder read (the manifest)

    @property
    def nrows(self) -> int:
        return len(self.cols[0].cells) if self.cols else 0


def cell_lines(v, stacked: bool) -> list[str]:
    if isinstance(v, V):
        if v.ci and stacked:
            return [v.est, v.ci]
        return [f"{v.est} {v.ci}".strip()]
    if isinstance(v, list):
        return v
    return [v]


def x_text(v) -> str:
    if isinstance(v, V):
        return f"{v.est} {v.ci}".strip()
    if isinstance(v, list):
        return r" \newline ".join(v)
    return v


def measure(t: Tab, ctx: str, size: str, sep: float, stacked: frozenset) -> dict | None:
    """Widths, heads and height of one layout; None when it does not fit."""
    fs = SIZES[size]
    W = TARGET[ctx]
    n = len(t.cols)
    nat: dict[int, float] = {}
    heads: dict[int, list[str]] = {}
    xmin: dict[int, float] = {}
    for j, c in enumerate(t.cols):
        if c.x:
            xmin[j] = max([longest_word(x_text(v), fs) for v in c.cells] + [longest_word(c.head.replace(r"\\", " "), fs)])
            continue
        dw = max((text_w(line, fs) for v in c.cells for line in cell_lines(v, j in stacked)), default=0.0)
        if r"\\" in c.head:
            hl = [h.strip() for h in c.head.split(r"\\")]
        else:
            hl = wrap_lines(c.head, max(dw, longest_word(c.head, fs), 30.0), fs) if c.head else [""]
        heads[j] = hl
        nat[j] = max(dw, max(text_w(h, fs) for h in hl))
    seps = 2.0 * sep * (n - 1)
    fixed = sum(nat.values()) * SAFETY
    avail = W - fixed - seps
    xs = [j for j in range(n) if t.cols[j].x]
    if not xs:
        raise SystemExit(f"{t.name}: no X column")
    wsum = sum(t.cols[j].weight for j in xs)
    xw = {j: avail * t.cols[j].weight / wsum for j in xs}
    if avail <= 0 or any(xw[j] < xmin[j] * SAFETY + 2.0 for j in xs):
        return None
    for j in xs:
        c = t.cols[j]
        heads[j] = [h.strip() for h in c.head.split(r"\\")] if r"\\" in c.head else wrap_lines(c.head, xw[j], fs)
        if max(text_w(h, fs) for h in heads[j]) > xw[j]:
            return None
    width = {j: nat.get(j, xw.get(j)) for j in range(n)}
    # spanning heads must fit their columns
    tops = []
    for text, a, b in t.top:
        span = sum(width[j] for j in range(a, b + 1)) + 2.0 * sep * (b - a)
        lines = wrap_lines(text, span, fs)
        if max(text_w(x, fs) for x in lines) > span:
            return None
        tops.append((lines, a, b))
    for g in t.groups.values():
        if text_w(r"\textit{" + g + "}", fs) > W:
            return None
    # height
    line = LINE[size]
    head_n = max(len(h) for h in heads.values()) + (max(len(x[0]) for x in tops) if tops else 0)
    body = 0
    for i in range(t.nrows):
        k = 1
        for j, c in enumerate(t.cols):
            v = c.cells[i]
            if c.x:
                k = max(k, n_lines(x_text(v), xw[j], fs))
            else:
                k = max(k, len(cell_lines(v, j in stacked)))
        body += k
    titled = sum(1 for g in t.groups.values() if g)
    rules = 14.0 + 4.0 * len(t.groups) + (6.0 if tops else 0.0)
    height = (head_n + body + titled + (1 if t.title else 0)) * line + rules
    total = fixed + seps + sum(xmin[j] * SAFETY for j in xs)
    return {"ctx": ctx, "size": size, "sep": sep, "stacked": stacked, "heads": heads, "tops": tops, "xw": xw,
            "nat": nat, "total": total, "height": height, "slack": W - total}


def caption_height(caption: str, ctx: str, supplement: bool) -> float:
    w = text_w(("Table S00. " if supplement else "Table 00. ") + caption, 9.0)
    return (math.ceil(w / (0.96 * CAPTION_WIDTH[ctx])) + 0.0) * CAPTION_LINE + 14.0


EXHAUSTIVE_STACK = 9  # up to this many interval columns every subset is tried (all tables S1-S31)


def search(t: Tab, ctx: str, size: str) -> dict | None:
    """Least-height layout of t at one context and font size (then the widest separation, then the
    fewest stacked columns). With more than EXHAUSTIVE_STACK interval columns the subsets tried are the
    prefixes of the columns ordered by the width that stacking saves (for a given number of stacked columns
    that prefix gives the narrowest table)."""
    stackable = [j for j, c in enumerate(t.cols)
                 if not c.x and c.stack == "auto" and any(isinstance(v, V) and v.ci for v in c.cells)]
    forced = frozenset(j for j, c in enumerate(t.cols) if not c.x and c.stack == "always")
    if len(stackable) <= EXHAUSTIVE_STACK:
        subsets = [s for k in range(len(stackable) + 1) for s in itertools.combinations(stackable, k)]
    else:
        fs = SIZES[size]

        def saving(j: int) -> float:
            cells = t.cols[j].cells
            return (max((text_w(x, fs) for v in cells for x in cell_lines(v, False)), default=0.0)
                    - max((text_w(x, fs) for v in cells for x in cell_lines(v, True)), default=0.0))
        ranked = sorted(stackable, key=lambda j: (-saving(j), j))
        subsets = [tuple(ranked[:k]) for k in range(len(ranked) + 1)]
    best = None
    for subset in subsets:
        stacked = forced | frozenset(subset)
        for sep in SEPS:
            m = measure(t, ctx, size, sep, stacked)
            if m is None:
                continue
            key = (round(m["height"], 1), -sep, len(stacked))
            if best is None or key < best[0]:
                best = (key, m)
    return best[1] if best else None


def choose(t: Tab, supplement: bool) -> None:
    """Context, font size, kind and layout of t (and of its panels)."""
    order = []
    for ctx in t.contexts:
        for size in t.sizes:
            order.append((ctx, size))
    # a larger font is preferred over a narrower context in the supplement: text/fs, wide/fs, text/ss, ...
    if supplement:
        rank = {"text": 0, "wide": 1, "landscape": 2}
        order.sort(key=lambda cs: (rank[cs[0]] if cs[0] == "landscape" else 0,
                                   list(SIZES).index(cs[1]), rank[cs[0]]))
    tried = []
    for ctx, size in order:
        parts = t.panels or [t]
        ms = [search(p, ctx, size) for p in parts]
        if any(m is None for m in ms):
            tried.append(f"{ctx}/{size}: no fit")
            continue
        cap = caption_height(t.caption, ctx, supplement)
        height = sum(m["height"] for m in ms) + cap + 10.0 * (len(parts) - 1)
        kind = "float"
        if height > HEIGHT_LIMIT[ctx]:
            if not (t.long_ok and not t.panels and ctx in ("text", "landscape")):
                tried.append(f"{ctx}/{size}: height {height:.0f} pt")
                continue
            kind = "long"
        for p, m in zip(parts, ms):
            p.layout = dict(m)
        summary = {"ctx": ctx, "size": size, "kind": kind, "height": height,
                   "total": max(m["total"] for m in ms), "slack": min(m["slack"] for m in ms)}
        t.layout = {**(t.layout if not t.panels else {}), **summary}
        return
    raise SystemExit(f"{t.name}: no layout fits: {'; '.join(tried)}")


# ------------------------------------------------------------------------------------------------
# emission
def render(v, stacked: bool, align: str) -> str:
    if isinstance(v, V):
        if v.ci and stacked:
            return rf"\cellstack[{align}]{{{v.est}\\{{}}{v.ci}}}"
        return f"{v.est} {v.ci}".strip()
    if isinstance(v, list):
        if len(v) == 1:
            return v[0]
        return rf"\cellstack[{align}]{{" + r"\\".join(v) + "}"
    return v


def head_cell(lines: list[str], align: str) -> str:
    if len(lines) == 1:
        return lines[0]
    return rf"\headstack[{align}]{{" + r"\\".join(lines) + "}"


def colspec(t: Tab) -> str:
    xs = [c for c in t.cols if c.x]
    weights = [c.weight for c in xs]
    equal = len(set(weights)) <= 1
    norm = len(xs) / sum(weights)
    out = []
    for c in t.cols:
        if c.x and not equal:
            out.append(rf">{{\hsize={c.weight * norm:.3f}\hsize}}{c.align}")
        else:
            out.append(c.align)
    return "@{}" + "".join(out) + "@{}"


def head_block(t: Tab) -> list[str]:
    lay = t.layout
    n = len(t.cols)
    out = []
    if lay["tops"]:
        cells, rules, j = [], [], 0
        for lines, a, b in lay["tops"]:
            while j < a:
                cells.append("")
                j += 1
            left = "@{}" if a == 0 else ""
            right = "@{}" if b == n - 1 else ""
            cells.append(rf"\multicolumn{{{b - a + 1}}}{{{left}c{right}}}{{{head_cell(lines, 'c')}}}")
            rules.append(rf"\cmidrule(lr){{{a + 1}-{b + 1}}}")
            j = b + 1
        while j < n:
            cells.append("")
            j += 1
        out.append(" & ".join(cells) + r" \\")
        out.append(" ".join(rules))
    heads = []
    for j, c in enumerate(t.cols):
        al = {"L": "l", "C": "c", "R": "r"}.get(c.align, c.align)
        heads.append(head_cell(lay["heads"][j], al))
    out.append(" & ".join(heads) + r" \\")
    return out


def body_lines(t: Tab, longtable: bool) -> list[str]:
    lay = t.layout
    n = len(t.cols)
    out = []
    for i in range(t.nrows):
        if i in t.groups:
            if i > 0:
                out.append(r"\addlinespace")
            if t.groups[i]:  # an empty title gives the space only
                out.append(rf"\multicolumn{{{n}}}{{@{{}}l}}{{\textit{{{t.groups[i]}}}}} \\" + ("*" if longtable else ""))
        cells = []
        for j, c in enumerate(t.cols):
            al = {"L": "l", "C": "c", "R": "r"}.get(c.align, c.align)
            cells.append(render(c.cells[i], j in lay["stacked"], al))
        out.append(" & ".join(cells) + r" \\")
    return out


def layout_comment(t: Tab) -> str:
    lay = t.layout
    parts = t.panels or [t]
    stacked = "; ".join(",".join(p.cols[j].head.replace(r"\\", " ") for j in sorted(p.layout["stacked"])) or "none"
                        for p in parts)
    seps = ",".join("%gpt" % p.layout["sep"] for p in parts)
    return (f"% v2-layout: context={lay['ctx']} target={TARGET[lay['ctx']]:.0f}pt kind={lay['kind']} "
            f"size={lay['size']} sep={seps} estimated-width={lay['total']:.0f}pt "
            f"estimated-height={lay['height']:.0f}pt stacked: {stacked}")


def emit(t: Tab, supplement: bool) -> str:
    lay = t.layout
    ctx, size, kind = lay["ctx"], lay["size"], lay["kind"]
    lines = [f"% {t.name}: {t.note}",
             f"% Written by {GENERATED_BY}; do not edit by hand (rerun the script).",
             layout_comment(t)]
    width = WIDTH_EXPR[ctx]
    if kind == "long":
        lay_t = t.layout
        lines += [r"\begingroup", f"\\{size}", rf"\setlength{{\tabcolsep}}{{{lay_t['sep']:g}pt}}",
                  r"\renewcommand{\tabularxcolumn}[1]{p{#1}}",
                  rf"\begin{{xltabular}}{{{width}}}{{{colspec(t)}}}",
                  rf"\caption{{{t.caption}}}\label{{{t.label}}} \\"]
        head = [r"\toprule"] + head_block(t) + [r"\midrule"]
        lines += head + [r"\endfirsthead", r"\caption[]{\textit{Cont.}} \\"] + head + [
            r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
        lines += body_lines(t, longtable=True)
        lines += [r"\end{xltabular}", r"\endgroup"]
        return "\n".join(lines) + "\n"
    # the tables of the main text float (tbp): placed [H] the long ones (laws, E1, E5, E4) left half pages empty
    # before them or after them in the compiled PDF of 8 October 2026; the supplement keeps [H], where floating
    # tables left the short section introductions alone on their pages
    placement = "H" if (t.name[:1] == "S" and t.name[1:2].isdigit()) else "tbp"
    lines += [rf"\begin{{table}}[{placement}]", rf"\caption{{{t.caption}}}", rf"\label{{{t.label}}}"]
    if ctx == "wide":
        lines.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    lines += [f"\\{size}", r"\renewcommand{\tabularxcolumn}[1]{p{#1}}"]
    for k, p in enumerate(t.panels or [t]):
        if k:
            lines.append(r"\par\medskip")
        lines.append(rf"\setlength{{\tabcolsep}}{{{p.layout['sep']:g}pt}}")
        lines.append(rf"\begin{{tabularx}}{{{width}}}{{{colspec(p)}}}")
        if p.title:
            lines.append(rf"\multicolumn{{{len(p.cols)}}}{{@{{}}l}}{{{p.title}}} \\")
        lines += [r"\toprule"] + head_block(p) + [r"\midrule"] + body_lines(p, longtable=False)
        lines += [r"\bottomrule", r"\end{tabularx}"]
    if ctx == "wide":
        lines.append(r"\end{adjustwidth}")
    if t.footer:
        lines.append(r"\noindent{\footnotesize " + t.footer + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def groups_of(values: list[str], titles: dict[str, str] | None = None) -> dict[int, str]:
    out, last = {}, None
    for i, v in enumerate(values):
        if v != last:
            out[i] = (titles or {}).get(v, v)
            last = v
    return out


def p_statement(pairs: list[tuple[str, float]], what: str) -> str:
    """'p < 0.001 for every X except A (p = 0.375)' from (name, p) pairs."""
    big = [(n, p) for n, p in pairs if p is not None and p >= 0.001]
    if not big:
        return f"$p < 0.001$ for every {what}"
    rest = ", ".join(f"{n}, $p = {f_p(p)}$" for n, p in big)
    return f"$p < 0.001$ for every {what} except {rest}"


# ================================================================================================
# tables of the main text (text column, 393 pt)
def arrow(a: float | None, b: float | None, d: int) -> str:
    return f"{f_num(a, d)} $\\to$ {f_num(b, d)}"


def opt_num(r: dict, key: str, d: int) -> str:
    """A number of the row, or --- when the row has none (the pole-inclusive shares of the memoryless models)."""
    x = fnum(r[key])
    return "---" if x is None else f_num(x, d)


def main_e1() -> Tab:
    rows = read_csv(SRC / "e1.csv")
    order = ["mlp", "pidl", "gru", "lstm", "perl", "residual_idm", "persistence", "newell", "ovm", "idm", "knn"]
    assert [r["model"] for r in rows] == order, [r["model"] for r in rows]
    runs = {r["model"]: int(fnum(r["runs"])) for r in rows}
    assert {runs[m] for m in order[:6]} == {25} and {runs[m] for m in order[6:]} == {5}, runs
    assert [r["model"] for r in rows if fnum(r["rmse_vs_ref"]) is None] == ["idm"]
    cols = [
        Col("model", [ARCH[r["model"]] for r in rows], "L"),
        Col(r"spacing\\RMSE (m)", [num_ci(r, "rmse_s", 2) for r in rows]),
        Col(r"RMSE vs\\IDM (\%)", [num_ci(r, "rmse_vs_ref", 1, "pct", unit=False) or "---" for r in rows]),
        Col(r"unstable\\among\\equilibria", [num_ci(r, "unstable_eq", 2) for r in rows]),
        Col(r"poles\\incl.", [opt_num(r, "unstable_eq_full", 2) for r in rows]),
        Col(r"not stable\\in band", [f_num(fnum(r["not_stable"]), 2) for r in rows]),
        Col(r"poles\\incl.", [opt_num(r, "not_stable_full", 2) for r in rows]),
        Col("outside", [f_num(fnum(r["outside"]), 2) for r in rows]),
        Col("none", [f_num(fnum(r["none"]), 2) for r in rows]),
    ]
    for r in rows:  # the informational verdict with local instability included never differs from the verdict
        assert r["h1_1_poles"] in ("", r["h1_1"]), (r["model"], r["h1_1"], r["h1_1_poles"])
    p = p_statement([(ARCH[r["model"]], fnum(r["rmse_vs_ref_p"])) for r in rows if r["model"] != "idm"], "model")
    caption = (
        r"Audit of the unconstrained models (E1) on the highD events of the FollowNet benchmark: 25 runs per "
        r"learned model (5 folds $\times$ 5 seeds) and 5 per baseline (5 folds). Spacing RMSE: closed-loop "
        r"root-mean-square error of the spacing on the test events (m; unit: driver, i.e., event on highD, which carries no "
        r"driver identifier); RMSE vs IDM: relative difference to the intelligent driver model, paired over those "
        r"units (Wilcoxon signed-rank " + p + "). "
        r"Unstable among equilibria: share of the string-unstable equilibria among the equilibria found; not "
        r"stable in band, outside, none: shares of the grid speeds in the speed support whose equilibrium is not "
        r"stable inside the spacing band of the data, lies outside the band, or does not exist (unit: run); poles "
        r"incl.: the same two shares with a speed counted as not stable when a pole of the full-history loop lies "
        r"outside the unit circle (recurrent models, Section~\ref{sec:audit}; in E1 the pole check rejects no "
        r"additional equilibrium that passes the numerical rule, so the reconciled shares equal the pre-specified "
        r"ones and the rule of H1.1 on them gives the same verdicts). "
        r"Baselines: persistence, Newell's model, the optimal velocity model (OVM), the intelligent driver model "
        r"(IDM) and $k$-nearest neighbours ($k$-NN). " + CI_NOTE)
    return Tab("e1", "tab:e1", caption, cols, groups={0: "Learned models (25 runs each)", 6: "Baselines (5 runs each)"},
               note="from runs/_report/tables/e1.csv")


def main_e2() -> Tab:
    rows = read_csv(SRC / "e2.csv")
    order = ["mlp", "pidl", "gru", "lstm", "perl", "residual_idm"]
    assert [r["architecture"] for r in rows] == order
    assert {(fnum(r["runs_e1"]), fnum(r["runs_e2"])) for r in rows} == {(25, 25)}
    cols = [
        Col("architecture", [ARCH[r["architecture"]] for r in rows], "L"),
        Col("penalty", [PENALTY[r["kind"]] for r in rows], "l"),
        Col("weight", [f_w(fnum(r["weight"])) for r in rows]),
        Col(r"RMSE\\change (\%)", [num_ci(r, "rmse_change", 1, "pct", unit=False) for r in rows]),
        Col(r"not stable\\in band,\\E1 $\to$ E2",
            [V(arrow(fnum(r["not_stable_e1"]), fnum(r["not_stable_e2"]), 2),
               f"[{f_num(fnum(r['not_stable_e2_low']))}, {f_num(fnum(r['not_stable_e2_high']))}]") for r in rows]),
        Col(r"unstable\\among\\equilibria,\\E1 $\to$ E2",
            [arrow(fnum(r["unstable_eq_e1"]), fnum(r["unstable_eq_e2"]), 2) for r in rows]),
        Col(r"poles incl.,\\E2: not stable /\\unstable eq.",
            ["---" if fnum(r["not_stable_full_e2"]) is None else
             f"{f_num(fnum(r['not_stable_full_e2']), 2)} / {f_num(fnum(r['unstable_eq_full_e2']), 2)}" for r in rows]),
        Col(r"hysteresis\\area (m$^2$/s),\\E1 $\to$ E2",
            [arrow(fnum(r["hysteresis_e1"]), fnum(r["hysteresis_e2"]), 1) for r in rows]),
        Col("H1.2", [r["h1_2"] for r in rows], "l"),
    ]
    for r in rows:  # the informational verdict with local instability included never differs from the verdict
        assert r["h1_2_poles"] in ("", r["h1_2"]), (r["architecture"], r["h1_2"], r["h1_2_poles"])
    p = p_statement([(ARCH[r["architecture"]], fnum(r["rmse_change_p_holm"])) for r in rows], "architecture")
    caption = (
        r"Experiment E2: the chosen penalty weight against E1 on highD (25 runs per architecture and experiment: "
        r"5 folds $\times$ 5 seeds); Jacobian penalty for the memoryless models, rollout gain penalty (rollout) for "
        r"the recurrent ones. RMSE change: relative change of the spacing RMSE, paired over the drivers (events on highD) (Wilcoxon "
        r"signed-rank test with Holm's correction over the architectures, " + p + "); not stable in band and "
        r"unstable among equilibria as in Table~\ref{tab:e1}, E1 $\to$ E2 with the interval of E2 (unit: run); "
        r"poles incl.: the two E2 shares with a speed counted as not stable when a pole of the full-history loop "
        r"lies outside the unit circle (recurrent models, Section~\ref{sec:audit}; the rule of H1.2 on them gives "
        r"the same verdicts); "
        r"hysteresis area of the first follower behind a braking pulse (m$^2$/s), mean over the pairs of runs in "
        r"which both have a value; H1.2: verdict of the pre-specified rule on the share not stable in band and the "
        r"RMSE change (Table~\ref{tab:verdicts}). " + CI_NOTE)
    return Tab("e2", "tab:e2", caption, cols, note="from runs/_report/tables/e2.csv")


def main_e2_arms() -> Tab:
    ex = read_csv(SRC / "e2_existence.csv")
    lf = [r for r in read_csv(SRC / "e2_lowfreq.csv") if r["chosen"] == "True"]
    mono = read_csv(M8 / "e2_monotone.csv")
    assert [r["architecture"] for r in ex] == ["gru", "lstm", "perl"]
    assert [r["architecture"] for r in lf] == ["gru", "lstm", "perl"]
    assert {fnum(r["runs"]) for r in ex} == {5} and {fnum(r["runs"]) for r in lf} == {5}
    arch, weight, rmse, notst, uneq, note = [], [], [], [], [], []
    for r in ex:  # the existence term puts an equilibrium in the band at every speed
        assert fnum(r["outside"]) == 0.0 and fnum(r["none"]) == 0.0, r["architecture"]
        arch.append(ARCH[r["architecture"]])
        weight.append("---")
        rmse.append(num_ci(r, "rmse_vs_e1", 1, "pct", unit=False))
        notst.append(num_ci(r, "not_stable", 2))
        uneq.append(f_num(fnum(r["unstable_eq"]), 2))
        note.append(f"max gain {f_num(fnum(r['max_gain_median']), 2)}")
    for r in lf:
        arch.append(ARCH[r["architecture"]])
        weight.append(f_w(fnum(r["weight"])))
        rmse.append(num_ci(r, "rmse_change", 1, "pct", unit=False))
        notst.append(num_ci(r, "not_stable", 2))
        uneq.append(f_num(fnum(r["unstable_eq"]), 2))
        note.append(f"H1.2 {r['h1_2']}")
    for a in ("mlp", "pidl", "residual_idm"):
        mine = [r for r in mono if r["architecture"] == a]
        m = [r for r in mine if r["arm"].startswith("monotone")]
        e1 = [r for r in mine if r["arm"] == "E1"]
        e2 = [r for r in mine if r["arm"].startswith("E2")]
        assert len(m) == len(e1) == len(e2) == 1 and {fnum(x["runs"]) for x in mine} == {1}, a
        m, e1, e2 = m[0], e1[0], e2[0]
        w = re.search(r"_w([\d.]+)$", m["experiment"]).group(1)
        w2 = re.search(r"weight ([\d.]+)\)$", e2["arm"]).group(1)
        arch.append(ARCH[a])
        weight.append(f_w(float(w)))
        rmse.append(num_ci(m, "rmse_change", 1, "pct", unit=False))
        notst.append(f_num(fnum(m["not_stable"]), 2))  # one run: no interval
        uneq.append(f_num(fnum(m["unstable_eq"]), 2))
        note.append(f"E1 {f_num(fnum(e1['not_stable']), 2)}; Jacobian ($w = {f_w(float(w2))}$) "
                    f"{f_num(fnum(e2['not_stable']), 2)} at {f_pct(fnum(e2['rmse_change']), 1)}")
    cols = [
        Col("architecture", arch, "L"),
        Col("weight", weight),
        Col(r"RMSE change\\vs E1 (\%)", rmse),
        Col(r"not stable\\in band", notst),
        Col(r"unstable\\among\\equilibria", uneq),
        Col("note", note, "L", weight=1.6),
    ]
    groups = {0: "Existence term alone (5 folds, seed 0)",
              3: "Combined penalty at the chosen Jacobian weight (5 folds, seed 0)",
              6: "Monotonicity terms alone (fold 0, seed 0, one run each)"}
    caption = (
        r"Arms of E2 on highD that isolate parts of the penalties: the existence term alone, the combined penalty of "
        r"the low-frequency arm and the monotonicity terms alone. Existence "
        r"term alone: an equilibrium inside the spacing band at every speed and no stability term (every speed then "
        r"has its equilibrium in the band; max gain: largest numerical gain, median over the runs). Combined "
        r"penalty: the rollout gain penalty, the Jacobian penalty of the memoryless view and the guard against stiff "
        r"equilibria at the chosen Jacobian weight (weight); note: verdict of the rule of H1.2. Monotonicity terms "
        r"alone (weight 1): "
        r"$\mathrm{relu}(-f_s) + \mathrm{relu}(f_{\Delta v}) + \mathrm{relu}(f_v)$ at the anchored equilibria, "
        r"without a string-stability term; note: not stable in band of E1 and of the Jacobian penalty of E2 on the "
        r"same fold, with the RMSE change of the latter. RMSE change: relative change of the spacing RMSE against E1 "
        r"of the same folds and seed, paired over the drivers (events on highD); not stable in band and unstable among equilibria as in "
        r"Table~\ref{tab:e1} (unit: run). Full tables: S2--S4. " + CI_NOTE)
    return Tab("e2_arms", "tab:e2_arms", caption, cols, groups=groups,
               note="from runs/_report/tables/e2_existence.csv, e2_lowfreq.csv (chosen weight) and "
                    "runs/_tables/m8/e2_monotone.csv (monotone arm)")


H13 = {"confirmed": "conf.", "refuted": "ref.", "open": "open", "": ""}


def main_e5() -> Tab:
    rows = read_csv(SRC / "e5.csv")
    label = {"openacc_acc": "ACC", "openacc_human": "human"}
    assert [r["view"] for r in rows] == ["openacc_acc"] * 8 + ["openacc_human"] * 8
    expect = [("idm", "none"), ("mlp", "none"), ("gru", "none"), ("lstm", "none"), ("mlp", "chosen"),
              ("gru", "chosen"), ("lstm", "chosen"), ("pooled", "chosen")] * 2
    assert [(r["model"], r["penalty"]) for r in rows] == expect
    assert {fnum(r["runs"]) for r in rows if r["model"] != "pooled"} == {5}
    view = [label[r["view"]] if i in (0, 8) else "" for i, r in enumerate(rows)]
    cols = [
        Col("view", view, "l"),
        Col("model", [ARCH[r["model"]] for r in rows], "L"),
        Col("penalty", [r["penalty"] for r in rows], "l"),
        Col(r"unstable\\among\\equilibria", [num_ci(r, "unstable_eq", 2) for r in rows]),
        Col(r"poles\\incl.", [opt_num(r, "unstable_eq_full", 2) for r in rows]),
        Col(r"collided/\\profiles", [f"{f_int(fnum(r['collided']))}/{f_int(fnum(r['profiles']))}" for r in rows]),
        Col(r"first\\collided\\position", [f_num(fnum(r["first_collided"]), 1) for r in rows]),
        Col(r"growth\\error", [f_num(fnum(r["growth_error"]), 3) for r in rows]),
        Col("pairs", [f_int(fnum(r["reduction_pairs"])) for r in rows]),
        Col(r"reduction\\(\%)", [num_ci(r, "reduction", 1, "pct", unit=False) for r in rows]),
        Col("H1.3", [H13[r["h1_3"]] for r in rows], "l"),
    ]
    caption = (
        r"Experiment E5 on OpenACC (5 folds, seed 0; 5 runs per row): models trained on the ACC-driven (ACC) or on "
        r"the human-driven (human) events, without penalty (none) and with the chosen E2 weight (chosen), and "
        r"the platoon test of 50-vehicle platoons behind the leader profiles of the same driving mode. Unstable "
        r"among equilibria as in Table~\ref{tab:e1} (unit: run), poles incl.: with a speed counted as unstable when a "
        r"pole of the full-history loop lies outside the unit circle (recurrent models); collided: profiles whose "
        r"platoon collided, of the "
        r"profiles, summed over the runs; first collided position: one past the last follower without collision (51 "
        r"without collision), mean over the runs; growth error on the collision-free prefix of the platoon (the "
        r"followers ahead of the first collision, at least 3); reduction: 1 $-$ growth error with/without penalty, "
        r"paired by fold over the folds in which both runs have a growth error (pairs; pooled: over the three "
        r"penalised architectures of a view); H1.3: conf.\ confirmed, ref.\ refuted, open (fewer than 3 pairs or an "
        r"undecided interval). Full table: S6. " + CI_NOTE)
    return Tab("e5", "tab:e5", caption, cols, groups={8: ""}, note="from runs/_report/tables/e5.csv")


def main_e4() -> Tab:
    e4 = read_csv(SRC / "e4.csv")
    rmax = read_csv(M5 / "e4_rmax.csv")
    laws = {(r["corridor"], r["law"]): r for r in read_csv(SRC / "laws.csv")}

    def one(**kw: str) -> dict:
        found = [r for r in e4 if all(r[k] == v for k, v in kw.items())]
        assert len(found) == 1, kw
        return found[0]

    free_ft = one(model="residual_idm", variant="fine-tuned, free core")
    free_e1 = one(model="residual_idm", variant="E1, free core")
    mlp_ft = one(model="mlp", variant="fine-tuned")
    mlp_e1 = one(model="mlp", variant="E1")
    assert free_ft["data"] == mlp_ft["data"] == "ngsim_i80" and free_e1["data"] == mlp_e1["data"] == "follownet_highd"
    assert {fnum(r["runs"]) for r in (free_ft, free_e1, mlp_ft, mlp_e1)} == {25}
    R = {(r["corridor"], float(r["r_max"]), r["core"]): r for r in rmax}
    f1 = R[("I-80", 1.0, "free")]  # the free hybrid of E1 is the r_max = 1 row of the sweep
    assert same(f1["rmse_highd"], free_e1["rmse_s"]) and same(f1["unstable_eq_ngsim"], free_ft["unstable_eq"])
    assert f1["law"] == "residual_idm"
    for c in ("I-80", "US-101"):
        assert same(R[(c, 1.0, "free")]["macro_error"], laws[(c, "residual_idm")]["macro_error"])

    def corridor(law: str) -> tuple:
        a, b = laws[("I-80", law)], laws[("US-101", law)]
        assert fnum(a["runs"]) == fnum(b["runs"]) == 30, law
        return (num_ci(a, "macro_error", 3), num_ci(b, "macro_error", 3),
                f"{f_coll(fnum(a['collisions_per_1000_vkm']))} / {f_coll(fnum(b['collisions_per_1000_vkm']))}")

    rows = [(r"free hybrid, $r_{\max} = 1$\textsuperscript{a}", "---", f_int(fnum(free_ft["a_priori_holds"])),
             f_num(fnum(free_ft["unstable_eq"]), 2), f_num(fnum(free_e1["rmse_s"]), 2),
             f_num(fnum(free_ft["rmse_s"]), 2)) + corridor("residual_idm")]
    assert fnum(mlp_ft["certificates"]) == 0.0
    rows.append((r"free MLP\textsuperscript{b}", "---", "---", f_num(fnum(mlp_ft["unstable_eq"]), 2),
                 f_num(fnum(mlp_e1["rmse_s"]), 2), f_num(fnum(mlp_ft["rmse_s"]), 2)) + corridor("mlp"))
    for rm, core in ((0.1, "certified"), (0.2, "certified"), (0.3, "certified"), (0.5, "certified"), (0.3, "free")):
        a, b = R[("I-80", rm, core)], R[("US-101", rm, core)]
        for k in ("a_priori_highd", "a_priori_ngsim", "unstable_eq_ngsim", "rmse_highd", "rmse_ngsim"):
            assert same(a[k], b[k]), (rm, core, k)
        assert a["law"] == b["law"], (rm, core)
        assert fnum(a["runs_highd"]) == fnum(a["runs_ngsim"]) == 25 and fnum(a["runs_corridor"]) == fnum(b["runs_corridor"]) == 30
        name = ("certified" if core == "certified" else "free core") + rf", $r_{{\max}} = {rm:g}$"
        rows.append((name, f_int(fnum(a["a_priori_highd"])), f_int(fnum(a["a_priori_ngsim"])),
                     f_num(fnum(a["unstable_eq_ngsim"]), 2), f_num(fnum(a["rmse_highd"]), 2),
                     f_num(fnum(a["rmse_ngsim"]), 2), num_ci(a, "macro_error", 3), num_ci(b, "macro_error", 3),
                     f"{f_coll(fnum(a['collisions_per_1000_vkm']))} / {f_coll(fnum(b['collisions_per_1000_vkm']))}"))
    heads = ["variant", "highD", r"after\\fine-tuning", r"unstable\\among\\equilibria\\(fine-tuned)", "highD", "I-80",
             "I-80",
             "US-101", r"collisions per\\1000 veh-km,\\I-80 / US-101"]
    cols = [Col(heads[0], [r[0] for r in rows], "L")]
    for j in range(1, 9):
        cols.append(Col(heads[j], [r[j] for r in rows]))
    top = [("certificate (of 25)", 1, 2), ("spacing RMSE (m)", 4, 5), ("macro error", 6, 7)]
    caption = (
        r"Experiment E4: the certified hybrid (ResidualIDM with a margin-constrained IDM core and a "
        r"Lipschitz-bounded residual of amplitude $r_{\max}$, m/s$^2$) against its free counterparts, on highD and "
        r"after fine-tuning on NGSIM I-80 (25 runs per row and stage: 5 folds $\times$ 5 seeds). Certificate: runs "
        r"whose a priori certificate holds (---: no certificate); unstable among equilibria after fine-tuning "
        r"(unit: run); spacing RMSE of the test parts (m; unit: driver, event on highD); macro error of the corridor law of the "
        r"fine-tuned runs (unit: run, 30 per corridor) and its collisions per 1000 vehicle-km on I-80 / US-101. Free "
        r"hybrid: the ResidualIDM of E1 (no certificate step); free core: the hybrid with $r_{\max} = 0.3$ and no "
        r"certificate; the certified hybrid of the corridor design is the one with $r_{\max} = 0.3$ (law "
        r"residual\_idm\_certified). Full tables: S7 and S8. " + CI_NOTE)
    footer = (r"\textsuperscript{a} Corridor columns: the law residual\_idm. "
              r"\textsuperscript{b} Corridor columns: the law mlp (the fine-tuned MLP is the corridor MLP).")
    return Tab("e4", "tab:e4", caption, cols, top=top, footer=footer,
               note="from runs/_report/tables/e4.csv, runs/_tables/m5/e4_rmax.csv (fine-tuned rows) and "
                    "runs/_report/tables/laws.csv (corridor columns of the free hybrid and the MLP)")


LAWS_DEF = (
    r"idm\_global, the IDM with one parameter set per fold calibrated on I-80; idm\_heterogeneous, one interior "
    r"per-event IDM estimate per vehicle (\_all: all estimates); knn, mlp, pidl, gru, lstm, perl, the networks "
    r"fine-tuned on I-80, and \_penalty, the same with the chosen stability penalty; residual\_idm, the free "
    r"hybrid; residual\_idm\_certified, the certified hybrid ($r_{\max} = 0.3$\,m/s$^2$; \_het: per-event "
    r"certified cores)")


def main_laws() -> Tab:
    laws = {(r["corridor"], r["law"]): r for r in read_csv(SRC / "laws.csv")}
    inst = {(r["corridor"], r["law"]): r for r in read_csv(SRC / "instability.csv")}
    power = read_csv(SRC / "power.csv")
    scen, seeds = {r["scenarios"] for r in power}, {r["seeds"] for r in power}
    assert len(scen) == len(seeds) == 1
    names = DESIGN_LAWS + ["idm_heterogeneous_all", "residual_idm_certified_het"]
    unst, me1, me2, dyn, coll, ins = [], [], [], [], [], []
    for law in names:
        a, b = laws[("I-80", law)], laws.get(("US-101", law))
        assert fnum(a["runs"]) == 30 and (b is None or fnum(b["runs"]) == 30), law
        u = fnum(inst[("I-80", law)]["unstable_eq"])
        if ("US-101", law) in inst:
            assert fnum(inst[("US-101", law)]["unstable_eq"]) == u, law
        unst.append(f_num(u, 2))
        me1.append(num_ci(a, "macro_error", 3))
        me2.append(num_ci(b, "macro_error", 3) if b else "---")
        dyn.append(f"{f_num(fnum(a['macro_error_dynamic']), 3)} / "
                   + (f_num(fnum(b["macro_error_dynamic"]), 3) if b else "---"))
        coll.append(f"{f_coll(fnum(a['collisions_per_1000_vkm']))} / "
                    + (f_coll(fnum(b["collisions_per_1000_vkm"])) if b else "---"))
        ins.append(f_num(fnum(b["inserted_share"]), 3) if b else "---")
    only_i80 = [n for n in names if ("US-101", n) not in laws]
    assert only_i80 == ["idm_heterogeneous_all"], only_i80
    cols = [
        Col("law", [ident(n, True) for n in names], "L"),
        Col(r"unstable\\among\\equilibria", unst),
        Col(r"macro error,\\I-80", me1),
        Col(r"macro error,\\US-101", me2),
        Col(r"dynamic\\macro error,\\I-80 / US-101", dyn),
        Col(r"collisions per\\1000 veh-km,\\I-80 / US-101", coll),
        Col(r"inserted\\share,\\US-101", ins),
    ]
    caption = (
        r"Corridor laws on NGSIM I-80 (in sample) and US-101 (out of sample): 30 runs per law and corridor "
        + f"({scen.pop()} periods, {seeds.pop()} seeds each; unit: run). Laws: " + LAWS_DEF
        + r"; idm\_heterogeneous\_all has no runs on US-101. Unstable among equilibria: share of the "
        r"string-unstable equilibria among the equilibria of the members (audits of the member runs; for "
        r"idm\_heterogeneous, \_all and the cores of residual\_idm\_certified\_het the exact gain of their parameter "
        r"sets). Macro error: mean of the absolute signed relative errors of the components against the ground "
        r"truth; dynamic: fundamental diagram, wave speed, number of waves and wave amplitude only; collisions: "
        r"contact episodes per 1000 vehicle-km inside the analysis window; inserted: share of the demand inserted. "
        r"Further laws and the raw metrics: Tables S13, S14 and S17. " + CI_NOTE)
    return Tab("laws", "tab:laws", caption, cols,
               note="from runs/_report/tables/laws.csv and instability.csv (the 13 laws of the design, "
                    "idm_heterogeneous_all, residual_idm_certified_het)")


# ---------------------------------------------------------------- verdicts of every hypothesis
NUM = r"[+-]?[\d.]+"


def pct_ci_text(m: re.Match, k: int) -> str:
    """Groups k, k+1, k+2 of a match ('+12.7', '-14.7', '+34.3') -> '+12.7\\% [$-$14.7, +34.3]'."""
    return minus(m.group(k)) + r"\% [" + minus(m.group(k + 1)) + ", " + minus(m.group(k + 2)) + "]"


def basis_h11(b: str) -> str:
    m = re.fullmatch(rf"unstable among equilibria ({NUM}) \[({NUM}), ({NUM})\], RMSE vs idm ({NUM}) % "
                     rf"\[({NUM}) %, ({NUM}) %\]", b)
    assert m, b
    return f"{m.group(1)} [{m.group(2)}, {m.group(3)}]; RMSE " + pct_ci_text(m, 4)


def basis_h12(b: str) -> str:
    m = re.fullmatch(rf"band share not stable ({NUM}) \[({NUM}), ({NUM})\], RMSE change ({NUM}) % "
                     rf"\[({NUM}) %, ({NUM}) %\]", b)
    assert m, b
    return f"{m.group(1)} [{m.group(2)}, {m.group(3)}]; RMSE " + pct_ci_text(m, 4)


RULES = {"no weight reaches stable >= 0.9: largest stable share": "largest stable share",
         "stable >= 0.9: smallest validation RMSE": r"stable $\geq 0.9$"}


def basis_h12c(b: str) -> str:
    m = re.fullmatch(rf"weight ({NUM}) \((.*)\): band share not stable ({NUM}) \[({NUM}), ({NUM})\], RMSE change "
                     rf"({NUM}) % \[({NUM}) %, ({NUM}) %\] against E1 \(seed 0\)", b)
    assert m, b
    assert m.group(2) in RULES, m.group(2)
    return f"$w = {m.group(1)}$; {m.group(3)} [{m.group(4)}, {m.group(5)}]; RMSE " + pct_ci_text(m, 6)


def h13_parts(b: str) -> re.Match:
    m = re.match(rf"reduction ({NUM}) % \[({NUM}) %, ({NUM}) %\] over (\d+) pairs? \(collision-free prefix\); "
                 rf".*collided profiles (\d+)/(\d+) without, (\d+)/(\d+) with penalty", b)
    assert m, b
    return m


def basis_h12_1(b: str) -> str:
    parts = re.findall(rf"(throughput|travel time|wave speed) ({NUM}) \[({NUM}), ({NUM})\] "
                       r"(degraded by >= 0\.15|degraded|not degraded)", b)
    assert [p[0] for p in parts] == ["throughput", "travel time", "wave speed"], b
    strong = sum(p[4] == "degraded by >= 0.15" for p in parts)
    return " / ".join(minus(p[1]) for p in parts) + rf"; {strong} of 3 by $\geq 0.15$"


def main_verdicts() -> Tab:
    v4 = read_csv(SRC / "verdicts.csv")
    v5 = read_csv(SRC / "verdicts_h12.csv")
    assert {r["complete"] for r in v4 + v5} == {"True"}

    def get4(h: str, u: str) -> dict:
        found = [r for r in v4 if r["hypothesis"] == h and r["unit"] == u]
        assert len(found) == 1, (h, u)
        return found[0]

    def get5(h: str, u: str, c: str) -> dict:
        found = [r for r in v5 if r["hypothesis"] == h and r["unit"] == u and r["corridors"] == c]
        assert len(found) == 1, (h, u, c)
        return found[0]

    rows: list[tuple] = []  # hypothesis, unit, verdict, basis
    for u in ("mlp", "gru", "lstm"):
        r = get4("H1.1", u)
        rows.append(("H1.1", ARCH[u], r["verdict"], basis_h11(r["basis"])))
    r = get4("H1.1", "overall")
    assert r["basis"] == "holds for at least 2 of 3", r["basis"]
    rows.append(("H1.1", "overall", r["verdict"], "holds for at least 2 of the 3 networks"))
    for u in ("mlp", "pidl", "gru", "lstm", "perl", "residual_idm"):
        r = get4("H1.2", u)
        rows.append(("H1.2", ARCH[u], r["verdict"], basis_h12(r["basis"])))
    for u in ("gru", "lstm", "perl"):
        r = get4("H1.2 (combined)", u)
        rows.append(("H1.2c", ARCH[u], r["verdict"], basis_h12c(r["basis"])))
    for view, name in (("openacc_acc", "ACC view"), ("openacc_human", "human view")):
        r = get4("H1.3", f"{view}/pooled")
        m = h13_parts(r["basis"])
        rows.append(("H1.3", f"{name}, pooled", r["verdict"],
                     f"{pct_ci_text(m, 1)}, {m.group(4)} pairs; collided {m.group(5)}/{m.group(6)} $\\to$ "
                     f"{m.group(7)}/{m.group(8)}"))
    r = get4("H1.3", "openacc_human/gru")
    m = h13_parts(r["basis"])
    rows.append(("H1.3", "human view, GRU", r["verdict"], f"{pct_ci_text(m, 1)}, {m.group(4)} pairs"))
    ra, rh = get4("H1.3", "openacc_acc/mlp"), get4("H1.3", "openacc_human/mlp")
    assert ra["verdict"] == rh["verdict"]
    ma, mh = h13_parts(ra["basis"]), h13_parts(rh["basis"])
    assert ma.group(4) == mh.group(4)
    rows.append(("H1.3", "MLP, both views", ra["verdict"],
                 f"{pct_ci_text(ma, 1)} and {pct_ci_text(mh, 1)}, {ma.group(4)} pairs each"))
    r = get4("H1.5", "overall")
    m = re.fullmatch(rf"certified, fine-tuned: band share unstable ({NUM}) \[({NUM}), ({NUM})\]; fine-tuned MLP: "
                     rf"({NUM}) \[({NUM}), ({NUM})\]", r["basis"])
    assert m, r["basis"]
    rows.append(("H1.5", "overall", r["verdict"],
                 f"certified hybrid {m.group(1)} [{m.group(2)}, {m.group(3)}]; free MLP {m.group(4)} "
                 f"[{m.group(5)}, {m.group(6)}]"))
    for corridor in ("I-80", "US-101"):
        for u in ("mlp", "gru", "lstm"):
            r = get5("H12.1", u, corridor)
            rows.append(("H12.1", f"{ARCH[u]}, {corridor}", r["verdict"], basis_h12_1(r["basis"])))
    r = get5("H12.1", "overall", "I-80 and US-101")
    m = re.fullmatch(r"confirmed for >= 2 of 3 networks on every corridor: I-80: (.*); US-101: (.*)", r["basis"])
    assert m, r["basis"]
    conf = []
    for k, corridor in ((1, "I-80"), (2, "US-101")):
        units = dict(x.split(" ") for x in m.group(k).split(", "))
        conf.append(f"{corridor}: " + (", ".join(ARCH[u] for u, v in units.items() if v == "confirmed") or "none"))
    rows.append(("H12.1", "overall", r["verdict"], f"confirmed on {conf[0]}; {conf[1]}"))
    r = get5("H12.2", "residual_idm_certified", "I-80 and US-101")
    b = r["basis"]
    ws = re.findall(rf"(I-80|US-101): throughput ({NUM}) % \[[^\]]*\] (equivalent|not equivalent[^,]*), travel time "
                    rf"({NUM}) % \[[^\]]*\] (equivalent|not equivalent[^,]*), wave speed ({NUM}) % \[({NUM}) %, "
                    rf"({NUM}) %\] (equivalent|not equivalent)", b)
    assert len(ws) == 2 and all(w[2] == w[4] == "equivalent" and w[8] == "not equivalent" for w in ws), b
    mm = re.search(rf"micro: RMSE vs idm ({NUM}) % \[({NUM}) %, ({NUM}) %\] over (\d+) drivers, (not superior|superior)",
                   b)
    assert mm, b
    wave = ", ".join(f"{minus(w[5])}\\% [{minus(w[6])}, {minus(w[7])}] ({w[0]})" for w in ws)
    rows.append(("H12.2", "certified hybrid", r["verdict"],
                 f"wave speed not equivalent: {wave}; RMSE {pct_ci_text(mm, 1)}, {mm.group(5)}"))
    for corridor in ("I-80", "US-101"):
        r = get5("H12.3", "Spearman", corridor)
        m = re.fullmatch(rf"r ({NUM}) \[({NUM}), ({NUM})\] over (\d+) laws, p \(permutation\) ({NUM})", r["basis"])
        assert m, r["basis"]
        s = [x.lstrip("+") for x in m.groups()]
        rows.append(("H12.3", f"Spearman, {corridor}", r["verdict"],
                     f"$r = {s[0]}$ [{s[1]}, {s[2]}], {s[3]} laws, $p = {f_p(float(s[4]))}$"))
    r = get5("H12.3", "overall", "I-80 and US-101")
    assert r["basis"] == ("Spearman, r > 0.6, p < 0.05 and >= 10 laws on every corridor: I-80 confirmed, "
                          "US-101 confirmed"), r["basis"]
    rows.append(("H12.3", "overall", r["verdict"], "confirmed on both corridors"))
    labels, groups, last = [], {}, None
    for i, row in enumerate(rows):
        if row[0] != last:
            labels.append(["H1.2", "combined"] if row[0] == "H1.2c" else row[0])
            if i:
                groups[i] = ""
            last = row[0]
        else:
            labels.append("")
    cols = [
        Col("hypothesis", labels, "l"),
        Col("unit", [r[1] for r in rows], "L", weight=0.55),
        Col("verdict", [r[2] for r in rows], "l"),
        Col("basis", [r[3] for r in rows], "L", weight=1.45),
    ]
    caption = (
        r"Verdicts of the pre-specified hypotheses and the numbers that decide them; every verdict is drawn from "
        r"every run of the design. Basis: H1.1 (E1), share of unstable equilibria and RMSE against the IDM; H1.2 "
        r"(E2) and H1.2 combined (low-frequency arm, chosen Jacobian weight $w$), share not stable in band with the "
        r"penalty and RMSE change against E1; H1.3 (E5), reduction of the growth error on the collision-free prefix "
        r"of the platoons and its pairs (MLP: ACC view, then human view), collided profiles without $\to$ with "
        r"penalty; H1.5 (E4), share unstable in band after fine-tuning; H12.1, degradation $|e_{\mathrm{network}}| "
        r"- |e_{\mathrm{idm\_global}}|$ of throughput / travel time / wave speed and how many of them reach 0.15 "
        r"(two for a network; overall: two networks on both corridors); H12.2, two one-sided tests (margin "
        r"$\pm$10\%) of residual\_idm\_certified against idm\_global, equivalent in throughput and travel time on "
        r"both corridors, and the spacing RMSE against the IDM on the NGSIM I-80 test parts; H12.3, Spearman's $r$ "
        r"over the laws of a corridor between macro error and share of unstable equilibria, permutation $p$ (rule: "
        r"$r > 0.6$, $p < 0.05$, at least 10 laws). " + CI_NOTE + r" Every unit: Tables S23 and S31.")
    return Tab("verdicts", "tab:verdicts", caption, cols, groups=groups, sizes=("footnotesize",),
               note="from runs/_report/tables/verdicts.csv and verdicts_h12.csv (bases shortened)")


VARIANTS = [("baseline", "baseline"), ("gain0.01", "gain 0.01"), ("gain0.04", "gain 0.04"),
            ("nofeedback", "no feedback"), ("lc_low", "LC timid"), ("lc_high", "LC assertive")]


def main_sensitivity() -> Tab:
    rows = read_csv(M5 / "sensitivity.csv")
    lawset = ["idm_global", "residual_idm_certified", "gru"]
    S = {(r["variant"], r["law"]): r for r in rows}
    assert sorted(S) == sorted((v, law) for v, _ in VARIANTS for law in lawset)
    assert {fnum(r["runs"]) for r in rows} == {5}
    cells: dict[str, list] = {law: [] for law in lawset}
    first = []
    for v, _ in VARIANTS:
        for law in lawset:
            cells[law].append(num_ci(S[(v, law)], "macro_error", 3))
        ranked = sorted(lawset, key=lambda law: fnum(S[(v, law)]["macro_error"]))
        assert S[(v, ranked[0])]["rank"] == "1", v
        first.append(ident(ranked[0], True))
    cols = [Col("variant", [name for _, name in VARIANTS], "L")]
    for law in lawset:  # a long identifier breaks after its second underscore
        parts = law.split("_")
        head = ident("_".join(parts[:2]) + "_") + r"\\" + ident("_".join(parts[2:])) if len(parts) > 2 else ident(law)
        cols.append(Col(head, cells[law]))
    cols.append(Col("first", first, "L", weight=1.2))
    caption = (
        r"Sensitivity of the macro error to the corridor model: period 1 of I-80 (scenario i80\_p1, seeds 0--4, "
        r"5 runs per cell; unit: run) under the baseline boundary (downstream speed feedback of gain 0.02, LC2013 "
        r"lane-change defaults), the feedback gains 0.01 and 0.04, a speed-only boundary without feedback, and a "
        r"timid and an assertive lane-change setting (lcSpeedGain, lcCooperative, lcAssertive 0.5, 0.5, 0.5 and "
        r"2.0, 1.0, 1.5); first: the law with the smallest macro error. Law names as in Table~\ref{tab:laws}. "
        r"Full table with the macro triple and the collisions: S26. " + CI_NOTE)
    return Tab("sensitivity", "tab:sensitivity", caption, cols, note="pivot of runs/_tables/m5/sensitivity.csv")


def main_correlation() -> Tab:
    ic = read_csv(SRC / "instability_correlation.csv")
    cp = read_csv(M8 / "correlation_pooled.csv")

    def pick(corridor: str, method: str) -> dict:
        found = [r for r in ic if r["corridor"] == corridor and r["method"] == method and r["error"] == "macro error"
                 and r["measure"] == "unstable among equilibria" and r["laws"] == "all laws"]
        assert len(found) == 1, (corridor, method)
        return found[0]

    def pooled(scope: str, laws: str) -> dict:
        found = [r for r in cp if r["scope"] == scope and r["error"] == "macro error" and r["laws"].startswith(laws)]
        assert len(found) == 1, (scope, laws)
        return found[0]

    scope_name = {"I-80": "I-80", "US-101": "US-101", "pooled (corridor as stratum)": "pooled"}
    rows = []
    for laws_key, laws_name in (("laws of H12.3", "design"), ("all laws with runs", "all with runs")):
        for scope in ("I-80", "US-101", "pooled (corridor as stratum)"):
            pr = pooled(scope, laws_key)
            if laws_key == "laws of H12.3" and scope != "pooled (corridor as stratum)":
                s, p = pick(scope, "spearman"), pick(scope, "pearson")
                assert s["n"] == pr["n"] and abs(fnum(s["estimate"]) - fnum(pr["estimate"])) < 1e-12, scope
                rows.append((scope_name[scope], f"{laws_name} ({s['n']})", num_ci(s, "estimate", 3),
                             f_p(fnum(s["p_permutation"])), num_ci(p, "estimate", 3), f_p(fnum(p["p_permutation"]))))
            else:
                rows.append((scope_name[scope], f"{laws_name} ({pr['n']})", num_ci(pr, "estimate", 3),
                             f_p(fnum(pr["p_permutation"])), "---", ""))
    design = pooled("pooled (corridor as stratum)", "laws of H12.3")["law_list"].split(", ")
    assert design == DESIGN_LAWS + ["idm_heterogeneous_all", "residual_idm_certified_het"], design
    allr = pooled("pooled (corridor as stratum)", "all laws with runs")["law_list"].split(", ")
    extra = [x for x in allr if x not in design]
    assert extra == ["residual_idm_certified_r0.1", "residual_idm_certified_r0.2", "residual_idm_certified_r0.5",
                     "residual_idm_free_r0.3"], extra
    cols = [
        Col("scope", [r[0] for r in rows], "L"),
        Col(r"laws ($n$)", [r[1] for r in rows], "l"),
        Col("Spearman", [r[2] for r in rows]),
        Col(r"$p$", [r[3] for r in rows]),
        Col("Pearson", [r[4] for r in rows]),
        Col(r"$p$", [r[5] for r in rows]),
    ]
    caption = (
        r"Correlation over the laws between the mean macro error of a law and the share of unstable equilibria "
        r"among the equilibria of its members, per corridor and pooled with the corridor as a stratum (ranks within "
        r"the corridor; $n$: law--corridor pairs). Design: the laws of Table~\ref{tab:laws} (idm\_heterogeneous\_all "
        r"on I-80 only); all with runs: also residual\_idm\_certified\_r0.1, \_r0.2, \_r0.5 (other residual "
        r"amplitudes) and residual\_idm\_free\_r0.3 (free core). Percentile bootstrap over the laws within the "
        r"corridors (1000 resamples); $p$: two-sided permutation $p$-value (10\,000 permutations of the instability "
        r"within the corridors); Pearson's coefficient for the per-corridor design only. Full tables: S18, S20 and "
        r"S21.")
    return Tab("correlation", "tab:correlation", caption, cols, groups={3: ""},
               note="from runs/_report/tables/instability_correlation.csv (per corridor, design laws) and "
                    "runs/_tables/m8/correlation_pooled.csv (pooled, all laws)")


def main_temporal() -> Tab:
    rows = read_csv(M8 / "temporal.csv")
    assert [r["versus"] for r in rows] == ["idm_global", "residual_idm_certified", "mlp", "gru"]
    for r in rows:
        assert r["law"] == r["versus"] + "_p0" and fnum(r["runs"]) == fnum(r["runs_versus"]) == 20
        assert r["scenarios"] == "i80_p1, i80_p2"
    cols = [
        Col("law", [ident(r["versus"], True) for r in rows], "L"),
        Col(r"macro error,\\period 0", [num_ci(r, "macro_error_law", 3) for r in rows]),
        Col(r"macro error,\\all periods", [num_ci(r, "macro_error_versus", 3) for r in rows]),
        Col(r"relative\\difference (\%)", [num_ci(r, "macro_error_relative", 1, "pct", unit=False) for r in rows]),
        Col(r"$p$", [f_p(fnum(r["macro_error_p"])) for r in rows]),
        Col("outcome", [r["outcome"] for r in rows], "l"),
    ]
    caption = (
        r"Temporal hold-out on I-80: each law fine-tuned on period 0 only (period 0; law name with the suffix "
        r"\_p0) against the same law fine-tuned on all periods, both simulated on periods 1 and 2 (20 runs each: "
        r"2 periods $\times$ 10 seeds; unit: run). Relative difference: mean ratio of the paired macro errors minus "
        r"1 (paired by period and seed), $p$: Wilcoxon signed-rank test; outcome: worse or better when the interval "
        r"of the paired difference excludes 0. Law names as in Table~\ref{tab:laws}. Full table: S28. " + CI_NOTE)
    return Tab("temporal", "tab:temporal", caption, cols, note="from runs/_tables/m8/temporal.csv")


MAIN_BUILDERS = [main_e1, main_e2, main_e2_arms, main_e5, main_e4, main_laws, main_verdicts, main_sensitivity,
                 main_correlation, main_temporal]


def build_main(out: Path) -> list[Tab]:
    (out / OUT_MAIN).mkdir(parents=True, exist_ok=True)
    tabs = []
    for b in MAIN_BUILDERS:
        READ.clear()
        t = b()
        t.sources = list(READ)
        t.contexts = ("text",)
        choose(t, supplement=False)
        (out / OUT_MAIN / f"{t.name}.tex").write_text(emit(t, supplement=False), encoding="utf-8")
        tabs.append(t)
    return tabs


# ================================================================================================
# tables of the Supplementary Materials (full column sets of version 1)
@dataclass
class S:
    """A column of a supplementary table: CSV key, head, kind (text flag int num pct p weight), decimals,
    interval; fn(row) overrides the cell."""
    key: str
    head: str
    kind: str = "num"
    d: int = 2
    ci: bool = False
    align: str = ""
    fn: object = None


def cols_of(rows: list[dict], specs: list[S]) -> list[Col]:
    cols = []
    for s in specs:
        cells = []
        for r in rows:
            if s.fn is not None:
                v = s.fn(r)
            elif s.kind == "text":
                v = tex(r[s.key])
            elif s.kind == "flag":
                v = "yes" if r[s.key] in ("True", "1", "1.0") else ""
            elif s.kind == "int":
                v = f_int(fnum(r[s.key]))
            elif s.kind == "weight":
                v = f_w(fnum(r[s.key]))
            elif s.kind == "p":
                v = f_p(fnum(r[s.key]))
            elif s.kind == "pct":
                v = num_ci(r, s.key, s.d, "pct", unit=False) if s.ci else f_pct(fnum(r[s.key]), s.d, unit=False)
            else:
                v = num_ci(r, s.key, s.d) if s.ci else f_num(fnum(r[s.key]), s.d)
            cells.append(v)
        cols.append(Col(s.head, cells, s.align or ("l" if s.kind in ("text", "flag") else "r")))
    return cols


def runs_total(rows: list[dict], key: str = "runs") -> int:
    return int(sum(fnum(r[key]) or 0 for r in rows))


SUPP_LAW_NAMES = (
    r"Laws: idm\_global, the IDM with one parameter set per fold calibrated on I-80; idm\_heterogeneous, one "
    r"interior per-event IDM estimate per vehicle (\_all: all estimates); knn, mlp, pidl, gru, lstm, perl, the "
    r"networks fine-tuned on I-80, and \_penalty, with the chosen stability penalty; residual\_idm, the free hybrid; "
    r"residual\_idm\_certified, the certified hybrid ($r_{\max} = 0.3$\,m/s$^2$; \_het: per-event certified cores; "
    r"\_r0.1, \_r0.2, \_r0.5: other residual amplitudes); residual\_idm\_free\_r0.3, a free core with $r_{\max} = "
    r"0.3$\,m/s$^2$; suffix \_p0: fine-tuned on period 0 only and run on periods 1 and 2.")
LAWS_REF = r"Law names as in Table~\ref{tab:Slaws}."


def law_cell(r: dict) -> str:
    return ident(r["law"], True)


def s_e2_sweep() -> Tab:
    rows = read_csv(SRC / "e2_sweep.csv")
    cols = cols_of(rows, [
        S("architecture", "architecture", fn=lambda r: ARCH[r["architecture"]], align="L"),
        S("kind", "penalty", fn=lambda r: PENALTY[r["kind"]], align="l"),
        S("weight", "weight", "weight"), S("runs", "runs", "int"),
        S("val_rmse_s", r"validation spacing RMSE (m)", d=2), S("test_rmse_s", r"test spacing RMSE (m)", d=2),
        S("stable", "stable", d=2), S("feasible", "feasible", "int"), S("chosen", "chosen", "flag")])
    caption = (
        rf"Sweep of the penalty weight in E2 on highD: {runs_total(rows)} runs (5 folds, seed 0, four weights per "
        r"architecture); penalty: Jacobian penalty or rollout gain penalty (rollout). Spacing RMSE (m): mean over the "
        r"runs of the mean spacing RMSE of the validation and the test events; stable: share of the grid speeds in "
        r"support with a stable equilibrium inside the spacing band; feasible: runs whose best epoch meets the penalty "
        r"(at most 0.01 at the grid speeds); chosen: the smallest validation RMSE among the weights with a stable share "
        r"of at least 0.9, otherwise the largest stable share.")
    return Tab("S01_e2_sweep", "tab:Se2_sweep", caption, cols, note="from runs/_report/tables/e2_sweep.csv")


def s_e2_existence() -> Tab:
    rows = read_csv(SRC / "e2_existence.csv")
    cols = cols_of(rows, [
        S("architecture", "architecture", fn=lambda r: ARCH[r["architecture"]], align="L"),
        S("runs", "runs", "int"), S("drivers", "drivers", "int"), S("rmse_s", "spacing RMSE (m)", ci=True),
        S("rmse_vs_e1", r"RMSE vs E1 (\%)", "pct", 1, ci=True), S("rmse_vs_e1_p", "$p$", "p"),
        S("stable", "stable"), S("unstable", "unstable"), S("outside", "outside"), S("none", "none"),
        S("unstable_eq", "unstable among equilibria", ci=True), S("max_gain_median", "max gain (median)", d=2),
        S("collided_profiles", "collided profiles", d=1)])
    caption = (
        r"Existence arm of E2: the existence term alone (an equilibrium inside the spacing band at every speed, no "
        rf"stability term) for the recurrent models, {runs_total(rows)} runs (5 folds, seed 0) against E1 of seed 0. "
        r"Spacing RMSE (m; unit: driver, i.e., event on highD); RMSE vs E1: relative difference, paired over those units, $p$: Wilcoxon "
        r"signed-rank test; shares of the grid speeds in support (unit: run); max gain: largest numerical gain, median "
        r"over the runs; collided profiles: OpenACC platoon profiles with a collision, mean over the runs. " + CI_NOTE)
    return Tab("S02_e2_existence", "tab:Se2_existence", caption, cols, note="from runs/_report/tables/e2_existence.csv")


def s_e2_lowfreq() -> Tab:
    rows = read_csv(SRC / "e2_lowfreq.csv")
    cols = cols_of(rows, [
        S("architecture", "architecture", fn=lambda r: ARCH[r["architecture"]], align="L"),
        S("weight", "Jacobian weight", "weight"), S("chosen", "chosen", "flag"), S("runs", "runs", "int"),
        S("rmse_s", "spacing RMSE (m)", ci=True), S("rmse_change", r"RMSE change vs E1 (\%)", "pct", 1, ci=True),
        S("stable", "stable"), S("not_stable", "not stable", ci=True),
        S("unstable_eq", "unstable among equilibria", ci=True), S("max_gain_median", "max gain (median)", d=3),
        S("collided", "collided", "int"), S("profiles", "profiles", "int"), S("h1_2", "H1.2 (combined)", "text")])
    caption = (
        rf"Low-frequency arm of E2: {runs_total(rows)} runs of the combined penalty (the rollout gain penalty of E2, "
        r"the Jacobian penalty of the memoryless view and the guard against stiff equilibria) per architecture and "
        r"Jacobian weight (the pilot on fold 0, the chosen weight on all 5 folds; seed 0). Spacing RMSE (m; unit: "
        r"driver, i.e., event on highD); RMSE change against E1 of the same folds and seed, paired over those units; shares of the grid speeds "
        r"in support (unit: run); max gain: median over the runs; collided: OpenACC platoon profiles that collided, "
        r"of the profiles; H1.2 (combined): the rule of H1.2 applied to the chosen weight. " + CI_NOTE)
    return Tab("S03_e2_lowfreq", "tab:Se2_lowfreq", caption, cols, note="from runs/_report/tables/e2_lowfreq.csv")


def s_e2_monotone() -> Tab:
    rows = read_csv(M8 / "e2_monotone.csv")

    def arm(r: dict) -> str:
        if r["arm"] == "E1":
            return "E1"
        if r["arm"].startswith("E2"):
            m = re.fullmatch(r"E2 \(jacobian, weight ([\d.]+)\)", r["arm"])
            return rf"E2 (Jacobian, $w = {m.group(1)}$)"
        assert r["arm"].startswith("monotone"), r["arm"]
        weight = re.search(r"_w([\d.]+)$", r["experiment"]).group(1)
        return rf"monotone ($w = {weight}$)"

    cols = cols_of(rows, [
        S("architecture", "architecture", fn=lambda r: ARCH[r["architecture"]], align="L"),
        S("arm", "arm", fn=arm, align="l"), S("runs", "runs", "int"), S("rmse_s", "spacing RMSE (m)", ci=True),
        S("rmse_change", r"RMSE change vs E1 (\%)", "pct", 1, ci=True), S("rmse_change_p", "$p$", "p"),
        S("stable", "stable"), S("unstable", "unstable"), S("outside", "outside"), S("none", "none"),
        S("not_stable", "not stable"), S("unstable_eq", "unstable among equilibria"),
        S("max_gain", "max gain", d=3), S("collided", "collided profiles", "int")])
    caption = (
        r"Monotonicity control of E2: penalty $\mathrm{relu}(-f_s) + \mathrm{relu}(f_{\Delta v}) + \mathrm{relu}(f_v)$ "
        r"at the anchored equilibria of E2 (the sign constraints of RACER, without a string-stability term) next to E1 "
        r"and the chosen E2 weight; fold 0, seed 0, one run per row (highD). Spacing RMSE of the test part (m; unit: "
        r"driver, i.e., event on highD) and its change against E1, paired over those units, $p$: Wilcoxon signed-rank test; band shares of the "
        r"audit, unstable among equilibria and the largest measured gain; collided profiles: OpenACC platoon profiles "
        r"with a collision, of 5. " + CI_NOTE)
    assert {r["profiles"] for r in rows} == {"5"}
    return Tab("S04_e2_monotone", "tab:Se2_monotone", caption, cols, note="from runs/_tables/m8/e2_monotone.csv")


def s_e3() -> Tab:
    rows = read_csv(SRC / "e3.csv")
    cols = cols_of(rows, [
        S("penalty", "penalty", "text", align="L"), S("weight", "weight", "weight"),
        S("target", "target", fn=lambda r: TARGET_DATA[r["target"]], align="l"), S("drivers", "drivers", "int"),
        S("source_rmse_s", "source spacing RMSE (m)"), S("rmse_s_h", "spacing RMSE, first 15 s (m)", ci=True),
        S("rmse_s_full", "spacing RMSE, whole events (m)", ci=True),
        S("degradation", r"degradation (\%)", "pct", 1, ci=True),
        S("degradation_change", r"with $-$ without penalty (\%)", "pct", 1, ci=True)])
    groups = groups_of([r["architecture"] for r in rows], ARCH)
    caption = (
        r"Experiment E3 (transfer): the highD models of seed 0 (5 folds) on all events of the target data sets NGSIM "
        rf"I-80, NGSIM US-101 and the Waymo car-following pairs; {runs_total(rows)} evaluations; penalty: none (E1) or "
        r"the chosen E2 weight. Spacing RMSE (m) of the first 15\,s and of the whole events (unit: driver of the "
        r"target); degradation: RMSE of the first 15\,s relative to the source test RMSE, minus 1; with $-$ without "
        r"penalty: difference of the degradations, drivers paired. " + CI_NOTE)
    return Tab("S05_e3", "tab:Se3", caption, cols, groups=groups, note="from runs/_report/tables/e3.csv")


def s_e5() -> Tab:
    rows = read_csv(SRC / "e5.csv")
    titles = {"openacc_acc": "OpenACC, models of the ACC view", "openacc_human": "OpenACC, models of the human view"}
    ra = [r for r in rows if r["model"] != "pooled"]
    a = cols_of(ra, [
        S("model", "model", fn=lambda r: ARCH[r["model"]], align="L"), S("penalty", "penalty", "text"),
        S("runs", "runs", "int"), S("rmse_s", "spacing RMSE (m)", ci=True),
        S("unstable_eq", "unstable among equilibria", ci=True), S("not_stable", "not stable", ci=True),
        S("max_gain_median", "max gain (median)", d=3)])
    b = cols_of(rows, [
        S("model", "model", fn=lambda r: ARCH[r["model"]], align="L"), S("penalty", "penalty", "text"),
        S("collided", "collided", "int"), S("profiles", "profiles", "int"),
        S("first_collided", "first collided position", d=1, ci=True),
        S("collision_free", "collision-free share", d=2, ci=True), S("growth_error", "growth error", d=3),
        S("reduction_pairs", "pairs", "int"), S("reduction", r"reduction (\%)", "pct", 1, ci=True),
        S("reduction_p", "$p$", "p"), S("h1_3", "H1.3", "text"), S("reduction_full_pairs", "whole-curve pairs", "int"),
        S("reduction_full", r"whole-curve reduction (\%)", "pct", 1, ci=True)])
    pa = Tab("e5a", "", "", a, groups=groups_of([r["view"] for r in ra], titles),
             title=r"\textit{(a) Accuracy and stability}")
    pb = Tab("e5b", "", "", b, groups=groups_of([r["view"] for r in rows], titles), title=r"\textit{(b) Platoon test}")
    caption = (
        r"Experiment E5 on the OpenACC views (models trained on the ACC-driven or on the human-driven events): "
        rf"{runs_total(ra)} runs (5 folds, seed 0); penalty: none or the chosen E2 weight (chosen). (a) Spacing RMSE "
        r"(m; unit: driver); stability shares (unit: run); max gain: median over the runs. (b) Platoon test on the "
        r"profiles of the driving mode of the view: the growth error of a profile is taken on the collision-free "
        r"prefix of its simulated platoon (the followers ahead of the first collided position, at least 3), that of a "
        r"run is the mean over the profiles that have one; collided: profiles whose platoon collided, of the "
        r"profiles, summed over the runs; first collided position: one past the last follower without collision (51 "
        r"without collision); reduction: 1 $-$ growth error with/without penalty, paired by fold over the folds in "
        r"which both runs have a growth error (pairs; pooled: over the architectures of a view; H1.3 open with fewer "
        r"than 3 pairs), $p$: Wilcoxon signed-rank test; whole curve: the same on the growth error over the whole "
        r"curve (none after a collision). " + CI_NOTE)
    return Tab("S06_e5", "tab:Se5", caption, [], panels=[pa, pb], note="from runs/_report/tables/e5.csv, two panels")


def s_e4() -> Tab:
    rows = read_csv(SRC / "e4.csv")
    cols = cols_of(rows, [
        S("model", "model", fn=lambda r: ARCH[r["model"]], align="L"), S("variant", "variant", "text"),
        S("data", "data", fn=lambda r: DATA[r["data"]], align="l"), S("runs", "runs", "int"),
        S("a_priori_holds", "a priori holds", "int"), S("at_equilibria_holds", "at equilibria holds", "int"),
        S("all_equilibria_certified", "all equilibria certified", "int"), S("unstable", "unstable", ci=True),
        S("not_stable", "not stable", ci=True), S("unstable_eq", "unstable among equilibria", ci=True),
        S("rmse_s", "spacing RMSE (m)", ci=True)])
    caption = (
        rf"Experiment E4 (certified hybrid and fine-tuning): {runs_total(rows)} runs (5 folds $\times$ 5 seeds per "
        r"row). Certificates: runs in which the a priori certificate holds, in which the certificate at the "
        r"equilibria holds, and in which every equilibrium found is certified; shares of the grid speeds in support "
        r"(unit: run); spacing RMSE: test part of the data set of the row (m; unit: driver, event on highD). Free core: ResidualIDM "
        r"without certificate; fine-tuned: on NGSIM I-80. " + CI_NOTE)
    return Tab("S07_e4", "tab:Se4", caption, cols, note="from runs/_report/tables/e4.csv")


def s_e4_rmax() -> Tab:
    rows = read_csv(M5 / "e4_rmax.csv")
    cols = cols_of(rows, [
        S("r_max", r"$r_{\max}$ (m/s$^2$)", "weight"), S("core", "core", "text", align="L"),
        S("runs_highd", "runs", "int"), S("a_priori_highd", "a priori holds", "int"),
        S("rmse_highd", "spacing RMSE (m)", ci=True), S("runs_ngsim", "runs", "int"),
        S("a_priori_ngsim", "a priori holds", "int"), S("unstable_eq_ngsim", "unstable among equilibria", ci=True),
        S("rmse_ngsim", "spacing RMSE (m)", ci=True), S("macro_error", "macro error", d=3, ci=True),
        S("macro_error_dynamic", "macro error (dynamic)", d=3, ci=True),
        S("collisions_per_1000_vkm", "collisions per 1000 veh-km", d=3)])
    top = [("highD", 2, 4), ("fine-tuned on NGSIM I-80", 5, 8), ("corridor law", 9, 11)]
    caption = (
        r"Residual-amplitude sweep of the hybrid: per residual amplitude $r_{\max}$ and core (certified at 0.1, 0.2, "
        r"0.3 and 0.5\,m/s$^2$; free at 0.3 and 1\,m/s$^2$, the latter being the ResidualIDM of E1) the runs on highD "
        r"and after fine-tuning on NGSIM I-80 (5 folds $\times$ 5 seeds each): runs whose a priori certificate holds, "
        r"unstable among equilibria after fine-tuning (unit: run), spacing RMSE of the test parts (m; unit: driver, event on highD), "
        r"and the corridor results of the law (macro error, its dynamic part and collisions per 1000 vehicle-km; "
        r"unit: the run, i.e., scenario and seed). " + CI_NOTE)
    return Tab("S08_e4_rmax", "tab:Se4_rmax", caption, cols, top=top,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_tables/m5/e4_rmax.csv; columns grouped by data set")


def s_certificate_tightness() -> Tab:
    rows = read_csv(SRC / "certificate_tightness.csv")
    assert {r["runs"] for r in rows} == {"25"}
    cols = cols_of(rows, [
        S("stage", "stage (data)", fn=lambda r: f"{r['stage']} ({DATA[r['data']]})", align="L"),
        S("bound", "bound", "text"), S("runs", "runs", "int"), S("speeds", "speeds", "int"),
        S("certified_median", "certified bound, median", d=3), S("audited_median", "audited margin, median", d=3),
        S("slack_median", "slack, median", d=3), S("slack_q10", r"slack, 10\% quantile", d=3),
        S("slack_min", "slack, minimum", d=3), S("negative", "negative slack", "int"),
        S("negative_share", "share negative", d=3)])
    groups = groups_of([r["r_max"] for r in rows], {x: rf"$r_{{\max}} = {float(x):g}$\,m/s$^2$" for x in
                                                    {r["r_max"] for r in rows}})
    caption = (
        r"Tightness of the certificate of E4: every run of the certified hybrid (ResidualIDM, IDM core with margin "
        r"0.2\,s$^{-2}$, certified Lipschitz budget; 5 folds $\times$ 5 seeds, 25 runs per row) at the residual "
        r"amplitudes $r_{\max}$, before fine-tuning (highD) and after fine-tuning (NGSIM I-80). Per grid speed, the "
        r"audited margin $M$ of the hybrid at its equilibrium against the certified lower bound, a priori (every "
        r"spacing at which an equilibrium is possible; compared at every speed with an equilibrium) or at the "
        r"equilibria (the anchored equilibria the certificate uses). Slack = audited margin $-$ bound (s$^{-2}$): "
        r"median, 10\% quantile and minimum over the speeds of all runs; negative slack would violate the "
        r"certificate.")
    return Tab("S09_certificate_tightness", "tab:Scertificate_tightness", caption, cols, groups=groups,
               note="from runs/_report/tables/certificate_tightness.csv; experiment and data merged into the stage")


def s_lowfreq_expansion() -> Tab:
    rows = read_csv(SRC / "lowfreq_expansion.csv")

    def arm(r: dict) -> str:
        if r["experiment"] == "e1":
            return "E1"
        weight = re.search(r"_w([\d.]+)$", r["experiment"]).group(1)
        return rf"E2, $w = {weight}$"

    cols = cols_of(rows, [
        S("arm", "arm", fn=arm, align="L"), S("omega", r"$\omega$ (rad/s)", d=3),
        S("equilibria", "equilibria", "int"), S("flagged", "flagged", "int"), S("speeds", "speeds", "int"),
        S("abs_diff_median", r"$|\mathrm{num}-\mathrm{exp}|$, median", d=3),
        S("abs_diff_q90", r"$|\mathrm{num}-\mathrm{exp}|$, 90\% quantile", d=3),
        S("sign_agreement", r"same sign of $|G|-1$", d=3), S("sign_disagree", "sign differs", "int"),
        S("disagree_blind", r"of which $-\omega^2<M<0$", "int"),
        S("abs_diff_exact_median", r"$|\mathrm{num}-\mathrm{exact}|$, median", d=3),
        S("slow_pole_median", "slow pole, median (rad/s)", d=3), S("locally_unstable", "locally unstable", "int"),
        S("residual_large", r"fit residual $>0.1$", "int")])
    caption = (
        r"Low-frequency expansion of the gain: fold-0 runs (seed 0, highD) of E1 and of the chosen E2 weight $w$ "
        r"(IDM: E1 only). Per grid speed with an equilibrium, the numerical gain of the audit at the three lowest "
        r"audit frequencies against the expansion $|G| = \sqrt{\max(0, 1 - \omega^2 M / f_s^2)}$ with $M = f_v^2 + 2 "
        r"f_v f_{\Delta v} - 2 f_s$ from the stored partial derivatives (memoryless view, derivatives summed over the "
        r"window, for GRU, LSTM and PERL). Flagged: speeds whose rollout at that frequency was clipped, stopped or "
        r"collided, left out (speeds = equilibria $-$ flagged); $|\mathrm{num}-\mathrm{exp}|$: absolute difference "
        r"between the numerical gain and the expansion; same sign: share of the speeds with the same sign of $|G| - "
        r"1$, and the number of speeds where it differs, of which inside $-\omega^2 < M < 0$; "
        r"$|\mathrm{num}-\mathrm{exact}|$: absolute difference to the exact gain of the continuous memoryless "
        r"linearisation; slow pole: $|f_s| / |f_{\Delta v} + f_v|$; locally unstable (memoryless view not locally "
        r"stable) and fit residual $> 0.1$: speeds without a linear stationary response, kept in the numbers.")
    return Tab("S10_lowfreq_expansion", "tab:Slowfreq_expansion", caption, cols,
               groups=groups_of([r["architecture"] for r in rows], ARCH),
               note="from runs/_report/tables/lowfreq_expansion.csv")


def s_band_width() -> Tab:
    rows = read_csv(SRC / "band_width.csv")
    for r in rows:
        assert float(r["v"]).is_integer(), r["v"]
    cols = cols_of(rows, [
        S("v", r"$v$ (m/s)", "weight", align="R"), S("samples", "samples", "int"), S("band", "band", "flag"),
        S("s_low", r"$s_{5}$ (m)", d=3), S("s_median", r"$s_{50}$ (m)", d=3), S("s_high", r"$s_{95}$ (m)", d=3),
        S("relative_width", r"$(s_{95}-s_{5})/s_{50}$", d=3), S("drivers", "drivers", "int"),
        S("driver_std", "driver SD (m)", d=3), S("driver_std_relative", r"driver SD$/s_{50}$", d=3)])
    groups = groups_of([r["data"] for r in rows], {"follownet_highd": "highD (FollowNet events)",
                                                   "ngsim_i80": "NGSIM I-80"})
    caption = (
        r"Relative width of the spacing band and per-driver dispersion. Spacing band of the data: per grid speed $v$ "
        r"the 5\%, 50\% and 95\% quantiles $s_5$, $s_{50}$, $s_{95}$ of the spacing of the near-steady samples of all "
        r"events of the set ($|\Delta v| < 0.5$\,m/s, $|a| < 0.3$\,m/s$^2$, speed within 0.5\,m/s of the grid "
        r"speed); a speed with fewer than 200 such samples has no band. Driver SD: standard deviation over the drivers "
        r"with at least 20 near-steady samples at that speed of their median near-steady spacing; the highD events "
        r"carry no driver identifiers, so every event is its own driver.")
    return Tab("S11_band_width", "tab:Sband_width", caption, cols, groups=groups,
               note="from runs/_report/tables/band_width.csv")


def s_band_sensitivity() -> Tab:
    rows = read_csv(SRC / "band_sensitivity.csv")
    for h in ("runs", "runs_expected", "pairs"):
        assert {r[h] for r in rows} == {"25"}, h
    for h in ("indifferent", "undefined"):
        assert {fnum(r[h]) for r in rows} == {0.0} and {fnum(r[h + "_high"]) for r in rows} == {0.0}, h
    cols = cols_of(rows, [
        S("band", "band", "text", align="L"), S("stable", "stable", d=3, ci=True),
        S("unstable", "unstable", d=3, ci=True), S("outside", "outside", d=3, ci=True),
        S("none", "none", d=3, ci=True), S("unstable_eq", "unstable among equilibria (H1.1)", d=3, ci=True),
        S("unstable_eq_change", "change vs 5/95", d=3, ci=True), S("h1_1_share_rule", "H1.1 share rule", "text")])
    caption = (
        r"Band sensitivity of the audit: the E1 runs of the learned models on highD (5 folds $\times$ 5 seeds, 25 runs "
        r"per row) audited with three spacing bands: 1/99, 5/95 (the band of the main audit) and 10/90, i.e., the "
        r"1\%, 5\% or 10\% and the 99\%, 95\% or 90\% quantiles of the near-steady spacing. Stable, unstable, "
        r"outside, none: shares of the grid speeds in support (no speed is indifferent or undefined); unstable among "
        r"equilibria: the statistic of H1.1, and its change against the 5/95 band, paired by run; H1.1 share rule: "
        r"share $\geq 0.5$ with the lower end $> 0.3$. " + CI_NOTE)
    return Tab("S12_band_sensitivity", "tab:Sband_sensitivity", caption, cols,
               groups=groups_of([r["architecture"] for r in rows], ARCH),
               note="from runs/_report/tables/band_sensitivity.csv; constant columns in the caption")


def s_laws() -> Tab:
    rows = read_csv(SRC / "laws.csv")
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("runs", "runs", "int"),
        S("scenario_set", "scenarios", fn=lambda r: SCENARIOS[r["scenario_set"]], align="l"),
        S("macro_error", "macro error", d=3, ci=True), S("macro_error_dynamic", "macro error (dynamic)", d=3, ci=True),
        S("n_components", "components", d=1), S("collisions_per_1000_vkm", "collisions per 1000 veh-km", d=3, ci=True),
        S("runs_with_collisions", "runs with collisions", "int"), S("inserted_share", "inserted", d=3)])
    n = {c: sum(1 for r in rows if r["corridor"] == c) for c in ("I-80", "US-101")}
    caption = (
        rf"Corridor laws: {runs_total(rows)} runs (I-80: {n['I-80']} laws, US-101: {n['US-101']} laws; 3 scenarios, "
        r"10 seeds per scenario; unit: run). Macro error: mean of the absolute signed relative errors of the "
        r"components against the ground truth (components: mean number available); dynamic: fundamental diagram (FD), "
        r"wave speed, number of waves and wave amplitude only; collisions: contact episodes per 1000 vehicle-km inside "
        r"the analysis window; inserted: share of the demand inserted. " + SUPP_LAW_NAMES + " " + CI_NOTE)
    return Tab("S13_laws", "tab:Slaws", caption, cols, groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/laws.csv")


def s_laws_raw() -> Tab:
    rows = read_csv(SRC / "laws_raw.csv")
    for r in rows:
        if r["law"] == "ground truth":
            r["_label"] = f"ground truth, period {r['scenario'].split('_p')[-1]}"
        else:
            assert r["scenario"] == "all", r
            r["_label"] = ident(r["law"], True)
    order = []
    for c in ("I-80", "US-101"):
        order += [r for r in rows if r["corridor"] == c and r["law"] != "ground truth"]
        order += [r for r in rows if r["corridor"] == c and r["law"] == "ground truth"]
    rows = order
    cols = cols_of(rows, [
        S("law", "law", fn=lambda r: r["_label"], align="L"), S("runs", "runs", "int"),
        S("throughput_vph", "throughput (veh/h)", d=0, ci=True), S("mean_speed", "mean speed (m/s)", d=2, ci=True),
        S("queue_discharge_flow", "queue discharge (veh/h)", d=0, ci=True),
        S("capacity_drop", "capacity drop", d=3, ci=True), S("n_waves", "waves", d=1, ci=True),
        S("wave_speed_xcorr", "wave speed, xcorr (m/s)", d=2, ci=True),
        S("wave_amplitude", "wave amplitude (m/s)", d=2, ci=True),
        S("travel_time_mean", "mean travel time (s)", d=1, ci=True),
        S("collisions_per_1000_vkm", "collisions per 1000 veh-km", d=3, ci=True)])
    caption = (
        rf"Raw corridor metrics per law, mean over its runs ({runs_total(rows)} runs; unit: run; all scenarios of the "
        r"corridor, periods 1 and 2 for the laws with suffix \_p0), and the ground truth of every period (no "
        r"interval). Flows in veh/h at the throughput detector, speeds in m/s, travel times in s, collisions per 1000 "
        r"vehicle-km; wave speed by cross-correlation (xcorr). " + LAWS_REF + " " + CI_NOTE)
    return Tab("S14_laws_raw", "tab:Slaws_raw", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/laws_raw.csv; ground-truth rows at the end of their corridor")


def s_m5_components() -> Tab:
    rows = read_csv(SRC / "m5_components.csv")
    keys = ["anchored_throughput", "anchored_mean_speed", "anchored_queue_discharge_flow", "anchored_travel_time",
            "dynamic_fd", "dynamic_wave_speed", "dynamic_waves", "dynamic_wave_amplitude"]
    heads = ["throughput", "mean speed", "queue discharge", "travel time (W1)", "FD", "wave speed (xcorr)", "waves",
             "wave amplitude"]
    cols = cols_of(rows, [S("law", "law", fn=law_cell, align="L")]
                   + [S(k, h, d=3, ci=True) for k, h in zip(keys, heads)])
    caption = (
        rf"Components of the macro-error vector per law: signed relative errors against the ground truth "
        rf"({runs_total(rows)} runs; unit: run). The anchored components follow the demand and the downstream "
        r"boundary of the data, the dynamic ones are decided by the law. FD: RMSE of the binned flow of the "
        r"fundamental diagram over the mean flow of the ground truth; waves: (run $-$ truth)/max(truth, 1); travel "
        r"time: Wasserstein-1 distance (W1) over the mean travel time of the ground truth; wave speed: "
        r"cross-correlation estimate (xcorr). " + LAWS_REF + " " + CI_NOTE)
    return Tab("S15_m5_components", "tab:Sm5_components", caption, cols, top=[("anchored", 1, 4), ("dynamic", 5, 8)],
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/m5_components.csv")


def s_macro_error_dynamic() -> Tab:
    rows = read_csv(SRC / "macro_error_dynamic.csv")
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("runs", "runs", "int"),
        S("unstable_eq", "unstable among equilibria", d=3), S("not_stable", "not stable", d=3),
        S("macro_error_dynamic", "macro error (dynamic)", d=3, ci=True), S("collides", "collides", "flag")])
    caption = (
        r"Dynamic macro error per law: mean of the absolute dynamic components (FD, wave speed, waves, wave "
        rf"amplitude) with its interval ({runs_total(rows)} runs; unit: run); instability of the members as in "
        r"Table~\ref{tab:Sinstability}; collides: collisions per 1000 vehicle-km above 0. " + LAWS_REF + " "
        + CI_NOTE)
    return Tab("S16_macro_error_dynamic", "tab:Smacro_error_dynamic", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/macro_error_dynamic.csv")


def s_instability() -> Tab:
    rows = read_csv(SRC / "instability.csv")
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("members", "members", "int"), S("audited", "audited", "int"),
        S("unstable_eq", "unstable among equilibria", d=3), S("not_stable", "not stable", d=3),
        S("runs", "runs", "int"), S("macro_error", "macro error", d=3, ci=True),
        S("macro_error_dynamic", "macro error (dynamic)", d=3, ci=True),
        S("collisions_per_1000_vkm", "collisions per 1000 veh-km", d=3), S("collides", "collides", "flag"),
        S("scenario_set", "scenarios", fn=lambda r: SCENARIOS[r["scenario_set"]], align="l"),
        S("in_correlation", "in the correlation", "flag")])
    caption = (
        rf"Instability of the members of every corridor law and the macro errors: {runs_total(rows)} runs (unit: "
        r"run). Instability: audits of the member runs; for idm\_heterogeneous, idm\_heterogeneous\_all, the "
        r"certified cores of residual\_idm\_certified\_het and idm\_global\_p0 the exact gain of their parameter "
        r"sets; collides: collisions per 1000 vehicle-km above 0. The residual amplitudes are in "
        r"Table~\ref{tab:Se4_rmax}; the temporal hold-out laws (suffix \_p0) run on periods 1 and 2 only and are not "
        r"in the correlations. " + LAWS_REF + " " + CI_NOTE)
    return Tab("S17_instability", "tab:Sinstability", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/instability.csv")


def s_instability_correlation() -> Tab:
    rows = read_csv(SRC / "instability_correlation.csv")
    cols = cols_of(rows, [
        S("error", "error", "text", align="L"), S("measure", "instability", "text"), S("laws", "laws", "text"),
        S("method", "method", fn=lambda r: METHOD[r["method"]], align="l"), S("n", "$n$", "int"),
        S("estimate", "correlation", d=3, ci=True), S("n_valid", "valid resamples", "int"),
        S("p_permutation", "$p$ (permutation)", "p")])
    caption = (
        r"Spearman and Pearson correlation over the laws of a corridor between the mean macro error (all components or "
        r"the dynamic ones) and the instability of the members; percentile bootstrap over the laws (1000 resamples; "
        r"valid resamples: those in which neither variable is constant) and two-sided permutation $p$-value "
        r"(10\,000 permutations of the instability over the laws); all laws and the laws without collisions.")
    return Tab("S18_instability_correlation", "tab:Sinstability_correlation", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/instability_correlation.csv")


def s_macro_error_dynamic_correlation() -> Tab:
    rows = read_csv(SRC / "macro_error_dynamic_correlation.csv")
    cols = cols_of(rows, [
        S("measure", "instability", "text", align="L"), S("laws", "laws", "text"),
        S("method", "method", fn=lambda r: METHOD[r["method"]], align="l"), S("n", "$n$", "int"),
        S("estimate", "correlation", d=3, ci=True), S("n_valid", "valid resamples", "int"),
        S("p_permutation", "$p$ (permutation)", "p")])
    caption = (
        r"Spearman and Pearson correlation over the laws of a corridor between the dynamic macro error and the "
        r"instability of the members; percentile bootstrap over the laws (1000 resamples; valid resamples: those in "
        r"which neither variable is constant) and two-sided permutation $p$-value (10\,000 permutations); all laws "
        r"and the laws without collisions.")
    return Tab("S19_macro_error_dynamic_correlation", "tab:Smacro_error_dynamic_correlation", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/macro_error_dynamic_correlation.csv")


def s_correlation_pooled() -> Tab:
    rows = read_csv(M8 / "correlation_pooled.csv")
    cols = cols_of(rows, [
        S("scope", "corridor", "text", align="L"), S("error", "error", "text"),
        S("laws", "laws", fn=lambda r: {"all laws with runs": "all laws with runs",
                                        "laws of H12.3": "laws of the H12.3 verdicts"}[DCODE.sub("", r["laws"])],
          align="l"),
        S("n", "$n$", "int"), S("estimate", "Spearman", d=3, ci=True), S("p_permutation", "$p$ (permutation)", "p"),
        S("h12_3_rule", "rule of H12.3", "text")])
    caption = (
        r"Spearman correlation between the mean macro error of a law and the share of unstable equilibria of its "
        r"members, per corridor over the laws and pooled with the corridor as a stratum (ranks within the corridor); "
        r"bootstrap over the laws within the corridors (1000 resamples), two-sided permutation $p$-value within the "
        r"corridors (10\,000 permutations); all laws with runs and the laws of the H12.3 verdicts; rule of H12.3: "
        r"$r > 0.6$, $p < 0.05$ and $n \geq 10$. " + LAWS_REF)
    return Tab("S20_correlation_pooled", "tab:Scorrelation_pooled", caption, cols,
               note="from runs/_tables/m8/correlation_pooled.csv")


def s_correlation_pooled_laws() -> Tab:
    rows = read_csv(SRC / "correlation_pooled_laws.csv")
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("unstable_eq", "unstable among equilibria", d=3),
        S("macro_error", "macro error", d=3), S("macro_error_dynamic", "macro error (dynamic)", d=3)])
    caption = (
        r"Values of the pooled correlation per corridor and law: share of unstable equilibria among the equilibria of "
        r"the members (audits; for the closed-form IDM laws the exact gain of their parameter sets) and the mean "
        r"macro errors over the runs. " + LAWS_REF)
    return Tab("S21_correlation_pooled_laws", "tab:Scorrelation_pooled_laws", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/correlation_pooled_laws.csv")


def s_tost() -> Tab:
    rows = read_csv(SRC / "tost.csv")
    cols = cols_of(rows, [
        S("header", "metric", "text", align="L"), S("pairs", "pairs", "int"),
        S("mean_reference", r"idm\_global", d=3), S("mean_candidate", r"residual\_idm\_certified", d=3),
        S("relative_difference", r"relative difference (\%)", "pct", 1, ci=True),
        S("wilcoxon_p", "$p$ (Wilcoxon)", "p"), S("p_value", "$p$ (TOST)", "p"),
        S("equivalent", "equivalent", "flag")])
    cols[3].head = r"residual\_idm\_\\certified"
    caption = (
        r"Equivalence of residual\_idm\_certified and idm\_global per corridor, paired by scenario and seed (pairs): "
        r"means of the two laws, relative difference with its 95\% percentile bootstrap interval (1000 resamples), "
        r"Wilcoxon signed-rank $p$-value, and the $p$-value of the two one-sided tests (TOST, paired $t$-tests) for "
        r"the margin $\pm$10\% of the mean of idm\_global; equivalent: $p$ (TOST) below 0.05. A metric with a "
        r"negative mean (wave speed) is tested on its magnitude.")
    return Tab("S22_tost", "tab:Stost", caption, cols, groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/tost.csv")


def basis_text(b: str) -> str:
    b = b.replace("over 1 pairs", "over 1 pair").replace("RMSE vs idm ", "RMSE vs IDM ")
    return tex(b)


def s_verdicts_h12() -> Tab:
    rows = read_csv(SRC / "verdicts_h12.csv")
    assert {r["complete"] for r in rows} == {"True"}
    cols = cols_of(rows, [
        S("hypothesis", "hypothesis", "text"),
        S("unit", "unit", fn=lambda r: ARCH.get(r["unit"], ident(r["unit"], True)), align="L"),
        S("corridors", "corridors", "text"), S("verdict", "verdict", "text"),
        S("basis", "basis", fn=lambda r: basis_text(r["basis"]), align="L")])
    cols[1].weight, cols[4].weight = 0.5, 1.5
    caption = (
        r"Verdicts of the corridor hypotheses H12.1--H12.3 per network and corridor, overall (naming the corridors "
        r"used), for the certified hybrid and per correlation coefficient; every verdict is drawn from every run of "
        r"the design on both corridors. Intervals: 95\% percentile bootstrap (1000 resamples).")
    return Tab("S23_verdicts_h12", "tab:Sverdicts_h12", caption, cols,
               note="from runs/_report/tables/verdicts_h12.csv; the column complete (true in every row) is in the caption")


def s_h12_1() -> Tab:
    rows = read_csv(SRC / "h12_1.csv")
    cols = cols_of(rows, [
        S("network", "network", fn=lambda r: ARCH[r["network"]], align="L"), S("header", "metric", "text"),
        S("pairs", "pairs", "int"), S("abs_error_network", r"$|e|$ network", d=3),
        S("abs_error_reference", r"$|e|$ idm\_global", d=3), S("degradation", "degradation", d=3, ci=True),
        S("degraded", "degraded", "flag"), S("strong", r"degraded by $\geq 0.15$", "flag"),
        S("h12_1", "H12.1", "text")])
    caption = (
        r"H12.1: the networks MLP, GRU and LSTM against idm\_global per corridor. Degradation "
        r"$|e_{\mathrm{network}}| - |e_{\mathrm{idm\_global}}|$ of the signed relative errors of the throughput, the "
        r"travel time (Wasserstein-1 distance, W1) and the wave speed (cross-correlation estimate, xcorr), paired by "
        r"scenario and seed (pairs), mean with its 95\% percentile bootstrap interval (1000 resamples); degraded: "
        r"lower end $> 0$; degraded by $\geq 0.15$: degraded and the mean at least 0.15.")
    return Tab("S24_h12_1", "tab:Sh12_1", caption, cols, groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/h12_1.csv")


def s_h12_2() -> Tab:
    rows = read_csv(SRC / "h12_2.csv")
    names = {("e3_reference", "idm"): "IDM calibrated on NGSIM I-80",
             ("e4_stable_ft", "residual_idm"): "certified hybrid, fine-tuned",
             ("e4_free_ft", "residual_idm"): "free hybrid, fine-tuned", ("e4_free_ft", "mlp"): "MLP, fine-tuned"}
    cols = cols_of(rows, [
        S("role", "role", fn=lambda r: {"reference row": "comparison"}.get(r["role"], r["role"]), align="l"),
        S("model", "model", fn=lambda r: names[(r["experiment"], r["model"])], align="L"), S("runs", "runs", "int"),
        S("drivers", "drivers", "int"), S("rmse_s", "spacing RMSE (m)", ci=True),
        S("rmse_runs", "mean over runs (m)"), S("rmse_vs_reference", r"RMSE vs IDM (\%)", "pct", 1, ci=True),
        S("rmse_vs_reference_p", "$p$", "p"), S("superior", "superior", "flag")])
    caption = (
        r"H12.2, micro part: closed-loop spacing RMSE on the test parts of NGSIM I-80 (m; unit: driver) of the "
        r"fine-tuned certified hybrid and of two comparison rows against the IDM calibrated on NGSIM I-80 with the same "
        r"folds, paired over drivers, $p$: Wilcoxon signed-rank test; superior: upper end of the relative difference "
        r"below 0; mean over runs: mean over the runs of the event-mean RMSE. " + CI_NOTE)
    return Tab("S25_h12_2", "tab:Sh12_2", caption, cols, note="from runs/_report/tables/h12_2.csv")


def s_sensitivity() -> Tab:
    rows = read_csv(M5 / "sensitivity.csv")
    assert {r["order_holds"] for r in rows} == {"False"}
    titles = {"baseline": "Baseline: boundary gain 0.02, LC2013 defaults", "gain0.01": "Boundary gain 0.01",
              "gain0.04": "Boundary gain 0.04", "nofeedback": "No feedback (speed-only boundary)",
              "lc_low": "Lane change: lcSpeedGain, lcCooperative, lcAssertive 0.5, 0.5, 0.5",
              "lc_high": "Lane change: lcSpeedGain, lcCooperative, lcAssertive 2.0, 1.0, 1.5"}
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("runs", "runs", "int"),
        S("macro_error", "macro error", d=3, ci=True), S("error_throughput", "throughput", d=3, ci=True),
        S("error_travel_time", "travel time (W1)", d=3, ci=True),
        S("error_wave_speed", "wave speed (xcorr)", d=3, ci=True),
        S("collisions_per_1000_vkm", "collisions per 1000 veh-km", d=3), S("rank", "rank", "int")])
    caption = (
        r"Variants of period 1 of I-80 (scenario i80\_p1) and the scenario itself (baseline; the same seeds 0--4): "
        r"the gain of the downstream boundary feedback, a speed-only boundary, and the lane-change parameters of "
        r"LC2013. Per variant and law: the macro error, the signed relative errors of throughput, travel time (W1) and "
        r"wave speed (xcorr), and the collisions per 1000 vehicle-km (unit: run, i.e., seed); rank of the laws by "
        r"macro error within the variant. The order residual\_idm\_certified $<$ idm\_global $<$ gru holds in none of "
        r"the variants. " + LAWS_REF + " " + CI_NOTE)
    return Tab("S26_sensitivity", "tab:Ssensitivity", caption, cols,
               groups=groups_of([r["variant"] for r in rows], titles),
               note="from runs/_tables/m5/sensitivity.csv; the order column (false in every row) is in the caption")


def s_power() -> Tab:
    rows = read_csv(SRC / "power.csv")
    assert {(r["law_a"], r["law_b"]) for r in rows} == {("idm_global", "residual_idm_certified")}
    cols = cols_of(rows, [
        S("header", "metric", "text", align="L"), S("pairs", "pairs", "int"), S("mean_a", "mean A", d=3),
        S("mean_b", "mean B", d=3), S("sd_a", "SD A", d=4), S("sd_b", "SD B", d=4),
        S("sd_difference", r"SD of B $-$ A", d=4), S("n_one_scenario", "seeds, one scenario", "int"),
        S("n_per_scenario", "seeds per scenario", "int"), S("power_design", "power of the design", d=3),
        S("mdd_relative", r"detectable difference (\%)", "pct", 1),
        S("observed", r"observed difference B/A $-$ 1 (\%)", "pct", 1, ci=True)])
    caption = (
        r"Seed-to-seed spread and power of the corridor comparison of A = idm\_global and B = "
        r"residual\_idm\_certified, paired by seed within every scenario (pairs). SD: standard deviation over the "
        r"seeds within a scenario, pooled over the scenarios; seeds needed to detect a 10\% difference at $\alpha = "
        r"0.05$ with power 0.8 (paired $t$-test), for one scenario and per scenario when all scenarios contribute; "
        r"power of the design and the smallest difference it detects; observed difference with its 95\% percentile "
        r"bootstrap interval (1000 resamples).")
    return Tab("S27_power", "tab:Spower", caption, cols, groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/power.csv")


def s_temporal() -> Tab:
    rows = read_csv(M8 / "temporal.csv")
    for r in rows:
        assert r["law"] == r["versus"] + "_p0"
    cols = cols_of(rows, [
        S("versus", "law", fn=lambda r: ident(r["versus"], True), align="L"), S("runs", "runs", "int"),
        S("macro_error_law", "macro error, period 0", d=3, ci=True),
        S("macro_error_versus", "macro error, all periods", d=3, ci=True),
        S("macro_error_difference", "difference", d=3, ci=True),
        S("macro_error_relative", r"relative difference (\%)", "pct", 1, ci=True),
        S("macro_error_dynamic_law", "dynamic, period 0", d=3, ci=True),
        S("macro_error_dynamic_versus", "dynamic, all periods", d=3, ci=True),
        S("collisions_per_1000_vkm_law", "collisions per 1000 veh-km, period 0", d=3),
        S("outcome", "period-0 law", "text")])
    caption = (
        r"Temporal hold-out on I-80: each law fine-tuned on period 0 only (suffix \_p0) against the same law "
        r"fine-tuned on all periods, both simulated on periods 1 and 2 (20 runs each): macro error and dynamic macro "
        r"error (unit: run), their paired difference and relative difference (by scenario and seed); period-0 law: "
        r"worse or better when the interval of the difference excludes 0. " + CI_NOTE)
    return Tab("S28_temporal", "tab:Stemporal", caption, cols, note="from runs/_tables/m8/temporal.csv")


def s_asymmetry() -> Tab:
    rows = read_csv(SRC / "asymmetry.csv")
    for r in rows:
        if r["law"] == "ground truth":
            r["_label"] = ("ground truth, mean" if r["scenario"] == "mean"
                           else f"ground truth, period {r['scenario'].split('_p')[-1]}")
        else:
            assert r["scenario"] == "all", r
            r["_label"] = ident(r["law"], True)
    order = []
    for c in ("I-80", "US-101"):
        order += [r for r in rows if r["corridor"] == c and r["law"] != "ground truth"]
        order += [r for r in rows if r["corridor"] == c and r["law"] == "ground truth"]
    rows = order
    cols = cols_of(rows, [
        S("law", "law", fn=lambda r: r["_label"], align="L"), S("runs", "runs", "int"),
        S("asymmetry_index", "asymmetry index", d=3, ci=True), S("accelerating_share", "accelerating", d=3, ci=True),
        S("decelerating_share", "decelerating", d=3, ci=True), S("peak_frequency", "peak frequency (Hz)", d=4),
        S("centroid", "spectral centroid (Hz)", d=4, ci=True), S("band_rms", "band RMS (m/s)", d=3, ci=True),
        S("error_asymmetry_index", r"error of the index (\%)", "pct", 1, ci=True),
        S("error_peak_frequency", r"error of the peak (\%)", "pct", 1),
        S("error_centroid", r"error of the centroid (\%)", "pct", 1, ci=True)])
    caption = (
        r"Acceleration asymmetry and oscillation spectrum per corridor and law (unit: run; mean over all scenarios of "
        r"the corridor, periods 1 and 2 for the laws with suffix \_p0) and of the ground truth of every period and "
        r"their mean. Asymmetry index: mean acceleration above 0.1\,m/s$^2$ over the mean magnitude of the "
        r"decelerations below $-0.1$\,m/s$^2$ (1\,s differences of the speeds); accelerating, decelerating: shares "
        r"of the vehicle-seconds; peak frequency, spectral centroid (0.002--0.05\,Hz) and band root-mean-square (RMS) "
        r"of the Welch spectrum of the detector speeds (2\,s samples, 128\,s segments); errors: signed relative errors "
        r"against the ground truth of the run's scenario. " + LAWS_REF + " " + CI_NOTE)
    return Tab("S29_asymmetry", "tab:Sasymmetry", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/asymmetry.csv; ground-truth rows at the end of their corridor")


def s_asymmetry_contrasts() -> Tab:
    rows = read_csv(SRC / "asymmetry_contrasts.csv")
    cols = cols_of(rows, [
        S("law", "law", fn=law_cell, align="L"), S("versus", "free law", fn=lambda r: ident(r["versus"]), align="l"),
        S("pairs", "pairs", "int"), S("asymmetry_index_difference", "index difference", d=3, ci=True),
        S("index_changed", "changed", "flag"), S("closer_asymmetry_index", r"$|e|$ difference", d=3, ci=True),
        S("index_closer", "to the truth", "text"),
        S("accelerating_share_difference", "accelerating share", d=3, ci=True),
        S("centroid_difference", "centroid (Hz)", d=4, ci=True), S("band_rms_difference", "band RMS (m/s)", d=3, ci=True)])
    caption = (
        r"Penalised and certified laws against their free counterparts, paired by scenario and seed (pairs): "
        r"differences (law $-$ free law) of the asymmetry index, of its distance to the ground truth "
        r"($|e_{\mathrm{law}}| - |e_{\mathrm{free}}|$, negative: closer), of the accelerating share, of the spectral "
        r"centroid and of the band RMS; changed: the interval of the index difference excludes 0; to the truth: closer "
        r"or further when the interval of $|e|$ difference excludes 0. " + LAWS_REF + " " + CI_NOTE)
    return Tab("S30_asymmetry_contrasts", "tab:Sasymmetry_contrasts", caption, cols,
               groups=groups_of([r["corridor"] for r in rows], CORRIDOR_TITLE),
               note="from runs/_report/tables/asymmetry_contrasts.csv")


def verdict_unit(u: str) -> str:
    if "/" in u:
        v, m = u.split("/")
        return {"openacc_acc": "ACC view", "openacc_human": "human view"}[v] + ", " + ARCH.get(m, m)
    return ARCH.get(u, u)


def s_verdicts() -> Tab:
    rows = read_csv(SRC / "verdicts.csv")
    assert {r["complete"] for r in rows} == {"True"}
    cols = cols_of(rows, [
        S("hypothesis", "hypothesis", "text"), S("unit", "unit", fn=lambda r: verdict_unit(r["unit"]), align="L"),
        S("verdict", "verdict", "text"), S("basis", "basis", fn=lambda r: basis_text(r["basis"]), align="L")])
    cols[1].weight, cols[3].weight = 0.5, 1.5
    caption = (
        r"Verdicts of the pre-specified hypotheses H1.1 (E1), H1.2 (E2), H1.2 combined (the low-frequency arm of "
        r"E2), H1.3 (E5, growth error on the collision-free prefix of the platoons) and H1.5 (E4) with their basis; "
        r"every verdict is drawn from every run of the design. Intervals: 95\% percentile bootstrap (1000 resamples).")
    return Tab("S31_verdicts", "tab:Sverdicts", caption, cols,
               note="from runs/_report/tables/verdicts.csv; the column complete (true in every row) is in the caption")


# ================================================================================================
# generic tables of the Supplementary Materials: a CSV of runs/_tables/<milestone>/ and the markdown report of the
# same stem, whose tables give the columns, their heads, their rounding and the order of the rows. Every cell of the
# report is reproduced from the full-precision CSV (the CSV key of a column is the one whose values, formatted as
# the report formats them, give every cell of the column), so a report out of step with its CSV stops the build.
@dataclass
class MdTable:
    title: str  # the line before the table in the report ("" if none)
    heads: list[str]
    rows: list[list[str]]


@dataclass
class MdDoc:
    title: str
    notes: list[str]
    tables: list[MdTable]


def md_cells(line: str) -> list[str]:
    """Cells of a markdown table row ('\\|' is a literal bar)."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def read_md(path: Path) -> MdDoc:
    lines = used(path).read_text(encoding="utf-8").splitlines()
    title, notes, tables, before, i = "", [], [], "", 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"\|\s*:?-{3}", lines[i + 1]):
            heads, rows = md_cells(line), []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(md_cells(lines[i]))
                i += 1
            assert all(len(r) == len(heads) for r in rows), (path.name, heads)
            tables.append(MdTable(before, heads, rows))
            before = ""
            continue
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("- "):
            notes.append(line[2:].strip())
        elif line.startswith("  ") and notes and line.strip():
            notes[-1] += " " + line.strip()
        elif line.strip():
            before = line.strip().rstrip(":")
        i += 1
    return MdDoc(title, notes, tables)


MD_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
MD_FORMS = [  # numeric cells of the reports (units % removed first)
    ("ci", re.compile(rf"({MD_NUM}) \[({MD_NUM}), ({MD_NUM})\]")),  # estimate [interval]
    ("iv", re.compile(rf"\[({MD_NUM}), ({MD_NUM})\]")),  # interval alone
    ("pair", re.compile(rf"({MD_NUM}) \(({MD_NUM})\)")),  # median (quantile)
    ("one", re.compile(rf"({MD_NUM})")),
]


def md_cell(cell: str) -> tuple[str, list[str], bool]:
    """(kind, number strings, percent) of a cell: blank, na (n/a), p (<0.001), ci, iv, pair, one or text."""
    if cell == "":
        return "blank", [], False
    if cell == "n/a":
        return "na", [], False
    if cell == "<0.001":
        return "p", [], False
    pct = "%" in cell
    c = cell.replace(" %", "").replace("%", "") if pct else cell
    for kind, rx in MD_FORMS:
        m = rx.fullmatch(c)
        if m:
            return kind, list(m.groups()), pct
    return "text", [cell], False


@dataclass(frozen=True)
class NumFmt:
    """The rounding of a report column: decimals (of the mantissa when scientific), sign, percent."""
    d: int
    sci: bool = False
    sign: bool = False
    pct: bool = False

    def md(self, x: float) -> str:
        v = 100.0 * x if self.pct else x
        return format(v, f"{'+' if self.sign else ''}.{self.d}{'e' if self.sci else 'f'}")

    def tex(self, x: float) -> str:
        s = self.md(x)
        if not self.sci:
            return minus(s)
        mant, sg, ex = re.fullmatch(r"([+-]?[\d.]+)e([+-])(\d+)", s).groups()
        return rf"${mant}\times 10^{{{'-' if sg == '-' else ''}{int(ex)}}}$"


def num_fmt(strings: list[str], pct: bool) -> NumFmt:
    def decimals(s: str) -> int:
        mant = s.lower().split("e")[0]
        return len(mant.split(".")[1]) if "." in mant else 0
    return NumFmt(max((decimals(s) for s in strings), default=0), any("e" in s.lower() for s in strings),
                  any(s.startswith("+") for s in strings), pct)


@dataclass
class MdCol:
    """A column of a report table and the CSV keys that reproduce it."""
    head: str
    kind: str  # text, num, p, ci (estimate, low, high), iv (low, high), pair (median, quantile)
    keys: tuple
    fmts: tuple
    cells: list[str]
    part: int = 0


def md_kind(head: str, parsed: list) -> str:
    kinds = {k for k, _, _ in parsed} - {"blank", "na"}
    if not kinds:
        return "empty"
    if kinds <= {"one", "p"}:
        return "p" if "p" in kinds or re.match(r"p\b", head) else "num"
    if "ci" in kinds and kinds <= {"ci", "one"}:
        return "ci"
    if kinds == {"iv"}:
        return "iv"
    if "pair" in kinds and kinds <= {"pair", "one"}:
        return "pair"
    return "text"


def md_fmts(kind: str, parsed: list) -> tuple:
    pct = any(p for _, _, p in parsed)
    if kind == "num":
        return (num_fmt([n[0] for k, n, _ in parsed if k == "one"], pct),)
    if kind in ("ci", "iv"):
        f = num_fmt([x for k, n, _ in parsed if k in ("ci", "iv", "one") for x in n], pct)
        return (f, f, f) if kind == "ci" else (f, f)
    if kind == "pair":
        return (num_fmt([n[0] for k, n, _ in parsed if k in ("pair", "one")], pct),
                num_fmt([n[1] for k, n, _ in parsed if k == "pair"], pct))
    return ()


TEXT_EQ = {("True", "yes"), ("False", "no"), ("", "n/a"), ("nan", "n/a")}


def md_fits(kind: str, keys: tuple, fmts: tuple, parsed: list, cells: list[str], rows: list[dict]) -> bool:
    """Do the CSV columns keys, formatted as the report formats them, give every cell of the column?"""
    for (ck, nums, _), cell, r in zip(parsed, cells, rows):
        if ck == "blank":  # left empty by the report (e.g. a difference to itself): no constraint
            continue
        if kind == "text":
            v = (r.get(keys[0]) or "").strip()
            if not (v == cell or (v, cell) in TEXT_EQ):
                return False
            continue
        vals = [fnum(r.get(k)) for k in keys]
        if ck == "na":
            if vals[0] is not None:
                return False
            continue
        if any(v is None for v in vals[:1 if ck == "one" else len(vals)]):
            return False
        if kind == "p":
            if (ck == "p") != (vals[0] < 0.001) or (ck == "one" and f"{vals[0]:.3f}" != nums[0]):
                return False
        elif kind == "num":
            if fmts[0].md(vals[0]) != nums[0]:
                return False
        elif ck == "one":  # an estimate without its interval (ci) or quantile (pair)
            if fmts[0].md(vals[0]) != nums[0]:
                return False
        elif any(f.md(v) != s for f, v, s in zip(fmts, vals, nums)) or len(nums) != len(keys):
            return False
    return True


def is_number(v: str | None) -> bool:
    if v is None or v.strip() == "":
        return True
    try:
        float(v)
    except ValueError:
        return False
    return True


def similarity(head: str, key: str) -> float:
    h = set(re.findall(r"[a-z0-9]+", head.lower()))
    k = set(key.lower().split("_"))
    return len(h & k) / len(h | k) if h | k else 0.0


def md_match(name: str, head: str, cells: list[str], rows: list[dict], used: set) -> MdCol | None:
    """The CSV keys of a report column; among several that fit, the one closest to the head (then the first)."""
    keys = list(rows[0])
    parsed = [md_cell(c) for c in cells]
    kind = md_kind(head, parsed)
    if kind == "empty":
        return None
    fmts = md_fmts(kind, parsed)
    numeric = [k for k in keys if all(is_number(r.get(k)) for r in rows)]
    have = set(keys)
    if kind == "text":
        cands = [(k,) for k in keys]
    elif kind in ("num", "p"):
        cands = [(k,) for k in numeric]
    elif kind == "ci":
        cands = [(k, f"{k}_low", f"{k}_high") for k in numeric if f"{k}_low" in have and f"{k}_high" in have]
    elif kind == "iv":
        cands = [(k, k[:-4] + "_high") for k in numeric if k.endswith("_low") and k[:-4] + "_high" in have]
    else:
        cands = [(a, b) for a in numeric for b in numeric if a != b]
    good = [c for c in cands if md_fits(kind, c, fmts, parsed, cells, rows)]
    if not good:
        raise SystemExit(f"{name}: no CSV column reproduces the column {head!r} ({kind}) of the report")
    fresh = [c for c in good if c[0] not in used] or good
    best = min(fresh, key=lambda c: (-similarity(head, c[0]), keys.index(c[0])))
    return MdCol(head, kind, best, fmts, cells)


def md_columns(name: str, doc: MdDoc, rows: list[dict]) -> list[MdCol]:
    """The columns of every table of the report; a later table that repeats the row keys of the first (same
    text columns) is joined to it, its own columns appended (part 1, 2, ...)."""
    cols: list[MdCol] = []
    used: set = set()
    shared: set = set()
    for p, mt in enumerate(doc.tables):
        assert len(mt.rows) == len(rows), (name, mt.title, len(mt.rows), len(rows))
        for j, head in enumerate(mt.heads):
            c = md_match(name, head, [r[j] for r in mt.rows], rows, used)
            if c is None:
                continue
            if p and (c.keys in shared or any(c.keys == x.keys for x in cols)):
                continue
            c.part = p
            cols.append(c)
            used.add(c.keys[0])
            if p == 0 and c.kind == "text":
                shared.add(c.keys)
    return cols


def md_text(key: str, cell: str) -> str:
    """A text cell: architectures and data sets by name, law identifiers breakable after an underscore."""
    if key in ("architecture", "model", "network"):
        return ARCH.get(cell, tex(cell))
    if key in ("law", "versus"):
        return ident(cell, True)
    if key == "data":
        return DATA.get(cell, tex(cell))
    return tex(cell).replace("k-NN", "$k$-NN")


def md_values(c: MdCol, rows: list[dict]) -> list:
    out: list = []
    for cell, r in zip(c.cells, rows):
        ck = md_cell(cell)[0]
        if ck == "blank":
            out.append("")
            continue
        if ck == "na":
            out.append("---")
            continue
        if c.kind == "text":
            out.append(md_text(c.keys[0], cell))
            continue
        vals = [fnum(r[k]) for k in c.keys]
        if c.kind == "p":
            out.append(f_p(vals[0]))
        elif c.kind == "num":
            out.append(c.fmts[0].tex(vals[0]))
        elif c.kind == "iv":
            out.append(f"[{c.fmts[0].tex(vals[0])}, {c.fmts[1].tex(vals[1])}]")
        elif ck == "one":
            out.append(V(c.fmts[0].tex(vals[0])))
        elif c.kind == "ci":
            f = c.fmts[0]
            out.append(V(f.tex(vals[0]), f"[{f.tex(vals[1])}, {f.tex(vals[2])}]"))
        else:
            out.append(V(c.fmts[0].tex(vals[0]), f"({c.fmts[1].tex(vals[1])})"))
    return out


def md_head(c: MdCol) -> str:
    """A head of the report as LaTeX: no decision codes; p, n and |e| in math; the unit (%) of a percentage."""
    h = tex(DCODE.sub("", c.head).strip())
    h = re.sub(r"^p\b", "$p$", h)
    h = re.sub(r"^n$", "$n$", h).replace("|e|", "$|e|$")
    if c.fmts and c.fmts[0].pct and r"\%" not in h:
        h += r" (\%)"
    return h


def md_caption(doc: MdDoc) -> str:
    """Default caption: the title and the first note of the report, without milestone tags, decision codes, file
    names and paths, with the data sets by name."""
    def clean(s: str) -> str:
        s = re.sub(r"\s*\((?:M\d+|review of M\d+|revision[^)]*)\)|,\s*review of M\d+|review of M\d+,\s*", "", s)
        s = re.sub(r"\s*\(D\d+[^)]*\)|\s+(?:of|in|by) D\d+\b|\bD\d+\b", "", s)  # decision codes
        words = []
        for w in s.split(" "):  # paths (two slashes, an underscore or an extension) and file names; units stay
            core = w.strip("(),;:.")
            path = "/" in core and (core.count("/") > 1 or "_" in core or re.search(r"\.[a-z]{2,7}$", core))
            name = re.fullmatch(r"[\w.-]+\.(?:json|csv|md|txt|yaml|py|parquet|pdf|png)", core)
            words.append(w.replace(core, "") if core and (path or name) else w)
        s = " ".join(words)
        for k, v in DATA.items():
            s = s.replace(k, v)
        for _ in range(2):  # punctuation left behind
            s = re.sub(r"\(\s*[,;]\s*", "(", s)
            s = re.sub(r"\s*[,;]\s*\)", ")", s)
            s = re.sub(r"\(\s*\)", "", s)
            s = re.sub(r"\s+([,;:.)])", r"\1", re.sub(r"\s{2,}", " ", s))
        return tex(s.strip().rstrip(".;:,")).replace(" x ", r" $\times$ ")
    parts = [clean(doc.title)] + ([clean(doc.notes[0])] if doc.notes else [])
    return ". ".join(p for p in parts if p) + "."


def md_tab(name: str, label: str, path: Path, *, caption: str | None = None, lead: str = "",
           group: str | tuple | None = None, group_style: str = "title", group_title=None,
           heads: dict | None = None, tops: list | None = None, drop: tuple = (), keep_row=None) -> Tab:
    """A supplementary table from any CSV of runs/_tables/<m>/ and its report <stem>.md: the columns, heads, rounding
    and rows of the report's tables (tables with the same row keys joined; their titles span their columns),
    intervals as \\cellstack candidates (the layout search puts them under their estimates when that is lower).
    caption: as given, else lead + the title and first note of the report; group: CSV key(s) whose blocks become
    group rows (group_style title) or stay in their column, named on the first row of a block (first); heads:
    report head -> LaTeX head; tops: (LaTeX title, first report head, last report head) spanning heads; drop: report
    heads left out; keep_row(csv row): the rows kept."""
    rows = read_csv(path)
    md_path = path.with_suffix(".md")
    doc = read_md(md_path)
    assert doc.tables, md_path
    mcols = md_columns(name, doc, rows)
    keep = [i for i, r in enumerate(rows) if keep_row is None or keep_row(r)]
    sub = [rows[i] for i in keep]
    for c in mcols:
        c.cells = [c.cells[i] for i in keep]
    gkeys = (group,) if isinstance(group, str) else tuple(group or ())
    groups: dict[int, str] = {}
    shown = [c for c in mcols if c.head not in drop
             and not (group_style == "title" and c.kind == "text" and c.keys[0] in gkeys)]
    values = {id(c): md_values(c, sub) for c in shown}
    if gkeys:
        blocks = [tuple(r[k] for k in gkeys) for r in sub]
        default = {"architecture": ARCH, "corridor": CORRIDOR_TITLE}.get(gkeys[0], {})
        title = group_title or (lambda v: default.get(v[0], tex(", ".join(v))))
        for i, b in enumerate(blocks):
            if i == 0 or b != blocks[i - 1]:
                groups[i] = title(b) if group_style == "title" else ("" if i == 0 else "")
        if group_style == "first":
            groups.pop(0, None)
            for c in shown:
                if c.kind == "text" and c.keys[0] in gkeys:
                    values[id(c)] = [v if i == 0 or blocks[i] != blocks[i - 1] else "" for i, v in enumerate(values[id(c)])]
    text_cols = [c for c in shown if c.kind == "text"]
    label_col = text_cols[0] if text_cols else shown[0]
    cols = []
    for c in shown:
        align = "L" if c is label_col else ("l" if c.kind == "text" else "r")
        cols.append(Col((heads or {}).get(c.head) or md_head(c), values[id(c)], align))
    if tops is None and len({c.part for c in shown}) > 1:
        tops = []
        for p in sorted({c.part for c in shown}):
            own = [c for c in shown if c.part == p and c.kind != "text"]
            if own:
                tops.append((tex(doc.tables[p].title), own[0].head, own[-1].head))
    index = {}
    for j, c in enumerate(shown):
        index.setdefault(c.head, j)
    top = [(text, index[a], index[b]) for text, a, b in (tops or [])]
    has_ci = any(c.kind in ("ci", "iv") for c in shown)
    if caption is None:
        caption = lead + md_caption(doc) + (" " + CI_NOTE if has_ci else "")
    mapping = "; ".join(f"{c.head} = {'/'.join(c.keys)}" for c in shown)
    return Tab(name, label, caption, cols, groups=groups, top=top,
               note=f"from {path.relative_to(ROOT).as_posix()}, columns and rounding of {md_path.name} ({mapping})")


ERROR_TITLE = {"macro error": "Macro error", "macro error (dynamic)": "Dynamic macro error"}


def s_full_history() -> Tab:
    caption = (
        r"Linearisation of the recurrent laws over their whole window of 30 states at the equilibria of the audit "
        r"(highD): every run of GRU, LSTM and PERL in E1 (no penalty) and E2 (rollout gain penalty, weight 0.1), 5 "
        r"folds $\times$ 5 seeds each, in the combined arm of E2 (5 folds, seed 0) and in the long-window arm (fold 0, "
        r"seed 0), and the GRU and LSTM of the E5 control on the OpenACC views (ACC and human, without and with the "
        r"penalty; 5 folds, seed 0); speeds with an equilibrium in "
        r"the audit (part all) and those in the speed range of the training data (support), summed over the runs. "
        r"Shares of the speeds (unit: run): poles inside, every pole of the windowed two-vehicle loop (semi-implicit "
        r"Euler at 0.1\,s, as in the rollouts) strictly inside the unit circle (local stability); memoryless locally "
        r"stable, $f_s > 0$ and $f_v + f_{\Delta v} < 0$; windowed $> 1.02$, the windowed gain "
        r"$|G_w(e^{i\omega\,\mathrm{d}t})|$ above 1.02 at one of the 25 audit frequencies; numerical unstable, the "
        r"verdict of the audit; agreement, the same windowed and numerical verdict (rollouts that broke down left "
        r"out); low-frequency limit, $M < 0$ (memoryless view) or $M_w = M + 2(f_v m_s - f_s m_v) - \mathrm{d}t\, "
        r"f_s f_v < 0$ ($m_x$: first lag moments of the history Jacobian) exactly when $|G_w| > 1$ at 0.02\,rad/s, "
        r"and the same sign of $M$ and $M_w$. $|\mathrm{num}-\mathrm{win}|$: median (90\% quantile), over the speeds "
        r"of all runs with every pole inside and an unflagged rollout, of the difference between the numerical and "
        r"the windowed gain at 0.02\,rad/s. " + CI_NOTE)
    return md_tab(
        "S32_full_history", "tab:Sfull_history", M9 / "full_history.csv", caption=caption, group="architecture",
        heads={"memoryless locally stable": r"memoryless\\locally stable", "windowed > 1.02": r"windowed\\$> 1.02$",
               "numerical unstable": r"numerical\\unstable", "agreement windowed vs numerical": "agreement",
               "|num - win| at 0.02": r"$|\mathrm{num}-\mathrm{win}|$",
               "sign(M) vs |G_w(0.02)|": r"$M$", "sign(M_w) vs |G_w(0.02)|": r"$M_w$",
               "sign(M) = sign(M_w)": r"same sign"},
        tops=[("poles and gain verdicts", "poles inside", "|num - win| at 0.02"),
              ("low-frequency limit", "sign(M) vs |G_w(0.02)|", "sign(M) = sign(M_w)")])


def s_threshold_sensitivity() -> Tab:
    caption = (
        r"Sensitivity of the audit to the threshold of its numerical rule: every E1 run on highD (learned models: 5 "
        r"folds $\times$ 5 seeds; baselines: 5 folds, seed 0) re-evaluated from the stored largest gain of every "
        r"equilibrium over the 25 audit frequencies, without new rollouts. Unstable: largest gain above the threshold "
        r"(a strict inequality, as in the audit, whose threshold is 1.02); a speed without a defined largest gain has "
        r"no verdict. Unstable among equilibria: the statistic of H1.1 (string-unstable equilibria among the "
        r"equilibria the audit analysed, speeds in support) and its change against the threshold 1.02, paired by "
        r"run; band unstable, band not stable (1 $-$ stable), stable, outside, none: shares of the grid speeds in "
        r"support (outside and none do not depend on the threshold); flagged: equilibria in support above the "
        r"threshold, summed over the runs; share rule of H1.1: share $\geq 0.5$ with the lower end $> 0.3$ (the "
        r"verdict of H1.1 also needs its RMSE part). Unit: run. " + CI_NOTE)
    return md_tab(
        "S33_threshold_sensitivity", "tab:Sthreshold_sensitivity", M9 / "threshold_sensitivity.csv",
        caption=caption, group="architecture", group_style="first",
        heads={"H1.1: unstable among equilibria": "unstable among equilibria (H1.1)",
               "H1.1 share rule": "share rule of H1.1"})


def s_equilibrium_roots() -> Tab:
    caption = (
        r"Zero crossings of the equilibrium condition $f(s, 0, v) = 0$ of every E1 run of the learned models on highD "
        r"(5 folds $\times$ 5 seeds), evaluated as the equilibrium search evaluates the model (memoryless view, the "
        r"window filled with the constant state), at every grid speed 5--30\,m/s (part all) or at the speeds in the "
        r"speed range of the training data (support; speeds summed over the runs), on 400 log-spaced gaps over "
        r"$[1, 200]$\,m (neighbours 1.34\% apart; two roots between neighbouring gaps are not seen). Up 0, up 1, up "
        r"2+: shares of the speeds with no, one, or two or more upward crossings ($f < 0$ at a gap and $f \geq 0$ at "
        r"the next, the convention of the search); any down: at least one downward crossing ($f_s < 0$ there, an "
        r"equilibrium that a gap disturbance moves away from); down below: a downward crossing below the first upward "
        r"one, or without any; down above: one above the first upward crossing. Audit equilibria: speeds with an "
        r"equilibrium in the audit, summed over the runs; unique: among them, the share at which the scan has exactly "
        r"one upward crossing and its bracket holds the spacing of the audit; not bracketed: the spacing of the audit "
        r"lies in no upward bracket of the scan; audit none, scan up: among the speeds without an equilibrium in the "
        r"audit, those at which the scan finds an upward crossing (---: no such speed). Unit: run. " + CI_NOTE)
    return md_tab("S34_equilibrium_roots", "tab:Sequilibrium_roots", M9 / "equilibrium_roots.csv", caption=caption,
                  group="architecture", group_style="first")


def s_correlation_clustered() -> Tab:
    caption = (
        r"Correlation over the laws between the mean macro error of a law and the share of unstable equilibria among "
        r"the equilibria of its members (Spearman; pooled: ranks within every corridor, the corridor as a stratum), "
        r"with intervals that treat the law--corridor pairs as clustered: laws, a bootstrap of the laws as clusters "
        r"(the rows of a law on both corridors move together); families, a bootstrap of the architecture families "
        r"(IDM: idm\_global, idm\_heterogeneous, idm\_heterogeneous\_all; ResidualIDM: residual\_idm, "
        r"residual\_idm\_certified with its amplitudes \_r0.1, \_r0.2, \_r0.5 and \_het, residual\_idm\_free\_r0.3; "
        r"$k$-NN: knn; MLP: mlp, mlp\_penalty; PIDL: pidl; GRU: gru, gru\_penalty; LSTM: lstm, lstm\_penalty; PERL: "
        r"perl); pairs, the rows resampled within every corridor as in Table~\ref{tab:Scorrelation_pooled}. 95\% "
        r"percentile intervals from 5000 resamples each (resamples with a constant variable have no correlation; per "
        r"corridor the law clusters are the laws themselves); $p$: two-sided permutation test (10\,000 permutations) "
        r"of the instability over the laws as wholes (a law takes another law's value on every corridor it runs on). "
        r"Subset: all laws with runs; without a family: its rows left out, ranks recomputed; learned laws only: "
        r"without the families IDM and ResidualIDM. $n$: law--corridor pairs; clusters: laws and families in the "
        r"subset. Rule of H12.3, for information (the verdicts stay as reported): $r > 0.6$, $p < 0.05$ and "
        r"$n \geq 10$ confirmed, $r < 0.3$ refuted, otherwise open. " + LAWS_REF)
    return md_tab(
        "S35_correlation_clustered", "tab:Scorrelation_clustered", M8 / "correlation_clustered.csv", caption=caption,
        group=("error", "scope"), group_title=lambda v: f"{ERROR_TITLE[v[0]]}, {tex(v[1])}",
        heads={"laws": "subset", "laws (clusters)": "laws", "families (clusters)": "families",
               "law clusters": "laws", "family clusters": "families", "pairs independent (D122)": "pairs",
               "p (laws permuted)": "$p$"},
        tops=[("clusters", "laws (clusters)", "families (clusters)"),
              ("interval, bootstrap of", "law clusters", "pairs independent (D122)")])


def s_contacts_absolute() -> Tab:
    path = M8 / "contacts_absolute.csv"
    rows = read_csv(path)

    def has_runs(r: dict) -> bool:
        return r["law"] == "ground truth" or (fnum(r["runs"]) or 0) > 0

    left_out = sorted({r["law"] for r in rows if not has_runs(r)})
    shown = {r["law"] for r in rows if has_runs(r)}
    new_laws = [x for x in ("idm_core_margin", "idm_margin_i80", "residual_idm_margin_free_r0.3") if x in shown]
    names = LAWS_REF if not new_laws else (r"Law names as in Tables~\ref{tab:Slaws} and "
                                           r"\ref{tab:Sfactorial_ablation}.")
    out = (" Laws without runs yet are left out (" + ", ".join(ident(x) for x in left_out)
           + r"; Table~\ref{tab:Sfactorial_ablation}).") if left_out else ""
    caption = (
        r"Contact episodes with their exposure and the demand shortfall per corridor and law (unit: run, i.e., "
        r"scenario and seed; mean over the runs). Contact episodes: episodes of followers with a gap of zero or less "
        r"that begin inside the analysis window (180--840\,s on I-80, 90--870\,s on US-101); runs with contacts: runs "
        r"with at least one; vehicles in contact: vehicles of the window with an episode that begins inside it, per "
        r"run and as a share of the vehicles of the window; veh-km: vehicle-km driven inside the section and the "
        r"window (the denominator of the rate), per run and relative to the ground truth of the run's scenario; "
        r"episodes per 1000 veh-km: the mean of the rates of the runs (the rate of Table~\ref{tab:Slaws}) and the "
        r"pooled rate, 1000 times the sum of the episodes over the sum of the veh-km of the runs; inserted share: of "
        r"the vehicles planned to depart inside the window, the share inserted at all; shortfall: vehicles planned "
        r"inside the window, or in the whole run, and never inserted; insertion delay: mean departure delay of the "
        r"inserted vehicles. Ground truth: mean over the scenarios of the corridor." + out + " " + names + " "
        + CI_NOTE)
    return md_tab(
        "S36_contacts_absolute", "tab:Scontacts_absolute", path, caption=caption, group="corridor", keep_row=has_runs,
        heads={"contact episodes per run": "contact episodes per run", "vehicles in contact per run": "per run",
               "share of the vehicles": "share", "vehicles in the window": "vehicles in the window",
               "veh-km per run": "per run", "veh-km / ground truth": "/ ground truth",
               "episodes / 1000 veh-km (mean of runs)": "mean of runs",
               "episodes / 1000 veh-km (pooled)": "pooled", "inserted share (window)": "inserted share",
               "shortfall (window)": "window", "shortfall (run)": "run", "insertion delay (s)": "insertion delay (s)"},
        tops=[("vehicles in contact", "vehicles in contact per run", "share of the vehicles"),
              ("veh-km", "veh-km per run", "veh-km / ground truth"),
              ("episodes per 1000 veh-km", "episodes / 1000 veh-km (mean of runs)", "episodes / 1000 veh-km (pooled)"),
              ("shortfall", "shortfall (window)", "shortfall (run)")])


def s_h12_2_error() -> Tab:
    caption = (
        r"Exploratory comparison of residual\_idm\_certified with idm\_global on their errors against the ground "
        r"truth, per corridor and paired by scenario and seed (pairs); not a verdict: the pre-specified H12.2 used "
        r"the two one-sided tests of the raw metrics (Table~\ref{tab:Stost}) and stays as reported. Per component of "
        r"the macro-error vector (signed relative errors against the ground truth of the run's scenario; the macro "
        r"triple first: throughput, travel time (W1) and wave speed (xcorr)) and for the macro error and the dynamic "
        r"macro error (kind summary): the mean absolute errors $|e|$ of the two laws; difference of $|e|$: "
        r"$|e_{\mathrm{certified}}| - |e_{\mathrm{idm\_global}}|$ per pair, mean with its interval (negative: the "
        r"certified hybrid closer to the data); relative: ratio of the mean absolute errors minus 1, with its "
        r"interval; $p$ (Wilcoxon): two-sided signed-rank test of the differences; $p$ (Holm): Holm's correction "
        r"over the eight components of a corridor; certified hybrid: smaller or larger error when the interval of the "
        r"difference excludes 0, else no difference. " + LAWS_REF + " " + CI_NOTE)
    return md_tab(
        "S37_h12_2_error", "tab:Sh12_2_error", M8 / "h12_2_error.csv", caption=caption, group="corridor",
        heads={"|e| idm_global": r"$|e|$, idm\_global", "|e| residual_idm_certified": r"$|e|$, certified",
               "difference of |e|": r"difference of $|e|$", "residual_idm_certified": "certified hybrid"})


ABLATION_LAWS = [  # (law, variant, description, highD experiment, NGSIM I-80 experiment) of the factorial ablation
    ("idm_global", "A", "calibrated IDM", None, None),
    ("idm_core_margin", "B", "margin core of the hybrid, no residual", None, None),
    ("idm_margin_i80", "B$'$", "margin IDM of I-80, no residual", None, None),
    ("residual_idm_free_r0.3", "C", "free core + bounded residual", "e4_free_r0.3", "e4_free_r0.3_ft"),
    ("residual_idm_margin_free_r0.3", "D", "margin core + residual without derivative budget", "e4_margin_free_r0.3",
     "e4_margin_free_r0.3_ft"),
    ("residual_idm_certified", "E", "certified hybrid", "e4_stable", "e4_stable_ft"),
]


def s_factorial_ablation() -> Tab:
    """Factorial ablation of the certified hybrid: e4_rmax.csv (and e4.csv) for the training stages, laws.csv of m5
    for the corridor, instability.csv for the members; a variant whose law has no runs yet gets --- (runs in
    progress)."""
    rmax = read_csv(M5 / "e4_rmax.csv")
    e4 = read_csv(SRC / "e4.csv")
    laws = {(r["corridor"], r["law"]): r for r in read_csv(M5 / "laws.csv")}
    inst = {(r["corridor"], r["law"]): r for r in read_csv(SRC / "instability.csv")}
    e1 = {r["model"]: r for r in read_csv(SRC / "e1.csv")}
    ref = [r for r in read_csv(SRC / "h12_2.csv") if (r["experiment"], r["model"]) == ("e3_reference", "idm")]
    assert len(ref) == 1 and fnum(ref[0]["runs"]) == 5 and fnum(e1["idm"]["runs"]) == 5
    rows, pending, no_traj = [], [], []
    for law, var, desc, exp_h, exp_n in ABLATION_LAWS:
        corr = [laws.get((c, law)) for c in ("I-80", "US-101")]
        corr = [r if r is not None and (fnum(r["runs"]) or 0) > 0 else None for r in corr]
        for r in corr:
            assert r is None or fnum(r["runs"]) == 30, law
        if all(r is None for r in corr):
            pending.append(var)
        # training stages: the row of the sweep (law), else the rows of e4.csv (experiment)
        tr = [r for r in rmax if r["law"] == law and r["corridor"] == "I-80"]
        assert len(tr) <= 1, law
        stage = None
        if tr:
            r = tr[0]
            stage = {k: fnum(r[k]) for k in ("runs_highd", "a_priori_highd", "rmse_highd", "runs_ngsim",
                                             "a_priori_ngsim", "unstable_eq_ngsim", "rmse_ngsim")}
            for c, cr in zip(("I-80", "US-101"), corr):
                er = [x for x in rmax if x["law"] == law and x["corridor"] == c]
                if cr is not None and er:
                    assert same(er[0]["macro_error"], cr["macro_error"]), (law, c)
        elif exp_h:
            h = [r for r in e4 if r["experiment"] == exp_h and r["model"] == "residual_idm"]
            n = [r for r in e4 if r["experiment"] == exp_n and r["model"] == "residual_idm"]
            if h and n:
                stage = {"runs_highd": fnum(h[0]["runs"]), "a_priori_highd": fnum(h[0]["a_priori_holds"]),
                         "rmse_highd": fnum(h[0]["rmse_s"]), "runs_ngsim": fnum(n[0]["runs"]),
                         "a_priori_ngsim": fnum(n[0]["a_priori_holds"]), "unstable_eq_ngsim": fnum(n[0]["unstable_eq"]),
                         "rmse_ngsim": fnum(n[0]["rmse_s"])}
        if law == "residual_idm_certified":  # the sweep row equals E4
            h = [r for r in e4 if r["experiment"] == "e4_stable" and r["model"] == "residual_idm"][0]
            assert stage and abs(stage["rmse_highd"] - fnum(h["rmse_s"])) < 1e-9
        # certificate
        if exp_h is None:
            cert = "n.a."
        elif stage and stage["a_priori_highd"] is not None and stage["a_priori_ngsim"] is not None:
            hold = (stage["a_priori_highd"], stage["a_priori_ngsim"])
            runs = (stage["runs_highd"], stage["runs_ngsim"])
            cert = ("yes" if hold == runs else "no" if hold == (0, 0)
                    else f"{f_int(hold[0])}/{f_int(runs[0])}, {f_int(hold[1])}/{f_int(runs[1])}")
        else:
            cert = "no" if law == "residual_idm_margin_free_r0.3" else "---"
        # instability of the members of the corridor law
        if ("I-80", law) in inst:
            unst = f_num(fnum(inst[("I-80", law)]["unstable_eq"]), 2)
            if stage and stage["unstable_eq_ngsim"] is not None:
                assert abs(stage["unstable_eq_ngsim"] - fnum(inst[("I-80", law)]["unstable_eq"])) < 1e-9, law
        elif stage and stage["unstable_eq_ngsim"] is not None:
            unst = f_num(stage["unstable_eq_ngsim"], 2)
        else:
            unst = "---"
        # spacing RMSE of the training stages (A: the IDM of E1 on highD and the members of idm_global on I-80)
        if law == "idm_global":
            rm_h, rm_n = f_num(fnum(e1["idm"]["rmse_s"]), 2), f_num(fnum(ref[0]["rmse_s"]), 2)
            var_cell = var + r"\textsuperscript{a}"
        elif stage:
            rm_h, rm_n = f_num(stage["rmse_highd"], 2), f_num(stage["rmse_ngsim"], 2)
            var_cell = var
        else:
            rm_h = rm_n = "---"
            var_cell = var
            if exp_h is None:
                no_traj.append(var)
        me = [num_ci(r, "macro_error", 3) if r else "---" for r in corr]
        coll = ("---" if all(r is None for r in corr) else
                " / ".join(f_coll(fnum(r["collisions_per_1000_vkm"])) if r else "---" for r in corr))
        rows.append((f"{var_cell}: {desc} ({ident(law, True)})", cert, unst, rm_h, rm_n, me[0], me[1], coll))
    heads = ["variant", "certificate", r"unstable\\among\\equilibria", "highD", "NGSIM I-80", "I-80", "US-101",
             r"collisions per\\1000 veh-km,\\I-80 / US-101"]
    cols = [Col(heads[0], [r[0] for r in rows], "L"), Col(heads[1], [r[1] for r in rows], "l")]
    for j in range(2, 8):
        cols.append(Col(heads[j], [r[j] for r in rows]))
    top = [("spacing RMSE (m)", 3, 4), ("macro error", 5, 6)]
    def listed(xs: list[str]) -> str:
        return ", ".join(xs[:-1]) + (" and " if len(xs) > 1 else "") + xs[-1]

    notes = []
    if pending:
        notes.append(f"---: no runs yet (runs in progress for {listed(pending)})")
    if no_traj:
        notes.append(f"{listed(no_traj)}: {'cores' if len(no_traj) > 1 else 'a core'} without trajectory runs "
                     f"(---)")
    pend = (" " + "; ".join(notes) + ".") if notes else ""
    caption = (
        r"Controlled ablation of the components of the certified hybrid (C, D and E share the architecture, the "
        r"scaler, the loss, the optimiser settings, the folds and the seeds and differ in the constraint under study; "
        r"their epochs differ through early stopping; A, B and B$'$ are calibrated cores without a training run): "
        r"A, the IDM calibrated on I-80; B, the margin-constrained IDM core of the certified hybrid alone, without "
        r"the residual (the frozen core of every member, calibrated on highD with the margin $m_c = 0.2$\,s$^{-2}$); "
        r"B$'$, the global IDM of I-80 calibrated with the same margin; C, a free core with the bounded residual "
        r"($r_{\max} = 0.3$\,m/s$^2$); D, the margin core with the same residual but without the derivative budget; "
        r"E, the certified hybrid ($r_{\max} = 0.3$\,m/s$^2$). Certificate: the a priori certificate holds in every "
        r"run on highD and after fine-tuning on NGSIM I-80 (yes), in none or is not part of the variant (no), or "
        r"does not apply (n.a.: an IDM without residual; B and B$'$ have the margin $m_c$ at every grid speed by "
        r"their calibration); unstable among equilibria: of the members of the corridor law (unit: run); spacing "
        r"RMSE of the test parts on highD and after fine-tuning on NGSIM I-80 (m; unit: driver, event on highD; 25 runs per stage); "
        r"macro error and collisions per 1000 vehicle-km of the corridor law on I-80 and US-101 (30 runs per law and "
        r"corridor; unit: run)." + pend + " " + CI_NOTE)
    footer = (r"\textsuperscript{a} Spacing RMSE of the IDM calibrated on highD in E1 (5 runs) and of the members "
              r"of idm\_global on NGSIM I-80 (5 runs).")
    return Tab("S38_factorial_ablation", "tab:Sfactorial_ablation", caption, cols, top=top, footer=footer,
               note="from runs/_tables/m5/e4_rmax.csv (and runs/_report/tables/e4.csv) for the training stages, "
                    "runs/_tables/m5/laws.csv for the corridor, runs/_report/tables/instability.csv, e1.csv and "
                    "h12_2.csv for the IDM")


def s_e2_horizon() -> Tab:
    """The long-window arm of the rollout penalty: the generic table once runs/_tables/m4/e2_horizon.csv exists,
    until then a placeholder that keeps the number S39."""
    lead = (r"Long-window arm of the rollout gain penalty of E2 on highD: GRU and LSTM on fold 0 (seed 0) at the "
            r"chosen weight 0.1 with a rollout of 380\,s whose gain is measured over the last 252\,s, i.e., two "
            r"periods of the lowest penalty frequency (0.05\,rad/s) after a warm-up of one period, instead of the last "
            r"20\,s of a 40\,s rollout. ")
    path = M4 / "e2_horizon.csv"
    if path.exists() and path.with_suffix(".md").exists():
        caption = (lead +
                   r"Per architecture and arm (E1 without penalty; E2 at the chosen weight with the standard rollout; "
                   r"the long window): runs; spacing RMSE of the test part (m; unit: event on highD) and its relative "
                   r"change against E1 of the same fold and seed, paired over the events; unstable, not stable: band "
                   r"shares of the grid speeds in support (unit: run); poles incl.: the same with a speed counted as "
                   r"not stable when a pole of the full-history loop lies outside the unit circle; unstable among "
                   r"equilibria: of the equilibria found; gain above threshold at $\omega \le 0.1$: share of the audited "
                   r"equilibria whose gain exceeds 1.02 at a frequency of at most 0.1\,rad/s, the band a 20-s window "
                   r"cannot resolve; max gain and the frequency at which it occurs; best epoch and training time of the "
                   r"run; collided profiles: OpenACC platoon profiles (of 5) whose platoon collided. One run per row, so "
                   r"the intervals of the shares equal the values. " + CI_NOTE)
        return md_tab("S39_e2_horizon", "tab:Se2_horizon", path, caption=caption, group="architecture",
                      drop=("drivers", "p", "audited", "stable", "outside", "none", "epochs", "profiles", "complete"))
    caption = lead + r"Runs in progress: the table is completed when the runs of the arm are finished."
    cols = [Col("arm", [r"rollout of 380\,s, gain over the last 252\,s (GRU, LSTM; fold 0, seed 0)"], "L"),
            Col("status", ["runs in progress"], "l")]
    return Tab("S39_e2_horizon", "tab:Se2_horizon", caption, cols,
               note="placeholder: runs/_tables/m4/e2_horizon.csv does not exist yet (runs in progress)")


def s_ablation_pairs() -> Tab:
    """Paired differences between the variants of the controlled ablation (review of 8 October 2026): the generic
    table of runs/_tables/m8/ablation_pairs.csv, grouped by corridor."""
    lead = (r"Paired differences between the variants of the controlled ablation of the certified hybrid (A, the IDM "
            r"calibrated on I-80, idm\_global; B, the margin core of the hybrid without residual, idm\_core\_margin; "
            r"B$'$, the I-80 IDM recalibrated with the margin, idm\_margin\_i80; C, a free core with the bounded "
            r"residual, residual\_idm\_free\_r0.3; D, the margin core with the bounded residual and no derivative "
            r"budget, residual\_idm\_margin\_free\_r0.3; E, the certified hybrid, residual\_idm\_certified), per "
            r"corridor and pair of laws, paired by scenario and seed. Exploratory: the 42 comparisons are reported with "
            r"their intervals and unadjusted Wilcoxon $p$-values, without a correction for multiplicity; the verdicts "
            r"of the hypotheses do not depend on them. ")
    caption = (lead +
               r"Pair: candidate $-$ reference by the letters of the variants, with their laws; metric: the macro "
               r"error, the dynamic macro error or the contact episodes per 1\,000 vehicle-km (as in "
               r"Table~\ref{tab:Slaws}); pairs: the runs (scenario, seed) with a value of the metric for both laws; mean "
               r"candidate, mean reference: the means over these pairs; difference: candidate $-$ reference per pair, "
               r"mean with its interval over the pairs (negative: the candidate has the smaller error or fewer "
               r"contacts); relative: mean candidate / mean reference $-$ 1 with its interval, only for a positive "
               r"reference mean; $p$: two-sided Wilcoxon signed-rank test of the differences; outcome: lower or higher "
               r"when the interval of the difference lies below or above 0, else no difference. " + CI_NOTE)
    return md_tab("S40_ablation_pairs", "tab:Sablation_pairs", M8 / "ablation_pairs.csv", caption=caption,
                  group="corridor")


def s_certificate_exact() -> Tab:
    """Exact verification of the a priori certificate (review of 8 October 2026): the generic table of
    runs/_tables/m9/certificate_exact.csv, one row per experiment."""
    lead = (r"Exact verification of the a priori certificate of the residual hybrid (Section 3.4 of the main text): "
            r"the minimum of the guaranteed margin over the feasible gaps $I(v_e)$ computed exactly (the guaranteed "
            r"margin is a piecewise polynomial in $1/s$ whose minimum lies at an interval end, a regime boundary or a "
            r"root of a cubic) against the stored scan of 400 gaps with three refinements, per experiment over its "
            r"runs and the 26 grid speeds; the existence conditions of the Remark on existence (a finite upper end of "
            r"$I(v_e)$, $a[1 - (v_e/v_0)^\delta] > r_{\max}$, and $I(v_e)$ inside the gap range $[1, 200]$\,m); and a "
            r"dense sweep of 2\,501 speeds (5 to 30\,m/s in steps of 0.01\,m/s) with the exact minimum at each. "
            r"Experiments as in Tables S7 and S8: e4\_stable, the certified hybrid on highD, \_ft its fine-tuning on "
            r"I-80, \_r0.1 to \_r0.5 the amplitude sweep, m8\_temporal\_stable\_ft the temporal hold-out; "
            r"e4\_free\_r0.3 and e4\_free\_ft the free cores (controls without certificate), e4\_margin\_free\_r0.3 "
            r"the margin core without the derivative budget (variant D). ")
    caption = (lead +
               r"Per experiment: $r_{\max}$ of the residual (m/s$^2$) and the runs; exact min: the smallest exact "
               r"minimum of the guaranteed margin over the runs and the 26 grid speeds (s$^{-2}$); max $|$exact $-$ "
               r"scan$|$: the largest difference to the stored scan over the (run, speed) cells (s$^{-2}$); holds "
               r"differ: cells where the verdict of the exact minimum differs from the scanned one; runs holding at "
               r"every grid speed: runs whose exact certificate holds at all grid speeds; sweep min and runs holding on "
               r"the sweep: the same over the 2\,501 speeds; min $a[1 - (v_e/v_0)^\delta] - r_{\max}$: the smallest "
               r"margin of the finite upper end over the runs and grid speeds (positive: every $I(v_e)$ bounded); "
               r"$I(v_e)$ within [1, 200]\,m: share of the (run, speed) cells whose interval lies inside the audited "
               r"gap range; min $s_{\mathrm{lo}}$, max $s_{\mathrm{hi}}$: the extreme ends of the intervals (m; blank "
               r"when some interval has no upper end). The margin core without the certificate (variant D) and the "
               r"free cores fail the condition in every run, the free cores also the existence condition.")
    return md_tab("S41_certificate_exact", "tab:Scertificate_exact", M9 / "certificate_exact.csv", caption=caption,
                  drop=("design", "data", "grid speeds", "scan min (s⁻²)", "minimum at the upper end of I(v)",
                        "runs holding at every grid speed, scan", "I(v) outside [1, 200] m"),
                  heads={"I(v) within [1, 200] m": r"$I(v_e)$ within\\{}[1, 200]\,m",
                         "r_max (m/s²)": r"$r_{\max}$\\(m/s$^2$)", "exact min (s⁻²)": r"exact min\\(s$^{-2}$)",
                         "max |exact - scan| (s⁻²)": r"max $|$exact\\$-$ scan$|$ (s$^{-2}$)",
                         "max \\|exact - scan\\| (s⁻²)": r"max $|$exact\\$-$ scan$|$ (s$^{-2}$)",
                         "holds differ": r"holds\\differ", "sweep min (s⁻²)": r"sweep min\\(s$^{-2}$)",
                         "runs holding at every grid speed, exact": r"runs holding\\at every\\grid speed",
                         "runs holding on the whole sweep": r"runs holding\\on the\\sweep",
                         "min a(1-(v/v0)^δ) - r_max (m/s²)": r"min $a[1 - (v_e/v_0)^\delta]$\\$- r_{\max}$ (m/s$^2$)",
                         "min s_low (m)": r"min $s_{\mathrm{lo}}$\\(m)", "max s_high (m)": r"max $s_{\mathrm{hi}}$\\(m)"},
                  keep_row=lambda r: not str(r["experiment"]).startswith(("m4_e4pilot", "m8_smoke")))


SUPP_BUILDERS = [  # binding numbering S1-S41 (the order of the document is SECTIONS)
    (s_e2_sweep, "Sweep of the penalty weight in E2"),
    (s_e2_existence, "Existence arm of E2"),
    (s_e2_lowfreq, "Low-frequency arm of E2 (combined penalty)"),
    (s_e2_monotone, "Monotonicity control of E2"),
    (s_e3, "Transfer of the highD models to NGSIM and Waymo (E3)"),
    (s_e5, "OpenACC views and platoon test (E5)"),
    (s_e4, "Certified hybrid and fine-tuning (E4)"),
    (s_e4_rmax, "Residual-amplitude sweep of the hybrid"),
    (s_certificate_tightness, "Tightness of the certificate"),
    (s_lowfreq_expansion, "Low-frequency expansion of the gain"),
    (s_band_width, "Width of the spacing band and per-driver dispersion"),
    (s_band_sensitivity, "Band sensitivity of the audit"),
    (s_laws, "Corridor laws, all laws"),
    (s_laws_raw, "Raw corridor metrics and ground truth"),
    (s_m5_components, "Components of the macro error"),
    (s_macro_error_dynamic, "Dynamic macro error"),
    (s_instability, "Instability of the members and macro errors"),
    (s_instability_correlation, "Correlation of macro error and instability per corridor"),
    (s_macro_error_dynamic_correlation, "Correlation of dynamic macro error and instability"),
    (s_correlation_pooled, "Correlation over all laws, per corridor and pooled"),
    (s_correlation_pooled_laws, "Values of the pooled correlation per law"),
    (s_tost, "Equivalence tests of the certified hybrid and the IDM"),
    (s_verdicts_h12, "Verdicts of the corridor hypotheses"),
    (s_h12_1, "Degradation of the macro metrics by the networks (H12.1)"),
    (s_h12_2, "Micro comparison with the IDM on NGSIM I-80 (H12.2)"),
    (s_sensitivity, "Sensitivity to the downstream boundary and the lane-change model"),
    (s_power, "Seed-to-seed spread and power of the corridor comparison"),
    (s_temporal, "Temporal hold-out on I-80"),
    (s_asymmetry, "Acceleration asymmetry and oscillation spectrum"),
    (s_asymmetry_contrasts, "Asymmetry of the penalised and certified laws against the free laws"),
    (s_verdicts, "Verdicts of the single-vehicle hypotheses"),
    (s_full_history, "Full-history linearisation of the recurrent laws"),
    (s_threshold_sensitivity, "Threshold of the numerical stability rule"),
    (s_equilibrium_roots, "Zero crossings of the equilibrium condition"),
    (s_correlation_clustered, "Clustered inference of the correlation of macro error and instability"),
    (s_contacts_absolute, "Contact episodes, exposure and demand shortfall"),
    (s_h12_2_error, "Absolute errors of the certified hybrid and the IDM (H12.2, exploratory)"),
    (s_factorial_ablation, "Controlled ablation of the certified hybrid"),
    (s_e2_horizon, "Long-window arm of the rollout penalty"),
    (s_ablation_pairs, "Paired differences between the variants of the ablation"),
    (s_certificate_exact, "Exact verification of the a priori certificate"),
]


def build_supplement(out: Path) -> list[Tab]:
    folder = out / OUT_SUPP
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("S*.tex"):
        old.unlink()
    tabs = []
    for k, (b, _title) in enumerate(SUPP_BUILDERS, start=1):
        READ.clear()
        t = b()
        t.sources = list(READ)
        assert t.name.startswith(f"S{k:02d}_"), (k, t.name)
        t.contexts = ("text", "wide", "landscape")
        t.long_ok = not t.panels
        for p in t.panels:
            p.contexts = t.contexts
        choose(t, supplement=True)
        (folder / f"{t.name}.tex").write_text(emit(t, supplement=True), encoding="utf-8")
        tabs.append(t)
    return tabs


# ================================================================================================
# figures
MAIN_FIGURES = [("rmse_vs_instability", "text"), ("e2_tradeoff", "text"), ("gain_curves", "wide"),
                ("growth_curves", "wide"), ("corridor_speed_contours", "wide"), ("fundamental_diagrams", "wide"),
                ("macro_error_vs_instability", "text")]
SUPP_FIGURES = ([(f"stability_map_{a}", "text") for a in ("mlp", "pidl", "residual_idm", "gru", "lstm", "perl")]
                + [("lowfreq_expansion", "wide"), ("certificate_tightness", "wide"), ("band_width", "text")]
                + [(f"contours_{s}_p{p}", "wide") for s in ("i80", "us101") for p in range(3)]
                + [(f"fd_{s}_p{p}", "wide") for s in ("i80", "us101") for p in range(3)]
                + [("full_history_gain", "wide")])
# figures whose caption is edited from the caption file of the figure (runs/_report/supplement/figures/<name>.txt)
# rather than from version 1; the numbers the caption copies must still be in that file
FIGURE_TXT_NUMBERS = {
    "full_history_gain": ["900 of 3717", "252 numerical gains above", "54 below", "GRU 8e-05 / 9e-04",
                          "LSTM 1e-04 / 2e-03", "PERL 4e-05 / 1e-03", "21 of these 2815 points",
                          "0.07 or more", "median of 1e-03", "A / ω = 10 m",
                          "GRU 0.96 / 0.74, LSTM 0.90 / 0.98, PERL 0.86 / 0.12"]}
STAB = {"mlp": ("MLP", "Jacobian", "1", "stable 32, unstable 20"),
        "pidl": ("PIDL", "Jacobian", "10", "stable 33, unstable 19"),
        "residual_idm": ("ResidualIDM", "Jacobian", "0.1", "stable 44, unstable 6, none 2"),
        "gru": ("GRU", "rollout gain", "0.1", "unstable 33, outside 15, none 4"),
        "lstm": ("LSTM", "rollout gain", "0.1", "stable 19, unstable 18, outside 10, none 5"),
        "perl": ("PERL", "rollout gain", "0.1", "unstable 32, outside 18, none 2")}
SITES = {"i80": ("I-80", "1--6", "180--840\\,s", "20--500\\,m", 19),
         "us101": ("US-101", "1--5", "90--870\\,s", "25--640\\,m", 18)}
VMAX = {"i80_p0": 14, "i80_p1": 12, "i80_p2": 12, "us101_p0": 20, "us101_p1": 16, "us101_p2": 16}


def figure_captions() -> dict[str, str]:
    """The captions of version 1 (paper/figures/*_env.tex); references to tables of the Supplementary Materials
    point to their S labels there."""
    cap = {
        "rmse_vs_instability": (
            r"Audit of the unconstrained models (E1, highD). Left: closed-loop spacing RMSE of the test events against "
            r"the share of string-unstable equilibria among the equilibria found (mean over the runs with 95\% "
            r"bootstrap intervals; open square: the IDM reference). Right: shares of the grid speeds in the speed "
            r"support whose equilibrium is stable or unstable inside the spacing band of the data, lies outside the "
            r"band, or does not exist; hatched: laws without an isolated equilibrium (indifferent: persistence and "
            r"Newell's model)."),
        "e2_tradeoff": (
            r"Sweep of the penalty weight in E2 (highD, 5 folds, seed 0): test spacing RMSE against the share of the "
            r"grid speeds that are stable inside the spacing band, per architecture (Jacobian penalty for MLP, PIDL and "
            r"ResidualIDM, rollout gain penalty for GRU, LSTM and PERL). Points are labelled with the penalty weight; "
            r"circled: the chosen weight (the smallest validation RMSE among the weights with a stable share of at "
            r"least 0.9, otherwise the largest stable share); square: E1 without penalty."),
        "gain_curves": (
            r"Gain $|G(\omega)|$ of the frequency response of the fold-0 runs (highD, seed 0) without penalty (E1) and "
            r"with the chosen E2 weight: thin lines, one speed in the support; thick lines, median over the speeds; "
            r"dashed: the threshold $|G| = 1.02$ of the numerical stability rule."),
        "growth_curves": (
            r"Platoon test (E5): standard deviation of the speed per platoon position for the data (OpenACC, dots) and "
            r"for the laws of the matching driving mode (IDM, MLP, GRU and LSTM, solid; penalised MLP, GRU and LSTM, "
            r"dashed), mean over the folds, on five OpenACC profiles (Vicolungo and ZalaZone test tracks; human or ACC "
            r"driving) and behind a braking pulse (laws of the human view); logarithmic vertical axis. A model curve "
            r"that ends early ends in a cross: the platoon of a fold collided at the next position, where the mean over "
            r"the folds is undefined; 130 of the 210 simulated platoons collided."),
        "corridor_speed_contours": (
            r"Speed fields of period 1 of I-80 (scenario i80\_p1, seed 0): space--time speed of 20\,m $\times$ 2\,s "
            r"cells over all main lanes by Edie's definitions~\cite{edie1963}, for the ground truth and five laws "
            r"(idm\_global, residual\_idm\_certified, mlp, gru, lstm; names as in Table~\ref{tab:laws}); grey: cells "
            r"without a vehicle; dashed: the analysis window (180--840\,s)."),
        "fundamental_diagrams": (
            r"Fundamental diagrams of period 1 of I-80 (scenario i80\_p1, seed 0): flow against density of the Edie "
            r"cells of 100\,m $\times$ 30\,s over all lanes (section 20--500\,m, analysis window 180--840\,s), for the "
            r"ground truth and five laws (names as in Table~\ref{tab:laws}); black line on every panel: mean flow of "
            r"the ground truth per density bin of 10\,veh/km."),
        "macro_error_vs_instability": (
            r"Macro error (left) and dynamic macro error (right) of the corridor laws on I-80 against the share of "
            r"unstable equilibria among the equilibria of their members (mean over the runs with 95\% bootstrap "
            r"intervals); filled circles: laws without collisions; open squares: laws with collisions; above each "
            r"panel: Spearman's and Pearson's coefficients with their bootstrap intervals. Law names as in "
            r"Table~\ref{tab:laws}."),
        "lowfreq_expansion": (
            r"Low-frequency expansion of the gain: fold-0 runs (highD, seed 0) of E1 and of the chosen E2 weight (IDM: "
            r"E1 only). At every grid speed with an equilibrium, the numerical gain $|G|$ of the audit at the lowest "
            r"audit frequency, 0.02\,rad/s (vertical axis), against the expansion $|G| = \sqrt{\max(0, 1 - \omega^2 M "
            r"/ f_s^2)}$ with $M = f_v^2 + 2 f_v f_{\Delta v} - 2 f_s$ from the stored partial derivatives (memoryless "
            r"view for GRU, LSTM and PERL; horizontal axis); open markers: equilibria that are not locally stable in "
            r"the memoryless view (no stationary response). Solid line: numerical gain = expansion; dotted: $|G| = 1$ "
            r"(points in the upper-left and lower-right quadrants differ in the sign of $|G| - 1$). Speeds whose rollout "
            r"at that frequency was clipped, stopped or collided are left out (16 of 324). The numbers at the three "
            r"lowest frequencies are in Table~\ref{tab:Slowfreq_expansion}."),
        "certificate_tightness": (
            r"Tightness of the certificate of E4: per grid speed of every run of the certified hybrid (ResidualIDM with "
            r"the margin-constrained IDM core and the certified Lipschitz budget; 5 folds $\times$ 5 seeds per residual "
            r"amplitude, on highD before fine-tuning and on NGSIM I-80 after) the audited margin $M$ of the hybrid "
            r"(vertical axis) against the certified lower bound of the margin (horizontal axis). Columns: residual "
            r"amplitude $r_{\max}$; top row: the a priori bound (every speed with an equilibrium); bottom row: the bound "
            r"at the anchored equilibria. Line: margin = bound; a point below it would violate the certificate (none of "
            r"the 10\,068 points does). Slack per amplitude, stage and bound: Table~\ref{tab:Scertificate_tightness}."),
        "band_width": (
            r"The price of a common equilibrium: per grid speed, the relative width $(s_{95} - s_5)/s_{50}$ of the "
            r"spacing band of the data (quantiles of the near-steady samples with $|\Delta v| < 0.5$\,m/s, $|a| < "
            r"0.3$\,m/s$^2$ and a speed within 0.5\,m/s of the grid speed; all events of the set, speeds with at least "
            r"200 samples) and the standard deviation over the drivers (with at least 20 near-steady samples at that "
            r"speed) of their median near-steady spacing, divided by $s_{50}$ (open squares: fewer than 10 drivers). "
            r"Left: highD (panel follownet\_highd; the events carry no driver identifiers, so every 15\,s event is its "
            r"own driver); right: NGSIM I-80 (panel ngsim\_i80). Numbers: Table~\ref{tab:Sband_width}."),
        "full_history_gain": (
            r"Full-history linearisation of the recurrent laws against the audit: every grid speed with an equilibrium "
            r"of every E1 run (no penalty) and E2 run (rollout gain penalty at the chosen weight 0.1) of GRU, LSTM and "
            r"PERL on highD (5 folds $\times$ 5 seeds): the gain of the follower speed at 0.02\,rad/s, the lowest audit "
            r"frequency, from the rollout of the audit (vertical axis) against the gain $|G_w(e^{i\omega\,\mathrm{d}t})|$ "
            r"of the linearised two-vehicle loop with the whole window of 30 states (horizontal axis; history Jacobian "
            r"of the network, semi-implicit Euler at 0.1\,s). Filled markers: every closed-loop pole inside the unit "
            r"circle; open markers: a pole on or outside it (the loop is not locally stable, the rollout grows instead "
            r"of settling and the two gains need not agree; 900 of 3717 points). Axes per panel over the range of its "
            r"windowed gains; 252 numerical gains above the upper limit and 54 below the lower one are drawn on the "
            r"edge as triangles. Solid line: numerical = windowed; dotted: $|G| = 1$. Where every pole is inside and "
            r"the rollout is not clipped, stopped or collided, the absolute difference between the numerical and the "
            r"windowed gain has the median and 90\% quantile (E1) $8\times 10^{-5}$ and $9\times 10^{-4}$ (GRU), "
            r"$1\times 10^{-4}$ and $2\times 10^{-3}$ (LSTM), $4\times 10^{-5}$ and $1\times 10^{-3}$ (PERL); 21 of these "
            r"2815 points (E1 and E2) differ by more than 0.05, and their rollouts are not sinusoidal (fit residual of "
            r"the audit 0.07 or more, against a median of $10^{-3}$ elsewhere): a finite-amplitude response, the gap "
            r"moving by $|1 - G|\,A/\omega$ with $A/\omega = 10$\,m at $A = 0.2$\,m/s and 0.02\,rad/s. Share of the "
            r"speeds with every pole inside, E1 / E2 (mean over the runs): GRU 0.96 / 0.74, LSTM 0.90 / 0.98, PERL "
            r"0.86 / 0.12. The combined arm and the intervals: Table~\ref{tab:Sfull_history}."),
    }
    for name, numbers in FIGURE_TXT_NUMBERS.items():
        txt = (REPORT / "supplement" / "figures" / f"{name}.txt").read_text(encoding="utf-8")
        missing = [x for x in numbers if x not in txt]
        assert not missing, f"{name}.txt changed, update the caption of the figure: {missing}"
    for arch, (name, pen, w, counts) in STAB.items():
        cap[f"stability_map_{arch}"] = (
            rf"Equilibria of the {name} of fold 0 (highD, seed 0) on the speed grid, without penalty (E1, left) and "
            rf"with the {pen} penalty at the chosen weight {w} (E2, right). Grey band: spacing band of the training "
            r"data of the fold (5\%--95\% quantiles of the near-steady samples, linear between the grid speeds); "
            r"dashed: its median; points: the equilibrium of every grid speed (searched inside the band where the "
            r"speed has one), coloured by the status of the audit with the numerical rule (gain above 1.02): stable or "
            r"unstable inside the band, outside the band (the first equilibrium in 1--200\,m), or none (no "
            r"equilibrium; triangle at the top of the panel); open markers: speeds without a band; shaded: speeds "
            r"outside the 1\%--99\% speed range of the training data. Logarithmic spacing axis. Speeds by status over "
            r"both panels: " + counts + ".")
    for site, (sname, lanes, window, section, n) in SITES.items():
        for p in range(3):
            sc = f"{site}_p{p}"
            note = ""
            if site == "i80":
                note = (r" The temporal hold-out laws (suffix \_p0) have no run on period 0." if p == 0 else
                        r" The four temporal hold-out laws (suffix \_p0) are left out of the grid; their numbers are in "
                        r"Table~\ref{tab:Stemporal}.")
            sce = sc.replace("_", r"\_")
            cap[f"contours_{sc}"] = (
                rf"Speed fields of {sname}, period {p} (scenario {sce}): space--time speed of 20\,m $\times$ 2\,s "
                rf"cells by Edie's definitions over the main lanes {lanes} together, one colour scale "
                rf"0--{VMAX[sc]}\,m/s (the 99\% quantile of the ground truth); grey: cells without a vehicle; dashed: "
                rf"the analysis window ({window}). Panels: the ground truth (first) and every law of "
                rf"Table~\ref{{tab:Slaws}} with a run of seed 0 ({n} laws); the variants of the controlled ablation "
                rf"(Table~\ref{{tab:Sfactorial_ablation}}) are not drawn." + note)
            cap[f"fd_{sc}"] = (
                rf"Fundamental diagrams of {sname}, period {p} (scenario {sce}): flow against density of the Edie cells "
                rf"of 100\,m $\times$ 30\,s over all lanes (section {section}, analysis window {window}); black line on "
                rf"every panel: mean flow of the ground truth per density bin of 10\,veh/km. Panels: the ground truth "
                rf"(first) and every law of Table~\ref{{tab:Slaws}} with a run of seed 0 ({n} laws); the variants of "
                rf"the controlled ablation (Table~\ref{{tab:Sfactorial_ablation}}) are not drawn." + note)
    return cap


SUPP_FIGURE_TITLES = (
    [f"Equilibria of the {STAB[a][0]} without and with penalty" for a in ("mlp", "pidl", "residual_idm", "gru", "lstm",
                                                                          "perl")]
    + ["Low-frequency expansion of the gain against the numerical gain", "Tightness of the certificate",
       "Width of the spacing band and per-driver dispersion"]
    + [f"Speed fields of {SITES[s][0]}, period {p}" for s in ("i80", "us101") for p in range(3)]
    + [f"Fundamental diagrams of {SITES[s][0]}, period {p}" for s in ("i80", "us101") for p in range(3)]
    + ["Full-history linearisation against the numerical gain"])


def figure_env(name: str, ctx: str, path: str, caption: str, label: str, src: str) -> str:
    origin = (f"{Path(src).with_suffix('.txt').as_posix()}" if name in FIGURE_TXT_NUMBERS
              else f"paper/figures/{name}_env.tex (v1)")
    lines = [f"% Figure {name}: {src} (copied unchanged); caption edited from {origin}.",
             f"% Written by {GENERATED_BY}; do not edit by hand (rerun the script).",
             # the full-page corridor figures and the figures that follow the long tables of E1, E2 and E5 may
             # float (tbp): placed [H] they leave half pages empty
             rf"\begin{{figure}}[{'tbp' if name in ('corridor_speed_contours', 'fundamental_diagrams', 'rmse_vs_instability', 'gain_curves', 'growth_curves') else 'H'}]"]
    if ctx == "wide":
        lines += [r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering",
                  rf"\includegraphics[width=\fulllength]{{{path}}}", r"\end{adjustwidth}"]
    else:
        lines += [r"\centering", rf"\includegraphics[width=\textwidth]{{{path}}}"]
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}"]
    return "\n".join(lines) + "\n"


def build_figures(out: Path) -> None:
    caps = figure_captions()
    main_dir, supp_dir = out / OUT_FIG, out / OUT_SUPP_FIG
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    for old in list(supp_dir.glob("*_env.tex")):
        old.unlink()
    for name, ctx in MAIN_FIGURES:
        src = REPORT / "figures" / f"{name}.pdf"
        shutil.copy2(src, main_dir / f"{name}.pdf")
        (main_dir / f"{name}_env.tex").write_text(
            figure_env(name, ctx, f"figures/{name}.pdf", caps[name], f"fig:{name}",
                       f"runs/_report/figures/{name}.pdf"), encoding="utf-8")
    for k, (name, ctx) in enumerate(SUPP_FIGURES, start=1):
        src = REPORT / "supplement" / "figures" / f"{name}.pdf"
        shutil.copy2(src, supp_dir / f"{name}.pdf")
        assert r"\cite" not in caps[name], name
        (supp_dir / f"S{k:02d}_{name}_env.tex").write_text(
            figure_env(name, ctx, f"supplement/figures/{name}.pdf", caps[name], f"fig:S{name}",
                       f"runs/_report/supplement/figures/{name}.pdf"), encoding="utf-8")


# ================================================================================================
# the Supplementary Materials
TITLE = (r"String Stability of Learned Car-Following Models: Audit, Differentiable Penalties, a Certified Hybrid and "
         r"Corridor-Level Validation")
AUTHOR_BLOCK = r"""% Author ORCID iDs (the macro names are fixed by the class)
\newcommand{\orcidauthorA}{0000-0003-1739-9831}
\newcommand{\orcidauthorB}{0000-0002-9778-124X}

% Authors
\Author{Mikhail Gorodnichev $^{1,}$*\orcidA{} and Marina Moseva $^{1}$\orcidB{}}

% MDPI internal command: authors, for the metadata of the PDF
\AuthorNames{Mikhail Gorodnichev and Marina Moseva}

% Affiliation
\address[1]{Faculty of Information Technology, Moscow Technical University of Communication and Informatics, Moscow 111024, Russia}

% Corresponding author
\corres{Correspondence: m.g.gorodnichev@mtuci.ru}"""
HELPERS = f"% Helpers of the generated tables ({Path(GENERATED_BY).name}):\n" + r"""% \cellstack[r]{<estimate>\\{}<[interval]>}: two lines in one cell, aligned at the first line;
% \headstack[r]{<line>\\<line>}: a column head on several lines, aligned at the last line.
\newcommand{\cellstack}[2][r]{\begin{tabular}[t]{@{}#1@{}}#2\end{tabular}}
\newcommand{\headstack}[2][c]{\begin{tabular}[b]{@{}#1@{}}#2\end{tabular}}"""
SECTIONS = [  # (heading, sentence, table numbers in the order of the document); the numbers are binding, a table
    # placed out of their order gets its number by \setcounter
    ("Penalties", r"Full tables of the penalty experiment E2: the sweep of the weight, the existence arm, the "
                  r"low-frequency arm, the monotonicity control and the long-window arm of the rollout penalty.",
     [1, 2, 3, 4, 39]),
    ("Transfer, Platoon Test and Certificate",
     r"Full tables of the transfer experiment E3, the OpenACC control E5 and the certified hybrid E4, with the exact "
     r"verification of the a priori certificate.", [5, 6, 7, 8, 9, 41]),
    ("Audit: Gain Expansion and Spacing Band",
     r"The exact gain against the numerical gains of the audit, the width of the spacing band, the band "
     r"sensitivity of the audit, the full-history linearisation of the recurrent laws, the threshold of the "
     r"numerical rule and the zero crossings of the equilibrium condition.", [10, 11, 12, 32, 33, 34]),
    ("Corridor", r"Full tables of the corridor validation: the laws, their raw metrics and components, the "
                 r"instability of their members, the correlations, the equivalence tests, the verdicts of the "
                 r"corridor hypotheses, the sensitivity study, the power analysis, the temporal hold-out, the "
                 r"acceleration asymmetry, the clustered inference of the correlation, the absolute contact exposure "
                 r"and the errors of the certified hybrid and the IDM against the ground truth.",
     list(range(13, 31)) + [35, 36, 37]),
    ("Controlled Ablation of the Certified Hybrid",
     r"The parts of the certified hybrid separated in a controlled ablation of components: the calibrated IDM, the "
     r"margin-constrained IDM cores without residual, a free and a margin core with the bounded residual, and the "
     r"certified hybrid; then the paired differences between the variants.", [38, 40]),
    ("Verdicts of the Single-Vehicle Hypotheses", r"Every unit of the verdicts of H1.1--H1.5.", [31]),
]


def write_supplementary(tabs: list[Tab], out: Path) -> None:
    order = [k for _h, _s, ks in SECTIONS for k in ks]
    assert sorted(order) == list(range(1, len(tabs) + 1)), order
    lines = [
        "%  Supplementary Materials of the manuscript (MDPI Mathematics), version 2. Written by "
        f"{Path(GENERATED_BY).name};",
        "%  rerun it rather than editing this file. Build: pdflatex supplementary (twice); see README.md.",
        f"%  Tables S1-S{len(tabs)} and Figures S1-S{len(SUPP_FIGURES)} carry binding numbers (the class option "
        "supfile gives the",
        "%  S prefix; the main text refers to them by these numbers): in the order of the sections here, a table",
        "%  placed out of that numbering gets its number by \\setcounter{table}.",
        r"\documentclass[mathematics,supfile,submit,moreauthors]{Definitions/mdpi}",
        "",
        "% MDPI internal commands - do not modify",
        r"\firstpage{1}", r"\makeatletter", r"\setcounter{page}{\@firstpage}", r"\makeatother",
        r"\pubvolume{1}", r"\issuenum{1}", r"\articlenumber{0}", r"\pubyear{2026}", r"\copyrightyear{2026}",
        r"\datereceived{ }", r"\daterevised{ }", r"\dateaccepted{ }", r"\datepublished{ }",
        "",
        r"\usepackage{xltabular} % long tables of fixed width (longtable + tabularx)",
        r"\setlength{\LTcapwidth}{\textwidth}",
        HELPERS,
        r"\renewcommand{\thesection}{S\arabic{section}}",
        "",
        r"\Title{" + TITLE + "}",
        "",
        AUTHOR_BLOCK,
        "",
        r"\begin{document}",
        "",
    ]
    landscape = False
    last = 0  # number of the table input last
    for h, s, numbers in SECTIONS:
        # a section whose first table is set on a landscape page starts on that page
        wanted = tabs[numbers[0] - 1].layout["ctx"] == "landscape"
        if landscape and not wanted:
            lines.append(r"\finishlandscape")
            landscape = False
        elif wanted and not landscape:
            lines.append(r"\startlandscape")
            landscape = True
        lines += ["", rf"\section{{{h}}}", "", s, ""]
        for k in numbers:
            t = tabs[k - 1]
            if t.layout["ctx"] == "landscape" and not landscape:
                lines.append(r"\startlandscape")
                landscape = True
            elif t.layout["ctx"] != "landscape" and landscape:
                lines.append(r"\finishlandscape")
                landscape = False
            if k != last + 1:  # out of the numbering: the binding number by the counter
                lines.append(rf"\setcounter{{table}}{{{k - 1}}}% the next table is Table S{k}")
            lines.append(rf"\input{{supplement/tables/{t.name}}}")
            last = k
    if landscape:
        lines.append(r"\finishlandscape")
    lines += ["", r"\section{Supplementary Figures}", "",
              r"Stability maps of the audit (Figures S1--S6), the gain expansion (S7), the certificate (S8), the "
              r"spacing band (S9), the speed fields and fundamental diagrams of every law and period of both "
              r"corridors (S10--S21) and the full-history linearisation of the recurrent laws (S22).", ""]
    for k, (name, _ctx) in enumerate(SUPP_FIGURES, start=1):
        lines.append(rf"\input{{supplement/figures/S{k:02d}_{name}_env}}")
    cites = [f.name for f in sorted((out / OUT_SUPP).glob("*.tex")) + sorted((out / OUT_SUPP_FIG).glob("*_env.tex"))
             if r"\cite" in f.read_text(encoding="utf-8")]
    if cites:  # a caption cites: the references of the main text, as the class prints them
        lines += ["", r"\begin{adjustwidth}{-\extralength}{0cm}", r"\reftitle{References}",
                  r"\externalbibliography{yes}", r"\bibliography{references}", r"\end{adjustwidth}"]
    lines += ["", r"\end{document}", ""]
    (out / "supplementary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_items(out: Path) -> None:
    """The list of the Supplementary Materials for \\supplementary{...} of main.tex."""
    items = [f"Table S{k}: {title}" for k, (_b, title) in enumerate(SUPP_BUILDERS, start=1)]
    items += [f"Figure S{k}: {title}" for k, title in enumerate(SUPP_FIGURE_TITLES, start=1)]
    (out / "supplement" / "items.txt").write_text("; ".join(items) + ".\n", encoding="utf-8")


# ================================================================================================
# the map of the materials of the manuscript (--manifest)
MAIN_FLOATS = [  # the floats of the main text in the order of the manuscript (paper/v2/main.tex and its sections):
    # (kind, label, content, generated name or None, basis of a table written by hand in the manuscript)
    ("table", "tab:positioning", "The closest works and this paper", None, "comparison with the literature (no data)"),
    ("table", "tab:datasets", "Event sets", None,
     "the `describe.json` of every extracted event set (`scripts/describe_events.py`; needs the data of level 3)"),
    ("table", "tab:hypotheses", "Pre-specified hypotheses and their decision rules", None,
     "the hypotheses and thresholds of `docs/study_plan.md`"),
    ("table", "tab:e1", "Audit of the unconstrained models (E1)", "e1", None),
    ("figure", "fig:rmse_vs_instability", "Spacing RMSE against the share of unstable equilibria (E1)",
     "rmse_vs_instability", None),
    ("table", "tab:e5", "OpenACC control and platoon test (E5)", "e5", None),
    ("figure", "fig:growth_curves", "Growth of the speed oscillations along the platoons (E5)", "growth_curves", None),
    ("figure", "fig:e2_tradeoff", "Sweep of the penalty weight (E2)", "e2_tradeoff", None),
    ("table", "tab:e2", "Chosen penalty weight against E1 (E2)", "e2", None),
    ("figure", "fig:gain_curves", "Gain of the frequency response without and with the penalty", "gain_curves", None),
    ("table", "tab:loopholes", "How stability penalties get gamed", None,
     "the loophole table of `docs/theory_notes.md` (training runs of E2 and of its arms)"),
    ("table", "tab:e2_arms", "Arms of E2: existence term, combined penalty, monotonicity terms", "e2_arms", None),
    ("table", "tab:e4", "Certified hybrid against its free counterparts (E4)", "e4", None),
    ("table", "tab:laws", "Corridor laws on I-80 and US-101", "laws", None),
    ("figure", "fig:corridor_speed_contours", "Speed fields of period 1 of I-80", "corridor_speed_contours", None),
    ("figure", "fig:fundamental_diagrams", "Fundamental diagrams of period 1 of I-80", "fundamental_diagrams", None),
    ("figure", "fig:macro_error_vs_instability", "Macro error against the share of unstable equilibria",
     "macro_error_vs_instability", None),
    ("table", "tab:correlation", "Correlation of macro error and instability over the laws", "correlation", None),
    ("table", "tab:sensitivity", "Sensitivity of the macro error to the corridor model", "sensitivity", None),
    ("table", "tab:temporal", "Temporal hold-out on I-80", "temporal", None),
    ("table", "tab:verdicts", "Verdicts of the pre-specified hypotheses", "verdicts", None),
]
STEPS = {  # the commands that write the sources, in the order of the pipeline (README.md, section Reproduce)
    "full_history_audit": "python scripts/analysis/full_history_audit.py",
    "threshold_sensitivity": "python scripts/analysis/threshold_sensitivity.py",
    "equilibrium_roots": "python scripts/analysis/equilibrium_roots.py",
    "certificate_exact": "python scripts/analysis/certificate_exact.py",
    "band_sensitivity": "python scripts/analysis/band_sensitivity.py",
    "tables": "python scripts/make_tables.py",
    "corridor": "python scripts/corridor_metrics.py compute=false tables=true",
    "supplement": "python scripts/make_figures_supplement.py",
    "report": "python scripts/make_report.py",
}
STEP_OUTPUTS = {  # what each command writes (the docstrings of the scripts)
    "full_history_audit": "the table `full_history` of `runs/_tables/m9/` and Figure S22 (`full_history_gain` of "
                          "`runs/_report/supplement/figures/`); rewrites the `full_history.json` of the recurrent "
                          "runs, which make_tables.py reads",
    "threshold_sensitivity": "the table `threshold_sensitivity` of `runs/_tables/m9/`",
    "equilibrium_roots": "the table `equilibrium_roots` of `runs/_tables/m9/`",
    "certificate_exact": "the table `certificate_exact` of `runs/_tables/m9/`",
    "band_sensitivity": "the table `band_sensitivity` of `runs/_tables/m8/` (needs the training events of level 3)",
    "tables": "`runs/_tables/m4/` (E1-E5, the arms of E2, the verdicts of H1) and the table `e2_monotone` of "
              "`runs/_tables/m8/`",
    "corridor": "`runs/_tables/m5/` (laws, components, instability, TOST, H12, e4_rmax, sensitivity) and the corridor "
                "tables of `runs/_tables/m8/` (asymmetry, asymmetry_contrasts, correlation_pooled, "
                "correlation_pooled_laws, power, temporal, correlation_clustered, contacts_absolute, h12_2_error, "
                "ablation_pairs) from the `macro.json` and `asymmetry.json` of every corridor run and ground truth "
                "(`asymmetry.json`: `python scripts/corridor_asymmetry.py`, level 3)",
    "supplement": "the tables `lowfreq_expansion`, `certificate_tightness` and `band_width` of `runs/_tables/m8/` and "
                  "the supplementary figures except S22 (`runs/_report/supplement/figures/`); band_width needs the "
                  "event sets of level 3, the corridor grids S10-S21 the `trajectories.npz` (level 3), else the "
                  "`fields.npz` of the corridor runs",
    "report": "`runs/_report/tables/` (the tables above, copied, and the tables derived from them), the seven figures "
              "of `runs/_report/figures/` (Figures 5 and 6 from the `trajectories.npz`, else the `fields.npz` of the "
              "corridor runs) and `runs/_report/report.md`",
}
M8_WRITERS = {  # runs/_tables/m8/<stem>: the step that writes it
    "e2_monotone": "tables", "lowfreq_expansion": "supplement", "certificate_tightness": "supplement",
    "band_width": "supplement", "band_sensitivity": "band_sensitivity",
    **{stem: "corridor" for stem in ("asymmetry", "asymmetry_contrasts", "correlation_pooled",
                                     "correlation_pooled_laws", "power", "temporal", "correlation_clustered",
                                     "contacts_absolute", "h12_2_error", "ablation_pairs")},
}
M9_WRITERS = {"full_history": "full_history_audit", "threshold_sensitivity": "threshold_sensitivity",
              "equilibrium_roots": "equilibrium_roots", "certificate_exact": "certificate_exact"}
REPORT_DERIVED = {  # runs/_report/tables/<stem> that make_report.py derives (None: from the run files)
    "e2_existence": None, "laws_raw": "runs/_tables/m5/laws.csv", "m5_components": "runs/_tables/m5/components.csv",
    "macro_error_dynamic": "runs/_tables/m5/instability.csv",
    "macro_error_dynamic_correlation": "runs/_tables/m5/instability_correlation.csv",
    "verdicts": "runs/_tables/m4/verdicts.csv", "verdicts_h12": "runs/_tables/m5/verdicts.csv",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def writers(name: str) -> list[str]:
    """The steps that write the file `name` (relative to the repository root), upstream first."""
    stem = Path(name).stem
    if name.startswith("runs/_report/tables/"):
        if stem in REPORT_DERIVED:
            up = REPORT_DERIVED[stem]
        else:  # a copy of the table of the same name
            found = [f"runs/_tables/{m}/{stem}.csv" for m in ("m4", "m5", "m8", "m9")
                     if (ROOT / "runs" / "_tables" / m / f"{stem}.csv").exists()]
            if len(found) != 1:
                raise SystemExit(f"{name}: no single table of runs/_tables behind it ({found})")
            up = found[0]
        return (writers(up) if up else []) + ["report"]
    if name.startswith("runs/_report/figures/"):
        return ["report"]
    if name.startswith("runs/_report/supplement/figures/"):
        return ["full_history_audit" if stem == "full_history_gain" else "supplement"]
    for prefix, table in (("runs/_tables/m4/", None), ("runs/_tables/m5/", None), ("runs/_tables/m8/", M8_WRITERS),
                          ("runs/_tables/m9/", M9_WRITERS)):
        if name.startswith(prefix):
            if table is None:
                return ["tables" if prefix.endswith("m4/") else "corridor"]
            if stem not in table:
                raise SystemExit(f"{name}: no command known for it")
            return [table[stem]]
    raise SystemExit(f"{name}: no command known for it")


def commands(sources: list[str]) -> str:
    steps = {s for name in sources for s in writers(name)}
    return " → ".join(f"`{STEPS[s]}`" for s in STEPS if s in steps)


def manuscript_floats(master: Path) -> list[tuple[str, str]]:
    """(kind, label) of the tables and figures of a LaTeX document in the order of the text (inputs followed)."""
    out: list[tuple[str, str]] = []

    def walk(f: Path) -> None:
        text = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in f.read_text(encoding="utf-8").splitlines())
        for m in re.finditer(r"\\begin\{(table|figure|xltabular|longtable)\}|\\input\{([^}]+)\}", text):
            if m.group(2):
                target = master.parent / (m.group(2) if m.group(2).endswith(".tex") else m.group(2) + ".tex")
                if target.exists():
                    walk(target)
                continue
            lab = re.search(r"\\label\{([^}]+)\}", text[m.end():])
            out.append(("figure" if m.group(1) == "figure" else "table", lab.group(1) if lab else "?"))

    walk(master)
    return out


def write_manifest(path: Path, main_tabs: list[Tab], supp_tabs: list[Tab]) -> int:
    """docs/paper_materials.md: one row per table and figure of the manuscript; returns the number of rows."""
    names = {t.name: t for t in main_tabs}
    assert sorted(n for kind, _l, _c, n, _h in MAIN_FLOATS if kind == "table" and n) == sorted(names)
    assert sorted(n for kind, _l, _c, n, _h in MAIN_FLOATS if kind == "figure") == sorted(n for n, _ in MAIN_FIGURES)
    if (V2 / "main.tex").exists():  # the authors' copy of the manuscript: its order of the floats
        found = manuscript_floats(V2 / "main.tex")
        expected = [(kind, label) for kind, label, *_ in MAIN_FLOATS]
        if found != expected:
            raise SystemExit(f"MAIN_FLOATS is not the order of the floats of paper/v2/main.tex: {found}")

    def files(xs: list[str]) -> str:
        return ", ".join(f"`{x}`" for x in xs)

    rows: list[tuple[str, ...]] = []
    count = {"table": 0, "figure": 0}
    for kind, label, content, name, hand in MAIN_FLOATS:
        count[kind] += 1
        item = f"{kind.capitalize()} {count[kind]}"
        if name is None:
            rows.append((item, content, f"`{label}`", "written by hand in the manuscript", hand, "—"))
            continue
        if kind == "table":
            t = names[name]
            assert t.label == label, (t.label, label)
            sources, generated = [rel(p) for p in t.sources], [f"tables/{name}.tex"]
        else:
            sources, generated = [f"runs/_report/figures/{name}.pdf"], [f"figures/{name}_env.tex", f"figures/{name}.pdf"]
        rows.append((item, content, f"`{label}`", files(generated), files(sources), commands(sources)))
    for k, ((_b, title), t) in enumerate(zip(SUPP_BUILDERS, supp_tabs), start=1):
        sources = [rel(p) for p in t.sources]
        rows.append((f"Table S{k}", title, f"`{t.label}`", files([f"supplement/tables/{t.name}.tex"]),
                     files(sources) if sources else "placeholder (no table yet)", commands(sources) or "—"))
    for k, ((name, _ctx), title) in enumerate(zip(SUPP_FIGURES, SUPP_FIGURE_TITLES), start=1):
        sources = [f"runs/_report/supplement/figures/{name}.pdf"]
        if name in FIGURE_TXT_NUMBERS:  # the caption copies numbers of the caption file
            sources.append(f"runs/_report/supplement/figures/{name}.txt")
        rows.append((f"Figure S{k}", title, f"`fig:S{name}`",
                     files([f"supplement/figures/S{k:02d}_{name}_env.tex", f"supplement/figures/{name}.pdf"]),
                     files(sources), commands(sources)))
    for r in rows + [(v,) for v in STEP_OUTPUTS.values()]:  # a bar would split a cell of the Markdown tables
        assert all("|" not in cell for cell in r), r
    n_main = sum(count.values())
    hand = [r[0].split(" ")[1] for r in rows[:n_main] if r[3] == "written by hand in the manuscript"]
    head = "| item | content | label | generated files | built from | commands |\n|---|---|---|---|---|---|"
    intro = (
        "The manuscript (*" + TITLE + "*, submitted to *Mathematics*, MDPI, 2026) is not part of this repository. Its "
        "generated tables and figure environments are written by `python scripts/paper_assets.py --out <dir>` (default "
        "`paper/v2`, the authors' copy of the manuscript; `--out paper_assets` in a checkout). One row per table and "
        "figure, numbered as in the manuscript: its label, the generated files (relative to `<dir>`), the files of this "
        "repository it is built from (every CSV and report its builder reads; a figure is copied unchanged from its "
        "PDF) and the commands that write them, upstream first. Tables " + ", ".join(hand[:-1]) + " and " + hand[-1]
        + " of the main text are written by hand in the manuscript.")
    lines = [
        "# Materials of the manuscript",
        "",
        "Written by `python scripts/paper_assets.py --manifest`; do not edit by hand.",
        "",
        *textwrap.wrap(intro, 115, break_long_words=False, break_on_hyphens=False),
        "",
        "## Commands",
        "",
        "The section Reproduce of `README.md` gives the inputs of every command and the three levels of reproduction.",
        "",
        "| command | writes |",
        "|---|---|",
        *(f"| `{STEPS[s]}` | {STEP_OUTPUTS[s]} |" for s in STEPS),
        "| `python scripts/paper_assets.py --out <dir>` | the generated files of the tables below |",
        "",
        f"## Main text ({count['table']} tables, {count['figure']} figures)",
        "",
        head,
        *("| " + " | ".join(r) + " |" for r in rows[:n_main]),
        "",
        f"## Supplementary Materials ({len(supp_tabs)} tables, {len(SUPP_FIGURES)} figures)",
        "",
        head,
        *("| " + " | ".join(r) + " |" for r in rows[n_main:]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return len(rows)


def report(tabs: list[Tab], title: str) -> None:
    print(title)
    for t in tabs:
        lay = t.layout
        parts = t.panels or [t]
        stacked = "; ".join(",".join(p.cols[j].head.replace("\\\\", " ") for j in sorted(p.layout["stacked"])) or "-"
                            for p in parts)
        print(f"  {t.name:36s} {lay['ctx']:9s} {lay['kind']:5s} {lay['size']:12s} "
              f"sep={','.join('%g' % p.layout['sep'] for p in parts):7s} est={lay['total']:5.0f}/"
              f"{TARGET[lay['ctx']]:.0f}pt h={lay['height']:4.0f}pt stacked: {stacked[:70]}")


def from_root(path: str) -> Path:
    """A path of the command line: absolute, or relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="paper/v2",
                        help="output directory, absolute or relative to the repository root (default: paper/v2, the "
                             "manuscript); created when missing")
    parser.add_argument("--manifest", nargs="?", const=MANIFEST, default=None, metavar="PATH",
                        help=f"also write the map of the materials of the manuscript (default PATH: {MANIFEST})")
    args = parser.parse_args(argv)
    out = from_root(args.out)
    main_tabs = build_main(out)
    report(main_tabs, "main text")
    supp = build_supplement(out)
    report(supp, "supplementary materials")
    build_figures(out)
    write_supplementary(supp, out)
    write_items(out)
    print(f"{len(main_tabs)} main tables, {len(supp)} supplementary tables, {len(MAIN_FIGURES)} main figures, "
          f"{len(SUPP_FIGURES)} supplementary figures; supplementary.tex and supplement/items.txt written to "
          f"{shown(out)}")
    if args.manifest:
        path = from_root(args.manifest)
        print(f"{shown(path)}: {write_manifest(path, main_tabs, supp)} rows")
    return 0


def shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


if __name__ == "__main__":
    sys.exit(main())
