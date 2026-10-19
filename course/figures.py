"""Figures, charts and generated tables for the course.

Everything here is computed from the team repository at build time (the Othello rules, the docs, the
training records and the anchor results), so a board, a token number or a chart value can never drift
from the code it describes. init(repo) must be called first.
"""

from __future__ import annotations

import csv
import glob
import html
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO: Path = Path(".")
HERE = Path(__file__).resolve().parent
oth = None
frag = None
train = None
COLS = "ABCDEFGH"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def init(repo: Path) -> None:
    global REPO, oth, frag, train
    REPO = repo
    oth = _load("othello", "src/domains/othello.py")
    frag = _load("fragility", "src/fragility/othello.py")
    train = _load("zoo_train", "src/zoo/train.py")


def sq(name: str) -> int:
    return (int(name[1]) - 1) * 8 + COLS.index(name[0])


def nm(s: int) -> str:
    return COLS[s % 8] + str(s // 8 + 1)


def f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def svg(label: str, w: float, h: float, body: list[str], marker: str | None = None, extra_defs: str = "",
        min_width: int | None = None) -> str:
    defs = extra_defs
    if marker:
        defs += (f'<marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                 f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker>')
    style = f' style="min-width:{min_width}px;max-width:{f(w)}px"' if min_width else f' style="max-width:{f(w)}px"'
    return (f'<svg viewBox="0 0 {f(w)} {f(h)}" role="img" aria-label="{esc(label)}"{style}>'
            + (f"<defs>{defs}</defs>" if defs else "") + "\n".join(body) + "</svg>")


# ==========================================================================
# Boards
# ==========================================================================

def board_after(names: list[str]) -> tuple[np.ndarray, list[list[int]]]:
    """Board after the moves (Black first, no passes) and the flips of each move."""
    b, p, flips = oth.start_board(), oth.BLACK, []
    for n in names:
        fl = oth.flips_for(b, sq(n), p)
        assert fl, n
        b[fl] = p
        b[sq(n)] = p
        flips.append(fl)
        p = oth.other(p)
    return b, flips


def ctr(ox: float, oy: float, cell: float, s: int) -> tuple[float, float]:
    return ox + cell * (s % 8) + cell / 2, oy + cell * (s // 8) + cell / 2


def board_g(ox, oy, cell, board, *, coords=True, coord_size=11, legal=(), last=None, flipped=(), ghost=None, ring=()):
    S = cell * 8
    p = [f'<rect class="board" x="{f(ox)}" y="{f(oy)}" width="{f(S)}" height="{f(S)}" rx="2"/>']
    d = " ".join([f"M{f(ox + cell * i)},{f(oy)}V{f(oy + S)}" for i in range(1, 8)]
                 + [f"M{f(ox)},{f(oy + cell * i)}H{f(ox + S)}" for i in range(1, 8)])
    p.append(f'<path class="board-grid" d="{d}"/>')
    if coords:
        for c in range(8):
            p.append(f'<text class="t-coord" font-size="{coord_size}" x="{f(ox + cell * c + cell / 2)}" '
                     f'y="{f(oy - coord_size * 0.55)}" text-anchor="middle">{COLS[c]}</text>')
        for r in range(8):
            p.append(f'<text class="t-coord" font-size="{coord_size}" x="{f(ox - coord_size * 0.55)}" '
                     f'y="{f(oy + cell * r + cell / 2 + coord_size * 0.36)}" text-anchor="end">{r + 1}</text>')
    rd = cell * 0.39
    for s in range(64):
        if board[s]:
            cx, cy = ctr(ox, oy, cell, s)
            cls = "disc-b" if board[s] == oth.BLACK else "disc-w"
            p.append(f'<circle class="{cls}" cx="{f(cx)}" cy="{f(cy)}" r="{f(rd)}"/>')
    for s in flipped:
        cx, cy = ctr(ox, oy, cell, s)
        p.append(f'<circle class="mark-flip" cx="{f(cx)}" cy="{f(cy)}" r="{f(rd + cell * 0.07)}"/>')
    for s in ring:
        cx, cy = ctr(ox, oy, cell, s)
        p.append(f'<rect class="mark-square" x="{f(cx - cell / 2 + 1.5)}" y="{f(cy - cell / 2 + 1.5)}" width="{f(cell - 3)}" height="{f(cell - 3)}"/>')
    if last is not None:
        cx, cy = ctr(ox, oy, cell, last)
        p.append(f'<circle class="mark-last" cx="{f(cx)}" cy="{f(cy)}" r="{f(cell * 0.11)}"/>')
    for s in legal:
        cx, cy = ctr(ox, oy, cell, s)
        p.append(f'<circle class="legal-dot" cx="{f(cx)}" cy="{f(cy)}" r="{f(cell * 0.1)}"/>')
    if ghost is not None:
        cx, cy = ctr(ox, oy, cell, ghost)
        p.append(f'<circle class="ghost" cx="{f(cx)}" cy="{f(cy)}" r="{f(rd)}"/>')
    return "\n".join(p)


def fig_start() -> str:
    b = oth.start_board()
    legal = oth.legal_moves(b, oth.BLACK)
    ox, oy, cell = 44, 40, 38
    g = [board_g(ox, oy, cell, b, coord_size=12, legal=legal)]
    lx = 392

    def swatch(y, inner):
        return f'<rect class="board" x="{lx}" y="{y - 13}" width="26" height="26" rx="3"/>' + inner
    g.append(swatch(96, f'<circle class="disc-b" cx="{lx + 13}" cy="96" r="10"/>'))
    g.append(f'<text class="t-body" x="{lx + 38}" y="100">Black</text>')
    g.append(swatch(132, f'<circle class="disc-w" cx="{lx + 13}" cy="132" r="10"/>'))
    g.append(f'<text class="t-body" x="{lx + 38}" y="136">White</text>')
    g.append(swatch(168, f'<circle class="legal-dot" cx="{lx + 13}" cy="168" r="3.8"/>'))
    g.append(f'<text class="t-body" x="{lx + 38}" y="172">a legal move for Black</text>')
    g.append(f'<text class="t-strong" x="{lx}" y="226">Black moves first</text>')
    g.append(f'<text class="t-body" x="{lx}" y="262">A1 = square 0, top left</text>')
    g.append(f'<text class="t-body" x="{lx}" y="284">H8 = square 63, bottom right</text>')
    g.append(f'<text class="t-body" x="{lx}" y="306">D3 = column D, row 3</text>')
    label = ("The Othello start position: white discs on D4 and E5, black on E4 and D5. Dots mark Black's "
             "four legal first moves, D3, C4, F5 and E6. Columns are A to H, rows 1 to 8 from the top.")
    return svg(label, 620, 372, g)


def fig_filmstrip() -> str:
    seq = ["D3", "C3", "C4", "E3"]
    cell, top, gap = 17, 46, 38
    S = cell * 8
    g = []
    panels = [("Start", oth.start_board(), None, [], "Black to move")]
    for k in range(1, 5):
        b, flips = board_after(seq[:k])
        who = "Black" if k % 2 else "White"
        flipped = flips[-1]
        panels.append((f"{k} · {who} {seq[k - 1]}", b, sq(seq[k - 1]), flipped,
                       "flips " + " and ".join(nm(s) for s in flipped)))
    for i, (title, b, last, flipped, text) in enumerate(panels):
        ox = 30 + i * (S + gap)
        g.append(f'<text class="t-title-sm" x="{ox}" y="18">{title}</text>')
        g.append(board_g(ox, top, cell, b, coord_size=9, last=last, flipped=flipped))
        nb, nw = int((b == oth.BLACK).sum()), int((b == oth.WHITE).sum())
        g.append(f'<text class="t-body" x="{ox}" y="{top + S + 22}">{text}</text>')
        g.append(f'<text class="t-muted" x="{ox}" y="{top + S + 40}">Black {nb} · White {nw}</text>')
        if i < 4:
            x0, x1 = ox + S + 9, ox + S + gap - 9
            g.append(f'<path class="wire" d="M{x0},{top + S / 2} H{x1}" marker-end="url(#fs-arrow)"/>')
    width = 30 + 5 * S + 4 * gap + 14
    label = ("Four opening moves. 1: Black D3 flips D4. 2: White C3 flips D4 back. 3: Black C4 flips D4 again. "
             "4: White E3 flips D3 and E4 at once. Rings mark discs that just flipped; a dot marks the disc just played.")
    return svg(label, width, top + S + 50, g, marker="fs-arrow")


def fig_rays() -> str:
    b, _ = board_after(["D3", "C3", "C4"])
    ox, oy, cell = 44, 40, 38
    e3, d3, e4, c5 = sq("E3"), sq("D3"), sq("E4"), sq("C5")
    assert sorted(oth.flips_for(b, e3, oth.WHITE)) == sorted([d3, e4])
    g = [board_g(ox, oy, cell, b, coord_size=12, flipped=[d3, e4], ghost=e3)]
    cx, cy = ctr(ox, oy, cell, e3)
    rd = cell * 0.39 + 2
    c3x, _ = ctr(ox, oy, cell, sq("C3"))
    _, e5y = ctr(ox, oy, cell, sq("E5"))
    c5x, c5y = ctr(ox, oy, cell, c5)
    diag = rd / 2 ** 0.5
    g.append(f'<path class="ray-hit" d="M{f(cx - rd)},{f(cy)} H{f(c3x + rd + 3)}" marker-end="url(#ray-arrow)"/>')
    g.append(f'<path class="ray-hit" d="M{f(cx)},{f(cy + rd)} V{f(e5y - rd - 3)}" marker-end="url(#ray-arrow)"/>')
    g.append(f'<path class="ray-miss" d="M{f(cx - diag)},{f(cy + diag)} L{f(c5x + 7)},{f(c5y - 7)}"/>')
    g.append(f'<path class="ray-x" d="M{f(c5x - 6)},{f(c5y - 6)} L{f(c5x + 6)},{f(c5y + 6)} M{f(c5x + 6)},{f(c5y - 6)} L{f(c5x - 6)},{f(c5y + 6)}"/>')
    for dr, dc in [(-1, 0), (0, 1), (-1, -1), (-1, 1), (1, 1)]:
        norm = (dc * dc + dr * dr) ** 0.5
        ux, uy = dc / norm, dr / norm
        g.append(f'<path class="ray-stub" d="M{f(cx + ux * rd)},{f(cy + uy * rd)} L{f(cx + ux * (rd + 13))},{f(cy + uy * (rd + 13))}"/>')
    lx = 392
    g.append(f'<text class="t-mono" x="{lx}" y="74">flips_for(board, E3, WHITE)</text>')
    rows = [("ray-hit", "left: D3 black, then C3 white → D3 flips"),
            ("ray-hit", "down: E4 black, then E5 white → E4 flips"),
            ("ray-miss", "down-left: D4 black, then C5 empty → none"),
            ("ray-stub", "5 other directions: empty at once → none")]
    for k, (cls, text) in enumerate(rows):
        y = 116 + k * 32
        g.append(f'<path class="{cls}" d="M{lx},{y - 4} H{lx + 26}"/>')
        g.append(f'<text class="t-body" x="{lx + 38}" y="{y}">{text}</text>')
    g.append(f'<text class="t-mono" x="{lx}" y="270">returns [19, 28]</text>')
    g.append(f'<text class="t-muted" x="{lx}" y="292">squares 19 and 28 are D3 and E4;</text>')
    g.append(f'<text class="t-muted" x="{lx}" y="310">a non-empty list means E3 is legal</text>')
    label = ("flips_for walking out from E3 for White after D3 C3 C4. Left, D3 is black and C3 white, so D3 flips. "
             "Down, E4 is black and E5 white, so E4 flips. Down-left, D4 is black but C5 is empty, so nothing flips. "
             "The other five directions start on an empty square. The function returns squares 19 and 28.")
    defs = ('<marker id="ray-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ray-head"/></marker>')
    return svg(label, 720, 372, g, extra_defs=defs)


def fig_tokenmap() -> str:
    ox, oy, cell = 40, 36, 52
    hl = {sq(n) for n in ("D3", "C3", "C4", "E3")}
    g = []
    for c in range(8):
        g.append(f'<text class="t-coord" font-size="12" x="{ox + cell * c + cell / 2}" y="{oy - 10}" text-anchor="middle">{COLS[c]}</text>')
    for r in range(8):
        g.append(f'<text class="t-coord" font-size="12" x="{ox - 10}" y="{oy + cell * r + cell / 2 + 4}" text-anchor="end">{r + 1}</text>')
    start = oth.start_board()
    for s in range(64):
        x, y = ox + cell * (s % 8), oy + cell * (s // 8)
        if s in oth.CENTER:
            g.append(f'<rect class="tm-center" x="{x}" y="{y}" width="{cell}" height="{cell}"/>')
            col = "disc-w" if start[s] == oth.WHITE else "disc-b"
            g.append(f'<circle class="{col}" cx="{x + cell / 2}" cy="{y + cell / 2 + 3}" r="9"/>')
            g.append(f'<text class="tm-sq" x="{x + 5}" y="{y + 13}">{s}</text>')
            continue
        cls = "tm-hl" if s in hl else "tm-cell"
        g.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cell}" height="{cell}"/>')
        g.append(f'<text class="tm-sq" x="{x + 5}" y="{y + 13}">{s}</text>')
        g.append(f'<text class="tm-token" x="{x + cell / 2}" y="{y + cell / 2 + 9}" text-anchor="middle">{oth.SQ_TO_TOKEN[s]}</text>')
    label = ("An 8 by 8 grid giving, for every square, its index in the code (0 to 63, row by row from A1) and its "
             "token (1 to 60). The four start squares have no token. D3, C3, C4 and E3 are tokens 20, 19, 27 and 21.")
    return svg(label, ox + cell * 8 + 12, oy + cell * 8 + 12, g, min_width=420)


def build_example() -> str:
    seq = ["D3", "C3", "C4", "E3"]
    game = np.full(60, -1, dtype=np.int8)
    game[:4] = [sq(n) for n in seq]
    d = oth.build(game[None], legal=True)
    rows = []
    for t, n in enumerate(seq):
        mover = int(d["movers"][0, t])
        toks = np.flatnonzero(d["legal_next"][0, t])
        squares = ", ".join(nm(oth.TOKEN_TO_SQ[int(k)]) for k in toks)
        nxt = "White" if mover == oth.BLACK else "Black"
        rows.append(f'<tr><td class="num">{t}</td><td><code>{n}</code></td><td class="num">{d["tokens"][0, t]}</td>'
                    f'<td class="num">{mover} <span class="dim">({"Black" if mover == oth.BLACK else "White"})</span></td>'
                    f'<td class="num">{" ".join(str(int(k)) for k in toks)}</td><td>{squares} <span class="dim">({nxt} to move)</span></td></tr>')
    return ('<div class="table-scroll"><table class="ex-table"><thead><tr><th scope="col">t</th><th scope="col">Move</th>'
            '<th scope="col"><code>tokens</code></th><th scope="col"><code>movers</code></th>'
            '<th scope="col"><code>legal_next</code> (tokens)</th><th scope="col">The same, as squares</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def fig_d2_pairs() -> str:
    """The compression pair (same board by two orders) and the distinction pair (boundary move B4)."""
    def board_of(names):
        b, _ = board_after(names)
        mover = oth.BLACK if len(names) % 2 else oth.WHITE
        return b, sorted(oth.legal_next(b, mover))

    a1, l1 = board_of(["D3", "C3", "C4"])
    a2, l2 = board_of(["C4", "C3", "D3"])
    assert np.array_equal(a1, a2) and l1 == l2 == sorted([sq("C5"), sq("E3")])
    b1, m1 = board_of(["E6", "D6", "C5"])
    b2, m2 = board_of(["F5", "D6", "C6"])
    boundary = sorted(set(m1) ^ set(m2))
    assert [nm(s) for s in boundary] == ["B4"], [nm(s) for s in boundary]
    assert [nm(s) for s in m1] == ["B4", "F4", "B6", "F6"] and [nm(s) for s in m2] == ["F4", "B6", "F6"], ([nm(s) for s in m1], [nm(s) for s in m2])
    cell = 19
    S = cell * 8
    g = []
    xs = [40, 236, 492, 688]
    titles = [("D3 C3 C4", l1, a1, []), ("C4 C3 D3", l2, a2, []), ("E6 D6 C5", m1, b1, boundary), ("F5 D6 C6", m2, b2, boundary)]
    g.append('<text class="t-title" x="40" y="22">Same board, different order</text>')
    g.append('<text class="t-title" x="492" y="22">Same length, different boards</text>')
    for x, (moves, legal, b, ring) in zip(xs, titles, strict=True):
        g.append(f'<text class="t-mono" x="{x}" y="52">{moves}</text>')
        g.append(board_g(x, 72, cell, b, coord_size=9, legal=legal, ring=ring))
        g.append(f'<text class="t-body" x="{x}" y="{72 + S + 24}">legal: {", ".join(nm(s) for s in legal)}</text>')
    g.append(f'<text class="t-muted" x="40" y="{72 + S + 46}">compression: the model should accept the same moves after both</text>')
    g.append(f'<text class="t-muted" x="492" y="{72 + S + 46}">distinction: the square B4 is the boundary</text>')
    g.append(f'<line class="divider" x1="458" y1="34" x2="458" y2="{72 + S + 50}"/>')
    label = ("Left: D3 C3 C4 and C4 C3 D3 reach the identical board, White to move, legal moves C5 and E3 after both. "
             "Right: E6 D6 C5 and F5 D6 C6 are both three moves long but reach different boards. B4 is legal after the "
             "first and not after the second, so B4 is the boundary between them.")
    return svg(label, 880, 72 + S + 60, g)


def fig_symmetry() -> str:
    seq = ["D3", "C3", "C4", "E3"]
    cell = 16
    S = cell * 8
    g = []
    names = [("original", None), ("rot180", "rot180"), ("transpose", "transpose"), ("anti_transpose", "anti_transpose")]
    for i, (title, sym) in enumerate(names):
        game = np.full(60, -1, dtype=np.int8)
        game[:4] = [sq(n) for n in seq]
        if sym:
            game = frag.transform_games(game[None], sym)[0]
        moves = [int(s) for s in game if s >= 0]
        out = oth.replay(np.array(moves))
        assert out is not None
        boards, _ = out
        ox = 30 + i * (S + 50)
        g.append(f'<text class="t-title-sm" x="{ox}" y="18">{title}</text>')
        g.append(f'<text class="t-mono" x="{ox}" y="38">{" ".join(nm(m) for m in moves)}</text>')
        g.append(board_g(ox, 56, cell, boards[-1], coord_size=8.5))
    # the 90 degree rotation that is NOT used: it swaps the colours of the start squares
    ox = 30 + 4 * (S + 50) + 20
    start = oth.start_board()
    rot = np.zeros_like(start)
    for s in range(64):
        r, c = divmod(s, 8)
        rot[c * 8 + (7 - r)] = start[s]
    assert rot[sq("D4")] == oth.BLACK and rot[sq("E4")] == oth.WHITE
    g.append(f'<line class="divider" x1="{ox - 26}" y1="6" x2="{ox - 26}" y2="{56 + S + 30}"/>')
    g.append(f'<text class="t-title-sm" x="{ox}" y="18">rot90 (not used)</text>')
    g.append(f'<text class="t-mono" x="{ox}" y="38">start position</text>')
    g.append(board_g(ox, 56, cell, rot, coord_size=8.5, ring=[sq("D4"), sq("E4"), sq("D5"), sq("E5")]))
    g.append(f'<text class="t-muted" x="{ox}" y="{56 + S + 22}">colours of the start swap</text>')
    g.append(f'<text class="t-muted" x="30" y="{56 + S + 22}">the same game with the squares renamed: every move stays legal, same movers</text>')
    label = ("The four-move opening D3 C3 C4 E3 under the three symmetries the fragility suite uses: 180 degree rotation "
             "gives E6 F6 F5 D6, transpose gives C4 C3 D3 C5, anti-transpose gives F5 F6 E6 F4. Each is the same game "
             "relabelled. A 90 degree rotation is not used because it swaps the colours of the four start discs.")
    return svg(label, ox + S + 24, 56 + S + 34, g)


def byte_strip(values: list[int], groups: list[tuple[int, int, str, str]], decode=None, per_row: int = 20) -> str:
    """HTML strip of bytes. groups: (start, end, css class, label). decode(i, v) -> small text under a byte."""
    cells = []
    for i, v in enumerate(values):
        cls = next((c for a, b, c, _ in groups if a <= i < b), "")
        sub = decode(i, v) if decode else ""
        cells.append(f'<span class="byte {cls}"><b>{v}</b><i>{esc(sub)}</i></span>')
    legend = "".join(f'<li><span class="byte-key {c}"></span>{esc(lbl)} <span class="dim">bytes {a}–{b - 1}</span></li>'
                     for a, b, c, lbl in groups)
    return (f'<div class="bytes" style="--per-row:{per_row}">' + "".join(cells) + "</div>"
            f'<ul class="legend byte-legend">{legend}</ul>')


SYNTH_GAME = [19, 20, 21, 18, 17, 11, 34, 16, 37, 22, 3, 45, 23, 33, 53, 4, 42, 2, 24, 50, 41, 32, 9, 14, 48, 29,
              59, 61, 30, 0, 26, 49, 8, 58, 40, 25, 6, 39, 13, 31, 43, 52, 44, 15, 5, 60, 54, 55, 62, 63, 57, 56, 10,
              51, 38, 1, 12, 46, 47, 7]
WTHOR_REC = [9, 0, 9, 15, 110, 13, 31, 37, 56, 64, 53, 46, 35, 63, 34, 66, 65, 52, 43, 36, 74, 73, 62, 85, 61, 75,
             83, 41, 42, 25, 84, 82, 16, 24, 57, 33, 32, 23, 15, 47, 26, 48, 37, 38, 31, 21, 76, 14, 13, 22, 12, 86,
             67, 58, 78, 72, 81, 71, 87, 27, 18, 17, 28, 11, 68, 77, 51, 0]


def bytes_synthetic() -> str:
    vals = SYNTH_GAME[:20]
    return byte_strip(vals, [(0, 20, "b-move", "move squares (0 to 63), one byte each")],
                      decode=lambda i, v: nm(v))


def bytes_wthor() -> str:
    vals = WTHOR_REC[:28]

    def dec(i, v):
        if i >= 8:
            if v == 0:
                return "none"
            s = (v // 10 - 1) * 8 + (v % 10 - 1)
            return nm(s)
        return ""
    assert nm((56 // 10 - 1) * 8 + (56 % 10 - 1)) == "F5"
    return byte_strip(vals, [(0, 2, "b-hdr", "tournament"), (2, 4, "b-hdr2", "black player"),
                             (4, 6, "b-hdr", "white player"), (6, 8, "b-hdr2", "two scores"),
                             (8, 28, "b-move", "moves as 10 × row + column")], decode=dec, per_row=14)


# ==========================================================================
# Charts: SVG with data-tip hover marks, a legend for 2+ series, and a table view
# ==========================================================================

def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float, digits: int = 2) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:.{digits}f}"


def table_view(headers: list[str], rows: list[list], caption: str = "Show the numbers") -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="{"num" if isinstance(c, (int, float)) else ""}">'
                                    f'{esc(_fmt(c, 4) if isinstance(c, float) else c)}</td>' for c in r) + "</tr>"
                   for r in rows)
    return (f'<details class="numbers"><summary>{esc(caption)}</summary><div class="table-scroll">'
            f'<table class="num-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>')


def legend(items: list[tuple[str, str, str]]) -> str:
    """items: (kind: bar|line|band|dot|dash, css class, label)."""
    out = []
    for kind, cls, label in items:
        if kind == "line":
            key = f'<svg viewBox="0 0 22 10" aria-hidden="true"><line class="{cls}" x1="1" y1="5" x2="21" y2="5"/></svg>'
        elif kind == "dash":
            key = f'<svg viewBox="0 0 22 10" aria-hidden="true"><line class="{cls}" x1="1" y1="5" x2="21" y2="5"/></svg>'
        elif kind == "dot":
            key = f'<svg viewBox="0 0 12 12" aria-hidden="true"><circle class="{cls}" cx="6" cy="6" r="4.5"/></svg>'
        else:
            key = f'<svg viewBox="0 0 14 12" aria-hidden="true"><rect class="{cls}" x="0" y="1" width="14" height="10" rx="2"/></svg>'
        out.append(f'<li>{key}{esc(label)}</li>')
    return f'<ul class="legend chart-legend">{"".join(out)}</ul>'


def chart(svg_markup: str, caption: str, legend_html: str = "", table_html: str = "") -> str:
    return (f'<figure class="chart"><div class="fig-box chart-box">{svg_markup}</div>{legend_html}'
            f'<figcaption><b><!--FIGNUM--></b> {caption}</figcaption>{table_html}</figure>')


def _bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar: square at the baseline (left), rounded data end (right)."""
    r = min(r, w / 2, h / 2) if w > 0 else 0
    return (f"M{f(x0)},{f(y)} H{f(x0 + w - r)} Q{f(x0 + w)},{f(y)} {f(x0 + w)},{f(y + r)} "
            f"V{f(y + h - r)} Q{f(x0 + w)},{f(y + h)} {f(x0 + w - r)},{f(y + h)} H{f(x0)} Z")


def _col_path(x: float, y0: float, w: float, h: float, r: float = 4) -> str:
    """Vertical column: square at the baseline (bottom y0), rounded data end (top)."""
    r = min(r, w / 2, h / 2) if h > 0 else 0
    top = y0 - h
    return (f"M{f(x)},{f(y0)} V{f(top + r)} Q{f(x)},{f(top)} {f(x + r)},{f(top)} H{f(x + w - r)} "
            f"Q{f(x + w)},{f(top)} {f(x + w)},{f(top + r)} V{f(y0)} Z")


def chart_throughput(caption: str) -> str:
    rows = []
    for line in (REPO / "src/docs/throughput.md").read_text().splitlines():
        m = re.match(r"\|\s*(transformer|lstm|mamba)\s*\|\s*(small|medium|large)\s*\|\s*([\dx ]+)\|\s*([\d.]+M)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3).strip(), m.group(4), float(m.group(5)), float(m.group(6))))
    assert len(rows) == 9, rows
    W, left, right, top, rowh, gap_group = 760, 170, 70, 28, 26, 12
    vmax = 150
    plot_w = W - left - right
    y = top
    g, ys = [], []
    for k, (arch, scale, *_rest) in enumerate(rows):
        if k and rows[k - 1][0] != arch:
            y += gap_group
        ys.append(y)
        y += rowh
    H = y + 34
    for t in _ticks(0, vmax, 5):
        x = left + plot_w * t / vmax
        g.append(f'<line class="grid" x1="{f(x)}" y1="{top - 8}" x2="{f(x)}" y2="{H - 30}"/>')
        g.append(f'<text class="tick" x="{f(x)}" y="{H - 12}" text-anchor="middle">{_fmt(t)}</text>')
    g.append(f'<text class="tick" x="{left + plot_w}" y="{top - 14}" text-anchor="end">minutes per model at the full budget</text>')
    for (arch, scale, layers, params, sps, minutes), yy in zip(rows, ys, strict=True):
        w = plot_w * minutes / vmax
        cls = "em" if arch == "mamba" else "de"
        tip = f"{arch} {scale}: {minutes:g} min per model · {sps:g} s/step · {params} params"
        g.append(f'<text class="row-label" x="{left - 12}" y="{yy + 15}" text-anchor="end">{arch} · {scale}</text>')
        g.append(f'<path class="{cls}" d="{_bar_path(left, yy + 3, max(w, 1.5), 16)}"/>')
        g.append(f'<rect class="hit" x="{left}" y="{yy}" width="{plot_w + right}" height="{rowh - 2}" data-tip="{esc(tip)}" tabindex="0"/>')
        g.append(f'<text class="val" x="{f(left + w + 8)}" y="{yy + 15}">{minutes:g}</text>')
    g.append(f'<line class="axis" x1="{left}" y1="{top - 8}" x2="{left}" y2="{H - 30}"/>')
    label = "Training minutes per model at the full budget: " + "; ".join(f"{a} {s} {m:g}" for a, s, _l, _p, _s, m in rows)
    table = table_view(["Architecture", "Scale", "Layers × width", "Parameters", "Seconds per step", "Minutes per model"],
                       [[a, s, l, p, sp, m] for a, s, l, p, sp, m in rows])
    leg = legend([("bar", "em", "Mamba"), ("bar", "de", "transformer and LSTM")])
    return chart(svg(label, W, H, g), caption, leg, table)


def chart_lr_loss(caption: str) -> str:
    budget = train.matrix.TRAINING_BUDGET
    steps = list(range(0, int(budget["steps"]) + 1, 25))
    lrs = [train.lr_at(min(s, int(budget["steps"]) - 1), budget["lr"], int(budget["warmup_steps"]), int(budget["steps"])) for s in steps]
    loss = json.loads((HERE / "data/loss_curve.json").read_text())
    W, H = 780, 250
    panel_w, gap = 310, 70
    g = []

    def panel(ox, xs, ys, ylo, yhi, ytitle, fmt, key, tipname):
        top, bottom = 30, H - 40
        ph = bottom - top
        sx = lambda v: ox + panel_w * v / 5000  # noqa: E731
        sy = lambda v: bottom - ph * (v - ylo) / (yhi - ylo)  # noqa: E731
        for t in _ticks(ylo, yhi, 4):
            g.append(f'<line class="grid" x1="{ox}" y1="{f(sy(t))}" x2="{ox + panel_w}" y2="{f(sy(t))}"/>')
            g.append(f'<text class="tick" x="{ox - 8}" y="{f(sy(t) + 4)}" text-anchor="end">{fmt(t)}</text>')
        for t in (0, 1000, 2000, 3000, 4000, 5000):
            g.append(f'<text class="tick" x="{f(sx(t))}" y="{bottom + 18}" text-anchor="middle">{_fmt(t)}</text>')
        g.append(f'<line class="axis" x1="{ox}" y1="{bottom}" x2="{ox + panel_w}" y2="{bottom}"/>')
        g.append(f'<text class="panel-title" x="{ox}" y="{top - 14}">{ytitle}</text>')
        pts = " ".join(f"{f(sx(x))},{f(sy(y))}" for x, y in zip(xs, ys, strict=True))
        g.append(f'<polyline class="{key}" points="{pts}"/>')
        g.append(f'<circle class="{key}-dot" cx="{f(sx(xs[-1]))}" cy="{f(sy(ys[-1]))}" r="4"/>')
        bw = panel_w / (len(xs) - 1)
        for x, y in zip(xs, ys, strict=True):
            g.append(f'<rect class="hit band-hit" x="{f(sx(x) - bw / 2)}" y="{top}" width="{f(bw)}" height="{ph}" '
                     f'data-tip="step {x:,}: {tipname} {fmt(y, True)}"/>')
        g.append(f'<text class="tick" x="{ox + panel_w}" y="{bottom + 34}" text-anchor="end">optimizer step</text>')

    panel(64, steps, lrs, 0, 3e-4, "Learning rate (lr_at)", lambda v, full=False: f"{v * 1e4:.1f}e-4" if v else "0", "line-em", "learning rate")
    panel(64 + panel_w + gap, loss["steps"], loss["loss"], 2.0, 4.5, "Training loss of one zoo model",
          lambda v, full=False: f"{v:.3f}" if full else f"{v:.1f}", "line-ink", "loss")
    label = ("Two panels. Left: the learning rate rises linearly to 3e-4 over the first 200 steps, then falls along a "
             "cosine to zero at step 5,000. Right: the training loss of one zoo model falls from 4.28 at step 1 to about 2.47 at step 5,000.")
    rows = []
    for x in (1, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000):
        lr = train.lr_at(min(x, 4999), budget["lr"], int(budget["warmup_steps"]), int(budget["steps"]))
        ls = loss["loss"][loss["steps"].index(x)] if x in loss["steps"] else ""
        rows.append([x, f"{lr:.2e}", ls])
    table = table_view(["Step", "Learning rate", "Loss (logged every 50 steps)"], rows)
    return chart(svg(label, W, H, g), caption, "", table)


def chart_legal_rates(caption: str) -> str:
    recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(REPO / "results/zoo_training/*.json")))]
    recs = [r for r in recs if r.get("domain", "othello") == "othello"]
    archs, scales, dists = ["transformer", "lstm", "mamba"], ["small", "medium", "large"], ["synthetic", "championship", "mixed"]
    W, H, left, top, bottom = 900, 330, 54, 34, 270
    groups = [(a, s) for a in archs for s in scales]
    gw = (W - left - 20) / len(groups)
    ylo, yhi = 0.4, 1.0
    sy = lambda v: bottom - (bottom - top) * (v - ylo) / (yhi - ylo)  # noqa: E731
    g = []
    for t in _ticks(ylo, yhi, 6):
        g.append(f'<line class="grid" x1="{left}" y1="{f(sy(t))}" x2="{W - 20}" y2="{f(sy(t))}"/>')
        g.append(f'<text class="tick" x="{left - 8}" y="{f(sy(t) + 4)}" text-anchor="end">{t:.1f}</text>')
    g.append(f'<line class="threshold" x1="{left}" y1="{f(sy(0.5))}" x2="{W - 20}" y2="{f(sy(0.5))}"/>')
    g.append(f'<text class="tick strong" x="{W - 20}" y="{f(sy(0.5) + 15)}" text-anchor="end">admission line: 0.50</text>')
    for i, (a, s) in enumerate(groups):
        x0 = left + i * gw
        if i and groups[i - 1][0] != a:
            g.append(f'<line class="grid" x1="{f(x0)}" y1="{top - 10}" x2="{f(x0)}" y2="{bottom + 34}"/>')
        g.append(f'<text class="tick" x="{f(x0 + gw / 2)}" y="{bottom + 18}" text-anchor="middle">{s}</text>')
        if s == "medium":
            g.append(f'<text class="row-label" x="{f(x0 + gw / 2)}" y="{bottom + 38}" text-anchor="middle">{a}</text>')
        for j, dname in enumerate(dists):
            for r in recs:
                if r["arch"] == a and r["scale"] == s and r.get("distribution", r["config"]["distribution"]) == dname:
                    v = r["eval"]["legal_move_rate"]
                    seed = r.get("seed", r["config"]["seed"])
                    jx = x0 + gw * (0.14 + 0.72 * (j * 3 + seed) / 8)
                    tip = f'{r["model_id"]}: legal move rate {v:.4f} on 1,000 validation games'
                    g.append(f'<circle class="dot s{j + 1}" cx="{f(jx)}" cy="{f(sy(v))}" r="4"/>')
                    g.append(f'<circle class="hit" cx="{f(jx)}" cy="{f(sy(v))}" r="8" data-tip="{esc(tip)}"/>')
    g.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{W - 20}" y2="{bottom}"/>')
    g.append(f'<text class="tick" x="{left}" y="{top - 14}">validation legal move rate</text>')
    label = f"Validation legal move rate of {len(recs)} trained Othello zoo models, grouped by architecture and scale, coloured by training distribution, with the 0.50 admission line."
    rows = sorted([[r["arch"], r["scale"], r.get("distribution", r["config"]["distribution"]), r.get("seed", r["config"]["seed"]),
                    round(r["eval"]["legal_move_rate"], 4)] for r in recs],
                  key=lambda x: (archs.index(x[0]), scales.index(x[1]), dists.index(x[2]), x[3]))
    table = table_view(["Architecture", "Scale", "Distribution", "Seed", "Validation legal move rate"], rows)
    leg = legend([("dot", "dot s1", "synthetic"), ("dot", "dot s2", "championship"), ("dot", "dot s3", "mixed"),
                  ("dash", "threshold", "admission line")])
    return chart(svg(label, W, H, g), caption, leg, table)


def zoo_stats() -> dict[str, str]:
    recs = [json.loads(Path(p).read_text()) for p in glob.glob(str(REPO / "results/zoo_training/*.json"))]
    rates = [r["eval"]["legal_move_rate"] for r in recs]
    by = {}
    for r in recs:
        by.setdefault(r["arch"], []).append(r["eval"]["legal_move_rate"])
    n_othello = sum(r.get("domain", "othello") == "othello" for r in recs)
    status = ("All 81 Othello models have a training record." if n_othello >= 81 else
              f"{n_othello} of the 81 Othello models had a training record when this page was built.")
    return {"zoo-n": str(len(recs)), "zoo-min": f"{min(rates):.3f}", "zoo-max": f"{max(rates):.3f}",
            "zoo-below": str(sum(v < 0.5 for v in rates)), "zoo-status": status,
            "lstm-min": f"{min(by.get('lstm', [0])):.3f}", "lstm-max": f"{max(by.get('lstm', [0])):.3f}",
            "transformer-min": f"{min(by.get('transformer', [0])):.3f}", "transformer-max": f"{max(by.get('transformer', [0])):.3f}",
            "mamba-min": f"{min(by.get('mamba', [0])):.3f}", "mamba-max": f"{max(by.get('mamba', [0])):.3f}"}


def _anchor(diag_folder: str, which: str) -> dict:
    return json.loads((REPO / f"results/{diag_folder}/othello_gpt_{which}.json").read_text())


def chart_d2_lengths(caption: str) -> str:
    syn = _anchor("D2_state_equiv", "synthetic")["details"]
    ch = _anchor("D2_state_equiv", "championship")["details"]
    W, left, panel_h, gap = 780, 56, 110, 70
    lengths = list(range(3, 34))
    pw = W - left - 20
    cw = pw / len(lengths)
    g = []
    for k, (name, d, cls) in enumerate([("Othello-GPT synthetic", syn, "s1"), ("Othello-GPT championship", ch, "s2")]):
        top = 30 + k * (panel_h + gap)
        bottom = top + panel_h
        counts = {int(a): b for a, b in d["same_state_pair_lengths"].items()}
        vmax = 300 if k == 0 else 80
        for t in _ticks(0, vmax, 3):
            y = bottom - panel_h * t / vmax
            g.append(f'<line class="grid" x1="{left}" y1="{f(y)}" x2="{W - 20}" y2="{f(y)}"/>')
            g.append(f'<text class="tick" x="{left - 8}" y="{f(y + 4)}" text-anchor="end">{_fmt(t)}</text>')
        g.append(f'<text class="panel-title" x="{left}" y="{top - 12}">{name}: {d["n_same_state_pairs"]} same-state pairs from {d["games_used"]:,} games</text>')
        for i, L in enumerate(lengths):
            c = counts.get(L, 0)
            if c:
                h = panel_h * c / vmax
                x = left + i * cw + 2
                g.append(f'<path class="{cls}" d="{_col_path(x, bottom, cw - 4, h, 3)}"/>')
                g.append(f'<rect class="hit" x="{f(left + i * cw)}" y="{top}" width="{f(cw)}" height="{panel_h}" '
                         f'data-tip="{esc(name)}: {c} pairs of length {L}"/>')
        g.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{W - 20}" y2="{bottom}"/>')
        for L in (3, 5, 10, 15, 20, 25, 30, 33):
            i = lengths.index(L)
            g.append(f'<text class="tick" x="{f(left + i * cw + cw / 2)}" y="{bottom + 16}" text-anchor="middle">{L}</text>')
    H = 30 + 2 * panel_h + gap + 36
    g.append(f'<text class="tick" x="{W - 20}" y="{H - 4}" text-anchor="end">history length (moves) of the pair</text>')
    rows = [[L, syn["same_state_pair_lengths"].get(str(L), 0), ch["same_state_pair_lengths"].get(str(L), 0)] for L in lengths
            if str(L) in syn["same_state_pair_lengths"] or str(L) in ch["same_state_pair_lengths"]]
    table = table_view(["History length", "Pairs, synthetic anchor", "Pairs, championship anchor"], rows)
    label = ("Number of sampled same-state pairs by history length for the two anchors. Synthetic: lengths 3 to 10, "
             "most at 5 to 7. Championship: lengths 5 to 33, most at 9 to 15.")
    return chart(svg(label, W, H, g), caption, "", table)


def chart_d3_gaps(caption: str) -> str:
    syn = _anchor("D3_inductive_bias", "synthetic")["details"]
    ch = _anchor("D3_inductive_bias", "championship")["details"]
    tasks = list(syn["gap_by_task"])
    W, left, right, top, grp = 760, 130, 70, 30, 52
    vmax = 0.2
    pw = W - left - right
    g = []
    H = top + grp * len(tasks) + 30
    for t in _ticks(0, vmax, 4):
        x = left + pw * t / vmax
        g.append(f'<line class="grid" x1="{f(x)}" y1="{top - 8}" x2="{f(x)}" y2="{H - 28}"/>')
        g.append(f'<text class="tick" x="{f(x)}" y="{H - 10}" text-anchor="middle">{t:.2f}</text>')
    for i, task in enumerate(tasks):
        y0 = top + i * grp
        g.append(f'<text class="row-label" x="{left - 12}" y="{y0 + 22}" text-anchor="end">{task}</text>')
        for j, (name, d, cls) in enumerate([("synthetic", syn, "s1"), ("championship", ch, "s2")]):
            v = d["gap_by_task"][task]
            yy = y0 + 4 + j * 20
            w = pw * v / vmax
            g.append(f'<path class="{cls}" d="{_bar_path(left, yy, w, 16)}"/>')
            g.append(f'<rect class="hit" x="{left}" y="{yy - 2}" width="{pw}" height="20" data-tip="{esc(task)}, {name} anchor: gap {v:.3f}"/>')
            g.append(f'<text class="val" x="{f(left + w + 8)}" y="{yy + 12}">{v:.3f}</text>')
    g.append(f'<line class="axis" x1="{left}" y1="{top - 8}" x2="{left}" y2="{H - 28}"/>')
    g.append(f'<text class="tick" x="{left + pw}" y="{top - 14}" text-anchor="end">balanced accuracy, pretrained minus random weights</text>')
    rows = [[t, round(syn["gap_by_task"][t], 4), round(ch["gap_by_task"][t], 4)] for t in tasks]
    table = table_view(["Task", "Gap, synthetic anchor", "Gap, championship anchor"], rows)
    leg = legend([("bar", "s1", "Othello-GPT synthetic"), ("bar", "s2", "Othello-GPT championship")])
    label = "D3 gap per task for the two anchors: " + "; ".join(f"{t}: {syn['gap_by_task'][t]:.3f} and {ch['gap_by_task'][t]:.3f}" for t in tasks)
    return chart(svg(label, W, H, g), caption, leg, table)


def chart_d4_layers(caption: str) -> str:
    syn = _anchor("D4_probing", "synthetic")["details"]
    ch = _anchor("D4_probing", "championship")["details"]
    W, H, pw, gap, left = 780, 290, 320, 70, 56
    top, bottom = 34, 236
    ylo, yhi = 0.5, 1.0
    g = []
    rows = []
    for k, (name, d, band) in enumerate([("Othello-GPT synthetic", syn, "band-s1"), ("Othello-GPT championship", ch, "band-s2")]):
        ox = left + k * (pw + gap)
        L = d["n_layers"]
        sx = lambda i: ox + pw * i / (L - 1)  # noqa: E731
        sy = lambda v: bottom - (bottom - top) * (v - ylo) / (yhi - ylo)  # noqa: E731
        for t in _ticks(ylo, yhi, 5):
            g.append(f'<line class="grid" x1="{ox}" y1="{f(sy(t))}" x2="{ox + pw}" y2="{f(sy(t))}"/>')
            g.append(f'<text class="tick" x="{ox - 8}" y="{f(sy(t) + 4)}" text-anchor="end">{t:.1f}</text>')
        p, c = d["probe_acc_by_layer"], d["control_acc_by_layer"]
        poly = " ".join(f"{f(sx(i))},{f(sy(v))}" for i, v in enumerate(p)) + " " + \
            " ".join(f"{f(sx(i))},{f(sy(v))}" for i, v in reversed(list(enumerate(c))))
        g.append(f'<polygon class="{band}" points="{poly}"/>')
        g.append(f'<polyline class="line-ink" points="{" ".join(f"{f(sx(i))},{f(sy(v))}" for i, v in enumerate(p))}"/>')
        g.append(f'<polyline class="line-muted" points="{" ".join(f"{f(sx(i))},{f(sy(v))}" for i, v in enumerate(c))}"/>')
        b = d["best_layer"]
        g.append(f'<line class="best" x1="{f(sx(b))}" y1="{f(sy(p[b]))}" x2="{f(sx(b))}" y2="{f(sy(c[b]))}"/>')
        g.append(f'<circle class="line-ink-dot" cx="{f(sx(b))}" cy="{f(sy(p[b]))}" r="4.5"/>')
        g.append(f'<text class="val" x="{f(sx(b) + (8 if b < L - 2 else -8))}" y="{f((sy(p[b]) + sy(c[b])) / 2 + 4)}" '
                 f'text-anchor="{"start" if b < L - 2 else "end"}">selectivity {d["selectivity_by_layer"][b]:.3f}</text>')
        for i in range(L):
            g.append(f'<text class="tick" x="{f(sx(i))}" y="{bottom + 18}" text-anchor="middle">{i}</text>')
            g.append(f'<rect class="hit band-hit" x="{f(sx(i) - pw / (L - 1) / 2)}" y="{top}" width="{f(pw / (L - 1))}" height="{bottom - top}" '
                     f'data-tip="{esc(name)}, layer {i}: probe {p[i]:.3f}, control {c[i]:.3f}, selectivity {d["selectivity_by_layer"][i]:.3f}"/>')
            rows.append([name, i, p[i], c[i], d["selectivity_by_layer"][i], "best" if i == b else ""])
        g.append(f'<line class="axis" x1="{ox}" y1="{bottom}" x2="{ox + pw}" y2="{bottom}"/>')
        g.append(f'<text class="panel-title" x="{ox}" y="{top - 14}">{name} · best layer {b}</text>')
        g.append(f'<text class="tick" x="{ox + pw}" y="{bottom + 36}" text-anchor="end">layer</text>')
    table = table_view(["Anchor", "Layer", "Probe accuracy", "Control accuracy", "Selectivity", ""], rows)
    leg = legend([("line", "line-ink", "probe accuracy"), ("line", "line-muted", "control accuracy"),
                  ("bar", "band-s1 solid", "selectivity, synthetic"), ("bar", "band-s2 solid", "selectivity, championship")])
    label = (f"Per-layer linear probe accuracy and control accuracy on test games for the two anchors. Synthetic: probe "
             f"rises from {syn['probe_acc_by_layer'][0]:.3f} to {max(syn['probe_acc_by_layer']):.3f}, control near "
             f"{syn['control_acc_by_layer'][0]:.2f}, best layer {syn['best_layer']}. Championship: probe peaks at "
             f"{max(ch['probe_acc_by_layer']):.3f}, best layer {ch['best_layer']}.")
    return chart(svg(label, W, H, g), caption, leg, table)


def fig_d3_squares() -> str:
    """The four D3 task squares on a real position, with the true answer to each question."""
    seq = ["D3", "C3", "C4", "E3"]
    b, _ = board_after(seq)
    mover = oth.WHITE
    tasks = ["C4", "F5", "C3", "F6"]
    ox, oy, cell = 44, 40, 34
    g = [board_g(ox, oy, cell, b, coord_size=11, ring=[sq(t) for t in tasks])]
    lx = 360
    g.append(f'<text class="t-mono" x="{lx}" y="58">after {" ".join(seq)}: White just moved</text>')
    g.append(f'<text class="t-muted" x="{lx}" y="80">so “mine” means White’s disc</text>')
    for k, t in enumerate(tasks):
        v = int(b[sq(t)])
        ans = "yes" if v == mover else "no"
        why = "White’s disc" if v == mover else ("Black’s disc" if v == oth.BLACK else "empty")
        y = 122 + k * 34
        g.append(f'<text class="t-strong" x="{lx}" y="{y}">is {t} mine?</text>')
        g.append(f'<text class="t-mono" x="{lx + 120}" y="{y}">{ans}</text>')
        g.append(f'<text class="t-muted" x="{lx + 160}" y="{y}">{why}</text>')
    g.append(f'<text class="t-muted" x="{lx}" y="276">occupied squares are easy to spot from the moves;</text>')
    g.append(f'<text class="t-muted" x="{lx}" y="294">whose disc it is needs every flip since</text>')
    label = ("The four D3 task squares C4, F5, C3 and F6 marked on the board after D3 C3 C4 E3, where White just moved. "
             "Is C4 mine: no, Black's disc. Is F5 mine: no, empty. Is C3 mine: yes. Is F6 mine: no, empty.")
    return svg(label, 680, 330, g)


CHARTS = {
    "throughput": chart_throughput,
    "legal-rates": chart_legal_rates,
    "lr-loss": chart_lr_loss,
    "d2-lengths": chart_d2_lengths,
    "d3-gaps": chart_d3_gaps,
    "d4-layers": chart_d4_layers,
}

FIGS = {
    "start": fig_start, "filmstrip": fig_filmstrip, "rays": fig_rays, "tokenmap": fig_tokenmap,
    "d2-pairs": fig_d2_pairs, "symmetry": fig_symmetry, "d3-squares": fig_d3_squares,
}

HTML_BLOCKS = {
    "build-example": build_example, "bytes-synthetic": bytes_synthetic, "bytes-wthor": bytes_wthor,
}


def zoo_counts() -> dict:
    recs = [json.loads(Path(p).read_text()) for p in glob.glob(str(REPO / "results/zoo_training/*.json"))]
    return {"trained": len(recs)}


def slices_table() -> str:
    rows = list(csv.DictReader((REPO / "src/docs/zoo_slices.csv").open()))
    by = {}
    for r in rows:
        s = by.setdefault(r["slice"], {"owners": [], "n": 0, "min": 0.0, "mamba": 0.0})
        if r["owner"] not in s["owners"]:
            s["owners"].append(r["owner"])
        s["n"] += 1
        s["min"] += float(r["est_minutes"])
        if r["arch"] == "mamba":
            s["mamba"] += float(r["est_minutes"])
    body = "".join(f'<tr><td class="num">{k}</td><td>{", ".join(v["owners"])}</td><td class="num">{v["n"]}</td>'
                   f'<td class="num">{v["min"] / 60:.1f} h</td><td class="num">{100 * v["mamba"] / v["min"]:.0f}%</td></tr>'
                   for k, v in sorted(by.items()))
    return ('<div class="table-scroll"><table class="slice-table"><thead><tr><th scope="col">Slice</th><th scope="col">Owner</th>'
            '<th scope="col">Models</th><th scope="col">Estimated time</th><th scope="col">Share spent on Mamba</th></tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


HTML_BLOCKS["slices"] = slices_table
