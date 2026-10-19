"""Build the WM-Concord course into one static page.

    python course/build.py --repo ../fall-2026-group6            # writes index.html at the repo root
    python course/build.py --repo ../fall-2026-group6 --artifact /tmp/course.html

The team repository is read, never written: code excerpts and line numbers come from its files, boards from its
Othello rules, charts from its docs, training records and anchor results. Run it again after the team code
changes and every excerpt, line number and chart follows.

Placeholders a lesson file (course/lessons/NN.html) can use:
    <!--LESSON_HEAD-->               eyebrow and title from lessons.py
    <!--MINUTES-->                   reading time from lessons.py
    <!--FIG:name-->                  a generated figure (figures.FIGS)
    <!--CHART:name|caption html-->  a generated chart with legend, caption and table view (figures.CHARTS)
    <!--HTML:name-->                 a generated block (figures.HTML_BLOCKS)
    <!--CODE:path:start:end-->       lines start to end of a team repo file, numbered and coloured
    <!--QUIZ:7.6-->                  the lesson's multiple-choice check as section 7.6 (quizzes.py)
    <!--STAT:key-->                  a number computed from the team repo (figures.zoo_stats)
    <!--FIGNUM-->                    "Figure N.k.", numbered in order within the lesson
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import figures  # noqa: E402
from lessons import LESSONS, PARTS  # noqa: E402
from quizzes import QUIZZES  # noqa: E402

COURSE_REPO_URL = "https://github.com/Asbetos/wm-concord-course"
TEAM_REPO_URL = "https://github.com/nik-hill-323/fall-2026-group6"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# ==========================================================================
# Code excerpts
# ==========================================================================

KW = {"def", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or", "is", "None", "True",
      "False", "import", "from", "as", "with", "class", "break", "continue", "raise", "lambda", "yield", "try",
      "except", "finally", "pass", "assert", "del", "global", "nonlocal"}
BUILTINS = {"range", "len", "int", "list", "dict", "set", "enumerate", "zip", "print", "sum", "max", "min",
            "isinstance", "float", "str", "sorted", "any", "all", "bool", "tuple", "divmod", "ValueError",
            "TypeError", "FileNotFoundError", "Exception", "super", "hasattr", "getattr", "callable", "round", "abs"}
STRING_START = re.compile(r"([rRbBuUfF]{0,2})(\"\"\"|'''|\"|')")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")
SHELL_KW = {"if", "then", "else", "fi", "for", "do", "done", "while", "continue", "break", "exit", "case", "esac", "in"}


def tokens(src: str, shell: bool = False) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    i, n, prev_word = 0, len(src), ""
    keywords = SHELL_KW if shell else KW
    while i < n:
        c = src[i]
        if c == "#" and (not shell or i == 0 or src[i - 1] in " \t\n"):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(("com", src[i:j]))
            i = j
            continue
        m = STRING_START.match(src, i) if not shell else re.compile(r"()(\"|')").match(src, i)
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            q, j = m.group(2), m.end()
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src.startswith(q, j):
                    j += len(q)
                    break
                if len(q) == 1 and src[j] == "\n" and not shell:
                    break
                j += 1
            out.append(("str", src[i:j]))
            i = j
            continue
        if c.isalpha() or c == "_":
            w = WORD.match(src, i).group(0)
            cls = "kw" if w in keywords else "fn" if prev_word == "def" else "bi" if (w in BUILTINS and not shell) else None
            out.append((cls, w))
            prev_word = w
            i += len(w)
            continue
        if c.isdigit():
            w = NUMBER.match(src, i).group(0)
            out.append(("num", w))
            i += len(w)
            continue
        out.append((None, c))
        i += 1
    return out


def code(repo: Path, rel: str, start: int, end: int) -> str:
    path = repo / rel
    lines = path.read_text().splitlines()
    if end > len(lines):
        raise ValueError(f"{rel} has {len(lines)} lines, asked for {start}-{end}")
    lines = lines[start - 1:end]
    shell = rel.endswith((".sh", ".yml", ".yaml"))
    rows: list[list[str]] = [[]]
    for cls, text in tokens("\n".join(lines), shell=shell):
        for k, part in enumerate(text.split("\n")):
            if k:
                rows.append([])
            if part:
                rows[-1].append(f'<span class="tk-{cls}">{esc(part)}</span>' if cls else esc(part))
    body = "".join(f'<span class="line"><span class="ln">{start + k}</span>{"".join(r)}</span>' for k, r in enumerate(rows))
    return (f'<figure class="code"><figcaption><code>{rel}</code><span>lines {start}–{end}</span></figcaption>'
            f'<pre><code>{body}</code></pre></figure>')


# ==========================================================================
# Quizzes
# ==========================================================================

def quiz(n: int, num: str) -> str:
    qid = f"l{n}"
    qs = QUIZZES[qid]
    p = [f'<section class="section" id="{qid}-check"><h3><span class="num">{num}</span>Check yourself</h3>',
         f'<div class="quiz" data-quiz="{qid}"><div class="quiz-bar"><p class="quiz-score" aria-live="polite">'
         f'{len(qs)} questions. Pick an answer to see why it is right or wrong.</p>'
         '<button type="button" class="quiz-reset">Clear my answers</button></div>']
    for i, (stem, options) in enumerate(qs, 1):
        key = f"{qid}q{i}"
        if sum(c for _, _, c in options) != 1 or len(options) != 4:
            raise ValueError(f"{key}: needs 4 options with exactly one correct")
        p.append(f'<fieldset class="mcq" data-q="{key}"><legend><span class="qn">Q{i}</span>'
                 f'<span class="stem">{esc(stem)}</span></legend><div class="opts">')
        for j, (text, why, correct) in enumerate(options):
            p.append(f'<button type="button" class="opt"{" data-correct" if correct else ""} aria-describedby="{key}-fb">'
                     f'<span class="letter">{"ABCD"[j]}</span><span class="otext">{esc(text)}</span>'
                     f'<span class="why" hidden>{esc(why)}</span></button>')
        p.append(f'</div><p class="feedback" id="{key}-fb" aria-live="polite"></p></fieldset>')
    p.append("</div></section>")
    return "".join(p)


# ==========================================================================
# Page assembly
# ==========================================================================

def course_map() -> str:
    out = []
    for part, name in PARTS.items():
        items = []
        for les in (x for x in LESSONS if x["part"] == part):
            n = les["n"]
            items.append(
                f'<li class="lesson-item"><a href="#lesson-{n}" data-lesson-link><span class="no">{n:02d}</span>'
                f'<span><span class="name">{esc(les["title"])}</span><span class="goal">{esc(les["goal"])}</span>'
                f'<span class="files">{esc(les["files"])}</span></span><span class="pill" data-pill="{n}"></span></a></li>')
        out.append(f'<div class="part"><span class="eyebrow">Part {part} · {esc(name)}</span><ol>{"".join(items)}</ol></div>')
    return "\n".join(out)


def next_card(n: int) -> str:
    if n < len(LESSONS):
        nxt = LESSONS[n]
        return (f'<aside class="next" aria-label="Next lesson"><p class="eyebrow">Next · Lesson {nxt["n"]}</p>'
                f'<h4>{esc(nxt["title"])}</h4><p>{esc(nxt["goal"])}</p>'
                f'<a class="next-link" href="#lesson-{nxt["n"]}">Go to Lesson {nxt["n"]} →</a></aside>')
    return ('<aside class="next" aria-label="End of the course"><p class="eyebrow">End of the course</p>'
            '<h4>You have read every file that matters</h4><p>From here the repository should read like a map you know. '
            'Revisit any lesson from the course map, or go back to the start.</p>'
            '<a class="next-link" href="#lesson-1">Back to Lesson 1 →</a></aside>')


def render_lesson(repo: Path, les: dict, stats: dict[str, str]) -> str:
    n = les["n"]
    src = (HERE / "lessons" / f"{n:02d}.html").read_text()
    head = (f'<p class="eyebrow">Part {les["part"]} · {esc(PARTS[les["part"]])} · Lesson {n}</p>'
            f'<h2>{esc(les["title"])}</h2>')
    src = src.replace("<!--LESSON_HEAD-->", head).replace("<!--MINUTES-->", str(les["minutes"]))
    src = re.sub(r"<!--FIG:([\w-]+)-->", lambda m: figures.FIGS[m.group(1)](), src)
    src = re.sub(r"<!--CHART:([\w-]+)\|(.*?)-->", lambda m: figures.CHARTS[m.group(1)](m.group(2).strip()), src, flags=re.S)
    src = re.sub(r"<!--HTML:([\w-]+)-->", lambda m: figures.HTML_BLOCKS[m.group(1)](), src)
    src = re.sub(r"<!--CODE:([\w/.-]+):(\d+):(\d+)-->", lambda m: code(repo, m.group(1), int(m.group(2)), int(m.group(3))), src)
    src = re.sub(r"<!--QUIZ:([\d.]+)-->", lambda m: quiz(n, m.group(1)), src)
    src = re.sub(r"<!--STAT:([\w-]+)-->", lambda m: stats[m.group(1)], src)
    counter = iter(range(1, 100))
    src = re.sub(r"<!--FIGNUM-->", lambda m: f"Figure {n}.{next(counter)}.", src)
    title = f"Lesson {n} · {les['title']}"
    return (f'<article class="lesson" id="lesson-{n}" data-title="{html.escape(title, quote=True)}">\n{src}\n'
            f'{next_card(n)}\n</article>')


def build_info(repo: Path) -> str:
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    today = dt.datetime.now(dt.timezone.utc).strftime("%-d %B %Y")
    n_records = figures.zoo_counts()["trained"]
    return (f'<p>Built on {today} from <a href="{TEAM_REPO_URL}">nik-hill-323/fall-2026-group6</a> at commit '
            f'<code>{commit}</code>, with {n_records} training records present. Boards, token numbers, code excerpts, '
            f'line numbers and charts are generated from that code, so they match it exactly.</p>'
            f'<p>Course source and rebuild instructions: <a href="{COURSE_REPO_URL}">Asbetos/wm-concord-course</a>. '
            f'The team’s own draw.io diagrams for D1–D4 are in <code>reports/diagrams/</code> of the project repository.</p>')


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="A guided course through the WM-Concord repository (GWU Data Science Capstone, Group 6): every stage, every script, and how the code works.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%231b6544'/%3E%3Ccircle cx='11' cy='11' r='6' fill='%23fbfbf8'/%3E%3Ccircle cx='21' cy='11' r='6' fill='%23111814'/%3E%3Ccircle cx='11' cy='21' r='6' fill='%23111814'/%3E%3Ccircle cx='21' cy='21' r='6' fill='%23fbfbf8'/%3E%3C/svg%3E">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def check(page: str) -> None:
    left = re.findall(r"<!--(?:LESSON_HEAD|MINUTES|FIG|CHART|HTML|CODE|QUIZ|STAT|FIGNUM|TABLE)[^>]*-->", page)
    if left:
        raise SystemExit(f"unreplaced placeholders: {left[:5]}")
    ids = re.findall(r'\sid="([^"]+)"', page)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"duplicate ids: {dupes[:10]}")
    missing = sorted({h for h in re.findall(r'href="#([\w.-]+)"', page)} - set(ids))
    if missing:
        raise SystemExit(f"links to missing anchors: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="path to a checkout of nik-hill-323/fall-2026-group6")
    ap.add_argument("--out", default=str(ROOT / "index.html"))
    ap.add_argument("--artifact", default=None, help="also write the page without the document wrapper")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    figures.init(repo)
    stats = figures.zoo_stats()
    tpl = (HERE / "template.html").read_text()
    head, body = tpl.split("<!--BODY-->")
    body = body.replace("<!--COURSE_MAP-->", course_map())
    body = body.replace("<!--PROGRESS-->", "".join(f'<span class="sq" data-square="{x["n"]}"></span>' for x in LESSONS))
    body = body.replace("<!--LESSONS-->", "\n".join(render_lesson(repo, les, stats) for les in LESSONS))
    body = body.replace("<!--BUILD_INFO-->", build_info(repo))
    check(head + body)
    Path(a.out).write_text(STANDALONE_HEAD + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    print(f"wrote {a.out} ({len(head + body) / 1024:.0f} KB, {len(LESSONS)} lessons)")
    if a.artifact:
        Path(a.artifact).write_text(head + body)
        print(f"wrote {a.artifact}")


if __name__ == "__main__":
    main()
