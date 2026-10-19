# WM-Concord Course

A guided, interactive course through the [WM-Concord repository](https://github.com/nik-hill-323/fall-2026-group6)
(GWU Data Science Capstone, Fall 2026, Group 6). Fifteen lessons take you from "what is this project
testing?" to every script, file and decision rule in it, with diagrams, code walkthroughs, charts drawn
from the real results, a playable Othello board, and a multiple-choice check at the end of each lesson.

**Read it:** open the hosted site (see the repository's About link), or open `index.html` in any browser.
Everything is in that one file; nothing is installed and nothing leaves your browser. Quiz answers are kept
in your own browser's storage only.

## The lessons

| Part | Lessons |
|---|---|
| A · Orientation | 1 The big picture · 2 How Othello is played |
| B · Building the zoo | 3 The rules engine · 4 The game data · 5 Planning the experiment · 6 Training one model · 7 Training the whole zoo |
| C · Measuring world models | 8 Rules before results · 9 The shared plumbing · 10 D1 next-token validity · 11 D2 state equivalence · 12 D3 inductive bias · 13 D4 representation probing |
| D · Judging the diagnostics | 14 The fragility suite · 15 From scores to answers |

## How it is built

`index.html` is generated. The lessons are plain HTML fragments in `course/lessons/`, and `course/build.py`
fills them with material computed from a checkout of the project repository at build time: code excerpts
with their real line numbers, Othello boards replayed by the project's own rules, and charts drawn from
`src/docs/throughput.md`, the training records and the anchor results. So after the project code changes,
a rebuild keeps every excerpt, line number and chart in step.

```
# from this repository's root, with the project's Python environment (it imports othello.py, so it needs numpy and torch)
python course/build.py --repo ../fall-2026-group6
```

The build refuses to write a page with a leftover placeholder, a duplicate id, a link to a missing anchor,
or a quiz question without exactly one correct answer.

| File | What it holds |
|---|---|
| `course/lessons.py` | The outline: titles, goals, files and reading time of every lesson |
| `course/lessons/NN.html` | The body of lesson NN |
| `course/quizzes.py` | Every multiple-choice question, with an explanation for each option |
| `course/figures.py` | Boards, figures and charts generated from the project |
| `course/template.html` | The page shell: styles, course map, navigation, quizzes, the playable board |
| `course/data/loss_curve.json` | The logged training loss used in Lesson 6 (from a training log, which is not in git) |

Placeholders a lesson can use are listed at the top of `course/build.py`, for example
`<!--CODE:src/zoo/train.py:174:225-->` for a numbered code excerpt.

## Fixing a mistake

Edit the lesson file (or `quizzes.py`), rebuild, check the page, and open a pull request. If a line
reference in a walkthrough no longer matches the code after a project change, the excerpt will show the
new lines and the walkthrough text needs updating to match.

## Hosting

The site is static: `index.html` at the root, no build step on the host.

* **Vercel:** Add New → Project → import this repository → Framework Preset "Other", leave the build
  command empty and the output directory as the root → Deploy. Every push to `main` redeploys.
* **GitHub Pages:** Settings → Pages → Deploy from a branch → `main`, folder `/ (root)`.
