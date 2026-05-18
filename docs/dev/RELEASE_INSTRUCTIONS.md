# Release instructions: paper2md → GitHub → Zenodo

Use this whenever you cut a new version. The order of events is the
load-bearing detail: a small sequencing mistake forces a churn commit
to fix the citation. Follow it top-to-bottom and the citation in the
source code stays canonical.

The `__doi__` constant in `src/paper2md.py` is the single point of
truth for the per-version Zenodo DOI. Reserve the DOI on Zenodo
*before* tagging, paste it into `__doi__`, then tag and push. The
GitHub-Zenodo integration publishes the release with the DOI you
already reserved, so no follow-up commit is needed.

## Handoff convention

Each step is marked with who acts:

- **🧑 SARAH** — runs in browser or shell; involves github.com / zenodo.org / external accounts
- **🤖 CLAUDE** — local-only: edits files, runs `git commit` / `git tag` / `git remote add` in the local repo, runs tests

**Claude never pushes to GitHub.** All `git push`, `gh ...`, Zenodo
clicks, and PyPI uploads are Sarah's. See memory
`feedback_no_github_pushes`.

> **One-time setup** (skip on later releases): §1.

---

## 1. One-time setup

Done once for the lifetime of the project. After this section, every
release follows §2.

### 1.1 🧑 SARAH — Create the GitHub repository

Pick a name (suggested: `paper2md`). Create it under your GitHub
account or org through the GitHub web UI. Make it **public** —
Zenodo only archives public repos.

Do NOT initialize with a README, .gitignore, or LICENSE on GitHub
— the transfer repo already has all three, and an init would
create a non-empty remote that's harder to push to.

Copy the empty repo URL (HTTPS or SSH). Hand back to Claude — needed
in §1.2.

### 1.2 🤖 CLAUDE — Wire the local transfer repo to the GitHub remote

From the transfer directory (e.g. `paper2md_transfer_v040/`):

```bash
git remote add origin <url-from-step-1.1>
git remote -v   # sanity check: should print `origin   <url>` twice
```

### 1.3 🧑 SARAH — Push the initial release

```bash
cd paper2md_transfer_v040
git push -u origin main
```

The `-u` makes `main` track `origin/main` so future pushes are just
`git push`. Tag pushes happen separately in §2.7.

### 1.4 🧑 SARAH — Connect Zenodo to GitHub

Go to [Zenodo's GitHub integration page](https://zenodo.org/account/settings/github/).
Sign in with your GitHub account. You'll see a list of your
repositories. Find `paper2md` and **flip its toggle to ON**. From
that moment on, every GitHub *release* (created from a tag) on this
repo will automatically be archived to Zenodo with a fresh version
DOI.

You don't have to publish a release to test the integration — the
toggle alone is enough; the first release you create afterwards
will be archived.

### 1.5 🤖 CLAUDE — `CITATION.cff` is already in the transfer dir

The transfer dir's `CITATION.cff` is built from the dev repo's
canonical copy. Updated `version:` and `date-released:` for every
release; nothing else changes.

---

## 2. Per-release workflow (this is what you'll do every time)

The shape of a release is: **dev commits → transfer dir (tagged with
DOI placeholder) → push main + tag → GitHub release → Zenodo mints
DOI → patch dev follow-up.**

Plan ~30 min total. The Zenodo handshake (§2.6 → DOI appears) takes
a couple of minutes; everything else is fast.

### 2.1 🤖 CLAUDE — Bump the version + tests in the DEV repo

In `/home/sts/datasets/pdf2md/claude/paper2md/`, edit:

- `src/paper2md.py`: `__version__ = "0.X.Y"` (also the line in the
  module docstring header)
- `src/paper2md.py`: `__doi__ = "10.5281/zenodo.RESERVED"`
  (placeholder; patched after Zenodo mints in §2.7. Carrying the
  previous version's DOI here briefly would be wrong — it'd be
  archived alongside the new version.)
- `pyproject.toml`: `version = "0.X.Y"`
- `CITATION.cff`: `version: 0.X.Y` + `date-released: <today>`
- `README.md`: footer "paper2md v0.X.Y (Month YYYY)"
- `docs/dev/CLAUDE.md`: "Version: 0.X.x" header
- `docs/comparisons/COMPARISON_DOCLING.md`: capability matrix header
- `docs/comparisons/COMPARISON_PYMUPDF4LLM.md`: capability matrix header
- `docs/comparisons/ALTERNATIVES_NO_CUDA.md`: "public anchor is the v0.X.0 release" line
- `docs/design/FLOWCHART.md`: "(v0.X.0)" intro line

(Do NOT auto-bump version stamps in `docs/comparisons/COMPARISON_MINERU.md`
or `COMPARISON_PYMUPDF4LLM.md`'s "verified on canup.pdf" section —
those are tied to the version the benchmark actually ran on.)

Run `grep -rnF "v0.<previous>" .` after — should hit exactly the
lines you want and nothing else.

### 2.2 🤖 CLAUDE — Test locally

```bash
cd /home/sts/datasets/pdf2md/claude/paper2md
python -m pytest tests/ -q     # all tests must pass
python src/paper2md.py --help | head -3   # banner shows new version
```

The runtime banner reads `paper2md v0.X.Y (MIT License)`. The cite
line still omits the DOI URL (placeholder) — that's expected.

### 2.3 🤖 CLAUDE — Commit the bump + any new features in the DEV repo

Per `feedback_commit_after_each_task`: split into one commit per
logical change (new feature, bug fix, doc edit), with the version
bump as its own final commit `Bump to v0.X.Y`. Don't accumulate
multi-task diffs into one mega-commit.

No tag yet. We tag in §2.4 inside the TRANSFER dir, not the dev
repo. (The dev repo carries the development history; the transfer
repo carries the public-facing v0.X.Y root commit.)

### 2.4 🤖 CLAUDE — Build the v0.X.Y transfer directory

Create `/home/sts/datasets/pdf2md/paper2md_transfer_v0XY/` alongside
the prior transfer dir(s). Copy from the dev tree, **exclude**:

- `.git/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`
- `workflow/`, `collections/`, anything corpus-private
- `.env`, any local config
- Host-specific test fixtures referenced by `$TEST_FILES_DIR`

**Include** (the public surface):

- `src/` (all `.py` files)
- `tests/` (full suite + `figure_match_truth/` + `table_extract_truth/`)
- `docs/` (USAGE.md, BATCH.md, design/, dev/, setup/, comparisons/)
- `examples/` (README pointer only; no PDFs)
- `README.md`, `LICENSE`, `CITATION.cff`, `CLAUDE.md`
- `environment-mac.yml`, `environment-gpu.yml`, `pyproject.toml`
- `.gitignore`

Verify:

```bash
cd paper2md_transfer_v0XY
python -m pytest tests/ -q     # same count as dev repo
```

Initialize fresh git history (single root commit) and tag:

```bash
git init -b main
git config user.name "Sarah T. Stewart"
git config user.email "sstewa56@asu.edu"
git add -A
git commit -m "paper2md v0.X.Y — initial public release"
git tag -a v0.X.Y -m "paper2md v0.X.Y"
git remote add origin git@github.com:ststewart/paper2md.git
```

> **Note**: `__doi__` stays at the `"10.5281/zenodo.RESERVED"`
> placeholder for now. Zenodo assigns the actual DOI when the
> GitHub release is published in §2.6, and we patch the dev tree
> in §2.7. The transfer commit's `__doi__` will remain RESERVED
> in the archived tarball — this is a documented cosmetic scar
> (see §2.7 note).

### 2.5 🧑 SARAH — Push main and the tag

In the **transfer directory**:

```bash
cd paper2md_transfer_v0XY
git push -u origin main          # first release: use -u to set upstream
git push origin v0.X.Y           # tag push is independent
```

For releases AFTER the first, use `git push --force origin main` only
if the root commit was amended for some other reason (rare).

> **Why the DOI is NOT pre-reserved.** Earlier versions of these
> instructions had a "reserve DOI before tagging" step (Zenodo
> Upload → Reserve DOI → paste into `__doi__` → tag). **That flow
> does not work with the current Zenodo InvenioRDM integration**:
> the GitHub-Zenodo webhook creates a fresh deposit with a fresh
> DOI on every release, ignoring previously-reserved drafts. The
> reserved draft just sits unpublished on your dashboard, and
> `__doi__` ends up pointing at a DOI that never resolves to
> anything. We tag with `__doi__ = "RESERVED"` placeholder and
> patch in a dev follow-up after Zenodo assigns the real DOI.

### 2.6 🧑 SARAH — Create the GitHub release (Zenodo assigns DOI here)

The Zenodo integration only archives **GitHub releases**, not raw
tags. Go to:

```
https://github.com/<your-username>/paper2md/releases/new
```

- **Choose a tag**: pick the `v0.X.Y` you just pushed.
- **Release title**: `paper2md v0.X.Y`.
- **Description**: Claude can draft this for you; paste it in.
  Bullet list of highlights is fine.
- Click **Publish release**.

Within a minute or two, Zenodo's webhook fires. Refresh
[Zenodo](https://zenodo.org/) and a new deposit appears with a
freshly-minted DOI of the form `10.5281/zenodo.NNNNNNNN`. Copy
that bare identifier and hand it to Claude for §2.7.

### 2.7 🤖 CLAUDE — Wire the assigned DOI into dev source

In the **dev repo only** (NOT the transfer dir — the transfer's
v0.X.Y root commit is already on Zenodo and immutable):

```bash
cd /home/sts/datasets/pdf2md/claude/paper2md
# Edit src/paper2md.py: __doi__ = "10.5281/zenodo.NNNNNNNN"
# Edit CITATION.cff: add `doi: 10.5281/zenodo.NNNNNNNN`
git add src/paper2md.py CITATION.cff
git commit -m "Wire v0.X.Y Zenodo DOI"
```

Smoke check:

```bash
python src/paper2md.py --help | head -3
```

The third line should now read:

```
Cite: Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). paper2md (v0.X.Y) [Software]. MIT License. https://doi.org/10.5281/zenodo.NNNNNNNN
```

Frontmatter / `.meta.json` outputs from this version onwards will
also carry `paper2md_doi: "10.5281/zenodo.NNNNNNNN"`.

> **Known cosmetic scar**: the v0.X.Y tarball archived on Zenodo
> still has `__doi__ = "10.5281/zenodo.RESERVED"` inside the
> source — because the DOI didn't exist when we tagged. Anyone
> who downloads that tarball and runs `--help` sees the no-DOI
> citation banner. The Zenodo deposit page itself shows the
> correct DOI and the `CITATION.cff` on GitHub's main branch is
> correct. Acceptable tradeoff; the alternative is a v0.X.Y+1
> point release just to bake the DOI in.

### 2.8 🧑 SARAH — Verify the DOI resolves

Open `https://doi.org/10.5281/zenodo.NNNNNNNN` in a browser. It
should redirect to your Zenodo deposit page, which lists the
release archive and metadata from `CITATION.cff`.

If the DOI 404s, the GitHub-Zenodo handshake didn't complete — see
§4.

---

## 3. What lives where

| Constant / file | Purpose |
|---|---|
| `src/paper2md.py:__version__` | Single source of truth for the version string. Read into the runtime banner, the YAML frontmatter (`run.paper2md_version`), and the `.meta.json` sidecar. |
| `src/paper2md.py:__doi__` | Per-version Zenodo DOI. Placeholder `"10.5281/zenodo.RESERVED"` means no DOI is wired yet; the runtime falls back to a no-DOI citation when it sees the placeholder. Update once per release between §2.5 and §2.6. |
| `src/paper2md.py:__citation__` / `__citation_with_doi__` | Two format strings; the runtime picks the DOI variant when `__doi__` is set, the no-DOI variant otherwise. **Do not edit these per release** — the version and DOI substitute in. |
| `CITATION.cff` | Zenodo / GitHub citation metadata. Update `version` and `date-released` per release. |
| Output frontmatter `run.paper2md_doi` | Carried in every paper2md-produced markdown / `.meta.json` once the DOI is wired. Omitted until then. |

---

## 4. Troubleshooting

**Zenodo deposit appears but with a different DOI than the one you
reserved.** This is the **expected** behavior of the current Zenodo
InvenioRDM integration — DOI reservations are NOT claimed by the
GitHub webhook. Don't try to reserve in advance. Use the §2.5-2.8
flow (tag + release first, patch `__doi__` in a follow-up dev commit
after Zenodo assigns the real DOI). If you have an old unpublished
draft sitting on your Zenodo dashboard from a reservation attempt,
delete it (top of the draft page → Delete / Discard; unpublished
drafts are self-deletable).

**GitHub release published but Zenodo never picked it up.** Check
that the GitHub-Zenodo integration is still ON for this repo at
[zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/).
A toggle that flipped off (e.g., due to a re-auth) will silently
skip the release. After re-enabling, re-trigger by clicking
"Re-run integration" or by deleting and recreating the GitHub
release.

**Two Zenodo deposits with the same release.** Likely cause: you
hit "Publish" on a manually-created Zenodo draft AND the GitHub
integration also published. The manual deposit can be deleted by
you within 24 h via Zenodo's UI if unpublished; once published,
contact Zenodo support. The integration-created one is the canonical
archive — keep that, delete the other.

**Runtime banner still says no-DOI after pasting `__doi__`.** The
fallback condition checks `__doi__.endswith("RESERVED")` — make
sure you pasted the actual DOI and didn't leave the placeholder
intact. `git diff src/paper2md.py | grep -A1 __doi__` will show.

**Tests fail in the transfer dir but pass in the dev repo.** Most
likely cause: the copy-out missed a file (a new test fixture, a new
source module). Re-run §2.4 copy step; or run
`diff -r src/ ../paper2md_transfer_v0XY/src/` to spot the gap.

---

## 5. Concept DOI vs version DOI

Zenodo mints two DOIs for every project at first release:

- **Concept DOI** (paper2md: `10.5281/zenodo.20263035`): always
  points at "all versions of paper2md". Doesn't change across
  releases. Always resolves to the latest. Found on any deposit's
  "Versions" panel ("Cite all versions? ... use the DOI ...").
- **Version DOI** (paper2md v0.4.0: `10.5281/zenodo.20263036`):
  points at one specific release archive. New value every release.

paper2md's wiring:

- **`src/paper2md.py:__doi__`** = version DOI. Reproducibility
  requires it — a consumer of the output markdown six months from
  now needs to know which exact release produced it. Re-wired in
  §2.7 of this doc after each release.
- **`CITATION.cff`** carries both: `doi:` field = version DOI,
  `identifiers:` block = concept DOI tagged "description: Concept
  DOI (all versions...)". Reference managers and citation tools
  pick the right one for context.
- **`README.md` badge** uses the concept DOI so the badge stays
  green and points at the latest release regardless of which
  version of the README a reader has.

When citing paper2md in a paper, use the version DOI. When linking
from your CV, another project's README, or any general "see
paper2md" reference, the concept DOI is the right choice.

The concept DOI is set once at v0.X.0 and never changes — no
per-release maintenance for it.

---

## 6. Sequencing summary

```
§2.1  🤖 CLAUDE  bump __version__, leave __doi__="RESERVED" placeholder, doc version stamps in DEV repo
§2.2  🤖 CLAUDE  pytest -q + --help smoke
§2.3  🤖 CLAUDE  git commit (split: feature-per-commit + final "Bump to v0.X.Y")
§2.4  🤖 CLAUDE  build paper2md_transfer_v0XY/; verify tests; git init + single root commit + tag v0.X.Y
§2.5  🧑 SARAH   cd transfer; git push -u origin main; git push origin v0.X.Y
§2.6  🧑 SARAH   GitHub → Releases → new release from v0.X.Y tag → Publish
                 → Zenodo assigns DOI within ~2 min → copy from deposit page → hand to Claude
§2.7  🤖 CLAUDE  paste DOI into dev/src/paper2md.py + dev/CITATION.cff; commit "Wire v0.X.Y Zenodo DOI"
                 (transfer dir is NOT updated -- its tarball is already on Zenodo and immutable)
§2.8  🧑 SARAH   open https://doi.org/<your-DOI> → verify Zenodo archived
```

The DOI flow is fundamentally asymmetric: Zenodo only mints the
DOI when the release is published, so the tagged source can't
contain its own DOI. We accept the cosmetic scar (archived
tarball's `__doi__` says RESERVED) in exchange for not needing a
v0.X.Y+1 point release to bake the DOI in.
