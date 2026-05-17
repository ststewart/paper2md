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

The shape of a release is: **dev commits → transfer dir → GitHub push
→ Zenodo DOI → DOI paste → amend + tag → tag push → GitHub release →
verify.**

Plan an hour for §2.5–2.9 (Zenodo + DOI plumbing); the rest is fast.

### 2.1 🤖 CLAUDE — Bump the version + tests in the DEV repo

In `/home/sts/datasets/pdf2md/claude/paper2md/`, edit:

- `src/paper2md.py`: `__version__ = "0.X.Y"` (also the line in the
  module docstring header)
- `src/paper2md.py`: `__doi__ = "10.5281/zenodo.RESERVED"`
  (placeholder until §2.5)
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

No tag yet. We tag in §2.6 inside the TRANSFER dir, not the dev
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

Initialize fresh git history (single root commit):

```bash
git init
git add -A
git commit -m "paper2md v0.X.Y — initial public release"
# (For releases after the first: keep the v0.X.0 root style but you
# can mention what's new in the commit body.)
```

### 2.5 🧑 SARAH — Reserve a Zenodo DOI

This is the load-bearing step. Reserve the DOI Zenodo would mint for
the next release, paste it into the source, THEN tag — so the tag,
the published release, and the DOI all match.

1. Go to [Zenodo](https://zenodo.org/) and sign in with the same
   GitHub account you used in §1.4.
2. Click **+ Upload** in the top-right (or **New upload**).
3. Choose the **Software** resource type.
4. Click **Reserve DOI**. Zenodo shows a DOI like
   `10.5281/zenodo.12345678` and assigns it to this draft.
5. Copy the bare identifier — `10.5281/zenodo.12345678`, no
   protocol prefix, no trailing slash. **Don't publish the draft
   yet.** Just leave the form open in a tab; the GitHub integration
   takes over in §2.8.

**Why we abandon the manual draft**: the GitHub integration creates
its own deposit when it sees a new release, and it'll claim the
reserved DOI as long as the DOI embedded in the release matches one
Zenodo has already reserved for your account. If you publish the
manual draft, you'll end up with two deposits. (See §4 if this
happens.)

Hand the DOI back to Claude.

### 2.6 🤖 CLAUDE — Paste DOI into source, amend transfer commit, tag

In the **transfer directory**:

```bash
cd paper2md_transfer_v0XY
# Edit src/paper2md.py: __doi__ = "10.5281/zenodo.12345678"
git add src/paper2md.py
git commit --amend --no-edit
git tag -a v0.X.Y -m "paper2md v0.X.Y"
```

Also apply the same DOI paste to the **dev repo's**
`src/paper2md.py:__doi__` and commit as `Wire v0.X.Y Zenodo DOI` —
keeps the dev tree's banner in sync with what the public archive shows.

Smoke check:

```bash
python src/paper2md.py --help | head -3
```

The third line should now read:

```
Cite: Stewart, S. T., & Claude (Anthropic, Opus 4.7). (2026). paper2md (v0.X.Y) [Software]. MIT License. https://doi.org/10.5281/zenodo.12345678
```

Frontmatter / `.meta.json` outputs from this version onwards will
also carry `paper2md_doi: "10.5281/zenodo.12345678"`.

### 2.7 🧑 SARAH — Push main and the tag

In the **transfer directory**:

```bash
cd paper2md_transfer_v0XY
git push --force origin main    # force needed: amended in §2.6
git push origin v0.X.Y          # tag pushes are independent
```

Force-push to a single-commit branch that only you control is safe.
(If this is the FIRST release per §1.3, replace `--force` with
`-u origin main`.)

### 2.8 🧑 SARAH — Create the GitHub release

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
[Zenodo](https://zenodo.org/) and you should see the new deposit
appear with the DOI you reserved in §2.5.

### 2.9 🧑 SARAH — Verify the DOI resolves

Open `https://doi.org/10.5281/zenodo.12345678` in a browser. It
should redirect to your Zenodo deposit page, which lists the release
archive and metadata from `CITATION.cff`.

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
reserved.** The reserved DOI was probably consumed by an unrelated
deposit between when you reserved it and when the GitHub release
fired. Edit the deposit on Zenodo to discard it (you can within the
first 24 h), reserve a new DOI on Zenodo, hand it to Claude to paste
into `__doi__`, amend + force-push the tag, recreate the GitHub
release. Force-push of a tag is fine here because v0.X.Y was never
visible to anyone outside your account.

**GitHub release published but Zenodo never picked it up.** Check
that the GitHub-Zenodo integration is still ON for this repo at
[zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/).
A toggle that flipped off (e.g., due to a re-auth) will silently
skip the release. After re-enabling, re-trigger by clicking
"Re-run integration" or by deleting and recreating the GitHub
release.

**Two Zenodo deposits with the same release.** You probably
published the manual draft from §2.5 instead of letting the GitHub
integration claim it. The first deposit can be deleted by you
within 24 h via Zenodo's UI; after that, it requires emailing
Zenodo support.

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

Zenodo mints two DOIs once your project has at least one published
release:

- **Concept DOI** (e.g. `10.5281/zenodo.99999998`): always points at
  "all versions of paper2md". Doesn't change across releases.
  Useful as a permanent reference when you don't care which
  version was used.
- **Version DOI** (e.g. `10.5281/zenodo.99999999`): points at one
  specific release. Reserved + embedded in `__doi__` per release.

paper2md's `__doi__` constant carries the **version DOI** because
that's what reproducibility requires — a consumer of the output
markdown six months from now needs to know which exact release
produced it. The concept DOI lives in `CITATION.cff` (if you set
that up) and on the Zenodo page itself.

When citing paper2md in a paper, use the version DOI. When linking
from your CV or another project's README, the concept DOI is fine.

---

## 6. Sequencing summary

```
§2.1  🤖 CLAUDE  bump __version__, __doi__="RESERVED", doc version stamps in DEV repo
§2.2  🤖 CLAUDE  pytest -q + --help smoke
§2.3  🤖 CLAUDE  git commit (split: feature-per-commit + final "Bump to v0.X.Y")
§2.4  🤖 CLAUDE  build paper2md_transfer_v0XY/; verify tests; git init + single root commit
§2.5  🧑 SARAH   Zenodo → New upload → Software → Reserve DOI → copy → hand to Claude
§2.6  🤖 CLAUDE  paste DOI into transfer/src/paper2md.py, amend, tag v0.X.Y
                 also paste into dev/src/paper2md.py, commit "Wire v0.X.Y Zenodo DOI"
§2.7  🧑 SARAH   cd transfer; git push --force origin main; git push origin v0.X.Y
§2.8  🧑 SARAH   GitHub → Releases → new release from v0.X.Y tag → Publish
§2.9  🧑 SARAH   open https://doi.org/<your-DOI> → verify Zenodo archived
```

If you can't follow §2.5 first (e.g. Zenodo is down), the fallback is
to publish without the DOI, then commit a follow-up that populates
`__doi__` once Zenodo's webhook fires. Single-commit sequencing is
the goal but not load-bearing — paper2md runs fine with the
placeholder.
