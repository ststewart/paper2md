# Acquiring test fixtures

The `test_files/` directory at the workspace top level holds PDFs
used by paper2md and (later) zmcp2 unit tests + end-to-end smoke
tests. The location is by convention:

    ~/datasets/pdf2md/claude/test_files/

Override via `TEST_FILES_DIR=/path/to/dir` in the environment.

PDFs are **not** in git for any of the code repos. They live outside
the repo trees entirely, so they cannot accidentally be committed
from inside any repo.

## On a fresh machine

To populate `test_files/`:

1. **Copy from another machine** that already has them — easiest if
   you're cloning the workspace fresh on a new host.
2. **Use the canonical sources** listed in
   `~/datasets/pdf2md/claude/test_files/INVENTORY.md`. Each entry
   gives the bibliographic information (DOI / URL / journal) needed
   to obtain the PDF.

The full list of fixtures expected today (~74 MB total when
populated):

```
aastex-template.pdf, agu-template.pdf, mnras-template.pdf,
rhoclass-template.pdf
canup.pdf, Elliott.pdf, jacquet.pdf, Merrill.pdf, ocampo.pdf,
ocampo_sm.pdf
millot.pdf + millot_si.pdf
root-maintext.pdf + root-supplemental.pdf
tracy.pdf + tracy_sm.pdf
wackerle.pdf + wackerle-table2.md + wackerle-table2-v2.md
young-full.pdf
enstatite/  (subdir of journal layouts)
```

## Tests with missing fixtures

`tests/conftest.py` provides a `test_pdf(filename)` fixture that
calls `pytest.skip()` when the requested file isn't present.
Unit tests on a fresh clone with no `test_files/` populated still
pass — only smoke and eval tests skip.

## Smoke-test pattern

```bash
export TEST_FILES_DIR="${TEST_FILES_DIR:-$HOME/datasets/pdf2md/claude/test_files}"
export WORKFLOW_DIR="${WORKFLOW_DIR:-$HOME/datasets/pdf2md/claude/workflow}"

# Pipe a fixture through the marker pipeline:
python src/paper2md.py "$TEST_FILES_DIR/canup.pdf" \
    -o "$WORKFLOW_DIR/md_database/test/canup-marker"

# Or through the MinerU wrapper:
python src/wrap_mineru.py "$TEST_FILES_DIR/canup.pdf" \
    -o "$WORKFLOW_DIR/md_database/test/canup-mineru"
```

The output convention is `$WORKFLOW_DIR/md_database/test/<descriptive-name>/`
— per-developer dev sink that survives reboots, is gitignored
implicitly (it's outside the repo), and shares a layout with
production runs so eval harnesses can A/B compare.
