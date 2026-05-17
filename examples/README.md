# Examples

PDF fixtures used by paper2md's end-to-end smoke tests live at the
workspace level under `~/datasets/pdf2md/claude/test_files/`, not
inside this directory. They are not tracked in git for any code
repo (different copyright statuses; same workflow regardless).

This `examples/` directory is intentionally kept as a stub so docs
and scripts that reference it don't have to be rewritten if the
convention changes again.

To run smoke tests:

```bash
# Tests use the test_pdf fixture (see tests/conftest.py) which reads
# $TEST_FILES_DIR (default: ~/datasets/pdf2md/claude/test_files/).
python -m pytest tests/

# End-to-end smoke against a fixture PDF:
python src/paper2md.py "$TEST_FILES_DIR/canup.pdf" \
    -o "$WORKFLOW_DIR/md_database/test/canup-smoke"
```

See [`docs/dev/EXAMPLES_FIXTURES.md`](../docs/dev/EXAMPLES_FIXTURES.md)
for the full convention and how to populate `test_files/` on a
fresh machine.
