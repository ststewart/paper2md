# Hybrid layout — implementation notes and known "fudges"

`--layout-source=hybrid` splices MinerU's figure/table layout into marker's body markdown. The base splice is structural and predictable, but several real-world MinerU oddities are papered over with targeted fudges. **Every fudge is a candidate to revisit when MinerU's version changes.**

Captured 2026-05-12. Updated 2026-05-15 (conservative-splice refactor). Current pinned MinerU: **3.1.7**.

## 2026-05-15 refactor: conservative splice, sidecar-first

The hybrid splice no longer **replaces** marker's pipe-md table content. The mutation surface is now purely additive:

- **Matched tables** (marker caption + MinerU number both present): the splice INSERTS `<image link> + [Table N — separate markdown](sidecar.md)` immediately after marker's natural table block. Marker's pipe-md stays verbatim. The high-quality VLM-rewritten table goes to the per-table `.md` sidecar in `assets/` — that's the primary product, intended for a future MCP hook to consume.
- **Unmatched tables** (MinerU detected but no marker caption to anchor): the splice writes the sidecar but does NOT insert inline. Every unmatched sidecar is collected into a single `## Extracted tables` section that gets PREPENDED at the START of the body (right after the frontmatter).

**Why start-of-doc and not end-of-doc**: three critical code paths assume the references section is the LAST section in the body (`_detect_refs_section` at L5432, the fallback cluster at L5492, `_swap_hybrid_refs_with_mineru` at L3167). An end-of-doc `## Extracted tables` block would confuse refs scoring, journal-rescues, and the Crossref/OpenAlex API fallback. Prepending sidesteps every refs-walks-from-end hook.

**Why marker's pipe-md is preserved verbatim**: marker's content can have surya OCR errors but is never CORRUPTED by the splice itself. Replacing it added a failure surface (wrong anchor → wrong region → marker's content lost or duplicated). With conservative-splice, this surface is gone. The high-quality version is one click away in the sidecar.

**New counters** (replace `tables_swapped` in `rescues.hybrid_splice`):
- `tables_linked_inline` — matched marker caption; sidecar link inserted next to it
- `tables_appended_index` — no marker anchor; sidecar link in start-of-doc `## Extracted tables`

**Obsolete-by-design fudges**:
- The `_find_marker_table_region` extend-across-blank-lines loop (was: catch surya emitting the same table twice; the splice would consume both chunks). Still present but no longer load-bearing — the splice never replaces, only uses `region_end` as the insert position.
- The HTML-fallback inline emission (was: drop raw HTML into body when pipe-md conversion fails). Now suppressed under `inline_in_body=False`; only the image link is emitted, marker's pipe-md remains the body's table content.

**Still-relevant fudges**: every entry in the TL;DR map below remains (image-link strip, refs swap, panel-prefix tolerance, etc.). They operate on the splice's image/caption mutations, which still happen — just not on table BODY content.

## TL;DR map

| Fudge | Code | Counter | Last touched | What breaks it |
|---|---|---|---|---|
| Marker image-link strip | `_strip_marker_image_links` (`paper2md.py`) | `marker_image_links_removed` | `21f1010` | Marker stops emitting `_page_N_*` filenames OR starts wrapping images in something other than `<span id>...</span>![](...)` |
| MinerU references swap | `_swap_hybrid_refs_with_mineru` (`paper2md.py`) | `mineru_refs_swapped` | `623c538` | MinerU's references heading style changes OR MinerU output stops segmenting refs as a heading-prefixed section |
| Nature `**Figure N**\|` caption regex | `HYBRID_FIG_LINE_RE` (`paper2md.py:324`) | n/a — feeds `figures_spliced` | `01cc48e` | Marker changes how it bold-wraps captions (e.g. switches to `__Figure 1__` underscore-bold) |
| Sub-panel reading-order sort | `_ordered_panels_for_emission` (`paper2md.py`) | n/a — drives alt-text letters | `4c5a185` | MinerU's middle.json starts emitting subpanels without bboxes, or `rescue_subpanel_groups` stops capturing `subpanel_bboxes` |
| Inferred fig-number pairing | `_align_marker_to_mineru_layout` final pass | `inferred_fig_number_pairings` | `ce99664` | MinerU caption parsing stops dropping `Fig. N` prefixes; both lists go to zero (the pass becomes inert, which is fine) |
| `(letter)` panel-prefix tolerance | `_PANEL_PREFIX_FRAG` + 4 regexes in `layout_mineru.py` | n/a — feeds subpanel consolidation + `figures_spliced` | `5cdba5c` | MinerU starts emitting captions with a different panel-prefix style (e.g. `[c] Figure 1` or `c. Figure 1`); regex needs another extension |
| Table VLM warmup ping | `_warmup_vlm_connection` (`paper2md.py`) | n/a — only visible in DEBUG log | this commit | vLLM/LM Studio start refusing `models.list()` (unlikely); or persistent-connection behavior changes such that the gap is no longer load-bearing (then the ping is harmless overhead) |
| Hybrid table concurrency | `table_workers` parameter on `_align_marker_to_mineru_layout` + `ThreadPoolExecutor` in the table-task loop | n/a — affects throughput, not output content | `d3a85bc` | `_render_mineru_table` or `wrap_mineru.table_block_to_md` gains shared mutable state outside `report.tables` |
| Table VLM SDK retries + app-level retry loop | `_vlm_table_rewrite` (`wrap_mineru.py`): `max_retries=5` to SDK + 2 app-level retries on `APIConnectionError` with 5s sleep between rounds | exception class in `VLM call failed (...)` log; `app-level retry` info lines | `517de3e` | vLLM's rejection window grows beyond ~22s (12s SDK + 5s sleep + 12s SDK + 5s sleep), or the rejection becomes a different exception type than `APIConnectionError` |
| Table VLM image source — PDF render instead of MinerU JPG | `_render_mineru_table` / `_build_eod_table_splice` use `render_crop(pdf_doc, page_idx, bbox)` and pass PIL image as `pil_image_override` to `wrap_mineru.table_block_to_md` | `Hybrid: rendered table N from PDF` info log | this commit | MinerU's bbox coordinate space diverges from fitz; PyMuPDF Document becomes non-thread-safe; the hypothesis turns out wrong and we revert |

Every fudge surfaces a counter in `rescues.hybrid_splice` in the YAML frontmatter + `.meta.json` sidecar. **Audit the counters per corpus when bumping MinerU.** A counter going to zero may mean MinerU fixed the underlying issue (good — consider deleting the fudge); a counter spiking may mean MinerU broke something new.

---

## 1. Marker image-link strip

**Pattern.** Under hybrid, marker writes its own per-page figure crops to `assets/` (filenames like `_page_1_Figure_2.jpeg`) and emits image links in the body alongside marker's captions. The hybrid splice then inserts MinerU's structurally-cleaner images at the same anchors. Without intervention the body shows two images per figure.

**Trigger / conditions.** Fires under `--layout-source=hybrid` unconditionally. Strips every body line of the form:

```
<span id="page-X-Y"></span><span id="...">...</span>![<alt>](<path-with-marker-basename>)
```

where the path basename is in the set of files marker reported writing. Leading `<span id="page-X-Y"></span>` cross-reference anchors are consumed as a unit so no orphan stubs remain.

**Files on disk are NOT removed.** This is intentional — Sarah wanted A/B comparison during evaluation. To revisit when hybrid is verified: add `(assets_dir / basename).unlink(missing_ok=True)` in `run_marker_plus_mineru_layout` after the strip pass.

**Code.** `_strip_marker_image_links` in `paper2md.py`. Called once from `run_marker_plus_mineru_layout`. The matched basename set is `marker_images.keys()` from `run_marker`'s return tuple.

**What changes break this.**
- Marker switches its image-naming scheme away from `_page_N_*` — set membership keeps working as long as `marker_images.keys()` matches what's in the body links.
- Marker stops emitting `<span id="page-X-Y">` cross-ref anchors — strip pass keeps working; isolated spans become rare.
- Marker starts wrapping images differently (`<figure>...</figure>` HTML, etc.) — strip pass would miss them; need a new pattern.

**Telemetry.** `rescues.hybrid_splice.marker_image_links_removed`. Verified on Hicks2006 (PRL) pre-fix: 4 stripped including span anchors.

---

## 2. MinerU references swap

**Pattern.** Marker's column-aware reference serialiser fails on 3-col Science/Nature papers — numbers dropped, refs glued in pairs across column boundaries, entire ref lists collapsed into one paragraph (lyzenga1980), refs from neighboring articles intermixed (alexander). MinerU's structural extractor produces cleaner numbered lists from the same source.

The baseline corpus that motivated this fudge is the moon / chondrules / silica-shock collections under `workflow/md_database/p2m-hybrid-*/`; diff future runs against those outputs.

**Trigger / conditions.** Fires when BOTH:
1. Marker's body contains a recognizable references heading.
2. `mineru/<stem>.md` also contains a recognizable references heading.

Both checks use the looser `_HYBRID_REFS_HEADING_RE` (`paper2md.py`) which matches:
- `## References`
- `### References`
- `#### References and Notes`
- `#### REFERENCES AND NOTES`
- `## **REFERENCES AND NOTES**`
- `#### **References**`
- heading-less paragraph forms: `References and Notes`, `REFERENCES AND NOTES`, `**References**`

When both match, marker's references section (from the heading to EOF) is replaced with MinerU's.

**Heading guard is load-bearing.** Wackerle1962 has footnote-style refs that MinerU silently dropped (no heading at all). Without the guard the swap would replace marker's refs with empty content. With the guard, papers where MinerU has no refs heading keep marker's version.

**Cross-article contamination guard (post-`66d7cbf` fix).** MinerU's `mineru/<stem>.md` sometimes contains multiple papers concatenated (alexander Science, Elliott Nature News & Views, millot). Before the fix `_extract_mineru_refs_section` used `.search()` which returned the FIRST refs heading — for alexander that was the perspective's refs, and its "section through EOF" extended through the alexander article's title + body + real refs. Swap-time this got pasted into marker's body and **the alexander paper appeared twice** in the output.

Fix: iterate all refs heading candidates; pick the first one whose section is NOT followed by a level-1 `^# Title` heading (`_MINERU_TOP_TITLE_RE`). Sections that ARE followed by a `# Title` belong to a preceding article and would carry the next article's body content with them. For alexander this picks the second match (the alexander article's own refs, no `# Title` after).

Fallback: if every candidate is followed by a `# Title` (reverse MinerU order — perspective at end), pick the last match defensively. Simple single-article papers have one refs heading and zero `# Title` after it — guard is a no-op.

Verified on `p2m-hybrid4-chondrules-single/mineru/alexander.md`: pre-fix the extracted section was 6.2KB (perspective refs + alexander title + alexander body + alexander refs); post-fix it's 3.7KB (alexander refs only).

**Code.** `_extract_mineru_refs_section`, `_swap_hybrid_refs_with_mineru` in `paper2md.py`. Called from `convert()` at line ~8228, AFTER the global `merge_reference_sections` and `normalise_references_section` hooks have run. The MinerU refs section is therefore inserted *raw* — none of the cleanup hooks re-fire on it. This is fine in practice because MinerU's refs are already structurally clean; if it becomes a problem, move the call site earlier in the post-hook pipeline.

**Name collision history.** Commit `6cdca59` introduced this with a constant called `_REFS_HEADING_RE` — same name as TWO pre-existing module-level constants. The existing `_REFS_HEADING_RE` at `paper2md.py:1715` has named groups `hashes` and `title` used by `_refs_section_line_spans`, `merge_reference_sections`, `normalise_references_section`, `inject_orphan_ref_clusters`. The collision crashed those consumers with `IndexError: no such group`. Fixed in `623c538` by renaming the hybrid-only constant to `_HYBRID_REFS_HEADING_RE`. **Do not reuse the `_REFS_HEADING_RE` name** for hybrid-specific work.

**What changes break this.**
- MinerU's `<stem>.md` stops emitting a `## References`-style heading for born-digital papers. Guard returns None, swap silently no-ops, marker's broken refs survive. Visible via `mineru_refs_swapped: false` in frontmatter even when marker refs are bad.
- MinerU starts emitting the refs section title in a new format (e.g. `**Bibliography**` instead of `## References`). Easy fix: extend `_HYBRID_REFS_HEADING_RE`.
- MinerU starts consistently extracting clean 3-col refs *in marker too*. The swap would still fire but emit equivalent content. Becomes inert — consider deleting once verified.

**Telemetry.** `rescues.hybrid_splice.mineru_refs_swapped: true`.

---

## 3. Nature `**Figure N**|` caption regex

**Pattern.** Marker emits Nature-style captions as `**Figure 1** | **Simulation...**` — with **closing** `**` between the figure number and the pipe. The original `HYBRID_FIG_LINE_RE` lookahead `(?=\s*[.|])` rejected the `**` between `1` and `|`. canup Figs 1+2 dropped to `unmatched_mineru_figs` even though MinerU had correctly parsed them.

**Trigger / conditions.** Always active. The lookahead is now `(?=\s*\*{0,2}\s*[.|])` — allows 0-2 asterisks between the figure id and the punctuation. Still rejects body cross-refs like `Fig. 5 shows that...` (no `.` or `|` after the id) and `Fig. 6 of Smith [1990]` (similarly).

**Code.** `HYBRID_FIG_LINE_RE` definition in `paper2md.py:324`.

**What changes break this.** Marker is the source of these captions, not MinerU — so the regex is decoupled from MinerU's version. But if marker ever switches to underscore-bold (`__Figure 1__ |`), the regex needs another extension.

**Telemetry.** None directly. Effect is visible as `figures_spliced` going up and `unmatched_mineru_figs` going down. canup before: 1 spliced / 2 unmatched. After: 3 spliced / 0 unmatched.

---

## 4. Sub-panel reading-order sort

**Pattern.** MinerU's `_group_subpanels` picks one block as "primary" (largest area or first-seen) and the rest as stubs. `_strip_leading_panel_letter` then extracts a letter prefix from the primary's caption if present. Three observed failure modes:

- **canup Figs 1, 2** (Nature 2-panel): primary letter is `b` (extracted from the right-side panel's caption text). Old emission was primary-first → labels `[b, a]` in that order → "b before a" in the markdown.
- **canup Fig 3** (Nature 3-panel a/b/c): primary letter is None. `_subpanel_letter_assignments` returned `[a, b, c]` sequentially with primary at index 0. But the primary was visually panel **c** (the rightmost block); labels said `a, b, c` while content was `c, a, b`.
- **cuk Fig 4** (top-bottom 2-panel): primary letter is None, primary is the BOTTOM block. Old emission labeled bottom as `a` and top as `b` — labels backwards from visual content.

**Trigger / conditions.** Fires for any multi-panel figure under hybrid where MinerU returns a primary with `subpanel_paths`.

**The fix.** `_ordered_panels_for_emission` (`paper2md.py`) has two branches:

1. **bboxes available (normal path):** Sort all panels by `(y_center, x_center)` reading order; assign sequential `a, b, c, ...` in sorted order. `primary.panel_letter` is **ignored** as a label hint — visual position is ground truth.
2. **No bboxes (legacy / test stubs):** Fall back to `_subpanel_letter_assignments` (honors `panel_letter`), then sort `(letter, path)` pairs alphabetically before emission.

For path 1 to fire we need stub bboxes plumbed through `rescue_subpanel_groups`. That required:
- Adding `subpanel_bboxes: list` to `_MBlock` parallel to `subpanel_paths`.
- Populating it in `rescue_subpanel_groups`: `primary.subpanel_bboxes = [s.bbox for s in adopted]`.

**Code.** Helper `_ordered_panels_for_emission` in `paper2md.py`; called by `_render_image_only` (matched-anchor path) and `_build_eod_figure_splice` (unmatched fallback). The `_MBlock.subpanel_bboxes` field lives in `src/layout_mineru.py`.

**What changes break this.**
- MinerU emits panels with degenerate or missing bboxes. The fallback branch kicks in and uses primary-letter-aware labeling (degraded but not broken).
- MinerU's reading order in `parsed.blocks` diverges from visual reading order. The `(y_center, x_center)` sort would still match VISUAL position because we use bbox geometry, not iteration order.
- A paper's panel layout is non-Western reading order (right-to-left, top-to-bottom Japanese). The sort would mislabel. None of our corpus hits this.

**Telemetry.** No specific counter. Visible by inspection of alt text in the body: panel letters should match alphabetical-by-position. Tests `test_render_image_only_subpanels_reading_order_*` lock in the four observed corpus patterns.

---

## 5. Inferred fig-number pairing

**Pattern.** MinerU sometimes extracts a real figure image but mangles or empties its caption text, so `_hybrid_fig_id_from_block` returns `None` and the block silently drops out of `mineru_figs`. Marker's matching `**Fig. N.**` caption then sits in the body with no image. Two observed shapes:

- **alexander Fig 2**: caption was `"The initial (core) and final (rim) olivine FigNa $K_D$ s..."` — the original `"Fig. 2."` prefix got merged mid-text into `"FigNa"` (the `"2."` disappeared into adjacent body text).
- **jacquet Fig 7**: caption text was empty entirely (`text=''`). Image still extracted with valid bbox.

**Trigger / conditions.** Post-main-splice pass. Activates when ALL hold:

1. At least one MinerU image/chart block has no parseable fid AND `min(width, height) > 50` (filters publisher logos / 29×29 DOI badges).
2. At least one marker figure number appears in `fig_anchors` but not `mineru_figs`.
3. The two list lengths are EQUAL.

When all three hold, pairs are formed sequentially: MinerU's `parsed.blocks` are already in reading order (page_idx → intra-page index), marker's anchors are sorted by document position. For born-digital papers the orders align.

**Mismatched-count guard is load-bearing.** A paper with 3 unmatched MinerU blocks and 2 unmatched marker numbers SKIPS the pairing — mislabeling is worse than missing-images. Logged via the counter staying zero while `marker_caption_no_mineru_asset` reflects the unrescued state.

**Size filter is load-bearing.** jacquet has a 29×29 publisher logo (blk[2], page 0) that MinerU also leaves unmatched. Without the filter, jacquet would have 2 unmatched MinerU blocks vs 1 unmatched marker number → counts mismatch → no pairing → Fig 7 stays missing.

**Code.** Inline pass at the end of `_align_marker_to_mineru_layout` (after the main `mineru_figs` loop, before tables). Splices via the same `_render_image_only` path the matched case uses.

**What changes break this.**
- MinerU's caption extraction improves and stops dropping `Fig. N` prefixes. The pass becomes inert (`inferred_fig_number_pairings: 0`). Consider deleting once verified across the corpus.
- MinerU starts emitting many small image blocks (e.g. inline equation glyphs misclassified as images). The size filter threshold may need tightening; alternatively, restrict to type=`chart` only (currently includes `image` too).
- A paper has a real figure smaller than 50×50 pixels (rare in scientific journals). Would get filtered out; need to lower threshold or add per-page heuristics.

**Telemetry.** `rescues.hybrid_splice.inferred_fig_number_pairings`. On 2026-05-12 corpus: alexander = 1 (Fig 2), jacquet = 1 (Fig 7), all other papers = 0.

---

## 6. `(letter)` panel-prefix tolerance

**Pattern.** Three caption-prefix styles observed in the corpus, all of which look the same to readers but differ in OCR transcription:

| Style | Example | Source |
|---|---|---|
| Bare letter | `b Figure 1 \| Simulation...` | Nature (canup) |
| Parenthesized | `(c) Figure 1. Experimental...` | J. Geophys. Res. (feng) |
| Closing-paren only | `b) Figure 1. (a) Schematic...` | AGU (ocampo) |

The original caption predicates in `layout_mineru.py` only matched the bare form, so the other two variants fell through every guard.

Cascade pre-fix on feng:
- `_is_figure_caption_primary` returned False on `(c) Figure 1. ...` → blk[40] not recognized as a Fig 1 primary.
- `_group_subpanels` found no primary on page 2 → no consolidation → the empty-caption stubs blk[38], blk[39] (panels a, b) stayed unmatched.
- `_hybrid_fig_id_from_block` couldn't extract a fig id either (its regex is also `(?:[a-h]\s+)?` bare-letter only).
- feng's Figs 1, 2, 5 all silently dropped from the splice; marker's `**Figure N.**` captions sat alone in the body.

**Trigger / conditions.** Always active. The shared fragment `_PANEL_PREFIX_FRAG` in `layout_mineru.py` accepts all three forms:

```regex
(?:(?:\(\s*[a-h]\s*\)|[a-h]\s*\)|[a-h])\s+)?
```

- Bare: `b ` (any single a-h letter + whitespace)
- Parens: `(b) `, `( b )`, `(B)` (case-insensitive, optional internal whitespace)
- Closing-paren only: `b) `, `B) `

Applied uniformly across all four caption predicates so primary/stub detection, letter extraction, and figure-number extraction all agree.

`_LEADING_PANEL_LETTER_RE` uses three capture groups (one per form) so `_strip_leading_panel_letter` recovers the letter regardless of which alternation matched.

`_STUB_FIG_CAPTION_RE` additionally accepts the multi-letter closing-paren variant `a b)` (ocampo blk[51], page 6 — two panel labels collapsed into one stub by MinerU's caption extractor).

**Code.** `_PANEL_PREFIX_FRAG` definition near top of `layout_mineru.py`. Used by `_PRIMARY_FIG_CAPTION_RE`, `_STUB_FIG_CAPTION_RE`, `_FIG_NUMBER_RE`, and `_LEADING_PANEL_LETTER_RE` (the latter uses a two-group alternation since it needs to *capture* the letter; `_strip_leading_panel_letter` checks both groups).

**Why this matters more than it looks.** Without this fix:
- Multi-panel figures with parens-prefixed primary captions can't be consolidated.
- `_hybrid_fig_id_from_block` returns None for those captions.
- Even the inferred-fig-number-pairing pass (fudge #5) won't help, because counts often mismatch when multiple sub-panels are also unmatched (e.g. feng Fig 1: 3 unmatched MinerU blocks vs 1 unmatched marker caption).

**What changes break this.**
- MinerU switches to yet another panel-prefix style: `[c] Figure 1`, `c. Figure 1`, etc. Each new style needs another alternation branch.
- A real caption begins with one of these prefixes for non-panel reasons (e.g. enumerated list item). Unlikely for figure captions — and the rest of the regex requires `Figure N` to follow, so false positives are bounded.

**Telemetry.** No specific counter. Visible as `figures_spliced` going up and `marker_caption_no_mineru_asset` going down. Concrete corpus impacts:
- feng (`5cdba5c`): `figures_spliced` 2 → 5, `subpanel_groups.groups` 0 → 3, all panel-prefixed figs recovered.
- ocampo (`018dddd`+): `figures_spliced` 7 → 8 (Fig 1 recovered), `subpanel_groups.groups` 0 → 2 (Figs 1 + 4 panels consolidated).

---

## 7. Table VLM warmup — real /chat/completions call

**Pattern: VLM call order matters.** Under hybrid the marker + MinerU phase runs for several minutes with no VLM activity. When the splice fires, it's the FIRST `/chat/completions` POST after the idle gap, and vLLM's model worker rejects the first batch of POSTs for ~10-15s before recovering.

**Why hybrid hits this and marker layout doesn't.** Compare the VLM call order in `convert()`:

| Step | Marker layout | Hybrid layout |
|---|---|---|
| Layout produces md | `run_marker()` — no VLM | `run_marker_plus_mineru_layout()` — **table VLM calls inside the splice** |
| Trim to first article | VLM #1 (small page composite) | VLM #N (after table batch) |
| Citation synthesis | VLM #2 (page 1) | VLM #N+1 (after table batch) |
| Table VLM rewrites | VLM #3+ (process_tables, warmed up) | (already done inside the splice as VLM #1) |

In marker layout, `trim_to_first_article` and `extract_citation` warm vLLM's model worker before `process_tables` fires. In hybrid layout, the table calls have no such prelude — they ARE the first chat/completions request.

**An earlier attempt failed.** Commit `d3a85bc` added a warmup ping calling `client.models.list()` (a `/v1/models` GET). That succeeded but didn't fix anything because `/v1/models` is served separately from `/v1/chat/completions` and doesn't exercise the model worker. Sarah's Wackerle1962 re-run after `d3a85bc` still had all 3 table calls fail (per fudge #9 evidence).

**Fix.** Replace the models.list ping with a **real `/chat/completions` call** using a 32×32 placeholder PIL image and `"Reply OK."` prompt (max_tokens=5, max_retries=5, timeout=30s). Same endpoint, same call shape as the table calls — this actually warms the model worker. Failures swallowed (non-fatal). Skipped for anthropic provider (different cold-call pattern).

**Cost.** ~1-5s per hybrid paper for the warmup call itself. Once. Worth it to avoid table-batch failures.

**What changes break this.**
- vLLM/LM Studio start refusing 32×32 image POSTs or "Reply OK." prompts (unlikely).
- The model worker's cold-call rejection pattern goes away — then the warmup is harmless overhead and could be deleted.
- Someone adds a different VLM call BEFORE the splice in the hybrid path; the explicit warmup becomes redundant.

**Telemetry.** INFO log lines:
- `VLM warmup chat/completions succeeded` — worker is responsive
- `VLM warmup chat/completions returned no content (model worker may not be ready -- table calls will retry on their own)` — warmup itself was rejected; fall back to fudge #9's app-level retry

---

## 8. Hybrid table concurrency (`--table-workers`)

**Pattern.** `process_tables` (the marker-layout table extractor) uses a `ThreadPoolExecutor(max_workers=table_workers)` to fan out VLM rewrites; the hybrid path didn't, so `--table-workers 4` was a silent no-op under `--layout-source=hybrid`. Tables with rowspan/colspan that can't pipe-convert ran sequentially, and a paper with 8 such tables took 8× the per-call VLM time.

**Fix.** Refactored `_align_marker_to_mineru_layout`'s table loop to a 3-phase pattern matching `process_tables`:

1. **Phase 1 (sequential):** Build `table_tasks` list — cheap, no VLM.
2. **Phase 2 (concurrent when `table_workers > 1` AND >1 task):** Submit each task to `ThreadPoolExecutor`. Each task may invoke `_vlm_table_rewrite` internally; vLLM batches concurrent requests server-side.
3. **Phase 3 (sequential):** Apply splices, sort `report.tables` by `index` so concurrent appends are restored to deterministic order.

`table_workers=1` (the default and single-thread baseline) skips the pool entirely — same code path as before.

**What changes break this.**
- `_render_mineru_table` / `wrap_mineru.table_block_to_md` start mutating shared state outside `report.tables` (e.g. writing per-table sidecar files into a shared dict). Today the only shared write is `report.tables.append()` which is GIL-safe.
- The openai SDK's client becomes non-thread-safe (the SDK is documented thread-safe today).

**Telemetry.** No new counter. INFO log emits `Hybrid layout: dispatching N table VLM call(s) concurrently (workers=K)` when the pool spawns.

---

## 9. Table VLM retries — SDK retries + app-level retry loop

**Pattern.** vLLM **actively refuses TCP connections** to `/chat/completions` for ~10-13 seconds after the marker+MinerU idle gap, specifically for the first table-image POST batch. The warmup `GET /v1/models` (fudge #7) succeeds — proves vLLM is up. Article-trim's POST 1 second after the last table failure succeeds — proves the endpoint is generally responsive. But the table batch itself fails its entire retry window. This is TCP-level rejection (`APIConnectionError`, instant per attempt, not a timeout).

**Two-stage Wackerle1962 evidence:**

Stage 1 (commit `d3a85bc`): bumped SDK `max_retries` from default 2 to 5. Saw all 15 retries (5 × 3 concurrent calls) fail across 13s with proper exponential backoff (0.4 → 0.85 → 2 → 4 → 7s). Article-trim worked at +15s.

Stage 2 (this commit): SDK retries weren't enough. Added an **application-level retry loop** in `_vlm_table_rewrite` that catches `APIConnectionError` AFTER the SDK gives up, sleeps 5s, and re-invokes `vlm()` for another full SDK retry round. Two app-level retries by default → worst case ~22s of total wait (12s SDK + 5s + 12s SDK + 5s) before bailing.

To make this work, `p2m.vlm()` gained a `raise_on_error: bool = False` parameter. The default preserves the existing "log and return None" semantics; callers that want to branch on specific exception types pass `raise_on_error=True` and catch `APIConnectionError` / `BadRequestError` / etc. as they see fit. `_vlm_table_rewrite` is the first such caller — non-connection errors (`BadRequestError`, `AuthenticationError`, etc.) bail immediately without app-level retry because more waiting won't help.

We don't know *why* vLLM rejects specifically the table-batch POSTs (and not the warmup GET or article-trim POST). Speculation: request scheduler prefill batch full while vision encoder serializes large images, KV cache initialization, or some server-side state. The "specifically table POSTs after idle gap" pattern is consistent across multiple runs, so we treat it as a known vLLM behavior to ride out rather than diagnose deeper.

**Why not retry forever?** Genuinely-down endpoints would burn ~22s per failing table on top of the SDK timeouts. For a 10-table paper that's an extra 4 minutes of dead time. Cap at 2 app-level retries trades a bit of resilience for not-hanging-on-real-failures.

**What changes break this.**
- vLLM's rejection window grows beyond ~22s. Bump app-level retry count or sleep duration.
- The rejection becomes a different exception type (e.g. `APITimeoutError` instead of `APIConnectionError`). Update the catch clause.
- openai SDK changes its retry policy or exception hierarchy.
- The actual cause is identified and fixed server-side. Then this fudge is harmless overhead — delete it.

**Telemetry.** Look in the log for:
- `VLM call failed (APIConnectionError)` — connection rejection happened
- `vlm_table_rewrite: APIConnectionError attempt N/3, sleeping 5s before app-level retry` — app-level loop kicked in
- `vlm_table_rewrite: gave up after 3 application-level attempt(s)` — rejection persisted past all retries (paper-quality regression; needs investigation)

---

## 10. Table VLM image source — render from PDF, not MinerU JPG (test variant)

**Pattern.** Sarah's vlm-log-2.txt and vlm-log-3.txt showed the table VLM rewrite path failing under hybrid even after the warmup ping (fudge #7) and the app-level retry loop (fudge #9). The Wackerle1962 hybrid run failed all 3 concurrent table calls across 50 seconds despite 51 retry attempts; cuk (which has zero tables) ran cleanly through the same pipeline.

The historical context Sarah surfaced: **the same VLM + table-rewrite path worked under marker+docling many times** before the docling→MinerU swap. Comparing the image-delivery paths reveals the architectural change that's never been called out:

| Path | How the table image reaches the VLM |
|---|---|
| Marker layout + any finder (docling / pymupdf / tatr) | finder locates bbox → `render_crop(doc, bbox)` (PyMuPDF directly from PDF) → PIL RGB Image → PNG encode |
| Hybrid layout | MinerU locates AND saves a cropped JPG to disk → `Image.open(mineru_jpg).convert("RGB")` → PIL RGB Image → PNG encode |

**Both marker paths render fresh from the PDF on demand. The hybrid path uses MinerU's pre-saved JPG.** That's the actual behavioral change between "vlm-tables worked many times" and "now it fails on hybrid" — and it's been hiding behind the docling-optional flip. Even with `--layout-source marker --vlm-tables --table-finder docling` (the historical recipe), the image path is still `render_crop`, never JPG-on-disk.

Hypothesis: MinerU's intermediate JPG carries compression artifacts that the vLLM vision encoder chokes on, while a PyMuPDF-direct render produces cleaner bytes the VLM handles correctly.

**Fix (test variant).** Plumbed `pdf_doc` through `run_marker_plus_mineru_layout` → `_align_marker_to_mineru_layout` → `_render_mineru_table` / `_build_eod_table_splice` → `wrap_mineru.table_block_to_md` → `_vlm_table_rewrite`. When `pdf_doc` is available (always for production runs; tests skip when PDF can't be opened):

- `_render_mineru_table` calls `render_crop(pdf_doc, blk.page_idx, blk.bbox)` to build a fresh PIL image.
- That PIL image is passed as `pil_image_override` to `_vlm_table_rewrite`.
- `_vlm_table_rewrite` uses the override directly, bypassing `Image.open(mineru_jpg).convert("RGB")`.
- On `render_crop` failure (rare; bad bbox or other PyMuPDF error), falls back silently to the MinerU JPG path.

MinerU's bbox coordinate space matches fitz page rect — both are points (verified on Wackerle1962: MinerU page_size [547, 740] == fitz `Rect(0,0,547,740)`). So no coordinate conversion is needed.

`fitz.Document` is thread-safe for reads, so concurrent table workers can share the handle.

**What this proves / disproves.**
- If Wackerle1962 table VLM calls succeed under hybrid after this change → the JPG intermediate was the cause. Roll the fudge into a permanent fix and consider whether the marker-image-strip on disk (fudge #1) can also be loosened since we no longer need MinerU's JPG for VLM.
- If they still fail → the issue is downstream (concurrency, max_tokens, vLLM-specific state). Move to server-side instrumentation.

**Telemetry.** INFO log emits:
- `Hybrid: rendered table N from PDF page P bbox=... for VLM (bypassing MinerU JPG)` — PDF-render path active
- `Hybrid: render_crop failed for table N (falling back to MinerU JPG): <reason>` — fallback fired

**What changes break this.**
- MinerU's bbox coordinate space diverges from fitz (e.g. switches to pixels instead of points). Detectable by comparing rendered crop dimensions to bbox dimensions.
- PyMuPDF's `Document` becomes non-thread-safe for reads. Currently safe per fitz docs.
- The hypothesis turns out to be wrong (tables still fail with PDF render). Document the failure mode and revert; the fudge is then known unnecessary overhead.

---

## When MinerU version changes — review checklist

Pin bump is risky for hybrid because every fudge above assumes MinerU's *specific* failure modes. After bumping `mineru` in the env files:

1. **Re-run the moon / chondrules / silica-shock collections** with `--layout-source=hybrid` against the baseline outputs in `workflow/md_database/p2m-hybrid-*/`.
2. **Diff frontmatter counters** per paper against the baseline:
   - `figures_spliced` going DOWN: regression — fewer figures anchor-matched. Likely caption-format change.
   - `inferred_fig_number_pairings` going DOWN: improvement (MinerU is parsing more captions). Consider deleting the pass.
   - `inferred_fig_number_pairings` going UP: regression (MinerU dropping more captions). Investigate.
   - `mineru_refs_swapped` going from `true` to `false`: regression (MinerU lost the refs heading). Re-tune `_HYBRID_REFS_HEADING_RE`.
   - `marker_image_links_removed` going to zero: regression in the strip pass (marker's filenames or wrappers changed). Check the body for `_page_*` leakage.
3. **Diff the per-paper rendered markdown** against the baseline outputs in `workflow/md_database/p2m-hybrid3-*/` for the spot-checked figures: canup Fig 1-3 (sub-panel order), cuk Fig 4 (top-bottom subpanels), alexander Fig 2 (inferred pairing), jacquet Fig 7 (inferred pairing), Hicks2006 (marker image strip with span anchors), young (refs swap on 3-col paper).
4. **For any fudge whose counter drops to zero corpus-wide**, propose a follow-up to delete it. Each fudge carries maintenance cost; we only keep them while they're load-bearing.

---

## File / counter map for grep

```
paper2md.py
├── HYBRID_FIG_LINE_RE                       (line 324)
├── HYBRID_TABLE_LINE_RE                     (alias, line 336)
├── _hybrid_fig_id_from_block                (line 380)
├── _strip_marker_image_links                # marker image-link strip
├── _HYBRID_REFS_HEADING_RE                  # ref heading regex (do NOT rename to _REFS_HEADING_RE; collides)
├── _extract_mineru_refs_section             # refs swap part 1
├── _swap_hybrid_refs_with_mineru            # refs swap part 2
├── _ordered_panels_for_emission             # sub-panel reading-order sort
├── _render_image_only                       # matched-anchor figure splice
├── _build_eod_figure_splice                 # unmatched-anchor fallback (also handles subpanels)
├── _align_marker_to_mineru_layout           # main splice driver; hosts the inferred-pairing pass
└── run_marker_plus_mineru_layout            # entry point; calls strip + swap

src/layout_mineru.py
├── _MBlock.subpanel_paths                   # list of stub image_paths
├── _MBlock.subpanel_bboxes                  # parallel list of stub bboxes (used by sort)
├── rescue_orphan_captions                   # always-on
└── rescue_subpanel_groups                   # populates subpanel_paths + subpanel_bboxes
```

Counters in `rescues.hybrid_splice` (YAML frontmatter + .meta.json):

```
figures_spliced
tables_swapped
unmatched_mineru_figs
unmatched_mineru_tbls
marker_caption_no_mineru_asset
duplicate_mineru_figs
marker_image_links_removed              # marker strip
mineru_refs_swapped                     # refs swap (bool)
inferred_fig_number_pairings            # inferred pairing
```

Linked: `workflow/md_database/p2m-hybrid-*/` (corpus baseline), `CLAUDE.md` (Stage 7 design memo).
