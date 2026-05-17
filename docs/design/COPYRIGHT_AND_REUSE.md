# Memo: Copyright, paper extraction, and how to make your work reusable

**Audience:** scientists who write papers and who use, build, or share
tools (like paper2md) that extract text, tables, figures, and data
from the published literature.

**TL;DR.** Most "free to read" papers are not free to *reuse*.
Extracting text, tables, or figures from a copyrighted paper into a
shareable corpus, training set, or vector DB sits in legally murky
territory in the United States and is restricted by default in much of
the rest of the world. The single most consequential thing you can do
to fix this — for your own work — is publish or self-deposit under
**CC-BY 4.0**, and re-deposit older preprints with a CC-BY license.
The checklist at the end gives the concrete steps.

This memo is written by a working scientist for working scientists.
It is **not legal advice.** Talk to your institution's research-library
copyright office before making decisions that involve large-scale
redistribution.

---

## 1. Why this matters now

Tools that automatically extract content from PDFs at scale —
paper2md, Marker, Docling, GROBID, olmOCR, Nougat, MinerU — make it
easy for any researcher to build a markdown/JSON corpus of the
literature in their field. That corpus then feeds the obvious
downstream uses:

- **Retrieval-augmented LLMs** that answer questions over your
  field's literature (vector DB + LLM).
- **Fine-tuning or pre-training** of domain-specific models.
- **Meta-analyses and systematic reviews** that need machine-readable
  tables across hundreds of papers.
- **Replication and reanalysis** that needs the actual numbers, not
  a screenshot of a table.
- **Teaching materials** and review articles that quote, excerpt, or
  reproduce figures.

All of these run into the same wall: most published scientific
literature is copyrighted, and the copyright is often held by the
publisher, not the author. The fact that you can read a paper does
not mean you can extract, redistribute, or train on its contents.

The wall is not symmetric. Authors can — under most journal contracts
— *use their own work*; what they cannot easily do is *let other
scientists reuse it* unless they actively choose an open license.
That choice is almost always available, almost always free, and
almost never the default.

---

## 2. What is actually copyrighted in a paper

A typical journal article has multiple layers of intellectual
property, with different default holders and different reusability:

| Element | Copyrightable? | Default holder | Notes |
|---|---|---|---|
| Body text (prose) | Yes — original creative expression | Author at first, then usually transferred or exclusively licensed to publisher at acceptance | The whole paper as a "literary work" |
| Figures (plots, diagrams, photos) | Yes — original creative expression. Even simple plots count. | Same as text | Each figure is a separately-protectable work |
| Tables of data | Mixed. Pure factual data is not copyrightable in the US (*Feist*); selection, arrangement, and creative captioning are. EU "sui generis" database rights protect even non-creative compilations. | Same as text | The numbers are usually free; the layout, headers, and choice of what to include are protected |
| Raw data / measurements | Generally not copyrightable as data; can be protected by contract, database rights (EU), or trade secret | Author / institution / funder | Often the *least* protected legally and the *most* locked up in practice (in supplementary PDFs, paywalled sites) |
| Mathematical expressions, equations | Not protectable as math; the typeset rendering may be | Public domain (math) / Author (rendering) | Equations themselves can always be republished |
| Citations, reference lists | Facts; not copyrightable | n/a | Always reusable |
| Title, abstract | Usually too short for substantial copyright on its own; many publishers explicitly grant abstract reuse | Mixed | Most aggregators (Crossref, NASA ADS, Semantic Scholar) redistribute abstracts under publisher API terms |

Two practical implications:

- **Figures are the hardest.** They are short, distinctive, and
  obviously creative — courts treat them as full works. Reproducing
  even one figure from a copyrighted paper in a review, a textbook,
  or a slide deck typically requires permission unless fair use
  applies.
- **Numbers in tables are the easiest** — the underlying data is
  generally not protected in the US — but the *table as published*
  (with its headers, footnotes, and arrangement) probably is.
  Extracting the numbers and republishing them in a new table is
  usually defensible; reproducing the published table verbatim is
  riskier.

---

## 3. The "free to read but not free to reuse" trap

Three categories of papers a reader can access today, with very
different reuse rights:

**(a) Subscription / hybrid journals, default license.**
Reader paid (via institution); author transferred copyright. Default
reuse rights for third parties: *none*, beyond fair use. Examples:
ApJ in subscription mode, MNRAS in non-OA mode, Icarus, Nature, most
Elsevier titles. You can read the PDF; you cannot put extracted
markdown of it on the open web.

**(b) Hybrid OA articles within subscription journals.**
Author paid an APC; license is typically **CC-BY 4.0** or in some
cases CC-BY-NC. Fully reusable with attribution. Examples: most
society journals (AAS, MNRAS, Icarus) now offer this option for an
extra fee.

**(c) Pure OA journals.**
Default license is CC-BY (sometimes CC-BY-NC for older deposits or
specific journals). Examples: PLOS, eLife, MDPI, Frontiers, AAS's OA
journals (PSJ has CC-BY by default since launch). Fully reusable.

**The trap is (a).** When a paper is on the journal's site behind no
paywall (because of a "free access" promotion, a delayed-OA policy
after embargo, or because your institution has a Read-and-Publish
deal), readers often assume that "free" means "open." It does not.
Free-to-read articles under category (a) carry the same restrictions
as paywalled ones.

**Preprints have the same trap.** The default arXiv submission
license — chosen by the majority of submitters historically — is the
"arXiv.org perpetual non-exclusive license" (or earlier, the
"non-exclusive license to distribute"). It lets arXiv host and
distribute the work. It does **not** grant readers the right to
redistribute, modify, or use the work in derivative products. Old
arXiv preprints — anything before authors started actively choosing
CC-BY at submission — are mostly under this restrictive default.
The same applies to bioRxiv, medRxiv, ChemRxiv, PsyArXiv, SSRN.

---

## 4. What is allowed today for extraction and analysis

Two legal frameworks matter most for the kinds of corpus-building
that paper2md enables:

**United States — fair use.** Four factors, weighed case-by-case:
purpose (transformative? commercial?), nature of the original
(factual vs creative — scientific papers lean factual, which helps),
amount used, market effect. Recent case law:

- **Authors Guild v. Google (2015):** Google Books's full-text
  scanning and snippet display ruled fair use.
- **Authors Guild v. HathiTrust (2014):** Library digitization for
  search and accessibility ruled fair use.
- **Andy Warhol Foundation v. Goldsmith (2023):** Narrowed
  "transformative use" — same purpose as the original weighs against
  fair use.
- **Bartz v. Anthropic (N.D. Cal. 2025):** Judge Alsup ruled in
  June 2025 that training Claude on **legally acquired** books was
  "quintessentially transformative" and fair use, while Anthropic's
  retention of a "central library" of **pirated** books from
  shadow-library sources (LibGen etc.) was not. The parties settled
  in August 2025 for ~$1.5 billion (preliminary approval September
  2025) — roughly $3,000 per title across ~500,000 books, the
  largest copyright settlement in US history. The takeaway is the
  legal/illegal sourcing distinction: training on works you have
  legitimate access to is plausibly fair use; training on a
  pirated corpus is not, even if the training itself is
  transformative.
- **Hachette v. Internet Archive (2024):** "Controlled Digital
  Lending" of full books was *not* fair use.

For research-corpus building, the practical read is:

- **Personal extraction for your own analysis** — almost certainly
  fair use, even on subscription journals you legitimately access.
- **Sharing extracted content with named collaborators** for a
  defined research project — fair use is plausible but
  fact-dependent.
- **Posting an extracted corpus publicly** (GitHub, HuggingFace) —
  high risk under current US law unless every paper is openly
  licensed.
- **Training a model on a corpus** — the *Bartz* settlement
  illustrates that "training is fair use" is not a free pass; the
  acquisition of the training set matters too.

**European Union — text and data mining (TDM) exceptions (DSM
Directive 2019/790).** "Text and data mining" in copyright law means
any automated analytical technique that extracts information from
digital text or data — exactly the kind of work paper2md and
downstream RAG/embedding pipelines do.

- **Article 3:** Research organizations and cultural-heritage
  institutions can perform TDM on works to which they have lawful
  access, for scientific research. Rightsholders cannot opt out of
  Article 3.
- **Article 4:** Anyone can perform TDM on lawfully-accessible works
  for any purpose, *unless* the rightsholder has expressly reserved
  rights via machine-readable means (e.g., `robots.txt`,
  `noai`/`noimageai` meta tags).

For EU-based researchers, Article 3 is materially broader than US
fair use for the corpus-building case. It does not cover *publishing*
the extracted corpus, only the act of extraction and analysis.

**United Kingdom.** UK has its own TDM exception (Section 29A CDPA),
narrower than the EU's — research only, non-commercial only, lawfully
accessible only. A 2022 proposal to broaden to commercial uses was
withdrawn after publisher pushback.

**Other jurisdictions.** Japan, Singapore, and Israel have explicit
TDM exceptions of varying breadth. Most of the rest of the world has
no explicit exception, leaving extraction and corpus-building in
copyright's general framework.

**Bottom line on legality.** Extracting a paper for your own
research is broadly defensible everywhere. *Sharing* the extracted
content is not, unless the source paper is openly licensed. The
checklist below is about creating that openly-licensed source.

---

## 5. The "working copy" allowance — operating under TDM rights

§4 establishes the *legal framework* for extracting content from
copyrighted papers. This section translates that framework into
operating procedure: when you can keep extracted figures and text on
disk, where, for how long, and what to delete when.

The principle: **TDM exceptions and publisher TDM licences both
contemplate that you cannot analyse data without first extracting
it.** When Elsevier, AIP, AGU, Wiley, and Springer Nature grant TDM
rights to your institution as part of the subscription, they are
licensing you to make a working copy of the text and figures for the
duration of the research project. The same is true under EU DSM
Directive Article 3, which permits research organisations to "retain
copies of works for the purposes of scientific research, including
for the verification of research results, with an appropriate level
of security." US fair use offers a parallel (case-by-case) backstop
for research use of legitimately accessed papers.

In practice this becomes a two-phase project lifecycle.

### 5.1 Phase 1 — During the project (the working phase)

You may:
- Download papers your institution legitimately licenses.
- Extract text, tables, and figures (using paper2md or any
  comparable tool).
- Store the extracted assets — `figure_1.jpg`, `figure_2.png`,
  cropped tables as Markdown, parsed reference lists — locally,
  on storage that only you and named collaborators can access.
- Run analysis on those assets: text embedded in a vector DB
  (ChromaDB, Qdrant, Weaviate), figures fed to a local
  vision-language model, classical statistics on table contents.
- Iterate for as long as the project genuinely needs it. EU
  Article 3 imposes no hard time limit, and publisher TDM clauses
  are typically scoped to "the duration of the research project,"
  which can be a year or more for sustained work.

### 5.2 Phase 2 — When the project concludes (the cleanup phase)

Once your paper is submitted, accepted, or published and the active
research is over, the temporary working-copy justification ends.

- **Delete the local cache of publisher-copyrighted figures and
  full-text extracts** that came from non-openly-licensed sources.
  Keep the *derived* outputs: extracted measurements, embedding
  vectors, annotation files, analysis code, citation strings.
- **Retain only what you can lawfully retain.** Article 3 explicitly
  permits retention "for the verification of research results"
  with appropriate security — practically, a locked-down archival
  copy for replication, not an active corpus you keep mining. Many
  institutions interpret this conservatively as "delete"; follow
  your institution's guidance.
- **Keep openly-licensed source files** (CC-BY, CC0) without
  restriction.

### 5.3 Why a local vision-language model matters

Uploading a copyrighted Elsevier figure to a third-party commercial
service (ChatGPT.com, Claude.ai, Gemini consumer chat) transmits a
copyrighted publisher asset to a third party that is not party to
your institution's TDM licence. Whether this is permitted depends
on three things at once:

- **Your publisher TDM agreement.** Most are silent on or
  prohibitive of redistribution to third-party services; some
  modern TDM clauses explicitly forbid sending content to
  generative-AI services.
- **The API or app's data-handling terms.** Anthropic, OpenAI, and
  Google each offer enterprise/zero-retention modes (Anthropic API
  zero data retention, OpenAI Enterprise) that materially reduce
  risk; consumer chat surfaces (ChatGPT free/Pro, Claude.ai)
  typically retain inputs for some period and may use them for
  service improvements. None makes a *copyright* warranty about
  content you upload — that obligation stays with you.
- **TDM right scope.** Your institution's TDM licence covers your
  institution; it does not automatically extend to a separate
  commercial entity you forward the content to.

**The cautious operating posture is: process figures and full text
locally with a local model.** paper2md's pipeline (`--provider
lmstudio` or `--provider vllm`) is built for this — your local
Apple Silicon or Sol GPU node never sends a single byte of the
source PDF to a third-party API. The text and numbers your local
model produces (citations, transcribed tables, short summaries) can
be shared more freely than the source images themselves.

If you do use a commercial API (`--provider openai` or `--provider
anthropic`), check three things first: (1) your institution's TDM
agreement permits it, (2) you have enabled the zero-retention or
Enterprise mode the provider offers, (3) the source paper's licence
does not specifically prohibit AI-service uploads — some publishers
added such clauses in 2024–2025.

### 5.4 What this looks like for paper2md outputs

paper2md produces three classes of output from a single PDF:

| Output | Contents | Treat as |
|---|---|---|
| `<paper>.md` body | Body text + tables in Markdown, often verbatim from the PDF | Working copy of copyrighted text — same rules as the source PDF |
| `assets/figure_*.png` | Cropped figure images | Working copy of copyrighted images — same rules |
| YAML front-matter, citation, embedding vectors, derived measurements | Bibliographic facts and derived numerical data | Generally *not* copyrightable in itself — keep |

The `--hdf5` bundle option packages all three together; convenient
for moving working copies between machines you control. The same
bundle should not be shared outside your TDM licence's scope.
`unpack_h5.py` and `repack_h5.py` exist for exactly this kind of
in-flight management.

When the project ends:

```bash
# Inspect what you'd delete first
find out/ -name 'figure_*.png' -o -name '*.md' | wc -l

# Then remove publisher-copyrighted working copies
find out/ -name 'figure_*.png' -delete
find out/ -name '*.md' -delete   # if these contain verbatim full text
```

Keep the `manifest.jsonl`, the citations, the derived data, and your
analysis code. If you want to retain working copies for replication,
follow your institution's archival procedures — typically an offline
encrypted archive, not an actively-mined directory.

### 5.5 Supplementary materials — a separate copyright analysis

Supplementary files (PDFs of extra figures, methods documents, raw
data tables, instrument calibration sheets) sit in an interesting
middle position: a publisher's standard Copyright Transfer Agreement
typically extends to the supplementary files just as it does to the
main paper, but the *content* inside those files often consists of
raw data that copyright cannot protect at all (Murray-Rust, 2008).
The analysis is case-by-case.

**The files themselves — copyrighted with the article.** When a
paywalled journal (Elsevier, AIP, Wiley/AGU, Springer Nature
subscription, etc.) holds copyright on the article, the
supplementary PDF, DOCX, and figure files come along. Many
publishers add explicit "do not reproduce or redistribute" notices
on the first page of the supplementary PDF.
- *Operational rule:* treat them exactly like the main text.
  Working copy on disk during the project; deleted at project
  conclusion; not redistributed.

**The data inside the files — usually not copyrightable.** Facts
and raw measurements are not copyrightable in the US (*Feist v.
Rural Telephone*, 1991). When a publisher hosts a `.csv`, `.xlsx`,
`.fits`, `.h5`, or VOTable file containing measurements, the
*numbers themselves* are free to extract, retain, and republish as
part of your own dataset. Two caveats:
- **EU sui generis database rights** (the same caveat from §2) can
  protect non-creative compilations where the publisher made
  "substantial investment" in obtaining/verifying/presenting the
  data. Extracting individual data points is fine; bulk-cloning a
  curated database may not be.
- **Selection and arrangement** in a curated supplementary table
  *can* be copyrightable even when the underlying numbers are not.
  Republish the *numbers* in your own table; don't republish the
  publisher's table layout verbatim.

**Open access papers — supplementary inherits the article licence.**
For CC-BY articles (AAS journals since 1 Jan 2022, eLife, PLOS,
gold-OA articles in hybrid journals, etc.), the CC-BY licence
almost always extends to the supplementary materials and figures.
You can keep, modify, and republish both the data and the
supplementary visuals permanently, with attribution.
- *Watch for split licences:* a few publishers issue the article
  as CC-BY but the supplementary data file as CC0 (more
  permissive) or under a separate dataset-licence. Read the data
  record's metadata, not just the article's.

**For the paper2md / RAG pipeline.** Treat supplementary materials
as first-class inputs — they often hold the most extraction-worthy
content (full method narratives, raw measurement tables, extra
figure panels, error budgets, instrument calibration). The
lifecycle distinction from §5.1–5.2 still applies, with one
refinement on what to keep:

| Content from supplement | Copyrightable? | Treat as |
|---|---|---|
| Verbatim text of the supplement PDF | Yes | Working copy — delete at project end |
| Supplementary figure images | Yes | Working copy — delete at project end |
| Numerical tables (CSV/XLSX/FITS/HDF5/VOTable) — the cell values | No (US); thin in EU | Facts — retain permanently, ingest into your dataset |
| Your derived analyses, summaries, embeddings | n/a — you authored | Yours — retain permanently |

paper2md already supports the supplement workflow:

```bash
python src/paper2md.py paper.pdf --supplement paper_SI.pdf -o out/paper/
```

The supplement's assets land alongside the main article's in
`out/paper/assets/` with an `si_` filename prefix
(`assets/si_*.png`, `assets/si_*.jpeg`, etc.) so they don't
collide with the main paper's figures. The §5.4 cleanup commands
extend to those:

```bash
# Inspect what's about to go
find out/ -name 'si_*' -o -name 'figure_*.png' | wc -l

# Remove publisher-copyrighted SI working copies along with main-paper ones
find out/ -name 'si_*' -delete
find out/ -name 'figure_*.png' -delete
find out/ -name '*.md' -delete   # if these contain verbatim full text
```

Keep the numerical tables you extracted from CSV/XLSX/FITS — those
are facts and stay in your dataset.

---

**Reference:** Murray-Rust, P. (2008). Open data in science.
*Nature Precedings*. <https://doi.org/10.1038/npre.2008.1526.1>

---

## 6. Why CC-BY (not CC-BY-NC, not CC-BY-ND, not CC-BY-SA)

Among Creative Commons licenses, **CC-BY 4.0** is the only one that
unambiguously enables the reuse cases that matter for science:

| License | Reuse | Modify | Redistribute | Use in commercial product / company-funded research | Combine with other CC-BY work |
|---|---|---|---|---|---|
| CC-BY | ✅ | ✅ | ✅ | ✅ | ✅ |
| CC-BY-SA | ✅ | ✅ | ✅ | ✅ but derivative must also be CC-BY-SA | ❌ incompatible with CC-BY |
| CC-BY-NC | ✅ | ✅ | ✅ | ❌ — blocks pharma, deeptech startups, contracted research | ⚠ |
| CC-BY-ND | ✅ | ❌ | ✅ | ⚠ no derivatives | ❌ |
| CC0 | ✅ | ✅ | ✅ | ✅ | ✅ — even fewer obligations |

The non-commercial restriction (CC-BY-NC) sounds appealing but in
practice it blocks legitimate scientific reuse: industry collaborators
can't include CC-BY-NC content in funded projects; AI training by
commercial labs is forbidden; you can't combine CC-BY-NC and CC-BY
content in the same derivative dataset cleanly.

**Plan S** (cOAlition S — most major European funders, Wellcome Trust,
Gates Foundation, etc.) requires CC-BY for funded outputs,
specifically because CC-BY-NC fragments the OA corpus.

**The recommended choice for any new paper you publish or deposit is
CC-BY 4.0.** CC0 is also acceptable and slightly more permissive (no
attribution requirement); a few funders accept either.

---

## 7. Checklist — what scientists can do

Concrete actions, ordered roughly by impact-per-effort.

### 7.1 For new papers

- [ ] **Choose a journal that allows CC-BY at submission.** Most
      society journals (AAS, AGU, RAS) and many commercial publishers
      now offer hybrid OA with CC-BY. Check Sherpa Romeo
      (<https://www.sherpa.ac.uk/romeo/>; being migrated to JISC's
      Open Policy Finder at <https://openpolicyfinder.jisc.ac.uk/>)
      for journal-specific terms.
- [ ] **Pay the APC if your funder allows it**, and select **CC-BY
      4.0** on the licensing page. If the publisher offers CC-BY-NC
      or CC-BY-ND as the "default" OA option, ask for CC-BY 4.0
      explicitly — most will allow it on request.
- [ ] **If APC is impossible, use Plan S Rights Retention Strategy
      (RRS).** Add this sentence to your submission cover letter and
      your acknowledgments: *"For the purpose of open access, the
      author has applied a CC-BY public copyright license to any
      Author Accepted Manuscript (AAM) version arising from this
      submission."* This pre-empts the publisher's standard
      copyright transfer for the AAM and preserves your right to
      deposit it CC-BY in a repository.
- [ ] **Negotiate the contract.** The SPARC Author Addendum
      (<https://sparcopen.org/our-work/author-rights/>) modifies
      standard publisher contracts to retain reuse rights. Most
      authors don't negotiate; many publishers accept the addendum
      when asked.
- [ ] **Deposit the AAM in your institutional repository** with
      CC-BY immediately on acceptance. Funder mandates (NIH, NSF,
      most EU national agencies) increasingly require this anyway.

### 7.2 For preprints — including re-licensing old ones

arXiv and most preprint servers now let you choose a Creative Commons
license at submission. They also let you change the license **on a
new version**.

- [ ] **For new preprints:** at submission, choose **CC-BY 4.0** or
      **CC0 1.0** instead of the default arXiv non-exclusive license.
      The license picker is on the metadata page; it takes one click.
- [ ] **For old preprints under the default license:** upload a new
      version (v2, v3, …) and select **CC-BY 4.0** on the license
      page. The new version will be CC-BY; older versions remain
      under their original license, but readers see and cite the
      latest version by default. Even minor updates (typo fixes,
      reference corrections) are a valid reason to upload a new
      version. Same workflow on bioRxiv, medRxiv, ChemRxiv,
      PsyArXiv, SSRN.
- [ ] **Post the publisher PDF (Version of Record)** if and only if
      your publisher contract allows it. Most don't — but most
      *do* allow the AAM. Check Sherpa Romeo for the specific
      journal's policy and embargo period.
- [ ] **In the preprint's title and abstract, include the explicit
      CC-BY notice.** Example: *"This preprint is released under a
      CC-BY 4.0 license."* This makes the license discoverable to
      tools that scan abstracts and helps readers who don't check
      the metadata.

### 7.3 For figures and tables specifically

Even when the paper is openly licensed, *separately depositing* the
figures and data lowers the friction for downstream reuse —
especially for AI training, where rebuilding figures from a PDF is
lossy.

- [ ] **Deposit figure source files on Zenodo or figshare** at the
      time of submission. Choose **CC-BY 4.0** for figures (which are
      creative works) or **CC0 1.0** for raw plots regenerated from
      data. Use the journal article's DOI as a related identifier.
- [ ] **Deposit tables as machine-readable CSV or VOTable**
      separately on Zenodo / VizieR / your domain's data archive.
      Choose **CC0** for raw measurements (lowest friction; matches
      the legal reality that data is generally not copyrightable in
      the US) or **CC-BY** for derived/curated tables.
- [ ] **Cite the data deposit in the paper.** A "Data Availability"
      section with a Zenodo / figshare DOI is now standard and is
      required by most funders.
- [ ] **Avoid putting data only in the supplementary PDF.** PDF
      tables are the worst format for downstream reuse — extraction
      tools (including paper2md) do their best, but you lose
      precision, units, and structure compared to a CSV.

### 7.4 For your group / collaboration

- [ ] **Set a default in your group's submission template.** "We
      submit under CC-BY 4.0 unless there's a specific reason not
      to" — make this the norm for your students and postdocs.
- [ ] **Add the Plan S RRS sentence to your group's standard
      cover-letter template.** One sentence, every submission.
- [ ] **Work with your institution's library** to confirm what
      Read-and-Publish ("transformative") agreements your
      institution has — these often cover the APC for CC-BY OA at
      participating publishers, at no cost to the author.
- [ ] **For collaborations:** if you're a corresponding author,
      ask co-authors before submission to confirm everyone is OK
      with CC-BY. Doing it after copyright is transferred is much
      harder.

### 7.5 For your data — separately from papers

- [ ] **Pick a long-lived archive.** Zenodo (CERN, EU-hosted),
      figshare, your domain archive (NASA ADS / VizieR for astronomy,
      PANGAEA for earth sciences, etc.) — not your personal website,
      not Dropbox, not your university page that disappears when
      you change jobs.
- [ ] **License data CC0 1.0.** It removes ambiguity for downstream
      users who need to combine your data with others'. Funders
      generally accept it; some require it.
- [ ] **DOI everything.** Zenodo and figshare mint DOIs on upload.
      Cite the data DOI in the paper, and cite the paper DOI in the
      data record.
- [ ] **Use FAIR-aligned formats.** CSV, HDF5, FITS, NetCDF, Parquet
      — anything readable without proprietary software. Avoid
      MS-Excel-only formats for archival data.

---

## 8. Walkthrough: re-depositing an old arXiv preprint with CC-BY

Most working scientists have a backlog of arXiv preprints from before
they started actively selecting a license. Re-licensing them is the
single highest-leverage thing you can do — old preprints become
reusable without renegotiating with publishers.

**One-time setup.** Make sure your arXiv account is linked to your
ORCID. This makes the "my submissions" list reliable.

**Per-preprint workflow:**

1. Log in to arXiv → "User → My submissions."
2. Click **"Replace"** next to the preprint you want to re-license.
3. Upload the same source files you originally used (or the latest
   AAM if you've made minor corrections; arXiv requires *some*
   substantive change for v2+, even a single comma or reference
   addition counts).
4. On the licensing screen, change the license from the arXiv
   default to **"Creative Commons Attribution 4.0 International
   (CC BY 4.0)"**.
5. In the comments / version-note field, add: *"v2: relicensed
   under CC-BY 4.0"* plus any actual changes.
6. Submit. The new version is live within ~1 business day.

**Important caveats:**

- arXiv's policy is that the license applies to *that submitted
  version*. v1 remains under its original license. But citations
  and links default to the latest version, so v2-onward is what
  readers and downstream tools see.
- You can only re-license **your own work**. Co-authored papers
  need the corresponding author or all co-authors to agree (arXiv
  typically requires the submitting account to be an author).
- If the paper was published in a journal and you transferred
  copyright, the **arXiv preprint** is still yours to license — the
  publisher owns the journal version (Version of Record), but the
  preprint and AAM are typically the author's. Check Sherpa Romeo
  for the specific journal to confirm. AAS journals, MNRAS, AJ,
  ApJ all permit author retention of preprint rights.
- Some old arXiv versions used the "non-exclusive distribution"
  license that explicitly grants arXiv a perpetual right but says
  nothing about reader rights. CC-BY on a new version supersedes
  this for the new version.

**Order of operations for a backlog:**

1. List your arXiv submissions sorted by citation count.
2. Re-license the top ~10 first — these get the most reuse.
3. Then walk down the list at one per week. The whole process is
   ~5 minutes per preprint.

---

## 9. Discipline-specific notes

**Astronomy / planetary science (AAS journals + arXiv).**
- AAS journals — ApJ, AJ, ApJL, ApJS — went **fully open access on
  1 January 2022** and now publish all articles under CC-BY by
  default; authors retain copyright. PSJ (Planetary Science Journal)
  has been fully OA since launch.
- Icarus (Elsevier): hybrid OA with CC-BY 4.0 as the recommended
  option; CC-BY-NC-ND is also offered, avoid it.
- Nature Astronomy / Nature Geoscience: hybrid OA available, CC-BY
  4.0 for OA articles. APC for the *Nature* flagship is currently
  **$12,850 / £9,390 / €10,850** (2026 rates, hybrid OA route);
  Nature Astronomy and other Nature-Portfolio journals have
  separate, comparable APCs. Always check the journal's current
  page before submission.
- arXiv is the universal preprint home in this field. Re-licensing
  old preprints is high-impact because the field's reading habits
  are arXiv-first.

**Physics — shock, plasma, condensed matter (APS, AIP, IOP + arXiv).**
- arXiv is the dominant preprint server in physics — it was created
  for physics in 1991. Subject classes relevant here:
  `physics.shock-phys`, `physics.plasm-ph`, `cond-mat.*`. Re-licensing
  old preprints to CC-BY (§8) is especially high-impact in physics
  because reading habits are arXiv-first; the journal version is
  often consulted only for the formal record.
- **APS (American Physical Society)** — *Physical Review B* (cond-mat
  flagship), *Physical Review E* (plasma, soft matter, nonlinear),
  *Physical Review Letters*, *Physical Review Materials*, *Physical
  Review Research*. Hybrid OA with **CC-BY 4.0** at extra fee.
  - APS has been "green" since 1998 — authors may post the accepted
    manuscript on arXiv and on institutional repositories at any
    time (the Version of Record / publisher PDF is the only version
    that's restricted). This makes APS preprint deposit unusually
    friction-free.
  - **Important caveat:** green OA on APS does **not** satisfy a
    CC-BY rights-retention requirement (e.g., Plan S RRS) on its
    own — the AAM you post is under APS's standard license, not
    CC-BY. To deliver CC-BY for an APS paper you must pay for gold
    OA.
  - **Physical Review X** is fully gold OA (CC-BY) and has been
    since launch in 2011 — the strong choice for high-impact
    cond-mat / plasma / HEDP work that would otherwise go to PRL.
    *PRX Energy* and *PRX Life* are similar gold-OA siblings.
- **AIP Publishing** — *Physics of Plasmas* (the flagship plasma
  journal), *Journal of Applied Physics* (shock physics, dynamic
  materials response), *Review of Scientific Instruments*, *Applied
  Physics Letters*. Hybrid OA with CC-BY 4.0 at extra fee. Default
  subscription license retains author copyright but grants AIP an
  exclusive license; choosing the OA route restores broader rights.
- **IOP Publishing** — *Plasma Physics and Controlled Fusion*,
  *Nuclear Fusion* (with IAEA), *Plasma Sources Science and
  Technology*, *Journal of Physics: Condensed Matter*. Hybrid OA
  with CC-BY 4.0. **New Journal of Physics** (IOP/DPG) has been
  **fully gold OA (CC-BY) since 1998** — one of physics' oldest
  open-access journals and a credible alternative to PRX for
  multi-subfield work.
- **Elsevier** — *High Energy Density Physics* (HEDP, ICF, shock,
  warm dense matter), *International Journal of Impact Engineering*,
  *Journal of Nuclear Materials*, *Combustion and Flame* (shock-
  reactive flow). Hybrid OA with CC-BY 4.0. **Avoid** the CC-BY-NC-ND
  option Elsevier sometimes presents as the "default" OA license.
- **Springer / Cambridge** — *Shock Waves* (Springer; the journal
  for shock-wave physics + engineering), *Journal of Plasma Physics*
  (Cambridge), *Journal of Dynamic Behavior of Materials* (Springer).
  Hybrid OA with CC-BY 4.0.
- **Nature Portfolio** — *Nature Physics*, *Nature Materials*,
  *Nature Communications* (multidisciplinary, fully OA), and the
  fully-OA **npj** family (*npj Computational Materials*, *npj
  Quantum Materials*, *npj 2D Materials and Applications*). All
  publish OA articles under CC-BY 4.0; APCs are comparable to other
  Nature-Portfolio titles.
- **Conference proceedings warning.** A lot of shock physics, ICF,
  and high-pressure physics moves through proceedings — *AIP
  Conference Proceedings* (HPSCC / Shock Compression of Condensed
  Matter, AIRAPT, IFSA), *Journal of Physics: Conference Series*
  (IOP), *EPJ Web of Conferences*. These often have **more
  restrictive author rights** than peer-reviewed journals; some
  proceedings publication agreements transfer copyright outright
  and don't permit a CC-BY preprint deposit at all. Read the
  conference's IP terms before signing, and **deposit on arXiv
  before the conference submission deadline** so the preprint
  rights are locked in independently of the proceedings contract.
- **Funder mandates that affect this field.** US DOE (Office of
  Science, NNSA — funds most US plasma / HEDP / shock work), NSF,
  and DOD all now have public-access requirements; cOAlition S
  members fund a sizeable share of European condensed-matter and
  fusion work and require CC-BY. Check funder requirements before
  defaulting to subscription publication.

**Earth sciences (AGU + Elsevier).**
- AGU journals (JGR family, GRL, etc.) have a CC-BY OA option.
- ESS journals adopted CC-BY 4.0 as default for new OA papers.
- ESSOAr / EarthArXiv preprints support CC-BY at submission.

**Biology (PubMed Central, bioRxiv).**
- Many funders (NIH, Wellcome, Gates) now mandate CC-BY for OA papers.
- bioRxiv supports CC-BY, CC-BY-NC, CC-BY-ND, CC0 at submission and
  on new versions.

**Computer science / ML (arXiv).**
- CS preprint culture is strong; many papers exist *only* on arXiv
  and never go to a journal. This makes preprint license choice
  even more consequential.
- NeurIPS, ICML, ACL increasingly accept CC-BY OA for proceedings.

---

## 10. Frequently asked questions

**"My publisher told me I can't deposit my paper publicly."**
Almost always means the *Version of Record* (the typeset journal
PDF). The Author Accepted Manuscript (your final post-peer-review
text, before publisher typesetting) is almost always allowed under
green OA, possibly after an embargo. Check Sherpa Romeo for the
specific journal's "AAM" (sometimes "Postprint") policy.

**"Can I extract content from copyrighted papers for my own
research?"**
In the US: almost certainly yes, as fair use, for personal scholarly
analysis. In the EU: yes, under DSM Directive Article 3 (research
TDM exception). In the UK: yes, under Section 29A CDPA, for
non-commercial research. Sharing the extracted corpus publicly is
the part that gets risky everywhere.

**"What about training an LLM on a corpus of papers?"**
The 2025 *Bartz v. Anthropic* ruling is the clearest US data point
so far: training itself can be fair use, but **how you sourced the
corpus matters separately**. Anthropic was cleared on the training
question and on the books they bought; they were not cleared on the
books they pulled from LibGen-style shadow libraries, and ended up
in a $1.5 billion settlement specifically over the pirated portion.
EU TDM exceptions cover training (Article 4 includes commercial AI
training, with rightsholder opt-out). Japan's exception is broadest.
Practical recommendation: train on content that is openly licensed
*or* that you have legitimate single-user access to, document
provenance, and don't redistribute the training set.

**"I can't change the license on a paper I co-authored without
asking everyone."**
For arXiv re-deposit, only the submitter (typically a corresponding
author) is required by the platform. Out of professional courtesy,
notify your co-authors. For published-paper rights retention, all
authors typically have to agree before signing the publisher
contract.

**"My paper is already published with copyright transferred. Can I
do anything?"**
You can almost always still re-license the preprint on arXiv. You
can usually deposit the AAM in your institutional repository under
CC-BY (Plan S RRS protects this prospectively, but for past papers
it's case-by-case). You usually cannot re-license the Version of
Record without the publisher's consent.

**"What about figures from other people's papers in my review
article?"**
You generally need permission from the publisher, unless (a) the
source paper is CC-BY, (b) the figure was reproduced from a
non-copyrightable source (like NASA / public-domain data), or (c)
you can make a fair-use argument (transformative, limited, no market
harm). Many publishers have streamlined permissions through
RightsLink. Plan ahead — this can take weeks.

---

## 11. Further reading and resources

**Practical guides:**
- Sherpa Romeo (journal policies) — <https://www.sherpa.ac.uk/romeo/>
  (the underlying service is migrating to JISC's consolidated **Open
  Policy Finder** — <https://openpolicyfinder.jisc.ac.uk/> — which
  combines Sherpa Romeo, Juliet, Fact, and OpenDOAR)
- Plan S Rights Retention Strategy — <https://www.coalition-s.org/rights-retention-strategy/>
- SPARC Author Addendum — <https://sparcopen.org/our-work/author-rights/>
- Creative Commons license chooser — <https://creativecommons.org/choose/>
- arXiv license options — <https://arxiv.org/licenses/>

**Background reading:**
- Suber, Peter. *Open Access* (MIT Press, 2012; free online) —
  the canonical introduction.
- COMMUNIA Association — EU copyright reform analysis,
  <https://communia-association.org/>
- Authors Alliance — author-side rights advocacy and templates,
  <https://www.authorsalliance.org/>

**For institutions and groups setting policy:**
- cOAlition S funder requirements — <https://www.coalition-s.org/>
- Open Access Scholarly Publishers Association — <https://oaspa.org/>

---

## 12. Disclaimer

This memo is written by a working scientist as practical guidance for
peers. It is not legal advice. Copyright law varies by jurisdiction
and changes; case law on AI training, TDM, and digital reproduction
is actively evolving. Talk to your institution's research-library
copyright office or general counsel before making decisions involving
large-scale redistribution, public corpus release, or commercial use.
