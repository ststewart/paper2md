# Running paper2md on an NVIDIA DGX Spark — setup + tuning guide

This is the day-one cheat-sheet for getting paper2md running on a
DGX Spark (Grace-Blackwell GB10, sm_121, ARM64, 96 GB unified
memory). Pairs with [`USAGE.md`](USAGE.md) for the full CLI
reference and [`BATCH.md`](BATCH.md) for production-batch operations.

For the ASU Sol HPC cluster, see [`SOL_SETUP.md`](SOL_SETUP.md)
(different hardware path: A100 batch + GH200 interactive).

## 1. Hardware: what's special about the Spark

The Spark is the desktop Grace-Blackwell box. Key things that
matter for paper2md tuning:

| Item | Value | Why it matters |
|---|---|---|
| GPU | NVIDIA GB10 (Blackwell desktop, sm_121) | Uses BF16, **not** FP8 — vllm 0.19.1's CUTLASS FP8 GEMM kernel crashes on sm_121. Stick to BF16-quant'd models. |
| CPU | 20-core Grace ARMv9 | aarch64 — pip wheels for some packages (notably older `torch`) need explicit ARM builds. The `environment-gpu.yml` env file pins working versions. |
| Memory | 96 GB unified (LPDDR5X + HBM3) | The "GPU memory" is the same physical pool as system memory. `--gpu-memory-utilization 0.65` (hybrid) or `0.80` (marker-only) trades KV-cache headroom for room to spawn MinerU alongside marker. See §3 below. |
| OS | Ubuntu-flavored Linux on aarch64 (`Linux 6.17.0-1014-nvidia` at time of writing) | Standard `apt`-based; no special procedure beyond NVIDIA's driver setup. |

The GB10 is *not* a Hopper or full Blackwell datacenter chip — some
optimizations (FP8 attention, sm_90-only kernels) won't work. The
config in this doc is the empirically-stable baseline.

## 2. Install: conda env + paper2md

One-time:

```bash
# 1. Clone the repo (or rsync your local copy onto the Spark).
cd ~ && git clone <your-fork-url> paper2md && cd paper2md

# 2. Create the env (uses environment-gpu.yml; pins are Spark-tested).
conda env create -f environment-gpu.yml
conda activate paper2md

# 3. Post-install: marker + surya + MinerU with PINNED versions.
#    These pins are load-bearing for paper2md's hooks; bumping them
#    without re-auditing the splice fudges WILL cause parsing errors
#    (see USAGE.md §2 and docs/dev/HYBRID_IMPLEMENTATION.md).
pip install --no-deps marker-pdf==1.10.2 surya-ocr==0.17.1
pip install "mineru[core]==3.1.7"

# 4. Sanity-check the GPU + torch.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'arch', torch.cuda.get_device_capability())"
# Expected: torch 2.x.x cuda True arch (12, 1)

# 5. Verify MinerU pinned correctly.
python -c "from importlib.metadata import version; print('mineru', version('mineru'))"
# Expected: mineru 3.1.7
```

The expected `(12, 1)` is `sm_121`. If you get `(8, 0)` or `(9, 0)`
you're not on the Spark — different host, different tuning.

**Why MinerU 3.1.7 is pinned**: paper2md's hybrid splice and several
layout-rescue hooks parse MinerU's `middle.json` schema. Minor-version
schema drift silently breaks caption detection, table HTML extraction,
and figure-asset paths. If you must upgrade, follow the procedure in
USAGE.md §2.

## 3. Start vllm

paper2md auto-pairs `--backend cuda` with `--provider vllm`, so
you just need vllm serving on the default port:

```bash
# In a tmux session (so it survives ssh disconnects):
tmux new -s vllm
conda activate paper2md

# Recommended for HYBRID layout (marker body + MinerU layout):
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65

# Marker-only or MinerU-only workflows can use 0.80:
# vllm serve ... --gpu-memory-utilization 0.80

# Detach: Ctrl-b d
# Reattach: tmux attach -t vllm
```

Why each flag:

- **`Qwen/Qwen3-VL-32B-Instruct`** — paper2md's default VLM. BF16
  weights, ~64 GB on disk. Hugging Face downloads to `~/.cache/huggingface`
  on first run; takes 10–20 minutes the first time.
- **`--port 8000`** — paper2md's `VLLM_BASE_URL` defaults to
  `http://localhost:8000/v1`. Override via env var if you need a
  different port.
- **`--max-model-len 32768`** — fits comfortably in the GB10's
  KV-cache budget at 4 concurrent sequences. Drop to 16384 if
  you see eviction in the vllm log.
- **`--gpu-memory-utilization`** — picks the trade-off between
  vLLM's KV-cache headroom and free system memory for other CUDA
  consumers in the pipeline. The Spark's *unified* memory pool
  means every claimant comes out of the same 96 GB:
  - **`0.65` (recommended for `--layout-source hybrid`)** — leaves
    ~32 GB free, room for marker (~6 GB) AND MinerU's subprocess
    (~6-8 GB) on the same paper. KV cache shrinks to ~14 GB which
    still supports ~1.7× concurrency at 32K tokens — fine for
    paper2md's 4-concurrent table-rewrite burst.
  - **`0.80` (marker-only / MinerU-only)** — leaves ~19 GB free.
    Plenty for a single CUDA consumer. KV cache ~36 GB. Use this
    for `--layout-source marker` or `mineru` runs.
  - **`0.85`+ — discouraged.** No safety margin; one large paper
    can OOM the whole pipeline. Empirically the boundary where
    MinerU subprocesses started failing with `error_kind=cuda_oom`
    in `manifest.jsonl`.

Pre-flight diagnostic: paper2md's hybrid path logs available memory
right before spawning MinerU. A WARN line "MinerU pre-flight: X.X
GiB available before spawning… below 8 GiB safe threshold" is the
signal to drop to 0.65.

For a long-lived setup, switch from tmux to a `systemd --user` unit:

```ini
# ~/.config/systemd/user/vllm.service
[Unit]
Description=vLLM Qwen3-VL-32B server
After=network.target

[Service]
Type=simple
WorkingDirectory=%h
Environment=PATH=%h/anaconda3/envs/paper2md/bin:/usr/bin:/bin
ExecStart=%h/anaconda3/envs/paper2md/bin/vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.65
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
```

```bash
loginctl enable-linger $USER          # one-time: keeps user services running after logout
systemctl --user daemon-reload
systemctl --user enable --now vllm
journalctl --user -u vllm -f          # follow logs
```

## 4. Verify vllm is up

Three quick checks:

```bash
# 4.1. Is the port listening?
ss -tlnp | grep :8000

# 4.2. Does /v1/models answer with the loaded model?
curl -s http://localhost:8000/v1/models | jq '.data[].id'
# Expected: "Qwen/Qwen3-VL-32B-Instruct"

# 4.3. Does it actually respond to a generation request?
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-32B-Instruct","max_tokens":4,
       "messages":[{"role":"user","content":"Reply with the digit 7."}]}' \
  | jq -r '.choices[0].message.content'
# Expected: 7 (or "7" or similar)
```

If 4.1 returns nothing → server not started; check `journalctl`
or the tmux session. If 4.2 returns empty → server still loading
weights (first start is slow). If 4.3 hangs > 30 s → likely OOM or
stuck mid-prefill; restart vllm.

## 5. Verify concurrency: can vllm handle 4 simultaneous requests?

This matters because **`--table-workers 4` is the single biggest
speed lever for paper2md** (see [§5.5 below](#55-tuning-paper2md-for-the-spark)).
Confirm the server can actually batch them before relying on it.

### 5a. Check vllm's continuous-batching config (upper bound)

```bash
ps -ef | grep 'vllm serve' | head -1
```

Look for `--max-num-seqs N` in the launch command. Default is
**256**, so 4 concurrent sequences is well below the limit unless
you explicitly capped it. (The actual constraint is KV-cache
headroom — see 5c.)

### 5b. Smoke test with 4 concurrent curl calls

```bash
seq 4 | xargs -P4 -I{} bash -c 'echo "req {} start $(date +%T.%N)"; \
  curl -s -o /dev/null -w "req {} done %{time_total}s http=%{http_code}\n" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-VL-32B-Instruct\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 50.\"}]}" \
    http://localhost:8000/v1/chat/completions'
```

Read the timing output:

- **All 4 finish within ~1–1.5× the slowest** → vllm is
  continuous-batching them. Use `--table-workers 4`.
- **First finishes at T, second at 2T, third at 3T, fourth at 4T** →
  serializing. Either `--max-num-seqs 1` was set, you're prefill-
  bound, or you're on an old vllm. Stick with `--table-workers 1`.

### 5c. Watch GPU saturation during a real run

In a second terminal:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

While paper2md is in hook 1 (you'll see `INFO VLM-rewriting table N`
lines):

- **GPU util pinned 95–100%, memory.used stable** → batching is
  saturated. 4 workers is fine; 6 might still help.
- **GPU util oscillates 30–95%, periodic dips** → decode-bound on
  each request individually; concurrent requests aren't filling
  the slack. 4 workers helps; more probably won't.
- **Memory.used climbing toward limit, then `preemption` warnings
  in vllm's log** → KV cache pressure. Drop to `--table-workers 2`
  or restart vllm with a smaller `--max-model-len`.

### 5d. The empirical answer everyone trusts: time the same paper at 1 vs 4

```bash
time python src/paper2md.py examples/canup.pdf -o /tmp/t1 \
    --table-finder docling --no-metadata-lookup --table-workers 1
time python src/paper2md.py examples/canup.pdf -o /tmp/t4 \
    --table-finder docling --no-metadata-lookup --table-workers 4
```

Expected ratios on a healthy server with a multi-table paper:

| `t4 / t1` | What it means | Action |
|---|---|---|
| 0.4 – 0.6 | continuous batching working | scale up to corpus |
| 0.7 – 0.9 | marginal gain (KV-cache or VRAM bound) | 4 workers OK but not free |
| ≈ 1.0 | server is serializing | investigate via 5a + vllm log |
| > 1.0 | queue contention or eviction churn | drop to 2 |

**Watch the vllm log too.** Lines like `Avg prompt throughput`,
`Avg generation throughput`, and `running: 4 reqs, swapped: 0 reqs,
pending: 0 reqs` mean it's happily batching 4. Anything with
`swapped > 0` means KV cache is being evicted — drop concurrency.

## 5.5 Tuning paper2md for the Spark

After confirming vllm handles 4 concurrent requests:

```bash
# Default fast path: docling table-finder + detector-text mode. Seconds
# per paper for the table phase. Other VLM hooks (citation, figures,
# rescue) still run.
python src/paper2md.py paper.pdf -o out/

# High-fidelity tables: opt INTO the per-table VLM rewrite. Adds 1-8
# min per dense table; recovers subscript / Greek / footnote-marker
# fidelity that the default detector-text path loses. Pair with
# --table-workers 4 to dispatch concurrently.
python src/paper2md.py paper.pdf -o out/ --vlm-tables --table-workers 4

# Batch with the same opt-in pattern (see BATCH.md for the full runbook):
python src/paper2md.py --batch corpus/ -o out/ --workers 2 \
    --vlm-tables --table-workers 4
```

| Setting | Recommended | Why on the Spark |
|---|---|---|
| `--table-finder` | `docling` (DEFAULT) | Strongest table detection across the four available finders. ~500 MB models, downloads on first use. Default since v0.2; the detector-text shortcut hands its pre-extracted markdown straight to the body. |
| `--vlm-tables` | conditional | OFF by default. Pass to opt INTO the per-table VLM crop rewrite when subscript / math / footnote-marker fidelity matters; pair with `--table-workers 4` for concurrent dispatch. |
| `--table-workers` | **4** with `--vlm-tables` | Single biggest speed lever for the VLM-rewrite path. Confirmed via §5d on the Spark with 32B BF16 + 32k context. No effect under default detector-text mode (no per-table VLM calls to dispatch). |
| `--workers` (batch only) | 2 | Marker holds `_MARKER_LOCK` so only one paper's marker run uses the GPU at a time; the second worker overlaps its VLM hooks with the first's marker. Going higher saturates without speedup. |
| `--gpu-memory-utilization` (vllm) | **0.65 for hybrid**, 0.80 for marker-only / mineru-only | At 0.65 leaves ~32 GB for marker (~6 GB) + MinerU subprocess (~6-8 GB) on the same paper; KV cache 14 GB. At 0.80 leaves ~19 GB which is fine when only marker OR mineru spawns under paper2md (not both). 0.85+ risks `error_kind=cuda_oom` in `manifest.jsonl` for hybrid runs. |
| `--max-model-len` (vllm) | 32768 | Comfortable for any single paper's table prompts (max ~5k input tokens). Drop to 16384 if you see preemption warnings or want to push `--table-workers` to 6. |

## 6. Known issues + workarounds on sm_121

### 6.1 vllm hangs at `max_tokens >= 2500`

paper2md caps table-rewrite generations at `max_tokens=2000` for
exactly this reason. **Don't raise it.** If a wide table truncates,
the failure is graceful (the body is kept as marker output and the
quality score reflects it). At max_tokens=2500+, vllm 0.19.1's
attention kernel hangs — the request never completes and the whole
table phase times out.

This is a documented vllm/sm_121 bug; the workaround is the cap.

### 6.2 FP8 weights won't load

If you try `Qwen/Qwen3-VL-32B-Instruct-FP8` (or any other FP8 model),
vllm crashes during the first forward pass with a CUTLASS GEMM
error on sm_121. **Stick to BF16.** Track <https://github.com/vllm-project/vllm/issues>
for sm_121 FP8 support; bump back when it lands.

### 6.3 Marker / surya pinning across Mac and Spark

As of v0.2.2, `environment-mac.yml` and `environment-gpu.yml` pin
the SAME exact ML stack versions (`marker-pdf==1.10.2`,
`surya-ocr==0.17.1`, `transformers==4.57.6`, `torch==2.10.0`,
`pymupdf==1.27.2.3`, `docling==2.92.0`, `pillow==10.4.0`) so
paper2md outputs are bit-comparable across Mac CPU and Spark
CUDA runs. **Bump these versions in both env files in the same
commit**; one-sided bumps cause frontmatter drift across machines
that breaks the user's corpus pipeline reproducibility.

Note: the conservative `marker-pdf==1.8.5` / `surya-ocr==0.15.4`
pin that earlier paper2md versions shipped on Apple Silicon (in
the mistaken belief that surya 0.16 introduced the MPS bug) is
gone as of v0.2.2. CPU on Apple Silicon doesn't care about the
MPS bug, and Mac CPU + the latest pins matches Spark CUDA on
canup.pdf and root-maintext.pdf.

### 6.4 First marker run is slow (model downloads)

Surya layout, OCR, and equation models are ~1 GB total. They
download to `~/.cache/datalab` on the first run, which adds 5–10
minutes to the first paper. Subsequent runs reuse the cache.

If you're behind a corporate proxy, set `HF_ENDPOINT` and
`HF_HUB_OFFLINE=0` before the first run.

### 6.5 Long DFT-MD-style tables are still slow

A 30-row × 8-column table can hit the `max_tokens=2000` ceiling
and take 5–8 minutes per VLM call (decode is ~40–80 tokens/sec on
the Spark). Mitigations:

- `--table-workers 4` — concurrency hides individual call latency
  for multi-table papers (each call is still long, but you run 4
  at once).
- Default mode (no `--vlm-tables`) — bypasses the per-table VLM call
  entirely. ~10× speedup vs `--vlm-tables`; loses subscript / Greek /
  footnote-marker semantics in the table body. This is the v0.2
  default; `--no-vlm-tables` is the deprecated alias.
- Smaller VLM (Qwen2-VL-7B, Gemma 3 4B) — ~4× decode throughput at
  some accuracy cost.

For "why is this slow" in detail, see [USAGE.md §6.2e](USAGE.md#62e-concurrent-table-vlm-calls---table-workers).

## 7. Environment variables (optional)

In a `.env` file at the repo root or in your shell:

```bash
VLLM_BASE_URL=http://localhost:8000/v1     # paper2md default
VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct       # paper2md default
UNPAYWALL_EMAIL=you@example.com            # required for the metadata pre-pass
OPENALEX_MAILTO=you@example.com            # polite-pool, optional
CROSSREF_MAILTO=you@example.com            # polite-pool, optional
PAPER2MD_USER="Your Name"                  # default --user value
PAPER2MD_COLLECTION=my-corpus              # default --collection value
```

paper2md's `python-dotenv` loader auto-picks up `.env` from the
script directory. See [USAGE.md §13](USAGE.md) for the full env-var
matrix across providers.

## 8. Day-one verification

A useful single-paper smoke test that exercises every hook:

```bash
python src/paper2md.py examples/canup.pdf -o out/canup_smoke \
    --table-finder docling --table-workers 4 \
    --user "$USER" --collection smoke-test \
    --note "first run on Spark"
```

Expected outputs:

- `out/canup_smoke/canup.md` — quality grade A or B in the
  `quality:` block.
- `out/canup_smoke/canup.meta.json` — `run.backend == "cuda"`,
  `run.vlm_provider == "vllm"`, `run.packages.marker-pdf` set.
- `out/canup_smoke/assets/` — figure JPEGs for Figs 1–3 (canup
  has 3 figures; all should match. Truth fixtures for the
  figure-caption matcher live in `tests/figure_match_truth/`).

If figure 3 is missing, you're on a paper2md version older than
`92dc1b2` (the looser figure-classify prompt commit). Pull and
retry.

## 9. Where to go next

- [`USAGE.md`](USAGE.md) — full CLI reference, hook-by-hook
  behavior, scoring rubric, all flags.
- [`BATCH.md`](BATCH.md) — production batch runbook (tmux, monitoring,
  manifest queries, when to hand off to a frontier model).
- [`FLOWCHART.md`](FLOWCHART.md) — stage-by-stage pipeline diagram
  and decision points.
