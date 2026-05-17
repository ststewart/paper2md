# Sol GH200 (Grace Hopper) ARM — interactive smoke test

Quick interactive walkthrough for running paper2md on Sol's Grace
Hopper (`-p arm`) node. Sister doc: `SOL_SETUP.md` (the long-form
setup guide); batch counterpart: `sol_batch.sbatch`.

## Before you start: A100 might be easier for your first test

GH200 is the better long-term fit for paper2md (more VRAM, exclusive
node, aarch64 matches the project's reference dev box), but for a
*first* smoke test on Sol the A100 partition is almost always the
easier path to iterate on:

| Dimension | A100 partition (4× A100 80 GiB) | GH200 (`-p arm`) |
|---|---|---|
| Architecture | x86_64 — mainstream; every wheel and stack-trace you'll google was built and tested for it | aarch64 — narrower ecosystem; more chance of an obscure wheel/build issue |
| Queue time | Multiple nodes × 4 GPUs each → typically minutes | One machine on the cluster → can be hours during busy times |
| Failure cost | Re-allocate in minutes if something blows up | Re-allocate after another long queue wait |
| VRAM | 80 GB HBM2e — fits Qwen3-VL-32B BF16 with ~16 GB free | 96 GB HBM3 — same model, ~30 GB free; can host 72B at 4-bit |
| Sharing | Share node with up to 3 other users (one GPU each) | Exclusive — whole machine is yours |
| CPU for marker | 48 cores AMD EPYC | 72 cores NVIDIA Grace ARM |

**The dimensions that dominate for a smoke test are queue time and
failure cost.** Smoke testing is iterative: build env → run → hit a
bug → fix → re-run. On A100 each cycle is ~5 minutes from `salloc` to
running pipeline. On GH200 each cycle can be 1–4 hours of queue wait,
and if you blow up the env mid-job you've burned that wait.

GH200's advantages — 96 GB HBM3, exclusive node, NVLink-C2C unified
memory — only start mattering when you (a) run models bigger than 32B
BF16, (b) saturate the KV cache with high `--table-workers`
concurrency, or (c) want stable wall-clock numbers free of GPU
neighbors. None of that applies to a smoke test.

**Practical recommendation:**
1. **Smoke test on A100** first — follow `SOL_SETUP.md`. Get one paper
   through end-to-end. Validate timing.
2. **Move to GH200** for production batches if you want the
   exclusive-node guarantee or plan to run 72B models. A100 is fine
   for production too if Qwen3-VL-32B BF16 is your default.
3. **If you do both**, build separate conda envs at separate paths
   (`paper2md-x86_64` and `paper2md-arm`) so they don't stomp each
   other when activated from `/scratch` — see §3 below.

If you've already done the A100 smoke test, or you're comfortable
with aarch64 already, continue below.

## Why this node is a good fit

| Spec | Sol `arm` partition node | Why it matters for paper2md |
|---|---|---|
| Architecture | aarch64 (NVIDIA Grace ARM CPU) | `environment-gpu.yml` already targets aarch64 (project's reference dev box is the DGX Spark, also aarch64) — no env changes needed. |
| GPU | 1× NVIDIA GH200 superchip — Hopper sm_90 GPU + Grace ARM CPU on one package, with **96 GB HBM3** GPU memory plus 480 GiB LPDDR5X CPU memory linked via NVLink-C2C (Sol's "GH200 480GB" labels the LPDDR tier, not the HBM tier) | Qwen3-VL-32B-Instruct BF16 (~64 GB weights) fits with ~30 GB HBM free for KV cache. Could host Qwen3-VL-72B at 4-bit with room to spare. |
| CPU | 72 cores (NVIDIA Grace, aarch64) | Marker's surya OCR pass parallelizes across CPUs — 72 cores is generous. |
| Host memory | 512 GiB | Conda env + HF cache + marker resident set + OS all fit easily. |
| Allocation | **Exclusive node** — one user owns the whole machine. | No `--mem-per-cpu` arithmetic; just take everything. |

The catch: aarch64 binaries are not interchangeable with x86_64. **The
conda env must be built on aarch64** (either inside this allocation or
on another aarch64 box). If the Sol login nodes are x86_64 and you
built the env there, you'll see ImportErrors for compiled wheels
(torch, vllm) inside the job. Rebuild from inside the allocation if
that happens — see §3.

**If `arm` is queued.** Sol's other GPU partitions can host the same
pipeline with minor adjustments — the GPU A100 partition (4× A100
80 GiB, x86_64) is the closest fit; GPU MIG (sliced A100 at 10/20 GiB)
only works with the AWQ 4-bit fallback (see SOL_SETUP.md §4); GPU
MI200 (AMD) needs a ROCm rebuild and is not recommended. On any
non-`arm` partition, build the conda env to a different path (e.g.
`/scratch/$USER/conda/paper2md-x86_64`) so aarch64 and x86_64 envs
don't collide.

## 1. Request the interactive allocation

Either the Sol wrapper (recommended for simplicity):

```bash
interactive -G 1 -p arm --mem=0 -c 72
```

Or raw SLURM:

```bash
salloc -p arm -G 1 --mem=0 -c 72 --time=04:00:00
```

Knobs:
- `-p arm` — the Grace Hopper partition. Required; `arm` nodes are
  excluded from the general partition because they're a different
  architecture.
- `-G 1` — the one GPU on the node.
- `--mem=0` — all node memory. The node is exclusive, so you may as
  well take it all.
- `-c 72` — all 72 Grace ARM CPU cores. Marker uses them.
- `--time=04:00:00` — 4 hours; smoke-test for one paper finishes in
  ~15–25 min once vLLM is warm. Buffer is for env build + cold-start
  + iterating.

When the prompt returns on a compute node, you have the machine.

## 2. Verify you're actually on aarch64 with a GPU

```bash
hostname
uname -m                 # must print: aarch64
nvidia-smi               # must show one GH200 (Hopper-class, 96 GB HBM3)
echo $SLURM_JOB_GPUS     # should be a GPU index (e.g. "0")
```

If `uname -m` reports `x86_64`, your `salloc` landed on the wrong
partition — exit and re-request with `-p arm`.

## 3. Bring the conda env up on aarch64

If you've never built the env on aarch64 before (i.e. your previous
work was on the x86_64 login node), do this **once** from inside the
arm allocation:

```bash
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load mamba/latest             # adjust to whatever `module avail mamba` shows

# Build to /scratch (large quota, fast filesystem)
export HF_HOME=/scratch/$USER/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

cd /scratch/$USER/paper2md-workspace/paper2md
mamba env create -f environment-gpu.yml -p /scratch/$USER/conda/paper2md-arm
```

**Note the `-arm` suffix.** Keep separate env paths per architecture
so a future x86_64 job doesn't try to import aarch64 wheels.

If you already built the env on aarch64 in a previous arm allocation:

```bash
module load mamba/latest
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/$USER/conda/paper2md-arm
export HF_HOME=/scratch/$USER/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

CUDA: torch 2.10 + cu130 wheels in the env carry their own CUDA
runtime, so you do **not** need a `module load cuda/...`. If
`torch.cuda.is_available()` is False below, only then try
`module load cuda/12.x` to satisfy any system-driver mismatch.

## 4. 60-second pre-flight before burning compute

Don't discover env bugs at minute 18 of marker. Cheap checks first:

```bash
# 1. Torch sees the GPU
python -c "import torch; assert torch.cuda.is_available(); \
           print('torch:', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))"
# expected: torch: 2.10.0+cu130 | gpu: NVIDIA GH200 ...   (or similar)

# 2. All paper2md imports resolve on aarch64
python -c "import marker, surya, fitz, openai, vllm, docling; print('imports OK')"

# 3. paper2md help renders
cd /scratch/$USER/paper2md-workspace/paper2md
python src/paper2md.py --help | head -20
```

Any failure here:
- `cuda.is_available() == False` → `nvidia-smi` first; if that works,
  driver/runtime mismatch — try `module load cuda/12.4` (or whatever
  `module avail cuda` lists).
- ImportError on `vllm` or `torch` → env was built on x86_64. Rebuild
  per §3 with the `-arm` suffix path.
- `vllm` imports but warns about `libcudart.so.12` → expected, see
  the env file comment at line 53; ignore.

## 5. Start vLLM and verify the endpoint *before* running paper2md

```bash
mkdir -p /scratch/$USER/logs
nohup vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  > /scratch/$USER/logs/vllm-arm-interactive.log 2>&1 &
echo $! > /scratch/$USER/logs/vllm-arm-interactive.pid
```

`--gpu-memory-utilization 0.85` (vs the SOL_SETUP.md default of 0.80)
because GH200 has 96 GB; you can afford a larger KV cache.

Cold-start for a 32B BF16 model is 5–10 min. Poll instead of guessing:

```bash
DEADLINE=$(( $(date +%s) + 20 * 60 ))
until curl -fsS http://localhost:8000/v1/models | grep -q Qwen3-VL; do
    if (( $(date +%s) > DEADLINE )); then
        echo "vLLM did not come up. Last 80 log lines:"
        tail -80 /scratch/$USER/logs/vllm-arm-interactive.log
        break
    fi
    if ! kill -0 "$(cat /scratch/$USER/logs/vllm-arm-interactive.pid)" 2>/dev/null; then
        echo "vLLM died during startup. Last 80 log lines:"
        tail -80 /scratch/$USER/logs/vllm-arm-interactive.log
        break
    fi
    echo "  $(date +%H:%M:%S) waiting..."
    sleep 30
done
echo "vLLM up."
```

End-to-end sanity check (text-only — confirms the chat endpoint
parses OpenAI-format requests):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-32B-Instruct",
       "messages":[{"role":"user","content":"reply with the single word OK"}],
       "max_tokens":5}' \
  | python -m json.tool
```

If `choices[0].message.content` contains `OK`, the VLM stack is
fully wired and you can run paper2md.

## 6. Run paper2md against one paper

```bash
cd /scratch/$USER/paper2md-workspace/paper2md
python src/paper2md.py examples/<known-good>.pdf -o out/smoke-arm/
```

Provider auto-pairs with the CUDA backend, so no `--provider`
needed. Expected wall-clock on GH200 (BATCH.md throughput model,
scaled for the faster CPU): ~10–18 min for a 16-page paper once
vLLM is warm.

Watch progress in another terminal (login node, no extra allocation
needed):

```bash
ssh <asurite>@sol.asu.edu
ssh <compute-node-from-step-2>          # name from `hostname` earlier
tail -f /scratch/$USER/paper2md-workspace/paper2md/out/smoke-arm/<paper>/...
# or just watch nvidia-smi:
watch -n 5 nvidia-smi
```

## 7. Verify output

```bash
ls out/smoke-arm/
head -40 out/smoke-arm/<paper>/<paper>.md
```

Same checks as `SOL_SETUP.md` §2.5: YAML front-matter at top, citation
on the first non-frontmatter line, GitHub-flavored markdown tables,
`assets/` directory containing figures referenced in the body.

## 8. Tear down

```bash
kill $(cat /scratch/$USER/logs/vllm-arm-interactive.pid)
exit                  # releases the salloc allocation
```

`exit` alone is sufficient (SLURM kills the whole job's process group),
but explicit kill is hygiene if you plan to re-run paper2md without
restarting the salloc.

## GH200-specific notes

- **96 GB VRAM headroom unlocks bigger models.** If 32B BF16 isn't
  giving you the table fidelity you want, try
  `Qwen/Qwen3-VL-72B-Instruct` (4-bit) — fits with room to spare.
  Pass `--model Qwen/Qwen3-VL-72B-Instruct-AWQ --quantization awq`
  to vllm and `VLM_MODEL=...` to paper2md.
- **FP8 might work on sm_90.** The env file comment about FP8
  crashing is for the Spark's sm_121, not Sol's sm_90. If you want
  to experiment, try `vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8` —
  cuts VRAM in half and Hopper has native FP8 tensor cores. Validate
  table fidelity carefully against a BF16 baseline before trusting
  it (see USAGE.md §4.2.4).
- **Marker can use all 72 cores.** No flag needed; surya parallelizes
  automatically based on `os.cpu_count()`.
- **Exclusive node, no contention.** Unlike shared GPU partitions,
  you don't have to worry about another user's process starving your
  vLLM. Wall-clock numbers should be more stable.
- **Queue time can be longer.** The `arm` node is one machine on the
  cluster — if someone else is using it, you wait. `squeue -p arm`
  shows current state. `squeue -u $USER --start` estimates your
  start time.
