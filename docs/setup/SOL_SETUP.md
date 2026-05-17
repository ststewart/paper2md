# Running paper2md on ASU Sol — setup guide

Goal: get one paper through the pipeline end-to-end on a Sol GPU node
using the documented CUDA path (vLLM + Qwen3-VL-32B-Instruct BF16). This
is the smoke-test setup. Once this works, see "Next steps" for batch
processing.

Target GPU: NVIDIA H100 or A100 **80 GB** (the 32B BF16 model needs
~64 GB VRAM; A100 40 GB will not fit — use AWQ 4-bit on those, see
"Smaller GPU fallback" at the bottom). Replace `<asurite>` with your ASU
NetID throughout.

> **GH200 vs A100 — which to test on first?** This doc targets the
> A100 partition (x86_64, 4× A100 80 GiB per node). Sol also has a
> Grace Hopper / GH200 ARM node — see `SOL_ARM_INTERACTIVE.md`. For
> *first-time* paper2md testing on Sol, A100 is generally the easier
> path: shorter queues, mainstream x86_64 ecosystem, faster failure
> recovery. GH200 wins on raw spec (96 GB HBM3, exclusive node) but
> the queue can be hours and aarch64 has more obscure wheel/build
> edge cases. See `SOL_ARM_INTERACTIVE.md` § "Before you start" for
> the full tradeoff table.

---

## 1. One-time login-node setup

Everything in this section runs on the Sol login node (`sol.asu.edu`)
and is done **once**. Compute jobs come later.

### 1.1 Pick filesystems

Sol has tight `/home` quotas; conda envs and especially Hugging Face
weight caches are large. Put everything on `/scratch`.

```bash
ssh <asurite>@sol.asu.edu

# Working directory
mkdir -p /scratch/$USER/paper2md-workspace
cd /scratch/$USER/paper2md-workspace

# Hugging Face cache (marker weights ~5 GB, Qwen3-VL-32B BF16 ~64 GB)
mkdir -p /scratch/$USER/hf-cache
```

Add the cache var to your shell so every future job picks it up:

```bash
echo 'export HF_HOME=/scratch/$USER/hf-cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub' >> ~/.bashrc
source ~/.bashrc
```

### 1.2 Get the code

```bash
cd /scratch/$USER/paper2md-workspace
# Either clone from your remote, or rsync from your laptop:
#   rsync -av --exclude='out/' --exclude='__pycache__' \
#         /local/path/to/paper2md/ <asurite>@sol.asu.edu:/scratch/$USER/paper2md-workspace/paper2md/
git clone <your-paper2md-repo-url> paper2md
cd paper2md
```

### 1.3 Create the conda environment

Sol exposes conda/mamba via the module system. Check what's available:

```bash
module avail mamba 2>&1 | head
module avail miniconda 2>&1 | head
# pick whichever loads — common names on Sol: mamba/latest, anaconda3/...
module load mamba/latest    # adjust to whatever appears above
```

Create the env from the project's pinned CUDA file:

```bash
mamba env create -f environment-gpu.yml -p /scratch/$USER/conda/paper2md
mamba activate /scratch/$USER/conda/paper2md
```

Why `-p` instead of `-n`: the env on `/scratch` doesn't count against
your `/home` quota and is faster on Sol's parallel filesystem.

Verify CUDA is wired up:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: e.g. 2.4.x True
```

If `cuda.is_available()` is `False` on the login node, that is normal —
login nodes have no GPU. The check that matters happens inside the
SLURM allocation in §2.

### 1.4 Pre-download model weights (do this on the login node, not in your job)

Compute-node walltime should not be spent on weight downloads. The
32B model is ~64 GB; pulling it once on the login node saves ~10 min
of compute time per job and avoids any compute-node egress weirdness.

```bash
mamba activate /scratch/$USER/conda/paper2md
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct
# verify it landed on /scratch:
du -sh /scratch/$USER/hf-cache/hub/models--Qwen--Qwen3-VL-32B-Instruct/
```

Marker downloads its own surya/text/layout weights on first run
(~5 GB). You can either let that happen inside the smoke-test job
(adds ~3 min the first time) or pre-pull on the login node:

```bash
python -c "from marker.models import create_model_dict; create_model_dict()"
```

### 1.5 (Optional) Pre-flight the conda env without a GPU

```bash
python src/paper2md.py --help | head -40
```

If this prints the help text, your conda env is structurally OK.

---

## 2. Interactive smoke-test session

Get a single GPU interactively, start vLLM in the background, run
paper2md against one paper from `examples/`, verify the output,
release the allocation.

### 2.1 Request a GPU allocation

Sol uses SLURM. The exact partition/QoS depends on your allocation —
ask your PI or `module avail` won't tell you, but `sinfo` will. The
canonical request shape for an H100 or A100 80 GB:

```bash
salloc \
  --partition=general \
  --qos=public \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=04:00:00
```

Knobs:

- `--gres=gpu:a100:1` — one A100. For H100 use `--gres=gpu:h100:1`. If
  you get "QOSMaxGRESPerJob" or similar, your allocation only allows
  certain GPU types — substitute what you have access to.
- `--mem=128G` — host RAM. vLLM weight loading + marker (~5 GB
  resident) + the OS + the HF download cache need elbow room.
- `--cpus-per-task=8` — marker's surya OCR pass parallelizes across
  CPUs.
- `--time=04:00:00` — 4 hours; smoke-test for one paper finishes in
  ~30 min once vLLM is warm. The remainder is a buffer.
- Add `--account=<group>` if your allocation requires it.

Once SLURM grants the job, your shell is now on the compute node.

### 2.2 Re-load modules and the env on the compute node

SLURM does not inherit `module load` from the login session by
default. Re-load:

```bash
module load mamba/latest          # same one you used in §1.3
module load cuda/12.4              # or whichever CUDA matches your env
mamba activate /scratch/$USER/conda/paper2md

export HF_HOME=/scratch/$USER/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# verify the GPU is visible from inside the job:
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA A100-SXM4-80GB   (or H100 etc.)
```

### 2.3 Start vLLM in the background

```bash
cd /scratch/$USER/paper2md-workspace/paper2md
mkdir -p /scratch/$USER/logs

nohup vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  > /scratch/$USER/logs/vllm.log 2>&1 &

echo $! > /scratch/$USER/logs/vllm.pid
```

Wait for it to come up. Cold-start for a 32B BF16 model on Sol's
filesystem is **5–10 minutes**. Poll instead of guessing:

```bash
until curl -s http://localhost:8000/v1/models | grep -q Qwen3-VL; do
  echo "waiting for vLLM... $(date +%H:%M:%S)"
  sleep 30
done
echo "vLLM up."
```

If it never comes up, `tail -200 /scratch/$USER/logs/vllm.log`. Common
failures: insufficient VRAM (drop `--gpu-memory-utilization` to 0.70,
or you're on a 40 GB card and need the AWQ fallback), CUDA version
mismatch (re-check `module load cuda/...`), HF cache miss because
`HF_HOME` wasn't exported in the new shell.

### 2.4 Run paper2md against one paper

```bash
cd /scratch/$USER/paper2md-workspace/paper2md
python src/paper2md.py examples/<known-good>.pdf -o out/smoke/
# pick any PDF in examples/ — see USAGE.md §2 for the rationale on each.
```

The provider auto-pairs with the CUDA backend (lines 3025–3029 of
`src/paper2md.py`), so you do not need `--provider vllm` explicitly.
If you want to be explicit:

```bash
python src/paper2md.py examples/<known-good>.pdf -o out/smoke/ \
  --backend cuda --provider vllm
```

Expected wall-clock for one ~16-page paper (per BATCH.md §1):
- Marker: 10–20 min (the dominant cost on CUDA)
- VLM hooks combined: 30–90 sec (vLLM batches concurrent table calls)
- Total: ~12–22 min after vLLM is already warm.

### 2.5 Verify the output

```bash
ls out/smoke/
# expected layout:
#   out/smoke/<paper>/
#     <paper>.md
#     assets/
#     manifest.jsonl   (when running batch mode; absent for single-file)
#     ...

head -40 out/smoke/<paper>/<paper>.md
# YAML front-matter at top: source_pdf, overall, grade, etc.
# Followed by the citation, headings, and converted body.
```

Sanity checks:
- The first non-frontmatter line should be a journal-style citation
  (hook 0).
- Tables should be GitHub-flavored markdown tables, not gibberish.
- `assets/` should contain figure images that match the body's
  `![](...)` references.

If any of those are wrong, see USAGE.md §7 (full troubleshooting
tree).

### 2.6 Tear down

```bash
kill $(cat /scratch/$USER/logs/vllm.pid)
exit         # release the salloc allocation
```

If you exit `salloc` without killing vLLM, SLURM tears the whole job
down anyway — vLLM dies with the allocation. The explicit kill is
just hygiene.

---

## 3. Common Sol-specific pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `cuda.is_available() == False` inside salloc | `--gres=gpu:...` missing or wrong; or `cuda/...` module not loaded | Check `nvidia-smi`; re-`module load cuda/X.Y` after the env activates |
| vLLM "no space left on device" | HF cache landed in `/home` (16 GB quota) | Re-export `HF_HOME=/scratch/...` and `huggingface-cli download` again |
| vLLM stalls 8 min then OOMs | A100 40 GB or smaller | See "Smaller GPU fallback" below — switch to AWQ 4-bit |
| `module: command not found` | Lmod not initialized in non-interactive shell | `source /etc/profile.d/lmod.sh` (or whatever Sol's docs specify) |
| Marker re-downloads weights every job | `HF_HOME` not preserved across jobs | Put the export in `~/.bashrc` (done in §1.1) |
| salloc waits in queue forever | Wrong partition/QoS for your allocation | Ask your PI; try other partitions; lower walltime |
| `OPENAI_API_KEY not set` error | You set `--provider openai` somewhere | Drop the flag — vLLM is the default on CUDA |
| Conda env activation extremely slow | Env is on `/home` instead of `/scratch` | Recreate with `-p /scratch/$USER/conda/paper2md` |

---

## 4. Smaller GPU fallback (40 GB cards)

Qwen3-VL-32B BF16 won't fit in 40 GB. Two options:

**Option A: AWQ 4-bit quant.** ~99% of full-precision quality with
roughly half the VRAM. Override the model:

```bash
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct-AWQ
vllm serve Qwen/Qwen3-VL-32B-Instruct-AWQ \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --quantization awq
```

Then point paper2md at it:

```bash
export VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-AWQ
python src/paper2md.py examples/<paper>.pdf -o out/smoke/
```

**Option B: smaller model.** Drop to Qwen2.5-VL-7B for capacity-bound
nodes. Quality drops on dense tables (USAGE.md §4.2.1, ≤16 GB row);
acceptable for figure captioning but you should spot-check tables.

---

## 5. Next steps — batch processing on Sol

Once the smoke test passes, switch from the interactive `salloc`
walkthrough to an unattended `sbatch` job.

A ready-to-edit template lives at **`docs/setup/sol_batch.sbatch`** in this
repo. It encodes everything from §2 (module loads, `HF_HOME`,
background vLLM with cold-start polling and a 20-min timeout, cleanup
trap so vLLM is killed on any exit, post-run summary from
`manifest.jsonl`) plus the SLURM directives for a 2-day run.

Workflow:

1. Read `docs/BATCH.md` for the throughput model and per-paper
   wall-clock budget. Use it to size `--time` for your corpus.
2. Copy `docs/setup/sol_batch.sbatch` somewhere you can edit (e.g.
   `/scratch/$USER/paper2md-workspace/`) and fill in the `EDIT_ME`
   block at the top — corpus path, output path, `CONDA_ENV`,
   optionally `--account`, `--mail-user`, partition/QoS.
3. Submit: `sbatch sol_batch.sbatch`. The job ID lets you tail the
   logs in real time:
   ```
   tail -f /scratch/$USER/logs/paper2md-<jobid>.out
   tail -f /scratch/$USER/logs/vllm-<jobid>.log
   ```
4. Monitor progress via `manifest.jsonl` in the output directory —
   one JSON line per paper with status and timing. To find failures:
   ```
   jq -c 'select(.status != "ok")' <output-dir>/manifest.jsonl
   ```
   The script also prints a status histogram and the first 10
   failures at the end of the SLURM `.out` log.
5. To resume after a partial run, point a new sbatch at the same
   `OUTPUT_DIR` — paper2md skips PDFs whose output dir already
   exists (see USAGE.md "Batch mode").

---

## 6. Where to look when stuck

- `docs/USAGE.md` — flag reference, hook-by-hook behavior, full
  troubleshooting tree (§7).
- `docs/BATCH.md` — operational runbook, throughput model, hand-off
  to a frontier model when the local pipeline is the wrong tool.
- `CLAUDE.md` — quick-start cheat sheet at the project root.
- vLLM log: `/scratch/$USER/logs/vllm.log`.
- Sol's RC docs: <https://asurc.atlassian.net/wiki/spaces/RC/>.
