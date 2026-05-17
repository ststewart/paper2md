#!/bin/bash
# paper2md batch run on ASU Sol — sbatch template.
#
# Usage:
#   1. Copy this file somewhere you can edit (e.g. /scratch/$USER/paper2md-workspace/).
#   2. Edit the EDIT_ME block below (corpus path, output path, account if needed).
#   3. Submit:   sbatch sol_batch.sbatch
#   4. Monitor:  tail -f /scratch/$USER/logs/paper2md-<jobid>.out
#                                 plus /scratch/$USER/logs/vllm-<jobid>.log
#                jq -c 'select(.status != "ok")' <output-dir>/manifest.jsonl
#
# Companion doc: docs/setup/SOL_SETUP.md (smoke-test walkthrough this is built on).

# ---------- SLURM directives ----------
#SBATCH --job-name=paper2md-batch
#SBATCH --partition=general              # adjust to your allocation; check `sinfo`
#SBATCH --qos=public                     # adjust to your allocation
#SBATCH --gres=gpu:a100:1                # or gpu:h100:1; need 80 GB VRAM for 32B BF16
#SBATCH --cpus-per-task=8                # marker's surya OCR pass parallelizes across CPUs
#SBATCH --mem=128G                       # vLLM weight load + marker + HF cache headroom
#SBATCH --time=2-00:00:00                # 2 days; raise/lower to match corpus size
#SBATCH --output=/scratch/%u/logs/paper2md-%j.out
#SBATCH --error=/scratch/%u/logs/paper2md-%j.err
##SBATCH --account=<your-group>          # uncomment + fill if your allocation requires it
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<asurite>@asu.edu

set -euo pipefail

# ====================================================================
# EDIT_ME — paths and run parameters
# ====================================================================
CORPUS_DIR="/scratch/$USER/corpora/my-aas-papers"           # input PDFs
OUTPUT_DIR="/scratch/$USER/paper2md-out/$(date +%Y%m%d)"    # results land here
PAPER2MD_DIR="/scratch/$USER/paper2md-workspace/paper2md"   # cloned repo
CONDA_ENV="/scratch/$USER/conda/paper2md"                   # conda env path
WORKERS=1                                                   # paper2md batch parallelism;
                                                            # single GPU + marker-bound,
                                                            # raise to 2 only if VRAM
                                                            # and CPU headroom allow.
VLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"                      # BF16; 40 GB cards: switch to
                                                            # Qwen/Qwen3-VL-32B-Instruct-AWQ
VLM_MAX_LEN=32768                                           # KV-cache budget vs context
VLM_GPU_UTIL=0.80                                           # AWQ on 40 GB: try 0.85
EXTRA_PAPER2MD_FLAGS=""                                     # e.g. "--hdf5" or "--no-pair"
# ====================================================================

# Logs
mkdir -p "/scratch/$USER/logs"
VLLM_LOG="/scratch/$USER/logs/vllm-${SLURM_JOB_ID}.log"
VLLM_PID_FILE="/scratch/$USER/logs/vllm-${SLURM_JOB_ID}.pid"

# Output dir
mkdir -p "$OUTPUT_DIR"

# Caches on /scratch (NOT /home — quota will bite you mid-batch)
export HF_HOME="/scratch/$USER/hf-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

# ---------- Modules + conda ----------
# Replace these names with whatever `module avail` shows on Sol.
# Lmod isn't always on $PATH in non-interactive shells:
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load mamba/latest        # adjust if Sol exposes a different name
module load cuda/12.4           # match your environment-gpu.yml CUDA

# `mamba activate` needs the shell hook in batch context:
eval "$(mamba shell hook --shell bash)"
mamba activate "$CONDA_ENV"

# ---------- Sanity checks before we burn an hour on cold-start ----------
echo "===== job ${SLURM_JOB_ID} on $(hostname) at $(date -Iseconds) ====="
echo "Corpus:       $CORPUS_DIR  ($(find "$CORPUS_DIR" -maxdepth 1 -name '*.pdf' | wc -l) PDFs)"
echo "Output:       $OUTPUT_DIR"
echo "VLM model:    $VLM_MODEL"
echo
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA inside the job'; \
           print('torch:', torch.__version__, 'gpu:', torch.cuda.get_device_name(0))"
test -d "$CORPUS_DIR" || { echo "FATAL: corpus not found at $CORPUS_DIR"; exit 1; }

# ---------- Cleanup trap (runs on any exit, including SLURM cancel) ----------
cleanup() {
    local rc=$?
    if [[ -f "$VLLM_PID_FILE" ]]; then
        local pid
        pid=$(cat "$VLLM_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$(date -Iseconds)] tearing down vLLM (pid=$pid)"
            kill "$pid" 2>/dev/null || true
            sleep 5
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    echo "[$(date -Iseconds)] job exiting with rc=$rc"
}
trap cleanup EXIT

# ---------- Start vLLM in the background ----------
echo "[$(date -Iseconds)] starting vLLM ($VLM_MODEL)"
QUANT_FLAG=""
if [[ "$VLM_MODEL" == *AWQ* ]]; then
    QUANT_FLAG="--quantization awq"
fi
nohup vllm serve "$VLM_MODEL" \
    --port 8000 \
    --max-model-len "$VLM_MAX_LEN" \
    --gpu-memory-utilization "$VLM_GPU_UTIL" \
    $QUANT_FLAG \
    > "$VLLM_LOG" 2>&1 &
echo $! > "$VLLM_PID_FILE"

# Poll for readiness with a hard timeout (32B BF16 takes 5-10 min cold)
echo "[$(date -Iseconds)] waiting for vLLM (timeout 20 min)"
DEADLINE=$(( $(date +%s) + 20 * 60 ))
while ! curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; do
    if (( $(date +%s) > DEADLINE )); then
        echo "FATAL: vLLM did not come up within 20 min. Last 80 log lines:"
        tail -80 "$VLLM_LOG"
        exit 2
    fi
    if ! kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
        echo "FATAL: vLLM process died during startup. Last 80 log lines:"
        tail -80 "$VLLM_LOG"
        exit 2
    fi
    sleep 30
    echo "  ... still waiting at $(date +%H:%M:%S)"
done
echo "[$(date -Iseconds)] vLLM is up."

# ---------- Run paper2md in batch mode ----------
cd "$PAPER2MD_DIR"
echo "[$(date -Iseconds)] starting paper2md --batch"
python src/paper2md.py \
    --batch "$CORPUS_DIR" \
    -o "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --backend cuda \
    --provider vllm \
    --model "$VLM_MODEL" \
    $EXTRA_PAPER2MD_FLAGS

# ---------- Post-batch summary from manifest.jsonl ----------
MANIFEST="$OUTPUT_DIR/manifest.jsonl"
if [[ -f "$MANIFEST" ]]; then
    echo
    echo "===== batch summary ====="
    python - <<PY
import json, collections, pathlib
p = pathlib.Path("$MANIFEST")
counts = collections.Counter()
fails = []
for line in p.read_text().splitlines():
    if not line.strip(): continue
    rec = json.loads(line)
    status = rec.get("status", "unknown")
    counts[status] += 1
    if status != "ok":
        fails.append(rec)
print(f"total: {sum(counts.values())}")
for k, v in counts.most_common():
    print(f"  {k}: {v}")
if fails:
    print("\nfailures (first 10):")
    for r in fails[:10]:
        print(f"  - {r.get('source_pdf', '?')}: {r.get('error', r.get('status'))}")
PY
else
    echo "WARN: no manifest.jsonl at $MANIFEST"
fi

echo "[$(date -Iseconds)] done."
