#!/usr/bin/env bash
#
# One SLURM array element = one BATCH of grid trials.
#
# The submitter (demod/slurm_demod_sweep.sh) passes a sourceable environment
# file; this worker runs the in-memory demodulation driver on exactly the batch
# selected by SLURM_ARRAY_TASK_ID.  It never loops over "one orbit per job": the
# driver's --demod-batch-size/--demod-batch-index does the batching, and
# --demod-concurrency parallelises the searches inside this process.
#
# Usage (normally via sbatch from the submitter):
#   sbatch ... slurm_demod_batch.sh /path/to/slurm_job.env

set -euo pipefail

ENVFILE="${1:?usage: slurm_demod_batch.sh ENVFILE}"
# shellcheck disable=SC1090
source "$ENVFILE"

if [ -n "${MODULE_LOAD:-}" ]; then eval "$MODULE_LOAD"; fi
: "${SLURM_ARRAY_TASK_ID:?slurm_demod_batch.sh must run under sbatch --array}"

BATCH_NO=$((SLURM_ARRAY_TASK_ID + 1))
if [ "$BATCH_NO" -lt 1 ] || [ "$BATCH_NO" -gt "$NBATCHES" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID maps to batch $BATCH_NO, outside 1..$NBATCHES" >&2
    exit 1
fi

CPUS="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"
LO=$(( (BATCH_NO - 1) * BATCH_SIZE + 1 ))
HI=$(( LO + BATCH_SIZE - 1 ))
if [ "$HI" -gt "$TOTAL_POINTS" ]; then HI=$TOTAL_POINTS; fi

mkdir -p "$LOGDIR"
cd "$OUTDIR"

GRID_FLAGS=()
for csv in "${GRID_CSVS[@]}"; do
    GRID_FLAGS+=(--demod-grid "$csv")
done

EXTRA=()
if [ "${GPU:-0}" = 1 ]; then EXTRA+=(--gpu); fi
if [ -n "${BLOCKSIZE:-}" ]; then EXTRA+=(--blocksize "$BLOCKSIZE"); fi

echo ">> batch $BATCH_NO/$NBATCHES: global rows $LO..$HI of $TOTAL_POINTS"
echo ">> $CPUS thread(s), --demod-concurrency=$DEMOD_CONCURRENCY, ${#GRID_CSVS[@]} grid CSV(s)"

"$JULIA" --project="$COH_REPO" -t "$CPUS" \
    "$COH_REPO/bin/coherent_search.jl" "$DAT" "${GRID_FLAGS[@]}" \
    --demod-batch-size "$BATCH_SIZE" \
    --demod-batch-index "$BATCH_NO" \
    --demod-batch-total "$TOTAL_POINTS" \
    --demod-concurrency "$DEMOD_CONCURRENCY" \
    --noprogress --outdir "$OUTDIR" \
    "${EXTRA[@]}" $SEARCH_ARGS \
    2>&1 | tee "$LOGDIR/batch_${BATCH_NO}.log"
