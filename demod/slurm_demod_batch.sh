#!/usr/bin/env bash
#
# One SLURM array element = one BATCH of hybrid-grid trials.
#
# The submitter passes a sourceable environment file; this worker runs the
# in-memory driver on the batch selected by SLURM_ARRAY_TASK_ID.  The batching
# happens inside the driver (`--demod-batch-*`), not by looping here.
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
if [ "$BATCH_NO" -gt "$NBATCHES" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID maps to batch $BATCH_NO, outside 1..$NBATCHES" >&2
    exit 1
fi

CPUS="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"

mkdir -p "$LOGDIR"
cd "$OUTDIR"

GRID_FLAGS=()
for csv in "${GRID_CSVS[@]}"; do
    GRID_FLAGS+=(--demod-grid "$csv")
done

echo ">> batch $BATCH_NO/$NBATCHES"

"$JULIA" --project="$COH_REPO" -t "$CPUS" \
    "$COH_REPO/bin/coherent_search.jl" "$DAT" "${GRID_FLAGS[@]}" \
    --demod-batch-size "$BATCH_SIZE" \
    --demod-batch-index "$BATCH_NO" \
    --demod-batch-total "$TOTAL_POINTS" \
    --demod-concurrency "$DEMOD_CONCURRENCY" \
    --noprogress --outdir "$OUTDIR" $SEARCH_ARGS \
    2>&1 | tee "$LOGDIR/batch_${BATCH_NO}.log"
