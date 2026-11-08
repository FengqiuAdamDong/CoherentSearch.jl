#!/usr/bin/env bash
#
# Submit a full demodulated search as a SLURM job array: one array element per
# batch of hybrid-grid trials.  The grid is circular below p_break and
# ajs/Taylor above it (demod/nsns_grid.py -mode hybrid).
#
# Each element runs the in-memory driver on BATCH_SIZE grid rows; the array size
# is ceil(N / BATCH_SIZE).  A dependent job combines the per-point .cohout files.
#
# Examples:
#   INF=/path/OBS.inf BATCH_SIZE=64 CPUS_PER_TASK=8 \
#       bash demod/slurm_demod_sweep.sh
#   DRY_RUN=1 INF=/path/OBS.inf BATCH_SIZE=64 bash demod/slurm_demod_sweep.sh
#   GRID_CSV_LIST=/path/circ.csv:/path/ajs.csv BATCH_SIZE=64 \
#       INF=/path/OBS.inf bash demod/slurm_demod_sweep.sh
#
# PARTITION, ACCOUNT, QOS, MEM and TIME are site settings.  Set MODULE_LOAD if
# Julia/Python need `module load`.
#
# Requires: bash 4+, Julia with the CoherentSearch.jl project, and a Python with
# numpy/scipy/matplotlib/yaml for nsns_grid.py and combine_cohout.py.

set -euo pipefail

# Input and batching.
: "${INF:=/home/fadong/Documents/CoherentSearch.jl/demod/inj.inf}"
: "${DAT:=}"                    # default: ${INF%.inf}.dat
: "${INBASE:=}"                 # default: basename of DAT without .dat
: "${BATCH_SIZE:=32}"           # grid rows per array element
: "${MAX_ARRAY_TASKS:=1000}"    # refuse an array larger than this
: "${OUTDIR:=}"                 # default: $PWD/nsns_slurm_<obs>
: "${GRID_CSV_LIST:=}"          # colon-separated CSVs; skips derivation

# Resources (site settings; no cluster defaults).
: "${CPUS_PER_TASK:=8}"
: "${MEM:=16G}"
: "${TIME:=04:00:00}"
: "${PARTITION:=}"
: "${ACCOUNT:=}"
: "${QOS:=}"

# Hybrid grid derivation, forwarded to run_nsns_sweep.sh.
: "${P0:=1.0}"
: "${PHASE_TOL:=0.1}"
: "${PO:=}"
: "${ECC:=}"
: "${SINI:=}"
: "${OMEGA_PERI:=}"
: "${DROPPCT:=10}"
: "${NPHASE:=120}"
: "${NSAMP:=4000}"
: "${P_BREAK:=}"
: "${BREAK_COV:=}"

# Search.
: "${SEARCH_ARGS:=}"            # whitespace-split, e.g. "--threshold 6"
: "${DEMOD_CONCURRENCY:=0}"     # 0 = one slot per Julia thread

# Tools.
: "${JULIA:=julia}"
: "${PYTHON:=python3}"
: "${COH_REPO:=}"
: "${MODULE_LOAD:=}"

: "${DRY_RUN:=0}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

if [ -z "$COH_REPO" ]; then
    COH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

case "$BATCH_SIZE" in
    ''|*[!0-9]*) die "BATCH_SIZE must be a positive integer (got '$BATCH_SIZE')" ;;
esac
[ "$BATCH_SIZE" -gt 0 ] || die "BATCH_SIZE must be > 0"
case "$MAX_ARRAY_TASKS" in
    ''|*[!0-9]*) die "MAX_ARRAY_TASKS must be a positive integer" ;;
esac
[ "$MAX_ARRAY_TASKS" -gt 0 ] || die "MAX_ARRAY_TASKS must be > 0"
case "$CPUS_PER_TASK" in
    ''|*[!0-9]*) die "CPUS_PER_TASK must be a positive integer" ;;
esac
[ "$CPUS_PER_TASK" -gt 0 ] || die "CPUS_PER_TASK must be > 0"

[ -f "$INF" ] || die "INF not found: $INF"
if [ -z "$DAT" ]; then DAT="${INF%.inf}.dat"; fi
[ -f "$DAT" ] || die "DAT not found: $DAT (set DAT explicitly if it is not beside INF)"
INF="$(cd "$(dirname "$INF")" && pwd)/$(basename "$INF")"
DAT="$(cd "$(dirname "$DAT")" && pwd)/$(basename "$DAT")"
if [ -z "$INBASE" ]; then INBASE="$(basename "${DAT%.dat}")"; fi

if [ -z "$OUTDIR" ]; then OUTDIR="$PWD/nsns_slurm_${INBASE}"; fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"
ENVFILE="$OUTDIR/slurm_job.env"

if [ -n "$MODULE_LOAD" ]; then eval "$MODULE_LOAD"; fi

# --------------------------------------------------------------------------
# Grid: one hybrid derivation, or a caller-supplied list
# --------------------------------------------------------------------------
count_csv() {
    # Same blank-line rule as the Julia parser: line 1 is the header, and a
    # whitespace-only line is not a row.
    awk 'NR > 1 && NF > 0 { n++ } END { print n + 0 }' "$1"
}

check_csv() {
    local csv="$1" hdr
    [ -f "$csv" ] || die "grid CSV not found: $csv"
    hdr="$(head -1 "$csv" | tr -d '\r' | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    case "$hdr" in
        accel,jerk,snap|pb,x,at) ;;
        *) die "$csv: unknown header '$hdr'; expected accel,jerk,snap or pb,x,at" ;;
    esac
}

GRID_CSVS=()
TOTAL_POINTS=0
if [ -n "$GRID_CSV_LIST" ]; then
    IFS=: read -r -a GRID_CSVS <<< "$GRID_CSV_LIST"
    for csv in "${GRID_CSVS[@]}"; do
        check_csv "$csv"
        TOTAL_POINTS=$((TOTAL_POINTS + $(count_csv "$csv")))
    done
else
    # run_nsns_sweep.sh owns the grid derivation and its physics knobs.
    export INF DAT OUTDIR PYTHON JULIA COH_REPO
    export P0 PHASE_TOL PO ECC SINI OMEGA_PERI DROPPCT NPHASE NSAMP P_BREAK BREAK_COV
    log="$OUTDIR/derive_hybrid.log"
    note "deriving hybrid grid (circular below p_break, ajs above)"
    GRID_CSV="" MODE=hybrid GRIDNAME=nsns_grid GRID_ONLY=1 \
        bash "$COH_REPO/demod/run_nsns_sweep.sh" | tee "$log"
    mapfile -t GRID_CSVS < <(sed -n 's/^GRID_CSV=//p' "$log")
    [ "${#GRID_CSVS[@]}" -ge 1 ] || die "no GRID_CSV= lines from $log"
    for csv in "${GRID_CSVS[@]}"; do check_csv "$csv"; done
    TOTAL_POINTS="$(sed -n 's/^GRID_POINTS=//p' "$log" | tail -1)"
    [ -n "$TOTAL_POINTS" ] || die "no GRID_POINTS= line from $log"
fi
[ "$TOTAL_POINTS" -gt 0 ] || die "the grid has no points"

NBATCHES=$(( (TOTAL_POINTS + BATCH_SIZE - 1) / BATCH_SIZE ))
if [ "$NBATCHES" -gt "$MAX_ARRAY_TASKS" ]; then
    min_bs=$(( (TOTAL_POINTS + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
    die "$TOTAL_POINTS point(s) at BATCH_SIZE=$BATCH_SIZE need $NBATCHES array elements, above MAX_ARRAY_TASKS=$MAX_ARRAY_TASKS; use BATCH_SIZE >= $min_bs (or raise MAX_ARRAY_TASKS)"
fi

# --------------------------------------------------------------------------
# Worker environment
# --------------------------------------------------------------------------
{
    printf 'COH_REPO=%q\n' "$COH_REPO"
    printf 'DAT=%q\n' "$DAT"
    printf 'OUTDIR=%q\n' "$OUTDIR"
    printf 'LOGDIR=%q\n' "$LOGDIR"
    printf 'INBASE=%q\n' "$INBASE"
    printf 'GRID_CSVS=('
    for csv in "${GRID_CSVS[@]}"; do printf ' %q' "$csv"; done
    printf ' )\n'
    printf 'BATCH_SIZE=%q\n' "$BATCH_SIZE"
    printf 'TOTAL_POINTS=%q\n' "$TOTAL_POINTS"
    printf 'NBATCHES=%q\n' "$NBATCHES"
    printf 'CPUS_PER_TASK=%q\n' "$CPUS_PER_TASK"
    printf 'DEMOD_CONCURRENCY=%q\n' "$DEMOD_CONCURRENCY"
    printf 'SEARCH_ARGS=%q\n' "$SEARCH_ARGS"
    printf 'JULIA=%q\n' "$JULIA"
    printf 'PYTHON=%q\n' "$PYTHON"
    printf 'MODULE_LOAD=%q\n' "$MODULE_LOAD"
} > "$ENVFILE"

# --------------------------------------------------------------------------
# Submit the batch array and a dependent combine job
# --------------------------------------------------------------------------
SITE_ARGS=""
if [ -n "$PARTITION" ]; then SITE_ARGS="$SITE_ARGS --partition=$PARTITION"; fi
if [ -n "$ACCOUNT" ]; then SITE_ARGS="$SITE_ARGS --account=$ACCOUNT"; fi
if [ -n "$QOS" ]; then SITE_ARGS="$SITE_ARGS --qos=$QOS"; fi

SBATCH_ARGS=(
    --parsable
    --job-name="${INBASE}_demod_hybrid"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$TIME"
    --array="0-$((NBATCHES - 1))"
    --output="$LOGDIR/slurm-%A_%a.out"
    --error="$LOGDIR/slurm-%A_%a.err"
    $SITE_ARGS
)

note "grid: ${#GRID_CSVS[@]} CSV(s), $TOTAL_POINTS point(s)"
note "batches: $NBATCHES x up to $BATCH_SIZE point(s)"
note "output: $OUTDIR"

if [ "$DRY_RUN" = 1 ]; then
    note "DRY_RUN: would submit:"
    printf '   '
    printf '%q ' sbatch "${SBATCH_ARGS[@]}" \
        "$COH_REPO/demod/slurm_demod_batch.sh" "$ENVFILE"
    printf '\n'
    exit 0
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" \
    "$COH_REPO/demod/slurm_demod_batch.sh" "$ENVFILE")"
JOBID="${JOBID%%;*}"
note "submitted array job $JOBID"
note "queue: squeue -j $JOBID"
note "logs:  $LOGDIR"

COMBINE_ARGS=(
    --parsable
    --job-name="${INBASE}_demod_hybrid_combine"
    --nodes=1
    --ntasks=1
    --cpus-per-task=1
    --mem=4G
    --time=01:00:00
    --dependency="afterok:$JOBID"
    --output="$LOGDIR/combine_%j.out"
    --error="$LOGDIR/combine_%j.err"
    $SITE_ARGS
)
CID="$(sbatch "${COMBINE_ARGS[@]}" \
    "$COH_REPO/demod/slurm_demod_combine.sh" "$ENVFILE")"
CID="${CID%%;*}"
note "submitted combine job $CID (afterok:$JOBID)"
note "combined candidates: $OUTDIR/combined_cohout.txt"
