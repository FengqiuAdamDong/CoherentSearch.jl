#!/usr/bin/env bash
#
# Submit a full demodulated pulsar search as a SLURM job array: one array
# element per BATCH of grid trials, not one orbit per job.
#
# The work list is the flattened rows of one or more grid CSVs from
# demod/nsns_grid.py.  `BATCH_SIZE` rows are assigned to each array element;
# `--demod-batch-size/--demod-batch-index/--demod-batch-total` make the
# in-memory driver stream and process exactly that block, so every grid row is
# searched once across the array.  Each element is one Julia process and uses
# `CPUS_PER_TASK` threads internally.
#
# Grid models (`MODE`):
#   both      (default) two independent full grids, circular and ajs/Taylor
#   circular  exact Keplerian (pb,x,at) grid only
#   ajs       cubic accel/jerk/snap (Taylor) grid only
#   taylor    alias of ajs
#   hybrid    one derivation: circular below p_break, ajs at/above it
# `MODE=both` and `MODE=hybrid` pass two CSVs to every task; the row's model
# travels with it, so a batch may contain both.
#
# Examples:
#   INF=/path/OBS.inf MODE=both BATCH_SIZE=64 bash demod/slurm_demod_sweep.sh
#   DRY_RUN=1 MODE=both BATCH_SIZE=64 bash demod/slurm_demod_sweep.sh
#   GRID_CSV_LIST=/path/circ.csv:/path/ajs.csv BATCH_SIZE=32 \
#       bash demod/slurm_demod_sweep.sh
#   RESTART=1 RERUN_BATCHES="3 7" OUTDIR=/path/OUT bash demod/slurm_demod_sweep.sh
#
# Site configuration is all environment variables; there are no cluster
# defaults for PARTITION/ACCOUNT/QOS.  The submitter runs on the login node;
# compute nodes need the same Julia/Python/modules (set MODULE_LOAD).
#
# Requires: bash 4+, Julia with the CoherentSearch.jl project, and a Python
# with numpy/scipy/matplotlib/yaml for nsns_grid.py and combine_cohout.py.

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
: "${INF:=/home/fadong/Documents/CoherentSearch.jl/demod/inj.inf}"
: "${DAT:=}"                     # default: ${INF%.inf}.dat
: "${INBASE:=}"                  # default: basename of DAT without .dat

: "${MODE:=both}"                # both | circular | ajs | taylor | hybrid

: "${BATCH_SIZE:=32}"            # grid rows per SLURM array element
: "${MAX_ARRAY_TASKS:=1000}"     # fail if ceil(N/BATCH_SIZE) exceeds this
: "${MAX_CONCURRENT:=0}"         # 0 = no % cap on the array

: "${CPUS_PER_TASK:=8}"          # Julia threads per task
: "${MEM:=16G}"
: "${TIME:=04:00:00}"
: "${PARTITION:=}"
: "${ACCOUNT:=}"
: "${QOS:=}"
: "${EXTRA_SBATCH:=}"            # whitespace-split extra sbatch options
: "${GPU:=0}"
: "${GRES:=gpu:1}"

: "${OUTDIR:=}"                  # default: $PWD/nsns_slurm_<obs>_<mode>
: "${GRIDNAME:=nsns_grid}"
: "${GRID_CSV_LIST:=}"           # colon-separated CSVs; skips derivation

# Grid derivation, forwarded to demod/run_nsns_sweep.sh.  Empty = its default.
: "${P0:=1.0}"
: "${PHASE_TOL:=0.1}"
: "${PO:=}"
: "${ECC:=}"
: "${SINI:=}"
: "${OMEGA_PERI:=}"
: "${DROPPCT:=10}"
: "${NPHASE:=120}"
: "${NSAMP:=4000}"
: "${MAX_ACCEL:=}"
: "${MAX_JERK:=}"
: "${MAX_SNAP:=}"
: "${P_BREAK:=}"
: "${BREAK_COV:=}"
: "${TERMS:=}"
: "${CIRC_STRICT:=}"
: "${PLOT:=}"

: "${SEARCH_ARGS:=}"           # whitespace-split, e.g. "--threshold 6"
: "${DEMOD_CONCURRENCY:=0}"    # 0 = one slot per Julia thread
: "${BLOCKSIZE:=}"

: "${JULIA:=julia}"
: "${PYTHON:=python3}"
: "${COH_REPO:=}"
: "${MODULE_LOAD:=}"

: "${COMBINE:=1}"
: "${COMBINE_MEM:=4G}"
: "${COMBINE_TIME:=01:00:00}"
: "${COMBINE_PLOT:=}"

: "${DRY_RUN:=0}"
: "${RESTART:=0}"
: "${RERUN_BATCHES:=}"         # 1-based batch numbers, e.g. "3 7"
: "${ARRAY_SPEC:=}"            # raw sbatch --array override

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

if [ -z "$COH_REPO" ]; then
    COH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

case "$MODE" in
    both|circular|ajs|taylor|hybrid) ;;
    *) die "MODE must be both|circular|ajs|taylor|hybrid (got '$MODE')" ;;
esac
MODE_RESOLVED="$MODE"
[ "$MODE" = taylor ] && MODE_RESOLVED=ajs

case "$BATCH_SIZE" in
    ''|*[!0-9]*) die "BATCH_SIZE must be a positive integer (got '$BATCH_SIZE')" ;;
esac
[ "$BATCH_SIZE" -gt 0 ] || die "BATCH_SIZE must be > 0"
case "$MAX_ARRAY_TASKS" in
    ''|*[!0-9]*) die "MAX_ARRAY_TASKS must be a positive integer" ;;
esac
[ "$MAX_ARRAY_TASKS" -gt 0 ] || die "MAX_ARRAY_TASKS must be > 0"
case "$MAX_CONCURRENT" in
    ''|*[!0-9]*) die "MAX_CONCURRENT must be a non-negative integer" ;;
esac
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

if [ -z "$OUTDIR" ]; then OUTDIR="$PWD/nsns_slurm_${INBASE}_${MODE}"; fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"
ENVFILE="$OUTDIR/slurm_job.env"
PLAN="$OUTDIR/batch_plan.tsv"
: "${JOB_NAME:=${INBASE}_demod_${MODE}}"
JOB_NAME="${JOB_NAME// /_}"

if [ -n "$MODULE_LOAD" ]; then eval "$MODULE_LOAD"; fi
if [ "$GPU" = 1 ]; then DEMOD_CONCURRENCY=1; fi

# --------------------------------------------------------------------------
# Grid selection / derivation
# --------------------------------------------------------------------------
GRID_CSVS=()
TOTAL_POINTS=0

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

derive_one() {
    # $1 = nsns_grid.py mode, $2 = outstem.  run_nsns_sweep.sh owns the grid
    # derivation (and all its physics knobs); this only asks it for the paths.
    local mode="$1" gname="$2" log="$OUTDIR/derive_${2}.log"
    local csvs=() c n
    note "deriving grid: mode=$mode outstem=$gname"
    GRID_CSV="" MODE="$mode" GRIDNAME="$gname" GRID_ONLY=1 \
        bash "$COH_REPO/demod/run_nsns_sweep.sh" | tee "$log"
    mapfile -t csvs < <(sed -n 's/^GRID_CSV=//p' "$log")
    [ "${#csvs[@]}" -ge 1 ] || die "no GRID_CSV= lines from $log"
    for c in "${csvs[@]}"; do
        check_csv "$c"
        GRID_CSVS+=("$c")
    done
    n="$(sed -n 's/^GRID_POINTS=//p' "$log" | tail -1)"
    [ -n "$n" ] || die "no GRID_POINTS= line from $log"
    TOTAL_POINTS=$((TOTAL_POINTS + n))
}

# Export what run_nsns_sweep.sh reads.  MODE/GRIDNAME/GRID_ONLY are set per call.
export INF DAT OUTDIR PYTHON JULIA COH_REPO
export P0 PHASE_TOL PO ECC SINI OMEGA_PERI DROPPCT NPHASE NSAMP
export MAX_ACCEL MAX_JERK MAX_SNAP P_BREAK BREAK_COV TERMS CIRC_STRICT PLOT

if [ "$RESTART" = 1 ]; then
    [ -f "$ENVFILE" ] || die "RESTART=1 but $ENVFILE does not exist"
    [ -f "$PLAN" ] || die "RESTART=1 but $PLAN does not exist"
    # Read the saved plan without letting the env file overwrite the settings
    # the caller just supplied for this submission.
    eval "$(bash -c 'source "$1"; printf "BATCH_SIZE=%q\nTOTAL_POINTS=%q\nNBATCHES=%q\n" \
        "$BATCH_SIZE" "$TOTAL_POINTS" "$NBATCHES"' _ "$ENVFILE")"
    mapfile -t GRID_CSVS < <(bash -c 'source "$1"; printf "%s\n" "${GRID_CSVS[@]}"' _ "$ENVFILE")
    [ "${#GRID_CSVS[@]}" -ge 1 ] || die "$ENVFILE has no GRID_CSVS"
    for c in "${GRID_CSVS[@]}"; do check_csv "$c"; done
    [ "$NBATCHES" -eq "$(( (TOTAL_POINTS + BATCH_SIZE - 1) / BATCH_SIZE ))" ] \
        || die "saved plan is inconsistent: NBATCHES=$NBATCHES, TOTAL_POINTS=$TOTAL_POINTS, BATCH_SIZE=$BATCH_SIZE"
    note "restarting from $ENVFILE: $TOTAL_POINTS point(s), $NBATCHES batch(es)"
elif [ -n "$GRID_CSV_LIST" ]; then
    IFS=: read -r -a GRID_CSVS <<< "$GRID_CSV_LIST"
    [ "${#GRID_CSVS[@]}" -ge 1 ] || die "GRID_CSV_LIST is empty"
    for c in "${GRID_CSVS[@]}"; do
        check_csv "$c"
        TOTAL_POINTS=$((TOTAL_POINTS + $(count_csv "$c")))
    done
else
    case "$MODE" in
        both)
            derive_one circular "${GRIDNAME}_circular"
            derive_one ajs "${GRIDNAME}_ajs"
            ;;
        hybrid)
            derive_one hybrid "$GRIDNAME"
            ;;
        *)
            derive_one "$MODE_RESOLVED" "$GRIDNAME"
            ;;
    esac
fi

[ "$TOTAL_POINTS" -gt 0 ] || die "the selected grid has no points"
if [ "$RESTART" != 1 ]; then
    NBATCHES=$(( (TOTAL_POINTS + BATCH_SIZE - 1) / BATCH_SIZE ))
fi
if [ "$NBATCHES" -gt "$MAX_ARRAY_TASKS" ]; then
    min_bs=$(( (TOTAL_POINTS + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
    die "$TOTAL_POINTS point(s) at BATCH_SIZE=$BATCH_SIZE need $NBATCHES array elements, above MAX_ARRAY_TASKS=$MAX_ARRAY_TASKS; use BATCH_SIZE >= $min_bs (or raise MAX_ARRAY_TASKS)"
fi

# --------------------------------------------------------------------------
# Plan and worker environment
# --------------------------------------------------------------------------
{
    printf 'batch\tfirst\tlast\tcount\n'
    for (( i = 1; i <= NBATCHES; i++ )); do
        lo=$(( (i - 1) * BATCH_SIZE + 1 ))
        hi=$(( lo + BATCH_SIZE - 1 ))
        if [ "$hi" -gt "$TOTAL_POINTS" ]; then hi=$TOTAL_POINTS; fi
        printf '%d\t%d\t%d\t%d\n' "$i" "$lo" "$hi" "$((hi - lo + 1))"
    done
} > "$PLAN"

write_env() {
    {
        printf '# generated by %s\n' "$0"
        printf 'MODE=%q\n' "$MODE"
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
        printf 'GPU=%q\n' "$GPU"
        printf 'BLOCKSIZE=%q\n' "$BLOCKSIZE"
        printf 'JULIA=%q\n' "$JULIA"
        printf 'PYTHON=%q\n' "$PYTHON"
        printf 'MODULE_LOAD=%q\n' "$MODULE_LOAD"
        printf 'COMBINE_PLOT=%q\n' "$COMBINE_PLOT"
    } > "$ENVFILE"
}
write_env

# --------------------------------------------------------------------------
# Array specification
# --------------------------------------------------------------------------
if [ -n "$ARRAY_SPEC" ]; then
    :
elif [ -n "$RERUN_BATCHES" ]; then
    ids=""
    for b in $RERUN_BATCHES; do
        case "$b" in
            ''|*[!0-9]*) die "RERUN_BATCHES must contain 1-based integer batch numbers" ;;
        esac
        if [ "$b" -lt 1 ] || [ "$b" -gt "$NBATCHES" ]; then
            die "batch $b is outside 1..$NBATCHES"
        fi
        if [ -z "$ids" ]; then ids="$((b - 1))"; else ids="$ids,$((b - 1))"; fi
    done
    ARRAY_SPEC="$ids"
else
    ARRAY_SPEC="0-$((NBATCHES - 1))"
fi
if [ "$MAX_CONCURRENT" -gt 0 ] && [[ "$ARRAY_SPEC" != *%* ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

# --------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------
SBATCH_ARGS=(
    --parsable
    --job-name="$JOB_NAME"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$TIME"
    --array="$ARRAY_SPEC"
    --output="$LOGDIR/slurm-%A_%a.out"
    --error="$LOGDIR/slurm-%A_%a.err"
)
if [ -n "$PARTITION" ]; then SBATCH_ARGS+=(--partition="$PARTITION"); fi
if [ -n "$ACCOUNT" ];   then SBATCH_ARGS+=(--account="$ACCOUNT"); fi
if [ -n "$QOS" ];       then SBATCH_ARGS+=(--qos="$QOS"); fi
if [ "$GPU" = 1 ];      then SBATCH_ARGS+=(--gres="$GRES"); fi
if [ -n "$EXTRA_SBATCH" ]; then
    # Intentionally word-split, like SEARCH_ARGS.
    SBATCH_ARGS+=($EXTRA_SBATCH)
fi

note "grid: ${#GRID_CSVS[@]} CSV(s), $TOTAL_POINTS point(s)"
note "batches: $NBATCHES x up to $BATCH_SIZE point(s), array=$ARRAY_SPEC"
note "worker: $CPUS_PER_TASK thread(s), --demod-concurrency=$DEMOD_CONCURRENCY, GPU=$GPU"
note "output: $OUTDIR"
note "plan:   $PLAN"
note "config: $ENVFILE"

if [ "$DRY_RUN" = 1 ]; then
    note "DRY_RUN: would submit:"
    printf '   '
    printf '%q ' sbatch "${SBATCH_ARGS[@]}" \
        "$COH_REPO/demod/slurm_demod_batch.sh" "$ENVFILE"
    printf '\n'
    if [ "$COMBINE" = 1 ]; then
        note "DRY_RUN: then a combine job with --dependency=afterok:<array_job_id>"
    fi
    exit 0
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" \
    "$COH_REPO/demod/slurm_demod_batch.sh" "$ENVFILE")"
JOBID="${JOBID%%;*}"
note "submitted array job $JOBID ($NBATCHES batch(es), $TOTAL_POINTS point(s))"
note "queue:     squeue -j $JOBID"
note "accounting: sacct -j $JOBID --format=JobID,State,Elapsed,MaxRSS"
note "logs:      $LOGDIR"

if [ "$COMBINE" = 1 ]; then
    COMBINE_ARGS=(
        --parsable
        --job-name="${JOB_NAME}_combine"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --mem="$COMBINE_MEM"
        --time="$COMBINE_TIME"
        --dependency="afterok:$JOBID"
        --output="$LOGDIR/combine_%j.out"
        --error="$LOGDIR/combine_%j.err"
    )
    if [ -n "$PARTITION" ]; then COMBINE_ARGS+=(--partition="$PARTITION"); fi
    if [ -n "$ACCOUNT" ];   then COMBINE_ARGS+=(--account="$ACCOUNT"); fi
    if [ -n "$QOS" ];       then COMBINE_ARGS+=(--qos="$QOS"); fi
    CID="$(sbatch "${COMBINE_ARGS[@]}" \
        "$COH_REPO/demod/slurm_demod_combine.sh" "$ENVFILE")"
    CID="${CID%%;*}"
    note "submitted combine job $CID (afterok:$JOBID)"
    note "combined candidates will be $OUTDIR/combined_cohout.txt"
else
    note "COMBINE=0; after the array finishes run:"
    note "  python $COH_REPO/demod/combine_cohout.py '$OUTDIR/${INBASE}_demod_*_red.cohout' -o $OUTDIR/combined_cohout.txt"
fi
