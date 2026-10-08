#!/usr/bin/env bash
#
# NS-NS demodulation sweep, in either grid model (MODE):
#
#   MODE=ajs (default) -- accel/jerk/snap grid (demod/nsns_grid.py, a copy of
#     FFA_stacking/orbit_simulation/kinematic_grid_spacing.py specialised to a
#     30 min - 10 d NS-NS orbit).  Each point writes
#     <INBASE>_demod_a{A}_j{J}_s{S}.*.
#   MODE=circular -- pure Keplerian circular-orbit grid (pb,x,at, uniform in
#     omega_b, x, A_T; no cubic-truncation gate since the demod is exact at the
#     truth, so -drop_pct drops the highest-x orbits first).  Each point writes
#     <INBASE>_demod_pb{Pb}_x{X}_at{A_T}.*.
#   MODE=hybrid -- circular below p_break and ajs at/above it, combining the two
#     regimes.  p_break is the shortest scanned p_o such that every p_o at/above
#     it has mean e=0 ajs coverage >= -break_coverage (default 60%); because ajs
#     coverage dips near p_o ~ T_obs, the switch clears the whole inadequate
#     band.  Override with P_BREAK (a period in yr) or BREAK_COV (a %).  Emits two
#     CSVs (<GRIDNAME>_circular.csv and <GRIDNAME>_ajs.csv) and one YAML; each
#     point keeps whichever filename pattern its model uses.
#
#   1. Derive the grid from one observation's .inf (nsns_grid.py -mode $MODE).
#      Emits a YAML spec and one CSV per model of every concrete trial.
#   2. Batch-demodulate the observation at every grid point, in ONE Julia
#      process (demod/demod_grid.jl -> demod/demod_dat.jl -> src/demod.jl).
#   3. For each point: realfft -> rednoise -> coherent search (coherent_search.jl
#      defaults; override via SEARCH_ARGS).  Every point keeps its own files,
#      flat under $OUTDIR.  On the CPU the searches run NPROC-wide in parallel
#      (default 8), each with NTHREADS Julia threads; keep NPROC*NTHREADS <=
#      cores to avoid oversubscription.
#
# GPU: set GPU=1 to pass --gpu to the search.  CUDA is a weak dependency, so
# point GPU_PROJECT at a Julia environment carrying CUDA.jl (see the README's
# "Installing CUDA.jl"; bench/gpu_probe_setup.sh builds one).  GPU=1 ignores
# NPROC: every point is prepared (realfft + rednoise) and then ALL the _red.fft
# files go to a single coherent_search.jl invocation, so CUDA loads once.
# BLOCKSIZE overrides --blocksize; empty uses coherent_search.jl's per-backend
# default.
#
# GPU: set GPU=1 to pass --gpu to the search.  CUDA is a weak dependency, so
# point GPU_PROJECT at a Julia environment carrying CUDA.jl (see the README's
# "Installing CUDA.jl"; bench/gpu_probe_setup.sh builds one).  GPU=1 ignores
# NPROC: every point is prepared (realfft + rednoise) and then ALL the _red.fft
# files go to a single coherent_search.jl invocation, so CUDA loads once.
# BLOCKSIZE overrides --blocksize; empty uses coherent_search.jl's per-backend
# default.
#
# Everything is configured through environment variables; see the defaults
# below.  The grid is derived from -phase_tol_cycles (the accumulated phase
# drift budget over the observation); run `python nsns_grid.py -h` for the
# full set of grid knobs.
#
#   bash demod/run_nsns_sweep.sh
#   INF=/path/obs.inf NPHASE=20 NSAMP=1000 bash demod/run_nsns_sweep.sh
#   GRID_CSV=/path/to/grid.csv bash demod/run_nsns_sweep.sh   # skip derivation
#   MODE=circular DROPPCT=50 bash demod/run_nsns_sweep.sh     # circular sweep
#   MODE=hybrid bash demod/run_nsns_sweep.sh                  # circular + ajs
#   MODE=hybrid P_BREAK=0.02 bash demod/run_nsns_sweep.sh     # manual p_break [yr]
#   GPU=1 GPU_PROJECT=~/gpuenv bash demod/run_nsns_sweep.sh   # NVIDIA P40
#
# Requires: julia (+ the CoherentSearch.jl project), a Python with numpy/scipy/
# matplotlib/yaml (the pixi/anaconda env carrying FFA_stacking's deps), and
# PRESTO's realfft + rednoise on PATH (or REALFFT/REDNOISE set).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
: "${INF:=/home/fadong/Documents/CoherentSearch.jl/demod/inj.inf}"
: "${INBASE:=}"                       # observation stem; default from INF

# Grid model: ajs (accel/jerk/snap, default), circular (pb,x,at), or hybrid
# (circular below p_break, ajs at/above it).  MODE picks the nsns_grid.py -mode
# and the per-point filename pattern (hybrid reads each point's model from its
# CSV header, so both patterns coexist).
: "${MODE:=ajs}"

# Spin period, used only by the grid derivation (nsns_grid.py -p0).
: "${P0:=1.0}"
# The search itself runs with coherent_search.jl's own defaults (0.1-125 Hz,
# nharms 60, maxdecim 6, threshold 8); override any of them by appending flags
# to SEARCH_ARGS.
: "${NTHREADS:=4}"                   # Julia threads per search process
: "${SEARCH_ARGS:=}"

# GPU (NVIDIA P40 or any CUDA card): GPU=1 passes --gpu to coherent_search.jl.
# CUDA is a weak dep, so GPU_PROJECT must name a Julia env that has it (default:
# the repo project, which works if you added CUDA there despite the advice not
# to).  One GPU is shared by every search process, so GPU=1 defaults NPROC to 1.
: "${GPU:=1}"
: "${GPU_PROJECT:=}"
: "${BLOCKSIZE:=}"                   # --blocksize; empty = per-backend default

# Concurrent search processes on the CPU (8); ignored under GPU=1, which runs
# one search over every prepared file.
: "${NPROC:=8}"

# Grid derivation (forwarded to nsns_grid.py).  Empty = use its own default.
: "${PO:=}"                           # orbital periods [yr], space separated
: "${ECC:=}"                          # eccentricities
: "${SINI:=}"                         # sin(inclination)
: "${OMEGA_PERI:=}"                   # arguments of periastron [rad]
: "${DROPPCT:=10}"                    # % of feasible segments to discard
: "${PHASE_TOL:=0.1}"                 # accumulated-phase budget [cycles]
: "${NPHASE:=120}"                    # segment start phases scanned per orbit
: "${NSAMP:=4000}"                    # time samples per span for the fits
: "${MAX_ACCEL:=}"                    # cap |accel| [m/s^2]; empty = derived
: "${MAX_JERK:=}"                     # cap |jerk| [m/s^3]
: "${MAX_SNAP:=}"                     # cap |snap| [m/s^4]
: "${P_BREAK:=}"                      # hybrid: manual switch period [yr]; empty = auto
: "${BREAK_COV:=}"                    # hybrid: auto-switch ajs coverage [%]; empty = 60
: "${PLOT:=}"                         # non-empty = write per-orbit phase plots

# Tool locations.
: "${COH_REPO:=/home/fadong/Documents/CoherentSearch.jl}"
: "${PYTHON:=/home/fadong/anaconda3/bin/python}"
: "${REALFFT:=/home/fadong/anaconda3/bin/realfft}"
: "${REDNOISE:=/home/fadong/anaconda3/bin/rednoise}"
: "${JULIA:=julia}"

# Search process environment and flags: GPU=1 searches from an env carrying
# CUDA.jl and passes --gpu; otherwise the repo project.  GPU_PROJECT defaults to
# the repo project (works only if CUDA was added there, against upstream advice).
SEARCH_PROJECT="$COH_REPO"
SEARCH_FLAGS=()
if [ -n "$GPU" ]; then
    : "${GPU_PROJECT:=$COH_REPO}"
    SEARCH_PROJECT="$GPU_PROJECT"
    SEARCH_FLAGS=(--gpu)
    [ -n "$BLOCKSIZE" ] && SEARCH_FLAGS+=(--blocksize "$BLOCKSIZE")
fi

: "${OUTDIR:=$PWD/nsns_sweep_inj}"
: "${GRIDNAME:=nsns_grid}"
: "${GRID_CSV:=}"                     # skip derivation and use this CSV

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
GRIDDIR="$OUTDIR/$GRIDNAME"
mkdir -p "$GRIDDIR"

DAT="${INF%.inf}.dat"
[ -f "$DAT" ] || { echo "ERROR: no .dat beside $INF" >&2; exit 1; }
: "${INBASE:=$(basename "${DAT%.dat}")}"

echo ">> observation: $DAT"
echo ">> output:      $OUTDIR"

# --------------------------------------------------------------------------
# 1. Derive the grid (or use GRID_CSV if the caller supplied one)
# --------------------------------------------------------------------------
GRID_CSVS=()   # one CSV per model: hybrid emits _circular.csv + _ajs.csv
if [ -n "$GRID_CSV" ]; then
    [ -f "$GRID_CSV" ] || { echo "ERROR: GRID_CSV $GRID_CSV not found" >&2; exit 1; }
    GRID_CSV="$(cd "$(dirname "$GRID_CSV")" && pwd)/$(basename "$GRID_CSV")"
    GRID_CSVS=("$GRID_CSV")
    echo ">> using caller-supplied grid $GRID_CSV"
else
    GRID_ARGS=(-inf "$INF" -outdir "$GRIDDIR" -outstem "$GRIDNAME"
               -p0 "$P0" -drop_pct "$DROPPCT" -mode "$MODE")
    if [ "$MODE" = "circular" ]; then
        # Circular grid: only (p_o, sin_i, drop_pct, phase_tol) apply; no
        # e/omega_peri/caps/terms.
        [ -n "$PO" ]   && GRID_ARGS+=(-p_o $PO)
        [ -n "$SINI" ] && GRID_ARGS+=(-sin_i $SINI)
        GRID_ARGS+=(-phase_tol_cycles "$PHASE_TOL")
    elif [ "$MODE" = "hybrid" ]; then
        # Hybrid: the ajs scan still needs e/omega_peri/n_phase/nsamp (it is
        # what measures coverage and grids the long-period side); the circular
        # side ignores them. -p_break overrides the auto threshold; -BREAK_COV
        # sets the coverage percentage.
        GRID_ARGS+=(-phase_tol_cycles "$PHASE_TOL" -n_phase "$NPHASE" -nsamp "$NSAMP")
        [ -n "$PO" ]         && GRID_ARGS+=(-p_o $PO)
        [ -n "$ECC" ]        && GRID_ARGS+=(-e $ECC)
        [ -n "$SINI" ]       && GRID_ARGS+=(-sin_i $SINI)
        [ -n "$OMEGA_PERI" ] && GRID_ARGS+=(-omega_peri $OMEGA_PERI)
        [ -n "$MAX_ACCEL" ]  && GRID_ARGS+=(-max_accel "$MAX_ACCEL")
        [ -n "$MAX_JERK" ]   && GRID_ARGS+=(-max_jerk "$MAX_JERK")
        [ -n "$MAX_SNAP" ]   && GRID_ARGS+=(-max_snap "$MAX_SNAP")
        [ -n "$P_BREAK" ]    && GRID_ARGS+=(-p_break "$P_BREAK")
        [ -n "$BREAK_COV" ]  && GRID_ARGS+=(-break_coverage "$BREAK_COV")
        [ -n "$PLOT" ]       && GRID_ARGS+=(-plot)
    else
        GRID_ARGS+=(-phase_tol_cycles "$PHASE_TOL" -n_phase "$NPHASE" -nsamp "$NSAMP")
        [ -n "$PO" ]         && GRID_ARGS+=(-p_o $PO)
        [ -n "$ECC" ]        && GRID_ARGS+=(-e $ECC)
        [ -n "$SINI" ]       && GRID_ARGS+=(-sin_i $SINI)
        [ -n "$OMEGA_PERI" ] && GRID_ARGS+=(-omega_peri $OMEGA_PERI)
        [ -n "$MAX_ACCEL" ]  && GRID_ARGS+=(-max_accel "$MAX_ACCEL")
        [ -n "$MAX_JERK" ]   && GRID_ARGS+=(-max_jerk "$MAX_JERK")
        [ -n "$MAX_SNAP" ]   && GRID_ARGS+=(-max_snap "$MAX_SNAP")
        [ -n "$PLOT" ]       && GRID_ARGS+=(-plot)
    fi

    echo ">> deriving grid (nsns_grid.py -mode $MODE)"
    "$PYTHON" "$COH_REPO/demod/nsns_grid.py" "${GRID_ARGS[@]}" \
        2>&1 | tee "$GRIDDIR/derive.log"

    if [ "$MODE" = "hybrid" ]; then
        for suffix in circular ajs; do
            csv="$GRIDDIR/${GRIDNAME}_${suffix}.csv"
            if [ -f "$csv" ]; then GRID_CSVS+=("$csv"); fi
        done
    else
        csv="$GRIDDIR/${GRIDNAME}.csv"
        if [ -f "$csv" ]; then GRID_CSVS+=("$csv"); fi
    fi
    [ "${#GRID_CSVS[@]}" -ge 1 ] || { echo "ERROR: no grid CSV produced" >&2; exit 1; }
fi
NPOINTS=0
for csv in "${GRID_CSVS[@]}"; do
    n=$(( $(grep -c . "$csv") - 1 ))
    [ "$n" -ge 0 ] || n=0
    NPOINTS=$((NPOINTS + n))
done
echo ">> grid: $NPOINTS point(s) in ${GRID_CSVS[*]}"
[ "$NPOINTS" -ge 1 ] || { echo "ERROR: empty grid" >&2; exit 1; }

# --------------------------------------------------------------------------
# 2. Batch-demodulate every point (one Julia process), flat under $OUTDIR:
#    <INBASE>_demod_a{A}_j{J}_s{S}.dat (+ .inf) or ..._demod_pb{Pb}_x{X}_at{AT}
# --------------------------------------------------------------------------
echo ">> batch demodulating $NPOINTS point(s)"
"$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_grid.jl" \
    "$DAT" "$OUTDIR" "${GRID_CSVS[@]}" --basename "$INBASE" \
    2>&1 | tee "$OUTDIR/demod_grid.log"

# --------------------------------------------------------------------------
# 3. realfft + rednoise + coherent search per point (files kept; search uses
#    coherent_search.jl defaults, override via SEARCH_ARGS).  rednoise writes
#    _red.fft in the CWD, so everything runs from $OUTDIR.  Points are
#    independent (distinct file names).
#
#    CPU: one search per point, NPROC of them at once (NPROC=1 serial).
#    GPU: no per-point searches -- prepare every point, then hand ALL the
#    _red.fft files to ONE coherent_search.jl invocation so CUDA loads once.
# --------------------------------------------------------------------------
cd "$OUTDIR"

# One entry per point: "MODE<TAB>f1<TAB>f2<TAB>f3".  MODE is read from each
# CSV's header so a hybrid run (two CSVs, two models) is handled uniformly.
points=()
saw_ajs=0
saw_circular=0
load_points() {
    local csv="$1" mode="" header=1 f1 f2 f3
    while IFS= read -r line; do
        line="${line%$'\r'}"
        if [ "$header" = 1 ]; then
            case "${line// /}" in
                accel,jerk,snap*) mode=ajs; saw_ajs=1 ;;
                pb,x,at*)         mode=circular; saw_circular=1 ;;
                *) echo "ERROR: $csv: unknown header '$line'" >&2; return 1 ;;
            esac
            header=0
            continue
        fi
        [ -n "$line" ] || continue
        IFS=, read -r f1 f2 f3 <<< "$line"
        f1="${f1%$'\r'}"; f2="${f2%$'\r'}"; f3="${f3%$'\r'}"
        [ -n "$f1" ] || continue
        points+=("$mode"$'\t'"$f1"$'\t'"$f2"$'\t'"$f3")
    done < "$csv"
}
for csv in "${GRID_CSVS[@]}"; do
    load_points "$csv" || exit 1
done

# Split a point entry into mode + its three fields.
split_spec() {  # $1 = "mode f1 f2 f3"
    p_mode="${1%%$'\t'*}"; local rest="${1#*$'\t'}"
    accel="${rest%%$'\t'*}"; rest="${rest#*$'\t'}"
    jerk="${rest%%$'\t'*}"; snap="${rest#*$'\t'}"
}

# Per-point filename stem, keyed on that point's model.  Matches
# CoherentSearch.demod_point_stem() and combine_cohout.py's regexes.
point_stem() {  # $1 mode, $2 f1, $3 f2, $4 f3
    if [ "$1" = "circular" ]; then
        echo "${INBASE}_demod_pb$2_x$3_at$4"
    else
        echo "${INBASE}_demod_a$2_j$3_s$4"
    fi
}

# realfft + rednoise one point; leaves <stem>_red.fft in $OUTDIR.
prep_point() {
    local stem; stem="$(point_stem "$1" "$2" "$3" "$4")"
    local base="$OUTDIR/$stem"
    [ -f "$base.dat" ] || { echo "ERROR: missing $base.dat" >&2; return 1; }

    "$REALFFT" "$base.dat" > "$base.realfft.log" 2>&1 \
        || { echo "ERROR: realfft failed for $stem" >&2; return 1; }
    "$REDNOISE" "$stem.fft" > "$base.rednoise.log" 2>&1 \
        || { echo "ERROR: rednoise failed for $stem" >&2; return 1; }
    [ -f "${base}_red.fft" ] || { echo "ERROR: no ${base}_red.fft" >&2; return 1; }
}

# Search one already-prepared point (CPU path).
search_point() {
    local base="$OUTDIR/$(point_stem "$1" "$2" "$3" "$4")"
    "$JULIA" --project="$SEARCH_PROJECT" -t "$NTHREADS" \
        "$COH_REPO/bin/coherent_search.jl" "${base}_red.fft" \
        "${SEARCH_FLAGS[@]}" $SEARCH_ARGS \
        -o "${base}_red.cohout" > "$base.search.log" 2>&1
}

run_point() { prep_point "$1" "$2" "$3" "$4" && search_point "$1" "$2" "$3" "$4"; }

# Human label for a split spec, model-aware.
spec_label() {
    if [ "$1" = "circular" ]; then echo "pb=$2 x=$3 at=$4"; else echo "a=$2 j=$3 s=$4"; fi
}

run_one() {
    split_spec "$1"
    if run_point "$p_mode" "$accel" "$jerk" "$snap"; then
        echo ">> done    $(spec_label "$p_mode" "$accel" "$jerk" "$snap")"
    else
        echo ">> FAILED  $(spec_label "$p_mode" "$accel" "$jerk" "$snap")" >&2
        return 1
    fi
}

if [ -n "$GPU" ]; then
    # Serial prepare, then ONE search over every _red.fft (single CUDA load).
    echo ">> preparing $NPOINTS point(s): realfft + rednoise (serial)"
    rc=0
    fftlist="$OUTDIR/gpu_fftfiles.txt"
    : > "$fftlist"
    for spec in "${points[@]}"; do
        split_spec "$spec"
        if prep_point "$p_mode" "$accel" "$jerk" "$snap"; then
            printf '%s\n' "$OUTDIR/$(point_stem "$p_mode" "$accel" "$jerk" "$snap")_red.fft" >> "$fftlist"
            echo ">> prepared $(spec_label "$p_mode" "$accel" "$jerk" "$snap")"
        else
            rc=1
        fi
    done
    [ "$rc" -eq 0 ] || { echo "ERROR: one or more points failed to prepare" >&2; exit 1; }

    echo ">> searching $NPOINTS point(s) on the GPU in ONE invocation" \
         "(-t $NTHREADS, single CUDA load)"
    "$JULIA" --project="$SEARCH_PROJECT" -t "$NTHREADS" \
        "$COH_REPO/bin/coherent_search.jl" --filelist "$fftlist" \
        "${SEARCH_FLAGS[@]}" $SEARCH_ARGS --outdir "$OUTDIR" \
        > "$OUTDIR/search.log" 2>&1 \
        || { echo "ERROR: GPU search failed (see $OUTDIR/search.log)" >&2; exit 1; }
elif [ "$NPROC" -le 1 ]; then
    echo ">> searching $NPOINTS point(s), serial (-t $NTHREADS)"
    rc=0
    for spec in "${points[@]}"; do
        run_one "$spec" || rc=1
    done
else
    echo ">> searching $NPOINTS point(s), $NPROC concurrent x -t $NTHREADS"
    rc=0
    declare -A pids=()
    reap_ok() {  # block until any one point finishes; record its success
        local pid
        for pid in "${!pids[@]}"; do
            if wait "$pid" 2>/dev/null; then rc=0; else rc=1; fi
            unset 'pids[$pid]'
            return
        done
    }
    for spec in "${points[@]}"; do
        run_one "$spec" &
        pids[$!]=1
        if [ "${#pids[@]}" -ge "$NPROC" ]; then reap_ok; fi
    done
    while [ "${#pids[@]}" -gt 0 ]; do reap_ok; done
fi
cd - >/dev/null

[ "$rc" -eq 0 ] || { echo "ERROR: one or more points failed" >&2; exit 1; }

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo
echo "=========================================================================="
echo "Grid:   ${GRID_CSVS[*]}"
[ -f "$GRIDDIR/${GRIDNAME}.yaml" ] && echo "Spec:   $GRIDDIR/${GRIDNAME}.yaml"
echo "Points: $NPOINTS in $OUTDIR (each keeps .dat/.fft/.cohout/logs)"
echo
if [ "$saw_ajs" = 1 ] && [ "$saw_circular" = 1 ]; then
    GLOB="${INBASE}_demod_[ap]*_red.cohout"   # ajs `_demod_a...`, circ `_demod_pb...`
elif [ "$saw_circular" = 1 ]; then
    GLOB="${INBASE}_demod_pb*_x*_at*_red.cohout"
else
    GLOB="${INBASE}_demod_a*_j*_s*_red.cohout"
fi
echo "Per-point candidates are in $OUTDIR/$GLOB."
echo "To list the strongest hits across the whole sweep:"
echo "  awk '\$1 ~ /^[0-9]/ {print \$2, FILENAME, \$0}' $OUTDIR/$GLOB | sort -rn | head"
echo "=========================================================================="
