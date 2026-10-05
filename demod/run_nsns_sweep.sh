#!/usr/bin/env bash
#
# NS-NS accel/jerk/snap demodulation sweep.
#
#   1. Derive an (accel, jerk, snap) grid from one observation's .inf
#      (demod/nsns_grid.py, a copy of FFA_stacking/orbit_simulation/
#      kinematic_grid_spacing.py specialised to a 30 min - 10 d NS-NS orbit).
#      Emits a YAML spec and a CSV of every concrete trial.
#   2. Batch-demodulate the observation at every grid point, in ONE Julia
#      process (demod/demod_grid.jl -> demod/demod_dat.jl -> resample.jl).
#   3. For each point: realfft -> rednoise -> narrow coherent search around
#      1/P0.  Every point keeps its own .dat/.fft/.cohout under $OUTDIR/points/.
#
# Everything is configured through environment variables; see the defaults
# below.  The grid is derived from -phase_tol_cycles (the accumulated phase
# drift budget over the observation); run `python nsns_grid.py -h` for the
# full set of grid knobs.
#
#   bash demod/run_nsns_sweep.sh
#   INF=/path/obs.inf NPHASE=20 NSAMP=1000 bash demod/run_nsns_sweep.sh
#   GRID_CSV=/path/to/grid.csv bash demod/run_nsns_sweep.sh   # skip derivation
#
# Requires: julia (+ the CoherentSearch.jl project), a Python with numpy/scipy/
# matplotlib/yaml (the pixi/anaconda env carrying FFA_stacking's deps), and
# PRESTO's realfft + rednoise on PATH (or REALFFT/REDNOISE set).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
: "${INF:=/home/fadong/Documents/CoherentSearch.jl/test_DM1778.00.inf}"
: "${INBASE:=}"                       # observation stem; default from INF

# Spin / search band (fundamentals, Hz): 1/P0 +/- BAND_FRAC.
: "${P0:=1.0}"
: "${BAND_FRAC:=0.02}"
: "${NHARMS:=16}"
: "${MAXDECIM:=1}"
: "${THRESHOLD:=6.0}"
: "${NTHREADS:=4}"

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

# Tool locations.
: "${COH_REPO:=/home/fadong/Documents/CoherentSearch.jl}"
: "${FFA_REPO:=/home/fadong/Documents/FFA_stacking}"
: "${PYTHON:=/home/fadong/anaconda3/bin/python}"
: "${REALFFT:=/home/fadong/anaconda3/bin/realfft}"
: "${REDNOISE:=/home/fadong/anaconda3/bin/rednoise}"
: "${JULIA:=julia}"

: "${OUTDIR:=$PWD/nsns_sweep}"
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
if [ -n "$GRID_CSV" ]; then
    [ -f "$GRID_CSV" ] || { echo "ERROR: GRID_CSV $GRID_CSV not found" >&2; exit 1; }
    GRID_CSV="$(cd "$(dirname "$GRID_CSV")" && pwd)/$(basename "$GRID_CSV")"
    echo ">> using caller-supplied grid $GRID_CSV"
else
    GRID_ARGS=(-inf "$INF" -outdir "$GRIDDIR" -outstem "$GRIDNAME"
               -p0 "$P0" -drop_pct "$DROPPCT"
               -phase_tol_cycles "$PHASE_TOL" -n_phase "$NPHASE" -nsamp "$NSAMP")
    [ -n "$PO" ]         && GRID_ARGS+=(-p_o $PO)
    [ -n "$ECC" ]        && GRID_ARGS+=(-e $ECC)
    [ -n "$SINI" ]       && GRID_ARGS+=(-sin_i $SINI)
    [ -n "$OMEGA_PERI" ] && GRID_ARGS+=(-omega_peri $OMEGA_PERI)
    [ -n "$MAX_ACCEL" ]  && GRID_ARGS+=(-max_accel "$MAX_ACCEL")
    [ -n "$MAX_JERK" ]   && GRID_ARGS+=(-max_jerk "$MAX_JERK")
    [ -n "$MAX_SNAP" ]   && GRID_ARGS+=(-max_snap "$MAX_SNAP")

    echo ">> deriving grid (nsns_grid.py)"
    "$PYTHON" "$COH_REPO/demod/nsns_grid.py" "${GRID_ARGS[@]}" \
        2>&1 | tee "$GRIDDIR/derive.log"

    GRID_CSV="$GRIDDIR/${GRIDNAME}.csv"
    [ -f "$GRID_CSV" ] || { echo "ERROR: no grid CSV produced" >&2; exit 1; }
fi
NPOINTS=$(($(grep -c . "$GRID_CSV") - 1))
echo ">> grid: $NPOINTS point(s) in $GRID_CSV"
[ "$NPOINTS" -ge 1 ] || { echo "ERROR: empty grid" >&2; exit 1; }

# --------------------------------------------------------------------------
# 2. Batch-demodulate every point (one Julia process)
# --------------------------------------------------------------------------
POINTS="$OUTDIR/points"
echo ">> batch demodulating $NPOINTS point(s)"
"$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_grid.jl" \
    "$DAT" "$GRID_CSV" "$OUTDIR" --basename "$INBASE" \
    2>&1 | tee "$OUTDIR/demod_grid.log"

# --------------------------------------------------------------------------
# 3. realfft + rednoise + coherent search per point (files kept)
# --------------------------------------------------------------------------
LOFREQ=$("$PYTHON" -c "print(1.0/$P0*(1.0-$BAND_FRAC))")
HIFREQ=$("$PYTHON" -c "print(1.0/$P0*(1.0+$BAND_FRAC))")
echo ">> search band: $LOFREQ .. $HIFREQ Hz, threshold $THRESHOLD"

idx=0
{
    read -r _header                    # skip the "accel,jerk,snap" header
    while IFS=, read -r accel jerk snap; do
    [ -n "$accel" ] || continue        # tolerate a trailing blank line
    idx=$((idx + 1))
    pdir="$POINTS/$(printf 'p%05d' "$idx")"
    base="$pdir/${INBASE}_demod"
    [ -f "$base.dat" ] || { echo "ERROR: missing $base.dat" >&2; exit 1; }

    echo ">> [$idx/$NPOINTS] a=$accel j=$jerk s=$snap"
    # realfft/rednoise write beside the input / in the CWD respectively, so run
    # them with the point directory as CWD.
    "$REALFFT" "$base.dat" > "$pdir/realfft.log" 2>&1
    ( cd "$pdir" && "$REDNOISE" "$(basename "$base").fft" ) > "$pdir/rednoise.log" 2>&1
    [ -f "${base}_red.fft" ] || { echo "ERROR: no ${base}_red.fft" >&2; exit 1; }

    "$JULIA" --project="$COH_REPO" -t "$NTHREADS" \
        "$COH_REPO/bin/coherent_search.jl" "${base}_red.fft" \
        --lofreq "$LOFREQ" --hifreq "$HIFREQ" \
        --nharms "$NHARMS" --maxdecim "$MAXDECIM" \
        --threshold "$THRESHOLD" \
        -o "${base}_red.cohout" > "$pdir/search.log" 2>&1
    done
} < "$GRID_CSV"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo
echo "=========================================================================="
echo "Grid:   $GRID_CSV"
[ -f "$GRIDDIR/${GRIDNAME}.yaml" ] && echo "Spec:   $GRIDDIR/${GRIDNAME}.yaml"
echo "Points: $NPOINTS under $POINTS/p*  (each keeps .dat/.fft/.cohout/logs)"
echo
echo "Per-point candidates are in $POINTS/p*/${INBASE}_demod_red.cohout."
echo "To list the strongest hits across the whole sweep:"
echo "  awk '\$1 ~ /^[0-9]/ {print \$2, FILENAME, \$0}' $POINTS/p*/${INBASE}_demod_red.cohout | sort -rn | head"
echo "=========================================================================="
