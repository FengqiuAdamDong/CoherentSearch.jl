#!/usr/bin/env bash
#
# Inject a pulsar in a binary (full DD model), derive a demodulation grid for
# that orbit, pick the grid point closest to the injected truth, demodulate the
# injection at it, and run the coherent search on the result.
#
# MODE=ajs (default): derive the accel/jerk/snap grid and demodulate at the
#   nearest (a, j, s).  MODE=circular: derive the pure-Keplerian circular grid
#   (pb, x, A_T) and demodulate at the nearest point with the exact orbital
#   remap.  Either way the output lands in $OUTDIR.
#
#   1. Inject a phase-coherent pulsar in a binary (default: a 1-hour circular
#      orbit) into a real observation (demod/pulsegen_gr.py, Damour-Deruelle
#      delay model).  It writes the .dat/.inf and a _truth.yaml carrying the
#      parameters the search should recover.
#   2. Derive the grid from the injected .inf (demod/nsns_grid.py -mode $MODE),
#      scoped to the injected orbit.
#   3. Pick the grid point nearest the injected truth (each axis snapped to its
#      nearest grid value -- the grid is a Cartesian product, so that is the
#      nearest point).
#   4. De-drift the injected .dat at that point (demod/demod_dat.jl, exact
#      time remap), writing a resampled .dat + .inf.
#   5. realfft -> rednoise -> coherent search on the demodulated file.
#
# Everything is configured through environment variables; see the defaults
# below.  Output lands in $OUTDIR.
#
#   bash demod/run_demod_search.sh
#   SNR=50 bash demod/run_demod_search.sh
#   PB_DAYS=0.25 SIN_I=0.7 bash demod/run_demod_search.sh
#   MODE=circular PB_DAYS=0.041666666666666664 bash demod/run_demod_search.sh
#   SEARCH_ARGS="--threshold 6" bash demod/run_demod_search.sh
#
# Requires: julia (+ the CoherentSearch.jl project), a Python with numpy/scipy/
# matplotlib/yaml (the pixi/anaconda env carrying FFA_stacking's deps), and
# PRESTO's realfft + rednoise on PATH (or REALFFT/REDNOISE set).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration: injected binary (pulsegen_gr.py)
# --------------------------------------------------------------------------
# MODE=circular demodulates with the exact circular-orbit remap (pb,x,A_T),
# matching a pure Keplerian injection (e=0, omega_peri absorbed by A_T).  It
# reuses the same injection; it just bypasses the accel/jerk/snap grid and
# takes the truth's (pb, x, A_T) directly.  MODE=ajs (default) keeps the
# original nearest-ajs-grid-point behaviour.
: "${MODE:=ajs}"
: "${PB_DAYS:=0.041666666666666664}"  # orbital period [days]; default 1 hour
: "${E:=0}"                           # eccentricity
: "${SIN_I:=0.5}"                     # sin(inclination)
: "${OMEGA_PERI:=0.3}"                # argument of periastron [rad]
: "${A_T:=0}"                         # mean anomaly at the epoch [rad]

: "${P0:=1.0}"                        # intrinsic spin period [s]
: "${PDOT:=0}"                        # intrinsic period derivative [s/s]
: "${SNR:=6}"                        # injected S/N per observation
: "${PULSE_WIDTH:=0.01}"              # Gaussian pulse sigma [s]
: "${SEED:=42}"

# Observation to inject into (supplies N/dt/epoch).  REAL_DAT is the noise
# background: unset -> the .dat beside $INF; empty -> Gaussian noise.
: "${INF:=/home/fadong/Documents/CoherentSearch.jl/test_DM1778.00.inf}"

# --------------------------------------------------------------------------
# Configuration: grid derivation (nsns_grid.py)
# --------------------------------------------------------------------------
: "${DROPPCT:=10}"                    # % of feasible segments to discard
: "${PHASE_TOL:=0.1}"                 # accumulated-phase budget [cycles]
: "${NPHASE:=120}"                   # segment start phases scanned per orbit
: "${NSAMP:=4000}"                   # time samples per span for the fits
: "${PO_YEARS:=}"                    # orbital period(s) [yr]; default PB_DAYS
: "${GRID_OMEGA_PERI:=}"             # default: nsns_grid's own omega_peri scan

# --------------------------------------------------------------------------
# Configuration: search (coherent_search.jl defaults; override via SEARCH_ARGS)
# --------------------------------------------------------------------------
: "${NTHREADS:=4}"
: "${SEARCH_ARGS:=}"

# Tool locations.
: "${COH_REPO:=/home/fadong/Documents/CoherentSearch.jl}"
: "${FFA_REPO:=/home/fadong/Documents/FFA_stacking}"
: "${PYTHON:=/home/fadong/anaconda3/bin/python}"
: "${REALFFT:=/home/fadong/anaconda3/bin/realfft}"
: "${REDNOISE:=/home/fadong/anaconda3/bin/rednoise}"
: "${JULIA:=julia}"

: "${OUTDIR:=$PWD/demod_out}"
: "${OUTBASE:=inj}"
: "${GRIDNAME:=grid}"

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
GRIDDIR="$OUTDIR/$GRIDNAME"
mkdir -p "$GRIDDIR"

if [ -z "${REAL_DAT+x}" ]; then REAL_DAT="${INF%.inf}.dat"; fi
[ -n "$PO_YEARS" ] || PO_YEARS="$("$PYTHON" -c 'import sys; print(float(sys.argv[1])/365.25)' "$PB_DAYS")"

echo ">> output directory: $OUTDIR"

# --------------------------------------------------------------------------
# 1. Inject the pulsar (full DD model)
# --------------------------------------------------------------------------
REAL_ARG=()
[ -n "$REAL_DAT" ] && REAL_ARG=(-real "$REAL_DAT")

# MODE=circular removes only the Roemer term of a circular (e=0) orbit, so
# inject a pure Keplerian: omega_peri=0 (for e=0 it is degenerate with A_T, and
# a nonzero one would fold into A_T) and Shapiro off (its delay is not
# Roemer).  The truth's A_T_rad/x_lt_s are then exactly the effective phase and
# amplitude the demod must remove.  MODE=ajs keeps the full-DD injection.
ORB_EXTRA=()
if [ "$MODE" = "circular" ]; then
    ORB_EXTRA=(-omega_peri 0 -shapiro_r 0 -shapiro_s 0)
fi

echo ">> injecting binary: Pb=$PB_DAYS d, e=$E, sin_i=$SIN_I, p0=$P0 s," \
     "SNR=$SNR into ${REAL_DAT:-Gaussian noise} (MODE=$MODE)"
"$PYTHON" "$COH_REPO/demod/pulsegen_gr.py" \
    -inf "$INF" \
    -outdir "$OUTDIR" -outbasename "$OUTBASE" \
    -pb "$PB_DAYS" -e "$E" -sin_i "$SIN_I" -omega_peri "$OMEGA_PERI" -A_T "$A_T" \
    -p0 "$P0" -pdot "$PDOT" -snr "$SNR" -pulse_width "$PULSE_WIDTH" \
    -seed "$SEED" \
    "${ORB_EXTRA[@]}" \
    "${REAL_ARG[@]}" \
    2>&1 | tee "$OUTDIR/pulsegen.log"

INJ="$OUTDIR/${OUTBASE}.dat"
INJ_INF="$OUTDIR/${OUTBASE}.inf"
TRUTH="$OUTDIR/${OUTBASE}_truth.yaml"
[ -f "$INJ" ] || { echo "ERROR: no $INJ" >&2; exit 1; }

# --------------------------------------------------------------------------
# 2. Derive the grid, scoped to the injected orbit
# --------------------------------------------------------------------------
if [ "$MODE" = "circular" ]; then
    GRID_ARGS=(-inf "$INJ_INF" -outdir "$GRIDDIR" -outstem "$GRIDNAME"
               -mode circular -p0 "$P0" -drop_pct "$DROPPCT"
               -phase_tol_cycles "$PHASE_TOL" -p_o "$PO_YEARS" -sin_i "$SIN_I")
else
    GRID_ARGS=(-inf "$INJ_INF" -outdir "$GRIDDIR" -outstem "$GRIDNAME"
               -p0 "$P0" -drop_pct "$DROPPCT" -phase_tol_cycles "$PHASE_TOL"
               -n_phase "$NPHASE" -nsamp "$NSAMP"
               -p_o "$PO_YEARS" -e "$E" -sin_i "$SIN_I")
    [ -n "$GRID_OMEGA_PERI" ] && GRID_ARGS+=(-omega_peri $GRID_OMEGA_PERI)
fi

echo ">> deriving grid (nsns_grid.py -mode $MODE) for p_o=$PO_YEARS yr"
"$PYTHON" "$COH_REPO/demod/nsns_grid.py" "${GRID_ARGS[@]}" \
    2>&1 | tee "$GRIDDIR/derive.log"

GRID_CSV="$GRIDDIR/${GRIDNAME}.csv"
[ -f "$GRID_CSV" ] || { echo "ERROR: no grid CSV produced" >&2; exit 1; }
NPOINTS=$(($(grep -c . "$GRID_CSV") - 1))
echo ">> grid: $NPOINTS point(s) in $GRID_CSV"

# --------------------------------------------------------------------------
# 3. Nearest grid point to the injected truth
# --------------------------------------------------------------------------
if [ "$MODE" = "circular" ]; then
    { read -r BEST_PB BEST_X BEST_AT; read -r TRUTH_PB TRUTH_X TRUTH_AT; } <<< "$(
    "$PYTHON" - "$GRID_CSV" "$TRUTH" <<'PY'
import csv, sys, math, yaml
grid_csv, truth_yaml = sys.argv[1], sys.argv[2]
t = yaml.safe_load(open(truth_yaml))
truth = (float(t["pb_days"]), float(t["x_lt_s"]), float(t["A_T_rad"]))
cols = ([], [], [])
for row in csv.reader(open(grid_csv)):
    if not row or not row[0] or not row[0][0].isdigit():
        continue
    for i in range(3):
        cols[i].append(float(row[i]))
best = [min(sorted(set(c)), key=lambda v: abs(v - tv))
        for c, tv in zip(cols, truth)]
print(" ".join(repr(x) for x in best))
print(" ".join(repr(x) for x in truth))
PY
    )"
    echo ">> truth   pb=$TRUTH_PB x=$TRUTH_X at=$TRUTH_AT"
    echo ">> nearest pb=$BEST_PB x=$BEST_X at=$BEST_AT"
else
    { read -r BEST_A BEST_J BEST_S; read -r TRUTH_A TRUTH_J TRUTH_S; } <<< "$(
    "$PYTHON" - "$GRID_CSV" "$TRUTH" <<'PY'
import csv, sys, yaml
grid_csv, truth_yaml = sys.argv[1], sys.argv[2]
with open(truth_yaml) as fh:
    t = yaml.safe_load(fh)
truth = (t["best_fit_accel"], t["best_fit_jerk"], t["best_fit_snap"])
cols = ([], [], [])
with open(grid_csv) as fh:
    r = csv.reader(fh)
    next(r)
    for row in r:
        if not row:
            continue
        for i in range(3):
            cols[i].append(float(row[i]))
best = [min(sorted(set(c)), key=lambda v: abs(v - tv))
        for c, tv in zip(cols, truth)]
print(" ".join(repr(x) for x in best))
print(" ".join(repr(x) for x in truth))
PY
    )"
    echo ">> truth   a=$TRUTH_A j=$TRUTH_J s=$TRUTH_S"
    echo ">> nearest a=$BEST_A j=$BEST_J s=$BEST_S"
fi

# --------------------------------------------------------------------------
# 4. De-drift at the chosen point (exact remap)
# --------------------------------------------------------------------------
DEMOD="$OUTDIR/${OUTBASE}_demod.dat"
echo ">> demodulating at the nearest grid point (demod/demod_dat.jl)"
if [ "$MODE" = "circular" ]; then
    "$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_dat.jl" \
        "$INJ" "$DEMOD" \
        --pb "$BEST_PB" --x "$BEST_X" --a-t "$BEST_AT" \
        2>&1 | tee "$OUTDIR/demod.log"
else
    "$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_dat.jl" \
        "$INJ" "$DEMOD" \
        --accel "$BEST_A" --jerk "$BEST_J" --snap "$BEST_S" \
        2>&1 | tee "$OUTDIR/demod.log"
fi

# --------------------------------------------------------------------------
# 5. realfft + rednoise + coherent search on the demodulated file
# --------------------------------------------------------------------------
cd "$OUTDIR"
DEMOD_FFT="${DEMOD%.dat}_red.fft"
echo ">> realfft + rednoise ${OUTBASE}_demod.dat"
"$REALFFT" "$DEMOD" > "$OUTDIR/realfft.log" 2>&1
"$REDNOISE" "${DEMOD%.dat}.fft" > "$OUTDIR/rednoise.log" 2>&1
[ -f "$DEMOD_FFT" ] || { echo "ERROR: no $DEMOD_FFT" >&2; exit 1; }

echo ">> coherent search on $DEMOD_FFT"
"$JULIA" --project="$COH_REPO" -t "$NTHREADS" "$COH_REPO/bin/coherent_search.jl" \
    "$DEMOD_FFT" $SEARCH_ARGS \
    -o "$OUTDIR/best.cohout" 2>&1 | tee "$OUTDIR/search.log"
cd - >/dev/null

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo
echo "=========================================================================="
echo "Injected truth: Pb=$PB_DAYS d, e=$E, sin_i=$SIN_I, p0=$P0 s, SNR=$SNR"
if [ "$MODE" = "circular" ]; then
    echo "  truth   pb=$TRUTH_PB x=$TRUTH_X at=$TRUTH_AT"
    echo "Demod at pb=$BEST_PB x=$BEST_X at=$BEST_AT"
else
    echo "  best-fit a=$TRUTH_A j=$TRUTH_J s=$TRUTH_S"
    echo "Demod at a=$BEST_A j=$BEST_J s=$BEST_S"
fi
echo "Grid:    $GRID_CSV ($NPOINTS point(s))"
echo "Candidates ($OUTDIR/best.cohout):"
cat "$OUTDIR/best.cohout"
echo "=========================================================================="
