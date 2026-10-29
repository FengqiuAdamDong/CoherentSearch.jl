#!/usr/bin/env bash
#
# End-to-end test of the circular-orbit demodulation path, a circular analogue
# of run_demod_search.sh:
#
#   1. Inject a phase-coherent pulsar in a pure Keplerian CIRCULAR binary
#      (pulsegen_gr.py, Roemer only: e=0, omega_peri=0, Shapiro off).
#   2. Derive the circular (pb, x, A_T) grid scoped to the injected orbit
#      (nsns_grid.py -mode circular).
#   3. Pick the grid point nearest the injected truth (each axis snapped).
#   4. De-drift at that point (demod_dat.jl --pb/--x/--a-t, exact voc remap).
#   5. realfft -> rednoise -> coherent search.
#   6. Assert the pulsar is recovered at ~1/p0 with S/N above THRESH.
#
#   bash demod/test_circular_demod.sh
#   SNR=12 P0=2.0 bash demod/test_circular_demod.sh
#
# Requires the same tools as run_demod_search.sh (julia, the CoherentSearch.jl
# project, a Python with numpy/scipy/matplotlib/yaml, PRESTO realfft+rednoise).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
: "${PB_DAYS:=0.041666666666666664}"  # orbital period [days]; 1 hour
: "${SIN_I:=0.5}"                     # sin(inclination)
: "${A_T:=0.7}"                       # mean anomaly at the obs midpoint [rad]
: "${P0:=1.0}"                        # intrinsic spin period [s]
: "${PDOT:=0}"                        # intrinsic period derivative [s/s]
: "${SNR:=30}"                        # injected S/N per observation
: "${PULSE_WIDTH:=0.01}"              # Gaussian pulse sigma [s]
: "${SEED:=7}"

: "${INF:=/home/fadong/Documents/CoherentSearch.jl/test_DM1778.00.inf}"

: "${PHASE_TOL:=0.1}"                 # accumulated-phase budget [cycles]
: "${THRESH:=6}"                      # search threshold
: "${MIN_SNR:=15}"                   # S/N the recovered pulsar must beat
: "${NTHREADS:=4}"

: "${COH_REPO:=/home/fadong/Documents/CoherentSearch.jl}"
: "${PYTHON:=/home/fadong/anaconda3/bin/python}"
: "${REALFFT:=/home/fadong/anaconda3/bin/realfft}"
: "${REDNOISE:=/home/fadong/anaconda3/bin/rednoise}"
: "${JULIA:=julia}"

: "${OUTDIR:=$PWD/circtest_out}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
: "${REAL_DAT:=${INF%.inf}.dat}"
PO_YEARS="$("$PYTHON" -c 'import sys; print(float(sys.argv[1])/365.25)' "$PB_DAYS")"

echo ">> output directory: $OUTDIR"

# --------------------------------------------------------------------------
# 1. Inject a pure circular Keplerian binary (Roemer only)
# --------------------------------------------------------------------------
echo ">> injecting circular binary: Pb=$PB_DAYS d, sin_i=$SIN_I, A_T=$A_T rad, p0=$P0 s, SNR=$SNR"
"$PYTHON" "$COH_REPO/demod/pulsegen_gr.py" \
    -inf "$INF" -outdir "$OUTDIR" -outbasename inj \
    -pb "$PB_DAYS" -e 0 -sin_i "$SIN_I" -omega_peri 0 -A_T "$A_T" \
    -shapiro_r 0 -shapiro_s 0 \
    -p0 "$P0" -pdot "$PDOT" -snr "$SNR" -pulse_width "$PULSE_WIDTH" \
    -seed "$SEED" -real "$REAL_DAT" \
    2>&1 | tee "$OUTDIR/pulsegen.log"

INJ="$OUTDIR/inj.dat"
TRUTH="$OUTDIR/inj_truth.yaml"
[ -f "$INJ" ] || { echo "ERROR: no $INJ" >&2; exit 1; }

# --------------------------------------------------------------------------
# 2. Derive the circular grid, scoped to the injected orbit
# --------------------------------------------------------------------------
echo ">> deriving circular grid (nsns_grid.py -mode circular) for p_o=$PO_YEARS yr"
"$PYTHON" "$COH_REPO/demod/nsns_grid.py" \
    -inf "$OUTDIR/inj.inf" -outdir "$OUTDIR" -outstem circgrid \
    -mode circular -p0 "$P0" \
    -p_o "$PO_YEARS" -sin_i "$SIN_I" \
    -phase_tol_cycles "$PHASE_TOL" \
    2>&1 | tee "$OUTDIR/derive.log"

GRID_CSV="$OUTDIR/circgrid.csv"
[ -f "$GRID_CSV" ] || { echo "ERROR: no grid CSV produced" >&2; exit 1; }

# --------------------------------------------------------------------------
# 3. Nearest grid point to the injected truth (uniform coords omega_b, x, A_T)
# --------------------------------------------------------------------------
read -r BEST_PB BEST_X BEST_AT TRUTH_PB TRUTH_X TRUTH_AT <<< "$(
"$PYTHON" - "$GRID_CSV" "$TRUTH" <<'PY'
import csv, sys, math, yaml
grid_csv, truth_yaml = sys.argv[1], sys.argv[2]
t = yaml.safe_load(open(truth_yaml))
truth = (float(t["pb_days"]), float(t["x_lt_s"]), float(t["A_T_rad"]))
rows = list(csv.DictReader(open(grid_csv)))

def key(row):
    pb, x, at = float(row["pb"]), float(row["x"]), float(row["at"])
    dw = 2*math.pi/(pb*86400.0) - 2*math.pi/(truth[0]*86400.0)   # rad/s, uniform
    dx = x - truth[1]
    da = (at - truth[2] + math.pi) % (2*math.pi) - math.pi      # wrapped
    return (dw, dx, da)

# nearest by per-axis snap on the actual grid values (a Cartesian product)
cols = [sorted({float(r[k]) for r in rows}) for k in ("pb", "x", "at")]
def snap(axis, val):
    return min(cols[axis], key=lambda v: abs(v - val))
best = (snap(0, truth[0]), snap(1, truth[1]), snap(2, truth[2]))
print(best[0], best[1], best[2], truth[0], truth[1], truth[2])
PY
)"

echo ">> truth   pb=$TRUTH_PB x=$TRUTH_X at=$TRUTH_AT"
echo ">> nearest pb=$BEST_PB x=$BEST_X at=$BEST_AT"

# --------------------------------------------------------------------------
# 4. De-drift at the chosen point (exact voc remap)
# --------------------------------------------------------------------------
DEMOD="$OUTDIR/inj_demod.dat"
echo ">> demodulating at the nearest circular grid point (demod_dat.jl)"
"$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_dat.jl" \
    "$INJ" "$DEMOD" --pb "$BEST_PB" --x "$BEST_X" --a-t "$BEST_AT" \
    2>&1 | tee "$OUTDIR/demod.log"

# --------------------------------------------------------------------------
# 5. realfft + rednoise + coherent search
# --------------------------------------------------------------------------
cd "$OUTDIR"
DEMOD_FFT="${DEMOD%.dat}_red.fft"
echo ">> realfft + rednoise inj_demod.dat"
"$REALFFT" "$DEMOD" > realfft.log 2>&1
"$REDNOISE" "${DEMOD%.dat}.fft" > rednoise.log 2>&1
[ -f "$DEMOD_FFT" ] || { echo "ERROR: no $DEMOD_FFT" >&2; exit 1; }

echo ">> coherent search on $DEMOD_FFT"
"$JULIA" --project="$COH_REPO" -t "$NTHREADS" "$COH_REPO/bin/coherent_search.jl" \
    "$DEMOD_FFT" --threshold "$THRESH" \
    -o "$OUTDIR/best.cohout" > search.log 2>&1
cd - >/dev/null

# --------------------------------------------------------------------------
# 6. Assert detection at ~1/p0
# --------------------------------------------------------------------------
EXPECT_F="$("$PYTHON" -c 'import sys; print(1.0/float(sys.argv[1]))' "$P0")"
echo
echo "=========================================================================="
echo "Injected: Pb=$PB_DAYS d, sin_i=$SIN_I, A_T=$A_T rad, p0=$P0 s (f=$EXPECT_F Hz), SNR=$SNR"
echo "Demod at: pb=$BEST_PB x=$BEST_X at=$BEST_AT"
cat "$OUTDIR/best.cohout"

"$PYTHON" - "$OUTDIR/best.cohout" "$EXPECT_F" "$MIN_SNR" "$P0" <<'PY'
import sys
cohout, expect_f, min_snr, p0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
rows = [l.split() for l in open(cohout) if l[:1].isdigit()]
if not rows:
    sys.exit("FAIL: no candidates in .cohout")
best = max(rows, key=lambda r: float(r[1]))
snr, freq = float(best[1]), float(best[2])
# frequency must be within one Fourier bin (1/T) of 1/p0; read T from the .inf
inf = cohout.replace("best.cohout", "inj_demod.inf")
N = dt = None
for line in open(inf):
    if line.strip().startswith("Number of bins"):
        N = int(line.split("=")[1])
    if line.strip().startswith("Width of each time series bin"):
        dt = float(line.split("=")[1])
df = 1.0 / (N * dt) if (N and dt) else abs(expect_f) * 1e-3
ok = snr >= min_snr and abs(freq - expect_f) <= 3 * df
print(f"\nBest candidate: S/N {snr:.2f} at {freq:.9f} Hz (expected {expect_f:.9f} +/- {3*df:.2e})")
print(f"Require S/N >= {min_snr} and |df| <= 3 bins ({3*df:.3e} Hz)")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo "=========================================================================="
