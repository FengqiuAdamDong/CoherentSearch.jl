#!/usr/bin/env bash
#
# End-to-end accel/jerk/snap demodulation + narrow coherent search.
#
#   1. Inject a phase-coherent pulsar with accel/jerk/snap into a real
#      observation (test_DM1778.00.dat by default) using the sibling repo's
#      orbit_simulation/generate_search_timeseries.py.
#   2. De-drift the injected .dat with the exact time remap
#      (demod/demod_dat.jl, a Julia port of FFA_stacking/demodulation/utils.py
#      resample_ts_shift_snap), writing a resampled .dat + .inf.
#   3. realfft the resampled .dat -> .fft.
#   4. rednoise the .fft -> _red.fft (normalises powers to mean 1 and whitens).
#   5. Run the narrow coherent search around 1/P0 on the demodulated _red.fft.
#      The same chain is also run on the un-demodulated file as a control.
#
# Everything is configured through environment variables; see the defaults
# below.  Output lands in $OUTDIR.
#
#   bash demod/run_demod_search.sh
#   A_LOS=250 J_LOS=3e-4 S_LOS=8e-7 bash demod/run_demod_search.sh
#
# Requires: julia (+ the CoherentSearch.jl project), Python with numpy/scipy/
# yaml (the pixi/anaconda env that carries FFA_stacking's deps), and PRESTO's
# realfft + rednoise on PATH (or REALFFT/REDNOISE set).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
: "${MODE:=snap}"                     # accel | jerk | snap
: "${A_LOS:=200}"                    # injected LOS acceleration [m/s^2]
: "${J_LOS:=0.0002}"                 # injected LOS jerk [m/s^3]
: "${S_LOS:=5e-7}"                   # injected LOS snap [m/s^4]
: "${V_LOS0:=0.0}"                   # constant LOS velocity [m/s]

: "${P0:=1.0}"                       # intrinsic spin period at the epoch [s]
: "${DM:=1778.0}"                    # DM recorded in the .inf
: "${SNR:=30}"                       # injected S/N per observation
: "${PULSE_WIDTH:=0.01}"             # Gaussian pulse sigma [s]
: "${SEED:=42}"
: "${TSAMP:=0.0098304}"              # sampling interval [s]
: "${TOBS:=1509.94944}"              # observation length [s]

# The real observation to draw the noise background from.  The generator
# takes a random tobs-length slice (seeded by SEED) and injects into it.
: "${REAL_DAT:=/home/fadong/Documents/CoherentSearch.jl/test_DM1778.00.dat}"

# Search band (fundamentals, Hz): 1/P0 +/- BAND_FRAC.
: "${BAND_FRAC:=0.02}"
: "${NHARMS:=16}"
: "${MAXDECIM:=1}"
: "${THRESHOLD:=6.0}"
: "${NTHREADS:=4}"

# Tool locations.
: "${COH_REPO:=/home/fadong/Documents/CoherentSearch.jl}"
: "${FFA_REPO:=/home/fadong/Documents/FFA_stacking}"
: "${PYTHON:=/home/fadong/anaconda3/bin/python}"
: "${REALFFT:=/home/fadong/anaconda3/bin/realfft}"
: "${REDNOISE:=/home/fadong/anaconda3/bin/rednoise}"
: "${JULIA:=julia}"

: "${OUTDIR:=$PWD/demod_out}"
: "${OUTBASE:=inj_${MODE}}"

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

echo ">> output directory: $OUTDIR"

# --------------------------------------------------------------------------
# 1. Inject the pulsar
# --------------------------------------------------------------------------
REAL_ARG=()
if [ -n "$REAL_DAT" ]; then REAL_ARG=(-real "$REAL_DAT"); fi

echo ">> injecting mode=$MODE a=$A_LOS j=$J_LOS s=$S_LOS into ${REAL_DAT:-Gaussian noise}"
"$PYTHON" "$FFA_REPO/orbit_simulation/generate_search_timeseries.py" \
    -mode "$MODE" \
    -outdir "$OUTDIR" \
    -outbasename "$OUTBASE" \
    -n_obs 1 \
    -tobs "$TOBS" \
    -tsamp "$TSAMP" \
    -p0 "$P0" \
    -dm "$DM" \
    -snr "$SNR" \
    -pulse_width "$PULSE_WIDTH" \
    -a_los "$A_LOS" \
    -j_los="$J_LOS" \
    -s_los="$S_LOS" \
    -v_los0 "$V_LOS0" \
    -seed "$SEED" \
    "${REAL_ARG[@]}" \
    2>&1 | tee "$OUTDIR/generate.log"

INJ="$OUTDIR/${OUTBASE}_000.dat"
INJ_INF="$OUTDIR/${OUTBASE}_000.inf"
[ -f "$INJ" ] || { echo "ERROR: generator produced no $INJ" >&2; exit 1; }

# --------------------------------------------------------------------------
# 2. De-drift with the exact accel/jerk/snap remap (Julia)
# --------------------------------------------------------------------------
DEMOD="$OUTDIR/${OUTBASE}_000_demod.dat"
echo ">> demodulating with demod/demod_dat.jl"
"$JULIA" --project="$COH_REPO" "$COH_REPO/demod/demod_dat.jl" \
    "$INJ" "$DEMOD" \
    --accel "$A_LOS" --jerk "$J_LOS" --snap "$S_LOS" --v0 "$V_LOS0" \
    2>&1 | tee "$OUTDIR/demod.log"

# --------------------------------------------------------------------------
# 3-4. realfft + rednoise, for both the demodulated and the raw file
# --------------------------------------------------------------------------
cd "$OUTDIR"
for base in "${OUTBASE}_000_demod" "${OUTBASE}_000"; do
    echo ">> realfft $base.dat"
    "$REALFFT" "$base.dat" > "$OUTDIR/realfft_$base.log" 2>&1
    echo ">> rednoise $base.fft"
    "$REDNOISE" "$base.fft" > "$OUTDIR/rednoise_$base.log" 2>&1
    [ -f "$base"'_red.fft' ] || { echo "ERROR: rednoise produced no ${base}_red.fft" >&2; exit 1; }
done
cd - >/dev/null

DEMOD_FFT="$OUTDIR/${OUTBASE}_000_demod_red.fft"
RAW_FFT="$OUTDIR/${OUTBASE}_000_red.fft"

# --------------------------------------------------------------------------
# 5. Narrow coherent search around 1/P0 on both
# --------------------------------------------------------------------------
LOFREQ=$("$PYTHON" -c "print(1.0/$P0*(1.0-$BAND_FRAC))")
HIFREQ=$("$PYTHON" -c "print(1.0/$P0*(1.0+$BAND_FRAC))")
echo ">> coherent search band: $LOFREQ .. $HIFREQ Hz"

run_search() {
    local fft=$1 out=$2
    "$JULIA" --project="$COH_REPO" -t "$NTHREADS" "$COH_REPO/bin/coherent_search.jl" \
        "$fft" \
        --lofreq "$LOFREQ" --hifreq "$HIFREQ" \
        --nharms "$NHARMS" --maxdecim "$MAXDECIM" \
        --threshold "$THRESHOLD" \
        -o "$out"
}

echo ">> searching the DEMODULATED file"
run_search "$DEMOD_FFT" "$OUTDIR/demod.cohout" 2>&1 | tee "$OUTDIR/search_demod.log"

echo ">> searching the RAW (un-demodulated) file, as a control"
run_search "$RAW_FFT" "$OUTDIR/raw.cohout" 2>&1 | tee "$OUTDIR/search_raw.log"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo
echo "=========================================================================="
echo "Demodulated search ($DEMOD_FFT):"
cat "$OUTDIR/demod.cohout"
echo
echo "Raw control search ($RAW_FFT):"
cat "$OUTDIR/raw.cohout"
echo "=========================================================================="
echo "Injected truth: mode=$MODE a=$A_LOS m/s^2 j=$J_LOS m/s^3 s=$S_LOS m/s^4,"
echo "                P0=$P0 s at the anchor, band $LOFREQ..$HIFREQ Hz."
echo "Expected demodulated peak at $("$PYTHON" -c "print(1.0/$P0)") Hz."
