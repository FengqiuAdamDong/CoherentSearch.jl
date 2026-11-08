#!/usr/bin/env bash
#
# bin/prep_observation.sh -- on-disk PRESTO preprocessing for CoherentSearch.jl
#
#   rfifind  ->  prepsubband  ->  realfft  [-> rednoise]  [-> zapbirds]
#
# rfifind finds narrow- and wide-band RFI and writes a time/frequency mask;
# prepsubband dedisperses the raw data at each requested DM and applies that
# mask; realfft writes the PRESTO .fft the search reads.  The script stops
# after realfft by default so the FFT can be inspected and its Fourier bins
# masked.  --rednoise also writes <stem>_red.fft, and --zapfile zaps a list of
# Fourier frequencies (birdies) in it with PRESTO's zapbirds.
#
# ORDER MATTERS: zap the _red.fft, never the raw .fft.
#   * rednoise normalizes each bin by a running median of the RAW spectrum.
#     Zapping first would make it normalize against holes and rescale the
#     surrounding bins.
#   * zapbirds (PRESTO >= 6) replaces zapped bins with noise, so the unit-mean
#     power the search's analytic sigma assumes is preserved.  Do NOT pass
#     -constamp, and if you zero bins by hand instead, pass --sigma measured
#     to coherent_search.jl.
#   * the raw <stem>.fft is never modified, so <stem>_red.fft is always
#     reproducible from it; re-zapping with a changed list regenerates it.
#
# Every step is idempotent: it is skipped when its outputs already exist (and,
# for the zap, when the recorded zaplist hash still matches), and --force
# rebuilds everything.  This is the on-disk counterpart to
# `coherent_search.jl --demod-grid`, which dereddens unconditionally in RAM and
# never writes an .fft (see demod/README.md section 11).
#
# Examples
#   bin/prep_observation.sh OBS.fil --dm 238.60 --ncpus 8
#   bin/prep_observation.sh OBS.fil --dm 238.60,445.0 --no-bary --ncpus 8
#   bin/prep_observation.sh OBS.fil --dm 238.60 --rednoise --zapfile zaplist.txt
#   bin/prep_observation.sh --dry-run OBS.fil --dm 238.60
#
# Tools default to $PATH and then to /home/fadong/anaconda3/bin (override with
# RFIFIND, PREPSUBBAND, REALFFT, REDNOISE, ZAPBIRDS).
#
# Requires: bash 4+, PRESTO (rfifind, prepsubband, realfft, rednoise, zapbirds),
# and TEMPO/tempo2 for barycentering (use --no-bary if it is not installed).

set -euo pipefail

PROG=$(basename "$0")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }
warn() { echo "WARNING: $*" >&2; }

usage() {
    cat <<EOF
Usage: $PROG [OPTIONS] RAWFILE [RAWFILE ...]

On-disk preprocessing: rfifind -> prepsubband -> realfft [-> rednoise]
[-> zapbirds].  Stops after realfft unless --rednoise/--zapfile is given.

Options (env defaults in parentheses):
  -o, --outdir DIR          output directory (\$PWD/prep_<NAME>)
  -n, --name NAME           observation stem (first input's basename without
                            its suffix).  Required when more than one raw file
                            is given (split raw files are one observation).
  -d, --dm LIST             comma/space-separated dispersion measures
      --dmprec P            decimals in the DM filename field (2; 0-6)
      --nsub N              prepsubband -nsub N
      --downsamp N          prepsubband -downsamp N
      --no-bary             do not barycenter (prepsubband -nobary)
      --mask FILE           use this rfifind mask and skip rfifind
      --no-mask             skip rfifind and run prepsubband unmasked
      --rfifind-time S      rfifind -time (30)
      --ncpus N             -ncpus for rfifind/prepsubband (1)
      --rfifind-args S      extra rfifind arguments (whitespace-split)
      --prepsubband-args S  extra prepsubband arguments (whitespace-split)
      --rednoise            also write <stem>_red.fft and <stem>_red.inf
      --zapfile FILE        zap the listed Fourier bins in <stem>_red.fft
                            (implies --rednoise)
      --baryv V             zapbirds -baryv (0; requires --zapfile)
      --force               rebuild every step, even if its outputs exist
      --dry-run             print the commands and the reuse decisions only
  -h, --help                show this help and exit

Outputs (in OUTDIR):
  <NAME>_rfifind.{mask,stats,inf}    rfifind mask
  <NAME>_DM<dm>.{dat,inf}            dedispersed time series
  <NAME>_DM<dm>.fft                   raw FFT; never modified afterwards
  <NAME>_DM<dm>_red.{fft,inf}         dereddened FFT (--rednoise/--zapfile)
  <NAME>_DM<dm>_red.zapped            sha256 of the zaplist last applied

The zaplist format is PRESTO's: one "frequency_Hz width_Hz" pair per line,
'#' comments allowed.  zapbirds replaces the zapped bins with noise.  If you
zero bins yourself, search with --sigma measured.
EOF
}

# ---------------------------------------------------------------------------
# Defaults and option parsing
# ---------------------------------------------------------------------------
OUTDIR=""
NAME=""
NAME_GIVEN=0
DMS=""
DMPREC=2
NSUB=""
DOWNSAMP=""
NOBARY=0
MASK=""
MASK_GIVEN=0
NO_MASK=0
RFITIME=30
NCPUS="${NCPUS:-1}"
RFIFIND_ARGS="${RFIFIND_ARGS:-}"
PREPSUBBAND_ARGS="${PREPSUBBAND_ARGS:-}"
DO_REDNOISE=0
ZAPFILE=""
ZAP_GIVEN=0
BARYV="${BARYV:-0}"
BARYV_GIVEN=0
FORCE=0
DRY_RUN=0
INPUTS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -o|--outdir)   [ "$#" -ge 2 ] || die "$1 needs a value"; OUTDIR=$2; shift 2 ;;
        --outdir=*)    OUTDIR=${1#*=}; shift ;;
        -n|--name)     [ "$#" -ge 2 ] || die "$1 needs a value"; NAME=$2; NAME_GIVEN=1; shift 2 ;;
        --name=*)      NAME=${1#*=}; NAME_GIVEN=1; shift ;;
        -d|--dm)       [ "$#" -ge 2 ] || die "$1 needs a value"; DMS=$2; shift 2 ;;
        --dm=*)        DMS=${1#*=}; shift ;;
        --dmprec)      [ "$#" -ge 2 ] || die "$1 needs a value"; DMPREC=$2; shift 2 ;;
        --dmprec=*)    DMPREC=${1#*=}; shift ;;
        --nsub)        [ "$#" -ge 2 ] || die "$1 needs a value"; NSUB=$2; shift 2 ;;
        --nsub=*)      NSUB=${1#*=}; shift ;;
        --downsamp)    [ "$#" -ge 2 ] || die "$1 needs a value"; DOWNSAMP=$2; shift 2 ;;
        --downsamp=*)  DOWNSAMP=${1#*=}; shift ;;
        --no-bary)     NOBARY=1; shift ;;
        --mask)        [ "$#" -ge 2 ] || die "$1 needs a value"; MASK=$2; MASK_GIVEN=1; shift 2 ;;
        --mask=*)      MASK=${1#*=}; MASK_GIVEN=1; shift ;;
        --no-mask)     NO_MASK=1; shift ;;
        --rfifind-time)   [ "$#" -ge 2 ] || die "$1 needs a value"; RFITIME=$2; shift 2 ;;
        --rfifind-time=*) RFITIME=${1#*=}; shift ;;
        --ncpus)       [ "$#" -ge 2 ] || die "$1 needs a value"; NCPUS=$2; shift 2 ;;
        --ncpus=*)     NCPUS=${1#*=}; shift ;;
        --rfifind-args)   [ "$#" -ge 2 ] || die "$1 needs a value"; RFIFIND_ARGS=$2; shift 2 ;;
        --rfifind-args=*) RFIFIND_ARGS=${1#*=}; shift ;;
        --prepsubband-args)   [ "$#" -ge 2 ] || die "$1 needs a value"; PREPSUBBAND_ARGS=$2; shift 2 ;;
        --prepsubband-args=*) PREPSUBBAND_ARGS=${1#*=}; shift ;;
        --rednoise)    DO_REDNOISE=1; shift ;;
        --zapfile)     [ "$#" -ge 2 ] || die "$1 needs a value"; ZAPFILE=$2; ZAP_GIVEN=1; DO_REDNOISE=1; shift 2 ;;
        --zapfile=*)   ZAPFILE=${1#*=}; ZAP_GIVEN=1; DO_REDNOISE=1; shift ;;
        --baryv)       [ "$#" -ge 2 ] || die "$1 needs a value"; BARYV=$2; BARYV_GIVEN=1; shift 2 ;;
        --baryv=*)     BARYV=${1#*=}; BARYV_GIVEN=1; shift ;;
        --force)       FORCE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --)            shift; INPUTS+=("$@"); break ;;
        -*)            die "unknown option: $1 (try --help)" ;;
        *)             INPUTS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
[ "${#INPUTS[@]}" -gt 0 ] || die "no input files (try --help)"
[ -n "$DMS" ] || die "--dm is required"

if [ "$NO_MASK" = 1 ] && [ "$MASK_GIVEN" = 1 ]; then
    die "--mask and --no-mask are mutually exclusive"
fi
if [ "$MASK_GIVEN" = 1 ] && [ -z "$MASK" ]; then
    die "--mask needs a non-empty path"
fi
if [ "$ZAP_GIVEN" = 1 ] && [ -z "$ZAPFILE" ]; then
    die "--zapfile needs a non-empty path"
fi
if [ "$NAME_GIVEN" = 1 ] && [ -z "$NAME" ]; then
    die "--name needs a non-empty value"
fi
if [ "$BARYV_GIVEN" = 1 ] && [ -z "$ZAPFILE" ]; then
    die "--baryv only applies with --zapfile"
fi

case "$DMPREC" in
    ''|*[!0-9]*) die "--dmprec must be an integer between 0 and 6" ;;
esac
[ "$DMPREC" -le 6 ] || die "--dmprec must be between 0 and 6"

case "$NCPUS" in
    ''|*[!0-9]*|0) die "--ncpus must be a positive integer" ;;
esac
for pair in "NSUB:$NSUB" "DOWNSAMP:$DOWNSAMP"; do
    key=${pair%%:*}; val=${pair#*:}
    [ -n "$val" ] || continue
    case "$val" in
        ''|*[!0-9]*|0) die "--${key,,} must be a positive integer" ;;
    esac
done
if ! awk -v x="$RFITIME" 'BEGIN { exit !(x ~ /^[0-9]+([.][0-9]*)?$/ && x > 0) }'; then
    die "--rfifind-time must be a positive number"
fi
if ! awk -v x="$BARYV" 'BEGIN { exit !(x ~ /^-?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/) }'; then
    die "--baryv must be a number"
fi

# Inputs must be raw data, not something this script already produced.
first="${INPUTS[0]}"
for f in "${INPUTS[@]}"; do
    case "$(basename "$f")" in
        *.fft|*.dat|*.inf|*.mask|*.stats|*.cohout)
            die "input '$f' is already processed; this script starts from raw data (for an existing .dat, run realfft directly)" ;;
    esac
    case "$(basename "$f")" in
        *.*) ;;
        *) die "input '$f' has no suffix; rfifind/prepsubband require one" ;;
    esac
done
base=$(basename "$first")

if [ "$NAME_GIVEN" = 0 ]; then
    if [ "${#INPUTS[@]}" -gt 1 ]; then
        die "more than one raw file: pass --name OBS (they are treated as one observation)"
    fi
    NAME="${base%.*}"
fi
case "$NAME" in
    */*) die "--name must be a basename, not a path" ;;
    *_red) die "--name must not end in _red: that suffix is reserved for the dereddened FFT" ;;
esac

# Resolve inputs before changing directory.
_abs() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd -P)" "$(basename "$1")"
    fi
}
RESOLVED=()
for f in "${INPUTS[@]}"; do
    [ -e "$f" ] || die "input not found: $f"
    RESOLVED+=("$(_abs "$f")")
done
INPUTS=("${RESOLVED[@]}")

if [ "$MASK_GIVEN" = 1 ]; then
    [ -f "$MASK" ] || die "mask not found: $MASK"
    MASK=$(_abs "$MASK")
fi
if [ -n "$ZAPFILE" ]; then
    [ -f "$ZAPFILE" ] || die "zaplist not found: $ZAPFILE"
    [ -s "$ZAPFILE" ] || die "zaplist is empty: $ZAPFILE"
    ZAPFILE=$(_abs "$ZAPFILE")
fi

# ---------------------------------------------------------------------------
# Outdir, logs, tools
# ---------------------------------------------------------------------------
if [ -z "$OUTDIR" ]; then
    OUTDIR="$PWD/prep_$NAME"
fi
if [ "$DRY_RUN" = 1 ]; then
    OUTDIR=$(realpath -m "$OUTDIR" 2>/dev/null || printf '%s' "$OUTDIR")
    LOGDIR="$OUTDIR/logs"
else
    mkdir -p "$OUTDIR" || die "cannot create $OUTDIR"
    OUTDIR=$(cd "$OUTDIR" && pwd -P)
    LOGDIR="$OUTDIR/logs"
    mkdir -p "$LOGDIR" || die "cannot create $LOGDIR"
fi

resolve_tool() {
    # $1 = environment variable name, $2 = fallback absolute path
    local var=$1 fallback=$2 val
    val="${!var:-}"
    if [ -n "$val" ]; then
        if [ -x "$val" ] || command -v "$val" >/dev/null 2>&1; then
            printf '%s\n' "$val"
            return 0
        fi
        return 1
    fi
    if command -v "$(basename "$fallback")" >/dev/null 2>&1; then
        command -v "$(basename "$fallback")"
        return 0
    fi
    if [ -x "$fallback" ]; then
        printf '%s\n' "$fallback"
        return 0
    fi
    return 1
}

RFIFIND=""
PREPSUBBAND=""
REALFFT=""
REDNOISE=""
ZAPBIRDS=""
if [ "$MASK_GIVEN" = 0 ] && [ "$NO_MASK" = 0 ]; then
    RFIFIND=$(resolve_tool RFIFIND /home/fadong/anaconda3/bin/rfifind) \
        || die "rfifind not found (set RFIFIND=/path/to/rfifind)"
fi
PREPSUBBAND=$(resolve_tool PREPSUBBAND /home/fadong/anaconda3/bin/prepsubband) \
    || die "prepsubband not found (set PREPSUBBAND=/path/to/prepsubband)"
REALFFT=$(resolve_tool REALFFT /home/fadong/anaconda3/bin/realfft) \
    || die "realfft not found (set REALFFT=/path/to/realfft)"
if [ "$DO_REDNOISE" = 1 ]; then
    REDNOISE=$(resolve_tool REDNOISE /home/fadong/anaconda3/bin/rednoise) \
        || die "rednoise not found (set REDNOISE=/path/to/rednoise)"
fi
if [ -n "$ZAPFILE" ]; then
    ZAPBIRDS=$(resolve_tool ZAPBIRDS /home/fadong/anaconda3/bin/zapbirds) \
        || die "zapbirds not found (set ZAPBIRDS=/path/to/zapbirds)"
fi

# ---------------------------------------------------------------------------
# DM list
# ---------------------------------------------------------------------------
DM_LIST=()
DM_STEMS=()
declare -A DM_SEEN=()
IFS=', ' read -r -a _dm_tokens <<< "$DMS"
for tok in "${_dm_tokens[@]}"; do
    [ -n "$tok" ] || continue
    if ! awk -v x="$tok" 'BEGIN { exit !(x ~ /^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/) }'; then
        die "invalid DM: '$tok'"
    fi
    fmt=$(printf "%.${DMPREC}f" "$tok")
    if [ -n "${DM_SEEN[$fmt]:-}" ]; then
        die "two DMs format to the same filename at --dmprec $DMPREC: '$tok' -> $fmt"
    fi
    DM_SEEN[$fmt]=1
    DM_LIST+=("$tok")
    DM_STEMS+=("${NAME}_DM${fmt}")
done
[ "${#DM_LIST[@]}" -gt 0 ] || die "--dm listed no usable DMs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_step() {
    local log=$1; shift
    if [ "$DRY_RUN" = 1 ]; then
        printf '   (cd %q &&' "$OUTDIR"
        printf ' %q' "$@"
        printf ')\n'
        return 0
    fi
    mkdir -p "$LOGDIR"
    {
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    } > "$log"
    if ! ( cd "$OUTDIR" && "$@" ) >> "$log" 2>&1; then
        echo "ERROR: command failed (log: $log):" >&2
        printf '  ' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        tail -n 20 "$log" >&2 || true
        return 1
    fi
}

newer_than() {
    local ref=$1; shift
    local f
    for f in "$@"; do
        if [ "$f" -nt "$ref" ]; then
            return 0
        fi
    done
    return 1
}

sz() { wc -c < "$1" | tr -d '[:space:]'; }

inf_int() {
    # $1 = .inf, $2 = field prefix
    awk -F= -v key="$2" 'index($1, key) { gsub(/[ \t\r]/, "", $2); print $2; exit }' "$1"
}

check_stem_sizes() {
    local stem=$1 n want got
    n=$(inf_int "$OUTDIR/$stem.inf" "Number of bins in the time series")
    case "$n" in
        ''|*[!0-9]*) die "cannot read a usable 'Number of bins' from $OUTDIR/$stem.inf" ;;
    esac
    want=$((4 * n))
    got=$(sz "$OUTDIR/$stem.dat")
    [ "$got" -eq "$want" ] || die "$stem.dat is $got bytes, expected 4*N=$want; rerun with --force"
    got=$(sz "$OUTDIR/$stem.fft")
    [ "$got" -eq "$want" ] || die "$stem.fft is $got bytes, expected 4*N=$want; rerun with --force"
}

file_sha() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "need sha256sum or shasum to guard the zap step"
    fi
}

RFI_ARGS=()
if [ -n "$RFIFIND_ARGS" ]; then read -r -a RFI_ARGS <<< "$RFIFIND_ARGS"; fi
PREP_ARGS=()
if [ -n "$PREPSUBBAND_ARGS" ]; then read -r -a PREP_ARGS <<< "$PREPSUBBAND_ARGS"; fi

# ---------------------------------------------------------------------------
# 1. rfifind
# ---------------------------------------------------------------------------
if [ "$MASK_GIVEN" = 1 ]; then
    note "rfifind: skip (using supplied mask $MASK)"
elif [ "$NO_MASK" = 1 ]; then
    note "rfifind: skip (--no-mask)"
else
    rfi_base="$OUTDIR/${NAME}_rfifind"
    if [ "$FORCE" = 0 ] \
       && [ -s "${rfi_base}.mask" ] && [ -s "${rfi_base}.stats" ] && [ -s "${rfi_base}.inf" ]; then
        note "rfifind: skip (exists) ${rfi_base}.mask"
        if newer_than "${rfi_base}.mask" "${INPUTS[@]}"; then
            warn "an input is newer than ${rfi_base}.mask; rerun with --force to rebuild it"
        fi
    else
        note "rfifind: $NAME"
        run_step "$LOGDIR/rfifind.log" "$RFIFIND" -ncpus "$NCPUS" -time "$RFITIME" \
            -o "$NAME" "${RFI_ARGS[@]}" -- "${INPUTS[@]}"
        if [ "$DRY_RUN" = 0 ]; then
            [ -s "${rfi_base}.mask" ]  || die "rfifind did not write ${rfi_base}.mask"
            [ -s "${rfi_base}.stats" ] || die "rfifind did not write ${rfi_base}.stats"
            [ -s "${rfi_base}.inf" ]   || die "rfifind did not write ${rfi_base}.inf"
        fi
    fi
    MASK="${rfi_base}.mask"
fi

MASK_ARGS=()
if [ -n "$MASK" ]; then MASK_ARGS=(-mask "$MASK"); fi
NSUB_ARGS=()
if [ -n "$NSUB" ]; then NSUB_ARGS=(-nsub "$NSUB"); fi
DOWNSAMP_ARGS=()
if [ -n "$DOWNSAMP" ]; then DOWNSAMP_ARGS=(-downsamp "$DOWNSAMP"); fi
BARY_ARGS=()
if [ "$NOBARY" = 1 ]; then BARY_ARGS=(-nobary); fi

OUTPUTS=()
if [ -n "$MASK" ]; then OUTPUTS+=("$MASK"); fi

# ---------------------------------------------------------------------------
# 2-5. per-DM chain
# ---------------------------------------------------------------------------
for idx in "${!DM_LIST[@]}"; do
    dm="${DM_LIST[$idx]}"
    stem="${DM_STEMS[$idx]}"

    # ---- prepsubband ------------------------------------------------------
    if [ "$FORCE" = 0 ] && [ -s "$OUTDIR/$stem.dat" ] && [ -s "$OUTDIR/$stem.inf" ]; then
        note "prepsubband: skip (exists) $stem"
        if newer_than "$OUTDIR/$stem.dat" "${INPUTS[@]}"; then
            warn "an input is newer than $stem.dat; rerun with --force to rebuild it"
        fi
    else
        note "prepsubband: DM $dm -> $stem"
        run_step "$LOGDIR/prepsubband_${stem}.log" "$PREPSUBBAND" -ncpus "$NCPUS" \
            -lodm "$dm" -dmstep 1 -numdms 1 -dmprec "$DMPREC" \
            "${MASK_ARGS[@]}" "${NSUB_ARGS[@]}" "${DOWNSAMP_ARGS[@]}" "${BARY_ARGS[@]}" \
            "${PREP_ARGS[@]}" -o "$NAME" -- "${INPUTS[@]}"
        if [ "$DRY_RUN" = 0 ]; then
            [ -s "$OUTDIR/$stem.dat" ] || die "prepsubband did not write $stem.dat"
            [ -s "$OUTDIR/$stem.inf" ] || die "prepsubband did not write $stem.inf"
        fi
    fi
    if [ "$DRY_RUN" = 0 ] && [ "$NOBARY" = 0 ]; then
        grep -qE 'Barycentered\?.*= *1' "$OUTDIR/$stem.inf" \
            || die "$stem.inf is not barycentered; install TEMPO/tempo2 or pass --no-bary"
    fi
    OUTPUTS+=("$OUTDIR/$stem.dat" "$OUTDIR/$stem.inf")

    # ---- realfft ----------------------------------------------------------
    if [ "$FORCE" = 0 ] && [ -s "$OUTDIR/$stem.fft" ] && [ -s "$OUTDIR/$stem.inf" ]; then
        note "realfft: skip (exists) $stem.fft"
        if newer_than "$OUTDIR/$stem.fft" "$OUTDIR/$stem.dat"; then
            warn "$stem.dat is newer than $stem.fft; rerun with --force to rebuild it"
        fi
    else
        note "realfft: $stem"
        run_step "$LOGDIR/realfft_${stem}.log" "$REALFFT" -- "$stem.dat"
    fi
    if [ "$DRY_RUN" = 0 ]; then
        [ -s "$OUTDIR/$stem.fft" ] || die "realfft did not write $stem.fft"
        check_stem_sizes "$stem"
    fi
    OUTPUTS+=("$OUTDIR/$stem.fft")

    # ---- rednoise ---------------------------------------------------------
    if [ "$DO_REDNOISE" = 1 ]; then
        red="$OUTDIR/${stem}_red.fft"
        redinf="$OUTDIR/${stem}_red.inf"
        if [ "$FORCE" = 0 ] && [ -s "$red" ] && [ -s "$redinf" ] \
           && [ "$(sz "$red")" -eq "$(sz "$OUTDIR/$stem.fft")" ] \
           && ! [ "$OUTDIR/$stem.fft" -nt "$red" ]; then
            note "rednoise: skip (exists) ${stem}_red.fft"
        else
            note "rednoise: $stem"
            run_step "$LOGDIR/rednoise_${stem}.log" "$REDNOISE" -- "$stem.fft"
            if [ "$DRY_RUN" = 0 ]; then
                [ -s "$red" ]    || die "rednoise did not write ${stem}_red.fft"
                [ -s "$redinf" ] || die "rednoise did not write ${stem}_red.inf"
                [ "$(sz "$red")" -eq "$(sz "$OUTDIR/$stem.fft")" ] \
                    || die "${stem}_red.fft has the wrong size; rerun with --force"
            fi
        fi
        OUTPUTS+=("$red" "$redinf")
    fi

    # ---- zapbirds ---------------------------------------------------------
    if [ -n "$ZAPFILE" ]; then
        red="$OUTDIR/${stem}_red.fft"
        redinf="$OUTDIR/${stem}_red.inf"
        marker="$OUTDIR/${stem}_red.zapped"
        if [ "$DRY_RUN" = 0 ]; then
            [ -s "$red" ]    || die "missing $red (the zap runs on the dereddened FFT)"
            [ -s "$redinf" ] || die "missing $redinf (zapbirds reads it)"
        fi
        zsha=$(file_sha "$ZAPFILE")
        do_zap=1
        if [ "$FORCE" = 0 ] && [ -s "$marker" ]; then
            old=$(awk 'NR == 1 { print $1 }' "$marker")
            if [ "$old" = "$zsha" ]; then
                if [ "$DRY_RUN" = 1 ] || [ "$marker" -nt "$red" ]; then
                    do_zap=0
                    note "zapbirds: skip (already zapped with this list) $stem"
                fi
            else
                note "zapbirds: zaplist changed; regenerate ${stem}_red.fft from the raw ${stem}.fft"
                run_step "$LOGDIR/rednoise_${stem}.log" "$REDNOISE" -- "$stem.fft"
            fi
        fi
        if [ "$do_zap" = 1 ]; then
            note "zapbirds: $stem"
            run_step "$LOGDIR/zap_${stem}.log" "$ZAPBIRDS" -zap -zapfile "$ZAPFILE" \
                -baryv "$BARYV" -- "${stem}_red.fft"
            if [ "$DRY_RUN" = 0 ]; then
                [ "$(sz "$red")" -eq "$(sz "$OUTDIR/$stem.fft")" ] \
                    || die "zapbirds changed the size of ${stem}_red.fft"
                printf '%s  %s\n' "$zsha" "$ZAPFILE" > "$marker"
            fi
        fi
        OUTPUTS+=("$marker")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
if [ "$DRY_RUN" = 1 ]; then
    note "dry run: no files were written"
else
    note "done"
fi
for f in "${OUTPUTS[@]}"; do
    printf '   %s\n' "$f"
done

echo
echo "next steps:"
if [ -n "$MASK" ] && [ -s "$MASK" ] && command -v rfifind_stats.py >/dev/null 2>&1; then
    printf '  mask summary : rfifind_stats.py %q\n' "$MASK"
fi
printf '  inspect FFT  : explorefft %q\n' "$OUTDIR/${DM_STEMS[0]}.fft"
if [ "$DO_REDNOISE" = 0 ]; then
    printf '  normalize    : (cd %q && rednoise %q)\n' "$OUTDIR" "${DM_STEMS[0]}.fft"
fi
if [ -n "$ZAPFILE" ]; then
    printf '  zap bins     : (cd %q && zapbirds -zap -zapfile %q -baryv %q -- %q)\n' \
        "$OUTDIR" "$ZAPFILE" "$BARYV" "${DM_STEMS[0]}_red.fft"
fi
for stem in "${DM_STEMS[@]}"; do
    target="$OUTDIR/$stem.fft"
    if [ "$DO_REDNOISE" = 1 ]; then
        target="$OUTDIR/${stem}_red.fft"
    fi
    printf '  search       : julia --project=%q -t auto %q %q\n' \
        "$REPO" "$REPO/bin/coherent_search.jl" "$target"
done
cat <<'EOF'

Zap AFTER rednoise: rednoise normalizes the raw spectrum, and zapbirds' default
noise replacement preserves the unit-mean power the analytic sigma assumes.
Do not use zapbirds -constamp.  If you zero bins by hand instead, add
--sigma measured to the search command: the closed form assumes every bin
still carries noise power.
EOF
