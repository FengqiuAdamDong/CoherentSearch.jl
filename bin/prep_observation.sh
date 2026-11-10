#!/usr/bin/env bash
#
# bin/prep_observation.sh -- on-disk PRESTO preprocessing for CoherentSearch.jl
#
#   rfifind  ->  prepsubband  ->  realfft  [-> rednoise]  [-> zapbirds]
#
# One .fft per DM.  Stops after realfft by default so the FFT can be inspected;
# --rednoise also writes <stem>_red.fft, and --zapfile zaps birdies in it.
#
# ORDER MATTERS: zap the _red.fft, never the raw .fft.
#   * rednoise normalizes each bin by a running median of the RAW spectrum;
#     zapping first would make it normalize against holes.
#   * zapbirds (PRESTO >= 6) replaces zapped bins with noise, preserving the
#     unit-mean power the search's analytic sigma assumes.  Do NOT pass
#     -constamp; if you zero bins by hand, search with --sigma measured.
#   * the raw .fft is never modified, so _red.fft is always regenerated from it.
#
# Every step is skipped when its outputs exist, and rerun (with everything
# after it) when they do not; --force reruns everything.  This is the on-disk
# counterpart to `coherent_search.jl --demod-grid` (demod/README.md section 11).
#
# Tools come from $PATH, then /home/fadong/anaconda3/bin; override with
# RFIFIND, PREPSUBBAND, REALFFT, REDNOISE, ZAPBIRDS.  Barycentering needs
# TEMPO/tempo2 (or pass --no-bary).

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] RAWFILE [RAWFILE ...]

  -o, --outdir DIR          output directory (\$PWD/prep_<NAME>)
  -n, --name NAME           observation stem (first input's basename without
                            its suffix); required for several raw files
  -d, --dm LIST             comma/space-separated dispersion measures
      --dmprec P            decimals in the DM filename field (2)
      --no-bary             do not barycenter (prepsubband -nobary)
      --mask FILE           use this rfifind mask and skip rfifind
      --no-mask             skip rfifind and run prepsubband unmasked
      --rfifind-blocks N    rfifind -blocks (4)
      --rfifind-time S      rfifind -time S instead of -blocks
      --ncpus N             -ncpus for rfifind/prepsubband (1)
      --rfifind-args S      extra rfifind arguments (whitespace-split)
      --prepsubband-args S  extra prepsubband arguments, e.g. "-nsub 128"
      --rednoise            also write <stem>_red.{fft,inf}
      --zapfile FILE        zap these birdies in <stem>_red.fft (implies --rednoise)
      --baryv V             zapbirds -baryv (0)
      --force               rerun every step
      --dry-run             print the commands only
  -h, --help                show this help

Outputs (in OUTDIR), with <stem> = <NAME>_DM<dm>:
  <NAME>_rfifind.{mask,stats,inf}   <stem>.{dat,inf,fft}
  <stem>_red.{fft,inf}              <stem>_red.zapped (sha256 of the zaplist)

The zaplist is PRESTO's format: one "frequency_Hz width_Hz" pair per line.
EOF
}

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
OUTDIR="" NAME="" DMS="" DMPREC=2 NOBARY=0 MASK="" NO_MASK=0
RFI_INT=(-blocks 4) NCPUS=1 RFI_ARGS=() PREP_ARGS=()
DO_REDNOISE=0 ZAPFILE="" BARYV=0 FORCE=0 DRY_RUN=0 INPUTS=()

while [ "$#" -gt 0 ]; do
    case "$1" in --*=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;; esac
    v=${2:-}
    case "$1" in
        -h|--help)          usage; exit 0 ;;
        -o|--outdir)        OUTDIR=$v ;;
        -n|--name)          NAME=$v ;;
        -d|--dm)            DMS=$v ;;
        --dmprec)           DMPREC=$v ;;
        --mask)             MASK=$v ;;
        --rfifind-blocks)   RFI_INT=(-blocks "$v") ;;
        --rfifind-time)     RFI_INT=(-time "$v") ;;
        --ncpus)            NCPUS=$v ;;
        --rfifind-args)     read -r -a RFI_ARGS <<< "$v" ;;
        --prepsubband-args) read -r -a PREP_ARGS <<< "$v" ;;
        --zapfile)          ZAPFILE=$v; DO_REDNOISE=1 ;;
        --baryv)            BARYV=$v ;;
        --no-bary)          NOBARY=1;      shift; continue ;;
        --no-mask)          NO_MASK=1;     shift; continue ;;
        --rednoise)         DO_REDNOISE=1; shift; continue ;;
        --force)            FORCE=1;       shift; continue ;;
        --dry-run)          DRY_RUN=1;     shift; continue ;;
        --)                 INPUTS+=("${@:2}"); break ;;
        -*)                 die "unknown option: $1 (try --help)" ;;
        *)                  INPUTS+=("$1"); shift; continue ;;
    esac
    [ -n "$v" ] || die "$1 needs a value"
    shift 2
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
posint() { [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
isnum()  { [[ $1 =~ ^-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$ ]]; }

[ "${#INPUTS[@]}" -gt 0 ] || die "no input files (try --help)"
[ -n "$DMS" ] || die "--dm is required"
[ -z "$MASK" ] || [ "$NO_MASK" = 0 ] || die "--mask and --no-mask are mutually exclusive"
[[ $DMPREC =~ ^[0-6]$ ]] || die "--dmprec must be between 0 and 6"
posint --ncpus "$NCPUS"
isnum "${RFI_INT[1]}" || die "--rfifind-blocks/--rfifind-time must be a number"
isnum "$BARYV" || die "--baryv must be a number"

for f in "${INPUTS[@]}"; do
    case "$(basename "$f")" in
        *.fft|*.dat|*.inf|*.mask|*.stats|*.cohout)
            die "'$f' is already processed; this script starts from raw data" ;;
        *.*) ;;
        *) die "'$f' has no suffix; rfifind/prepsubband require one" ;;
    esac
done

if [ -z "$NAME" ]; then
    [ "${#INPUTS[@]}" -eq 1 ] || die "several raw files: pass --name (they are one observation)"
    NAME=$(basename "${INPUTS[0]%.*}")
fi
case "$NAME" in
    */*)   die "--name must be a basename, not a path" ;;
    *_red) die "--name must not end in _red (reserved for the dereddened FFT)" ;;
esac

# Absolute paths, since every step runs inside OUTDIR.
abs() { realpath -e -- "$1" || die "not found: $1"; }
for i in "${!INPUTS[@]}"; do INPUTS[$i]=$(abs "${INPUTS[$i]}"); done
[ -z "$MASK" ] || MASK=$(abs "$MASK")
if [ -n "$ZAPFILE" ]; then
    ZAPFILE=$(abs "$ZAPFILE")
    [ -s "$ZAPFILE" ] || die "zaplist is empty: $ZAPFILE"
fi

OUTDIR=$(realpath -m -- "${OUTDIR:-$PWD/prep_$NAME}")
LOGDIR="$OUTDIR/logs"
[ "$DRY_RUN" = 1 ] || mkdir -p "$LOGDIR"

# DM list -> one file stem per DM.
DMS_OK=() STEMS=()
IFS=', ' read -r -a toks <<< "$DMS"
for dm in "${toks[@]}"; do
    isnum "$dm" && [[ $dm != -* ]] || die "invalid DM: '$dm'"
    stem="${NAME}_DM$(printf "%.${DMPREC}f" "$dm")"
    [[ " ${STEMS[*]} " != *" $stem "* ]] || die "two DMs give the same stem $stem at --dmprec $DMPREC"
    DMS_OK+=("$dm") STEMS+=("$stem")
done
[ "${#STEMS[@]}" -gt 0 ] || die "--dm listed no DMs"

# ---------------------------------------------------------------------------
# Tools and helpers
# ---------------------------------------------------------------------------
# Assign the result (X=$(tool x)) so a failure stops the script.
tool() {  # $1 = tool name; $UPPERCASE_NAME overrides
    local var=${1^^}
    local t=${!var:-$(command -v "$1" || echo "/home/fadong/anaconda3/bin/$1")}
    command -v "$t" >/dev/null || die "$1 not found (set $var=/path/to/$1)"
    echo "$t"
}

# Run "$@" in OUTDIR, logging to logs/$name.log (or just print it on --dry-run).
run() {
    local log="$LOGDIR/$1.log"; shift
    if [ "$DRY_RUN" = 1 ]; then
        printf '   (cd %q &&' "$OUTDIR"; printf ' %q' "$@"; printf ')\n'
        return
    fi
    { printf '+'; printf ' %q' "$@"; printf '\n'; } > "$log"
    ( cd "$OUTDIR" && "$@" ) >> "$log" 2>&1 \
        || { tail -n 20 "$log" >&2; die "$(basename "$1") failed (log: $log)"; }
}

# Do the files all exist (and is nothing upstream of them being rebuilt)?
STALE=$FORCE
have() {
    [ "$STALE" = 0 ] || return 1
    local f; for f; do [ -s "$OUTDIR/$f" ] || return 1; done
}

# After a step: its outputs must exist; everything downstream reruns.
wrote() {
    STALE=1
    [ "$DRY_RUN" = 0 ] || return 0
    local f; for f; do [ -s "$OUTDIR/$f" ] || die "missing $f after the step that writes it"; done
}

sz() { stat -c %s "$OUTDIR/$1"; }

# ---------------------------------------------------------------------------
# 1. rfifind
# ---------------------------------------------------------------------------
if [ -n "$MASK" ]; then
    note "rfifind: skip (using $MASK)"
elif [ "$NO_MASK" = 1 ]; then
    note "rfifind: skip (--no-mask)"
else
    rfi=${NAME}_rfifind
    if have "$rfi".{mask,stats,inf}; then
        note "rfifind: skip (exists) $rfi.mask"
    else
        note "rfifind: $NAME"
        RFIFIND=$(tool rfifind)
        run rfifind "$RFIFIND" -ncpus "$NCPUS" "${RFI_INT[@]}" \
            -o "$NAME" "${RFI_ARGS[@]}" -- "${INPUTS[@]}"
        wrote "$rfi".{mask,stats,inf}
    fi
    MASK="$OUTDIR/$rfi.mask"
fi

PREPSUBBAND=$(tool prepsubband)
PREP=("$PREPSUBBAND" -ncpus "$NCPUS" -dmstep 1 -numdms 1 -dmprec "$DMPREC")
[ -z "$MASK" ]       || PREP+=(-mask "$MASK")
[ "$NOBARY" = 0 ]    || PREP+=(-nobary)
REALFFT=$(tool realfft)
[ "$DO_REDNOISE" = 0 ] || REDNOISE=$(tool rednoise)
[ -z "$ZAPFILE" ]      || { ZAPBIRDS=$(tool zapbirds); ZAPSHA=$(sha256sum "$ZAPFILE" | cut -d' ' -f1); }

# ---------------------------------------------------------------------------
# 2-5. per DM: prepsubband -> realfft [-> rednoise -> zapbirds]
# ---------------------------------------------------------------------------
RFI_STALE=$STALE
for i in "${!STEMS[@]}"; do
    dm=${DMS_OK[$i]} stem=${STEMS[$i]} red=${STEMS[$i]}_red
    STALE=$RFI_STALE

    if have "$stem".{dat,inf}; then
        note "prepsubband: skip (exists) $stem.dat"
    else
        note "prepsubband: DM $dm -> $stem"
        run "prepsubband_$stem" "${PREP[@]}" -lodm "$dm" "${PREP_ARGS[@]}" \
            -o "$NAME" -- "${INPUTS[@]}"
        wrote "$stem".{dat,inf}
    fi
    if [ "$DRY_RUN" = 0 ] && [ "$NOBARY" = 0 ]; then
        grep -qE 'Barycentered\?.*= *1' "$OUTDIR/$stem.inf" \
            || die "$stem.inf is not barycentered; install TEMPO/tempo2 or pass --no-bary"
    fi

    if have "$stem.fft"; then
        note "realfft: skip (exists) $stem.fft"
    else
        note "realfft: $stem"
        run "realfft_$stem" "$REALFFT" -- "$stem.dat"
        wrote "$stem.fft"
    fi
    # A truncated file from an interrupted run is the failure worth catching.
    if [ "$DRY_RUN" = 0 ] && [ "$(sz "$stem.dat")" -ne "$(sz "$stem.fft")" ]; then
        die "$stem.dat and $stem.fft differ in size; rerun with --force"
    fi

    [ "$DO_REDNOISE" = 1 ] || continue
    # _red.fft is rednoise of the raw .fft, then zapped with the list whose
    # hash is in the marker (none = unzapped).  Rebuild it unless that matches.
    marker="$OUTDIR/$red.zapped"
    zapped=$(cut -d' ' -f1 "$marker" 2>/dev/null || true)
    if have "$red".{fft,inf} && [ "$zapped" = "${ZAPSHA:-}" ]; then
        note "rednoise${ZAPFILE:+/zapbirds}: skip (exists) $red.fft"
        continue
    fi
    note "rednoise: $stem"
    [ "$DRY_RUN" = 1 ] || rm -f "$marker"
    run "rednoise_$stem" "$REDNOISE" -- "$stem.fft"
    wrote "$red".{fft,inf}
    if [ -n "$ZAPFILE" ]; then
        note "zapbirds: $red"
        run "zapbirds_$stem" "$ZAPBIRDS" -zap -zapfile "$ZAPFILE" -baryv "$BARYV" -- "$red.fft"
        [ "$DRY_RUN" = 1 ] || echo "$ZAPSHA  $ZAPFILE" > "$marker"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
note "$([ "$DRY_RUN" = 1 ] && echo "dry run: nothing written" || echo "done"): $OUTDIR"
echo "next steps:"
printf '  inspect : explorefft %q\n' "$OUTDIR/${STEMS[0]}.fft"
[ "$DO_REDNOISE" = 1 ] || echo "  then rerun with --rednoise [--zapfile FILE]"
for stem in "${STEMS[@]}"; do
    [ "$DO_REDNOISE" = 0 ] || stem=${stem}_red
    printf '  search  : julia --project=%q -t auto %q %q\n' \
        "$REPO" "$REPO/bin/coherent_search.jl" "$OUTDIR/$stem.fft"
done
