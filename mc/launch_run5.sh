#!/bin/bash
# Run 5 of the detection-efficiency Monte Carlo (2026-09-30): two patch passes
# answering Vincent Morello's comments on the draft.
#
#   white  (fitzroy)  rseek configuration C -- 20..120 bins below 7.2 ms,
#                     120..140 above -- on EXACTLY run 4's 22,368 white
#                     realisations, so it pairs with rseek_W and has the same
#                     per-band tails.  Patches go INTO run 2's directory.
#   red    (eiger)    rseek_A, rseek_B (1 in 10) and rseek_W on run 3's red
#                     realisations with runner-up hits stored: the whitened-
#                     riptide measurement the paper and the email lacked, and a
#                     measurement in place of the red displaced-hit bound.
#                     Patches go INTO run 3's directory.
#
# --outdir MUST be the directory being patched: mc_analyze keys records by
# (directory, index), and a patch in its own directory is dropped silently.
# Run under screen; a login shell going away killed run 3's first launch.
#   screen -dmS run5 bash mc/launch_run5.sh white    # on fitzroy
#   screen -dmS run5 bash mc/launch_run5.sh red      # on eiger
set -euo pipefail

PIXI=/data1/environments/pixiPSR/.pixi/envs/default/bin
REPO=/data1/git/CoherentSearch.jl
TPA=${TPA:-/data1/mc/table_1.csv}

# See launch_eiger.sh: libpresto.so and liberfa must be on the loader path.
export LD_LIBRARY_PATH="$PIXI/../lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
$PIXI/python -c 'from presto import sifting' || {
    echo "presto.sifting will not import -- fix LD_LIBRARY_PATH before running" >&2
    exit 1
}

cd "$REPO"
common=(--fa-top 4000 --ncands 4000 --sigma-every 0 --hits-per-inj 8
        --presto-bin $PIXI --rseek $PIXI/rseek --tpa "$TPA")

case "${1:-}" in
white)
    OUT=/data1/mc/run2
    $PIXI/python mc/mc_simulate.py --outdir $OUT --workers ${NWORK:-19} \
        --arms rseekc --indices-from $OUT \
        --only-indices-in "$OUT/mcpatch_accel+coherent+rseek+rseekw_eiger_*.jsonl" \
        "${common[@]}" 2>&1 | tee -a $OUT/launch_run5.log ;;
red)
    OUT=/data1/mc/run3
    $PIXI/python mc/mc_simulate.py --outdir $OUT --workers ${NWORK:-15} \
        --arms rseek,rseekw --indices-from $OUT --deep-every 10 \
        --rednoise-knee 0.1 50 \
        "${common[@]}" 2>&1 | tee -a $OUT/launch_run5.log ;;
*)
    echo "usage: $0 white|red" >&2; exit 2 ;;
esac
