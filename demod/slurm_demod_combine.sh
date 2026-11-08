#!/usr/bin/env bash
#
# Combine every batch's .cohout files after the array has finished.
# Submitted by demod/slurm_demod_sweep.sh with --dependency=afterok:<array>.
#
# Usage (normally via sbatch from the submitter):
#   sbatch ... slurm_demod_combine.sh /path/to/slurm_job.env

set -euo pipefail

ENVFILE="${1:?usage: slurm_demod_combine.sh ENVFILE}"
# shellcheck disable=SC1090
source "$ENVFILE"

if [ -n "${MODULE_LOAD:-}" ]; then eval "$MODULE_LOAD"; fi
cd "$OUTDIR"

# Count with find rather than a bash glob: a real sweep can have more .cohout
# files than ARG_MAX, and combine_cohout.py globs internally anyway.
found=$(find "$OUTDIR" -maxdepth 1 -type f -name "${INBASE}_demod_*_red.cohout" | wc -l)
echo ">> found $found .cohout file(s); expected $TOTAL_POINTS"
if [ "$found" -ne "$TOTAL_POINTS" ]; then
    echo "WARNING: expected $TOTAL_POINTS candidate file(s), found $found." >&2
    echo "WARNING: an array element may have failed, or OUTDIR has files from another run." >&2
fi
if [ "$found" -eq 0 ]; then
    echo "ERROR: no '${INBASE}_demod_*_red.cohout' files in $OUTDIR" >&2
    exit 1
fi

"$PYTHON" "$COH_REPO/demod/combine_cohout.py" \
    "$OUTDIR/${INBASE}_demod_*_red.cohout" \
    -o "$OUTDIR/combined_cohout.txt"

echo ">> wrote $OUTDIR/combined_cohout.txt"
