#!/usr/bin/env python3
"""Emit ajs-predicted known-pulsar mask centres for `fold_demod.jl`.

This is a deliberately thin, machine-readable wrapper around
`plot_corner_ajs.py`: the fold script and the corner plot therefore use the
*same* par parser, observation reader and ajs residual-phase prediction.  The
prediction needs the DD/BT orbit model from `pulsar_optimal_ajs_circular.py`
(and hence the sibling FFA_stacking checkout), so it lives on the Python side
rather than being reimplemented in Julia.

Inputs
------
`--ajs CSV` has one trial point per line, `accel,jerk,snap` (an `a,j,s` or
`accel,jerk,snap` header is ignored).  `--par` is one or more .par files,
directories, or globs.  `--inf` is the observation .inf.

Output
------
CSV on stdout:

    par_index,ajs_index,f_pred_hz,par_name,f0_hz,half_width_hz

`par_index` and `ajs_index` are zero-based positions in the order given.
`f_pred_hz` is the frequency the coherent search would report for that known
pulsar at that trial ajs point (Section 5.6's absorbed linear phase).  The
half-width is the same one `plot_corner_ajs.py` uses; Julia applies the
harmonic scaling `h*half` when masking.

Usage
-----
    python predict_par_masks.py --inf OBS.inf --ajs points.csv \
        --par par_files/ [--par-tol HZ] [--par-frac-tol FRAC]
"""

import argparse
import os
import sys

import numpy as np

# Same directory: `plot_corner_ajs` imports pulsar_optimal_ajs_circular, which
# owns the DD model and the search's velocity convention.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plot_corner_ajs as PC  # noqa: E402


def read_ajs(path):
    """Read `accel,jerk,snap` rows, ignoring blank lines and a header."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 3:
                raise SystemExit(f"{path}: expected 3 columns, got {line!r}")
            if fields[0].lower() in ("a", "accel", "acceleration"):
                continue
            rows.append([float(fields[0]), float(fields[1]), float(fields[2])])
    if not rows:
        raise SystemExit(f"{path}: no ajs points")
    return np.asarray(rows, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ajs-predicted known-pulsar mask centres for fold_demod.jl.")
    ap.add_argument("--par", nargs="+", action="extend", required=True,
                    metavar="PATH",
                    help=".par file(s), directory(ies), or glob(s)")
    ap.add_argument("--inf", required=True,
                    help="observation .inf (must be barycentered for the "
                         "prediction to be meaningful)")
    ap.add_argument("--ajs", required=True,
                    help="CSV of trial accel,jerk,snap points")
    ap.add_argument("--par-tol", type=float, default=None, metavar="HZ",
                    help="absolute frequency half-width (default: "
                         "max(0.005, 5e-4*F0))")
    ap.add_argument("--par-frac-tol", type=float, default=0.0, metavar="FRAC",
                    help="fractional-F0 half-width (combined with --par-tol "
                         "using the larger value)")
    args = ap.parse_args(argv)

    pars = PC.gather_pars(args.par)
    if not pars:
        raise SystemExit("no usable par files found")

    inf = PC.read_inf(args.inf)
    _, _, _, _, bary = inf
    if not bary:
        print("warning: .inf says the data are NOT barycentered; the "
              "par-based prediction assumes barycentric frequencies",
              file=sys.stderr)

    pts = read_ajs(args.ajs)
    # predict_ajs_frequencies only reads columns 2:5 (the ajs triple).
    data = np.column_stack([np.zeros(len(pts)), np.zeros(len(pts)), pts])
    uniq, inv, preds = PC.predict_ajs_frequencies(data, pars, inf)

    print("par_index,ajs_index,f_pred_hz,par_name,f0_hz,half_width_hz")
    for pi, (psr, _path, f0, _par) in enumerate(pars):
        half = PC.par_halfwidth(f0, args.par_tol, args.par_frac_tol)
        for ii in range(len(pts)):
            gi = int(inv[ii])
            f_pred = float(preds[(pi, gi)])
            print(f"{pi},{ii},{f_pred:.12g},{psr},{f0:.12g},{half:.12g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
