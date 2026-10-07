#!/usr/bin/env python3
"""Combine per-point NS-NS sweep .cohout files into one candidate list.

Each input file is either <stem>_demod_a{A}_j{J}_s{S}_red.cohout (ajs grid) or
<stem>_demod_pb{Pb}_x{X}_at{A_T}_red.cohout (circular grid), written by
run_nsns_sweep.sh.  Every file carries the same one-line header and zero or
more candidate rows.  This appends the grid point's parameters, taken from the
filename, to each row and writes every candidate from every file to a single
output, sorted by S/N descending.

A hybrid sweep (run_nsns_sweep.sh MODE=hybrid) emits both filename patterns, so
one glob picks up both models; both are handled in a single call.

Optionally, with --plot, it also draws the three pairwise projections of the
grid (marker colour scaling with S/N).  If --truth names a truth YAML (e.g.
demod_out/inj_truth.yaml), a crosshair marks the injected parameters.  The
three columns are the model's axes: (a, j, s) for ajs, (pb, x, at) for circular.
Both models in one call is supported.

Usage:
    python combine_cohout.py 'nsns_sweep_inj/*.cohout' -o all_candidates.txt
    python combine_cohout.py 'nsns_sweep_inj/*.cohout' -o all.txt \
        --plot all.png --truth demod_out/inj_truth.yaml
"""
import argparse
import glob
import re
import sys

# Per-model: (filename regex, extra output header, axis keys, plot pairs,
# truth-YAML key per axis).  Matched in order.
MODELS = (
    dict(
        name="ajs",
        fname=re.compile(
            r"_demod_a(?P<a>[^_]+)_j(?P<j>[^_]+)_s(?P<s>[^_]+)_red\.cohout$"),
        keys=("a", "j", "s"),
        header="    Accel (m/s^2)    Jerk (m/s^3)    Snap (m/s^4)",
        labels={"a": "Accel (m/s^2)", "j": "Jerk (m/s^3)", "s": "Snap (m/s^4)"},
        truth={"a": "best_fit_accel", "j": "best_fit_jerk", "s": "best_fit_snap"},
    ),
    dict(
        name="circular",
        fname=re.compile(
            r"_demod_pb(?P<pb>[^_]+)_x(?P<x>[^_]+)_at(?P<at>[^_]+)_red\.cohout$"),
        keys=("pb", "x", "at"),
        header="    Pb (d)    x (lt-s)    A_T (rad)",
        labels={"pb": "Pb (d)", "x": "x (lt-s)", "at": "A_T (rad)"},
        truth={"pb": "pb_days", "x": "x_lt_s", "at": "A_T_rad"},
    ),
)

PAIRS = (("0", "1"), ("0", "2"), ("1", "2"))  # positions into model["keys"]


def parse_row(line):
    """Return (snr, original_line) for a candidate row, or None for non-data."""
    fields = line.split()
    if len(fields) < 2:
        return None
    try:
        snr = float(fields[1])
    except ValueError:
        return None
    return snr, line.rstrip("\n")


def read_truth(path, model):
    """Return {axis_key: value} from a flat truth YAML, using the model's
    truth-key mapping (best_fit_* for ajs, pb_days/x_lt_s/A_T_rad for circular)."""
    values = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0]
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            for short, long in model["truth"].items():
                if key == long:
                    try:
                        values[short] = float(val.strip())
                    except ValueError:
                        pass
    missing = [model["truth"][k] for k in model["keys"] if k not in values]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)} "
                         f"for {model['name']} model")
    return values


def make_plot(rows, model, truth, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = model["keys"]
    labels = model["labels"]
    snrs = [r[0] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xi, yi) in zip(axes, PAIRS):
        xk, yk = keys[int(xi)], keys[int(yi)]
        sc = ax.scatter([r[2][xk] for r in rows], [r[2][yk] for r in rows],
                        c=snrs, cmap="viridis", s=60, alpha=0.6,
                        edgecolors="none")
        if truth is not None:
            ax.axvline(truth[xk], color="red", ls="--", lw=1.0)
            ax.axhline(truth[yk], color="red", ls="--", lw=1.0)
        ax.set_xlabel(labels[xk])
        ax.set_ylabel(labels[yk])
        fig.colorbar(sc, ax=ax, label="S/N")
    title = f"{len(rows)} candidate(s); colour ~ S/N"
    if truth is not None:
        title += "; truth " + " ".join(f"{k}={truth[k]:.6g}" for k in keys)
    fig.suptitle(f"{model['name']} grid: " + title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="paths or globs (quote globs so this script expands them)")
    ap.add_argument("-o", "--output", default="combined_cohout.txt",
                    help="output file (default: combined_cohout.txt)")
    ap.add_argument("--plot", metavar="PNG",
                    help="write the three pairwise a/j/s scatter panels here")
    ap.add_argument("--truth", metavar="YAML",
                    help="truth YAML with best_fit_accel/jerk/snap, for crosshairs")
    args = ap.parse_args(argv)

    if args.truth and not args.plot:
        print("warning: --truth has no effect without --plot", file=sys.stderr)

    paths = []
    for pattern in args.inputs:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"warning: no files match {pattern!r}", file=sys.stderr)
        paths.extend(hits)
    if not paths:
        print("error: no input files", file=sys.stderr)
        return 1

    # Each path is classified by which model regex matches; the header line
    # differs between models, so keep a header per model.
    header_of = {}
    rows = []
    skipped = 0
    model_of_row = None  # model of the rows list, for --plot; mixed not plotted
    mixed = False
    for path in paths:
        model = next((m for m in MODELS if m["fname"].search(path)), None)
        if model is None:
            print(f"warning: skipping {path!r} (unrecognised name)", file=sys.stderr)
            skipped += 1
            continue
        m = model["fname"].search(path)
        coords = {k: float(m.group(k)) for k in model["keys"]}
        with open(path) as fh:
            lines = fh.read().splitlines()
        if not lines:
            skipped += 1
            continue
        header_of.setdefault(model["name"], lines[0])
        if model_of_row is None:
            model_of_row = model
        elif model_of_row is not model:
            mixed = True
        for line in lines[1:]:
            parsed = parse_row(line)
            if parsed is None:
                continue
            snr, text = parsed
            suffix = "    ".join(m.group(k) for k in model["keys"])
            rows.append((snr, f"{text}    {suffix}", coords, model["name"]))

    rows.sort(key=lambda r: r[0], reverse=True)
    # One model: emit its header.  Mixed: group by model, each with its header.
    with open(args.output, "w") as out:
        if not mixed and model_of_row is not None:
            out.write(header_of[model_of_row["name"]] + model_of_row["header"] + "\n")
        else:
            for m in MODELS:
                if m["name"] in header_of:
                    out.write(f"# {m['name']}\n")
                    out.write(header_of[m["name"]] + m["header"] + "\n")
        for _, text, _, _ in rows:
            out.write(text + "\n")

    if args.plot:
        if not rows:
            print("warning: no candidates, not plotting", file=sys.stderr)
        elif mixed:
            print("warning: --plot skipped (mixed-model candidate files)",
                  file=sys.stderr)
        else:
            make_plot(rows, model_of_row,
                      read_truth(args.truth, model_of_row) if args.truth else None,
                      args.plot)
            print(f"plot -> {args.plot}", file=sys.stderr)

    print(f"{len(rows)} candidate(s) from {len(paths) - skipped} file(s) "
          f"-> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
