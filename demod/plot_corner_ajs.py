#!/usr/bin/env python3
"""Corner plot of every candidate in a set of per-grid-point .cohout files.

Each input file is named

    <stem>_demod_a{A}_j{J}_s{S}_red.cohout

and contains the candidates found at that single (accel, jerk, snap) grid
point.  Every candidate row contributes one point:

    x/y/z = the grid point's accel, jerk, snap (from the filename)
    frequency = column 3 of the row
    marker area = S/N (column 2), scaled between the 1st and 99th percentile
                  of the sample so the narrow range is still visible

The result is a 4-parameter corner plot: histograms on the diagonal and
lower-triangle scatter panels for every pair.  Because accel/jerk/snap are
shared by all candidates from one file, the non-frequency projections show
one marker per candidate stacked on its grid point.

Candidates below an S/N threshold can be dropped before plotting with
--snr-min (default: no cut).

Known pulsars can be masked out by passing TEMPO/TEMPO2 .par files (or a
directory/glob of them) with --par.  Candidates within a search band around
each par's F0 are dropped before plotting.  The default half-width is the same
one run_ajs_pipeline.sh uses, max(0.005 Hz, 5e-4*F0); override it with
--par-tol (absolute Hz) and/or --par-frac-tol (fraction of F0).  Harmonics up
to --par-harmonics N are masked as well (default: fundamental only).

If the observation .inf is also supplied with --inf, the mask frequency is
predicted separately for every ajs grid point: the par gives the true orbit and
spin, the ajs gives the trial demodulation, and the residual phase's linear
term is the frequency the search would report.  The constant-velocity term is
kept because the spin search absorbs it as a frequency offset.  Without --inf
the mask stays centred on the par's F0 at PEPOCH.  --inf must describe a
barycentered observation; if the .inf says otherwise the prediction is only
approximate because the candidates then include the Earth's Doppler shift.

Usage:
    python plot_corner_ajs.py OUT/ -o corner_ajs.png
    python plot_corner_ajs.py OUT/ --snr-min 8 -o corner_ajs.png
    python plot_corner_ajs.py OUT/ --par par_files/ -o corner_ajs.png
    python plot_corner_ajs.py OUT/ --par par_files/ --inf OBS.inf -o corner.png
    python plot_corner_ajs.py 'OUT/*.cohout' -o corner_ajs.png --linear-freq

The figure is written to --output and then shown with plt.show().  Pass
--no-show for batch/pipeline use (this also forces the Agg backend).
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
# Force a headless backend only when the window is explicitly disabled;
# otherwise let matplotlib pick an interactive backend (it falls back to Agg
# by itself when none is available, e.g. over SSH without DISPLAY).
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

try:
    # Same-directory sibling: owns the DD/BT orbit model and the search's
    # velocity convention.  Only needed for --inf (ajs-aware prediction).
    import pulsar_optimal_ajs_circular as PAJS
except Exception:  # pragma: no cover - reported only if --inf is used
    PAJS = None

FNAME_RE = re.compile(
    r"_demod_a(?P<a>[^_]+)_j(?P<j>[^_]+)_s(?P<s>[^_]+)_red\.cohout$"
)

# Plot order requested by the user: accel, jerk, snap, frequency.
PARAMS = (
    dict(key="a", label=r"Accel (m s$^{-2}$)"),
    dict(key="j", label=r"Jerk (m s$^{-3}$)"),
    dict(key="s", label=r"Snap (m s$^{-4}$)"),
    dict(key="f", label="Frequency (Hz)"),
)

POINT_COLOUR = "#1f4e79"
HIST_COLOUR = "#2b6ca3"


def parse_row(line):
    """Return (snr, frequency) for a candidate row, or None for a header/blank."""
    fields = line.split()
    if len(fields) < 3:
        return None
    try:
        snr = float(fields[1])
        freq = float(fields[2])
    except ValueError:
        return None
    return snr, freq


def collect(paths):
    """Return (n, 5) array of (snr, freq, accel, jerk, snap) and the number of
    files that actually contributed candidates."""
    rows = []
    used_files = 0
    skipped = 0
    for path in sorted(paths):
        m = FNAME_RE.search(path)
        if m is None:
            print(f"warning: skipping {path!r} (unrecognised filename)",
                  file=sys.stderr)
            skipped += 1
            continue
        accel = float(m.group("a"))
        jerk = float(m.group("j"))
        snap = float(m.group("s"))
        file_rows = 0
        with open(path) as fh:
            for line in fh:
                parsed = parse_row(line)
                if parsed is None:
                    continue
                snr, freq = parsed
                rows.append((snr, freq, accel, jerk, snap))
                file_rows += 1
        if file_rows:
            used_files += 1
        else:
            skipped += 1
    return np.asarray(rows, dtype=float), used_files, skipped


def read_par_dict(path):
    """First value of every TEMPO/TEMPO2 key, with Fortran D exponents fixed."""
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            key = fields[0].upper()
            # Fix only Fortran exponent D/d, not letters in names like DE440.
            value = re.sub(r"(?<=\d)[Dd](?=[+-]?\d+$)", "E", fields[1])
            out.setdefault(key, value)
    return out


def par_f0(par):
    """Spin frequency at PEPOCH in Hz, from F0 or P0."""
    if "F0" in par:
        try:
            return float(par["F0"])
        except ValueError:
            return None
    if "P0" in par:
        try:
            p0 = float(par["P0"])
        except ValueError:
            return None
        return 1.0 / p0 if p0 > 0 else None
    return None


def par_frequency_at(par, mjd):
    """Intrinsic spin frequency at MJD from F0/F1/F2... (or P0/P1)."""
    f0 = par_f0(par)
    if f0 is None:
        return None
    try:
        pepoch = float(par.get("PEPOCH", par.get("POSEPOCH", mjd)))
    except ValueError:
        pepoch = mjd
    dt = (mjd - pepoch) * 86400.0

    f1 = None
    if "F1" in par:
        try:
            f1 = float(par["F1"])
        except ValueError:
            pass
    elif "P1" in par:
        try:
            f1 = -float(par["P1"]) / (1.0 / f0) ** 2
        except ValueError:
            pass
    f = f0
    if f1 is not None:
        f += f1 * dt
    fact = 2.0
    for order in range(2, 10):
        key = f"F{order}"
        if key not in par:
            break
        try:
            f += float(par[key]) * dt ** order / fact
        except ValueError:
            break
        fact *= order + 1
    return f


def read_inf(path):
    """Return (N, dt, epoch_mjd, t_obs_s, barycentered) from a PRESTO .inf."""
    vals = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            vals[key.strip()] = val.strip()

    def get(fragment):
        for key, val in vals.items():
            if fragment.lower() in key.lower():
                return val
        return None

    n = int(get("Number of bins").split()[0])
    dt = float(get("Width of each time series").split()[0])
    epoch = float(get("Epoch of observation").split()[0])
    bary = get("Barycentered")
    bary = bool(bary is not None and bary.split()[0].lstrip().startswith("1"))
    return n, dt, epoch, n * dt, bary


def gather_pars(patterns):
    """Expand par files/directories/globs into [(name, path, F0, par_dict), ...]."""
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            hits = sorted(glob.glob(os.path.join(pattern, "*.par")))
        else:
            hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"warning: no .par files match {pattern!r}", file=sys.stderr)
        paths.extend(hits)

    pars = []
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        par = read_par_dict(path)
        f0 = par_f0(par)
        if f0 is None or f0 <= 0:
            print(f"warning: skipping {path!r} (no usable F0/P0)", file=sys.stderr)
            continue
        psr = par.get("PSR") or par.get("PSRJ") or \
            os.path.splitext(os.path.basename(path))[0]
        pars.append((psr, path, f0, par))
    return pars


def par_halfwidth(f0, abs_tol, frac_tol):
    """Frequency half-width for masking one known pulsar.

    With neither override, reproduce run_ajs_pipeline.sh's search band.
    """
    if abs_tol is None and frac_tol <= 0:
        return max(0.005, 5e-4 * f0)
    return max(abs_tol or 0.0, frac_tol * f0)


def predict_ajs_frequencies(data, pars, inf):
    """Predict the searched frequency of each par at each ajs grid point.

    The par supplies the true orbit and spin.  For a trial ajs model the
    demodulated residual phase is

        phi(t) = -integral [v_true(t) - v_model(t)] dt / (c p0).

    The constant velocity term is retained: the FFT search absorbs it as a
    frequency offset, which is precisely the shift being predicted.  The
    search reports the least-squares linear slope of that phase, so

        f_pred = F(t_mid) + dphi/dt.

    Returns `(uniq_ajs, inverse_index, {(par_index, ajs_index): f_pred})`.
    """
    if PAJS is None:
        raise RuntimeError("ajs prediction needs demod/pulsar_optimal_ajs_circular.py")

    _, _, epoch, t_obs, _ = inf
    uniq, inv = np.unique(data[:, 2:5], axis=0, return_inverse=True)
    t = np.linspace(0.0, t_obs, 4001)
    tau = t - 0.5 * t_obs
    t0 = t - t.mean()
    denom = float(np.dot(t0, t0))
    dt = np.diff(t)
    preds = {}

    for pi, (psr, path, f0, par) in enumerate(pars):
        try:
            prm, p0, pdot, _ = PAJS.par_to_dd(par, epoch)
            v_true = PAJS.model_velocity(t, prm, p0, pdot)
        except Exception as exc:
            print(f"warning: {psr}: ajs prediction failed ({exc}); using F0",
                  file=sys.stderr)
            for gi in range(len(uniq)):
                preds[(pi, gi)] = f0
            continue

        f_true = par_frequency_at(par, epoch + 0.5 * t_obs / 86400.0)
        if f_true is None:
            f_true = f0
        for gi, (accel, jerk, snap) in enumerate(uniq):
            v_model = (accel * tau + jerk * tau ** 2 / 2.0
                       + snap * tau ** 3 / 6.0)
            dv = v_true - v_model
            phase = np.empty_like(t)
            phase[0] = 0.0
            phase[1:] = np.cumsum(0.5 * (dv[1:] + dv[:-1]) * dt)
            phase /= (PAJS.C * p0)
            # phase here is +integral(v_true-v_model); the actual demodulated
            # phase carries the opposite sign, so subtract its slope.
            slope = float(np.dot(t0, phase) / denom)
            preds[(pi, gi)] = f_true - slope
    return uniq, inv, preds


def mask_known(data, pars, abs_tol=None, frac_tol=0.0, max_harmonic=1,
               inf=None):
    """Drop candidates within tolerance of any known pulsar frequency.

    Without `inf` the centre is the par's F0 at PEPOCH.  With an observation
    `.inf`, the centre is predicted separately for every ajs grid point from
    the full par model and the demodulation residual phase.

    Returns (filtered_data, n_removed, per_par_counts, halfwidths,
    predicted_ranges).
    """
    freq = data[:, 1]
    keep = np.ones(len(freq), dtype=bool)
    counts = {}
    halfwidths = {}
    pred_ranges = {}
    max_harmonic = max(1, max_harmonic)

    if inf is None:
        for psr, path, f0, par in pars:
            half = par_halfwidth(f0, abs_tol, frac_tol)
            halfwidths[psr] = half
            hit = np.zeros(len(freq), dtype=bool)
            for h in range(1, max_harmonic + 1):
                # The uncertainty/band scales with the harmonic number.
                hit |= np.abs(freq - h * f0) <= h * half
            newly = hit & keep
            counts[psr] = int(newly.sum())
            keep &= ~hit
    else:
        uniq, inv, preds = predict_ajs_frequencies(data, pars, inf)
        for pi, (psr, path, f0, par) in enumerate(pars):
            half = par_halfwidth(f0, abs_tol, frac_tol)
            halfwidths[psr] = half
            count = 0
            values = []
            for gi in range(len(uniq)):
                f_pred = preds[(pi, gi)]
                values.append(f_pred)
                idx = np.flatnonzero(inv == gi)
                fg = freq[idx]
                hit = np.zeros(len(idx), dtype=bool)
                for h in range(1, max_harmonic + 1):
                    hit |= np.abs(fg - h * f_pred) <= h * half
                newly = idx[hit & keep[idx]]
                keep[newly] = False
                count += len(newly)
            counts[psr] = count
            pred_ranges[psr] = (min(values), max(values))

    removed = ~keep
    return (data[keep], int(removed.sum()), counts, halfwidths, pred_ranges)


def axis_limits(values, log=False):
    """Data limits with a little padding; a constant axis gets a fake span so
    the marker line/bar is visible."""
    if log:
        values = values[values > 0]
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            return lo / 1.5, hi * 1.5
        pad = 0.06 * (np.log10(hi) - np.log10(lo))
        return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        # Constant parameter: no real range, but keep the single value centred.
        return lo - 0.5, hi + 0.5
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def marker_sizes(snr, size_min, size_max):
    """Map S/N linearly onto marker area, using the 1-99 percentile range so a
    single loud candidate does not flatten every other marker."""
    lo, hi = np.percentile(snr, [1, 99])
    if hi <= lo:
        lo, hi = float(snr.min()), float(snr.max())
    if hi <= lo:
        return np.full_like(snr, 0.5 * (size_min + size_max))
    frac = np.clip((snr - lo) / (hi - lo), 0.0, 1.0)
    return size_min + frac * (size_max - size_min)


def size_for(snr_value, snr, size_min, size_max):
    """Marker area for a representative S/N, using the same mapping as the data."""
    return float(marker_sizes(np.asarray([snr_value], dtype=float),
                              size_min, size_max)[0])


def make_plot(data, nfiles, outpath, logfreq=False, dpi=200,
              size_min=2.0, size_max=70.0, alpha=0.30,
              npars=0, nremoved=0, ajs_predicted=False, snr_min=0.0,
              show=False):
    # Draw faint/low-S/N candidates first so the loudest marker at each grid
    # point is the one left visible in the non-frequency projections.
    data = data[np.argsort(data[:, 0])]
    snr = data[:, 0]
    freq = data[:, 1]
    accel = data[:, 2]
    jerk = data[:, 3]
    snap = data[:, 4]

    values = {
        "a": accel,
        "j": jerk,
        "s": snap,
        "f": freq,
    }
    constant = {k: np.ptp(v) == 0 for k, v in values.items()}
    logscale = {"f": bool(logfreq)}

    limits = {
        k: axis_limits(v, log=logscale.get(k, False))
        for k, v in values.items()
    }

    sizes = marker_sizes(snr, size_min, size_max)

    n = len(PARAMS)
    fig, axes = plt.subplots(n, n, figsize=(13.5, 13.5))

    for i, pi in enumerate(PARAMS):
        ki = pi["key"]
        for j, pj in enumerate(PARAMS):
            kj = pj["key"]
            ax = axes[i, j]

            if j > i:                       # upper triangle: unused
                ax.axis("off")
                continue

            if i == j:                      # diagonal: 1-D distribution
                v = values[ki]
                if constant[ki]:
                    ax.axvline(v[0], color=POINT_COLOUR, lw=2.5)
                    ax.text(0.5, 0.82,
                            f"fixed at {v[0]:.6g}\n({len(v)} candidates)",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=8, color="#444444")
                else:
                    bins = 45
                    if logscale.get(ki, False):
                        lo, hi = limits[ki]
                        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
                    ax.hist(v, bins=bins, histtype="stepfilled",
                            color=HIST_COLOUR, alpha=0.75, linewidth=0.8,
                            edgecolor=POINT_COLOUR)
                    if logscale.get(ki, False):
                        ax.set_xscale("log")
                ax.set_ylim(bottom=0)
                ax.tick_params(labelleft=False)
            else:                           # lower triangle: pair scatter
                x = values[kj]
                y = values[ki]
                if logscale.get(kj, False):
                    ax.set_xscale("log")
                if logscale.get(ki, False):
                    ax.set_yscale("log")
                ax.scatter(x, y, s=sizes, c=POINT_COLOUR, alpha=alpha,
                           edgecolors="none", rasterized=True)
                if constant[kj]:
                    ax.axvline(x[0], color="#d1495b", lw=1.2, ls="--")
                if constant[ki]:
                    ax.axhline(y[0], color="#d1495b", lw=1.2, ls="--")

            # Consistent limits for every panel in the same column/row.
            if logscale.get(kj, False):
                ax.set_xscale("log")
            if logscale.get(ki, False) and i != j:
                ax.set_yscale("log")
            if j <= i:
                ax.set_xlim(*limits[kj])
            if i != j:
                ax.set_ylim(*limits[ki])

            if i == n - 1:
                ax.set_xlabel(pj["label"])
            if j == 0 and i != 0:
                ax.set_ylabel(pi["label"])
            if j == 0 and i == 0:
                ax.set_ylabel("count")

            if constant[kj]:
                ax.set_xticks([values[kj][0]])
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.6g"))
            if i != j and constant[ki]:
                ax.set_yticks([values[ki][0]])
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.6g"))

    # Use the empty top-right panel for the S/N size legend and a summary.
    ax = axes[0, n - 1]
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    qs = np.array([snr.min(), np.median(snr), snr.max()])
    handles = [
        plt.scatter([], [], s=size_for(q, snr, size_min, size_max),
                    c=POINT_COLOUR, alpha=0.65, edgecolors="none",
                    label=f"{q:.2f}")
        for q in qs
    ]
    ax.legend(handles=handles, title="S/N", loc="upper center",
              bbox_to_anchor=(0.5, 1.0), frameon=False, labelspacing=1.5,
              handletextpad=1.2, borderpad=0.2, borderaxespad=0.0)
    fixed = [f"{p['key']} = {values[p['key']][0]:.6g}"
             for p in PARAMS if constant[p["key"]]]
    note = f"{len(snr):,} candidates\n{nfiles} grid file(s)"
    if snr_min > 0:
        note += f"\nS/N >= {snr_min:g}"
    if fixed:
        note += "\nconstant: " + ", ".join(fixed)
    if npars:
        note += f"\nremoved {nremoved:,} near {npars} par(s)"
        if ajs_predicted:
            note += "\n(ajs-predicted frequencies)"
    ax.text(0.5, 0.02, note, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#444444")

    title = "Candidate corner plot (marker area ~ S/N"
    if snr_min > 0:
        title += f", S/N >= {snr_min:g}"
    if npars:
        title += "; known-pulsar bands removed"
    title += ")"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        # Leave the figure open; the caller runs plt.show().
        return fig
    plt.close(fig)
    return None


def backend_is_interactive():
    """True when the current matplotlib backend can actually open a window."""
    try:
        from matplotlib.backends.registry import backend_registry
        _, gui = backend_registry.resolve_backend(matplotlib.get_backend())
        return gui is not None
    except Exception:
        pass
    name = matplotlib.get_backend().lower()
    return name not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="an OUT directory, or quoted glob(s) of *.cohout")
    ap.add_argument("-o", "--output", default="corner_ajs.png",
                    help="output image (default: corner_ajs.png)")
    ap.add_argument("--par", "--pars", dest="par", nargs="+", action="extend",
                    metavar="PATH",
                    help=".par file(s), directory, or quoted glob of known "
                         "pulsars; candidates within their F0 search bands (or "
                         "ajs-predicted frequencies with --inf) are removed")
    ap.add_argument("--par-tol", type=float, default=None, metavar="HZ",
                    help="absolute frequency half-width for par masking "
                         "(default: max(0.005, 5e-4*F0) Hz)")
    ap.add_argument("--par-frac-tol", type=float, default=0.0, metavar="FRAC",
                    help="fractional-F0 half-width for par masking; combined "
                         "with --par-tol using the larger value")
    ap.add_argument("--par-harmonics", type=int, default=1, metavar="N",
                    help="also mask harmonics 2..N of each par F0 (default: 1, "
                         "fundamental only)")
    ap.add_argument("--inf", metavar="PATH",
                    help="PRESTO .inf for the observation; enables predicting "
                         "the masked frequency separately for every ajs grid "
                         "point instead of using the par F0 at PEPOCH")
    ap.add_argument("--linear-freq", action="store_true",
                    help="use a linear frequency axis (default: log)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="do not open an interactive window (default: save the "
                         "figure and then show it with plt.show)")
    ap.add_argument("--snr-min", type=float, default=0.0, metavar="S/N",
                    help="drop candidates with S/N below this value before "
                         "plotting (default: 0, keep all)")
    ap.add_argument("--dpi", type=int, default=200,
                    help="output resolution (default: 200)")
    ap.add_argument("--size-min", type=float, default=2.0,
                    help="marker area for the lowest S/N (default: 2)")
    ap.add_argument("--size-max", type=float, default=70.0,
                    help="marker area for the highest S/N (default: 70)")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="marker transparency (default: 0.30)")
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.inputs:
        if os.path.isdir(pattern):
            hits = sorted(glob.glob(os.path.join(pattern, "*.cohout")))
        else:
            hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"warning: no .cohout files match {pattern!r}", file=sys.stderr)
        paths.extend(hits)
    if not paths:
        print("error: no input files", file=sys.stderr)
        return 1

    data, nfiles, skipped = collect(paths)
    if len(data) == 0:
        print("error: no candidate rows found", file=sys.stderr)
        return 1

    if args.snr_min > 0:
        n_before = len(data)
        data = data[data[:, 0] >= args.snr_min]
        print(f"S/N cut: kept {len(data)} of {n_before} candidates with "
              f"S/N >= {args.snr_min:g}", file=sys.stderr)
        if len(data) == 0:
            print(f"error: every candidate has S/N < {args.snr_min:g}",
                  file=sys.stderr)
            return 1

    npars = 0
    nremoved = 0
    ajs_predicted = False
    if args.par:
        pars = gather_pars(args.par)
        if not pars:
            print("error: --par was given but no usable par files were found",
                  file=sys.stderr)
            return 1

        inf = None
        if args.inf:
            if PAJS is None:
                print("error: --inf needs demod/pulsar_optimal_ajs_circular.py "
                      "(and its orbit model) on the import path", file=sys.stderr)
                return 1
            try:
                inf = read_inf(args.inf)
            except Exception as exc:
                print(f"error: cannot read {args.inf!r}: {exc}", file=sys.stderr)
                return 1
            n, dt, epoch, t_obs, bary = inf
            print(f"observation: {args.inf}  epoch={epoch:.9f}  "
                  f"T={t_obs:g} s  barycentered={'yes' if bary else 'no'}",
                  file=sys.stderr)
            if not bary:
                print("warning: .inf says the data are NOT barycentered; the "
                      "par-based prediction assumes barycentric frequencies",
                      file=sys.stderr)

        n_original = len(data)
        data, nremoved, counts, halfwidths, pred_ranges = mask_known(
            data, pars, abs_tol=args.par_tol,
            frac_tol=args.par_frac_tol,
            max_harmonic=args.par_harmonics, inf=inf)
        npars = len(pars)
        ajs_predicted = inf is not None
        mode = "ajs-predicted" if ajs_predicted else "F0"
        print(f"par mask ({mode}): removed {nremoved} of {n_original} "
              f"candidates around {npars} known pulsar(s) "
              f"(harmonics 1..{max(1, args.par_harmonics)})", file=sys.stderr)
        for psr, _, f0, _ in pars:
            line = (f"  {psr:<16s} F0={f0:.9f} Hz  "
                    f"half-width={halfwidths[psr]:.6g} Hz  "
                    f"removed={counts.get(psr, 0)}")
            if psr in pred_ranges:
                lo, hi = pred_ranges[psr]
                line += f"  predicted={lo:.6f}..{hi:.6f} Hz"
            print(line, file=sys.stderr)
        if len(data) == 0:
            print("error: every candidate was removed by the par mask",
                  file=sys.stderr)
            return 1

    fig = make_plot(data, nfiles, args.output, logfreq=not args.linear_freq,
                    dpi=args.dpi, size_min=args.size_min,
                    size_max=args.size_max, alpha=args.alpha, npars=npars,
                    nremoved=nremoved, ajs_predicted=ajs_predicted,
                    snr_min=args.snr_min, show=args.show)
    print(f"{len(data)} candidate(s) from {nfiles} file(s) "
          f"(skipped {skipped}) -> {args.output}", file=sys.stderr)
    if fig is not None:
        if backend_is_interactive():
            plt.show()
        else:
            print(f"note: matplotlib backend "
                  f"{matplotlib.get_backend()!r} is non-interactive, so no "
                  f"window will open (use --no-show to silence this)",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
