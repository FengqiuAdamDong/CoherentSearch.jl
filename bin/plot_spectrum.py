#!/usr/bin/env python3
"""
plot_spectrum.py -- plot the per-trial S/N spectrum written by ``--spectrum``.

``coherent_search.jl --spectrum FILE.fft`` writes ``FILE.spectrum``: an
8192-byte ASCII header followed by one little-endian ``Float32`` array per
decimation rung.  Each array holds every scored trial's peak boxcar S/N ``z``
on that rung's own trial grid, ``f = f0 + j*df``, with ``NaN`` where a rung
stops at Nyquist.  The header documents the statistic and the conversion to a
whitened FFT power.

By default every rung is drawn on one pair of axes, because the rungs share a
common scale once their S/N is converted to the *single-bin* whitened power::

    P = 1 + z^2 / 2

i.e. all of the trial's signal power placed in one Fourier bin (the header's
``P(H=1)`` row).  Horizontal guides mark the same conversion for S/N 5, 8, 10
and 12, so a peak can be read against a theoretical detection threshold.  The
rungs are *not* interchangeable -- their false-alarm floors differ because
each maximizes over its own profile length and width bank -- so ``--stack``
draws them in separate panels, and the header's per-rung ``FAP=1e-4`` note is
the authoritative comparison.

With ``--snr`` the raw statistic is plotted instead of the power.

Examples::

    python3 bin/plot_spectrum.py PM0063_034C1_DM445.0_red.spectrum
    python3 bin/plot_spectrum.py FILE.spectrum --stack --log --xlim 5 20
    python3 bin/plot_spectrum.py FILE.spectrum --snr -o zoom.png --xlim 7.0 7.3

Requires numpy and matplotlib (the search itself has no Python dependency).
"""

import argparse
import os
import sys
import warnings

import numpy as np

HEADER_BYTES = 8192
SNR_GUIDES = (5.0, 8.0, 10.0, 12.0)


def read_header(path):
    """Parse the fixed-size ASCII header into (metadata dict, list of rungs)."""
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_BYTES)
    if not raw.startswith(b"# CoherentSearch.jl per-trial spectrum"):
        raise ValueError("%s is not a CoherentSearch .spectrum file" % path)
    if len(raw) < HEADER_BYTES:
        raise ValueError("%s is truncated: the header alone is %d bytes"
                         % (path, HEADER_BYTES))
    text = raw.split(b"\0", 1)[0].decode("ascii", "replace")
    lines = text.splitlines()

    meta = {}
    rungs = []
    for line in lines:
        if line.startswith("#rung "):
            rung = {}
            for tok in line.split()[1:]:
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    rung[key] = val
            rungs.append(rung)
        elif line.startswith("# "):
            # The #grid block wraps across lines; keep the first occurrence of
            # each key ("first" because later prose repeats words like NaN).
            for tok in line[2:].split():
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    meta.setdefault(key, val)
    if not rungs:
        raise ValueError("%s has no #rung lines; was it written by --spectrum?" % path)
    return meta, rungs


def decimate(y, max_points):
    """Block-max decimation: ``(indices, values)`` keeping each block's peak."""
    n = len(y)
    if max_points <= 0 or n <= max_points:
        return np.arange(n), y
    step = int(np.ceil(n / max_points))
    m = n // step
    if m < 2:
        return np.arange(n), y
    # Every block contributes its strongest value: an envelope, not a sample.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks
        yr = np.nanmax(y[:m * step].reshape(m, step), axis=1)
    return np.arange(m) * step, yr


def iter_rungs(path, rungs, power=True, max_points=300000):
    """Yield ``(k, Hk, n, n_scored, x, y)`` one rung at a time.

    Only the decimated arrays are kept, so a default-band file (tens of
    millions of trials per rung) plots without the payload ever existing in
    float64: it is read as Float32 and converted after decimation.  ``x`` is
    the searched frequency of each plotted point, ``y`` the statistic (power
    ``1 + z^2/2`` or ``z`` itself).
    """
    size = os.path.getsize(path)
    for rung in rungs:
        k = int(rung["k"])
        Hk = int(rung["Hk"])
        f0 = float(rung["f0"])
        df = float(rung["df"])
        n = int(rung["n"])
        off = int(rung["offset"])
        have = max(0, (size - off) // 4)
        if off < HEADER_BYTES or have < n:
            sys.exit("truncated %s: rung k=%d wants %d values at byte %d, file has %d"
                     % (path, k, n, off, have))
        z = np.fromfile(path, dtype="<f4", count=n, offset=off)
        bad = np.isnan(z)
        n_scored = n - int(np.count_nonzero(bad)) if bad.any() else n
        z = z[:n_scored]                     # NaN is always a contiguous tail
        y = z
        if power:                            # P = 1 + z^2/2, in place
            y **= 2
            y *= np.float32(0.5)
            y += np.float32(1.0)
        idx, yd = decimate(y, max_points)
        yield k, Hk, n, n_scored, f0 + idx * df, yd.astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot a CoherentSearch.jl .spectrum file (frequency axis S/N).")
    ap.add_argument("spectrum", help="the .spectrum file written by --spectrum")
    ap.add_argument("-o", "--out", help="output PNG (default: <spectrum stem>.png)")
    ap.add_argument("--stack", action="store_true",
                    help="one panel per decimation rung instead of one overlaid axes")
    ap.add_argument("--snr", action="store_true",
                    help="plot the peak boxcar S/N z instead of the whitened power")
    ap.add_argument("--log", action="store_true",
                    help="logarithmic power axis (incompatible with --snr)")
    ap.add_argument("--xlim", nargs=2, type=float, metavar=("FMIN", "FMAX"),
                    help="searched-frequency limits in Hz")
    ap.add_argument("--max-points", type=int, default=300000,
                    help="per-rung point cap for display (block-max decimation; 0 = all)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", help="override the figure title")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args(argv)

    if args.snr and args.log:
        ap.error("--log is a power-axis option; drop it (or --snr)")

    meta, rungs = read_header(args.spectrum)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power = not args.snr
    nharms = meta.get("nharms", "?")
    lo = meta.get("lofreq", "?")
    hi = meta.get("hifreq", "?")
    tsec = meta.get("T", "?")

    def ylabel():
        return "whitened power  P = 1 + z$^2$/2" if power else "peak boxcar S/N  z"

    def guides(ax, annotate=True):
        # Grey guides at the same conversion for a theoretical detection S/N.
        for g in SNR_GUIDES:
            y = 1.0 + 0.5 * g * g if power else g
            ax.axhline(y, color="0.55", ls=":", lw=1.0, zorder=0)
            if annotate:
                ax.annotate("S/N %g" % g, xy=(0.004, y), xycoords=("axes fraction", "data"),
                            xytext=(0, 2), textcoords="offset points",
                            ha="left", va="bottom", fontsize=7, color="0.35",
                            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))

    rungs_iter = lambda: iter_rungs(args.spectrum, rungs, power, args.max_points)
    total = scored = 0

    if args.stack:
        fig, axes = plt.subplots(len(rungs), 1, sharex=True, sharey=True,
                                 figsize=(12, 1.9 * len(rungs) + 1.2), squeeze=False)
        axes = axes[:, 0]
        for i, (k, Hk, n, nv, x, y) in enumerate(rungs_iter()):
            total += n
            scored += nv
            ax = axes[i]
            if len(x):
                ax.plot(x, y, lw=0.5, color="C0")
            else:
                ax.text(0.5, 0.5, "no trials below Nyquist at this depth",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="0.5")
            guides(ax, annotate=(i == 0))     # shared y: label the top panel only
            ax.set_ylabel("k=%d\nH=%d" % (k, Hk), fontsize=8)
            ax.margins(x=0.01)
            ax.grid(alpha=0.2, lw=0.4)
        axes[-1].set_xlabel("searched frequency (Hz)")
        ax0 = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(12, 5.0))
        for i, (k, Hk, n, nv, x, y) in enumerate(rungs_iter()):
            total += n
            scored += nv
            if not len(x):
                continue
            ax.plot(x, y, lw=0.55, alpha=0.85, color="C%d" % (i % 10),
                    label="k=%d  (H=%d, %d bins)" % (k, Hk, 2 * Hk))
        guides(ax)
        ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
        ax.set_xlabel("searched frequency (Hz)")
        ax.margins(x=0.01)
        ax.grid(alpha=0.2, lw=0.4)
        ax0 = ax

    ax0.set_ylabel(ylabel())
    if args.log:
        for a in (axes if args.stack else [ax0]):
            a.set_yscale("log")
    if args.xlim:
        ax0.set_xlim(*args.xlim)
    title = args.title or ("%s  --  %.4g-%.4g Hz, nharms=%s, T=%s s, %d rung%s"
                           % (os.path.basename(args.spectrum), float(lo), float(hi),
                              nharms, tsec, len(rungs), "" if len(rungs) == 1 else "s"))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    out = args.out or (os.path.splitext(args.spectrum)[0] + ".png")
    fig.savefig(out, dpi=args.dpi)
    print("wrote %s (%d rungs, %d trials, %d scored)" % (out, len(rungs), total, scored))
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        sys.exit("plot_spectrum.py: error: %s" % exc)
