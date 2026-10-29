#!/usr/bin/env python
"""
End-to-end test for demod/pulsar_optimal_ajs_circular.py.

All input files are command-line arguments; nothing about the observation is
hard-coded:

    python demod/test_pulsar_optimal_ajs_circular.py \
        GRID.csv PULSAR.par OBS.inf OBS.dat \
        [-o OUTDIR] [--lofreq HZ --hifreq HZ] ...

The test:

  real data
    1. runs the new script on GRID.csv + PULSAR.par + OBS.inf;
    2. checks the written row is exactly the nearest ajs grid row;
    3. searches OBS.dat and its demodulated version with the same in-memory
       driver (bin/coherent_search.jl --demod-grid), around the par's F0;
    4. folds both with PRESTO prepfold and asserts the demodulated fold has the
       higher S/N and the right period.

  controlled injection
    5. injects the same ephemeris into a fresh series at high S/N with
       pulsegen_gr.py (same N/dt/epoch as OBS.inf);
    6. runs the new script on that injection and compares its best-fit ajs with
       the injection truth YAML;
    7. demodulates at the snapped row, searches + folds, and asserts a strong
       recovery while the un-demodulated injection is smeared.

Use --no-injection to run only the real-data part.

Exit status is non-zero on failure.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pulsar_optimal_ajs_circular import par_to_dd, read_par  # noqa: E402
from pulsegen_gr import read_inf  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log_run(cmd, logpath):
    """Run a command, tee its combined output to logpath, fail loudly."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    with open(logpath, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              text=True)
        fh.write(f"\n# exit={proc.returncode}\n")
    if proc.returncode != 0:
        tail = "".join(open(logpath).readlines()[-30:])
        raise RuntimeError(f"command failed ({proc.returncode}):\n{printable}\n{tail}")
    return logpath


def read_text(path):
    with open(path) as fh:
        return fh.read()


def cohout_best(path):
    """(snr, freq, period) of the highest-S/N candidate in a .cohout."""
    best = None
    for line in read_text(path).splitlines():
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        snr, freq, period = float(parts[1]), float(parts[2]), float(parts[3])
        if best is None or snr > best[0]:
            best = (snr, freq, period)
    if best is None:
        raise AssertionError(f"no candidates in {path}")
    return best


def bestprof_sigma_period(path):
    """(sigma, P_topo_ms) from a prepfold .bestprof."""
    sigma = 0.0
    period = None
    for line in read_text(path).splitlines():
        if "Prob(Noise)" in line:
            m = re.search(r"~([0-9.]+)\s*sigma", line)
            if m:
                sigma = float(m.group(1))
        if "P_topo (ms)" in line:
            m = re.search(r"=\s*([0-9.eE+-]+)", line)
            if m:
                period = float(m.group(1))
    return sigma, period


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise AssertionError(f"no file matching {pattern}")
    return files[-1]


def nearest_expected_row(grid, target):
    """Independent per-axis nearest unique value (the grid is a product)."""
    arr = np.loadtxt(grid, delimiter=",", skiprows=1)
    out = []
    for i in range(3):
        u = np.unique(arr[:, i])
        out.append(float(u[np.argmin(np.abs(u - target[i]))]))
    return np.array(out)


def parse_best_fit(stdout):
    m = re.search(r"best-fit ajs: accel=([0-9.eE+-]+)\s+jerk=([0-9.eE+-]+)\s+"
                  r"snap=([0-9.eE+-]+)", stdout)
    if not m:
        raise AssertionError("could not parse 'best-fit ajs' from the script output")
    return np.array([float(m.group(i)) for i in (1, 2, 3)])


def run_script(python, grid, par, inf, out, logpath):
    cmd = [python, os.path.join(HERE, "pulsar_optimal_ajs_circular.py"),
           grid, par, inf, "-o", out]
    log_run(cmd, logpath)
    return read_text(logpath)


def inmem_search(args, dat, grid, outsub, tag, lofreq, hifreq):
    """Run bin/coherent_search.jl --demod-grid; return the best candidate."""
    os.makedirs(outsub, exist_ok=True)
    cmd = [args.julia, f"--project={REPO}", "-t", str(args.nthreads),
           os.path.join(REPO, "bin", "coherent_search.jl"), dat,
           "--demod-grid", grid, "--outdir", outsub,
           "--lofreq", f"{lofreq:.15g}", "--hifreq", f"{hifreq:.15g}",
           "--threshold", f"{args.threshold:g}"]
    log_run(cmd, os.path.join(outsub, f"{tag}.log"))
    cohout = newest(os.path.join(outsub, "*.cohout"))
    return cohout_best(cohout), cohout


def fold(args, dat, freq, stem, logpath):
    cmd = [args.prepfold, "-topo", "-nosearch", "-n", "64", "-noxwin",
           "-f", f"{freq:.15g}", "-o", stem, dat]
    log_run(cmd, logpath)
    bestprof = newest(stem + "_*.bestprof")
    return bestprof_sigma_period(bestprof), bestprof


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="End-to-end test for pulsar_optimal_ajs_circular.py.")
    ap.add_argument("grid", help="ajs grid CSV (header accel,jerk,snap)")
    ap.add_argument("par", help="TEMPO/TEMPO2 ephemeris (.par)")
    ap.add_argument("inf", help="PRESTO .inf of the observation")
    ap.add_argument("dat", help="PRESTO .dat time series to demodulate/fold")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output directory (default: "
                         "<dat-dir>/optimal_ajs_test)")
    ap.add_argument("--lofreq", type=float, default=None,
                    help="Low search frequency [Hz] (default: F0-halfwidth)")
    ap.add_argument("--hifreq", type=float, default=None,
                    help="High search frequency [Hz] (default: F0+halfwidth)")
    ap.add_argument("--freq-halfwidth", type=float, default=None,
                    help="Half-width of the default search band [Hz] "
                         "(default: max(0.005, 5e-4*F0))")
    ap.add_argument("--expected-period-ms", type=float, default=None,
                    help="Expected folded period [ms] "
                         "(default: 1000/F0)")
    ap.add_argument("--period-tol-frac", type=float, default=1e-3,
                    help="Relative tolerance on the recovered period "
                         "(default: 1e-3)")
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="Coherent-search threshold (default: 4)")
    ap.add_argument("--nthreads", type=int,
                    default=int(os.environ.get("NTHREADS", "4")),
                    help="Julia search threads (default: 4 or $NTHREADS)")
    ap.add_argument("--julia", default=os.environ.get("JULIA", "julia"),
                    help="Julia executable")
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable),
                    help="Python executable for the scripts")
    ap.add_argument("--prepfold",
                    default=os.environ.get("PREPFOLD",
                                           "/home/fadong/anaconda3/bin/prepfold"),
                    help="PRESTO prepfold executable")
    ap.add_argument("--no-injection", action="store_true",
                    help="Skip the controlled-injection half of the test")
    ap.add_argument("--injection-snr", type=float, default=80.0,
                    help="Injected S/N (default: 80)")
    ap.add_argument("--injection-pulse-width", type=float, default=0.0005,
                    help="Injected Gaussian pulse sigma [s] (default: 5e-4)")
    ap.add_argument("--injection-seed", type=int, default=1234,
                    help="Injection RNG seed (default: 1234)")
    return ap


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def main(argv=None):
    args = build_parser().parse_args(argv)

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(args.dat)),
                              "optimal_ajs_test")
    os.makedirs(outdir, exist_ok=True)
    print(f"output directory: {outdir}")

    par = read_par(args.par)
    N, dt, epoch, _ = read_inf(args.inf)
    t_obs = N * dt
    prm, p0, pdot, info = par_to_dd(par, epoch)
    f_spin = 1.0 / p0

    expected_period_ms = (args.expected_period_ms if args.expected_period_ms
                          is not None else 1000.0 / f_spin)
    halfwidth = (args.freq_halfwidth if args.freq_halfwidth is not None
                 else max(0.005, 5e-4 * f_spin))
    lofreq = f_spin - halfwidth if args.lofreq is None else args.lofreq
    hifreq = f_spin + halfwidth if args.hifreq is None else args.hifreq
    print(f"par: F0={f_spin:.12g} Hz  search band {lofreq:.9f}-{hifreq:.9f} Hz  "
          f"expected period {expected_period_ms:.6f} ms")

    # ------------------------------------------------------------------
    # 1. real data: grid selection
    # ------------------------------------------------------------------
    print("\n[1/4] grid row from the ephemeris + .inf")
    out_csv = os.path.join(outdir, "nsns_grid_pulsar_optimal_ajs_circular.csv")
    stdout = run_script(args.python, args.grid, args.par, args.inf, out_csv,
                        os.path.join(outdir, "select.log"))
    lines = [ln for ln in read_text(out_csv).splitlines() if ln.strip()]
    check(len(lines) == 2, "output has a header and exactly one data row")
    check(lines[0].strip() == "accel,jerk,snap", "output header is accel,jerk,snap")
    chosen = np.array([float(x) for x in lines[1].split(",")])
    best_fit = parse_best_fit(stdout)
    expected = nearest_expected_row(args.grid, best_fit)
    check(np.allclose(chosen, expected, rtol=0.0, atol=0.0),
          f"chosen row {chosen} is the nearest grid row {expected}")

    # ------------------------------------------------------------------
    # 2. real data: search original vs demodulated (same metric)
    # ------------------------------------------------------------------
    print("\n[2/4] in-memory coherent search, original vs demodulated")
    zero_csv = os.path.join(outdir, "zero.csv")
    with open(zero_csv, "w") as fh:
        fh.write("accel,jerk,snap\n0,0,0\n")

    (snr0, f0, p_orig), _ = inmem_search(
        args, args.dat, zero_csv, os.path.join(outdir, "search_original"),
        "original", lofreq, hifreq)
    (snr1, f1, p_demod), coh1 = inmem_search(
        args, args.dat, out_csv, os.path.join(outdir, "search_demod"),
        "demod", lofreq, hifreq)
    print(f"  original:    S/N={snr0:.2f}  f={f0:.9f} Hz  P={p_orig:.6f} ms")
    print(f"  demodulated: S/N={snr1:.2f}  f={f1:.9f} Hz  P={p_demod:.6f} ms")
    check(snr1 > snr0 + 2.0,
          f"demodulated search improves S/N by >2 ({snr0:.2f} -> {snr1:.2f})")
    check(abs(p_demod - expected_period_ms) <
          args.period_tol_frac * expected_period_ms,
          f"demodulated candidate period {p_demod:.6f} ms is near "
          f"{expected_period_ms:.6f} ms")
    check(os.path.exists(coh1), "demodulated .cohout exists")

    # ------------------------------------------------------------------
    # 3. real data: fold both (the requested time-domain check)
    # ------------------------------------------------------------------
    print("\n[3/4] prepfold original vs demodulated at fixed period")
    demod_dat = os.path.join(outdir, "regular_demod.dat")
    cmd = [args.julia, f"--project={REPO}", os.path.join(HERE, "demod_dat.jl"),
           args.dat, demod_dat,
           "--accel", lines[1].split(",")[0],
           "--jerk", lines[1].split(",")[1],
           "--snap", lines[1].split(",")[2]]
    log_run(cmd, os.path.join(outdir, "demod_dat.log"))

    (sig0, per0), _ = fold(args, args.dat, f0,
                           os.path.join(outdir, "fold_original"),
                           os.path.join(outdir, "fold_original.log"))
    (sig1, per1), _ = fold(args, demod_dat, f1,
                           os.path.join(outdir, "fold_demod"),
                           os.path.join(outdir, "fold_demod.log"))
    print(f"  original fold:    {sig0:.2f} sigma  P={per0:.6f} ms")
    print(f"  demodulated fold: {sig1:.2f} sigma  P={per1:.6f} ms")
    check(sig1 > sig0 + 1.0,
          f"demodulated fold improves S/N by >1 sigma ({sig0:.2f} -> {sig1:.2f})")
    check(per1 is not None and
          abs(per1 - expected_period_ms) <
          args.period_tol_frac * expected_period_ms,
          f"folded period {per1} ms is near {expected_period_ms:.6f} ms")

    if args.no_injection:
        print("\n[4/4] controlled injection skipped (--no-injection)")
        print("\nALL TESTS PASSED")
        print(f"artifacts: {outdir}")
        return 0

    # ------------------------------------------------------------------
    # 4. controlled injection: the ajs values and the recovery
    # ------------------------------------------------------------------
    print("\n[4/4] controlled injection of the same ephemeris")
    if info["pb_days"] is None:
        raise AssertionError("controlled injection needs a binary par (PB/A1)")

    inj_dir = os.path.join(outdir, "injection")
    os.makedirs(inj_dir, exist_ok=True)
    inj_base = os.path.join(inj_dir, "inj_par")
    cmd = [args.python, os.path.join(HERE, "pulsegen_gr.py"),
           "-inf", args.inf, "-outdir", inj_dir, "-outbasename", "inj_par",
           "-pb", f"{info['pb_days']:g}", "-e", f"{info['e']:g}",
           "-x", f"{info['a1']:g}",
           "-omega_peri", f"{info['omega_peri']:.15g}",
           "-anchor", "start", "-A_T", f"{prm.A_T:.15g}",
           "-p0", f"{p0:.15g}", "-pdot", f"{pdot:.15g}",
           "-snr", f"{args.injection_snr:g}",
           "-pulse_width", f"{args.injection_pulse_width:g}",
           "-seed", str(args.injection_seed)]
    log_run(cmd, os.path.join(inj_dir, "pulsegen.log"))
    inj_inf = inj_base + ".inf"
    inj_dat = inj_base + ".dat"
    truth = read_text(inj_base + "_truth.yaml")

    def truth_value(key):
        m = re.search(rf"^{key}:\s*([0-9.eE+-]+)", truth, re.MULTILINE)
        if not m:
            raise AssertionError(f"{key} missing from injection truth YAML")
        return float(m.group(1))

    truth_ajs = np.array([truth_value("best_fit_accel"),
                          truth_value("best_fit_jerk"),
                          truth_value("best_fit_snap")])

    inj_csv = os.path.join(inj_dir, "inj_pulsar_optimal_ajs_circular.csv")
    inj_stdout = run_script(args.python, args.grid, args.par, inj_inf, inj_csv,
                            os.path.join(inj_dir, "select.log"))
    inj_fit = parse_best_fit(inj_stdout)
    tol = np.array([1e-2, 1e-5, 1e-9])
    check(np.all(np.abs(inj_fit - truth_ajs) <= tol),
          f"script best fit {inj_fit} matches injection truth {truth_ajs}")
    inj_lines = [ln for ln in read_text(inj_csv).splitlines() if ln.strip()]
    check(len(inj_lines) == 2, "injection output has one data row")

    # Search the injection with and without demodulation.
    (isnr0, if0, ip0), _ = inmem_search(
        args, inj_dat, zero_csv, os.path.join(inj_dir, "search_original"),
        "original", lofreq, hifreq)
    (isnr1, if1, ip1), _ = inmem_search(
        args, inj_dat, inj_csv, os.path.join(inj_dir, "search_demod"),
        "demod", lofreq, hifreq)
    print(f"  injection original:    S/N={isnr0:.2f}  f={if0:.9f} Hz")
    print(f"  injection demodulated: S/N={isnr1:.2f}  f={if1:.9f} Hz")
    check(isnr1 > isnr0 + 5.0,
          f"injection demodulation improves S/N by >5 "
          f"({isnr0:.2f} -> {isnr1:.2f})")
    check(abs(ip1 - expected_period_ms) <
          args.period_tol_frac * expected_period_ms,
          f"injection recovered period {ip1:.6f} ms is near "
          f"{expected_period_ms:.6f} ms")

    (isig0, _), _ = fold(args, inj_dat, if0,
                         os.path.join(inj_dir, "fold_original"),
                         os.path.join(inj_dir, "fold_original.log"))
    inj_demod = os.path.join(inj_dir, "inj_demod.dat")
    cmd = [args.julia, f"--project={REPO}", os.path.join(HERE, "demod_dat.jl"),
           inj_dat, inj_demod,
           "--accel", inj_lines[1].split(",")[0],
           "--jerk", inj_lines[1].split(",")[1],
           "--snap", inj_lines[1].split(",")[2]]
    log_run(cmd, os.path.join(inj_dir, "demod_dat.log"))
    (isig1, iper1), _ = fold(args, inj_demod, if1,
                             os.path.join(inj_dir, "fold_demod"),
                             os.path.join(inj_dir, "fold_demod.log"))
    print(f"  injection original fold:    {isig0:.2f} sigma")
    print(f"  injection demodulated fold: {isig1:.2f} sigma  "
          f"P={iper1:.6f} ms")
    check(isig1 > isig0 + 5.0,
          f"injection folded recovery improves by >5 sigma "
          f"({isig0:.2f} -> {isig1:.2f})")
    check(isig1 >= 10.0,
          f"injection is recovered at high significance ({isig1:.2f} sigma)")

    print("\nALL TESTS PASSED")
    print(f"artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 -- top-level test harness
        print(f"\nERROR: {exc}")
        sys.exit(2)
