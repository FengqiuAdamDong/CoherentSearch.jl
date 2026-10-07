#!/usr/bin/env python
"""MCMC check of the circular-orbit grid's phase-tolerance guarantee.

The grid (nsns_grid.py -mode circular) claims: for any true circular orbit whose
(p_o, sin_i, A_T) lies in the covered ranges, the nearest grid point keeps the
residual SPIN phase within phase_tol cycles across the observation.  This test
Monte-Carlos that claim directly, with no linearisation:

  1. Derive the grid exactly as nsns_grid.py does (imported, so the two cannot
     drift) for a chosen T, p0, p_o/sin_i ranges, phase_tol and drop_pct.
  2. Sample N random circular orbits uniformly in the covered ranges:
     log-uniform p_o, uniform sin_i, uniform A_T in [0, 2*pi).
  3. Derive x = a_p sin(i)/c the same way the grid does.
  4. Snap (omega_b, x, A_T) to the nearest grid value (the grid is a Cartesian
     product), i.e. find the actual grid point the search would test.
  5. Compute the exact residual spin phase
         R(t) = Phi_true(t) - Phi_grid(t),
         Phi(t) = (x/p0) sin(omega_b t + A_T),   t in [0, T] (A_T at the start),
     on a fine t grid, and record max_t |R| (the quantity phase_tol bounds;
     the ajs path's integrated_trunc is also a max absolute integrated phase).
  6. PASS iff the worst max|R| over all trials is <= phase_tol, AND the
     deterministic worst cell corner (which random sampling rarely hits) is
     within phase_tol on both the linear bound and the exact nonlinear R.

Usage:
    python demod/test_circular_grid.py [-inf OBS.inf | -t_obs T] [-p0 P0] \
        [-phase_tol_cycles 0.1] [-p_o ...] [-sin_i ...] [-drop_pct 0] [-n_mc 2000]

Run with -drop_pct 0 to verify the fundamental spacing (the full range is
covered).  With drop_pct > 0 the highest-x orbits are not covered, so the test
samples only orbits whose x falls in the kept [x_min, x_max] box and reports how
many were skipped.
"""
import argparse
import os
import sys

import numpy as np

# Import the grid code itself so the test cannot drift from the implementation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsns_grid as ng  # noqa: E402


def build_args(cli):
    """nsns_grid args namespace with this test's fields filled in."""
    ap = ng.build_parser()
    args = ap.parse_args([])          # grid defaults (p_o ladder, sin_i values, ...)
    for k in ("inf", "t_obs", "p0", "phase_tol_cycles", "drop_pct",
              "companion_mass", "pulsar_mass"):
        v = getattr(cli, k)
        if v is not None:
            setattr(args, k, v)
    if cli.p_o:
        args.p_o = cli.p_o
    if cli.sin_i:
        args.sin_i = cli.sin_i
    # Resolve T from the .inf when given and -t_obs absent.
    if args.t_obs is None:
        if args.inf is None:
            ap.error("give -inf or -t_obs")
        N, dt, _ = ng.read_inf(args.inf)
        args.t_obs = N * dt
    return args


def residual_phase(x_t, w_t, A_t, x_g, w_g, A_g, p0, T, n=20001):
    """Exact residual spin phase over the observation.

    ``A_T`` is defined at the observation START, so the phase is
    ``omega_b * t + A_T`` with ``t`` the absolute time since the start (t in
    [0, T]); the demod maps the midpoint anchor back to the start.
    Returns ``(max_abs, peak_to_peak)``.  ``max_abs`` = max_t |Phi_true -
    Phi_grid| is the quantity the ajs path budgets (its ``integrated_trunc`` is
    also a max absolute integrated phase), and is what phase_tol bounds."""
    t = np.linspace(0.0, T, n)
    phi_t = w_t * t + A_t
    phi_g = w_g * t + A_g
    R = (x_t / p0) * np.sin(phi_t) - (x_g / p0) * np.sin(phi_g)
    return np.max(np.abs(R)), R.max() - R.min()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-inf", type=str, default=None)
    ap.add_argument("-t_obs", type=float, default=None)
    ap.add_argument("-p0", type=float, default=1.0)
    ap.add_argument("-phase_tol_cycles", type=float, default=0.1)
    ap.add_argument("-p_o", type=float, nargs="+", default=None,
                    help="orbital periods [yr] to cover (default: nsns_grid's)")
    ap.add_argument("-sin_i", type=float, nargs="+", default=None,
                    help="sin(i) values to cover (default: nsns_grid's)")
    ap.add_argument("-drop_pct", type=float, default=0.0)
    ap.add_argument("-companion_mass", type=float, default=None)
    ap.add_argument("-pulsar_mass", type=float, default=None)
    ap.add_argument("-n_mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    args = build_args(cli)
    spec, omega_b, x, A_T, orbits = ng.derive_circular_grid(args)

    wlo, whi = omega_b[0], omega_b[-1]
    xlo, xhi = x[0], x[-1]
    p0, T, tol = args.p0, args.t_obs, args.phase_tol_cycles

    # Covered physical ranges (the kept orbits, for x-filtering under -drop_pct).
    kept = [o for o in orbits if o["kept"]]
    p_o_lo = min(o["p_o_s"] for o in kept)
    p_o_hi = max(o["p_o_s"] for o in kept)
    sin_lo = min(o["sin_i"] for o in kept)
    sin_hi = max(o["sin_i"] for o in kept)
    # The true covered amplitude top: the spacings were derived from this, NOT
    # from the padded grid top x[-1] (which overshoots by up to one spacing).
    x_covered_max = max(o["x"] for o in kept)

    rng = np.random.default_rng(cli.seed)
    n = cli.n_mc
    p_o = np.exp(rng.uniform(np.log(p_o_lo), np.log(p_o_hi), n))
    sin_i = rng.uniform(sin_lo, sin_hi, n)
    A_truth = rng.uniform(0.0, 2 * np.pi, n)
    x_truth = ng.circular_x_lt_s(p_o, sin_i, args.companion_mass, args.pulsar_mass)
    w_truth = 2 * np.pi / p_o

    # Snap to the nearest grid value on each axis.
    w_snap = omega_b[np.abs(w_truth.reshape(-1, 1) - omega_b).argmin(axis=1)]
    x_snap = x[np.abs(x_truth.reshape(-1, 1) - x).argmin(axis=1)]
    A_snap = A_T[np.abs(A_truth.reshape(-1, 1) - A_T).argmin(axis=1)]

    devs = np.empty(n)
    ptps = np.empty(n)
    skipped = 0
    for i in range(n):
        if not (xlo - 1e-12 <= x_truth[i] <= xhi + 1e-12):
            skipped += 1
            devs[i] = np.nan
            ptps[i] = np.nan
            continue
        devs[i], ptps[i] = residual_phase(
            x_truth[i], w_truth[i], A_truth[i],
            x_snap[i], w_snap[i], A_snap[i], p0, T)

    ok = ~np.isnan(devs)
    devs, ptps = devs[ok], ptps[ok]
    used = len(devs)

    print(f"Grid: {spec['n_trials']:.3e} trials, "
          f"p_o {2*np.pi/whi/86400:.4g}-{2*np.pi/wlo/86400:.4g} d, "
          f"x {xlo:.4g}-{xhi:.4g} lt-s, A_T {len(A_T)} values")
    print(f"phase_tol = {tol:g} cycles   T = {T:g} s   p0 = {p0} s")
    print(f"spacings: d_omega={spec['spacings'][0]:.4e} rad/s  "
          f"d_x={spec['spacings'][1]:.4e} lt-s  "
          f"d_A_T={spec['spacings'][2]:.4e} rad")
    print(f"MC trials: {n} ({skipped} skipped outside the covered x box), "
          f"{used} scored")
    if used == 0:
        sys.exit("FAIL: no trials inside the covered box")

    i_max = devs.argmax()
    print(f"\nresidual max|Phi_true - Phi_grid| over the observation:")
    print(f"  mean   = {devs.mean():.4f} cycles "
          f"({100*devs.mean()/tol:.1f}% of budget)")
    print(f"  median = {np.median(devs):.4f}")
    print(f"  max    = {devs.max():.4f} cycles "
          f"({100*devs.max()/tol:.1f}% of budget)")
    print(f"  peak-to-peak range: max = {ptps.max():.4f} cycles")
    print(f"  worst orbit: p_o={p_o[ok][i_max]:.6g} s, "
          f"sin_i={sin_i[ok][i_max]:.4g}, A_T={A_truth[ok][i_max]:.4f} rad")

    # Deterministic worst case: the far corner of a grid cell, on an orbit at
    # the largest covered x (the largest levers), with the phase chosen to align
    # the three contributions.  A true point is at most half a spacing from its
    # grid point, so this is the adversarial extreme random sampling will not
    # hit.  A covered orbit near x_max can snap to the padded grid top x[-1], so
    # the levers are sized at x[-1] (the grid does the same).
    d_w, d_x, d_A = spec["spacings"]
    x_lever = max(x_covered_max, x[-1])
    a = (d_x / 2.0) / p0
    b = x_lever * (d_A / 2.0) / p0
    # A_T is anchored at the observation START, so the omega axis sweeps the
    # full [0, T] and its contribution to the phase is d_w*t, t in [0, T].
    c = x_lever * (d_w / 2.0) * T / p0
    corner_lin = np.sqrt(a**2 + (b + c) ** 2)   # sup over t and phase
    # Direct (nonlinear) worst case at the cell corner: truth at x_lever, grid
    # point half a spacing away on each axis, A_T chosen to maximise the phase
    # excursion.  Scan A_T and the absolute time t in [0, T].
    t = np.linspace(0.0, T, 20001)
    x_t = x_lever
    x_g = x_t - d_x / 2.0
    w_t = whi                                  # omega_b at the top of the range
    w_g = w_t - d_w / 2.0
    corner_nl = 0.0
    for A_t in np.linspace(0.0, 2 * np.pi, 1441):
        A_g = A_t - d_A / 2.0
        R = (x_t / p0) * np.sin(w_t * t + A_t) \
            - (x_g / p0) * np.sin(w_g * t + A_g)
        corner_nl = max(corner_nl, np.max(np.abs(R)))
    # Random cell-corner scan (midpoints between adjacent grid lines), which
    # hits corners the uniform MC above essentially never samples.
    rngc = np.random.default_rng(12345)
    Wc = 0.5 * (omega_b[:-1] + omega_b[1:])
    Xc = 0.5 * (x[:-1] + x[1:])
    Ac = (A_T + d_A / 2.0) % (2 * np.pi)
    corner_scan = 0.0
    for _ in range(4000):
        wt = rngc.choice(Wc)
        xt = rngc.choice(Xc)
        if not (xlo - 1e-9 <= xt <= x_covered_max + 1e-9):
            continue
        At = rngc.choice(Ac)
        iw = int(np.argmin(np.abs(omega_b - wt)))
        ix = int(np.argmin(np.abs(x - xt)))
        iA = int(np.argmin(np.abs(A_T - At)))
        R = (xt / p0) * np.sin(wt * t + At) \
            - (x[ix] / p0) * np.sin(omega_b[iw] * t + A_T[iA])
        corner_scan = max(corner_scan, np.max(np.abs(R)))

    print(f"\nadversarial cell corners:")
    print(f"  a={a:.4e}  b={b:.4e}  c={c:.4e}")
    print(f"  linear bound sqrt(a^2+(b+c)^2) = {corner_lin:.4e} cycles")
    print(f"  direct nonlinear (x_max cell) = {corner_nl:.4e} cycles")
    print(f"  random cell-corner scan       = {corner_scan:.4e} cycles")

    passed = (devs.max() <= tol * (1 + 1e-9)
              and corner_lin <= tol * (1 + 1e-9)
              and corner_nl <= tol * (1 + 1e-9)
              and corner_scan <= tol * (1 + 1e-9))
    print(f"\nphase_tol never exceeded (MC, linear bound, corners): {passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
