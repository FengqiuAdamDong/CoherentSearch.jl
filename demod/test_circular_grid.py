#!/usr/bin/env python
"""Monte Carlo check of the circular-orbit grid's phase-tolerance guarantee.

The grid (nsns_grid.py -mode circular) claims: for any true circular orbit whose
(p_o, sin_i, A_T) lies in the covered ranges, the nearest grid point keeps the
residual SPIN phase within phase_tol cycles across the observation.  This test
Monte-Carlos that claim directly, with no linearisation:

  1. Derive the grid exactly as nsns_grid.py does (imported, so the two cannot
     drift) for a chosen T, p0, p_o/sin_i ranges and phase_tol.  The grid
     is one sub-grid per orbital-period band, and every residual is scored
     against the budget of the band the truth falls in.
  2. Sample N random circular orbits uniformly in the covered ranges:
     log-uniform p_o, uniform sin_i, uniform A_T in [0, 2*pi).
  3. Derive x = a_p sin(i)/c the same way the grid does.
  4. Snap to the nearest grid point of the truth's band: the nearest x row, then the nearest
     (omega_b, A_T) lattice point in that row in the L1 (diamond) metric --
     the rows are checkerboards, (i_omega + i_A) even.
  5. Compute the exact residual spin phase
         R(t) = Phi_true(t) - Phi_grid(t),
         Phi(t) = (x/p0) sin(omega_b (t - T/2) + A_T),   t in [0, T] (A_T at the midpoint),
     on a fine t grid, and record what phase_tol bounds: max_t |R| after the
     spin-frequency search absorbs a constant phase and a linear drift
     (nsns_grid.search_absorbed_max), or the raw max_t |R| under -circ_strict.
  6. PASS iff the worst residual over all trials is within its band's budget,
     AND so are every band's adversarial cell corners and a random scan of
     its diamond cell boundaries (which uniform sampling rarely hits), on the
     exact nonlinear R.

Usage:
    python demod/test_circular_grid.py [-inf OBS.inf | -t_obs T] [-p0 P0] \
        [-phase_tol_cycles 0.1] [-p_o ...] [-sin_i ...] [-n_mc 2000] \
        [-circ_strict] [--seed 0]

The circular grid drops nothing, so the whole (p_o, sin_i) box is sampled.
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
    for k in ("inf", "t_obs", "p0", "phase_tol_cycles",
              "companion_mass", "pulsar_mass", "circ_strict"):
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


def scored(R, tmid, strict):
    """The residual phase_tol bounds: raw max|R| (strict) or what is left
    after the spin search absorbs a constant + linear phase."""
    return np.max(np.abs(R)) if strict else float(ng.search_absorbed_max(R, tmid))


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
    ap.add_argument("-companion_mass", type=float, default=None)
    ap.add_argument("-pulsar_mass", type=float, default=None)
    ap.add_argument("-circ_strict", action="store_true", default=None,
                    help="score the raw max|R| (nsns_grid -circ_strict)")
    ap.add_argument("-n_mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    args = build_args(cli)
    spec, bands, orbits = ng.derive_circular_grid(args)
    p0, T = args.p0, args.t_obs
    sin_lo, sin_hi = min(args.sin_i), max(args.sin_i)
    p_lo, p_hi = bands[0]["p_lo_s"], bands[-1]["p_hi_s"]

    def snap(b, xt, wt, At):
        """Nearest row of band b, then the L1-nearest checkerboard point."""
        x = np.array([r[0] for r in b["rows"]])
        j = int(np.abs(x - xt).argmin())
        _, W, A = b["rows"][j]
        dw, dA, nA = W[1] - W[0] if len(W) > 1 else 1.0, A[1] - A[0], len(A)
        fw, fa = (wt - W[0]) / dw, (At % (2 * np.pi)) / dA
        best = None
        for iw in range(int(np.floor(fw)) - 1, int(np.floor(fw)) + 3):
            if not 0 <= iw < len(W):
                continue
            for ia in range(int(np.floor(fa)) - 1, int(np.floor(fa)) + 3):
                if (iw + ia) % 2:
                    continue
                d = abs(fw - iw) + abs(fa - ia)
                if best is None or d < best[0]:
                    best = (d, W[iw], A[ia % nA])
        return x[j], best[1], best[2]

    # Monte Carlo over the whole covered box; each residual is scored against
    # the budget of the period band the truth falls in, so every number below
    # is a fraction of ITS band's budget.
    rng = np.random.default_rng(cli.seed)
    n = cli.n_mc
    p_o = np.exp(rng.uniform(np.log(p_lo), np.log(p_hi), n))
    sin_i = rng.uniform(sin_lo, sin_hi, n)
    A_truth = rng.uniform(0.0, 2 * np.pi, n)
    x_truth = ng.circular_x_lt_s(p_o, sin_i, args.companion_mass, args.pulsar_mass)
    strict = bool(args.circ_strict)
    t = np.linspace(0.0, T, 20001)
    tmid = t - 0.5 * T
    frac = np.empty(n)
    for i in range(n):
        b = next(b for b in bands if b["p_lo_s"] <= p_o[i] <= b["p_hi_s"])
        xg, wg, Ag = snap(b, x_truth[i], 2 * np.pi / p_o[i], A_truth[i])
        wt = 2 * np.pi / p_o[i]
        R = (x_truth[i] / p0) * np.sin(wt * tmid + A_truth[i]) \
            - (xg / p0) * np.sin(wg * tmid + Ag)
        frac[i] = scored(R, tmid, strict) / b["tol"]

    print(f"Grid: {spec['n_trials']:.3e} trials, {len(bands)} period bands, "
          f"T = {T:g} s, p0 = {p0} s")
    what = "raw max|R|" if strict else "max|R| after the search absorbs const+linear"
    print(f"MC trials: {n}; {what} as a fraction of the truth's band "
          f"budget: mean {frac.mean():.3f}, max {frac.max():.4f}")

    # Adversarial cell corners, per band: truth at the band's top row edge
    # x_top + d_x/2 (the largest levers) and on the (omega, A_T) DIAMOND
    # boundary |dw/d_w| + |dA/d_A| = 1, which uniform sampling rarely hits.
    rngc = np.random.default_rng(12345)
    worst = frac.max()
    print(f"\n{'band [d]':>17} {'budget':>7} {'stretch':>17} {'nonlin':>8} {'scan':>8}"
          "   (fractions of the band budget)")
    for b in bands:
        tol, d_x = b["tol"], b["d_x"]
        x_top, W_top, A_top = b["rows"][-1]
        lever = x_top + d_x / 2.0
        d_w = W_top[1] - W_top[0] if len(W_top) > 1 else 0.0
        d_A = A_top[1] - A_top[0]
        # Nonlinear: half a step on every axis (a diamond edge point), top
        # omega of the band, A_T scanned to align the contributions.
        w_t = b["w_hi"]
        nl = 0.0
        for A_t in np.linspace(0.0, 2 * np.pi, 721):
            R = (lever / p0) * np.sin(w_t * tmid + A_t) \
                - (x_top / p0) * np.sin((w_t - d_w / 2) * tmid + A_t - d_A / 2)
            nl = max(nl, scored(R, tmid, strict))
        nl /= tol
        scan = 0.0
        for _ in range(4000):
            xg, W, A = b["rows"][rngc.integers(len(b["rows"]))]
            if len(W) < 2:
                continue
            iw = rngc.integers(len(W))
            iA = 2 * rngc.integers(len(A) // 2) + iw % 2     # on the checkerboard
            u = rngc.uniform(-1.0, 1.0)
            wt = W[iw] + rngc.choice((-1, 1)) * (1 - abs(u)) * (W[1] - W[0])
            if not b["w_lo"] <= wt <= b["w_hi"]:
                continue        # the truth must be an orbit of this band
            At = A[iA] + u * (A[1] - A[0])
            R = ((xg + d_x / 2) / p0) * np.sin(wt * tmid + At) \
                - (xg / p0) * np.sin(W[iw] * tmid + A[iA])
            scan = max(scan, scored(R, tmid, strict))
        scan /= tol
        worst = max(worst, nl, scan)
        st = "/".join(f"{v:.2f}" for v in b["stretch"])
        print(f"{b['p_lo_s']/86400:>8.4g}-{b['p_hi_s']/86400:<8.4g} {tol:>7.3g} "
              f"{st:>17} {nl:>8.4f} {scan:>8.4f}")

    passed = worst <= 1 + 1e-9
    print(f"\nevery band within its own phase budget (MC, corners): "
          f"{passed}  (worst {worst:.4f})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
