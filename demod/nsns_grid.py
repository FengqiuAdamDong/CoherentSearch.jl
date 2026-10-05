#!/usr/bin/env python
"""
NS-NS accel/jerk/snap demodulation grid for a single observation.

A copy of FFA_stacking/orbit_simulation/kinematic_grid_spacing.py specialised
to a double neutron-star binary on an orbit from 30 minutes out to 10 days,
searched over ONE PRESTO .fft/.inf observation (not a stacked campaign).

Differences from the original:

  * Companion/pulsar masses default to 1.4/1.4 Msun (NS-NS), not Sgr A*.
  * Orbital periods default to a log-spaced ladder over [30 min, 10 d].
  * The observation length T = N*dt is read from the .inf (-t_obs overrides);
    that same T is both the campaign span and the single-observation length,
    so the two parameters of the original collapse into one.
  * Everything is governed by -phase_tol_cycles, the total accumulated
    phase drift allowed across the observation:
      - a segment is FEASIBLE only if its own best-fit cubic already
        accumulates less than that budget (no grid density can fix a larger
        intrinsic truncation);
      - -drop_pct discards the worst-phase-error fraction of the feasible
        segments;
      - the template spacings are chosen so a within-cell offset adds no more
        integrated phase than the budget left after the worst kept segment.
    The instantaneous -n_bin FFA-bin budget of the original is gone; the
    integrated (PHASE_LEVERS) lever arms are used throughout.
  * -max_accel/-max_jerk/-max_snap cap the searched (a, j, s); defaults are
    derived from the shortest orbit in the scan, just covering the tightest
    NS-NS binary requested.

    v_model(tau) = v0 + a*tau + (j/2)*tau**2 + (s/6)*tau**3
    phase(tau)   = (1/(C*p0)) * integral v_model-v_true dtau, |tau| <= T/2
    budget       = phase_tol_cycles (cycles)
    spacing_x    = 2*eps/PHASE_LEVER_x(T),  eps = leftover / n_active_axes
    ranges       = best-fit (a,j,s) min/max over kept segments

Besides writing the human-readable report, this emits machine-readable outputs
for the demod sweep (demod/run_nsns_sweep.sh): a YAML spec (ranges, spacings,
active, caps, tolerances) and a CSV listing every concrete (accel, jerk, snap)
trial, one per line.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import get_context

# The orbit/physics helpers live in the sibling FFA_stacking repo.  Point at
# it with $FFA_REPO (default is the checkout used here).
_FFA_REPO = os.environ.get("FFA_REPO", "/home/fadong/Documents/FFA_stacking")
_ORBIT_SIM = os.path.join(_FFA_REPO, "orbit_simulation")
if _ORBIT_SIM not in sys.path:
    sys.path.insert(0, _ORBIT_SIM)

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import cumulative_trapezoid

from simulate_orbit_accel_jerk import (
    C,
    G,
    SOLAR_MASS,
    YEAR_S,
    los_acceleration,
    los_jerk,
    los_snap,
    los_velocity,
    observed_period,
)

# Phase-connection lever arms: max over |tau| <= T/2 of |integral tau'**k/k! dtau'|,
# k = 1, 2, 3 (accel, jerk, snap).  Midpoint + even k (accel, snap): the
# integrand is odd, extremum at tau=0 (half the naive T**(k+1)/(k+1)!); odd k
# (jerk): endpoint.
PHASE_LEVERS = {
    "start": lambda T: np.array([T**2 / 2.0, T**3 / 6.0, T**4 / 24.0]),
    "midpoint": lambda T: np.array([T**2 / 8.0, T**3 / 24.0, T**4 / 384.0]),
}
AXIS_NAMES = ["accel", "jerk", "snap"]
AXIS_UNITS = ["m/s^2", "m/s^3", "m/s^4"]
FACTORIALS = [1.0, 2.0, 6.0]  # k! for tau**k/k! basis

# NS-NS defaults: two ~1.4 Msun neutron stars.
NS_MASS = 1.4


def semi_major_axis_sini(p_o_s, sin_i, companion_mass_msun, pulsar_mass_msun=0.0):
    """Pulsar's projected semi-major axis a_p*sin(i) [m] for a two-body orbit.

    The upstream helper treats `companion_mass_msun` as the TOTAL mass and
    returns a_total*sin(i), which is only right when the companion dominates
    (Sgr A*).  For nearly equal masses it overstates the pulsar's orbit by the
    total/companion factor (2x for NS-NS).  Here the companion/pulsar split is
    explicit: a_p = a_total * m_c/M, a_total from Kepler III with M = m_c+m_p.
    pulsar_mass_msun=0 reproduces the upstream companion-dominates form."""
    M = companion_mass_msun + pulsar_mass_msun
    a_tot = (G * M * SOLAR_MASS / (4 * np.pi ** 2) * p_o_s ** 2) ** (1 / 3)
    return sin_i * a_tot * (companion_mass_msun / M)


def read_inf(path):
    """(N, dt, epoch_mjd) from a PRESTO .inf, via infodata if importable,
    else the raw key=value lines (only the three keys we need)."""
    try:
        import infodata
        d = infodata.infodata(path)
        return int(d.N), float(d.dt), float(d.epoch)
    except Exception:
        N = dt = epoch = None
        with open(path) as fh:
            for line in fh:
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key.startswith("Number of bins"):
                    N = int(val)
                elif key.startswith("Width of each time series bin"):
                    dt = float(val)
                elif key.startswith("Epoch of observation"):
                    epoch = float(val)
        if N is None or dt is None:
            raise ValueError(f"{path}: could not read N/dt")
        return N, dt, epoch


def peak_kinematics(p_min_s, e=0.0, sin_i=1.0, companion_mass=NS_MASS,
                    pulsar_mass=NS_MASS, n_omega=36, n_t=4000):
    """Peak |accel|, |jerk|, |snap| of a Keplerian orbit over one period,
    maximised over mean anomaly and argument of periastron.  Eccentric orbits
    peak far above the circular K*omega**k, so the true derivatives are
    sampled (central differences, as simulate_orbit_accel_jerk) rather than
    using the circular closed form.  Natural cap defaults for an NS-NS sweep."""
    omega_b = 2 * np.pi / p_min_s
    a_psini = semi_major_axis_sini(p_min_s, sin_i, companion_mass, pulsar_mass)
    t = np.linspace(0.0, p_min_s, n_t)
    best = [0.0, 0.0, 0.0]
    fns = (los_acceleration, los_jerk, los_snap)
    for omega_peri in np.linspace(0.0, 2 * np.pi, n_omega, endpoint=False):
        vals = [fn(t, omega_b, a_psini, 0.0, e, omega_peri) for fn in fns]
        best = [max(b, float(np.max(np.abs(v)))) for b, v in zip(best, vals)]
    return tuple(best)


def terms_mask(terms):
    """CLI -terms values -> [accel, jerk, snap] booleans."""
    return [n in terms or "all" in terms for n in AXIS_NAMES]


def terms_label(mask):
    """'accel+jerk+snap' style label for the active model terms."""
    return "+".join(n for n, m in zip(AXIS_NAMES, mask) if m)


def true_velocity(t, args, omega_b, a_psini, e, omega_peri):
    """True-observed-period LOS velocity: c*(P_true/p0 - 1). With -gr includes
    Einstein/Shapiro/periastron-advance + intrinsic pdot (fit absorbs as pseudo-accel)."""
    p_true = observed_period(
        t, args.p0, args.pdot, omega_b, a_psini, args.A_T, e,
        omega_peri, m_c=args.companion_mass, m_p=args.pulsar_mass,
        gr=args.gr,
    )
    return C * (p_true / args.p0 - 1.0)


_SPAN_GRID_CACHE = {}


def _span_grid(span_s, nsamp):
    """Cached np.linspace(0, span_s, nsamp): identical values every call
    within a run, so the allocation is paid once (see fit_velocity_model)."""
    key = (float(span_s), int(nsamp))
    g = _SPAN_GRID_CACHE.get(key)
    if g is None:
        g = np.linspace(0.0, span_s, nsamp)
        _SPAN_GRID_CACHE[key] = g
    return g


def fit_velocity_model(t0, span_s, nsamp, args, omega_b, a_psini, e,
                       omega_peri, anchor, mask):
    """Best LSQ velocity fit over one span (v0 + mask-selected tau**k/k! terms),
    anchored per `anchor`.  Returns (max |instantaneous resid|, fitted (a,j,s)
    with zeros for off terms, raw resid + tau axis from the anchor).  The
    integrated phase error is `integrated_trunc(resid, tau)`."""
    t = t0 + _span_grid(span_s, nsamp)
    v = true_velocity(t, args, omega_b, a_psini, e, omega_peri)
    t_anchor = t0 if anchor == "start" else t0 + span_s / 2.0
    tau = t - t_anchor
    # v0 (column of ones) is always fitted: the FFA period axis absorbs it.
    cols = [np.ones_like(tau)]
    cols += [tau ** (x + 1) / FACTORIALS[x] for x in range(3) if mask[x]]
    A = np.column_stack(cols)
    scale = np.max(np.abs(A), axis=0)  # tau**3 ~ 1e18: conditioning
    coef = np.linalg.lstsq(A / scale, v, rcond=None)[0] / scale
    resid = v - A @ coef
    ajs = np.zeros(3)
    ajs[np.flatnonzero(mask)] = coef[1:]
    trunc = np.max(np.abs(resid))
    return trunc, tuple(ajs), resid, tau


def integrated_trunc(resid_v, tau):
    """Path-length error (m) accumulated over the span: max |integral resid dtau|.
    Divide by C*p0 for cycles."""
    return float(np.max(np.abs(np.concatenate(
        ([0.0], cumulative_trapezoid(resid_v, tau))))))


def _scan_one_orbit(task):
    """Pool worker for scan_segments: fits all t0s of one orbit. Module scope
    so spawn-context Pool can pickle."""
    (span_s, nsamp, args, p_o, p_o_s, omega_b, sin_i, a_psini, e, omega_peri,
     t0s) = task
    recs = []
    for t0 in t0s:
        trunc, (a, j, s), resid, tau = fit_velocity_model(
            t0, span_s, nsamp, args, omega_b, a_psini, e,
            omega_peri, args.anchor, args.term_mask)
        recs.append(dict(
            p_o=p_o, sin_i=sin_i, e=e, omega_peri=omega_peri,
            t0=t0, omega_b=omega_b, a_psini=a_psini,
            p_o_s=p_o_s, trunc=trunc, a=a, j=j, s=s,
            trunc_phase=integrated_trunc(resid, tau)))
    return recs


def scan_segments(span_s, args, p_o_values, sin_i_values, e_values,
                  omega_peri_values, n_phase, nsamp, nproc=None):
    """Fit every segment start phase over every (p_o, sin_i, e, omega_peri)
    combo.  Recs carry the instantaneous and integrated truncation and the
    best-fit (a,j,s).  Spawn-context pool (nproc=None = all cores, nproc=1 =
    serial) with BLAS pinned to 1 thread: forked workers inherit an already
    initialized multi-threaded BLAS pool and ignore env set after import ->
    oversubscription."""
    tasks = []
    for p_o in p_o_values:
        p_o_s = p_o * YEAR_S
        omega_b = 2 * np.pi / p_o_s
        t0s = np.linspace(0.0, p_o_s, n_phase, endpoint=False)
        for sin_i in sin_i_values:
            a_psini = semi_major_axis_sini(p_o_s, sin_i, args.companion_mass,
                                           args.pulsar_mass)
            for e in e_values:
                for omega_peri in omega_peri_values:
                    tasks.append((span_s, nsamp, args, p_o, p_o_s, omega_b,
                                 sin_i, a_psini, e, omega_peri, t0s))

    workers = nproc if nproc is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks)))
    # spawn re-execs __main__ from disk; `python -c`/heredoc have no file ->
    # serial fallback (else pool fails before any task).
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if workers == 1 or not (main_file and os.path.exists(main_file)):
        results = [_scan_one_orbit(t) for t in tasks]
    else:
        thread_vars = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        saved = {v: os.environ.get(v) for v in thread_vars}
        try:
            for v in thread_vars:
                os.environ[v] = "1"
            with get_context("spawn").Pool(workers) as pool:
                results = pool.map(_scan_one_orbit, tasks)
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old

    recs = []
    for r in results:
        recs.extend(r)
    return recs


def mark_feasible(recs, phase_tol_m):
    """Tag feasibility and seed 'kept' = feasible.  A segment is feasible iff
    its own best-fit cubic integrates to less than phase_tol_m over the
    observation; no grid density can phase-connect a larger intrinsic
    truncation, so those segments are not claimed."""
    for r in recs:
        r["feasible"] = bool(r["trunc_phase"] < phase_tol_m)
        r["kept"] = r["feasible"]
    return [r for r in recs if r["feasible"]]


def apply_drop(feasible_recs, drop_pct):
    """Drop the worst-phase-error drop_pct% of feasible segments, pooled over
    ALL orbits (global budget -> globally worst dropped).  Dropped stay
    'feasible' but lose 'kept'.  >= 1 survives."""
    order = sorted(feasible_recs, key=lambda r: r["trunc_phase"])  # cheapest first
    n_keep = max(int(round(len(order) * (1.0 - drop_pct / 100.0))), 1)
    for r in order[n_keep:]:
        r["kept"] = False
    return order[:n_keep]


def allocate_spacings(budget_m, levers, ranges, mask):
    """Equal-eps split of the integrated-phase budget over axes spanning >1
    cell; saturated axes (range fits one cell) = 1 trial, (range/2)*lever fixed
    cost.  Off axes (mask): no spacing, no cost.  Returns (spacings, eps,
    active) or None."""
    active = list(mask)
    for _ in range(4):  # <= 3 saturation passes
        n_act = sum(active)
        fixed = sum((ranges[x][1] - ranges[x][0]) / 2.0 * levers[x]
                    for x in range(3) if mask[x] and not active[x])
        if budget_m - fixed <= 0:
            return None
        eps = (budget_m - fixed) / n_act if n_act else 0.0
        # saturated: spacing = full range (1 trial, worst offset range/2)
        spac = np.array([
            2.0 * eps / levers[x] if active[x] else
            (ranges[x][1] - ranges[x][0]) if mask[x] else 0.0
            for x in range(3)
        ])
        newly_sat = [x for x in range(3) if active[x]
                     and (ranges[x][1] - ranges[x][0]) <= 2.0 * eps / levers[x]]
        if not newly_sat:
            eps_arr = np.array([
                eps if active[x] else
                (ranges[x][1] - ranges[x][0]) / 2.0 * levers[x] if mask[x]
                else 0.0
                for x in range(3)
            ])
            return spac, eps_arr, active
        for x in newly_sat:
            active[x] = False
    return None


def coeff_ranges(kept_recs):
    """(min, max) best-fit a/j/s over kept segments = points grid must cover."""
    out = []
    for key in ("a", "j", "s"):
        vals = np.array([r[key] for r in kept_recs])
        out.append((float(np.min(vals)), float(np.max(vals))))
    return out


def n_templates(ranges, spacings, active, mask):
    """Template count; saturated/off axes = 1 trial."""
    n = 1
    for x in range(3):
        if mask[x] and active[x]:
            n *= int(np.ceil((ranges[x][1] - ranges[x][0]) / spacings[x])) + 1
    return n


def corner_feasible_fraction(groups, args, out_path):
    """Corner plot of % of each orbit kept (feasible, not dropped; start phases
    uniform in orbital time).  Diagonal: 1D marginals (mean over other
    params); lower triangle: 2D projections, cells averaged over hidden
    params."""
    axes_meta = [("p_o", args.p_o, "p_o [yr]"),
                 ("sin_i", args.sin_i, "sin_i"),
                 ("e", args.e, "e"),
                 ("omega_peri", args.omega_peri, r"$\omega_{peri}$ [rad]")]
    vals = [np.unique(m[1]) for m in axes_meta]
    labels = [m[2] for m in axes_meta]
    n = len(axes_meta)
    frac = {k: 100.0 * sum(r["kept"] for r in g) / len(g)
            for k, g in groups.items()}

    fig, axs = plt.subplots(n, n, figsize=(2.9 * n, 2.9 * n))
    im = None
    for i in range(n):
        for j in range(n):
            ax = axs[i][j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ys = [np.mean([frac[k] for k in frac if k[i] == v])
                      for v in vals[i]]
                ax.bar(range(len(vals[i])), ys, color="C0")
                ax.set_xticks(range(len(vals[i])))
                ax.set_xticklabels([f"{v:g}" for v in vals[i]],
                                   rotation=45, fontsize=7)
                ax.set_ylim(0, 100)
                ax.tick_params(axis="y", labelsize=7)
            else:
                grid = np.full((len(vals[i]), len(vals[j])), np.nan)
                for ri, vy in enumerate(vals[i]):
                    for ci, vx in enumerate(vals[j]):
                        sel = [frac[k] for k in frac
                               if k[i] == vy and k[j] == vx]
                        if sel:
                            grid[ri, ci] = np.mean(sel)
                im = ax.imshow(grid, origin="lower", aspect="auto",
                               vmin=0, vmax=100, cmap="viridis")
                for ri in range(len(vals[i])):
                    for ci in range(len(vals[j])):
                        if not np.isnan(grid[ri, ci]):
                            ax.text(ci, ri, f"{grid[ri, ci]:.0f}",
                                    ha="center", va="center", fontsize=6,
                                    color="w" if grid[ri, ci] < 60 else "k")
                ax.set_xticks(range(len(vals[j])))
                ax.set_xticklabels([f"{v:g}" for v in vals[j]],
                                   rotation=45, fontsize=7)
                ax.set_yticks(range(len(vals[i])))
                ax.set_yticklabels([f"{v:g}" for v in vals[i]], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=9)
    axs[0][0].set_ylabel("% searchable", fontsize=9)

    fig.suptitle(
        f"Percentage of orbit kept within {args.phase_tol_cycles:g} phase "
        f"cycles (T = {args.t_obs:g} s, model v0+{terms_label(args.term_mask)}, "
        f"drop_pct={args.drop_pct:g}%)"
        f"\ndiagonal = 1D marginal, "
        f"lower = 2D projection (mean over hidden params)", fontsize=11)
    if im is not None:
        cax = fig.add_axes([0.60, 0.55, 0.02, 0.28])
        fig.colorbar(im, cax=cax, label="% of orbit kept")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def claimed_fraction_plot(recs, args, out_path):
    """Headline: the percentage of each orbital period's phase space the grid
    claims (kept = feasible after the phase gate, -drop_pct and the caps).
    Mean and worst over the scanned e/sin_i/omega_peri at each p_o, with the
    overall kept fraction marked."""
    # Fraction of phase claimed per full orbit, then mean/worst over the
    # orbits sharing each p_o.
    orbit_frac = defaultdict(lambda: [0, 0])
    for r in recs:
        key = (r["p_o"], r["sin_i"], r["e"], r["omega_peri"])
        orbit_frac[key][0] += bool(r["kept"])
        orbit_frac[key][1] += 1
    by_po = defaultdict(list)
    for (p, _si, _e, _w), (k, n) in orbit_frac.items():
        by_po[p].append(100.0 * k / n)
    p_o = np.array(sorted(by_po))                      # years
    mean_pct = np.array([np.mean(by_po[p]) for p in p_o])
    worst_pct = np.array([np.min(by_po[p]) for p in p_o])
    overall = 100.0 * np.mean([r["kept"] for r in recs])

    days = p_o * 365.25
    x = np.arange(len(p_o))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(p_o) + 3.0), 4.6))
    ax.bar(x - 0.2, mean_pct, width=0.4, color="C0", label="mean over e, sin_i, $\\omega$")
    ax.bar(x + 0.2, worst_pct, width=0.4, color="C1", label="worst orbit (e, sin_i, $\\omega$)")
    ax.axhline(overall, color="k", ls="--", lw=1,
               label=f"overall kept = {overall:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.3g}" for d in days], rotation=45, fontsize=8)
    ax.set_xlabel("Orbital period $p_o$ [days]")
    ax.set_ylabel("% of phase claimed by the grid")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Grid coverage vs orbital period (T = {args.t_obs:g} s, "
        f"{args.phase_tol_cycles:g} cyc budget, drop_pct={args.drop_pct:g}%, "
        f"model v0+{terms_label(args.term_mask)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# NS-NS default orbital-period ladder: 30 minutes out to 10 days, in years.
_NSNS_PO_MIN_D = 30.0 / 1440.0
_NSNS_PO_MAX_D = 10.0
_DEFAULT_PO = list(np.geomspace(_NSNS_PO_MIN_D, _NSNS_PO_MAX_D, 12) / 365.25)


def build_parser():
    """NS-NS CLI.  Split from main() so callers can read defaults off
    build_parser().parse_args([]) instead of restating them (drift)."""
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-inf", type=str, default=None,
                    help="PRESTO .inf of the observation to be demodulated. "
                         "Sets -t_obs from N*dt unless given. Required unless "
                         "-t_obs is given explicitly")
    ap.add_argument("-p0", type=float, default=1.0,
                    help="Spin period [s]; use the MINIMUM of the search range "
                         "(phase error ~ 1/p0, so small p0 is the strict case)")
    ap.add_argument("-pdot", type=float, default=1e-15,
                    help="Intrinsic period derivative [s/s], included in the truth")
    ap.add_argument("-p_o", type=float, nargs="+", default=_DEFAULT_PO,
                    help="Orbital periods to scan [yr]; worst case over all of "
                         "them is used. Default is 12 log-spaced points from "
                         "30 min to 10 d (NS-NS range)")
    ap.add_argument("-e", type=float, nargs="+",
                    default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                    help="Eccentricities to scan; worst case over all of them "
                         "is used (0.0 = circular included)")
    ap.add_argument("-A_T", type=float, default=0.0,
                    help="Mean anomaly at t=0 [rad] (phase coverage comes from "
                         "scanning segment start times over a full orbit)")
    ap.add_argument("-omega_peri", type=float, nargs="+",
                    default=[0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2],
                    help="Arguments of periastron to scan [rad]; worst case over "
                         "all of them is used. Which part of the orbit is "
                         "searchable depends on this")
    ap.add_argument("-sin_i", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="sin(inclination) values to scan; worst case over all "
                         "of them is used.")
    ap.add_argument("-companion_mass", type=float, default=NS_MASS,
                    help="Companion mass [Msun] (default 1.4, NS-NS)")
    ap.add_argument("-pulsar_mass", type=float, default=NS_MASS,
                    help="Pulsar mass [Msun] (default 1.4, NS-NS)")
    ap.add_argument("-no_gr", dest="gr", action="store_false",
                    help="Disable GR in the truth (default: GR on)")
    ap.add_argument("-t_obs", type=float, default=None,
                    help="Observation length T [s]; both the span the cubic "
                         "must track and the periodogram baseline. Default: "
                         "N*dt from -inf")
    ap.add_argument("-max_accel", type=float, default=None,
                    help="Maximum |acceleration| [m/s^2] to search. Segments "
                         "whose best-fit accel exceeds it are not claimed. "
                         "Default: peak of the shortest orbit in -p_o")
    ap.add_argument("-max_jerk", type=float, default=None,
                    help="Maximum |jerk| [m/s^3] to search (see -max_accel)")
    ap.add_argument("-max_snap", type=float, default=None,
                    help="Maximum |snap| [m/s^4] to search (see -max_accel)")
    ap.add_argument("-phase_tol_cycles", type=float, default=0.1,
                    help="Total accumulated phase drift [cycles] allowed across "
                         "the observation.  A segment whose own best-fit cubic "
                         "already integrates to >= this is infeasible (no grid "
                         "density helps).  The remainder after the worst kept "
                         "segment is split over the active axes to set the "
                         "template spacings")
    ap.add_argument("-drop_pct", type=float, default=10.0,
                    help="Percentage of the FEASIBLE segments to discard, "
                         "worst-phase-error first, before the budget is split. "
                         "The worst segment alone sets the phase budget, so a "
                         "fraction of a percent here can cut the template count "
                         "by orders of magnitude; the price is not claiming "
                         "those orbital phases. 0 = claim everything feasible")
    ap.add_argument("-terms", nargs="+", default=["all"],
                    choices=("accel", "jerk", "snap", "all"),
                    help="Which kinematic terms the model carries and grids: "
                         "any subset of accel/jerk/snap, or 'all' for all "
                         "three (e.g. -terms accel jerk). Terms left out are "
                         "not fitted and not gridded, so the orbit curvature "
                         "they would have absorbed becomes truncation error")
    ap.add_argument("-anchor", choices=("midpoint", "start"), default="midpoint",
                    help="Where the polynomial is anchored within the observation. "
                         "'midpoint' is the convention everywhere (the demodulation "
                         "resampler's anchor) and gives spacings 2x/4x/8x coarser "
                         "than the legacy 'start', kept only for comparison")
    ap.add_argument("-n_phase", type=int, default=120,
                    help="Segment start phases scanned over one orbit")
    ap.add_argument("-nsamp", type=int, default=4000,
                    help="Time samples per span for the fits")
    ap.add_argument("-n_mc", type=int, default=300,
                    help="Monte Carlo mismatch draws for validation")
    ap.add_argument("-outdir", type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Base output directory")
    ap.add_argument("-outstem", type=str, default="nsns_grid",
                    help="Stem for the -grid_yaml/-grid_csv outputs")
    ap.add_argument("-plot_subdir", type=str, default="grid_size_plots",
                    help="Subdirectory of -outdir for plots (created if "
                         "missing), to avoid cluttering the cwd")
    return ap


def grids_from_spec(spec):
    """Concrete trial arrays from a spec.  Saturated axis (range fits one cell)
    collapses to a single trial at the range midpoint, matching
    allocate_spacings costing."""
    grids = []
    for x in range(3):
        lo, hi = spec["ranges"][x]
        if not spec["active"][x]:
            grids.append(np.array([0.5 * (lo + hi)]))
            continue
        D = spec["spacings"][x]
        n = int(np.ceil((hi - lo) / D)) + 1
        grids.append(lo + D * np.arange(n))
    return grids


def write_grid_outputs(args, spec):
    """Write the machine-readable grid: YAML spec + CSV of every concrete
    (accel, jerk, snap) trial."""
    os.makedirs(args.outdir, exist_ok=True)
    yaml_path = os.path.join(args.outdir, f"{args.outstem}.yaml")
    csv_path = os.path.join(args.outdir, f"{args.outstem}.csv")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False)

    grids = grids_from_spec(spec)
    n_tot = int(np.prod([len(g) for g in grids]))
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["accel", "jerk", "snap"])
        for a in grids[0]:
            for j in grids[1]:
                for s in grids[2]:
                    w.writerow([f"{a:.10g}", f"{j:.10g}", f"{s:.10g}"])
    print(f"\nWrote grid spec -> {yaml_path}")
    print(f"Wrote {n_tot} trial(s) -> {csv_path}")
    return yaml_path, csv_path


def apply_caps(recs, caps):
    """Mark a segment infeasible if any best-fit |coefficient| exceeds its cap.
    caps is [max_accel, max_jerk, max_snap] (None = no cap on that axis)."""
    for r in recs:
        if r.get("feasible", True) and any(
                cap is not None and abs(r[key]) > cap
                for key, cap in zip(("a", "j", "s"), caps)):
            r["feasible"] = False
            r["kept"] = False
    return [r for r in recs if r["feasible"]]


def main():
    ap = build_parser()
    args = ap.parse_args()

    if not 0.0 <= args.drop_pct < 100.0:
        ap.error("-drop_pct must be in [0, 100)")

    # Observation length from the .inf unless overridden.
    inf_N = inf_dt = None
    if args.inf is not None:
        inf_N, inf_dt, _epoch = read_inf(args.inf)
        if args.t_obs is None:
            args.t_obs = inf_N * inf_dt
    if args.t_obs is None:
        ap.error("give -inf, or set -t_obs explicitly")

    args.term_mask = terms_mask(args.terms)
    model_lbl = terms_label(args.term_mask)

    span_s = args.t_obs
    n_orbits = (len(args.p_o) * len(args.sin_i) * len(args.e)
                * len(args.omega_peri))
    phase_tol_m = args.phase_tol_cycles * C * args.p0

    # Natural NS-NS caps: the peak kinematics of the shortest orbit in the
    # scan (over all scanned sin_i), just covering the tightest binary asked
    # for.  A user cap tightens them.
    p_min_s = min(args.p_o) * YEAR_S
    a_pk, j_pk, s_pk = peak_kinematics(
        p_min_s, max(args.e), max(args.sin_i),
        args.companion_mass, args.pulsar_mass)
    caps = [
        args.max_accel if args.max_accel is not None else a_pk,
        args.max_jerk if args.max_jerk is not None else j_pk,
        args.max_snap if args.max_snap is not None else s_pk,
    ]

    def rng(vals):
        return (f"{min(vals):g}" if len(vals) == 1
                else f"{min(vals):g}-{max(vals):g} ({len(vals)})")

    print(f"Orbit grid ({n_orbits} orbits): p_o = {rng(args.p_o)} yr, "
          f"e = {rng(args.e)}, sin_i = {rng(args.sin_i)}, "
          f"omega_peri = {rng(args.omega_peri)} rad, "
          f"mass = {args.pulsar_mass:g}+{args.companion_mass:g} Msun, "
          f"GR = {'on' if args.gr else 'off'}")
    if inf_N is not None:
        print(f"Observation: {args.inf} (N = {inf_N}, dt = {inf_dt:g} s)")
    print(f"T = {args.t_obs:.6g} s ({args.t_obs / 86400.0:.6g} d), "
          f"p0 = {args.p0} s")
    print(f"Model terms: v0 + {model_lbl} "
          f"({sum(args.term_mask)} gridded axis/axes)")
    print(f"Caps: |accel| <= {caps[0]:.4g} m/s^2, |jerk| <= {caps[1]:.4g} m/s^3, "
          f"|snap| <= {caps[2]:.4g} m/s^4")
    print(f"Phase budget: {args.phase_tol_cycles:g} cycles "
          f"({phase_tol_m:.4g} m integrated) over the observation "
          f"(anchor: {args.anchor})")
    print(f"Marginal-segment trade: -drop_pct {args.drop_pct:g}% of the feasible "
          f"segments discarded (worst phase error first), pooled over all orbits")
    print()

    # ---- Stage 1: scan orbit grid x phase; phase gate + -drop_pct + caps ----
    recs = scan_segments(span_s, args, args.p_o, args.sin_i, args.e,
                         args.omega_peri, args.n_phase, args.nsamp)
    mark_feasible(recs, phase_tol_m)
    n_feasible_pre = sum(1 for r in recs if r["feasible"])
    feasible = apply_caps(recs, caps)
    if n_feasible_pre != len(feasible):
        print(f"Cap filter: {n_feasible_pre - len(feasible)} of "
              f"{n_feasible_pre} feasible segments dropped for exceeding the "
              f"(a,j,s) caps")

    print(f"{'e':>5} {'kept':>12} {'feasible':>12} "
          f"{'worst kept [cyc]':>18} {'worst phase [cyc]':>19}")
    for e in args.e:
        sub = [r for r in recs if r["e"] == e]
        k_e = [r for r in sub if r["kept"]]
        f_e = [r for r in sub if r["feasible"]]
        wk = (f"{max(r['trunc_phase'] for r in k_e) / (C * args.p0):.4f}"
              if k_e else "-")
        wt = max(r["trunc_phase"] for r in sub) / (C * args.p0)
        print(f"{e:>5.2f} {len(k_e):>5d}/{len(sub):<6d} {len(f_e):>5d}/{len(sub):<6d} "
              f"{wk:>18} {wt:>19.4f}")

    print(f"\nFeasible: {len(feasible)}/{len(recs)} segment start phases "
          f"({100.0 * len(feasible) / len(recs):.1f}% of orbital phase) can be "
          f"phase-connected within {args.phase_tol_cycles:g} cycles; the rest "
          f"are infeasible pieces of their orbit")

    if not feasible:
        print(f"\nINFEASIBLE: not one segment of any scanned orbit can be "
              f"phase-connected within {args.phase_tol_cycles:g} cycles over "
              f"T = {args.t_obs:.6g} s, so there is no grid to report: the "
              f"cubic truncation is irreducible.  Add terms to -terms or raise "
              f"-phase_tol_cycles.")
        return

    kept = apply_drop(feasible, args.drop_pct)
    per_orbit = defaultdict(lambda: [0, 0])
    for r in recs:
        c = per_orbit[(r["p_o"], r["sin_i"], r["e"], r["omega_peri"])]
        c[0] += bool(r["kept"])
        c[1] += 1
    worst_orbit = min(per_orbit.items(), key=lambda kv: kv[1][0] / kv[1][1])
    (wp, wsi, we, ww), (wk_n, wn) = worst_orbit
    print(f"Kept:     {len(kept)}/{len(recs)} "
          f"({100.0 * len(kept) / len(recs):.1f}% of orbital phase overall) "
          f"after dropping the worst {len(feasible) - len(kept)} feasible "
          f"segments (-drop_pct {args.drop_pct:g}%, pooled over all orbits)")
    print(f"          worst-covered orbit: {100.0 * wk_n / wn:.1f}% of its "
          f"phase (p_o = {wp:g} yr, e = {we:g}, sin_i = {wsi:g}, "
          f"omega_peri = {ww:.3f} rad) -- quote THIS as the per-orbit "
          f"coverage the grid below is claimed for")

    # ---- Stage 2: analytic spacings from the leftover phase budget ----
    worst = max(kept, key=lambda r: r["trunc_phase"])
    trunc_max = worst["trunc_phase"]
    print(f"\nPhase-connection truncation (best {model_lbl} fit vs truth), worst "
          f"over the {len(kept)} kept segments: "
          f"{trunc_max / (C * args.p0):.4f} cycles "
          f"(p_o = {worst['p_o']:g} yr, e = {worst['e']:g}, "
          f"sin_i = {worst['sin_i']:g}, omega_peri = {worst['omega_peri']:.3f} rad, "
          f"segment starts at orbital phase {worst['t0'] / worst['p_o_s']:.3f})")

    budget_m = phase_tol_m - trunc_max
    ranges = coeff_ranges(kept)
    alloc = allocate_spacings(budget_m, PHASE_LEVERS[args.anchor](span_s),
                               ranges, args.term_mask)
    if alloc is None:
        print("INFEASIBLE after accounting for saturated axes; raise "
              "-phase_tol_cycles or add -terms.")
        return
    spacings, eps_arr, active = alloc

    print(f"Phase budget for the grid: {budget_m / (C * args.p0):.4f} cycles "
          f"({budget_m:.4g} m), split equally over {sum(active)} active axes "
          f"({', '.join(n for n, a in zip(AXIS_NAMES, active) if a)})")
    print(f"Template ranges = best-fit coefficients over kept segments only")
    print()
    print(f"{'axis':>6} {'spacing':>12} {'unit':>7} {'range_lo':>12} {'range_hi':>12} "
          f"{'n_trials':>9} {'eps [cyc]':>10}")
    n_trials = []
    for x in range(3):
        lo_r, hi_r = ranges[x]
        if not args.term_mask[x]:
            n_trials.append(1)
            print(f"{AXIS_NAMES[x]:>6} {'off (-terms)':>12} {AXIS_UNITS[x]:>7} "
                  f"{'-':>12} {'-':>12} {1:>9d} {0.0:>10.3f}")
            continue
        n_x = int(np.ceil((hi_r - lo_r) / spacings[x])) + 1 if active[x] else 1
        n_trials.append(n_x)
        eps_cyc = eps_arr[x] / (C * args.p0)
        print(f"{AXIS_NAMES[x]:>6} {spacings[x]:>12.4e} {AXIS_UNITS[x]:>7} "
              f"{lo_r:>12.4e} {hi_r:>12.4e} {n_x:>9d} {eps_cyc:>10.4f}")
    print(f"\nTotal templates: {np.prod(n_trials):.3e}")

    # ---- Stage 3: end-to-end validation at the worst kept segment ----
    _, _, resid_v, tau = fit_velocity_model(
        worst["t0"], span_s, args.nsamp, args, worst["omega_b"],
        worst["a_psini"], worst["e"], worst["omega_peri"], args.anchor,
        args.term_mask)

    def total_cycles(d):
        quant_v = d[0] * tau + d[1] * tau ** 2 / 2.0 + d[2] * tau ** 3 / 6.0
        return integrated_trunc(resid_v + quant_v, tau) / (C * args.p0)

    half = spacings / 2.0
    corners = [(sa, sj, ss) for sa in (-1, 1) for sj in (-1, 1) for ss in (-1, 1)]
    corner_cyc = np.array([total_cycles(np.array(sgn) * half) for sgn in corners])
    mc_rng = np.random.default_rng(0)
    mc_cyc = np.array([
        total_cycles(mc_rng.uniform(-half, half)) for _ in range(args.n_mc)
    ])
    print(f"\nValidation at the worst kept segment (integrated phase error "
          f"[cycles], budget {args.phase_tol_cycles:g}):")
    print(f"  worst grid corner : {np.max(corner_cyc):.4f} cycles")
    print(f"  Monte Carlo (n={args.n_mc}), max : {np.max(mc_cyc):.4f}  "
          f"median : {np.median(mc_cyc):.4f} cycles")
    ok = np.max(corner_cyc) <= args.phase_tol_cycles * (1 + 1e-9)
    print(f"  guarantee holds: {ok}")

    spec = {
        "inf": os.path.abspath(args.inf) if args.inf else None,
        "p0": float(args.p0),
        "pdot": float(args.pdot),
        "t_obs": float(args.t_obs),
        "phase_tol_cycles": float(args.phase_tol_cycles),
        "drop_pct": float(args.drop_pct),
        "terms": model_lbl,
        "anchor": args.anchor,
        "companion_mass": float(args.companion_mass),
        "pulsar_mass": float(args.pulsar_mass),
        "gr": bool(args.gr),
        "p_o_yr": [float(x) for x in args.p_o],
        "e": [float(x) for x in args.e],
        "sin_i": [float(x) for x in args.sin_i],
        "omega_peri": [float(x) for x in args.omega_peri],
        "phase_tol_m": float(phase_tol_m),
        "trunc_max": float(trunc_max),
        "budget_m": float(budget_m),
        "caps": [float(x) for x in caps],
        "n_feasible": int(len(feasible)),
        "n_kept": int(len(kept)),
        "n_segments": int(len(recs)),
        "worst_corner_cycles": float(np.max(corner_cyc)),
        "guarantee_ok": bool(ok),
        "ranges": [[float(lo), float(hi)] for lo, hi in ranges],
        "spacings": [float(x) for x in spacings],
        "eps": [float(x) for x in eps_arr],
        "active": [bool(x) for x in active],
    }
    write_grid_outputs(args, spec)

    groups = defaultdict(list)
    for r in recs:
        groups[(r["p_o"], r["sin_i"], r["e"], r["omega_peri"])].append(r)
    plot_dir = os.path.join(args.outdir, args.plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    corner_path = os.path.join(plot_dir, "corner_feasible_fraction.png")
    corner_feasible_fraction(groups, args, corner_path)
    print(f"Saved corner plot of % orbit kept to {corner_path}")

    cover_path = os.path.join(plot_dir, "claimed_fraction.png")
    claimed_fraction_plot(recs, args, cover_path)
    print(f"Saved % phase claimed vs p_o to {cover_path}")


if __name__ == "__main__":
    main()
