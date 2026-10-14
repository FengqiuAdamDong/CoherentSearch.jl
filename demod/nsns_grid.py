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

# Phase-connection lever arms: max over the span of |integral tau'**k/k! dtau'|,
# k = 1, 2, 3 (accel, jerk, snap), integrated from the start of the span
# (integrated_trunc's convention).  Midpoint anchor (tau in [-T/2, T/2]): odd k
# (accel, snap) has an odd integrand, so the integral is extremal at tau = 0;
# even k (jerk) at the endpoint.  That is 1/4, 1/4, 1/16 of the 'start' levers.
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


def orbit_phase_param_plots(groups, args, model_lbl, plot_dir):
    """One figure per (p_o, sin_i, e, omega_peri) orbit: the worst phase offset
    (integrated phase error [cycles], the phase-connection analogue of the
    reference's model-truth bin error) and the best-fit accel/jerk/snap, both vs
    segment start orbital phase.  Kept/dropped/infeasible start phases are marked
    as in the feasibility plots; the phase panel carries the -phase_tol_cycles
    budget line.  One plot per orbit combination -- the reference gates this
    behind -plot_orbits because large grids are slow, but here the records are
    already scanned."""
    cyc_per_m = 1.0 / (C * args.p0)
    print(f"\nWriting {len(groups)} per-orbit phase/parameter plots to "
          f"{plot_dir}/ ...")
    for (p_o, sin_i, e, omega_peri), grp in groups.items():
        p_o_s = grp[0]["p_o_s"]
        ph = np.array([r["t0"] / p_o_s for r in grp])
        cyc = np.array([r["trunc_phase"] for r in grp]) * cyc_per_m
        keep = np.array([r["kept"] for r in grp])
        dropped = np.array([r["feasible"] and not r["kept"] for r in grp])
        infeas = np.array([not r["feasible"] for r in grp])
        ajs = {n: np.array([r[k] for r in grp])
               for n, k in (("accel", "a"), ("jerk", "j"), ("snap", "s"))}
        styles = ((keep, ".", 6, "C0", "claimed"),
                  (dropped, "s", 4, "C1", f"dropped (-drop_pct {args.drop_pct:g}%)"),
                  (infeas, "x", 5, "C3", "infeasible (over budget)"))

        fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))

        # Panel 0: this orbit's integrated phase error vs segment start phase.
        ax = axes[0]
        for sel, mk, ms, col, lbl in styles:
            if sel.any():
                ax.plot(ph[sel], cyc[sel], mk, ms=ms, color=col, label=lbl)
        ax.axhline(args.phase_tol_cycles, color="k", ls="--", lw=1,
                   label=f"{args.phase_tol_cycles:g}-cycle budget")
        # infeasible overshoot orders of magnitude: log scale keeps the budget
        # line readable
        if cyc.max() > 5.0 * args.phase_tol_cycles and cyc.min() > 0:
            ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Segment start orbital phase")
        ax.set_ylabel("Integrated phase error [cycles]")
        ax.set_title("Worst phase offset\n(cubic truncation vs truth)")
        ax.legend(fontsize=8)

        # Panels 1-3: best-fit accel/jerk/snap vs segment start phase.
        for ax, name in zip(axes[1:], AXIS_NAMES):
            unit = AXIS_UNITS[AXIS_NAMES.index(name)]
            for sel, mk, ms, col, lbl in styles:
                if sel.any():
                    ax.plot(ph[sel], ajs[name][sel], mk, ms=ms, color=col,
                            label=lbl)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Segment start orbital phase")
            ax.set_ylabel(f"{name} [{unit}]")
            ax.set_title(f"Best-fit {name}")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Orbit: p_o={p_o:g} yr, e={e:g}, sin_i={sin_i:g}, "
            f"omega_peri={omega_peri:.3f} rad  |  p0={args.p0} s, "
            f"model=v0+{model_lbl}, anchor={args.anchor}, "
            f"GR={'on' if args.gr else 'off'}, T={args.t_obs:g} s",
            fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fname = (f"orbit_phase_po{p_o:g}_sini{sin_i:g}_e{e:g}"
                 f"_w{omega_peri:.2f}_{model_lbl.replace('+', '')}.png")
        fig.savefig(os.path.join(plot_dir, fname), dpi=130)
        plt.close(fig)
    print(f"Done: {len(groups)} phase/parameter plots in {plot_dir}/")


# NS-NS default orbital-period ladder: 2 minutes out to 24 hours, in years.
_NSNS_PO_MIN_D = 2.0 / 1440.0
_NSNS_PO_MAX_D = 1.0
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
                         "2 min to 24 h (NS-NS range)")
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
                         "those orbital phases. 0 = claim everything feasible. "
                         "ajs grid only: the circular grid drops nothing (its "
                         "cost knob is -phase_tol_cycles)")
    ap.add_argument("-terms", nargs="+", default=["all"],
                    choices=("accel", "jerk", "snap", "all"),
                    help="Which kinematic terms the model carries and grids: "
                         "any subset of accel/jerk/snap, or 'all' for all "
                         "three (e.g. -terms accel jerk). Terms left out are "
                         "not fitted and not gridded, so the orbit curvature "
                         "they would have absorbed becomes truncation error")
    ap.add_argument("-anchor", choices=("midpoint", "start"), default="midpoint",
                    help="Where the ajs polynomial is anchored within the "
                         "observation. 'midpoint' is the convention everywhere "
                         "(the demodulation resampler's anchor) and gives "
                         "spacings 4x/4x/16x coarser than the legacy 'start', "
                         "kept only for comparison (the demod always anchors "
                         "the polynomial at the midpoint). The circular grid ignores "
                         "this: its A_T is always the midpoint phase")
    ap.add_argument("-n_phase", type=int, default=120,
                    help="Segment start phases scanned over one orbit")
    ap.add_argument("-nsamp", type=int, default=4000,
                    help="Time samples per span for the fits")
    ap.add_argument("-n_mc", type=int, default=300,
                    help="Monte Carlo mismatch draws for validation")
    ap.add_argument("-mode", choices=("ajs", "circular", "hybrid"),
                    default="ajs",
                    help="Grid model. 'ajs' (default): the accel/jerk/snap "
                         "phase-connection grid. 'circular': pure Keplerian "
                         "circular-orbit grid (pb,x,at), no cubic-truncation "
                         "gate -- the demod is exact at the truth, so the grid "
                         "is the pure parameter-mismatch phase budget, uniform "
                         "in (omega_b, x, A_T). 'hybrid': circular below "
                         "-p_break and ajs at/above it, combining the exact "
                         "short-orbit circular search with the general cubic "
                         "search for long orbits; emits two CSVs, one per model")
    ap.add_argument("-circ_strict", action="store_true",
                    help="circular/hybrid: bound the RAW residual spin phase "
                         "max|R|. Default: bound what is left after the spin-"
                         "frequency search absorbs a constant phase and a "
                         "linear drift, which widens the steps of the bands "
                         "whose orbit is not much shorter than T. Use this if "
                         "the demodulated series is folded at a FIXED spin "
                         "frequency")
    ap.add_argument("-p_break", type=float, default=None,
                    help="hybrid only: orbital period [yr] at which to switch "
                         "from the circular grid (below) to the ajs grid "
                         "(at/above). Default: derived from -break_coverage as "
                         "the shortest scanned p_o such that every p_o at/above "
                         "it has mean e=0 ajs coverage >= -break_coverage (ajs "
                         "coverage dips near p_o ~ T_obs, so the switch clears "
                         "the whole inadequate band)")
    ap.add_argument("-break_coverage", type=float, default=60.0,
                    help="hybrid only: the ajs coverage [%%] used to place the "
                         "automatic p_break. Coverage is the mean over the "
                         "scanned e=0 sin_i x omega_peri combos of the fraction "
                         "of segment start phases that pass the phase gate and "
                         "caps (before -drop_pct). Ignored if -p_break is given")
    ap.add_argument("-outdir", type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Base output directory")
    ap.add_argument("-outstem", type=str, default="nsns_grid",
                    help="Stem for the <outstem>.{yaml,csv} outputs "
                         "(hybrid: <outstem>_circular.csv, <outstem>_ajs.csv)")
    ap.add_argument("-plot_subdir", type=str, default="grid_size_plots",
                    help="Subdirectory of -outdir for plots (created if "
                         "missing), to avoid cluttering the cwd")
    ap.add_argument("-plot", action="store_true",
                    help="Also write the one-figure-per-orbit phase-error and "
                         "best-fit accel/jerk/snap plots. Off by default: one "
                         "figure per orbit combination is slow for large grids")
    ap.add_argument("-no_write", action="store_true",
                    help="Derive and report the grid but write no CSV, YAML "
                         "or ajs plots (the small circular template-share plot "
                         "is still written). A dry run: useful when "
                         "the grid is so large that serialising it to CSV "
                         "dwarfs the derivation itself")
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


def write_grid_csv(csv_path, spec):
    """CSV of every concrete (accel, jerk, snap) trial.  Returns the total."""
    grids = grids_from_spec(spec)
    n_tot = int(np.prod([len(g) for g in grids]))
    with open(csv_path, "w", newline="") as fh:
        # LF, not csv.writer's default CRLF: the demod sweep builds each output
        # filename from these literal fields in bash, where a trailing \r would
        # ride along and break the match against the file Julia writes.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["accel", "jerk", "snap"])
        for a in grids[0]:
            for j in grids[1]:
                for s in grids[2]:
                    w.writerow([f"{a:.10g}", f"{j:.10g}", f"{s:.10g}"])
    return n_tot


_SKIP = object()  # sentinel: "do not write this file" (None means "use default")


def write_grid_outputs(args, spec, csv_path=None, yaml_path=None):
    """Write the machine-readable grid: CSV of every concrete (a,j,s) trial
    plus the YAML spec.  Paths default to <outdir>/<outstem>.{csv,yaml}; hybrid
    passes an explicit per-model CSV path and yaml_path=_SKIP so only the
    combined YAML is written."""
    if args.no_write:
        n_tot = int(np.prod([len(g) for g in grids_from_spec(spec)]))
        print(f"\n[no_write] grid not written ({n_tot} trial(s))")
        return yaml_path, csv_path
    os.makedirs(args.outdir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(args.outdir, f"{args.outstem}.csv")
    if yaml_path is None:
        yaml_path = os.path.join(args.outdir, f"{args.outstem}.yaml")
    if yaml_path is not _SKIP:
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False)
    n_tot = write_grid_csv(csv_path, spec)
    print(f"\nWrote {n_tot} trial(s) -> {csv_path}")
    if yaml_path is not _SKIP:
        print(f"Wrote grid spec -> {yaml_path}")
    return yaml_path, csv_path


# ---------------------------------------------------------------------------
# Circular-orbit mode
#
# The circular demod removes the exact non-polynomial Roemer track, so it is
# exact at the true parameters and there is NO cubic-truncation feasibility
# gate (every point is "feasible").  The only residual at a grid point is the
# parameter mismatch, and the grid is uniform in the coordinates that make up
# the LOS velocity V/c = omega_b*x*cos(omega_b*(t - T/2) + A_T), x = a_p sin(i)/c
# (A_T at the observation midpoint):
#
#     delta_v/c = (domega_b) x cos - (dA_T) x sin + (dx) omega_b cos
#
# The demod's coherent observable is the orbit-induced residual of the SPIN
# phase (the phase the fold accumulates), Phi(t) = (1/p0) int V/c dt
#     = (x/p0) sin(omega_b (t - T/2) + A_T)   [cycles, up to a tau-independent
# offset], where A_T is the mean anomaly at the observation MIDPOINT, t is the
# absolute time since the start, and V/c is the LOS Doppler.  A grid offset
# perturbs it by
#     dPhi(t) = a sin(phi) + (b + c*(2(t-T/2)/T)) cos(phi),
#     phi = omega_b (t - T/2) + A_T,
# with a, b, c the x, A_T and omega contributions (see derive_circular_grid).
# The JOINT bound is a^2 + (b+c)^2 <= phase_tol^2, whose volume-optimal split is
# a = phase_tol/sqrt(3), b = c = phase_tol/sqrt(6) -- NOT the per-axis
# phase_tol/3 split (that only satisfies the triangle inequality and wastes
# ~2.6x).  Note omega_b (not P_b) is the uniform coordinate, and x (not sin i)
# decouples the amplitude from the period (a_p ~ P_b^(2/3)).
#
# That bound is on the RAW residual.  The spin-frequency search absorbs a
# constant phase and a linear drift for free, which is most of the residual
# for orbits not much shorter than T; `_band_stretch` widens those bands'
# steps and certifies the result on the exact residual (-circ_strict: off).
#
# The midpoint anchor is load-bearing for the GRID DENSITY: the omega_b lever
# is max|t - T/2| = T/2, so anchoring at the start (lever T) would need twice
# as many omega_b points for the same phase budget.  The demod is told the same
# convention with anchor="midpoint" (see demodulate_series).
# ---------------------------------------------------------------------------

def circular_x_lt_s(p_o_s, sin_i, companion_mass, pulsar_mass):
    """x = a_p sin(i)/c [light-seconds] for a two-body Keplerian orbit."""
    return semi_major_axis_sini(p_o_s, sin_i, companion_mass, pulsar_mass) / C


def search_absorbed_max(R, tm):
    """max_t |R| left after the spin-frequency search takes its free share:
    a constant phase (no effect on power) and a linear drift (a spin-frequency
    offset).  The better of no line and the least-squares line -- both are
    lines the search can land on, so this bounds the best (minimax) line from
    above.  R: (..., n_t) residual phase [cycles] on tm, centred on 0."""
    c0 = R.mean(-1, keepdims=True)
    c1 = (R * tm).sum(-1, keepdims=True) / (tm * tm).sum()
    return np.minimum(np.abs(R).max(-1), np.abs(R - c0 - c1 * tm).max(-1))


def _cell_worst(args, w_lo, w_hi, x_g, tol, s, n_t=513, n_A=128, n_w=5):
    """Worst exact (nonlinear) residual over one band's cell boundary for step
    multipliers s = (s_x, s_A, s_w): truth half an x step from the grid row
    (both signs) and on the (A_T, omega) diamond edge, scanned over A_T and
    omega in the band.  Returns (search-absorbed max, raw max) [cycles]."""
    p0, T = args.p0, args.t_obs
    tm = np.linspace(-T / 2.0, T / 2.0, n_t)
    d_x = 2.0 * tol / np.sqrt(3.0) * p0 * s[0]
    lever = x_g + d_x / 2.0
    B = tol * np.sqrt(2.0 / 3.0)
    d_A = 2 * np.pi / (2 * np.ceil(np.pi / (B * p0 * s[1] / lever)))  # as built
    d_w = B * p0 * s[2] / (lever * (T / 2.0))
    A = np.linspace(0.0, 2 * np.pi, n_A, endpoint=False)[:, None, None, None]
    W = np.geomspace(w_lo, w_hi, n_w)[None, :, None, None]
    f = np.linspace(0.0, 1.0, 11)[None, None, :, None]
    eff = raw = 0.0
    for sx in (1, -1):
        x_t = x_g + sx * d_x / 2.0
        if x_t <= 0:
            continue
        for sa in (1, -1):
            for sw in (1, -1):
                R = (x_t / p0) * np.sin(W * tm + A) - (x_g / p0) * np.sin(
                    (W - sw * f * d_w) * tm + A - sa * (1 - f) * d_A)
                eff = max(eff, float(search_absorbed_max(R, tm).max()))
                raw = max(raw, float(np.abs(R).max()))
    return eff, raw


def _band_stretch(args, w_lo, w_hi, x_lo, x_hi, tol, s_max=20.0):
    """Per-axis step multipliers (s_x, s_A, s_w) >= 1 for one omega_b band.

    The bound of `_band_rows` is on the RAW residual, but the spin search
    absorbs its constant + linear part for free.  When the band's orbit is not
    much shorter than T (omega_b T/2 <~ 2), that is most of it: the x and A_T
    perturbations (sin/cos phi) shrink by L_A = max over A_T, omega of the
    search-absorbed max of cos(phi), the omega one (2t'/T cos phi) by L_w.
    Seed s = 1/L, then certify on the exact residual (`_cell_worst`) and
    shrink uniformly until within tol, at both ends of the band's x range.
    Floored at 1: s = 1 is the raw-bound grid, which is guaranteed already.
    Capped at s_max: past it the absorbed drift is a large spin-frequency
    offset and the orbit is ajs territory anyway.
    ponytail: one uniform shrink, not a re-optimised split; the volume-optimal
    a/B split of the raw bound is kept."""
    if args.circ_strict:
        return np.ones(3), None
    T = args.t_obs
    tm = np.linspace(-T / 2.0, T / 2.0, 513)
    A = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)[:, None, None]
    phi = np.geomspace(w_lo, w_hi, 5)[None, :, None] * tm + A
    L_A = search_absorbed_max(np.cos(phi), tm).max()
    L_w = search_absorbed_max((2 * tm / T) * np.cos(phi), tm).max()
    s = np.clip(1.0 / np.array([L_A, L_A, L_w]), 1.0, s_max)
    for _ in range(8):
        eff, raw = map(max, zip(*(_cell_worst(args, w_lo, w_hi, x_g, tol, s)
                                  for x_g in (x_lo, x_hi))))
        if eff <= tol or np.all(s == 1.0):
            break
        s = np.maximum(1.0, s * 0.98 * tol / eff)
    return s, raw


def row_cost(row):
    """Templates in one checkerboard row (len(A) is always even)."""
    return len(row[1]) * len(row[2]) // 2


def _band_rows(args, w_lo, w_hi, sin_lo, sin_hi, tol):
    """Rows (x_j, omega_b array, A_T array) of the circular grid for one
    omega_b band at phase budget `tol` [cycles], and its x step d_x.

    Phase-budget derivation.  A_T is the mean anomaly at the observation
    MIDPOINT, so with t the absolute time since the start, t' = t - T/2:
        Phi(t) = (x/p0) sin(phi),   phi = omega_b t' + A_T,
    perturbed by (to first order in the offset)
        dPhi(t) = a sin(phi) + (b' + c' (2 t'/T)) cos(phi),
    a, b', c' the (signed) x, A_T and omega contributions:
        a = dx/p0,   b' = x dA_T/p0,   c' = x domega (T/2)/p0.
    The omega lever is max|t'| = T/2 -- the whole reason to anchor at the
    midpoint (anchoring at the start doubles the omega axis).  Worst over t'
    and phase, the exact bound is a^2 + (|b'| + |c'|)^2 <= tol^2.  So:
     (1) x: half-step a = tol/sqrt(3); (A_T, omega): |b'|+|c'| <= B with
         B = tol sqrt(2/3) -- the volume-optimal split of the bound.
     (2) The (A_T, omega) cell is a DIAMOND |b'|+|c'| <= B, which tiles on a
         checkerboard: steps of B in each axis, keeping (i_omega + i_A) even.
         Half the points of the inscribed rectangle, same guarantee.  n_A_T is
         even so the parity holds across the 2 pi wrap; every consumer applies
         the rule (write_circular_csv, test_circular_grid.py).
     (3) The A_T and omega levers are proportional to x, so each row is spaced
         for ITS OWN largest snapping x, x_j + d_x/2, not the grid top.
     (4) A circular orbit has x = k sin_i omega_b^(-2/3), so the row covering
         [x_j - d_x/2, x_j + d_x/2] only spans the omega_b that some sin_i in
         [sin_lo, sin_hi] can reach -- the physical band, not a box.
     (5) Every step is then multiplied by the band's stretch (`_band_stretch`):
         what the spin search absorbs (constant + linear phase) is not charged.
    Returns (rows, d_x, stretch, raw max|R| at the stretched cell boundary).
    test_circular_grid.py Monte-Carlos the exact residual and the cell
    corners against this."""
    p0, T = args.p0, args.t_obs
    a_bud = tol / np.sqrt(3.0)
    B = tol * np.sqrt(2.0 / 3.0)
    k1 = circular_x_lt_s(2 * np.pi, 1.0, args.companion_mass, args.pulsar_mass)
    x_min = k1 * sin_lo * w_hi ** (-2.0 / 3.0)
    x_max = k1 * sin_hi * w_lo ** (-2.0 / 3.0)
    s, raw = _band_stretch(args, w_lo, w_hi, x_min, x_max, tol)
    d_x = 2.0 * a_bud * p0 * s[0]
    rows = []
    for xj in x_min + d_x * np.arange(int(np.ceil((x_max - x_min) / d_x)) + 1):
        xa = max(xj - d_x / 2.0, x_min)
        xb = min(xj + d_x / 2.0, x_max)
        w_a = max(w_lo, (k1 * sin_lo / xb) ** 1.5)
        w_b = min(w_hi, (k1 * sin_hi / xa) ** 1.5)
        if xb < xa or w_b < w_a:
            continue
        lever = xj + d_x / 2.0
        d_A = B * p0 * s[1] / lever
        d_w = B * p0 * s[2] / (lever * (T / 2.0))
        w = w_a + d_w * np.arange(int(np.ceil((w_b - w_a) / d_w)) + 1)
        n_A = 2 * int(np.ceil(np.pi / d_A))      # even, so d_A_T only shrinks
        rows.append((float(xj), w, (2 * np.pi / n_A) * np.arange(n_A)))
    return rows, d_x, s, raw


def derive_circular_grid(args, p_o_values=None):
    """Phase-budget circular grid.  Returns (spec, bands, orbits).

    The orbital-period ladder is split into one band per ladder period (edges
    at the geometric midpoints between neighbours), each its own sub-grid
    (`_band_rows`) at the -phase_tol_cycles budget.  Nothing is dropped: the
    cost knob is the budget itself, and loosening it uniformly is the
    minimum-loss way to save templates (count ~ budget^-3 in every band, so
    re-allocating it between bands only trades where the loss lands).  `bands`
    carries each band's rows, d_x, budget and count; `orbits` carries each
    (p_o, sin_i) combination's template count for the plot.  `p_o_values` restricts the period ladder (hybrid uses
    it for the below-p_break side); None = args.p_o.

    A_T is the mean anomaly at the observation MIDPOINT; the downstream demod
    must be driven with anchor='midpoint' (`write_circular_csv`,
    `_grid_kwargs` in src/demodsearch.jl)."""
    p_o_values = args.p_o if p_o_values is None else p_o_values
    per = np.array(sorted(set(p_o_values))) * YEAR_S
    edges = np.concatenate(([per[0]], np.sqrt(per[:-1] * per[1:]), [per[-1]]))
    sin_lo, sin_hi = min(args.sin_i), max(args.sin_i)
    bands = [dict(p_o_s=float(p), p_lo_s=float(lo), p_hi_s=float(hi),
                  w_lo=2 * np.pi / hi, w_hi=2 * np.pi / lo,
                  sin_lo=sin_lo, sin_hi=sin_hi)
             for p, lo, hi in zip(per, edges[:-1], edges[1:])]
    s_vals = np.array(sorted(set(args.sin_i)))
    s_edges = np.concatenate(([-np.inf], 0.5 * (s_vals[:-1] + s_vals[1:]), [np.inf]))
    k1 = circular_x_lt_s(2 * np.pi, 1.0, args.companion_mass, args.pulsar_mass)
    orbits = []
    for b in bands:
        b["tol"] = args.phase_tol_cycles
        b["rows"], b["d_x"], b["stretch"], b["raw"] = _band_rows(
            args, b["w_lo"], b["w_hi"], sin_lo, sin_hi, b["tol"])
        b["n"] = sum(row_cost(r) for r in b["rows"])
        # Each template's implied sin_i = x omega^(2/3)/k1, binned to the
        # nearest -sin_i value: the templates spent on each (p_o, sin_i).
        per_s = np.zeros(len(s_vals))
        for xj, W, A in b["rows"]:
            s = xj * W ** (2.0 / 3.0) / k1
            per_s += np.histogram(s, s_edges)[0] * (len(A) // 2)
        for s, n_s in zip(s_vals, per_s):
            orbits.append(dict(p_o_s=b["p_o_s"], sin_i=float(s), n=float(n_s)))

    n_trials = sum(b["n"] for b in bands)
    spec = {
        "mode": "circular",
        "inf": os.path.abspath(args.inf) if args.inf else None,
        "p0": float(args.p0),
        "t_obs": float(args.t_obs),
        "phase_tol_cycles": float(args.phase_tol_cycles),
        "companion_mass": float(args.companion_mass),
        "pulsar_mass": float(args.pulsar_mass),
        "p_o_yr": [float(v) for v in p_o_values],
        "sin_i": [float(v) for v in args.sin_i],
        # A_T is the mean anomaly at the observation midpoint; the demod must
        # be driven with anchor="midpoint" (demod_dat.jl / src/demodsearch.jl).
        "A_T_anchor": "midpoint",
        # True: phase_tol bounds the residual left after the spin search
        # absorbs a constant + linear phase; False (-circ_strict): raw max|R|.
        "search_absorbs_linear": not args.circ_strict,
        "lattice": "checkerboard (i_omega + i_A_T even) in (omega_b, A_T)",
        "bands": [{"p_o_d": b["p_o_s"] / 86400.0,
                   "p_lo_d": b["p_lo_s"] / 86400.0,
                   "p_hi_d": b["p_hi_s"] / 86400.0,
                   "stretch_x_AT_omega": [float(v) for v in b["stretch"]],
                   "n_trials": int(b["n"])} for b in bands],
        "n_trials": int(n_trials),
    }
    return spec, bands, orbits


def circular_share_plot(orbits, spec, args, out_path):
    """Where the circular templates go: the % of all circular templates spent
    on each (p_o, sin_i) combination (a template counts towards the -sin_i
    value nearest its implied sin_i), annotated with the count.  Right:
    templates per period band.  Written even under -no_write (it is small;
    the CSV is what that skips)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p_vals = sorted({o["p_o_s"] for o in orbits})
    s_vals = sorted({o["sin_i"] for o in orbits})
    ip = {v: i for i, v in enumerate(p_vals)}
    js = {v: j for j, v in enumerate(s_vals)}
    n = np.zeros((len(s_vals), len(p_vals)))
    for o in orbits:
        n[js[o["sin_i"]], ip[o["p_o_s"]]] = o["n"]
    share = 100.0 * n / max(n.sum(), 1)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(15.5, 5.5),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    im = ax.imshow(share, origin="lower", aspect="auto", cmap="viridis")
    for j in range(len(s_vals)):
        for i in range(len(p_vals)):
            ax.text(i, j, f"{share[j, i]:.1f}%\n{n[j, i]:.2e}", ha="center",
                    va="center", fontsize=7,
                    color="white" if share[j, i] < 0.6 * share.max() else "black")
    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels([f"{p / 86400.0:.3g}" for p in p_vals], rotation=45,
                       fontsize=8)
    ax.set_yticks(range(len(s_vals)))
    ax.set_yticklabels([f"{s:g}" for s in s_vals], fontsize=8)
    ax.set_xlabel("Orbital period $p_o$ [days]")
    ax.set_ylabel("sin(i)")
    ax.set_title("Share of circular templates per (p_o, sin i), and count")
    fig.colorbar(im, ax=ax, label="% of all circular templates")

    x = np.arange(len(spec["bands"]))
    axb.bar(x, [b["n_trials"] for b in spec["bands"]], color="C0")
    axb.set_yscale("log")
    axb.set_xticks(x)
    axb.set_xticklabels([f"{b['p_o_d']:.3g}" for b in spec["bands"]],
                        rotation=45, fontsize=8)
    axb.set_xlabel("Period band centre $p_o$ [days]")
    axb.set_ylabel("templates")

    fig.suptitle(
        f"Circular grid (T = {args.t_obs:g} s, p0 = {args.p0:g} s, budget "
        f"{args.phase_tol_cycles:g} cyc, nothing dropped): "
        f"{spec['n_trials']:.3e} templates", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved circular template-share plot to {out_path}")


def write_circular_csv(csv_path, bands):
    """CSV (pb,x,at) of every concrete circular trial.  pb in days, x in
    light-seconds, at the mean anomaly [rad] at the observation MIDPOINT
    (demod_dat.jl / src/demodsearch.jl are driven with anchor="midpoint").
    Anchoring A_T at the midpoint (not the .inf epoch) is what keeps the
    omega_b lever at T/2 and halves the dominant grid axis."""
    n_tot = 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["pb", "x", "at"])
        for row in (r for b in bands for r in b["rows"]):
            xv, omega_b, A_T = row
            for i, pbd in enumerate(2 * np.pi / omega_b / 86400.0):
                for at in A_T[i % 2::2]:        # checkerboard lattice
                    w.writerow([f"{pbd:.10g}", f"{xv:.10g}", f"{at:.10g}"])
            n_tot += row_cost(row)
    return n_tot


def write_circular_outputs(args, spec, bands):
    """YAML spec + CSV (pb,x,at) of every concrete circular trial.  Paths
    are <outdir>/<outstem>.{yaml,csv}.  Hybrid does not call this: it writes
    <outstem>_circular.csv via write_circular_csv and its own combined YAML."""
    if args.no_write:
        print(f"\n[no_write] circular grid not written ({spec['n_trials']} trial(s))")
        return None, None
    os.makedirs(args.outdir, exist_ok=True)
    yaml_path = os.path.join(args.outdir, f"{args.outstem}.yaml")
    csv_path = os.path.join(args.outdir, f"{args.outstem}.csv")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False)
    n_tot = write_circular_csv(csv_path, bands)
    print(f"\nWrote circular grid spec -> {yaml_path}")
    print(f"Wrote {n_tot} trial(s) -> {csv_path}")
    return yaml_path, csv_path


def circular_summary(spec, bands):
    """Per-band report of the circular grid."""
    tot = max(spec["n_trials"], 1)
    lines = [f"{'p_o band [d]':>19} {'templates':>11} {'share':>7} "
             f"{'stretch x/A_T/omega':>21} {'raw max|R|':>10}"]
    for b, bb in zip(spec["bands"], bands):
        sx, sa, sw = bb["stretch"]
        raw = "-" if bb["raw"] is None else f"{bb['raw']:.3g}"
        lines.append(f"{b['p_lo_d']:>8.4g}-{b['p_hi_d']:<10.4g} "
                     f"{b['n_trials']:>11.3e} {100 * b['n_trials'] / tot:>6.1f}% "
                     f"{sx:>6.2f} {sa:>6.2f} {sw:>6.2f} {raw:>11}")
    lines.append("(stretch: step multiplier won from the constant + linear "
                 "phase the spin search absorbs; raw max|R| [cyc]: the drift it "
                 "absorbs, a spin-frequency offset of up to 2*raw/T Hz)")
    lines.append(f"{'total':>19} {spec['n_trials']:>11.3e}")
    return "\n".join(lines)


def main_circular(args):
    """Circular-orbit grid derivation (no cubic-truncation gate: the demod is
    exact at the truth)."""
    spec, bands, orbits = derive_circular_grid(args)
    print(f"Circular-orbit grid: sin_i = {min(args.sin_i):g}-{max(args.sin_i):g}, "
          f"A_T = 0-2pi (midpoint anchor), mass = {args.pulsar_mass:g}+"
          f"{args.companion_mass:g} Msun")
    print(f"T = {args.t_obs:.6g} s, p0 = {args.p0} s, phase budget "
          f"{args.phase_tol_cycles:g} cyc (nothing dropped; -drop_pct is ajs-only)")
    print(circular_summary(spec, bands))
    write_circular_outputs(args, spec, bands)
    circular_share_plot(orbits, spec, args, os.path.join(
        args.outdir, args.plot_subdir, "circular_template_share.png"))


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


def ajs_grid_from_scan(args, recs, caps, model_lbl, do_plots=True,
                       csv_path=None, yaml_path=None, plot_stem=""):
    """Gate / cap / drop / range / allocate / validate a set of scanned ajs
    records and write the grid.  `recs` is the exact subset to grid (hybrid
    passes only p_o >= p_break).  Flags (feasible/kept) are recomputed here.
    Returns the spec, or None if the subset is infeasible."""
    span_s = args.t_obs
    phase_tol_m = args.phase_tol_cycles * C * args.p0

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
        wt = (max(r["trunc_phase"] for r in sub) / (C * args.p0)
              if sub else 0.0)
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
        return None

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
        return None
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
        # the periods this sub-grid actually covers (hybrid filters the ladder)
        "p_o_yr": [float(x) for x in sorted({r["p_o"] for r in recs})],
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
    write_grid_outputs(args, spec, csv_path=csv_path, yaml_path=yaml_path)

    if do_plots and not args.no_write:
        groups = defaultdict(list)
        for r in recs:
            groups[(r["p_o"], r["sin_i"], r["e"], r["omega_peri"])].append(r)
        plot_dir = os.path.join(args.outdir, args.plot_subdir)
        os.makedirs(plot_dir, exist_ok=True)
        corner_path = os.path.join(plot_dir,
                                   f"{plot_stem}corner_feasible_fraction.png")
        corner_feasible_fraction(groups, args, corner_path)
        print(f"Saved corner plot of % orbit kept to {corner_path}")

        cover_path = os.path.join(plot_dir, f"{plot_stem}claimed_fraction.png")
        claimed_fraction_plot(recs, args, cover_path)
        print(f"Saved % phase claimed vs p_o to {cover_path}")

        if args.plot:
            orbit_phase_param_plots(groups, args, model_lbl, plot_dir)
    return spec


def _natural_caps(args, p_min_yr):
    """Default NS-NS caps from the peak kinematics of the orbit at p_min_yr
    [yr] (largest scanned e and sin_i), unless the user capped an axis."""
    p_min_s = p_min_yr * YEAR_S
    a_pk, j_pk, s_pk = peak_kinematics(
        p_min_s, max(args.e), max(args.sin_i),
        args.companion_mass, args.pulsar_mass)
    return [
        args.max_accel if args.max_accel is not None else a_pk,
        args.max_jerk if args.max_jerk is not None else j_pk,
        args.max_snap if args.max_snap is not None else s_pk,
    ]


def ajs_coverage_by_p_o(recs, args, caps, phase_tol_m):
    """Mean ajs feasibility (phase gate + caps, BEFORE -drop_pct) over the e=0
    records at each p_o, pooled over the scanned sin_i x omega_peri combos (each
    combo has the same n_phase, so this is also the combo mean).  Returns
    (coverage_dict, used_all_e); used_all_e is True when no e=0 records exist and
    the fallback pooled over every e."""
    mark_feasible(recs, phase_tol_m)
    apply_caps(recs, caps)
    e0 = [r for r in recs if r["e"] == 0.0]
    used_all_e = not e0
    pool = recs if used_all_e else e0
    tot = defaultdict(int)
    feas = defaultdict(int)
    for r in pool:
        tot[r["p_o"]] += 1
        feas[r["p_o"]] += int(r["feasible"])
    return {p: feas[p] / tot[p] for p in tot}, used_all_e


def main_ajs(args, inf_N, inf_dt):
    """The pure accel/jerk/snap grid (the original -mode ajs path)."""
    args.term_mask = terms_mask(args.terms)
    model_lbl = terms_label(args.term_mask)
    n_orbits = (len(args.p_o) * len(args.sin_i) * len(args.e)
                * len(args.omega_peri))
    phase_tol_m = args.phase_tol_cycles * C * args.p0
    caps = _natural_caps(args, min(args.p_o))

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

    # ---- Stage 1: scan orbit grid x phase ----
    recs = scan_segments(args.t_obs, args, args.p_o, args.sin_i, args.e,
                         args.omega_peri, args.n_phase, args.nsamp)
    ajs_grid_from_scan(args, recs, caps, model_lbl)


def main_hybrid(args, inf_N, inf_dt):
    """Hybrid grid: exact circular below p_break, ajs at/above it.

    p_break is the shortest scanned p_o such that every p_o at/above it has mean
    e=0 ajs coverage (phase gate + caps, before -drop_pct) >= -break_coverage
    (default 60%).  Because ajs coverage dips near p_o ~ T_obs, this clears the
    whole inadequate band rather than stopping at the first point that clears.
    A manual -p_break overrides.  Emits <stem>_circular.csv and <stem>_ajs.csv
    plus one combined YAML."""
    args.term_mask = terms_mask(args.terms)
    model_lbl = terms_label(args.term_mask)
    phase_tol_m = args.phase_tol_cycles * C * args.p0
    p_ladder = sorted(set(args.p_o))

    def rng(vals):
        return (f"{min(vals):g}" if len(vals) == 1
                else f"{min(vals):g}-{max(vals):g} ({len(vals)})")

    print(f"Hybrid orbit grid: p_o = {rng(args.p_o)} yr, "
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
    print()

    # ---- Stage 1: one full-ladder ajs scan (serves coverage AND the ajs side)
    recs = scan_segments(args.t_obs, args, args.p_o, args.sin_i, args.e,
                         args.omega_peri, args.n_phase, args.nsamp)

    # ---- Stage 2: p_break from the e=0 ajs coverage ----
    cov_caps = _natural_caps(args, min(args.p_o))
    coverage, used_all_e = ajs_coverage_by_p_o(recs, args, cov_caps, phase_tol_m)
    if used_all_e:
        print("WARNING: no e=0 records in the scan; judging p_break coverage "
              "over ALL e (the circular sub-grid still only covers e=0).")

    print(f"{'p_o [yr]':>14} {'p_o [d]':>10} {'ajs coverage':>14}")
    for p in p_ladder:
        c = coverage.get(p)
        print(f"{p:>14.6g} {p * 365.25:>10.5g} "
              f"{('-' if c is None else f'{100.0 * c:.1f}%'):>14}")

    if args.p_break is not None:
        p_break = float(args.p_break)
        p_break_auto = False
        print(f"\np_break = {p_break:g} yr (manual -p_break; the "
              f"-break_coverage {args.break_coverage:g}% rule is ignored)")
    else:
        p_break_auto = True
        # ajs coverage is NOT monotonic in p_o: it dips near p_o ~ T_obs, then
        # recovers for long orbits.  The switch must clear the WHOLE inadequate
        # band, so p_break = the shortest p_o such that every scanned p_o >= it
        # has coverage >= the threshold, i.e. the next ladder point above the
        # largest p_o that is still below threshold.
        thresh = args.break_coverage / 100.0
        bad = [p for p in p_ladder if coverage.get(p, 0.0) < thresh]
        if not bad:
            p_break = p_ladder[0]
            print(f"\nEvery scanned p_o already reaches {args.break_coverage:g}% "
                  f"ajs coverage; the whole band is gridded ajs.")
        elif bad[-1] == p_ladder[-1]:
            p_break = p_ladder[-1] * (1.0 + 1e-9)  # never clears: all circular
            print(f"\nThe longest scanned p_o still has < {args.break_coverage:g}% "
                  f"ajs coverage; no p_o clears the threshold, so the whole "
                  f"band is gridded circular.")
        else:
            i_bad = p_ladder.index(bad[-1])
            p_break = p_ladder[i_bad + 1]
            print(f"\np_break = {p_break:g} yr ({p_break * 365.25:.5g} d): "
                  f"circular below, ajs at/above. Last p_o below "
                  f"{args.break_coverage:g}% is {bad[-1]:g} yr "
                  f"({bad[-1] * 365.25:.5g} d); the switch clears the whole "
                  f"inadequate band (ajs coverage is non-monotonic, dipping "
                  f"near p_o ~ T_obs).")

    circ_periods = [p for p in args.p_o if p < p_break]
    ajs_recs = [r for r in recs if r["p_o"] >= p_break]
    ajs_p_min = min((r["p_o"] for r in ajs_recs), default=None)

    outdir = args.outdir
    plot_dir = os.path.join(outdir, args.plot_subdir)
    if not args.no_write:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

    # ---- Stage 3a: circular sub-grid (p_o < p_break, e=0) ----
    circ_spec = None
    if circ_periods:
        print(f"\n--- Circular side: {len(circ_periods)} period(s) < p_break ---")
        circ_spec, bands, orbits = derive_circular_grid(
            args, p_o_values=circ_periods)
        print(circular_summary(circ_spec, bands))
        print(f"Total circular templates: {circ_spec['n_trials']:.3e}")
        if args.no_write:
            print(f"[no_write] circular grid not written "
                  f"({circ_spec['n_trials']:.3e} trial(s))")
        else:
            write_circular_csv(
                os.path.join(outdir, f"{args.outstem}_circular.csv"), bands)
            print(f"Wrote {circ_spec['n_trials']} trial(s) -> "
                  f"{os.path.join(outdir, args.outstem + '_circular.csv')}")
        circular_share_plot(orbits, circ_spec, args, os.path.join(
            plot_dir, "hybrid_circular_template_share.png"))
    else:
        print("\nNo periods below p_break: no circular sub-grid.")

    # ---- Stage 3b: ajs sub-grid (p_o >= p_break), caps from p_break's orbit ----
    ajs_spec = None
    if ajs_recs:
        print(f"\n--- Ajs side: {len(ajs_recs)} scanned segments with "
              f"p_o >= p_break ---")
        ajs_caps = _natural_caps(args, ajs_p_min)
        print(f"Ajs caps (from p_o = {ajs_p_min:g} yr): "
              f"|accel| <= {ajs_caps[0]:.4g} m/s^2, "
              f"|jerk| <= {ajs_caps[1]:.4g} m/s^3, "
              f"|snap| <= {ajs_caps[2]:.4g} m/s^4")
        ajs_spec = ajs_grid_from_scan(
            args, ajs_recs, ajs_caps, model_lbl, do_plots=False,
            csv_path=os.path.join(outdir, f"{args.outstem}_ajs.csv"),
            yaml_path=_SKIP)
    else:
        print("\nNo scanned segments at/above p_break: no ajs sub-grid.")

    # ---- Combined YAML ----
    n_ajs_trials = 0
    if ajs_spec:
        n_ajs_trials = 1
        for i in range(3):
            if ajs_spec["active"][i]:
                lo, hi = ajs_spec["ranges"][i]
                n_ajs_trials *= int(np.ceil((hi - lo) / ajs_spec["spacings"][i])) + 1

    spec = {
        "mode": "hybrid",
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
        "break_coverage": float(args.break_coverage),
        "p_break": float(p_break),
        "p_break_auto": bool(p_break_auto),
        "coverage_by_p_o": {f"{p:.10g}": float(coverage.get(p, 0.0))
                            for p in p_ladder},
        "coverage_used_all_e": bool(used_all_e),
        "n_circular_trials": int(circ_spec["n_trials"]) if circ_spec else 0,
        "n_ajs_trials": int(n_ajs_trials),
        "circular": circ_spec,
        "ajs": ajs_spec,
    }
    yaml_path = os.path.join(outdir, f"{args.outstem}.yaml")
    if args.no_write:
        print(f"\n[no_write] hybrid spec not written "
              f"(circular {int(circ_spec['n_trials']) if circ_spec else 0} + "
              f"ajs {int(n_ajs_trials)} trials)")
        return
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False)
    print(f"\nWrote hybrid spec -> {yaml_path}")


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

    if args.mode == "circular":
        return main_circular(args)
    if args.mode == "hybrid":
        return main_hybrid(args, inf_N, inf_dt)
    return main_ajs(args, inf_N, inf_dt)


if __name__ == "__main__":
    main()
