#!/usr/bin/env python
"""
Pick the ajs grid row that best matches a pulsar's binary ephemeris.

The input grid is the CSV emitted by `demod/nsns_grid.py` in `-mode ajs`
(header `accel,jerk,snap`).  The ephemeris is a TEMPO/TEMPO2 `.par` file; the
observation metadata comes from the PRESTO `.inf` (N, dt, epoch).

The script:

  1. parses the binary model from the `.par` (BT/DD/DDGR, and the small-e ELL1
     form), turning it into the same DD parameter set `pulsegen_gr.py` uses;
  2. fits ONE accel/jerk/snap cubic to the model's line-of-sight velocity over
     the observation, anchored at the observation midpoint -- the exact
     convention `nsns_grid.py` used to derive the grid (`v0 + a*tau +
     j*tau^2/2 + s*tau^3/6`, GR from the two masses by default);
  3. snaps that best fit to the nearest row of the grid (per-axis nearest,
     which is exact for the Cartesian-product CSV `nsns_grid.py` writes; a
     phase-impact-weighted nearest row is used if the CSV is not a full
     product);
  4. writes `<grid-stem>_pulsar_optimal_ajs_circular.csv` with the same
     `accel,jerk,snap` header and exactly one data row.

The output row is what `bin/coherent_search.jl --demod-grid` and
`demod/demod_grid.jl` expect, so it can be fed straight back into the demod
pipeline.  The literal grid strings are preserved (rather than reformatted)
so the downstream per-row filenames match the grid text.

Usage:
    python demod/pulsar_optimal_ajs_circular.py GRID.csv PULSAR.par OBS.inf \
        [-o OUT.csv] [--method fit|instant] [--exact] [--no-gr]

`--exact` writes the fitted (a,j,s) values instead of snapping to a grid row;
the default is the closest grid row, which is what `--demod-grid` consumes.

The "circular" in the output name is the pipeline's label for this one-row
optimal solution.  The ajs themselves are computed from the full (possibly
eccentric) ephemeris: circularising an eccentric orbit would give a completely
different acceleration and would not demodulate it.
"""
import argparse
import csv
import os
import sys
import types

import numpy as np

# pulsegen_gr.py lives beside this file and already owns the DD model and the
# kinematic-fit convention.  Importing it here keeps the two from drifting.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pulsegen_gr import (  # noqa: E402
    C,
    DDParams,
    best_fit_kinematic,
    build_dd_params,
    dd_delay,
    projected_semi_major_axis,
    read_inf,
)

# The grid's midpoint-anchored phase levers (nsns_grid.PHASE_LEVERS["midpoint"]).
# Used only for the report and for a non-product grid's nearest-row metric.
PHASE_LEVERS = {
    "midpoint": lambda T: np.array([T**2 / 8.0, T**3 / 24.0, T**4 / 384.0]),
    "start": lambda T: np.array([T**2 / 2.0, T**3 / 6.0, T**4 / 24.0]),
}
AXES = ("accel", "jerk", "snap")
YEAR_S = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# TEMPO/TEMPO2 .par parsing
# ---------------------------------------------------------------------------

def read_par(path):
    """Return a dict of the first value of every key in a TEMPO-style .par.

    TEMPO lines are `KEY VALUE [flags]`; `#` starts a comment.  Only the first
    token after the key is kept, which is all this script needs (TEMPO's
    uncertainty/flag columns are not used)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].upper()
            out.setdefault(key, parts[1])
    return out


def _get_float(par, key, default=None):
    if key not in par:
        return default
    try:
        return float(par[key])
    except ValueError:
        raise ValueError(f"par: {key}={par[key]!r} is not a number")


def par_to_dd(par, epoch_mjd, companion_mass=None, pulsar_mass=None,
              no_gr=False):
    """Build a `pulsegen_gr.DDParams` and the spin parameters from a .par.

    Returns `(prm, p0, pdot, info)`.  `info` carries the human-readable model
    description and the parameter values used in the report.

    Supported binary models:
      * BT / DD / DDGR -- T0 is periastron passage, OM is in degrees;
      * ELL1          -- TASC is the ascending-node epoch; EPS1/EPS2 give e and
                         omega (the exact ELL1 convention is not needed for the
                         ajs cubic: only the instantaneous orbit matters).
    A par with no PB/A1 is treated as an isolated pulsar (ajs = 0)."""
    f0 = _get_float(par, "F0")
    p0_par = _get_float(par, "P0")
    if f0 is not None:
        p0 = 1.0 / f0
    elif p0_par is not None:
        p0 = p0_par
        f0 = 1.0 / p0
    else:
        raise ValueError("par: need F0 (or P0)")

    f1 = _get_float(par, "F1", 0.0)
    p1 = _get_float(par, "P1", None)
    pdot = p1 if p1 is not None else -f1 / f0**2

    model = par.get("BINARY", "").upper()
    pb_days = _get_float(par, "PB", None)
    a1 = _get_float(par, "A1", None)
    if a1 is None:
        a1 = _get_float(par, "X", None)  # some writers use X for A1

    mc = _get_float(par, "M2", None)
    mtot = _get_float(par, "MTOT", None)
    if companion_mass is not None:
        mc = companion_mass
    if pulsar_mass is not None:
        mp = pulsar_mass
    elif mtot is not None and mc is not None:
        mp = mtot - mc
    else:
        mp = 1.4
    if mc is None:
        mc = 1.4

    if pb_days is None or a1 is None:
        # Isolated pulsar: a purely constant velocity has zero ajs.  Keep a
        # valid DDParams-like object so the same fitting code path works.
        if pb_days is None and a1 is None:
            prm = DDParams(pb_s=1.0, x=0.0, e=0.0, omega_peri=0.0, A_T=0.0,
                           omega_dot=0.0, n_dot=0.0, gamma=0.0, r=0.0, s=0.0,
                           delta_r=0.0, delta_theta=0.0)
            info = dict(model="isolated", pb_days=None, a1=None, e=0.0,
                        omega_peri=0.0, t0_mjd=None, p_epoch=_get_float(par, "PEPOCH"),
                        companion_mass=mc, pulsar_mass=mp, A_T=0.0)
            return prm, p0, pdot, info
        raise ValueError("par: need both PB and A1 for a binary model")

    e = _get_float(par, "E", 0.0)
    omega = _get_float(par, "OM", None)
    t0 = _get_float(par, "T0", None)
    tasc = _get_float(par, "TASC", None)

    if model == "ELL1" or (t0 is None and tasc is not None):
        t0 = tasc if tasc is not None else t0
        eps1 = _get_float(par, "EPS1", 0.0)
        eps2 = _get_float(par, "EPS2", 0.0)
        if e == 0.0:
            e = float(np.hypot(eps1, eps2))
        omega = float(np.arctan2(eps1, eps2))
        if t0 is None:
            raise ValueError("par: ELL1 needs TASC (or T0)")
    else:
        if omega is None:
            omega = 0.0
        omega = np.deg2rad(omega)
        if t0 is None:
            raise ValueError("par: binary model needs T0 (or TASC)")

    pb_s = pb_days * 86400.0
    # Mean anomaly at the .inf epoch (t = 0 in pulsegen_gr's convention).
    A_T = 2.0 * np.pi * ((epoch_mjd - t0) * 86400.0) / pb_s
    A_T = float(np.mod(A_T, 2.0 * np.pi))

    # Shapiro shape s = sin(i).  Use the par's SINI when given; otherwise infer
    # it from A1 and the masses (falling back to 1 for an inconsistent mass
    # function).  It only enters the tiny Shapiro rate.
    sini = _get_float(par, "SINI", None)
    if sini is None:
        x_sini1 = projected_semi_major_axis(pb_s, 1.0, mc, mp)
        sini = float(np.clip(a1 / x_sini1, 0.0, 1.0)) if x_sini1 > 0 else 1.0

    # Let build_dd_params compute the PK terms from the two masses (the same
    # convention as pulsegen_gr.py and nsns_grid.py's gr=True truth), then
    # apply explicit par overrides where present.
    args = types.SimpleNamespace(
        pb=pb_days, x=a1, e=e, omega_peri=omega, A_T=A_T,
        companion_mass=mc, pulsar_mass=mp, sin_i=sini,
        gamma=None, omega_dot=None, n_dot=None,
        shapiro_r=None, shapiro_s=None,
    )
    prm = build_dd_params(args)

    if no_gr:
        prm.omega_dot = 0.0
        prm.n_dot = 0.0
        prm.gamma = 0.0
        prm.r = 0.0
        prm.s = 0.0
        prm.delta_r = 0.0
        prm.delta_theta = 0.0
    else:
        # Optional explicit PK values, if the par carries them.
        omdot = _get_float(par, "OMDOT", None)
        if omdot is not None:
            prm.omega_dot = np.deg2rad(omdot) / YEAR_S
        pbdot = _get_float(par, "PBDOT", None)
        if pbdot is not None:
            prm.n_dot = -(prm.n_b / prm.pb_s) * pbdot
        gamma = _get_float(par, "GAMMA", None)
        if gamma is not None:
            prm.gamma = gamma
        sini = _get_float(par, "SINI", None)
        if sini is not None:
            prm.s = sini

    info = dict(model=model or "BT", pb_days=pb_days, a1=a1, e=e,
                omega_peri=omega, t0_mjd=t0, p_epoch=_get_float(par, "PEPOCH"),
                companion_mass=mc, pulsar_mass=mp, A_T=A_T)
    return prm, p0, pdot, info


# ---------------------------------------------------------------------------
# ajs from the ephemeris
# ---------------------------------------------------------------------------

def model_velocity(t, prm, p0, pdot):
    """The searches' v = c*(P_obs/p0 - 1) for the DD delay model.

    This is the same expression `pulsegen_gr.best_fit_kinematic` uses, so the
    fitted coefficients are directly comparable with the grid."""
    t = np.asarray(t, dtype=float)
    h = max(float(np.max(np.abs(t))) if t.size else 1.0, 1.0) * 1e-6
    ddelay = (dd_delay(t + h, prm) - dd_delay(t - h, prm)) / (2.0 * h)
    return C * ((1.0 + (pdot / p0) * t) * (1.0 + ddelay) - 1.0)


def instant_ajs(prm, t_mid, no_gr, companion_mass, pulsar_mass):
    """Instantaneous a/j/s at `t_mid` (optional alternative to the LSQ fit)."""
    from simulate_orbit_accel_jerk import los_acceleration, los_jerk, los_snap
    A_T = prm.A_T + prm.n_b * t_mid
    kw = dict(m_c=companion_mass, m_p=pulsar_mass, gr=not no_gr)
    a = los_acceleration(t_mid, prm.n_b, prm.x * C, A_T, prm.e,
                         prm.omega_peri, **kw)
    j = los_jerk(t_mid, prm.n_b, prm.x * C, A_T, prm.e,
                 prm.omega_peri, **kw)
    s = los_snap(t_mid, prm.n_b, prm.x * C, A_T, prm.e,
                 prm.omega_peri, **kw)
    return np.array([float(a), float(j), float(s)])


def residual_phase(prm, p0, pdot, t_obs, ajs, t_anchor):
    """Worst integrated phase error of a cubic ajs model vs the true model.

    Returns `(raw_cycles, search_absorbed_cycles)`.  `raw` removes only the
    constant v0 (the spin search absorbs it as a frequency offset); the second
    value also removes the best linear phase drift (the spin-frequency search
    absorbs that too).  Both are in cycles, using the ephemeris' p0."""
    t = np.linspace(0.0, t_obs, 4001)
    tau = t - t_anchor
    v = model_velocity(t, prm, p0, pdot)
    vmod = ajs[0] * tau + ajs[1] * tau**2 / 2.0 + ajs[2] * tau**3 / 6.0
    dv = v - vmod
    dv = dv - np.mean(dv)
    phase = np.concatenate(([0.0],
                            np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(tau))))
    raw = float(np.max(np.abs(phase)) / (C * p0))
    A = np.column_stack([np.ones_like(tau), tau])
    coef = np.linalg.lstsq(A, phase, rcond=None)[0]
    absorbed = float(np.max(np.abs(phase - A @ coef)) / (C * p0))
    return raw, absorbed


# ---------------------------------------------------------------------------
# Grid selection
# ---------------------------------------------------------------------------

def read_grid(path):
    """Read an ajs grid CSV.  Returns `(header, arr)` with arr (N, 3)."""
    with open(path) as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty grid CSV")
        header = [h.strip().lower() for h in header]
        if header != ["accel", "jerk", "snap"]:
            raise ValueError(
                f"{path}: unsupported header {','.join(header)}; this tool "
                "needs an ajs grid with header 'accel,jerk,snap'")
        rows = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            if len(row) != 3:
                raise ValueError(f"{path}: expected 3 columns, got {len(row)}")
            rows.append([float(row[0]), float(row[1]), float(row[2])])
    if not rows:
        raise ValueError(f"{path}: grid has no data rows")
    return header, np.asarray(rows, dtype=float)


def select_row(path, arr, target, t_obs, p0):
    """Nearest grid row to `target` (a, j, s).

    For the full Cartesian product `nsns_grid.py` writes, the nearest point is
    the per-axis nearest and that is what is returned.  For a sparse/non-product
    CSV, the nearest row is found with a phase-impact-weighted metric using the
    grid's own midpoint phase levers, so an offset in j or s is not ignored
    merely because its numeric scale is tiny.
    """
    uniq = [np.unique(arr[:, i]) for i in range(3)]
    is_product = (len(uniq[0]) * len(uniq[1]) * len(uniq[2]) == len(arr))

    if is_product:
        chosen = np.array([
            float(uniq[i][np.argmin(np.abs(uniq[i] - target[i]))])
            for i in range(3)
        ])
        exact = np.flatnonzero(np.all(arr == chosen[None, :], axis=1))
        if exact.size:
            idx = int(exact[0])
        else:
            # A malformed/duplicated CSV: fall back to the nearest actual row.
            idx = int(np.argmin(np.sum((arr - chosen[None, :])**2, axis=1)))
            chosen = arr[idx].copy()
    else:
        # Phase per unit offset: lever / (c*p0), in cycles per (m/s^2 etc.).
        lever = PHASE_LEVERS["midpoint"](t_obs) / (C * p0)
        d2 = np.sum(((arr - target[None, :]) * lever[None, :])**2, axis=1)
        idx = int(np.argmin(d2))
        chosen = arr[idx].copy()

    # Preserve the CSV's literal number strings for this row: downstream
    # filenames are built from this text (see demod_point_stem).
    literal = None
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                vals = (float(row[0]), float(row[1]), float(row[2]))
            except (ValueError, IndexError):
                continue
            if vals == tuple(float(x) for x in chosen):
                literal = [row[0].strip(), row[1].strip(), row[2].strip()]
                break
    if literal is None:
        literal = [f"{v:.10g}" for v in chosen]
    return chosen, literal, idx, is_product


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Write the one-row ajs grid point that best matches a "
                    "pulsar ephemeris.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("grid", help="ajs grid CSV from demod/nsns_grid.py "
                                 "(header accel,jerk,snap)")
    ap.add_argument("par", help="TEMPO/TEMPO2 pulsar ephemeris (.par)")
    ap.add_argument("inf", help="PRESTO .inf of the observation (N, dt, epoch)")
    ap.add_argument("-o", "--out", default=None,
                    help="Output CSV; default: <grid-stem>_pulsar_optimal_"
                         "ajs_circular.csv beside the grid")
    ap.add_argument("--method", choices=("fit", "instant"), default="fit",
                    help="'fit': one LSQ cubic over the whole observation "
                         "(the grid's convention); 'instant': the a/j/s "
                         "derivatives at the anchor epoch")
    ap.add_argument("--anchor", choices=("midpoint", "start"), default="midpoint",
                    help="Where the cubic is expanded.  The demod resampler "
                         "and nsns_grid.py both use midpoint; 'start' is only "
                         "for inspection")
    ap.add_argument("--exact", action="store_true",
                    help="Write the fitted (a,j,s) values instead of snapping "
                         "to the nearest grid row")
    ap.add_argument("--companion-mass", type=float, default=None,
                    help="Override the companion mass [Msun] (default: par M2, "
                         "else 1.4)")
    ap.add_argument("--pulsar-mass", type=float, default=None,
                    help="Override the pulsar mass [Msun] (default: par MTOT-M2, "
                         "else 1.4)")
    ap.add_argument("--no-gr", action="store_true",
                    help="Zero the post-Keplerian terms in the model "
                         "(default: GR on, computed from the masses)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.anchor == "start":
        print("WARNING: nsns_grid.py and the demod resampler anchor the ajs "
              "polynomial at the observation midpoint.  --anchor start is "
              "for inspection only; the selected row will not be the right "
              "one to feed to --demod-grid.")

    N, dt, epoch_mjd, _ = read_inf(args.inf)
    t_obs = N * dt
    target_epoch = t_obs / 2.0 if args.anchor == "midpoint" else 0.0

    par = read_par(args.par)
    prm, p0, pdot, info = par_to_dd(
        par, epoch_mjd,
        companion_mass=args.companion_mass,
        pulsar_mass=args.pulsar_mass,
        no_gr=args.no_gr)

    if args.method == "fit":
        fit = best_fit_kinematic(prm, p0, pdot, t_obs, t_ref=0.0,
                                 t_anchor=target_epoch)
        ajs = np.array([fit["accel"], fit["jerk"], fit["snap"]])
        fit_note = (f"LSQ cubic over T={t_obs:g} s, anchor={args.anchor}, "
                    f"max |v residual|={fit['max_residual_v']:.3g} m/s")
    else:
        ajs = instant_ajs(prm, target_epoch, args.no_gr,
                          info["companion_mass"], info["pulsar_mass"])
        fit_note = f"instantaneous derivatives at t={target_epoch:g} s"

    header, arr = read_grid(args.grid)

    if args.exact:
        chosen = ajs.copy()
        literal = [f"{v:.10g}" for v in chosen]
        is_product = False
    else:
        chosen, literal, idx, is_product = select_row(
            args.grid, arr, ajs, t_obs, p0)

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.grid))[0]
        out = os.path.join(os.path.dirname(os.path.abspath(args.grid)),
                           f"{stem}_pulsar_optimal_ajs_circular.csv")

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["accel", "jerk", "snap"])
        w.writerow(literal)

    # Report.  The residual is computed against the *chosen* (snapped or exact)
    # values, so it is the phase error the demod will actually leave.
    raw_cyc, absorbed_cyc = residual_phase(prm, p0, pdot, t_obs, chosen,
                                           target_epoch)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    inside = np.all((ajs >= lo) & (ajs <= hi))

    print(f"par: {par.get('PSRJ', par.get('PSR', '<unnamed>'))}  "
          f"model={info['model']}  F0={1.0/p0:.12g} Hz  p0={p0:.10g} s  "
          f"pdot={pdot:.3g} s/s")
    if info["pb_days"] is not None:
        print(f"orbit: PB={info['pb_days']:.10g} d  A1={info['a1']:.10g} lt-s  "
              f"e={info['e']:.10g}  OM={np.rad2deg(info['omega_peri']):.8g} deg  "
              f"T0={info['t0_mjd']:.12g} MJD  "
              f"(M0={np.rad2deg(info['A_T']):.8g} deg at obs start)")
    else:
        print("orbit: none (isolated pulsar)")
    print(f"obs: N={N}  dt={dt:g} s  T={t_obs:g} s  epoch={epoch_mjd:.12g} MJD  "
          f"anchor={args.anchor}")
    print(f"model: {fit_note}; GR={'off' if args.no_gr else 'on'}  "
          f"m_c={info['companion_mass']:g}  m_p={info['pulsar_mass']:g} Msun")
    print(f"best-fit ajs: accel={ajs[0]:.10g}  jerk={ajs[1]:.10g}  "
          f"snap={ajs[2]:.10g}")
    print(f"grid: {len(arr)} row(s)  "
          f"ranges a[{lo[0]:.6g},{hi[0]:.6g}] "
          f"j[{lo[1]:.6g},{hi[1]:.6g}] s[{lo[2]:.6g},{hi[2]:.6g}]")
    if not args.exact:
        print(f"chosen {'(Cartesian product: per-axis nearest)' if is_product else '(weighted nearest row)'}: "
              f"accel={chosen[0]:.10g}  jerk={chosen[1]:.10g}  "
              f"snap={chosen[2]:.10g}")
        print(f"offsets: da={chosen[0]-ajs[0]:+.4g}  "
              f"dj={chosen[1]-ajs[1]:+.4g}  ds={chosen[2]-ajs[2]:+.4g}")
    if not inside:
        print("WARNING: best-fit ajs lies outside the grid ranges; the chosen "
              "row is on the boundary and the demod will not fully remove the "
              "orbit.")
    print(f"residual phase: raw={raw_cyc:.4g} cycles, "
          f"after spin-search (const+linear) absorption={absorbed_cyc:.4g} cycles")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
