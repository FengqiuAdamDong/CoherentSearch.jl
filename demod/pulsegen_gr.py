#!/usr/bin/env python
"""
Inject a pulsar in a relativistic (double-neutron-star) binary into a PRESTO
.dat/.inf, using the full Damour-Deruelle (DD) time-delay model.

The observed pulse arrival time is

    t_arr = T + Δ_R(T) + Δ_E(T) + Δ_S(T)

where T is the pulsar's proper time (which carries the intrinsic rotation
phase) and the three delay terms are, as functions of the eccentric anomaly E
and periastron longitude ω:

    Δ_R = x [ sin ω (cos E − e_r) + √(1 − e_θ²) cos ω sin E ]     (Roemer)
    Δ_E = γ sin E                                                 (Einstein)
    Δ_S = −2r ln{1 − e cos E − s[sin ω (cos E − e) + √(1−e²) cos ω sin E]}
                                                                  (Shapiro)

with the DD post-Keplerian parameters (Damour & Deruelle 1986), computed from
the two masses by pk_coeffs() in the sibling FFA_stacking repo:

    ω̇   periastron advance           γ   Einstein-delay amplitude
    ṅ    orbital decay (Ṗb)          r   Shapiro range  (= T_sun m_c)
    e_r = e(1+δ_r), e_θ = e(1+δ_θ)    s   Shapiro shape  (= sin i)
    M(T) = A_T + n T + ½ ṅ T²         ω(T) = ω_peri + ω̇ T

The intrinsic rotation phase is φ(T) = T/P0 − Ṗ0 T²/(2 P0²); pulse k is
emitted at the proper time T_k solving φ(T_k) = k, and lands at t_arr above.
Pulses are added to the series as Gaussians.

This is a DELAY model (the physically correct injection for a timing/DD fit),
not the velocity/period modulation that generate_search_timeseries.py's
orbital modes use.  Its observable is the arrival-time shift; dΔ/dt equals the
sibling code's los_velocity(gr=True)/c, which the validation checks.

Usage:
  python demod/pulsegen_gr.py -inf OBS.inf -outdir OUT -outbasename inj \
      -p0 1.0 -pb 0.5 -e 0.3 -omega_peri 0.3 -sin_i 0.5 -A_T 0.0 \
      -companion_mass 1.4 -pulsar_mass 1.4 -snr 30 -pulse_width 0.01 [-real OBS.dat]
"""
import argparse
import os
import sys

import numpy as np
import yaml

# Physics helpers from the sibling FFA_stacking repo (same code the
# orbit-simulation generators use, so the DD model cannot drift from theirs).
_FFA_REPO = os.environ.get("FFA_REPO", "/home/fadong/Documents/FFA_stacking")
if os.path.join(_FFA_REPO, "orbit_simulation") not in sys.path:
    sys.path.insert(0, os.path.join(_FFA_REPO, "orbit_simulation"))

from simulate_orbit_accel_jerk import (  # noqa: E402
    C,
    G,
    SOLAR_MASS,
    T_SUN,
    pk_coeffs,
    solve_kepler_equation,
)

FACTORIALS = [1.0, 2.0, 6.0]  # k! for the tau**k/k! accel/jerk/snap basis

# ---------------------------------------------------------------------------
# Observation I/O
# ---------------------------------------------------------------------------

def read_inf(path):
    """(N, dt, epoch_mjd, lines) from a PRESTO .inf."""
    N = dt = epoch = None
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
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
    if N is None or dt is None or epoch is None:
        raise ValueError(f"{path}: could not read N/dt/epoch")
    return N, dt, epoch, lines


def write_inf(src_lines, outpath, basenm, N, epoch):
    """Copy the source .inf, updating the data-file name, sample count and
    epoch.  Everything else (radio metadata, breaks) is preserved."""
    with open(outpath, "w") as fh:
        for line in src_lines:
            if "=" not in line:
                fh.write(line)
                continue
            key = line.split("=", 1)[0].strip()
            if key.startswith("Data file name"):
                fh.write(f" {key:<38} =  {basenm}\n")
            elif key.startswith("Number of bins"):
                fh.write(f" {key:<38} =  {N:<11d}\n")
            elif key.startswith("Epoch of observation"):
                fh.write(f" {key:<38} =  {epoch:05.15f}\n")
            else:
                fh.write(line)


# ---------------------------------------------------------------------------
# Damour-Deruelle model
# ---------------------------------------------------------------------------

def projected_semi_major_axis(pb_s, sin_i, companion_mass, pulsar_mass):
    """x = a_p sin(i)/c in light-seconds, proper two-body split (pulsar orbit
    a_p = a_total * m_c/M); companion-dominates would be a_total*sin_i."""
    M = companion_mass + pulsar_mass
    a_tot = (G * M * SOLAR_MASS / (4 * np.pi ** 2) * pb_s ** 2) ** (1 / 3)
    return sin_i * a_tot * (companion_mass / M) / C


class DDParams:
    """The full DD delay parameter set, in SI (delays in seconds)."""

    def __init__(self, pb_s, x, e, omega_peri, A_T,
                 omega_dot, n_dot, gamma, r, s, delta_r, delta_theta):
        self.pb_s = pb_s
        self.n_b = 2 * np.pi / pb_s
        self.x = x
        self.e = e
        self.omega_peri = omega_peri
        self.A_T = A_T
        self.omega_dot = omega_dot
        self.n_dot = n_dot
        self.gamma = gamma
        self.r = r
        self.s = s
        self.delta_r = delta_r
        self.delta_theta = delta_theta


def build_dd_params(args):
    """DD parameters from the two masses (pk_coeffs) with optional overrides."""
    pb_s = args.pb * 86400.0
    n_b = 2 * np.pi / pb_s
    x = args.x if args.x is not None else projected_semi_major_axis(
        pb_s, args.sin_i, args.companion_mass, args.pulsar_mass)
    omega_dot, n_dot, delta_r, delta_theta = pk_coeffs(
        n_b, args.e, args.companion_mass, args.pulsar_mass)
    gamma = (args.gamma if args.gamma is not None else
             args.e * n_b ** (-1 / 3) * T_SUN ** (2 / 3)
             * args.companion_mass * (args.pulsar_mass + 2 * args.companion_mass)
             / (args.companion_mass + args.pulsar_mass) ** (4 / 3))
    r = args.shapiro_r if args.shapiro_r is not None else T_SUN * args.companion_mass
    if args.omega_dot is not None:
        omega_dot = args.omega_dot
    if args.n_dot is not None:
        n_dot = args.n_dot
    s = args.shapiro_s if args.shapiro_s is not None else args.sin_i
    return DDParams(pb_s, x, args.e, args.omega_peri, args.A_T,
                    omega_dot, n_dot, gamma, r, s, delta_r, delta_theta)


def dd_delay(T, prm, components=False):
    """Total DD delay [s] at pulsar proper time T (s since the reference
    epoch); returns (Δ_R, Δ_E, Δ_S, total) if components else total."""
    M = prm.A_T + prm.n_b * T + 0.5 * prm.n_dot * T ** 2
    E = solve_kepler_equation(M, prm.e) if prm.e > 0.0 else M
    omega = prm.omega_peri + prm.omega_dot * T
    e_r = prm.e * (1.0 + prm.delta_r)
    e_theta = prm.e * (1.0 + prm.delta_theta)
    sinE, cosE = np.sin(E), np.cos(E)
    so, co = np.sin(omega), np.cos(omega)
    roemer = prm.x * (so * (cosE - e_r) + np.sqrt(1.0 - e_theta ** 2) * co * sinE)
    einstein = prm.gamma * sinE
    arg = 1.0 - prm.e * cosE - prm.s * (
        so * (cosE - prm.e) + np.sqrt(1.0 - prm.e ** 2) * co * sinE)
    arg = np.clip(arg, 1e-10, None)  # eclipse guard (log divergence)
    shapiro = -2.0 * prm.r * np.log(arg)
    total = roemer + einstein + shapiro
    return (roemer, einstein, shapiro, total) if components else total


def max_abs_delay(prm):
    """Upper bound on |delay| over the orbit, for sizing the pulse-index range."""
    shapiro_max = 2.0 * abs(prm.r) * abs(np.log(max(1e-12, 1.0 - abs(prm.s))))
    return abs(prm.x) + abs(prm.gamma) + shapiro_max


def fit_kinematic_coefficients(tau, v, terms=(True, True, True)):
    """LSQ fit of v_model(tau) = v0 + accel*tau + jerk*tau^2/2 + snap*tau^3/6
    with the tau**k/k! basis and free v0 -- same convention as
    generate_search_timeseries.py's fit_kinematic_coefficients and
    kinematic_grid_spacing.fit_velocity_model, so the coefficients are directly
    comparable with a kinematic_finder candidate.  Terms left out are not
    fitted (their coefficient is zero).  Returns a dict with the coefficients
    and the truncation residual (max/rms |v - fit|)."""
    tau = np.asarray(tau, dtype=float)
    v = np.asarray(v, dtype=float)
    cols = [np.ones_like(tau)]
    cols += [tau ** (k + 1) / FACTORIALS[k] for k in range(3) if terms[k]]
    A = np.column_stack(cols)
    scale = np.max(np.abs(A), axis=0)
    scale[scale == 0.0] = 1.0  # degenerate span: unscaled
    coef = np.linalg.lstsq(A / scale, v, rcond=None)[0] / scale
    resid = v - A @ coef
    ajs = np.zeros(3)
    ajs[np.asarray(terms, dtype=bool)] = coef[1:]
    return dict(v0=float(coef[0]), accel=float(ajs[0]), jerk=float(ajs[1]),
                snap=float(ajs[2]),
                max_residual_v=float(np.max(np.abs(resid))),
                rms_residual_v=float(np.sqrt(np.mean(resid ** 2))))


def best_fit_kinematic(prm, p0, pdot, t_obs, t_ref, t_anchor=0.0,
                       terms=(True, True, True), n_fit=2000):
    """ONE least-squares accel/jerk/snap fit to the DD-injected pulse train over
    the whole observation, in the searches' v = c*(P_obs/p0 - 1) convention.

    The delay model gives P_obs/P0 = 1 + dDelta/dt, so the fractional LOS
    Doppler is exactly the numerical derivative of dd_delay.  tau is measured
    from `t_anchor` (the epoch the coefficients are reported at): by default the
    observation midpoint, matching the kinematic searches' -anchor midpoint.
    The fit residual is the cubic-truncation error of the DD model over this
    span -- irreducible for an eccentric/GR orbit."""
    t = np.linspace(0.0, t_obs, n_fit)
    h = max(t_obs, 1.0) * 1e-6
    dd_delay_dot = (dd_delay(t + h, prm) - dd_delay(t - h, prm)) / (2.0 * h)
    # P_obs/P0 = (1 + pdot*T/p0) * (1 + dDelta/dT), as the searches see it.
    v_over_c = (1.0 + (pdot / p0) * t) * (1.0 + dd_delay_dot) - 1.0
    return fit_kinematic_coefficients(
        t - t_anchor, C * v_over_c, terms=terms)


def pulse_arrival_times(prm, p0, pdot, t_obs, t_ref=0.0):
    """Arrival times [s since obs start] of every pulse landing in [0, t_obs].

    Pulse k is emitted at the proper time T_k with φ(T_k) = T_k/p0 −
    Ṗ0 T_k²/(2 p0²) = k, and arrives at t_ref + T_k + Δ(T_k)."""
    span = t_obs - t_ref
    kmin = int(np.floor((0.0 - t_ref - max_abs_delay(prm)) / p0)) - 1
    kmax = int(np.ceil((t_obs - t_ref + max_abs_delay(prm)) / p0)) + 1
    k = np.arange(kmin, kmax + 1, dtype=float)
    T = k * p0
    for _ in range(8):  # solve T = k p0 + pdot T²/(2 p0) to machine precision
        T = k * p0 + pdot * T ** 2 / (2.0 * p0)
    t_arr = t_ref + T + dd_delay(T, prm)
    return np.sort(t_arr[(t_arr >= 0.0) & (t_arr <= t_obs)])


def inject_gaussian_pulses(data, pulse_times, tsamp, pulse_snr, pulse_width):
    """Add Gaussian pulses (sigma = pulse_width s, peak = pulse_snr * std) to
    1D float32 data in place.  Kernel built once for the fixed half-window."""
    std = np.std(data)
    height = pulse_snr * std
    width_bins = max(1, int(round(pulse_width / tsamp)))
    half_window = 5 * width_bins
    n = len(data)
    off = np.arange(-half_window, half_window)
    k = height * np.exp(-0.5 * (off / width_bins) ** 2)
    for ptoa in pulse_times:
        centre_bin = int(round(ptoa / tsamp))
        t_start = max(0, centre_bin - half_window)
        t_end = min(n, centre_bin + half_window)
        if t_start >= t_end:
            continue
        lo = t_start - (centre_bin - half_window)
        hi = lo + (t_end - t_start)
        data[t_start:t_end] += k[lo:hi]
    return data


# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-inf", type=str, required=True,
                    help="PRESTO .inf of the observation to inject into "
                         "(supplies N, dt, epoch).")
    ap.add_argument("-real", type=str, default=None,
                    help="Real .dat to draw the noise background from (a "
                         "random N-sample slice, seeded by -seed). Default: "
                         "Gaussian noise.")
    ap.add_argument("-outdir", type=str, default=".", help="Output directory.")
    ap.add_argument("-outbasename", type=str, default="pulsegr",
                    help="Output basename.")
    ap.add_argument("-seed", type=int, default=None, help="Random seed.")

    g_psr = ap.add_argument_group("pulsar / signal")
    g_psr.add_argument("-p0", type=float, default=1.0,
                       help="Intrinsic spin period at the reference epoch [s].")
    g_psr.add_argument("-pdot", type=float, default=0.0,
                       help="Intrinsic period derivative [s/s].")
    g_psr.add_argument("-snr", type=float, default=30.0,
                       help="Total integrated S/N per observation.")
    g_psr.add_argument("-pulse_width", type=float, default=0.01,
                       help="Gaussian pulse sigma [s].")
    g_psr.add_argument("-gaus_noise_std", type=float, default=1.0,
                       help="Std of the Gaussian noise when -real is not given.")

    g_orb = ap.add_argument_group("binary / DD model")
    g_orb.add_argument("-pb", type=float, default=0.5,
                       help="Orbital period [days].")
    g_orb.add_argument("-e", type=float, default=0.3, help="Eccentricity.")
    g_orb.add_argument("-omega_peri", type=float, default=0.3,
                       help="Argument of periastron [rad].")
    g_orb.add_argument("-A_T", type=float, default=0.0,
                       help="Mean anomaly at the reference epoch [rad].")
    g_orb.add_argument("-sin_i", type=float, default=0.5,
                       help="sin(inclination) (Shapiro shape s, and scales x "
                            "unless -x is given).")
    g_orb.add_argument("-x", type=float, default=None,
                       help="Projected semi-major axis a_p sin(i)/c [light-s]; "
                            "overrides the value derived from masses/sin_i.")
    g_orb.add_argument("-companion_mass", type=float, default=1.4,
                       help="Companion mass [Msun].")
    g_orb.add_argument("-pulsar_mass", type=float, default=1.4,
                       help="Pulsar mass [Msun].")
    g_orb.add_argument("-anchor", choices=("start", "midpoint"), default="start",
                       help="Reference epoch for p0, A_T and the delays: "
                            "'start' = the .inf epoch (first sample), "
                            "'midpoint' = obs start + T/2.")
    g_orb.add_argument("-fit_anchor", choices=("start", "midpoint"),
                       default="midpoint",
                       help="Epoch the reported best-fit accel/jerk/snap "
                            "coefficients are expanded about (midpoint matches "
                            "the kinematic searches).")
    g_orb.add_argument("-terms", nargs="+", default=["all"],
                       choices=("accel", "jerk", "snap", "all"),
                       help="Which of accel/jerk/snap the best-fit model fits "
                            "(any subset, or 'all').")
    # Post-Keplerian overrides (default: computed from the masses).
    g_orb.add_argument("-omega_dot", type=float, default=None,
                       help="Override periastron advance [rad/s].")
    g_orb.add_argument("-n_dot", type=float, default=None,
                       help="Override orbital decay dn/dt [rad/s^2].")
    g_orb.add_argument("-gamma", type=float, default=None,
                       help="Override Einstein-delay amplitude [s].")
    g_orb.add_argument("-shapiro_r", type=float, default=None,
                       help="Override Shapiro range r [s].")
    g_orb.add_argument("-shapiro_s", type=float, default=None,
                       help="Override Shapiro shape s = sin(i).")

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if not (0.0 <= args.e < 1.0):
        ap.error("-e must satisfy 0 <= e < 1")

    N, dt, epoch, inf_lines = read_inf(args.inf)
    t_obs = N * dt
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prm = build_dd_params(args)
    t_ref = 0.0 if args.anchor == "start" else t_obs / 2.0
    t_fit_anchor = 0.0 if args.fit_anchor == "start" else t_obs / 2.0
    fit_terms = tuple(n in args.terms or "all" in args.terms
                      for n in ("accel", "jerk", "snap"))

    # Best-fit accel/jerk/snap the searches should recover for this injection.
    fit = best_fit_kinematic(prm, args.p0, args.pdot, t_obs, t_ref,
                             t_anchor=t_fit_anchor, terms=fit_terms)

    # Noise background.
    if args.real is not None:
        raw = np.fromfile(args.real, dtype=np.float32)
        if len(raw) < N:
            ap.error(f"-real {args.real} has {len(raw)} samples, need {N}")
        start = int(rng.integers(0, len(raw) - N + 1))
        data = raw[start:start + N].copy()
    else:
        data = rng.normal(0.0, args.gaus_noise_std, N).astype(np.float32)

    pulse_times = pulse_arrival_times(prm, args.p0, args.pdot, t_obs, t_ref)
    n_pulses = len(pulse_times)
    pulse_snr = args.snr / np.sqrt(t_obs / args.p0)
    inject_gaussian_pulses(data, pulse_times, dt, pulse_snr, args.pulse_width)

    dat_path = os.path.join(args.outdir, args.outbasename + ".dat")
    data.tofile(dat_path)
    inf_path = os.path.join(args.outdir, args.outbasename + ".inf")
    write_inf(inf_lines, inf_path, args.outbasename, N, epoch)

    K_c = prm.n_b * prm.x / np.sqrt(1.0 - args.e ** 2)  # RV semi-amplitude / c
    truth = dict(
        model="DD_full",
        inf=os.path.abspath(args.inf),
        mjd_epoch=float(epoch),
        t_ref_s=float(t_ref),
        anchor=args.anchor,
        p0=float(args.p0), pdot=float(args.pdot),
        t_obs_s=float(t_obs), tsamp_s=float(dt), n_samples=int(N),
        n_pulses_injected=int(n_pulses),
        snr=float(args.snr), pulse_width=float(args.pulse_width),
        pb_days=float(args.pb), pb_s=float(prm.pb_s),
        x_lt_s=float(prm.x), e=float(args.e),
        omega_peri_rad=float(args.omega_peri), A_T_rad=float(args.A_T),
        sin_i=float(args.sin_i),
        companion_mass=float(args.companion_mass),
        pulsar_mass=float(args.pulsar_mass),
        omega_dot_rad_s=float(prm.omega_dot),
        n_dot_rad_s2=float(prm.n_dot),
        gamma_s=float(prm.gamma),
        shapiro_r_s=float(prm.r),
        shapiro_s=float(prm.s),
        delta_r=float(prm.delta_r),
        delta_theta=float(prm.delta_theta),
        K_over_c=float(K_c),
        fit_anchor=args.fit_anchor,
        fit_anchor_s=float(t_fit_anchor),
        fit_terms="+".join(n for n, m in zip(("accel", "jerk", "snap"),
                                            fit_terms) if m),
        best_fit_accel=fit["accel"],
        best_fit_jerk=fit["jerk"],
        best_fit_snap=fit["snap"],
        best_fit_v0=fit["v0"],
        best_fit_max_resid_velocity=fit["max_residual_v"],
        best_fit_rms_resid_velocity=fit["rms_residual_v"],
    )
    yaml_path = os.path.join(args.outdir, args.outbasename + "_truth.yaml")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(truth, fh, sort_keys=False)

    max_abs = max_abs_delay(prm)
    print(f"Wrote {dat_path} ({N} samples, {n_pulses} pulses, "
          f"per-pulse S/N {pulse_snr:.3f})")
    print(f"Wrote {inf_path}")
    print(f"Wrote {yaml_path}")
    print(f"DD model: Pb={args.pb:g} d, x={prm.x:.6g} lt-s (K/c={K_c:.4e}), "
          f"e={args.e:g}, sin_i={args.sin_i:g}")
    print(f"  omega_dot={prm.omega_dot:.4e} rad/s, n_dot={prm.n_dot:.4e} rad/s^2, "
          f"gamma={prm.gamma:.6e} s, r={prm.r:.6e} s, s={prm.s:.6g}")
    print(f"  max |delay| over orbit: {max_abs:.4g} s "
          f"({100.0 * max_abs / t_obs:.3g}% of T={t_obs:g} s)")
    print(f"Best-fit kinematic model (anchor {args.fit_anchor}, "
          f"terms {truth['fit_terms']}):")
    print(f"  accel={fit['accel']:.6e} m/s^2  jerk={fit['jerk']:.6e} m/s^3  "
          f"snap={fit['snap']:.6e} m/s^4")
    print(f"  residual (truncation): max {fit['max_residual_v']:.4e} m/s, "
          f"rms {fit['rms_residual_v']:.4e} m/s")


if __name__ == "__main__":
    main()
