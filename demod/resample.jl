# Julia port of the exact accel/jerk/snap time-remap resampler from
# FFA_stacking/demodulation/utils.py (`resample_ts_shift_snap` and its helpers).
#
# The remap u(tau) = integral_0^tau dt'/(1 + v_inj(t')/c) is evaluated by the
# same term-by-term series the Python original uses, so the two agree to
# rounding.  Polynomials are in *ascending* coefficient order (numpy
# `numpy.polynomial.polynomial` convention), and `tau` is offset from the
# anchor epoch: tau = (i + tstart_ind) * tsamp.

const CONST_C_VAL = 299792458.0
# Two-body / Kepler III constants, matching simulate_orbit_accel_jerk.py.
const CONST_G = 6.67430e-11
const CONST_SOLAR_MASS = 1.989e30

# np.round() already loses 0.5 samples, so a tighter tolerance buys nothing.
const RESAMPLE_TOL_SAMPLES = 0.1
# Each series order is suppressed by another ~|v/c|; the generator refuses
# |v|/c >= 1, so 8 orders is ample.
const RESAMPLE_MAX_ORDER = 8

# --- polynomial helpers (ascending coeffs), matching numpy.polynomial ---

function polyadd(a::AbstractVector, b::AbstractVector)
    out = zeros(max(length(a), length(b)))
    @inbounds for i in 1:length(a)
        out[i] += a[i]
    end
    @inbounds for i in 1:length(b)
        out[i] += b[i]
    end
    return out
end

function polymul(a::AbstractVector, b::AbstractVector)
    out = zeros(length(a) + length(b) - 1)
    @inbounds for j in 1:length(b), i in 1:length(a)
        out[i + j - 1] += a[i] * b[j]
    end
    return out
end

# int_0^x of a polynomial: prepend a zero and divide coeff k by k.
function polyint(p::AbstractVector)
    out = zeros(length(p) + 1)
    @inbounds for i in 1:length(p)
        out[i + 1] = p[i] / i
    end
    return out
end

# Horner, ascending.  x may be a scalar or an array.
function polyval(x::Real, c::AbstractVector)
    acc = float(c[end])
    @inbounds for i in (length(c)-1):-1:1
        acc = acc * float(x) + c[i]
    end
    return acc
end

polyval(x::AbstractVector, c::AbstractVector) = [polyval(xi, c) for xi in x]

# --- velocity / remap model -------------------------------------------------

"""
    _velocity_poly(accel, jerk, snap; v0=0.0)

Ascending coefficients of `v(tau) = v0 + accel*tau + jerk*tau^2/2 +
snap*tau^3/6`, trailing zeros trimmed (interior zeros kept).  Injected sign:
positive = receding.
"""
function _velocity_poly(accel, jerk, snap; v0 = 0.0)
    coeffs = [float(v0), float(accel), float(jerk) / 2.0, float(snap) / 6.0]
    n = length(coeffs)
    while n > 1 && coeffs[n] == 0.0
        n -= 1
    end
    return coeffs[1:n]
end

"""
    _exact_remap_poly(accel, jerk, snap, tau, tsamp; tol_samples, max_order, v0)

Ascending coefficients of the exact de-drift remap `u(tau)`, adding series
orders until the next order's peak contribution over `tau` is below
`tol_samples` samples.  Returns `(u_coeffs, order_used, converged)`.
"""
function _exact_remap_poly(accel, jerk, snap, tau, tsamp;
                           tol_samples = RESAMPLE_TOL_SAMPLES,
                           max_order = RESAMPLE_MAX_ORDER, v0 = 0.0)
    v = -_velocity_poly(accel, jerk, snap; v0 = v0)   # v_inj = -(trial as given)
    if length(v) == 1 && v[1] == 0.0                  # all-zero
        return [0.0, 1.0], 0, true
    end

    vn = copy(v)
    u = polyadd([0.0, 1.0], polyint(vn) ./ CONST_C_VAL)
    order_used = 1
    for order in 2:max_order
        vn = polymul(vn, v)
        term = polyint(vn) ./ CONST_C_VAL^order
        if maximum(abs.(polyval(tau, term))) / tsamp < tol_samples
            return u, order_used, true
        end
        u = polyadd(u, term)
        order_used = order
    end
    return u, order_used, false
end

# --- the resampler ----------------------------------------------------------

"""
    resample_ts_shift_snap(ts, accel, jerk, snap, tsamp, tstart_ind;
                           tol_samples, max_order, v0) -> (ts_new, tstart_new)

Resample `ts` to remove accel/jerk/snap via the exact time remap.  `tstart_ind`
is the first-sample offset in samples from the anchor epoch (negative before
it).  Returns the resampled series and the new starting-index offset.
"""
function resample_ts_shift_snap(ts::AbstractVector, accel, jerk, snap, tsamp,
                                tstart_ind;
                                tol_samples = RESAMPLE_TOL_SAMPLES,
                                max_order = RESAMPLE_MAX_ORDER, v0 = 0.0)
    if accel == 0 && jerk == 0 && snap == 0 && v0 == 0
        return ts, tstart_ind
    end
    tau = (collect(0:(length(ts) - 1)) .+ tstart_ind) .* float(tsamp)
    u_coeffs, order_used, converged = _exact_remap_poly(
        accel, jerk, snap, tau, tsamp;
        tol_samples = tol_samples, max_order = max_order, v0 = v0)
    if !converged
        @warn "resample_ts_shift_snap: exact-remap series did not converge " *
              "to $tol_samples samples within $max_order orders -- pass a " *
              "larger max_order, or shrink the campaign span/velocity."
    end
    ts_new_indices = round.(Int, polyval(tau, u_coeffs) ./ tsamp)
    resampled_tstart_ind = ts_new_indices[1]
    ts_new_indices .-= ts_new_indices[1]
    if minimum(diff(ts_new_indices)) < 0
        error("resample map is not monotonic: the coefficients are too large " *
              "for this offset from the anchor")
    end
    ts_new = zeros(eltype(ts), ts_new_indices[end] + 1)
    ts_new[ts_new_indices .+ 1] = ts            # +1: 0-based Python -> 1-based Julia
    return ts_new, resampled_tstart_ind
end

# Thin wrappers, matching the Python API.
resample_ts_shift_jerk(ts, accel, jerk, tsamp, tstart_ind; kw...) =
    resample_ts_shift_snap(ts, accel, jerk, 0.0, tsamp, tstart_ind; kw...)

resample_ts_shift(ts, accel, tsamp, tstart_ind; kw...) =
    resample_ts_shift_snap(ts, accel, 0.0, 0.0, tsamp, tstart_ind; kw...)

# --- non-polynomial (orbital) remap ----------------------------------------
#
# Port of FFA_stacking/demodulation/utils.py's _integrate_voc_remap /
# resample_ts_shift_voc.  A circular-orbit LOS Doppler is not polynomial in
# tau, so `_exact_remap_poly`'s series cannot represent it; instead
# u(tau) = int_0^tau dt'/(1+voc(t')) is done by cumulative trapezoid, with the
# grid refined x4 until the peak change over one refinement falls below
# `tol_samples` samples.

const RESAMPLE_VOC_N0 = 4000
const RESAMPLE_VOC_NMAX = 2_000_000

# np.linspace(lo, hi, n): n points, endpoints inclusive.
function _linspace(lo, hi, n)
    out = Vector{Float64}(undef, n)
    n == 1 && (out[1] = lo; return out)
    step = (hi - lo) / (n - 1)
    @inbounds for i in 1:n
        out[i] = lo + step * (i - 1)
    end
    return out
end

# np.interp: piecewise linear, clamped to the end values outside [xp[1], xp[end]].
function _interp(x::Real, xp::AbstractVector, fp::AbstractVector)
    n = length(xp)
    x <= xp[1] && return fp[1]
    x >= xp[n] && return fp[n]
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if xp[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])
end

_interp(x::AbstractVector, xp, fp) = [_interp(xi, xp, fp) for xi in x]

# scipy.integrate.cumulative_trapezoid(y, x, initial=0.0).
function _cumtrapz(y::AbstractVector, x::AbstractVector)
    n = length(y)
    out = Vector{Float64}(undef, n)
    out[1] = 0.0
    @inbounds for i in 2:n
        out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    end
    return out
end

"""
    _integrate_voc_remap(voc_func, tau, tol_samples, tsamp; n0, n_max)

`u(tau) = int_0^tau dt'/(1+voc_func(t'))` by cumulative trapezoid on
`[min(0,tau), max(0,tau)]`, quadrupling the sample count until the peak change
over one refinement is below `tol_samples` samples.
"""
function _integrate_voc_remap(voc_func, tau::AbstractVector, tol_samples, tsamp;
                              n0 = RESAMPLE_VOC_N0, n_max = RESAMPLE_VOC_NMAX)
    lo = min(0.0, minimum(tau))
    hi = max(0.0, maximum(tau))
    hi == lo && return zeros(length(tau))
    n = n0
    prev = nothing
    while true
        t_grid = _linspace(lo, hi, n)
        cum = _cumtrapz(1.0 ./ (1.0 .+ voc_func(t_grid)), t_grid)
        u = _interp(tau, t_grid, cum) .- _interp(0.0, t_grid, cum)
        if prev !== nothing && maximum(abs.(u .- prev)) / tsamp < tol_samples
            return u
        end
        if n >= n_max
            @warn "_integrate_voc_remap: quadrature did not converge to " *
                  "$tol_samples samples within $n points -- result may carry " *
                  "residual error."
            return u
        end
        prev = u
        n = min(n * 4, n_max)
    end
end

"""
    resample_ts_shift_voc(ts, voc_func, tsamp, tstart_ind; tol_samples) -> (ts_new, tstart_new)

Remove an arbitrary (non-polynomial) LOS Doppler track `voc_func(tau) = v_inj(tau)/c`
via the exact remap `u(tau) = int_0^tau dt'/(1+voc(tau'))`.  `voc_func` is
injected sign (positive = receding) and `tau` is measured from the anchor
(data midpoint), as in `resample_ts_shift_snap`.  Same index-remap convention
and output as the polynomial remapper.
"""
function resample_ts_shift_voc(ts::AbstractVector, voc_func, tsamp, tstart_ind;
                               tol_samples = RESAMPLE_TOL_SAMPLES)
    isempty(ts) && return ts, tstart_ind
    tau = (collect(0:(length(ts) - 1)) .+ tstart_ind) .* float(tsamp)
    u = _integrate_voc_remap(voc_func, tau, tol_samples, tsamp)
    ts_new_indices = round.(Int, u ./ tsamp)
    resampled_tstart_ind = ts_new_indices[1]
    ts_new_indices .-= ts_new_indices[1]
    if minimum(diff(ts_new_indices)) < 0
        error("resample map is not monotonic: voc_func implies |v|/c >= 1 " *
              "somewhere in this segment's span")
    end
    ts_new = zeros(eltype(ts), ts_new_indices[end] + 1)
    ts_new[ts_new_indices .+ 1] = ts
    return ts_new, resampled_tstart_ind
end

"""
    circular_voc(x_lt_s, pb_s, A_T; t_offset_s = 0.0) -> tau -> v_inj(tau)/c

Injected-sign LOS Doppler of a pure Keplerian circular orbit (e = 0, Roemer
only, no GR), as a function of `tau` measured from the data-midpoint anchor:

    delay  Δ(t) = x*sin(ω_b*t + A_T),      x = a_p sin(i)/c  [light-seconds]
    v/c         = dΔ/dt = x*ω_b*cos(ω_b*t + A_T)

`t_offset_s` shifts the midpoint anchor to the epoch at which `A_T` is defined
(`pulsegen_gr.py -anchor`): T/2 for `start` (the .inf epoch, pulsegen's
default), 0 for `midpoint`.  `A_T` absorbs the (degenerate, for e = 0)
argument of periastron.
"""
function circular_voc(x_lt_s, pb_s, A_T; t_offset_s = 0.0)
    omega_b = 2 * pi / pb_s
    return tau -> x_lt_s * omega_b .* cos.(omega_b .* (tau .+ t_offset_s) .+ A_T)
end
