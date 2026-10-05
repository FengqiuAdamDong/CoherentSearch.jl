# Julia port of the exact accel/jerk/snap time-remap resampler from
# FFA_stacking/demodulation/utils.py (`resample_ts_shift_snap` and its helpers).
#
# The remap u(tau) = integral_0^tau dt'/(1 + v_inj(t')/c) is evaluated by the
# same term-by-term series the Python original uses, so the two agree to
# rounding.  Polynomials are in *ascending* coefficient order (numpy
# `numpy.polynomial.polynomial` convention), and `tau` is offset from the
# anchor epoch: tau = (i + tstart_ind) * tsamp.

const CONST_C_VAL = 299792458.0

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
