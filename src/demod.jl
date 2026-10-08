# Time-domain demodulation and the PRESTO FFT/rednoise steps, all in memory.
#
# This is the in-process half of the `demod/` pipeline.  Before this file the
# chain for one grid point shelled out to disk three times: `demod_dat.jl` wrote
# a `.dat`, PRESTO's `realfft` wrote a `.fft`, PRESTO's `rednoise` wrote a
# `_red.fft`, and only then did the search read that back.  For a sweep of 220
# grid points that is ~1100 files written and read again, none of them wanted.
# Everything here is a pure function of its arguments and touches no files, so
# one process can hold the observation in RAM and run the whole chain per point.
#
# Three groups of code live here:
#
#   1. the exact accel/jerk/snap and circular-orbit time remaps (the Julia port
#      of FFA_stacking/demodulation/utils.py that used to be `demod/resample.jl`);
#   2. `presto_fft_amps` / `presto_deredden!`, ports of PRESTO's `.fft` packing
#      and of `dered_engine` (misc_utils.c);
#   3. `demodulate_series` + the grid CSV reader, which is what the driver in
#      `demodsearch.jl` loops over.
#
# The ports are pinned to the binaries, not to the C sources' intent: see
# `presto_deredden!` for the one place those disagree.

using FFTW: FFTW

# ---------------------------------------------------------------------------
# Exact time-remap resampler
#
# The remap u(tau) = integral_0^tau dt'/(1 + v_inj(t')/c) is evaluated by the
# same term-by-term series the Python original uses, so the two agree to
# rounding.  Polynomials are in *ascending* coefficient order (numpy
# `numpy.polynomial.polynomial` convention), and `tau` is offset from the
# anchor epoch: tau = (i + tstart_ind) * tsamp.
# ---------------------------------------------------------------------------

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

# x = a_p sin(i)/c [light-seconds] by Kepler III with the proper two-body
# split (pulsar orbit a_p = a_tot*m_c/M), the same formula pulsegen_gr.py and
# nsns_grid.py use.  A_T (mean anomaly) absorbs the argument of periastron for
# a circular orbit.
function projected_x(pb_s, sin_i, companion_mass, pulsar_mass)
    M = companion_mass + pulsar_mass
    a_tot = (CONST_G * M * CONST_SOLAR_MASS / (4 * pi^2) * pb_s^2)^(1 / 3)
    return sin_i * a_tot * (companion_mass / M) / CONST_C_VAL
end

# ---------------------------------------------------------------------------
# PRESTO `.fft` packing, in memory
# ---------------------------------------------------------------------------

"""
    presto_fft_amps(x, plans::Dict{Int,Any}) -> Vector{ComplexF32}

Pack the real FFT of the `Float32` time series `x` exactly as PRESTO's
`.fft` layout has it: `N÷2` interleaved `ComplexF32` bins in which bin 1 packs
the DC term (`sum(x)`) in its real part and the Nyquist term in its imaginary
part, and bins 2..N÷2 are the positive-frequency amplitudes.

`plans` caches one `FFTW.plan_rfft` per length, keyed by `N`; the plan is built
with `FFTW.ESTIMATE`.  `MEASURE` was measured at **35.5 s** for `N = 6e6`
against 15 ms for `ESTIMATE`, and the chain is re-planned for every grid point's
own (varying) length, so it is not affordable here.  `rfft` is the right engine:
it agrees with PRESTO's `realfft` to a worst-bin relative amplitude of 1.2e-3,
and *exactly* on the Nyquist value, where numpy's pocketfft does not.
"""
function presto_fft_amps(x::AbstractVector{Float32}, plans::Dict{Int,Any})
    N = length(x)
    iseven(N) || error("The number of time samples must be even (realfft requires it), got $N")
    H = N ÷ 2
    plan = get!(plans, N) do
        FFTW.plan_rfft(x; flags = FFTW.ESTIMATE)
    end
    X = plan * x
    a = Vector{ComplexF32}(undef, H)
    a[1] = ComplexF32(real(X[1]), real(X[H+1]))
    @inbounds for i in 2:H
        a[i] = ComplexF32(X[i])
    end
    return a
end

# ---------------------------------------------------------------------------
# PRESTO's rednoise, in memory
# ---------------------------------------------------------------------------

"""
    presto_invsqrt(x::Float32) -> Float32

The *bit-exact* fast inverse square root from PRESTO's `misc_utils.c:291`
(`0x5f3759df`, one Newton step).  Not `1/sqrt(x)`: the exact reciprocal is
`1.001` times this on average, and `dered_engine` uses this one, so a port that
"fixes" it would not reproduce `rednoise`'s output.
"""
@inline function presto_invsqrt(x::Float32)::Float32
    i = reinterpret(Int32, x)
    i = 0x5f3759df - (i >> 1)
    y = reinterpret(Float32, i)
    return y * (1.5f0 - 0.5f0 * x * y * y)
end

# PRESTO's `median()` (src/median.c) calls `gsl_stats_float_median`, but the
# `libpresto.so` actually linked returns the LOWER order statistic for even n,
# not GSL's documented mean of the two central values -- verified by calling
# `libpresto.so:median` directly on sorted data (n = 6, 8, 100).  This is not a
# detail: the mean-of-two form puts every block's median ~7% low on a
# log-distributed power spectrum, which is a silent 7% error in every S/N.
# The `$REDNOISE`-gated assertion in `test/test_demod.jl` is what catches it.
function _lower_median!(buf::AbstractVector{Float32}, n::Integer)
    n > 0 || throw(ArgumentError("cannot take the median of zero elements"))
    v = view(buf, 1:n)
    sort!(v)
    return @inbounds v[(n + 1) >> 1]
end

"""
    presto_deredden!(a::Vector{ComplexF32}, T; startwidth=6, endwidth=100,
                     endfreq=6.0) -> Int

Port of PRESTO's `dered_engine` (`src/misc_utils.c:900-1050`), operating in
place on a whole `N÷2`-bin `.fft` amplitude array, and returning the number of
bins it wrote.  `T` is the length of the original time series in seconds
(`N*dt`) -- of the *undemodulated* series, matching what `rednoise` reads from
the `.inf`.

The algorithm walks the band in blocks whose length grows logarithmically
(`startwidth*log(binnum)`, capped at `endwidth`, and frozen at `endwidth` once
`binnum/T >= endfreq`) and divides each bin by the square root of a running
median of the local power, linearly interpolated between block midpoints by
`dslope` over the second half of the old block and the first half of the new
one.  DC is forced to `1 + 0im`.

The defaults are `rednoise_cmd.c`'s.  Bin positions are tracked **0-based** as
in the C, so the index arithmetic can be read against the source line for line;
`binnum` is the count of bins consumed so far and doubles as a frequency over
`T`.

Two deliberate `Float64` widenings, both required to match the binary: the slope
arithmetic (`mean_old`, `mean_new`, `dslope`, and the `mean_old + dslope*ii`
argument, converted to `Float32` only at the `presto_invsqrt` call) is
`Float64` here, and `T` is widened through `Float32` once exactly as C's
`const float Tf = (float) T;`.  Forming that argument in `Float32` throughout,
as the C source does, measures 1.9e-4 against PRESTO where this measures 2e-7.

A full run returns `length(a)` — every bin, DC included, is rewritten.  A
smaller count means the walk stopped early (the C code processes `numbins` and
stops), and the caller must treat the unwritten tail as unnormalised.
"""
function presto_deredden!(a::Vector{ComplexF32}, T::Real;
                          startwidth::Integer = 6, endwidth::Integer = 100,
                          endfreq::Real = 6.0)
    numbins = length(a)
    numbins >= 1 || return 0
    Tf = Float32(T)
    # The C code's `inbuf1`/`inbuf2`/`outbuf`/`powbuf`, reused and swapped
    # rather than reallocated.  Sized for whichever of startwidth/endwidth is
    # larger, since the first block is `startwidth` wide before the cap applies.
    bufsz = max(Int(endwidth), Int(startwidth))
    oldbuf = Vector{ComplexF32}(undef, bufsz)
    newbuf = Vector{ComplexF32}(undef, bufsz)
    outbuf = Vector{ComplexF32}(undef, bufsz)
    powbuf = Vector{Float32}(undef, bufsz)

    # DC (with Nyquist packed into its imaginary part) is pinned to 1 + 0im and
    # never revisited: the running normalisation is relative, so the
    # zero-frequency power has to be fixed for the medians to mean anything.
    a[1] = ComplexF32(1, 0)
    numwrote = 1
    rpos = 1
    wpos = 1

    binnum = 1
    nblk_old = Int(startwidth)
    # The C code errors if the first read is short, so a file with nothing but
    # the DC bin has no normalisation to do.
    rpos + nblk_old > numbins && return numwrote
    @inbounds for ii in 1:nblk_old
        oldbuf[ii] = a[rpos + ii]
    end
    rpos += nblk_old
    mid_old = nblk_old >> 1

    @inbounds for ii in 1:nblk_old
        powbuf[ii] = real(oldbuf[ii])^2 + imag(oldbuf[ii])^2
    end
    mean_old = Float64(_lower_median!(powbuf, nblk_old)) / log(2.0)

    # The first half of the first block has no slope: only a few bins, and the
    # C code says that is probably OK.
    norm = presto_invsqrt(Float32(mean_old))
    @inbounds for ii in 1:mid_old
        outbuf[ii] = oldbuf[ii] * norm
    end
    @inbounds for ii in 1:mid_old
        a[wpos+ii] = outbuf[ii]
    end
    wpos += mid_old
    numwrote += mid_old

    binnum += nblk_old
    bufflen = min(Int(trunc(startwidth * log(binnum))), Int(endwidth))

    # `dslope` is 1.0 in the C declaration and reaches the final partial block
    # unchanged only if the while loop never ran.
    dslope = 1.0
    while true
        nblk_new = min(bufflen, numbins - rpos)
        nblk_new <= 0 && break
        @inbounds for ii in 1:nblk_new
            newbuf[ii] = a[rpos + ii]
        end
        rpos += nblk_new
        mid_new = nblk_new >> 1

        @inbounds for ii in 1:nblk_new
            powbuf[ii] = real(newbuf[ii])^2 + imag(newbuf[ii])^2
        end
        mean_new = Float64(_lower_median!(powbuf, nblk_new)) / log(2.0)

        dslope = (mean_new - mean_old) / (0.5 * (nblk_old + nblk_new))

        # The second half of the old block, then the first half of the new one,
        # both against the ramp from the old median to the new: the ramp's
        # coordinate `ii` starts at 0 at the old block's midpoint and reaches
        # the new block's midpoint (ii = mid_new + nblk_old - mid_old) at
        # mean_new.
        ii = 0
        @inbounds for ind in (mid_old+1):nblk_old
            ii += 1
            outbuf[ii] = oldbuf[ind] * presto_invsqrt(Float32(mean_old + dslope * (ii - 1)))
        end
        @inbounds for ind in 1:mid_new
            ii += 1
            outbuf[ii] = newbuf[ind] * presto_invsqrt(Float32(mean_old + dslope * (ii - 1)))
        end
        @inbounds for jj in 1:ii
            a[wpos+jj] = outbuf[jj]
        end
        wpos += ii
        numwrote += ii

        binnum += nblk_new
        if Float32(binnum) / Tf < Float32(endfreq)
            bufflen = min(Int(trunc(startwidth * log(binnum))), Int(endwidth))
        else
            bufflen = Int(endwidth)
        end
        oldbuf, newbuf = newbuf, oldbuf
        nblk_old = nblk_new
        mean_old = mean_new
        mid_old = mid_new
    end

    # The last partial block, assuming the previous slope still holds.
    ii = 0
    @inbounds for ind in (mid_old+1):nblk_old
        ii += 1
        outbuf[ii] = oldbuf[ind] * presto_invsqrt(Float32(mean_old + dslope * (ii - 1)))
    end
    @inbounds for jj in 1:ii
        a[wpos+jj] = outbuf[jj]
    end
    numwrote += ii
    return numwrote
end

# ---------------------------------------------------------------------------
# One demodulation point, in memory
# ---------------------------------------------------------------------------

"""
    demodulate_series(raw, dt, epoch; accel=0.0, jerk=0.0, snap=0.0, v0=0.0,
                      pb=nothing, x=nothing, A_T=0.0, anchor="start",
                      sin_i=nothing, companion_mass=1.4, pulsar_mass=1.4)
        -> (Vector{Float32}, Float64)

Apply the exact time remap to the `Float32` time series `raw` and return the
resampled series plus its new start epoch (MJD), without touching the
filesystem.  `dt` is the input's sample interval (s) and `epoch` its `Epoch of
observation (MJD)`.  This is `demod/demod_dat.jl`'s `demod_file` body with the
two `write` calls lifted out, so `demod_file` is a thin wrapper over it and the
two cannot drift; the epoch comes back because `demod_file`'s `.inf` rewrites it
and the resampled start index is otherwise internal.

Two mutually exclusive models:
  * accel/jerk/snap/v0 -- exact polynomial remap (the default);
  * circular orbit -- pass `pb` [days] and either `x` [light-seconds] or
    `sin_i` (+ `companion_mass`/`pulsar_mass` to derive x), with `A_T` [rad]
    the mean anomaly at `anchor` ("start" = the .inf epoch, pulsegen's
    default; "midpoint" = obs midpoint).

The anchor convention is load-bearing and was measured: the polynomial remap is
anchored at the **midpoint** (`epoch + 0.5*N*dt`) and the circular `A_T` is
shifted from the midpoint back to the `.inf` epoch by `T/2`.  Getting it wrong
moved a recovered S/N from 40.8 to 8.0.  The returned length is per-point (the
remap's own length, minus one sample when it lands odd, since PRESTO's
`realfft` needs an even count), so everything file-scaled -- `T`, `r_lo =
lofreq*T`, Nyquist -- must be recomputed per grid point.
"""
function demodulate_series(raw::Vector{Float32}, dt::Real, epoch::Real;
                           accel = 0.0, jerk = 0.0, snap = 0.0, v0 = 0.0,
                           pb = nothing, x = nothing, A_T = 0.0, anchor = "start",
                           sin_i = nothing, companion_mass = 1.4, pulsar_mass = 1.4)
    N = length(raw)
    # Zero-mean before resampling.  The remap can leave a few sample holes
    # (uncovered indices), which scatter to zero; on data with a large DC
    # baseline those holes read as huge spikes and their comb swamps the band.
    # The Python original never sees this because `load_data` calls
    # `TimeSeries.normalise` first -- this is the same de-meaning.  Done on a
    # copy, not in place: a driver loops this over a whole grid on one `raw`,
    # and a function that silently rewrites its input is a trap.
    ts = raw .- (sum(raw) / N)

    # Anchor = midpoint of first/last sample, exactly as utils.anchor_mjd
    # (len(ts.data)*tsamp after the first sample, so the midpoint sits at N/2).
    reference_mjd = epoch + 0.5 * N * dt / 86400.0
    tstart_ind = round(Int, (epoch - reference_mjd) * 86400.0 / dt)

    if pb === nothing
        ts_new, tstart_new = resample_ts_shift_snap(
            ts, accel, jerk, snap, dt, tstart_ind; v0 = v0)
    else
        pb_s = pb * 86400.0
        x_lt_s = x
        if x_lt_s === nothing
            sin_i === nothing && error("circular demod needs --x or --sini")
            x_lt_s = projected_x(pb_s, sin_i, companion_mass, pulsar_mass)
        end
        # A_T is the mean anomaly at `anchor`; the remap's tau is measured from
        # the data-midpoint, so shift the midpoint back to that epoch.  The
        # midpoint is at epoch + N/2*dt, i.e. tau = +T/2 relative to the start.
        t_offset_s = anchor == "start" ? 0.5 * N * dt : 0.0
        ts_new, tstart_new = resample_ts_shift_voc(
            ts, circular_voc(x_lt_s, pb_s, A_T; t_offset_s = t_offset_s),
            dt, tstart_ind)
    end

    # PRESTO's realfft requires an even sample count; the remap can land on an
    # odd length, so drop the final sample (one bin ~ dt seconds).
    if isodd(length(ts_new))
        ts_new = ts_new[1:end-1]
    end

    new_epoch = reference_mjd + tstart_new * dt / 86400.0
    return ts_new, new_epoch
end

# ---------------------------------------------------------------------------
# Grid CSVs and output naming
# ---------------------------------------------------------------------------

"""
    read_demod_grid(csv_path) -> (mode, rows)

Read one grid CSV from `demod/nsns_grid.py`.  The header selects the model --
`accel,jerk,snap` gives `:ajs`, `pb,x,at` gives `:circular` -- and each row is
returned as a tuple of its **raw literal column strings**, not parsed floats:
the output filename is built from that same text so it matches what
`combine_cohout.py` and `run_nsns_sweep.sh` build from the CSV.
"""
function read_demod_grid(csv_path)
    mode = nothing
    rows = Tuple[]
    header = true
    for line in eachline(csv_path)
        if header
            cols = lowercase.(strip.(split(strip(line), ",")))
            if cols == ["accel", "jerk", "snap"]
                mode = :ajs
            elseif cols == ["pb", "x", "at"]
                mode = :circular
            else
                error("$csv_path: unknown header $(join(cols, ",")); expected " *
                      "accel,jerk,snap or pb,x,at")
            end
            header = false
            continue
        end
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) == 3 || error("$csv_path: expected 3 columns, got $(length(parts))")
        push!(rows, (String(strip(parts[1])), String(strip(parts[2])),
                     String(strip(parts[3]))))
    end
    mode === nothing && error("$csv_path: empty file / no header")
    return mode, rows
end

"""
    demod_point_stem(stem, mode, row) -> String

Output filename stem for one grid row, built from its literal CSV strings.
`:ajs` gives `<stem>_demod_a<A>_j<J>_s<S>` and `:circular` gives
`<stem>_demod_pb<Pb>_x<X>_at<AT>`.  `demod/combine_cohout.py`'s `MODELS`
regexes match these with a `_red.cohout` suffix, so the sift/combine tooling
works on the driver's output unchanged.
"""
function demod_point_stem(stem, mode, row)
    if mode == :ajs
        return string(stem, "_demod_a", row[1], "_j", row[2], "_s", row[3])
    else
        return string(stem, "_demod_pb", row[1], "_x", row[2], "_at", row[3])
    end
end
