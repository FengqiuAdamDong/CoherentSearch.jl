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
using Libdl: Libdl

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
    _exact_remap_poly(accel, jerk, snap, tsamp, tstart_ind, nsamp; tol_samples,
                      max_order, v0)

Ascending coefficients of the exact de-drift remap `u(tau)`, adding series
orders until the next order's peak contribution over `tau` is below
`tol_samples` samples.  Returns `(u_coeffs, order_used, converged)`.

`tau` is not materialised: sample `i`'s time is `(i - 1 + tstart_ind) * tsamp`,
exactly the broadcast `resample_ts_shift_snap` used to build.  At
`N = 1.92e8` the old `polyval(tau, term)` plus `abs.()` pair allocated 3 GB per
series order, and this runs for every grid point of a demod sweep.  The
accumulation order (Horner, left to right) is unchanged, so the coefficients
are bit-identical to the array version.
"""
function _exact_remap_poly(accel, jerk, snap, tsamp, tstart_ind, nsamp;
                           tol_samples = RESAMPLE_TOL_SAMPLES,
                           max_order = RESAMPLE_MAX_ORDER, v0 = 0.0)
    # v = -(trial as given), so the plain sum tau + sum_n integral (v/c)^n below is
    # integral dtau/(1 + v_trial/c): the remap that cancels a receding (v > 0) trial.
    v = -_velocity_poly(accel, jerk, snap; v0 = v0)
    if length(v) == 1 && v[1] == 0.0                  # all-zero
        return [0.0, 1.0], 0, true
    end

    step = float(tsamp)
    vn = copy(v)
    u = polyadd([0.0, 1.0], polyint(vn) ./ CONST_C_VAL)
    order_used = 1
    for order in 2:max_order
        vn = polymul(vn, v)
        term = polyint(vn) ./ CONST_C_VAL^order
        peak = 0.0
        @inbounds for i in 1:nsamp
            peak = max(peak, abs(polyval((i - 1 + tstart_ind) * step, term)))
        end
        if peak / tsamp < tol_samples
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
    isempty(ts) && return ts, tstart_ind
    N = length(ts)
    step = float(tsamp)
    u_coeffs, order_used, converged = _exact_remap_poly(
        accel, jerk, snap, tsamp, tstart_ind, N;
        tol_samples = tol_samples, max_order = max_order, v0 = v0)
    if !converged
        @warn "resample_ts_shift_snap: exact-remap series did not converge " *
              "to $tol_samples samples within $max_order orders -- pass a " *
              "larger max_order, or shrink the campaign span/velocity."
    end
    # Two scalar passes instead of the old N-long `tau`, `u` and index arrays.
    # At N = 1.92e8 each of those is 1.4 GB, and a `--demod-grid` sweep paid
    # them once per point; the remap is only a few flops per sample, so
    # recomputing the indices is far cheaper than allocating and collecting
    # several GiB.  The arithmetic and `round` are those of the old broadcasts.
    first_idx = 0
    prev_idx = 0
    @inbounds for i in 1:N
        idx_i = round(Int, polyval((i - 1 + tstart_ind) * step, u_coeffs) / tsamp)
        i == 1 && (first_idx = idx_i)
        if i > 1 && idx_i < prev_idx
            error("resample map is not monotonic: the coefficients are too large " *
                  "for this offset from the anchor")
        end
        prev_idx = idx_i
    end
    last_idx = prev_idx
    ts_new = zeros(eltype(ts), last_idx - first_idx + 1)
    @inbounds for i in 1:N
        idx_i = round(Int, polyval((i - 1 + tstart_ind) * step, u_coeffs) / tsamp)
        ts_new[idx_i - first_idx + 1] = ts[i]
    end
    return ts_new, first_idx
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
    _integrate_voc_remap(voc_func, tsamp, tstart_ind, nsamp, tol_samples;
                         n0, n_max)

`u(tau) = int_0^tau dt'/(1+voc_func(t'))` by cumulative trapezoid on
`[min(0,tau), max(0,tau)]`, quadrupling the sample count until the peak change
over one refinement is below `tol_samples` samples.

`tau` is generated from the sample index rather than materialised.  The old
`_interp(tau, ...)` pair plus `u .- prev` allocated four N-long `Float64`
arrays per refinement (5.7 GB at N = 1.92e8); the two buffers below are reused
across refinements and the difference is reduced in the same loop that fills
`u`.
"""
function _integrate_voc_remap(voc_func, tsamp, tstart_ind, nsamp, tol_samples;
                              n0 = RESAMPLE_VOC_N0, n_max = RESAMPLE_VOC_NMAX)
    step = float(tsamp)
    tau_first = tstart_ind * step
    tau_last = (nsamp - 1 + tstart_ind) * step
    lo = min(0.0, tau_first, tau_last)
    hi = max(0.0, tau_first, tau_last)
    hi == lo && return zeros(nsamp)
    n = n0
    u = Vector{Float64}(undef, nsamp)
    prev = Vector{Float64}(undef, nsamp)
    have_prev = false
    while true
        t_grid = _linspace(lo, hi, n)
        cum = _cumtrapz(1.0 ./ (1.0 .+ voc_func(t_grid)), t_grid)
        zero_u = _interp(0.0, t_grid, cum)
        worst = 0.0
        @inbounds for i in 1:nsamp
            ui = _interp((i - 1 + tstart_ind) * step, t_grid, cum) - zero_u
            u[i] = ui
            have_prev && (worst = max(worst, abs(ui - prev[i])))
        end
        if have_prev && worst / tsamp < tol_samples
            return u
        end
        if n >= n_max
            @warn "_integrate_voc_remap: quadrature did not converge to " *
                  "$tol_samples samples within $n points -- result may carry " *
                  "residual error."
            return u
        end
        u, prev = prev, u
        have_prev = true
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
    N = length(ts)
    u = _integrate_voc_remap(voc_func, tsamp, tstart_ind, N, tol_samples)
    # Scalar index passes, as in `resample_ts_shift_snap`: the old
    # `round.(Int, u ./ tsamp)` index vector is another N-long allocation.
    first_idx = 0
    prev_idx = 0
    @inbounds for i in 1:N
        idx_i = round(Int, u[i] / tsamp)
        i == 1 && (first_idx = idx_i)
        if i > 1 && idx_i < prev_idx
            error("resample map is not monotonic: voc_func implies |v|/c >= 1 " *
                  "somewhere in this segment's span")
        end
        prev_idx = idx_i
    end
    last_idx = prev_idx
    ts_new = zeros(eltype(ts), last_idx - first_idx + 1)
    @inbounds for i in 1:N
        idx_i = round(Int, u[i] / tsamp)
        ts_new[idx_i - first_idx + 1] = ts[i]
    end
    return ts_new, first_idx
end

"""
    circular_voc(x_lt_s, pb_s, A_T; t_offset_s = 0.0) -> tau -> v_inj(tau)/c

Injected-sign LOS Doppler of a pure Keplerian circular orbit (e = 0, Roemer
only, no GR), as a function of `tau` measured from the data-midpoint anchor:

    delay  Δ(τ) = x*sin(ω_b*(τ + t_offset_s) + A_T),   x = a_p sin(i)/c [light-s]
    v/c         = dΔ/dτ = x*ω_b*cos(ω_b*(τ + t_offset_s) + A_T)

with `A_T` the mean anomaly at the epoch `t_offset_s` from the midpoint: 0
(midpoint, the default) for `anchor="midpoint"`, `+T/2` for `"start"`.

`t_offset_s` shifts the midpoint anchor to the epoch at which `A_T` is defined
(`pulsegen_gr.py -anchor`): 0 for `midpoint` (the default everywhere, so the
grid's omega_b lever is T/2), T/2 for `start` (the .inf epoch; kept for
injections that anchor there).  `A_T` absorbs the (degenerate, for e = 0)
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
# PRESTO's `realfft`, in memory -- a byte-for-byte port
# ---------------------------------------------------------------------------
#
# PRESTO does not call FFTW's real-to-complex transform.  `realfft()` in
# `fastffts.c` packs the N real samples *in place* as N/2 complex values,
# runs a length-N/2 **complex** FFT, and then unpacks that into PRESTO's
# `.fft` layout with a double-precision trig recursion.  Its complex FFT is
# `fftwcall()` (`fftcalls.c`), which uses FFTW_MEASURE for N/2 <= 16384 and
# FFTW_ESTIMATE above that.  `FFTW.rfft` is mathematically the same transform
# but a different algorithm (and a different FFTW build), so its last bits --
# and therefore the S/N -- differ.  We reproduce the actual chain instead.
#
# Bit-exactness needs the *same* libfftw3f that PRESTO used.  We take its path
# from the `realfft` executable itself (`ldd realfft`), falling back to
# `Libdl.find_library`, so the plan algorithms and SIMD codelets are the same
# ones PRESTO's binary runs.  Set `COHERENT_FFTW_LIB` to override.

const _PRESTO_FFTW_LOCK = ReentrantLock()
const _PRESTO_FFTW_READY = Ref(false)
const _PRESTO_FFTW_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _PRESTO_FFTW_PLAN = Ref{Ptr{Cvoid}}(C_NULL)
const _PRESTO_FFTW_EXECUTE = Ref{Ptr{Cvoid}}(C_NULL)
const _PRESTO_FFTW_DESTROY = Ref{Ptr{Cvoid}}(C_NULL)
const _PRESTO_FFTW_EXACT = Ref(false)
const _PRESTO_FFTW_PATH = Ref("")

# Pointers must not survive a precompiled image: a saved address is meaningless
# in the next process.  `demod_main` is in the precompile workload below, so it
# *does* run this code while precompiling; reset the lazy state on load.
function __init__()
    _PRESTO_FFTW_READY[] = false
    _PRESTO_FFTW_HANDLE[] = C_NULL
    _PRESTO_FFTW_PLAN[] = C_NULL
    _PRESTO_FFTW_EXECUTE[] = C_NULL
    _PRESTO_FFTW_DESTROY[] = C_NULL
    _PRESTO_FFTW_EXACT[] = false
    _PRESTO_FFTW_PATH[] = ""
    return nothing
end

function _presto_fftw_library()
    override = get(ENV, "COHERENT_FFTW_LIB", "")
    isempty(override) || return override

    # The strongest match: ask the very binary this is meant to reproduce which
    # libfftw3f it loads.  `ldd` resolves an absolute path, so a conda or
    # custom install is handled the same way as the system library.
    exe = get(ENV, "REALFFT", "")
    isempty(exe) && (exe = Sys.which("realfft"))
    if exe !== nothing && Sys.islinux()
        ldd = Sys.which("ldd")
        if ldd !== nothing
            try
                for line in eachline(IOBuffer(read(`$ldd $exe`, String)))
                    occursin("libfftw3f", line) || continue
                    parts = split(line)
                    k = findfirst(==("=>"), parts)
                    k === nothing && continue
                    k < length(parts) || continue
                    return String(parts[k + 1])
                end
            catch
                # An unreadable/incompatible binary is not fatal: fall through.
            end
        end
    end
    return Libdl.find_library(["libfftw3f.so.3", "libfftw3f", "libfftw3f.dylib"])
end

function _presto_fftw_init!()
    _PRESTO_FFTW_READY[] && return
    lock(_PRESTO_FFTW_LOCK) do
        _PRESTO_FFTW_READY[] && return
        _PRESTO_FFTW_READY[] = true          # even a failure is final
        path = _presto_fftw_library()
        isempty(path) && return
        try
            handle = Libdl.dlopen(path)
            plan = Libdl.dlsym(handle, :fftwf_plan_dft_1d)
            execute = Libdl.dlsym(handle, :fftwf_execute_dft)
            destroy = Libdl.dlsym(handle, :fftwf_destroy_plan)
            _PRESTO_FFTW_HANDLE[] = handle
            _PRESTO_FFTW_PLAN[] = plan
            _PRESTO_FFTW_EXECUTE[] = execute
            _PRESTO_FFTW_DESTROY[] = destroy
            _PRESTO_FFTW_PATH[] = Libdl.dlpath(handle)
            _PRESTO_FFTW_EXACT[] = true
        catch
            _PRESTO_FFTW_HANDLE[] = C_NULL
            _PRESTO_FFTW_PLAN[] = C_NULL
            _PRESTO_FFTW_EXECUTE[] = C_NULL
            _PRESTO_FFTW_DESTROY[] = C_NULL
            _PRESTO_FFTW_PATH[] = ""
            _PRESTO_FFTW_EXACT[] = false
        end
    end
    return nothing
end

"""
    presto_fftw_exact() -> Bool
    presto_fftw_library() -> String

Whether a PRESTO-compatible `libfftw3f` was found, and which one.  Without it
`presto_fft_amps` still works (through FFTW.jl's own library, with PRESTO's
exact unpacking) but cannot be byte-identical to PRESTO's `realfft`.
"""
function presto_fftw_exact()
    _presto_fftw_init!()
    return _PRESTO_FFTW_EXACT[]
end

function presto_fftw_library()
    _presto_fftw_init!()
    return _PRESTO_FFTW_PATH[]
end

# One cached complex-FFT plan.  `plan == C_NULL` means the fallback path stores
# a Julia FFTW plan in `jplan` instead.
mutable struct _PrestoFFTPlan
    n::Int
    plan::Ptr{Cvoid}
    jplan::Any
end

function _destroy_presto_fft_plan(p::_PrestoFFTPlan)
    p.plan == C_NULL && return nothing
    lock(_PRESTO_FFTW_LOCK) do
        if p.plan != C_NULL && _PRESTO_FFTW_DESTROY[] != C_NULL
            ccall(_PRESTO_FFTW_DESTROY[], Cvoid, (Ptr{Cvoid},), p.plan)
        end
        p.plan = C_NULL
    end
    return nothing
end

# FFTW allocates a plan's twiddle tables outside the Julia heap, so the GC
# cannot see them and will not reclaim them while the cache holds the plan.
# The tables scale with the transform length: measured ~0.74 GB per plan at
# N = 1.92e8 samples on the PRESTO-library path (~4 bytes per sample), and
# about twice that on the FFTW.jl fallback, whose plan also keeps the array it
# was planned on.  The budget therefore charges 8 bytes per sample.
#
# That matters because the demodulation driver plans once per *distinct*
# demodulated length, and `--demod-grid` produces a slightly different length
# for every grid point (the remap's endpoint moves with accel/jerk/snap).  An
# unbounded cache therefore pins one O(N) plan per grid point -- ~0.75 GB per
# point on the 192M-sample fixture -- and the process eventually runs out of
# RAM.  A byte budget instead keeps the cache bounded, which is the whole
# "peak memory independent of the grid size" property the driver promises.
#
# A `Ref` so the regression test can shrink the budget and exercise eviction on
# 512-sample transforms instead of allocating hundreds of MB.
const PRESTO_FFT_PLAN_CACHE_BYTES = Ref(512 * 1024 * 1024)

_presto_plan_bytes(N::Int) = 8 * N

"""
    _trim_presto_plans!(plans, reserve = 0)

Drop cached plans until `reserve` more bytes fit in
`PRESTO_FFT_PLAN_CACHE_BYTES`, destroying each evicted FFTW plan
immediately (finalizers alone would leave the C memory alive until the next
full GC).  Called *before* a new plan is built, so the transient peak is the
budget rather than the budget plus one plan.
"""
function _trim_presto_plans!(plans::AbstractDict{Int}, reserve::Int = 0)
    isempty(plans) && return nothing
    used = sum(_presto_plan_bytes, keys(plans); init = 0)
    while used + reserve > PRESTO_FFT_PLAN_CACHE_BYTES[] && !isempty(plans)
        k = first(keys(plans))
        used -= _presto_plan_bytes(k)
        _destroy_presto_fft_plan(pop!(plans, k))
    end
    return nothing
end

# FFTW_ESTIMATE = 1 << 6, FFTW_MEASURE = 0; PRESTO's `fftwcall()` uses the
# former above 16384 points and the latter at or below it.
const _FFTW_MEASURE = UInt32(0)
const _FFTW_ESTIMATE = UInt32(1 << 6)
const _FFTW_FORWARD = Cint(-1)

function _make_presto_fft_plan(data::Vector{ComplexF32}, N::Int)
    H = N ÷ 2
    if _PRESTO_FFTW_EXACT[]
        flag = H > 16384 ? _FFTW_ESTIMATE : _FFTW_MEASURE
        # FFTW planning may scribble on the input (MEASURE always does), so
        # copy it away and put it back exactly as `fftwcall()` does.
        saved = copy(data)
        plan = ccall(_PRESTO_FFTW_PLAN[], Ptr{Cvoid},
                     (Cint, Ptr{ComplexF32}, Ptr{ComplexF32}, Cint, Cuint),
                     H, pointer(data), pointer(data), _FFTW_FORWARD, flag)
        copyto!(data, saved)
        plan == C_NULL && error("PRESTO-compatible FFTW failed to plan N=$N")
        p = _PrestoFFTPlan(N, plan, nothing)
        finalizer(_destroy_presto_fft_plan, p)
        return p
    end
    jp = FFTW.plan_fft!(data; flags = FFTW.ESTIMATE)
    return _PrestoFFTPlan(N, C_NULL, jp)
end

"""
    presto_fft_amps(x, plans::AbstractDict{Int}) -> Vector{ComplexF32}

Byte-for-byte port of PRESTO's `realfft()` (`src/fastffts.c`) for a forward
transform.  The result has PRESTO's `.fft` layout: `N÷2` `ComplexF32` bins in
which bin 1 packs DC in its real part and Nyquist in its imaginary part, and
bins 2..N÷2 are the positive-frequency amplitudes.  `N` must be even.

`plans` caches the length-`N/2` complex-FFT plan per `N`.  That plan is the
same FFTW call PRESTO makes (including its MEASURE/ESTIMATE switch), against
the same `libfftw3f` when [`presto_fftw_exact`](@ref) is true; the unpacking
trig recursion is then transcribed operation for operation, including which
intermediates are rounded to `Float32` before being widened.  On the PRESTO
test series this reproduces `realfft`'s output bytes exactly.

The cache is **bounded** (see `PRESTO_FFT_PLAN_CACHE_BYTES`): a plan's
twiddles are FFTW-allocated, invisible to the GC, and O(`N`) each, so caching
every length a `--demod-grid` sweep visits would retain one plan per grid
point until the process exits.  Plans are evicted (and destroyed) when the
budget is exceeded; the victim is whichever key the dictionary yields first,
and the trim only runs when `N` is not already cached, so the plan about to be
reused is never the one evicted.  Callers that need a different policy can
pass their own dictionary and manage it.

One caveat is inherited from PRESTO, not added here: for `N÷2 <= 16384`
PRESTO asks FFTW for a `MEASURE` plan, and FFTW picks the fastest algorithm by
timing it, so two runs of PRESTO's own `realfft` can differ in the last bits.
Above that threshold PRESTO uses a deterministic `ESTIMATE` plan; that is the
regime of a real demodulated observation (`N` in the 10^5..10^7 range), and
there the match is exact.  (PRESTO switches to its own six-step FFT above
2×10^8 complex points; that is far outside the demod range and not ported.)
"""
function presto_fft_amps(x::AbstractVector{Float32}, plans::AbstractDict{Int})
    N = length(x)
    iseven(N) || error("The number of time samples must be even (realfft requires it), got $N")
    N >= 2 || error("realfft needs at least two samples, got $N")
    H = N ÷ 2

    _presto_fftw_init!()
    # PRESTO treats the N floats as N/2 complex values in place; `reinterpret`
    # is that memcpy, and the complex FFT below is length N/2 -- not N.
    xbuf = x isa Vector{Float32} ? x : Vector{Float32}(x)
    data = copy(reinterpret(ComplexF32, xbuf))
    plan = get(plans, N, nothing)
    if plan === nothing
        # Make room before planning, so the peak is the budget, not budget+plan.
        _trim_presto_plans!(plans, _presto_plan_bytes(N))
        plan = _make_presto_fft_plan(data, N)
        plans[N] = plan
    end
    plan.n == N || error("FFT plan cache key/length mismatch: $(plan.n) != $N")

    if plan.plan == C_NULL
        plan.jplan * data
    else
        ccall(_PRESTO_FFTW_EXECUTE[], Cvoid,
              (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{ComplexF32}),
              plan.plan, pointer(data), pointer(data))
    end

    # `realfft()`'s forward post-processing, exactly as written in
    # `src/fastffts.c`.  The C locals are double; the float sums inside the
    # parentheses are rounded to float *before* being promoted, so preserve
    # that distinction.
    n = N
    nby2 = n >> 1
    nby4 = n >> 2
    cc = -0.5
    theta = -2.0 * π / n
    wr = cos(theta)
    wi = sin(theta)
    tmp1 = sin(0.5 * theta)
    wpr = -2.0 * tmp1 * tmp1
    wpi = wi
    il = 1
    ih = nby2 - il
    @inbounds while il <= nby4
        dl = data[il+1]
        dh = data[ih+1]
        h1r = 0.5 * Float64(dl.re + dh.re)
        h1i = 0.5 * Float64(dl.im - dh.im)
        h2r = -cc * Float64(dl.im + dh.im)
        h2i = cc * Float64(dl.re - dh.re)
        h2rwr = h2r * wr
        h2rwi = h2r * wi
        h2iwr = h2i * wr
        h2iwi = h2i * wi
        data[il+1] = ComplexF32(Float32(h1r + h2rwr - h2iwi),
                                Float32(h1i + h2iwr + h2rwi))
        data[ih+1] = ComplexF32(Float32(h1r - h2rwr + h2iwi),
                                Float32(-h1i + h2iwr + h2rwi))
        tmp1 = wr
        wr = tmp1 * wpr - wi * wpi + wr
        wi = wi * wpr + tmp1 * wpi + wi
        il += 1
        ih -= 1
    end
    # Forward transform: data[0] becomes (DC, Nyquist).
    tmp1 = Float64(data[1].re)
    data[1] = ComplexF32(Float32(tmp1 + data[1].im),
                         Float32(tmp1 - data[1].im))
    return data
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
bins it wrote.  `T` is the length in seconds (`N*dt`) of the series that was
transformed: for the demodulation driver that is the **resampled** series
(`length(series)*dt`), because `demod_dat.jl` rewrites the `.inf` with the
resampled `N` and `rednoise` reads `N*dt` from that file.  Using the input
observation's `T` here would pick different block boundaries and different
window-freeze bins.

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

The arithmetic follows C's types exactly, which is what makes the output byte
for byte: `mean_old`/`mean_new`/`dslope`/`norm` are `Float32` (as declared in
`dered_engine`), but the expressions that C evaluates in `double` stay in
`Float64` until their assignment.  In particular

  * `median(...) / log(2.0)` is a double division, then rounded to `Float32`;
  * `(mean_new - mean_old)` is a `Float32` subtraction, and the division by
    `0.5*(nblk_old + nblk_new)` (a double) is done in `Float64`, then rounded;
  * `mean_old + dslope*ii` is entirely `Float32`; and
  * `(float) binnum / Tf` is compared against the *double* `endfreq`.

`T` is widened through `Float32` once, exactly as C's
`const float Tf = (float) T;`.  Rounded to `Float64` throughout instead, the
result differs from PRESTO in the last bits; rounded to `Float32` too early, it
differs in the fourth digit.

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
        powr = oldbuf[ii].re
        powi = oldbuf[ii].im
        powbuf[ii] = powr * powr + powi * powi
    end
    # C: float mean_old = median(...) / log(2.0) -- the division is double,
    # the assignment rounds to float.
    mean_old = Float32(Float64(_lower_median!(powbuf, nblk_old)) / log(2.0))

    # The first half of the first block has no slope: only a few bins, and the
    # C code says that is probably OK.
    norm = presto_invsqrt(mean_old)
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

    # `dslope` is 1.0f0 in the C declaration and reaches the final partial
    # block unchanged only if the while loop never ran.
    dslope = 1.0f0
    while true
        nblk_new = min(bufflen, numbins - rpos)
        nblk_new <= 0 && break
        @inbounds for ii in 1:nblk_new
            newbuf[ii] = a[rpos + ii]
        end
        rpos += nblk_new
        mid_new = nblk_new >> 1

        @inbounds for ii in 1:nblk_new
            powr = newbuf[ii].re
            powi = newbuf[ii].im
            powbuf[ii] = powr * powr + powi * powi
        end
        mean_new = Float32(Float64(_lower_median!(powbuf, nblk_new)) / log(2.0))

        # C: dslope = (mean_new - mean_old) / (0.5 * (nblk_old + nblk_new));
        # the numerator is a float subtraction, the 0.5 denominator term is a
        # double, so the division widens and only the result rounds to float.
        dslope = Float32(Float64(mean_new - mean_old) /
                         (0.5 * (nblk_old + nblk_new)))

        # The second half of the old block, then the first half of the new one,
        # both against the ramp from the old median to the new: the ramp's
        # coordinate `ii` starts at 0 at the old block's midpoint and reaches
        # the new block's midpoint (ii = mid_new + nblk_old - mid_old) at
        # mean_new.
        ii = 0
        @inbounds for ind in (mid_old+1):nblk_old
            ii += 1
            outbuf[ii] = oldbuf[ind] * presto_invsqrt(mean_old + dslope * (ii - 1))
        end
        @inbounds for ind in 1:mid_new
            ii += 1
            outbuf[ii] = newbuf[ind] * presto_invsqrt(mean_old + dslope * (ii - 1))
        end
        @inbounds for jj in 1:ii
            a[wpos+jj] = outbuf[jj]
        end
        wpos += ii
        numwrote += ii

        binnum += nblk_new
        # C: if ((float) binnum / Tf < endfreq) -- float quotient against the
        # double `endfreq`, so compare in Float64.
        if Float32(binnum) / Tf < Float64(endfreq)
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
        outbuf[ii] = oldbuf[ind] * presto_invsqrt(mean_old + dslope * (ii - 1))
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
                      pb=nothing, x=nothing, A_T=0.0, anchor="midpoint",
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
    the mean anomaly at `anchor` ("midpoint" = obs midpoint, the default and
    the convention the circular grid is generated with; "start" = the .inf
    epoch, pulsegen's old default).

The anchor convention is load-bearing and was measured: the polynomial remap is
anchored at the **midpoint** (`epoch + 0.5*N*dt`) and the circular `A_T` is
**also** defined at that midpoint by default.  The two anchors differ by a `T/2`
phase shift (`anchor="start"` shifts the midpoint back to the `.inf epoch`);
getting it wrong moved a recovered S/N from 40.8 to 8.0.  The returned length is
per-point (the remap's own length, minus one sample when it lands odd, since
PRESTO's `realfft` needs an even count), so everything file-scaled -- `T`,
`r_lo = lofreq*T`, Nyquist -- must be recomputed per grid point.
"""
function demodulate_series(raw::Vector{Float32}, dt::Real, epoch::Real;
                           accel = 0.0, jerk = 0.0, snap = 0.0, v0 = 0.0,
                           pb = nothing, x = nothing, A_T = 0.0, anchor = "midpoint",
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
        # A_T is the mean anomaly at `anchor`.  The remap's tau is measured from
        # the data midpoint, so tau already is t - T/2: `midpoint` (the
        # default) needs no shift, and `start` (A_T at the .inf epoch) shifts
        # the midpoint back by T/2 = N/2*dt.
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
