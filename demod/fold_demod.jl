#!/usr/bin/env julia
#
# Fold CoherentSearch `.cohout` candidates after demodulating the time series
# at the accel/jerk/snap (ajs) or circular-orbit point named in the `.cohout`
# filename.
#
# The demodulation is the package's own `CoherentSearch.demodulate_series`, so
# a fold here uses exactly the same time remap as `bin/coherent_search.jl
# --demod-grid` and `demod/demod_dat.jl`.  Nothing intermediate is written: the
# raw `.dat` is read once, demodulated in memory, and folded directly.
#
# Usage
# -----
#   julia --project=<repo> demod/fold_demod.jl INPUT [options]
#
# INPUT is either a directory containing `*.cohout` files or a single
# `.cohout` file.  The default example directory is
# `test/test_grid_ajs_efficacy/OUT`.
#
# Model parameters come from the filename, exactly as the search wrote it:
#
#   <stem>_demod_a<A>_j<J>_s<S>_red.cohout       accel/jerk/snap
#   <stem>_demod_pb<Pb>_x<X>_at<AT>_red.cohout  circular orbit (Pb [d],
#                                               x [lt-s], AT [rad] at midpoint)
#
# The source `.dat` is inferred from `<stem>` unless `--dat` is given; the
# search itself never writes a `.dat`, so for the bundled example the script
# finds `../Ter5_DM238.60.dat` beside the `OUT` directory.
#
# Candidate selection
# -------------------
# Every candidate row in a `.cohout` is filtered on its reported S/N,
# frequency (Hz) and period (ms -> seconds), then sorted by S/N.  `--top N`
# folds the N best surviving candidates of each file (default 1; 0 = all).
# The filters default to the full range: period/frequency 0 .. Inf and S/N
# 0 .. Inf.
#
#   --min-period SEC   --max-period SEC     (seconds; 0 / Inf default)
#   --min-freq HZ      --max-freq HZ        (Hz;      0 / Inf default)
#   --min-snr S        --max-snr S          (reported search S/N)
#
# Known pulsars
# -------------
# `--par` takes TEMPO/TEMPO2 .par files, directories, or quoted globs (repeat
# the flag for several paths) and removes candidates within the same search
# band `plot_corner_ajs.py` uses: max(0.005, 5e-4*F0) by default, overridable
# with --par-tol (absolute Hz) and --par-frac-tol (fraction of F0).
# --par-harmonics N also removes harmonics 2..N, with the band scaled by N.
# The mask is applied after the period/frequency/S/N windows and before
# --top, so the best surviving non-masked candidate is folded.
#
# With --inf, each known pulsar's searched frequency is predicted separately
# for every ajs trial point from the full par orbit and the demodulation
# residual (the same `--inf` mechanism as plot_corner_ajs.py, via
# predict_par_masks.py).  Without --inf the mask is centred on F0 at PEPOCH.
#
# DBSCAN clustering
# -----------------
# --dbscan clusters the globally surviving candidates in four dimensions:
#
#     (accel/da, jerk/dj, snap/ds, frequency/df)
#
# where da/dj/ds are the median positive spacings of the user-supplied
# --spacing-csv grid (e.g. my_grid_search/nsns.csv), and df is --freq-spacing,
# the CSV's frequency column if present, or 1/T from --inf/--dat.  It folds the
# top --top candidates per cluster (noise is discarded unless
# --dbscan-keep-noise, which makes each noise point a singleton).  The parsed
# spacings are printed and written to <outdir>/dbscan_spacings.txt.
#
# --dbscan-eps is in grid-spacing units (default 1 = one grid step) and
# --dbscan-minpts is the DBSCAN density threshold including the point itself
# (default 2).  Clustering runs once in the parent; --jobs then shards only the
# selected cluster representatives across workers (the parent writes
# dbscan_selection.csv and passes it to the workers).
#
# Output
# ------
# For each fold, under `--outdir` (default `<INPUT>/folds`):
#
#   <cohout-stem>_cand<NNN>.txt   phase, counts, mean profile and profile S/N
#   <cohout-stem>_cand<NNN>.png   summed profile + sub-integration waterfall
#   folds.csv                     one row per fold, plus the parameters used
#
# S/N and the default bin count
# -----------------------------
# If --nbins is omitted (or 0), the series is folded at 1024 bins and the
# Morello et al. (2014, MNRAS 443, 1651) boxcar optimisation is run over the
# widths 1,2,4,8,16,32,64,128,256:
#
#     S/N = sum_{i in W}(p_i - bbar) / (sigma * sqrt(w))
#
# where W is the circular on-pulse window (w bins) and bbar/sigma are the mean
# and standard deviation of the bins outside W.  Both the width and the window
# position are optimised.  The winning width w is then used to downsample the
# profile and sub-integration waterfall to 1024/w bins; the reported fold S/N
# is the winning boxcar S/N, and the plotted curve is the S/N of each
# downsampled window (so its peak equals the reported value).
#
# With an explicit --nbins N, the old behaviour is kept: fold at N bins and
# report the robust profile S/N (median-subtract, divide by IQR/1.349, peak).
# Neither is the coherent-search S/N; both are recorded.
#
# Options
# -------
#   --dat FILE          use this .dat instead of inferring one
#   --outdir DIR        output directory (default: <INPUT>/folds)
#   --nbins N           phase bins; 0 = auto boxcar optimisation (default),
#                        otherwise fold directly at N bins with profile S/N
#   --nsubints N        sub-integrations in the waterfall (default 16; 1 = off)
#   --top N             candidates per .cohout, or per DBSCAN cluster
#                       (default 1; 0 = all)
#   --dbscan            global DBSCAN in (ajs, frequency) before folding
#   --spacing-csv PATH  grid CSV supplying the accel/jerk/snap spacings
#   --freq-spacing HZ   frequency spacing (default: CSV column or 1/T)
#   --dbscan-eps E      DBSCAN eps in grid-spacing units (default 1)
#   --dbscan-minpts M   DBSCAN min points per cluster (default 2)
#   --dbscan-keep-noise fold noise points as singleton clusters
#   --recursive         also search subdirectories for *.cohout
#   --limit N           stop after N .cohout files (0 = all; useful for a smoke test)
#   --jobs N            independent worker processes for demod+fold (default 1;
#                       memory scales with N because each holds its own series)
#   --no-plot           write only the text profiles (skips loading CairoMakie)
#   --par PATH ...      known .par file(s), directory, or glob; repeatable
#   --par-tol HZ        absolute par-mask half-width (default max(0.005, 5e-4*F0))
#   --par-frac-tol F    fractional-F0 par-mask half-width (default 0)
#   --par-harmonics N   also mask harmonics 2..N (default 1)
#   --inf OBS.inf       ajs-predicted par-mask centres per trial point
#   --python EXE        Python used only for the --inf prediction
#   --accel/--jerk/--snap A J S
#                       override the filename's ajs parameters
#   --pb/--x/--at PB X AT
#                       override the filename's circular parameters
#
# Examples
# --------
#   # All files in the bundled output, best candidate of each, 10 ms .. 1 s:
#   julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT \
#       --min-period 0.010 --max-period 1.0 --min-snr 6
#
#   # Drop known pulsars and their first 12 harmonics:
#   julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT \
#       --par test/test_grid_ajs_efficacy/par_files --par-harmonics 12
#
#   # Same, with the ajs-predicted mask centre at every trial point
#   # (OBS.inf is the barycentered observation the search ran on):
#   julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT \
#       --par test/test_grid_ajs_efficacy/par_files --par-harmonics 12 \
#       --inf OBS.inf
#
#   # One file, ten best candidates, explicit data file:
#   julia --project=. demod/fold_demod.jl \
#       test/test_grid_ajs_efficacy/OUT/Ter5_DM238.60_demod_a-6.932130137_j-0.0001068209012_s-1.140293766e-08_red.cohout \
#       --dat test/test_grid_ajs_efficacy/Ter5_DM238.60.dat --top 10
#
#   # Four demod+fold jobs at once (memory ~4x a single job):
#   julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT \
#       --jobs 4 --limit 100
#
#   # Cluster globally in (ajs, frequency) using my_grid_search/nsns.csv
#   # spacings, then fold the best 2 candidates of every cluster:
#   julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT \
#       --dbscan --spacing-csv test/test_grid_ajs_efficacy/my_grid_search/nsns.csv \
#       --top 2 --no-plot

using CoherentSearch
using ArgParse
using Mmap
using Printf: @printf, @sprintf

# `parse_inf` (and `demod_file`, unused here) is the on-disk wrapper's reader.
# Its main guard is false when included, so this does not run its CLI.
include(joinpath(@__DIR__, "demod_dat.jl"))

# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------
#
# CairoMakie is expensive to load (~9 s), so only pull it in when the command
# line actually asks for PNGs.  `--help` must not pay for it either.  If the
# load fails, keep going with text profiles rather than aborting a long fold
# run over an optional extra.
const _WANT_PLOT = !any(==("--no-plot"), ARGS) &&
                   !any(==("-h"), ARGS) && !any(==("--help"), ARGS)
const _HAVE_MAKIE = if _WANT_PLOT
    try
        @eval using CairoMakie
        true
    catch err
        @warn "CairoMakie could not be loaded; writing text profiles only" exception = err
        false
    end
else
    false
end

# ---------------------------------------------------------------------------
# Filename model parsing
# ---------------------------------------------------------------------------

const AJS_RE = r"^(?<stem>.+)_demod_a(?<a>[^_]+)_j(?<j>[^_]+)_s(?<s>[^_]+)(?:_red)?\.cohout$"
const CIRC_RE = r"^(?<stem>.+)_demod_pb(?<pb>[^_]+)_x(?<x>[^_]+)_at(?<at>[^_]+)(?:_red)?\.cohout$"

struct DemodModel
    mode::Symbol                    # :ajs or :circular
    stem::String                    # source .dat stem, from the .cohout name
    params::NamedTuple               # accel/jerk/snap or pb/x/at
end

"""
    model_from_name(path) -> Union{DemodModel,Nothing}

Parse `<stem>_demod_a..._j..._s..._red.cohout` or
`<stem>_demod_pb..._x..._at..._red.cohout`.  Returns `nothing` for a `.cohout`
that did not come from the demod driver.
"""
function model_from_name(path)
    name = basename(path)
    m = match(AJS_RE, name)
    if m !== nothing
        p = (accel = parse(Float64, m[:a]),
             jerk = parse(Float64, m[:j]),
             snap = parse(Float64, m[:s]))
        return DemodModel(:ajs, m[:stem], p)
    end
    m = match(CIRC_RE, name)
    if m !== nothing
        p = (pb = parse(Float64, m[:pb]),
             x = parse(Float64, m[:x]),
             at = parse(Float64, m[:at]))
        return DemodModel(:circular, m[:stem], p)
    end
    return nothing
end

"""
    manual_model(accel, jerk, snap, pb, x, at, fallback_stem) -> Union{DemodModel,Nothing}

Build an override model from the CLI.  Supplying both an ajs override and a
circular override is an error.
"""
function manual_model(accel, jerk, snap, pb, x, at, fallback_stem)
    has_ajs = !isnan(accel) || !isnan(jerk) || !isnan(snap)
    has_circ = !isnan(pb) || !isnan(x) || !isnan(at)
    has_ajs && has_circ &&
        error("give either --accel/--jerk/--snap or --pb/--x/--at, not both")
    has_ajs || has_circ || return nothing
    if has_ajs
        any(isnan, (accel, jerk, snap)) &&
            error("an ajs override needs all three of --accel, --jerk, --snap")
        return DemodModel(:ajs, fallback_stem, (accel = accel, jerk = jerk, snap = snap))
    end
    any(isnan, (pb, x, at)) &&
        error("a circular override needs all three of --pb, --x, --at")
    return DemodModel(:circular, fallback_stem, (pb = pb, x = x, at = at))
end

function model_string(model::DemodModel)
    p = model.params
    if model.mode == :ajs
        return @sprintf("ajs a=%.10g j=%.10g s=%.10g", p.accel, p.jerk, p.snap)
    end
    return @sprintf("circular pb=%.10g d x=%.10g lt-s at=%.10g rad", p.pb, p.x, p.at)
end

# ---------------------------------------------------------------------------
# Known-pulsar mask (the same mechanism as plot_corner_ajs.py)
# ---------------------------------------------------------------------------

struct KnownPar
    name::String
    path::String
    f0::Float64
    par::Dict{String,String}
end

function _regex_escape(s::AbstractString)
    replace(s, r"([\\.^$|()\[\]{}+?])" => s"\\\1")
end

"""Expand a .par file, directory, or simple `*` glob into concrete paths."""
function expand_par_paths(pattern)
    if isdir(pattern)
        return sort([joinpath(pattern, f) for f in readdir(pattern)
                     if endswith(lowercase(f), ".par")])
    elseif isfile(pattern)
        return [pattern]
    end
    dir = dirname(pattern)
    isempty(dir) && (dir = pwd())
    isdir(dir) || return String[]
    rx = Regex("^" * join(_regex_escape.(split(basename(pattern), '*')), ".*") * "\$")
    return sort([joinpath(dir, f) for f in readdir(dir) if occursin(rx, f)])
end

"""First value of every TEMPO/TEMPO2 key, with Fortran D exponents fixed."""
function read_par_dict(path)
    out = Dict{String,String}()
    for line in eachline(path)
        line = strip(split(line, '#'; limit = 2)[1])
        isempty(line) && continue
        fields = split(line)
        length(fields) >= 2 || continue
        key = uppercase(fields[1])
        value = replace(fields[2], r"(?<=\d)[Dd](?=[+-]?\d+$)" => "E")
        haskey(out, key) || (out[key] = value)
    end
    return out
end

"""Spin frequency at PEPOCH in Hz, from F0 or P0."""
function par_f0(par)
    f0 = haskey(par, "F0") ? tryparse(Float64, par["F0"]) : nothing
    f0 !== nothing && return f0
    p0 = haskey(par, "P0") ? tryparse(Float64, par["P0"]) : nothing
    (p0 !== nothing && p0 > 0) && return 1.0 / p0
    return nothing
end

"""Intrinsic spin frequency at `mjd` from F0/F1/F2... (or P0/P1)."""
function par_frequency_at(par, mjd)
    f0 = par_f0(par)
    f0 === nothing && return nothing
    pepoch = haskey(par, "PEPOCH") ? tryparse(Float64, par["PEPOCH"]) :
             haskey(par, "POSEPOCH") ? tryparse(Float64, par["POSEPOCH"]) : mjd
    pepoch === nothing && (pepoch = mjd)
    dt = (mjd - pepoch) * 86400.0
    f1 = nothing
    if haskey(par, "F1")
        f1 = tryparse(Float64, par["F1"])
    elseif haskey(par, "P1")
        p1 = tryparse(Float64, par["P1"])
        f1 = p1 === nothing ? nothing : -p1 / (1.0 / f0)^2
    end
    f = f0
    f1 !== nothing && (f += f1 * dt)
    fact = 2.0
    for order in 2:9
        key = "F$order"
        haskey(par, key) || break
        val = tryparse(Float64, par[key])
        val === nothing && break
        f += val * dt^order / fact
        fact *= order + 1
    end
    return f
end

"""Expand par patterns, parse them, and skip any without a usable F0/P0."""
function gather_pars(patterns)
    paths = String[]
    for p in patterns
        append!(paths, expand_par_paths(p))
    end
    pars = KnownPar[]
    seen = Set{String}()
    for p in paths
        ap = abspath(p)
        ap in seen && continue
        push!(seen, ap)
        par = read_par_dict(ap)
        f0 = par_f0(par)
        if f0 === nothing || f0 <= 0
            @warn "skipping par without a usable F0/P0" par = ap
            continue
        end
        name = get(par, "PSR", get(par, "PSRJ", splitext(basename(ap))[1]))
        push!(pars, KnownPar(name, ap, f0, par))
    end
    return pars
end

"""
    par_halfwidth(f0, abs_tol, frac_tol)

Frequency half-width for masking one known pulsar.  With neither override,
reproduce `run_ajs_pipeline.sh`'s search band, `max(0.005, 5e-4*F0)` — the
same default `plot_corner_ajs.py` uses.
"""
function par_halfwidth(f0, abs_tol, frac_tol)
    if abs_tol === nothing && frac_tol <= 0
        return max(0.005, 5e-4 * f0)
    end
    return max(abs_tol === nothing ? 0.0 : abs_tol, frac_tol * f0)
end

"""
    mask_known(cands, pars, centers, abs_tol, frac_tol, max_harmonic)
        -> (kept_candidates, per_par_counts)

Drop candidates within `h` times the half-width of harmonic `h` of each known
pulsar's centre, for `h = 1 .. max_harmonic`.  `centers[i]` is the mask centre
for `pars[i]` at this trial point (F0 at PEPOCH, spin frequency at the
observation midpoint, or the ajs-predicted search frequency).  Counts are
disjoint, matching `plot_corner_ajs.py`'s `newly = hit & keep`.
"""
function mask_known(cands, pars, centers, abs_tol, frac_tol, max_harmonic)
    keep = trues(length(cands))
    counts = zeros(Int, length(pars))
    max_harmonic = max(1, max_harmonic)
    for (pi, par) in enumerate(pars)
        half = par_halfwidth(par.f0, abs_tol, frac_tol)
        center = centers[pi]
        for (ci, c) in enumerate(cands)
            keep[ci] || continue
            for h in 1:max_harmonic
                if abs(c.freq - h * center) <= h * half
                    keep[ci] = false
                    counts[pi] += 1
                    break
                end
            end
        end
    end
    return cands[keep], counts
end

"""
    predict_with_python(pars, infpath, ajs_points, python; abs_tol, frac_tol)
        -> Union{Vector{Vector{Float64}},Nothing}

Ask `predict_par_masks.py` (which calls the same functions as
`plot_corner_ajs.py`) for the search frequency of every known pulsar at every
trial ajs point.  `nothing` means the prediction failed and the caller should
fall back to the spin-frequency mask.
"""
function predict_with_python(pars, infpath, ajs_points, python;
                             abs_tol, frac_tol)
    helper = joinpath(@__DIR__, "predict_par_masks.py")
    if !isfile(helper)
        @warn "missing $helper; using the spin-frequency mask"
        return nothing
    end
    tmp, io = mktemp()
    try
        for (a, j, s) in ajs_points
            @printf(io, "%.17g,%.17g,%.17g\n", a, j, s)
        end
        close(io)

        args = String[python, helper, "--inf", infpath, "--ajs", tmp, "--par"]
        append!(args, [p.path for p in pars])
        if abs_tol !== nothing
            push!(args, "--par-tol", string(abs_tol))
        end
        push!(args, "--par-frac-tol", string(frac_tol))

        outbuf = IOBuffer()
        errbuf = IOBuffer()
        try
            run(pipeline(Cmd(args), stdout = outbuf, stderr = errbuf); wait = true)
        catch err
            msg = String(take!(errbuf))
            @warn "ajs-predicted par mask failed; falling back to the spin-frequency mask" python = python exception = err
            isempty(strip(msg)) || print(stderr, msg)
            return nothing
        end

        # Keep the helper's warnings on success (e.g. one par's orbit model
        # failed and that par fell back to F0 inside the Python code).
        warnmsg = String(take!(errbuf))
        isempty(strip(warnmsg)) || print(stderr, warnmsg)

        out = String(take!(outbuf))
        preds = [Vector{Float64}(undef, length(ajs_points)) for _ in pars]
        filled = [falses(length(ajs_points)) for _ in pars]
        for line in split(out, '\n')
            s = strip(line)
            (isempty(s) || startswith(s, "#") || startswith(s, "par_index")) && continue
            f = split(s, ',')
            length(f) >= 6 || continue
            pi = tryparse(Int, f[1])
            ai = tryparse(Int, f[2])
            fp = tryparse(Float64, f[3])
            (pi === nothing || ai === nothing || fp === nothing) && continue
            (1 <= pi + 1 <= length(pars) && 1 <= ai + 1 <= length(ajs_points)) || continue
            preds[pi+1][ai+1] = fp
            filled[pi+1][ai+1] = true
        end
        if any(v -> !all(v), filled)
            @warn "ajs-predicted par mask returned incomplete output; using the spin-frequency mask"
            return nothing
        end
        return preds
    finally
        rm(tmp; force = true)
    end
end

"""
    mask_centers(model, ajs_index, pred_by_ajs, inf_meta, pars)

Mask centres for one `.cohout`.  With an ajs-predicted table, an ajs job uses
its own trial point's prediction.  Otherwise (or for circular jobs) use the
par's spin frequency at the observation midpoint when `--inf` was supplied,
and F0 at PEPOCH when it was not.
"""
function mask_centers(model, ajs_idx, pred_by_ajs, inf_meta, pars)
    if pred_by_ajs !== nothing && model.mode == :ajs && ajs_idx > 0
        return [pred_by_ajs[pi][ajs_idx] for pi in eachindex(pars)]
    end
    if inf_meta !== nothing
        mjd_mid = inf_meta.epoch + 0.5 * inf_meta.tobs / 86400.0
        return [something(par_frequency_at(par.par, mjd_mid), par.f0) for par in pars]
    end
    return [par.f0 for par in pars]
end

# ---------------------------------------------------------------------------
# Candidate files
# ---------------------------------------------------------------------------

struct FoldCandidate
    index::Int
    snr::Float64
    freq::Float64          # Hz
    period_ms::Float64     # ms, exactly the column as written
    nharm::Int
    ducy::Float64          # percent
end

"""
    read_cohout(path) -> Vector{FoldCandidate}

Parse the candidate table (`#Num  'S/N'  Frequency (Hz)  Period (ms)  #Harm
Ducy(%)`).  Comment and blank lines are ignored.  Rows are returned sorted by
S/N descending, then frequency.
"""
function read_cohout(path)
    cands = FoldCandidate[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        cols = split(s)
        length(cols) >= 5 || continue
        idx = tryparse(Int, cols[1])
        snr = tryparse(Float64, cols[2])
        freq = tryparse(Float64, cols[3])
        per = tryparse(Float64, cols[4])
        (idx === nothing || snr === nothing || freq === nothing || per === nothing) &&
            continue
        nh = something(tryparse(Int, cols[5]), 0)
        ducy = length(cols) >= 6 ? something(tryparse(Float64, cols[6]), 0.0) : 0.0
        push!(cands, FoldCandidate(idx, snr, freq, per, nh, ducy))
    end
    sort!(cands; by = c -> (-c.snr, c.freq))
    return cands
end

struct CohoutEntry
    cohout::String
    model::DemodModel
    keep::Vector{FoldCandidate}
    n_masked::Int
end

"""
    surviving_candidates(cohout, model, freq_lo, freq_hi, a, pars,
                         pred_by_ajs, inf_meta, abs_tol, max_harmonic,
                         ajs_index)
        -> (keep, n_masked, per_par_counts)

Read one `.cohout`, apply the period/frequency/S/N windows, then the
known-pulsar mask.  Shared by the normal per-file path and the global DBSCAN
pre-scan so both see exactly the same surviving set.
"""
function surviving_candidates(cohout, model, freq_lo, freq_hi, a, pars,
                              pred_by_ajs, inf_meta, abs_tol, max_harmonic,
                              ajs_index)
    cands = read_cohout(cohout)
    isempty(cands) && return FoldCandidate[], 0, zeros(Int, length(pars))
    keep = [c for c in cands
            if freq_lo <= c.freq <= freq_hi &&
               a["min-period"] <= c.period_ms / 1000.0 <= a["max-period"] &&
               a["min-snr"] <= c.snr <= a["max-snr"]]
    isempty(keep) && return keep, 0, zeros(Int, length(pars))

    n_masked = 0
    counts = zeros(Int, length(pars))
    if !isempty(pars)
        ajs_idx = 0
        if model.mode == :ajs
            ajs_idx = get(ajs_index, (model.params.accel, model.params.jerk,
                                      model.params.snap), 0)
        end
        centers = mask_centers(model, ajs_idx, pred_by_ajs, inf_meta, pars)
        n_before = length(keep)
        keep, counts = mask_known(keep, pars, centers, abs_tol,
                                  a["par-frac-tol"], max_harmonic)
        n_masked = n_before - length(keep)
    end
    return keep, n_masked, counts
end

# ---------------------------------------------------------------------------
# DBSCAN in grid-spacing units
# ---------------------------------------------------------------------------

"""Median of the positive first differences of a sorted value set."""
function _median_positive_diff(v)
    s = sort(collect(v))
    length(s) >= 2 || return NaN
    d = sort([s[i+1] - s[i] for i in 1:length(s)-1 if s[i+1] > s[i]])
    isempty(d) && return NaN
    m = length(d)
    return isodd(m) ? d[(m+1)÷2] : 0.5 * (d[m÷2] + d[m÷2+1])
end

"""
    read_ajs_spacings(path; max_rows=200_000)

Read a grid CSV (`accel,jerk,snap`, optionally `frequency`) and return the
median positive spacing of each axis.  Only enough rows are read to see at
least three distinct values per axis (or `max_rows`), so the multi-GB
`my_grid/nsns.csv` is as cheap as the small `my_grid_search/nsns.csv`.
"""
function read_ajs_spacings(path; max_rows = 200_000)
    isfile(path) || error("--spacing-csv not found: $path")
    header = nothing
    icol = (accel = 0, jerk = 0, snap = 0, freq = 0)
    sets = (Set{Float64}(), Set{Float64}(), Set{Float64}())
    fset = Set{Float64}()
    rows = 0
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        parts = split(s, ',')
        if header === nothing
            header = lowercase.(strip.(parts))
            idx = Dict(h => i for (i, h) in enumerate(header))
            icol = (accel = get(idx, "accel", get(idx, "a", 0)),
                    jerk = get(idx, "jerk", get(idx, "j", 0)),
                    snap = get(idx, "snap", get(idx, "s", 0)),
                    freq = get(idx, "frequency", get(idx, "freq", get(idx, "f", 0))))
            (icol.accel > 0 && icol.jerk > 0 && icol.snap > 0) ||
                error("$path: need accel,jerk,snap columns (or a,j,s)")
            continue
        end
        rows += 1
        length(parts) >= max(icol.accel, icol.jerk, icol.snap) || continue
        for (k, col) in enumerate((icol.accel, icol.jerk, icol.snap))
            v = tryparse(Float64, strip(parts[col]))
            v === nothing || push!(sets[k], v)
        end
        if icol.freq > 0 && length(parts) >= icol.freq
            v = tryparse(Float64, strip(parts[icol.freq]))
            v === nothing || push!(fset, v)
        end
        need = all(length(st) >= 3 for st in sets) &&
               (icol.freq == 0 || length(fset) >= 3)
        if need || rows >= max_rows
            break
        end
    end
    header === nothing && error("$path: empty CSV / no header")
    spacings = (_median_positive_diff(sets[1]),
                _median_positive_diff(sets[2]),
                _median_positive_diff(sets[3]))
    any(x -> !(x > 0), spacings) &&
        error("$path: could not derive positive accel/jerk/snap spacings")
    fspacing = _median_positive_diff(fset)
    return (accel = spacings[1], jerk = spacings[2], snap = spacings[3],
            freq = fspacing, path = path)
end

"""
    dbscan_labels(points, eps, minpts) -> Vector{Int}

Grid-accelerated DBSCAN in 4D.  Labels are `1, 2, ...` for clusters and `-1`
for noise.  `minpts` counts the point itself.  A spatial hash with cell size
`eps` means only the 3^4 neighbouring cells are compared for each point.
"""
function dbscan_labels(points::AbstractMatrix{<:Real}, eps::Real,
                       minpts::Integer)
    n = size(points, 1)
    n == 0 && return Int[]
    eps > 0 || error("DBSCAN eps must be > 0")
    minpts = max(1, Int(minpts))
    inv = 1.0 / eps
    eps2 = eps * eps
    cells = Dict{NTuple{4,Int}, Vector{Int}}()
    for i in 1:n
        key = ntuple(k -> floor(Int, points[i, k] * inv), 4)
        push!(get!(cells, key, Int[]), i)
    end

    parent = collect(1:n)
    rank = zeros(Int, n)
    function findroot(i)
        j = i
        while parent[j] != j
            j = parent[j]
        end
        while parent[i] != i
            p = parent[i]
            parent[i] = j
            i = p
        end
        return j
    end
    function unite!(a, b)
        ra, rb = findroot(a), findroot(b)
        ra == rb && return
        if rank[ra] < rank[rb]
            ra, rb = rb, ra
        end
        parent[rb] = ra
        rank[ra] == rank[rb] && (rank[ra] += 1)
        return
    end

    offsets = Iterators.product(-1:1, -1:1, -1:1, -1:1)
    core = falses(n)
    for i in 1:n
        key = ntuple(k -> floor(Int, points[i, k] * inv), 4)
        neighbours = Int[]
        for off in offsets
            nkey = ntuple(k -> key[k] + off[k], 4)
            list = get(cells, nkey, nothing)
            list === nothing && continue
            for j in list
                d2 = 0.0
                for k in 1:4
                    delta = points[i, k] - points[j, k]
                    d2 += delta * delta
                end
                d2 <= eps2 && push!(neighbours, j)
            end
        end
        if length(neighbours) >= minpts
            core[i] = true
            for j in neighbours
                unite!(i, j)
            end
        end
    end

    root_has_core = Dict{Int,Bool}()
    for i in 1:n
        core[i] && (root_has_core[findroot(i)] = true)
    end
    labels = fill(-1, n)
    root_label = Dict{Int,Int}()
    nextlabel = 0
    for i in 1:n
        r = findroot(i)
        haskey(root_has_core, r) || continue
        lab = get(root_label, r, 0)
        if lab == 0
            nextlabel += 1
            lab = nextlabel
            root_label[r] = lab
        end
        labels[i] = lab
    end
    return labels
end

"""
    dbscan_select(entries, spacings, eps, minpts, top, keep_noise)
        -> (selected, stats)

Cluster every surviving ajs candidate in units of the grid spacings and return
`Dict(cohout => candidates)`, keeping the `top` highest-S/N members of each
cluster (`top == 0` keeps all).  Noise points are discarded unless
`keep_noise`, in which case each becomes a singleton cluster.
"""
function dbscan_select(entries, spacings, eps, minpts, top, keep_noise)
    row_entry = Int[]
    row_cand = Int[]
    ajs = Tuple{Float64,Float64,Float64}[]
    freqs = Float64[]
    ncirc = 0
    for (ei, e) in enumerate(entries)
        if e.model.mode != :ajs
            ncirc += 1
            continue
        end
        p = e.model.params
        for (ci, c) in enumerate(e.keep)
            push!(row_entry, ei)
            push!(row_cand, ci)
            push!(ajs, (p.accel, p.jerk, p.snap))
            push!(freqs, c.freq)
        end
    end
    n = length(freqs)
    n == 0 && return (Dict{String,Vector{FoldCandidate}}(),
                      (points = 0, clusters = 0, noise = 0, selected = 0,
                       circular = ncirc))
    # Subtract minima for numerical comfort; DBSCAN distances are unchanged.
    aref = minimum(x[1] for x in ajs)
    jref = minimum(x[2] for x in ajs)
    sref = minimum(x[3] for x in ajs)
    fref = minimum(freqs)
    points = Matrix{Float64}(undef, n, 4)
    for i in 1:n
        points[i, 1] = (ajs[i][1] - aref) / spacings.accel
        points[i, 2] = (ajs[i][2] - jref) / spacings.jerk
        points[i, 3] = (ajs[i][3] - sref) / spacings.snap
        points[i, 4] = (freqs[i] - fref) / spacings.freq
    end
    labels = dbscan_labels(points, eps, minpts)

    clusters = Dict{Int,Vector{Int}}()
    nnoise = 0
    for i in 1:n
        l = labels[i]
        if l > 0
            push!(get!(clusters, l, Int[]), i)
        else
            nnoise += 1
        end
    end
    groups = collect(values(clusters))
    if keep_noise
        for i in 1:n
            labels[i] == -1 && push!(groups, [i])
        end
    end

    selected = Dict{String,Vector{FoldCandidate}}()
    nsel = 0
    for g in groups
        sort!(g; by = i -> -entries[row_entry[i]].keep[row_cand[i]].snr)
        take = top == 0 ? length(g) : min(top, length(g))
        for k in 1:take
            i = g[k]
            e = entries[row_entry[i]]
            push!(get!(selected, e.cohout, FoldCandidate[]),
                  e.keep[row_cand[i]])
            nsel += 1
        end
    end
    return (selected,
            (points = n, clusters = length(clusters), noise = nnoise,
             selected = nsel, circular = ncirc))
end

# ---------------------------------------------------------------------------
# Source .dat discovery
# ---------------------------------------------------------------------------

"""
    infer_dat(cohout, stem, root) -> Union{String,Nothing}

Find `<stem>.dat` for a `.cohout`.  The search order is the places the search
pipeline normally leaves it: beside the `.cohout`, under the input root, one
directory up (the bundled `OUT/` case), then the working directory, then the
siblings of `root`'s parent.  `nothing` means the caller should ask for
`--dat`.
"""
function infer_dat(cohout, stem, root)
    tried = String[]
    for dir in (dirname(cohout), abspath(root), dirname(dirname(cohout)),
                dirname(abspath(root)), pwd())
        path = joinpath(dir, stem * ".dat")
        path in tried && continue
        push!(tried, path)
        isfile(path) && return path
    end
    # One level of sibling directories under the parent, for layouts like
    # `<run>/OUT/*.cohout` + `<run>/data/<stem>.dat`.
    parent = dirname(dirname(cohout))
    if isdir(parent)
        for entry in readdir(parent; join = true)
            isdir(entry) || continue
            path = joinpath(entry, stem * ".dat")
            isfile(path) && return path
        end
    end
    return nothing
end

"""
    inf_value(path, keyprefix) -> Union{String,Nothing}

First value in a PRESTO `.inf` whose key starts with `keyprefix`.  The keys
carry variable whitespace (e.g. `Barycentered?           (1 yes, 0 no)`), so a
prefix match is the only stable way to read one.
"""
function inf_value(path, keyprefix)
    for line in eachline(path)
        occursin("=", line) || continue
        k, v = split(line, "=", limit = 2)
        startswith(strip(k), keyprefix) && return strip(v)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

"""
    fold_series(ts, dt, freq, nbins, nsubints) -> (profile, subprofiles, counts)

Time-domain fold at `freq` [Hz], phase referenced to the midpoint of the
demodulated series (the same anchor the ajs/circular remaps use).  Returns the
summed mean profile (length `nbins`), an `nsubints × nbins` matrix of
sub-integration means, and the per-bin sample counts.  Samples are histogrammed
in place, so the work is O(N) in time and O(nbins·nsubints) in memory.
"""
function fold_series(ts::AbstractVector, dt::Real, freq::Real,
                     nbins::Integer, nsubints::Integer)
    n = length(ts)
    n > 0 || error("cannot fold an empty series")
    nsubints = max(1, nsubints)
    sums = zeros(Float64, nbins)
    counts = zeros(Int64, nbins)
    sub_sums = zeros(Float64, nsubints, nbins)
    sub_counts = zeros(Int64, nsubints, nbins)
    half = n / 2.0
    nb = Int(nbins)
    ns = Int(nsubints)
    @inbounds for i in 1:n
        t = (i - half) * dt
        # mod() (not rem) keeps negative times in [0, 1), which is what the
        # phase bin needs on both sides of the midpoint.
        b = Int(floor(mod(freq * t, 1.0) * nb)) + 1
        b > nb && (b = nb)
        s = Int(fld((i - 1) * ns, n)) + 1
        s > ns && (s = ns)
        v = Float64(ts[i])
        sums[b] += v
        counts[b] += 1
        sub_sums[s, b] += v
        sub_counts[s, b] += 1
    end
    prof = sums ./ max.(counts, 1)
    subprof = sub_sums ./ max.(sub_counts, 1)
    return prof, subprof, counts
end

# --- Morello et al. (2014) best-boxcar S/N --------------------------------

# The search ladder used when --nbins is not given.  All divide 1024.
const BOXCAR_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

"""
    best_boxcar(profile; widths=BOXCAR_WIDTHS) -> (width, start, snr)

Search every circular on-pulse window of the requested widths and return the
one maximising Morello et al. (2014, MNRAS 443, 1651) Eq. 1:

    S/N = sum_{i in W}(p_i - bbar) / (sigma * sqrt(w))

where `W` is the contiguous on-pulse window (w phase bins), and `bbar` and
`sigma` are the mean and standard deviation of the bins **outside** W.  Both
the window position and the width are optimised.  The profile is circular, so
a window may wrap through phase 1.  `start` is the 1-based first bin of the
winning window.
"""
function best_boxcar(profile::AbstractVector{<:Real};
                     widths = BOXCAR_WIDTHS)
    n = length(profile)
    n > 1 || return 1, 1, 0.0
    # Duplicate once so a circular window of any width is a contiguous slice.
    p2 = vcat(profile, profile)
    # Leading zero so sum(p2[a:b]) = cs[b+1] - cs[a].
    cs = vcat(0.0, cumsum(Float64.(p2)))
    cs2 = vcat(0.0, cumsum(Float64.(p2) .^ 2))
    total = cs[n+1]
    total2 = cs2[n+1]

    best_snr = -Inf
    best_w = 1
    best_start = 1
    for w in widths
        (w >= 1 && w < n) || continue
        noff = n - w
        noff >= 1 || continue
        sw2_sqrtw = sqrt(w)
        for s in 1:n
            e = s + w - 1
            sw = cs[e+1] - cs[s]
            sw2 = cs2[e+1] - cs2[s]
            soff = total - sw
            soff2 = total2 - sw2
            bbar = soff / noff
            var = soff2 / noff - bbar^2
            sigma = sqrt(max(var, 0.0))
            sigma > 0 || continue
            snr = (sw - w * bbar) / (sigma * sw2_sqrtw)
            if snr > best_snr
                best_snr = snr
                best_w = w
                best_start = s
            end
        end
    end
    isfinite(best_snr) || (best_snr = 0.0)
    return best_w, best_start, best_snr
end

"""
    boxcar_snr_groups(profile, w, start) -> Vector{Float64}

The Morello S/N of every disjoint `w`-bin window, starting at `start` and
stepping by `w` around the circle.  The first entry is the winning window when
`start` came from `best_boxcar`; the maximum is then the best S/N.  This is what
the profile plot shows in auto mode, so its peak and the title's S/N agree.
"""
function boxcar_snr_groups(profile::AbstractVector{<:Real}, w::Integer,
                           start::Integer)
    n = length(profile)
    n % w == 0 || error("profile length $n is not divisible by boxcar width $w")
    p2 = vcat(profile, profile)
    cs = vcat(0.0, cumsum(Float64.(p2)))
    cs2 = vcat(0.0, cumsum(Float64.(p2) .^ 2))
    total = cs[n+1]
    total2 = cs2[n+1]
    noff = n - w
    m = n ÷ w
    snrs = zeros(Float64, m)
    for j in 1:m
        s = mod1(start + (j - 1) * w, n)
        e = s + w - 1
        sw = cs[e+1] - cs[s]
        sw2 = cs2[e+1] - cs2[s]
        bbar = (total - sw) / noff
        sigma = sqrt(max((total2 - sw2) / noff - bbar^2, 0.0))
        snrs[j] = sigma > 0 ? (sw - w * bbar) / (sigma * sqrt(w)) : 0.0
    end
    return snrs
end

"""
    downsample_phase(prof, subprof, counts, w, offset)
        -> (prof_ds, subprof_ds, counts_ds)

Average consecutive circular groups of `w` phase bins, starting at `offset`
(the winning boxcar window), reducing the profile from N to N/w bins.  The
sub-integration waterfall is downsampled identically.  The result is rolled by
`m ÷ 2` output bins so the winning on-pulse window lands near phase 0.5; phase
zero is arbitrary in a fold, and this keeps the pulse centred in the plot.
"""
function downsample_phase(prof, subprof, counts, w::Integer, offset::Integer)
    n = length(prof)
    w >= 1 || error("downsample width must be >= 1")
    n % w == 0 || error("profile length $n is not divisible by boxcar width $w")
    m = n ÷ w
    prof_ds = zeros(Float64, m)
    counts_ds = zeros(Int64, m)
    sub_ds = zeros(Float64, size(subprof, 1), m)
    for j in 1:m
        start = mod1(offset + (j - 1) * w, n)
        prof_ds[j] = sum(prof[mod1(start + k, n)] for k in 0:w-1) / w
        counts_ds[j] = sum(counts[mod1(start + k, n)] for k in 0:w-1)
        for si in axes(subprof, 1)
            sub_ds[si, j] = sum(subprof[si, mod1(start + k, n)] for k in 0:w-1) / w
        end
    end
    # Put the winning window near phase 0.5.
    shift = m ÷ 2
    prof_ds = circshift(prof_ds, shift)
    counts_ds = circshift(counts_ds, shift)
    sub_ds = circshift(sub_ds, (0, shift))
    return prof_ds, sub_ds, counts_ds
end

# --- robust profile S/N (median, IQR/1.349), matching the repo convention ---

function _quantile_sorted(v::AbstractVector{<:Real}, p::Real)
    n = length(v)
    n == 0 && return NaN
    h = (n - 1) * p
    lo = floor(Int, h) + 1
    hi = min(lo + 1, n)
    return v[lo] + (h - (lo - 1)) * (v[hi] - v[lo])
end

"""
    profile_snr(profile) -> (snr_profile, peak_snr)

Median-subtract the profile, divide by IQR/1.349, and return both the
standardised profile and its peak.  This is the same robust S/N the FFA fold
diagnostics use; it is a shape diagnostic, not the search's matched-filter
S/N.
"""
function profile_snr(profile::AbstractVector{<:Real})
    v = sort(Float64.(profile))
    med = _quantile_sorted(v, 0.5)
    iqr = _quantile_sorted(v, 0.75) - _quantile_sorted(v, 0.25)
    sigma = iqr / 1.349
    if !(sigma > 0)
        return zeros(length(profile)), 0.0
    end
    out = (profile .- med) ./ sigma
    return out, maximum(out)
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

function write_profile(path, cand, model, freq, prof, snrprof, counts,
                       foldsnr, ndata, nbins, nsubints, datpath;
                       n_masked = 0, nbins_initial = nbins,
                       boxcar_width = 0, boxcar_start = 0,
                       snr_method = "profile")
    open(path, "w") do io
        println(io, "# Folded profile from ", basename(path))
        println(io, "# dat = ", datpath)
        println(io, "# model = ", model_string(model))
        println(io, "# candidate_index = ", cand.index)
        println(io, "# candidate_snr = ", cand.snr)
        println(io, "# frequency_hz = ", freq)
        println(io, "# period_s = ", 1.0 / freq)
        println(io, "# period_ms = ", cand.period_ms)
        println(io, "# nharm = ", cand.nharm)
        println(io, "# fold_snr = ", foldsnr)
        println(io, "# snr_method = ", snr_method)
        println(io, "# nbins_initial = ", nbins_initial)
        println(io, "# boxcar_width = ", boxcar_width)
        println(io, "# boxcar_start = ", boxcar_start)
        println(io, "# n_samples = ", ndata)
        println(io, "# nbins = ", nbins)
        println(io, "# nsubints = ", nsubints)
        println(io, "# par_mask_removed = ", n_masked)
        # `snr` is the best-boxcar S/N per displayed window in auto mode, and
        # the IQR-standardised profile in manual --nbins mode; # snr_method says
        # which.
        println(io, "phase\tcounts\tmean\tsnr")
        for b in 1:length(prof)
            @printf(io, "%.10f\t%d\t%.10g\t%.10g\n",
                    (b - 0.5) / length(prof), counts[b], prof[b], snrprof[b])
        end
    end
    return path
end

function plot_fold(path, cand, model, freq, prof, snrprof, subprof, foldsnr;
                   snr_method = "profile", nbins_initial = length(prof),
                   boxcar_width = 0)
    _HAVE_MAKIE || return nothing
    phase = ((1:length(prof)) .- 0.5) ./ length(prof)
    ns = size(subprof, 1)
    xt = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"])
    method = snr_method == "boxcar" ?
        @sprintf("boxcar w=%d, %d -> %d bins", boxcar_width,
                 nbins_initial, length(prof)) : "profile IQR"
    title = @sprintf("%s\nf = %.10f Hz   P = %.6f ms   cand #%d: S/N %.2f, #Harm %d   fold S/N %.2f (%s)\n%d phase bins, %d sub-integrations   %s",
                     splitext(basename(path))[1], freq, 1000.0 / freq, cand.index,
                     cand.snr, cand.nharm, foldsnr, method, length(prof), ns,
                     model_string(model))
    fig = CairoMakie.Figure(size = (1000, ns > 1 ? 800 : 560))
    ylabel = snr_method == "boxcar" ? "Boxcar S/N" : "Profile S/N (IQR)"
    ax1 = CairoMakie.Axis(fig[1, 1]; title = title, titlesize = 10,
                          xlabel = ns > 1 ? "" : "Pulse phase",
                          ylabel = ylabel, xticks = xt)
    CairoMakie.lines!(ax1, phase, snrprof; color = :navy, linewidth = 1.4)
    CairoMakie.xlims!(ax1, 0, 1)
    if ns > 1
        ax2 = CairoMakie.Axis(fig[2, 1]; xlabel = "Pulse phase",
                              ylabel = "Sub-integration", xticks = xt)
        # Remove each sub-integration's own mean so the waterfall is not
        # dominated by a slow baseline drift.
        disp = subprof .- sum(subprof; dims = 2) ./ size(subprof, 2)
        # Makie maps matrix[i, j] to (x[i], y[j]).  `disp` is
        # nsubints × nbins, so transpose it: phase must be the first (x)
        # index and sub-integration the second (y).  Phase edges make the
        # horizontal axis exactly 0..1.
        phase_edges = range(0.0, 1.0; length = size(subprof, 2) + 1)
        # Reversed greys: dark = more signal, white = background/no signal.
        CairoMakie.heatmap!(ax2, phase_edges, 1:ns, permutedims(disp);
                            colormap = CairoMakie.Reverse(:grays))
        # Same horizontal range, and linked so zoom/pan keeps profile and
        # waterfall aligned.
        CairoMakie.xlims!(ax2, 0, 1)
        CairoMakie.linkxaxes!(ax1, ax2)
    end
    CairoMakie.save(path, fig; px_per_unit = 2)
    return path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_cmdline(argv)
    s = ArgParseSettings(
        description = "Demodulate a PRESTO .dat at the accel/jerk/snap or circular-orbit point named in each .cohout and fold at its detected period/frequency.",
        epilog = "Example: julia --project=. demod/fold_demod.jl test/test_grid_ajs_efficacy/OUT --min-snr 6 --min-period 0.01 --max-period 1.0\n" *
                 "Put INPUT before --par when passing several par paths: --par consumes the bare words that follow it.",
    )
    @add_arg_table! s begin
        "input"
            help = "directory of *.cohout files, or one .cohout file"
            required = true
        "--dat"
            help = "source PRESTO .dat (default: infer <stem>.dat from each .cohout)"
            arg_type = String
        "--outdir"
            help = "output directory (default: <input>/folds)"
            arg_type = String
            default = ""
        "--min-period"
            help = "minimum candidate period [s] (default 0)"
            arg_type = Float64
            default = 0.0
        "--max-period"
            help = "maximum candidate period [s] (default Inf)"
            arg_type = Float64
            default = Inf
        "--min-freq"
            help = "minimum candidate frequency [Hz] (default 0)"
            arg_type = Float64
            default = 0.0
        "--max-freq"
            help = "maximum candidate frequency [Hz] (default Inf)"
            arg_type = Float64
            default = Inf
        "--min-snr"
            help = "minimum reported search S/N (default 0)"
            arg_type = Float64
            default = 0.0
        "--max-snr"
            help = "maximum reported search S/N (default Inf)"
            arg_type = Float64
            default = Inf
        "--top"
            help = "fold the N best candidates per .cohout (default 1; 0 = all). With --dbscan, N is per cluster instead"
            arg_type = Int
            default = 1
        "--nbins"
            help = "phase bins in the fold. Default 0 = auto: fold at 1024 bins, search the Morello et al. (2014) boxcar widths 1,2,4,8,16,32,64,128,256, then downsample by the best width (1024/w bins). Give a positive value to bypass the boxcar search and use that many bins"
            arg_type = Int
            default = 0
        "--nsubints"
            help = "sub-integrations in the waterfall (default 16; 1 = off)"
            arg_type = Int
            default = 16
        "--recursive"
            help = "also search subdirectories for *.cohout"
            action = :store_true
        "--limit"
            help = "stop after this many .cohout files (0 = all)"
            arg_type = Int
            default = 0
        "--jobs"
            help = "number of independent worker processes for demod+fold (default 1). With --dbscan, the parent clusters once and then shards only the selected candidates. Each worker holds its own .dat and demodulated series in memory, so memory scales with --jobs; 4 is a sensible starting point. A worker that finds no surviving candidate is reported separately, not as a failure"
            arg_type = Int
            default = 1
        "--dbscan"
            help = "cluster all surviving candidates in (accel,jerk,snap,frequency) before folding, and fold only the top --top candidates per DBSCAN cluster. Clustering is global; with --jobs the parent clusters once and then shards only the selected candidates"
            action = :store_true
        "--spacing-csv"
            help = "CSV of grid points (e.g. my_grid_search/nsns.csv) used to derive the accel/jerk/snap spacer sizes for --dbscan. Must have accel,jerk,snap columns; an optional frequency/freq column is used for the frequency spacing"
            arg_type = String
            default = ""
        "--freq-spacing"
            help = "frequency spacing [Hz] for --dbscan. If 0, use the spacing-csv frequency column, else 1/T from --inf, --dat's .inf, or the first inferred .dat"
            arg_type = Float64
            default = 0.0
        "--dbscan-eps"
            help = "DBSCAN eps in units of the grid spacings (default 1 = one accel/jerk/snap/frequency step)"
            arg_type = Float64
            default = 1.0
        "--dbscan-minpts"
            help = "DBSCAN minimum points per cluster, including the point itself (default 2)"
            arg_type = Int
            default = 2
        "--dbscan-keep-noise"
            help = "also fold DBSCAN noise points, each as its own singleton cluster (default: discard noise)"
            action = :store_true
        "--no-plot"
            help = "write text profiles only (do not load CairoMakie)"
            action = :store_true
        "--par"
            help = ".par file(s), directory(ies), or glob(s) of known pulsars whose F0 and harmonics are removed from the candidate list; repeat the flag for several paths (the same mask as plot_corner_ajs.py)"
            action = :append_arg
            nargs = '+'
            arg_type = String
            default = String[]
        "--pars"
            help = "alias for --par"
            action = :append_arg
            nargs = '+'
            arg_type = String
            default = String[]
        "--par-tol"
            help = "absolute frequency half-width for par masking [Hz] (default: max(0.005, 5e-4*F0))"
            arg_type = Float64
            default = NaN
        "--par-frac-tol"
            help = "fractional-F0 half-width for par masking; combined with --par-tol using the larger value (default 0)"
            arg_type = Float64
            default = 0.0
        "--par-harmonics"
            help = "also mask harmonics 2..N of each known-pulsar frequency (default 1, fundamental only)"
            arg_type = Int
            default = 1
        "--inf"
            help = "observation .inf; with --par, predict each known pulsar's searched frequency separately for every ajs trial point (same prediction as plot_corner_ajs.py --inf). Without it the mask is centred on F0 at PEPOCH"
            arg_type = String
            default = ""
        "--python"
            help = "Python executable used only for the --inf par-mask prediction (default: the PYTHON environment variable, else python3)"
            arg_type = String
            default = get(ENV, "PYTHON", "python3")
        "--accel"
            help = "override the filename accel [m/s^2] (requires --jerk and --snap)"
            arg_type = Float64
            default = NaN
        "--jerk"
            help = "override the filename jerk [m/s^3] (requires --accel and --snap)"
            arg_type = Float64
            default = NaN
        "--snap"
            help = "override the filename snap [m/s^4] (requires --accel and --jerk)"
            arg_type = Float64
            default = NaN
        "--pb"
            help = "override the filename circular period [d] (requires --x and --at)"
            arg_type = Float64
            default = NaN
        "--x"
            help = "override the filename circular projected axis [lt-s] (requires --pb and --at)"
            arg_type = Float64
            default = NaN
        "--at"
            help = "override the filename circular mean anomaly [rad] (requires --pb and --x)"
            arg_type = Float64
            default = NaN
    end
    return parse_args(argv, s)
end

function collect_cohouts(input; recursive)
    if isfile(input)
        endswith(lowercase(input), ".cohout") ||
            error("$input is not a .cohout file")
        return [abspath(input)]
    end
    isdir(input) || error("no such file or directory: $input")
    paths = String[]
    if recursive
        for (dir, _, files) in walkdir(input)
            for f in files
                endswith(lowercase(f), ".cohout") && push!(paths, joinpath(dir, f))
            end
        end
    else
        for f in readdir(input)
            endswith(lowercase(f), ".cohout") && push!(paths, joinpath(input, f))
        end
    end
    sort!(paths)
    return paths
end

"""
    read_selection(path) -> (sel, masked)

Read the parent's DBSCAN selection file:
`#cohout,candidate_index,n_masked`.  `sel` maps each absolute cohout path to
the candidate indices to fold; `masked` carries the per-file par-mask count so
the worker's profile header stays accurate.
"""
function read_selection(path)
    sel = Dict{String,Vector{Int}}()
    masked = Dict{String,Int}()
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s, ',')
        length(parts) >= 2 || continue
        coh = String(strip(parts[1]))
        idx = tryparse(Int, strip(parts[2]))
        idx === nothing && continue
        push!(get!(sel, coh, Int[]), idx)
        if length(parts) >= 3
            n = tryparse(Int, strip(parts[3]))
            n === nothing || (masked[coh] = n)
        end
    end
    return sel, masked
end

"""
    run_parallel(a, cohouts, outdir, argv) -> Union{Int,Nothing}

Spawn `a["jobs"]` independent worker **processes**, each running this same
script on a round-robin shard of the `.cohout` list.  A child writes its
profiles/PNGs into the shared `outdir` (names are unique per `.cohout`) and its
summary to `outdir/.shards/folds_<i>.csv`; this process waits and merges them
into `outdir/folds.csv`.  `nothing` means fewer than two jobs are useful.
"""
function run_parallel(a, cohouts, outdir, argv; selection_path = "")
    njobs = min(a["jobs"], length(cohouts))
    njobs > 1 || return nothing
    sharddir = joinpath(outdir, ".shards")
    isdir(sharddir) && rm(sharddir; recursive = true, force = true)
    mkpath(sharddir)
    summary = joinpath(outdir, "folds.csv")

    projfile = something(Base.active_project(),
                         joinpath(@__DIR__, "..", "Project.toml"))
    proj = dirname(projfile)
    script = abspath(@__FILE__)
    @printf("fold_demod: %d .cohout file(s) under %s\n", length(cohouts), a["input"])
    @printf("            output %s\n", outdir)
    @printf("            parallel: %d worker process(es)\n", njobs)
    flush(stdout)

    procs = Base.Process[]
    launch_failed = false
    for i in 1:njobs
        shard_summary = joinpath(sharddir, "folds_$(i).csv")
        cmd = `$(Base.julia_cmd()) --project=$proj $script $argv`
        cmd = addenv(cmd,
                     "COHERENT_FOLD_SHARD_INDEX" => string(i),
                     "COHERENT_FOLD_SHARD_TOTAL" => string(njobs),
                     "COHERENT_FOLD_SUMMARY" => shard_summary)
        if !isempty(selection_path)
            cmd = addenv(cmd, "COHERENT_FOLD_SELECTION" => selection_path)
        end
        try
            # Inherit this process's streams so worker progress/errors are
            # visible (plain `run(...; wait=false)` sends them to /dev/null).
            push!(procs, run(pipeline(cmd; stdout = stdout, stderr = stderr);
                             wait = false))
        catch err
            @error "could not launch worker" worker = i exception = err
            launch_failed = true
        end
    end

    ok = !launch_failed
    zero_fold = 0
    for (i, p) in enumerate(procs)
        wait(p)
        if p.exitcode == 2
            # A worker that completed but found no surviving candidate is not a
            # failure; exit code 2 keeps it distinct from a real error.
            zero_fold += 1
        elseif !success(p)
            ok = false
            @warn "worker failed (see its output above for the error)" worker = i status = p.exitcode
        end
    end
    if zero_fold > 0
        @printf("            %d worker(s) completed with no surviving candidate (filtered/masked)\n",
                zero_fold)
    end
    zero_fold == length(procs) && !isempty(procs) && (ok = false)

    open(summary, "w") do io
        wrote_header = false
        for i in 1:njobs
            path = joinpath(sharddir, "folds_$(i).csv")
            isfile(path) || continue
            for (ln, line) in enumerate(eachline(path))
                if ln == 1
                    wrote_header && continue
                    wrote_header = true
                end
                println(io, line)
            end
        end
    end
    @printf("fold_demod: parallel run finished; summary %s\n", summary)
    return ok ? 0 : 1
end

function main(argv)
    a = parse_cmdline(argv)

    # A child process (launched by run_parallel) is one shard and must not
    # spawn further workers.  The parent passes these through the environment.
    shard_total = something(tryparse(Int, get(ENV, "COHERENT_FOLD_SHARD_TOTAL", "1")), 1)
    shard_index = something(tryparse(Int, get(ENV, "COHERENT_FOLD_SHARD_INDEX", "1")), 1)
    selection_path = get(ENV, "COHERENT_FOLD_SELECTION", "")
    selection_mode = !isempty(selection_path)
    if shard_total > 1 || selection_mode
        a["jobs"] = 1
    end
    if shard_total > 1
        1 <= shard_index <= shard_total ||
            error("bad shard index $shard_index of $shard_total")
    end

    selmap = Dict{String,Vector{Int}}()
    masked_map = Dict{String,Int}()
    if selection_mode
        selmap, masked_map = read_selection(selection_path)
        isempty(selmap) && error("empty DBSCAN selection file: $selection_path")
    end

    input = abspath(a["input"])
    cohouts = selection_mode ? sort(collect(keys(selmap))) :
                              collect_cohouts(input; recursive = a["recursive"])
    isempty(cohouts) && error("no .cohout files found under $input")

    a["jobs"] < 1 && error("--jobs must be >= 1")
    if a["dbscan"]
        a["spacing-csv"] == "" &&
            error("--dbscan needs --spacing-csv (e.g. my_grid_search/nsns.csv)")
        isfile(a["spacing-csv"]) ||
            error("--spacing-csv not found: $(a["spacing-csv"])")
        a["dbscan-eps"] <= 0 && error("--dbscan-eps must be > 0")
        a["dbscan-minpts"] < 1 && error("--dbscan-minpts must be >= 1")
    end
    a["freq-spacing"] < 0 && error("--freq-spacing must be >= 0")
    a["min-period"] < 0 && error("--min-period must be >= 0")
    a["min-freq"] < 0 && error("--min-freq must be >= 0")
    a["min-snr"] < 0 && error("--min-snr must be >= 0")
    a["max-period"] < a["min-period"] &&
        error("--max-period must be >= --min-period")
    a["max-freq"] < a["min-freq"] &&
        error("--max-freq must be >= --min-freq")
    a["max-snr"] < a["min-snr"] && error("--max-snr must be >= --min-snr")
    a["top"] < 0 && error("--top must be >= 0")
    a["nbins"] != 0 && a["nbins"] < 2 && error("--nbins must be 0 (auto) or >= 2")
    a["nsubints"] < 1 && error("--nsubints must be >= 1")

    # Period and frequency windows are intersected.  max-period -> min
    # frequency, min-period -> max frequency; the defaults (0/Inf) add nothing.
    root = isdir(input) ? input : dirname(input)
    from_max_period = isfinite(a["max-period"]) && a["max-period"] > 0 ?
        1.0 / a["max-period"] : 0.0
    from_min_period = a["min-period"] > 0 ? 1.0 / a["min-period"] : Inf
    freq_lo = max(a["min-freq"], from_max_period)
    freq_hi = min(a["max-freq"], from_min_period)
    freq_lo <= freq_hi ||
        error(@sprintf("empty frequency window: %.6g .. %.6g Hz", freq_lo, freq_hi))

    outdir = isempty(a["outdir"]) ? joinpath(root, "folds") : abspath(a["outdir"])
    mkpath(outdir)

    if a["limit"] > 0 && !selection_mode && length(cohouts) > a["limit"]
        cohouts = cohouts[1:a["limit"]]
    end
    if shard_total > 1
        cohouts = cohouts[shard_index:shard_total:end]
    end

    # Non-DBSCAN parent: shard the (already limited) file list across workers.
    # DBSCAN is different: it must cluster globally first, then it shards the
    # selected candidates after the pre-scan below.
    if !a["dbscan"] && !selection_mode && a["jobs"] > 1 && shard_total == 1
        rc = run_parallel(a, cohouts, outdir, argv)
        rc !== nothing && return rc
    end

    fallback_stem = a["dat"] === nothing ? "" : splitext(basename(a["dat"]))[1]
    manual = manual_model(a["accel"], a["jerk"], a["snap"],
                          a["pb"], a["x"], a["at"], fallback_stem)
    manual !== nothing && a["dat"] === nothing &&
        error("a manual model override needs --dat (the filename no longer supplies the stem)")

    @printf("fold_demod: %d .cohout file(s) under %s\n", length(cohouts), input)
    @printf("            frequency window %.6g .. %.6g Hz, S/N window %.6g .. %.6g\n",
            freq_lo, freq_hi, a["min-snr"], a["max-snr"])
    @printf("            output %s\n", outdir)

    if selection_mode
        # DBSCAN worker: the parent already clustered, applied the mask and
        # wrote this selection file.  Build only the selected candidates.
        pars = KnownPar[]
        abs_tol = nothing
        max_harmonic = 1
        infpath_obs = nothing
        inf_meta = nothing
        ajs_index = Dict{Tuple{Float64,Float64,Float64},Int}()
        pred_by_ajs = nothing
        mask_mode = "F0"
        mask_counts = zeros(Int, 0)
        mask_total = 0
        mask_removed = 0
        dbscan_spacings = nothing
        nfold = 0
        nfile = 0
        nskip = 0
        warned_bary = Set{String}()
        entries = CohoutEntry[]
        for (ifi, cohout) in enumerate(cohouts)
            model = manual === nothing ? model_from_name(cohout) : manual
            if model === nothing
                @printf("[%d/%d] skip %s (not a demod .cohout)\n",
                        ifi, length(cohouts), basename(cohout))
                nskip += 1
                continue
            end
            idxset = Set(selmap[cohout])
            selected = [c for c in read_cohout(cohout) if c.index in idxset]
            if isempty(selected)
                nskip += 1
                continue
            end
            push!(entries, CohoutEntry(cohout, model, selected,
                                       get(masked_map, cohout, 0)))
        end
        selected_by_cohout = Dict{String,Vector{FoldCandidate}}()
    else
    # ---- known-pulsar mask setup (same mechanism as plot_corner_ajs.py) ----
    par_args = String[]
    for group in vcat(a["par"], a["pars"]), p in group
        push!(par_args, p)
    end
    pars = isempty(par_args) ? KnownPar[] : gather_pars(par_args)
    if isempty(par_args)
        a["inf"] == "" || @warn "--inf has no effect without --par"
    elseif isempty(pars)
        error("--par was given but no usable .par files were found")
    end
    abs_tol = isnan(a["par-tol"]) ? nothing : a["par-tol"]
    abs_tol !== nothing && abs_tol < 0 && error("--par-tol must be >= 0")
    a["par-frac-tol"] < 0 && error("--par-frac-tol must be >= 0")
    max_harmonic = max(1, a["par-harmonics"])

    infpath_obs = a["inf"] == "" ? nothing : abspath(a["inf"])
    inf_meta = nothing
    if infpath_obs !== nothing
        isfile(infpath_obs) || error("--inf file not found: $infpath_obs")
        Ninf, dtinf, epochinf = parse_inf(infpath_obs)
        inf_meta = (epoch = epochinf, tobs = Ninf * dtinf)
        bary_obs = inf_value(infpath_obs, "Barycentered?")
        if bary_obs !== nothing && startswith(bary_obs, "0")
            @warn "--inf says the observation is not barycentered; the par mask assumes barycentric frequencies" inf = infpath_obs
        end
    end

    # With --inf, predict the searched frequency of each known pulsar for every
    # unique ajs point ONCE, so a sweep pays one Python start-up, not one per
    # .cohout.  ajs points come from the filenames (or a manual override).
    ajs_index = Dict{Tuple{Float64,Float64,Float64},Int}()
    ajs_points = Tuple{Float64,Float64,Float64}[]
    pred_by_ajs = nothing
    if !isempty(pars) && infpath_obs !== nothing
        for cohout in cohouts
            model = manual === nothing ? model_from_name(cohout) : manual
            (model === nothing || model.mode != :ajs) && continue
            key = (model.params.accel, model.params.jerk, model.params.snap)
            if !haskey(ajs_index, key)
                push!(ajs_points, key)
                ajs_index[key] = length(ajs_points)
            end
        end
        if !isempty(ajs_points)
            @printf("            predicting the par mask for %d ajs point(s) and %d known pulsar(s)\n",
                    length(ajs_points), length(pars))
            pred_by_ajs = predict_with_python(pars, infpath_obs, ajs_points,
                                              a["python"]; abs_tol = abs_tol,
                                              frac_tol = a["par-frac-tol"])
        end
    end
    mask_mode = pred_by_ajs !== nothing ? "ajs-predicted" :
                (inf_meta !== nothing ? "spin-at-midpoint" : "F0")
    mask_counts = zeros(Int, length(pars))
    mask_total = 0
    mask_removed = 0

    # ---- DBSCAN spacing setup ------------------------------------------
    dbscan_spacings = nothing
    if a["dbscan"]
        sp = read_ajs_spacings(a["spacing-csv"])
        df = a["freq-spacing"]
        if !(df > 0)
            if isfinite(sp.freq) && sp.freq > 0
                df = sp.freq
            elseif inf_meta !== nothing
                df = 1.0 / inf_meta.tobs
            elseif a["dat"] !== nothing
                inffile = replace(abspath(a["dat"]), r"\.dat$" => ".inf")
                isfile(inffile) || error("--dbscan: missing $inffile for --freq-spacing")
                N0, dt0, _ = parse_inf(inffile)
                df = 1.0 / (N0 * dt0)
            else
                for coh in cohouts
                    m = manual === nothing ? model_from_name(coh) : manual
                    m === nothing && continue
                    dp = infer_dat(coh, m.stem, root)
                    dp === nothing && continue
                    inffile = replace(dp, r"\.dat$" => ".inf")
                    isfile(inffile) || continue
                    N0, dt0, _ = parse_inf(inffile)
                    df = 1.0 / (N0 * dt0)
                    break
                end
                df > 0 || error("--dbscan could not derive a frequency spacing; pass --freq-spacing, --inf or --dat")
            end
        end
        dbscan_spacings = (accel = sp.accel, jerk = sp.jerk, snap = sp.snap,
                           freq = df, path = sp.path)
        @printf("DBSCAN spacings from %s: accel=%.10g jerk=%.10g snap=%.10g freq=%.10g Hz\n",
                sp.path, sp.accel, sp.jerk, sp.snap, df)
        open(joinpath(outdir, "dbscan_spacings.txt"), "w") do io
            println(io, "source_csv = ", sp.path)
            println(io, "accel = ", sp.accel)
            println(io, "jerk = ", sp.jerk)
            println(io, "snap = ", sp.snap)
            println(io, "frequency_hz = ", df)
        end
    end

    # ---- candidate pre-scan --------------------------------------------
    # Read every .cohout and apply the windows/mask once.  DBSCAN needs the
    # global set before anything is folded; the normal path also uses this so
    # the two selection routes cannot diverge.
    nfold = 0
    nfile = 0
    nskip = 0
    warned_bary = Set{String}()
    entries = CohoutEntry[]
    for (ifi, cohout) in enumerate(cohouts)
        model = manual === nothing ? model_from_name(cohout) : manual
        if model === nothing
            @printf("[%d/%d] skip %s (not a demod .cohout)\n",
                    ifi, length(cohouts), basename(cohout))
            nskip += 1
            continue
        end
        keep, n_masked, counts = surviving_candidates(
            cohout, model, freq_lo, freq_hi, a, pars, pred_by_ajs, inf_meta,
            abs_tol, max_harmonic, ajs_index)
        if !isempty(pars)
            mask_counts .+= counts
            mask_total += n_masked + length(keep)
            mask_removed += n_masked
        end
        if isempty(keep)
            @printf("[%d/%d] skip %s (no surviving candidate)\n",
                    ifi, length(cohouts), basename(cohout))
            nskip += 1
            continue
        end
        push!(entries, CohoutEntry(cohout, model, keep, n_masked))
    end

    # ---- DBSCAN selection ----------------------------------------------
    selected_by_cohout = Dict{String,Vector{FoldCandidate}}()
    if a["dbscan"]
        selected_by_cohout, stats = dbscan_select(
            entries, dbscan_spacings, a["dbscan-eps"], a["dbscan-minpts"],
            a["top"], a["dbscan-keep-noise"])
        topmsg = a["top"] == 0 ? "all" : string(a["top"])
        @printf("DBSCAN: %d candidate(s) -> %d cluster(s) + %d noise; selected %d candidate(s) (top %s per cluster)\n",
                stats.points, stats.clusters, stats.noise, stats.selected, topmsg)
        stats.circular > 0 &&
            @warn "DBSCAN skipped circular-model .cohout files (no ajs coordinates)" count = stats.circular
        stats.points == 0 && @warn "DBSCAN found no ajs candidates to cluster"
    end

    # DBSCAN parent with --jobs: the global clustering above is done once;
    # now write the selected candidates and shard only those folds.
    if a["dbscan"] && a["jobs"] > 1 && shard_total == 1
        sel_cohouts = sort(collect(keys(selected_by_cohout)))
        if length(sel_cohouts) > 1
            selpath = joinpath(outdir, "dbscan_selection.csv")
            masked_by_cohout = Dict(e.cohout => e.n_masked for e in entries)
            open(selpath, "w") do io
                println(io, "#cohout,candidate_index,n_masked")
                for coh in sel_cohouts
                    for c in selected_by_cohout[coh]
                        @printf(io, "%s,%d,%d\n", coh, c.index,
                                get(masked_by_cohout, coh, 0))
                    end
                end
            end
            rc = run_parallel(a, sel_cohouts, outdir, argv;
                              selection_path = selpath)
            rc !== nothing && return rc
        end
    end
    end  # selection_mode vs normal setup

    # A worker writes its own shard summary; a normal run uses folds.csv.
    summary = get(ENV, "COHERENT_FOLD_SUMMARY", joinpath(outdir, "folds.csv"))
    mkpath(dirname(summary))
    open(summary, "w") do io
        println(io, "cohout,model,candidate_index,candidate_snr,frequency_hz," *
                    "period_s,nharm,fold_snr,snr_method,nbins_initial,nbins," *
                    "boxcar_width,boxcar_start,n_samples,nsubints,par_mask_removed," *
                    "dat,profile,png")
    end

    if !isempty(pars)
        @printf("            par mask: %d known pulsar(s), harmonics 1..%d, mode %s\n",
                length(pars), max_harmonic, mask_mode)
    end

    for (ifi, entry) in enumerate(entries)
        cohout = entry.cohout
        model = entry.model
        n_masked_file = entry.n_masked
        selected = if selection_mode
            entry.keep
        elseif a["dbscan"]
            get(selected_by_cohout, cohout, FoldCandidate[])
        elseif a["top"] == 0
            entry.keep
        else
            entry.keep[1:min(a["top"], length(entry.keep))]
        end
        if isempty(selected)
            # Expected for most files in --dbscan mode; the DBSCAN summary
            # already reports how many candidates were selected.
            nskip += 1
            continue
        end

        datpath = a["dat"] === nothing ?
            infer_dat(cohout, model.stem, root) : abspath(a["dat"])
        if datpath === nothing
            @warn "could not infer the .dat for this .cohout; pass --dat" cohout = cohout
            nskip += 1
            continue
        end
        infpath = replace(datpath, r"\.dat$" => ".inf")
        isfile(infpath) || error("missing .inf beside $datpath")
        N, dt, epoch = parse_inf(infpath)
        bary = inf_value(infpath, "Barycentered?")
        if bary !== nothing && startswith(bary, "0") && !(datpath in warned_bary)
            @warn "input .inf is not barycentered; Earth motion can smear the fold" dat = datpath
            push!(warned_bary, datpath)
        end

        raw = Mmap.mmap(datpath, Vector{Float32})
        length(raw) == N ||
            error("$datpath holds $(length(raw)) floats, $infpath says $N")

        @printf("[%d/%d] %s  (%s)\n", ifi, length(entries),
                basename(cohout), model_string(model))
        @printf("        dat %s  N=%d  dt=%.6g s\n", datpath, N, dt)

        kwargs = model.mode == :ajs ?
            (accel = model.params.accel, jerk = model.params.jerk,
             snap = model.params.snap) :
            (pb = model.params.pb, x = model.params.x,
             A_T = model.params.at, anchor = "midpoint")
        ts, new_epoch = demodulate_series(raw, dt, epoch; kwargs...)
        @printf("        demodulated N=%d  start MJD=%.12f\n", length(ts), new_epoch)

        stem = cohout[1:end-length(".cohout")]
        for cand in selected
            freq = cand.freq
            if a["nbins"] == 0
                # Morello et al.: fold finely, optimise the boxcar window,
                # then downsample by the winning width.
                nbins_initial = 1024
                prof0, sub0, counts0 = fold_series(ts, dt, freq,
                                                   nbins_initial, a["nsubints"])
                boxcar_width, boxcar_start, _ = best_boxcar(prof0)
                prof, subprof, counts = downsample_phase(prof0, sub0, counts0,
                                                         boxcar_width,
                                                         boxcar_start)
                nbins_final = length(prof)
                snr_method = "boxcar"
                # Plot the S/N of each downsampled (boxcar) window, not an
                # unrelated IQR normalisation, so the plotted peak is the
                # reported best-boxcar S/N.
                snrprof = circshift(
                    boxcar_snr_groups(prof0, boxcar_width, boxcar_start),
                    nbins_final ÷ 2)
                foldsnr = maximum(snrprof)
            else
                nbins_initial = a["nbins"]
                boxcar_width = 0
                boxcar_start = 0
                prof, subprof, counts = fold_series(ts, dt, freq,
                                                   nbins_initial, a["nsubints"])
                nbins_final = length(prof)
                snrprof, foldsnr = profile_snr(prof)
                snr_method = "profile"
            end

            tag = @sprintf("%s_cand%03d", basename(stem), cand.index)
            outstem = joinpath(outdir, tag)
            txt = write_profile(outstem * ".txt", cand, model, freq, prof,
                                snrprof, counts, foldsnr, length(ts),
                                nbins_final, a["nsubints"], datpath;
                                n_masked = n_masked_file,
                                nbins_initial = nbins_initial,
                                boxcar_width = boxcar_width,
                                boxcar_start = boxcar_start,
                                snr_method = snr_method)
            png = _HAVE_MAKIE ?
                plot_fold(outstem * ".png", cand, model, freq, prof, snrprof,
                          subprof, foldsnr; snr_method = snr_method,
                          nbins_initial = nbins_initial,
                          boxcar_width = boxcar_width) : nothing
            open(summary, "a") do io
                @printf(io, "%s,%s,%d,%.8g,%.12g,%.12g,%d,%.8g,%s,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s\n",
                        cohout, replace(model_string(model), "," => ";"),
                        cand.index, cand.snr, freq, 1.0 / freq, cand.nharm,
                        foldsnr, snr_method, nbins_initial, nbins_final,
                        boxcar_width, boxcar_start, length(ts), a["nsubints"],
                        n_masked_file, datpath, txt,
                        png === nothing ? "" : png)
            end
            if snr_method == "boxcar"
                @printf("        cand %d: search S/N %.2f  f=%.10f Hz  P=%.6f ms  fold S/N %.2f (boxcar w=%d; %d -> %d bins)  -> %s\n",
                        cand.index, cand.snr, freq, 1000.0 / freq, foldsnr,
                        boxcar_width, nbins_initial, nbins_final, basename(txt))
            else
                @printf("        cand %d: search S/N %.2f  f=%.10f Hz  P=%.6f ms  fold S/N %.2f  -> %s\n",
                        cand.index, cand.snr, freq, 1000.0 / freq, foldsnr,
                        basename(txt))
            end
            nfold += 1
        end
        nfile += 1
        # Explicitly release the big arrays between files so a long run does
        # not accumulate garbage while the next .dat is paged in.
        raw = nothing
        ts = nothing
        GC.gc()
    end

    if !isempty(pars)
        @printf("par mask (%s): removed %d of %d candidate(s) around %d known pulsar(s) (harmonics 1..%d)\n",
                mask_mode, mask_removed, mask_total, length(pars), max_harmonic)
        for (pi, par) in enumerate(pars)
            line = @sprintf("  %-16s F0=%.9f Hz  half-width=%.6g Hz  removed=%d",
                            par.name, par.f0,
                            par_halfwidth(par.f0, abs_tol, a["par-frac-tol"]),
                            mask_counts[pi])
            if pred_by_ajs !== nothing && !isempty(pred_by_ajs[pi])
                line *= @sprintf("  predicted=%.6f..%.6f Hz",
                                 minimum(pred_by_ajs[pi]),
                                 maximum(pred_by_ajs[pi]))
            end
            println(line)
        end
    end
    @printf("fold_demod: folded %d candidate(s) from %d file(s); skipped %d\n",
            nfold, nfile, nskip)
    @printf("            summary %s\n", summary)
    # A shard with no surviving candidates is not a failed worker; use a
    # distinct code so the parent can report it without a scary warning.
    nfold > 0 && return 0
    return shard_total > 1 ? 2 : 1
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(main(ARGS))
