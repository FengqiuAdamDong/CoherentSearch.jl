# In-memory demodulation driver: one observation, a whole grid, no intermediates.
#
# Entered from `main` when `--demod-grid` is given (see `cli.jl`).  The whole
# chain runs inside this process, per grid point:
#
#   raw .dat held in RAM  ->  exact time remap  ->  real FFT  ->  rednoise
#                         ->  coherent search    ->  <stem>_demod_..._red.cohout
#
# The point is the file count.  The shell pipeline this replaces
# (`demod/demod_grid.jl` writing a `.dat` + `.inf` per point, then PRESTO's
# `realfft` and `rednoise` on each, then one `coherent_search.jl` process per
# point) writes and re-reads ~5 files per grid point -- ~1100 for a 220-point
# sweep -- and every one of them is a pure intermediate.  Here the observation
# is read once and only the `.cohout` files are written, so peak memory is the
# raw series plus one demodulated series plus one amplitude array, independent
# of how many grid points there are.

"""
    demod_main(a) -> Nothing

Run the in-memory demodulation sweep described by the parsed CLI options `a`.
`a["demod-grid"]` lists the grid CSV(s) from `demod/nsns_grid.py` (one per model;
hybrid mode emits two), and the positional inputs are PRESTO `.dat` time series.

Each input is read **once**; then for every grid point the series is
demodulated, packed into PRESTO's `.fft` layout, dereddened with
[`presto_deredden!`](@ref) and searched with exactly the options the CLI would
use on a real `.fft` file ([`search_one`](@ref) is reused verbatim, so
`--ncands`, `--metricstats`, `measure_ducy` and every search knob behave
identically).  One `<stem>_demod_<grid>_red.cohout` is written per point, into
`--outdir` or beside the input.

**Sequential demodulation, concurrent searches.**  The search is the dominant
cost (~1.7 s of a ~1.75 s in-process point at `--nharms 60`), and a demodulation
is serial at ~25 ms — during which every worker would idle if one search held the
whole thread pool.  Measured: letting one search use all `-t 16` threads takes
14.6 s for 64 points at `--nharms 60`, while searching 16 points at once, each
single-threaded, takes 10.2 s — **1.43x**.  So `--demod-concurrency N` (default
one slot per thread) searches N grid points concurrently, each pinned to
`nthreads()/N` threads.  `N = 1` is the purely sequential driver.

Note what this is *not*: the parallel axis is inside this one process, never one
process per grid point.  Running a process per point was measured at only
1.06–1.13x over the sequential driver, because it re-reads the `.dat` and re-pays
Julia's start-up, plan building and wisdom import per point (~0.85 s against a
~1.8 s point).  The observation is read once here, as intended.

Concurrent searches must not share a `SearchCache`: `Workspace`s are indexed by
task index inside `_search_region!`, so two live searches would have their chunks
writing into the same workspaces.  Each slot in the pool owns one cache, which
also amortises FFTW planning across the sweep.  Candidates are bit-identical to
the sequential run.

`-o`/`--outputfilenm`, `--plot` and `--plotstem` are rejected: each names a
single output that a many-point sweep would clobber, and holding a whole sweep's
amplitudes for deferred plotting would defeat the memory argument above.  Plot
afterwards from the `.cohout` files with `bin/plot_candidates.jl`.
"""
function demod_main(a)
    isempty(a["outputfilenm"]) || throw(ArgumentError(
        "-o/--outputfilenm is not supported with --demod-grid: it names one output " *
        "file, but a sweep writes one .cohout per grid point.  Use --outdir to " *
        "choose the directory instead"))
    a["plot"] && throw(ArgumentError(
        "--plot is not supported with --demod-grid: it would hold every grid " *
        "point's amplitudes and metadata for the deferred plotting pass, which is " *
        "exactly the memory the in-memory chain is avoiding.  Plot afterwards from " *
        "the .cohout files with bin/plot_candidates.jl"))
    isempty(a["plotstem"]) || throw(ArgumentError(
        "--plotstem is not supported with --demod-grid: it names one plot stem, " *
        "but a sweep has as many as it has grid points"))

    dats = copy(a["fftfile"]::Vector{String})
    if !isempty(a["filelist"])
        for line in eachline(a["filelist"])
            path = strip(line)
            isempty(path) || push!(dats, path)
        end
    end
    isempty(dats) && throw(ArgumentError(
        "--demod-grid needs the time series to demodulate: pass positional .dat paths or --filelist FILE"))

    # Every grid CSV, flattened.  The mode travels with each row, since a hybrid
    # sweep passes one CSV per model and they are not interchangeable.
    points = Tuple{Symbol,Tuple{String,String,String}}[]
    for csv in a["demod-grid"]::Vector{String}
        mode, rows = read_demod_grid(csv)
        @info "Demodulation grid" csv=csv mode=mode npoints=length(rows)
        for row in rows
            push!(points, (mode, row))
        end
    end
    isempty(points) && throw(ArgumentError(
        "no demodulation points in $(join(a["demod-grid"], ", "))"))

    outdir = a["outdir"]
    isempty(outdir) && (outdir = dirname(abspath(dats[1])))
    isempty(a["outdir"]) || mkpath(outdir)

    nharms = a["nharms"]
    params = SearchParams(
        nharms = nharms,
        m = a["m"],
        hidr = a["hidr"],
        threshold = a["threshold"],
        decimations = decimation_set(nharms, a["maxdecim"]),
        precision = Symbol(a["precision"]),
        sigma = Symbol(a["sigma"]),
    )
    backend = resolve_backend(a)
    # A POOL of caches, one per concurrent slot, plus the sequential one.
    #
    # Concurrent searches must NOT share a `SearchCache`: its `Workspace`s are
    # indexed by task index inside `_search_region!`, so two searches running at
    # once would have their `@spawn`ed chunks writing into the same workspaces --
    # a silent data race, not a slow path. One cache per slot fixes that AND
    # amortises FFTW planning: the plans are built `nconc` times rather than once
    # per grid point, which matters because `cache = nothing` re-plans every point.
    nconc = _demod_concurrency(a, backend)
    # Slot 1 is `cache` (the sequential one, reused when nconc == 1); the rest are
    # extra slots. `Channel` is how `_demod_submit!` blocks when all are busy.
    cache = SearchCache()
    caches = Channel{SearchCache}(nconc)
    put!(caches, cache)
    for _ in 2:nconc
        put!(caches, SearchCache())
    end
    pending = Tuple{Task,AbstractString,Float64}[]
    plans = Dict{Int,Any}()          # FFTW rfft plans, keyed by the demodulated N

    npoints = length(points) * length(dats)
    done = 0
    try
        for dat in dats
            infpath = replace(dat, r"\.dat$" => ".inf")
            inf = SimpleInf(infpath)
            inf.N === nothing && error("Missing 'Number of bins' in $infpath")
            inf.dt === nothing && error("Missing 'Width of each time series bin' in $infpath")
            inf.epoch === nothing && error("Missing 'Epoch of observation (MJD)' in $infpath")
            N = inf.N
            raw = Vector{Float32}(undef, N)
            open(dat, "r") do io
                read!(io, raw)
            end
            length(raw) == N || error("$dat holds $(length(raw)) floats, .inf says $N")
            stem_base = first(splitext(Base.basename(dat)))

            for (mode, row) in points
                done += 1
                series, epoch = demodulate_series(raw, inf.dt, inf.epoch;
                                                  _grid_kwargs(mode, row)...)
                amps = presto_fft_amps(series, plans)
                stem = demod_point_stem(stem_base, mode, row)
                outfile = joinpath(outdir, stem * "_red.cohout")
                nwrote = presto_deredden!(amps, length(series) * inf.dt;
                                          startwidth = a["rednoise-startwidth"],
                                          endwidth = a["rednoise-endwidth"],
                                          endfreq = a["rednoise-endfreq"])
                if nwrote < length(amps)
                    @warn "Dereddening stopped early; the tail is unnormalised" file=dat stem=stem written=nwrote nbins=length(amps)
                end
                # The demodulated series has its own length and start epoch, so
                # they replace the input's in the metadata the search reads.  Its
                # `dt` is unchanged -- the remap shifts samples, it does not
                # resample them -- so T' = N'*dt and everything scaled by T
                # (`r_lo = lofreq*T`, Nyquist) is per-point, as it must be.
                Np = length(series)
                demin = SimpleInf(infpath, inf.object, epoch, Np, inf.dt, inf.DM)
                ft = FFTFile(stem, amps, demin, Np, Np * inf.dt, 1.0 / (Np * inf.dt),
                             true, true, real(amps[1]), imag(amps[1]))
                gridlabel = mode === :ajs ? "a=$(row[1]) j=$(row[2]) s=$(row[3])" :
                                            "pb=$(row[1]) x=$(row[2]) at=$(row[3])"
                @info "Demodulated point" point="$done/$npoints" mode=mode grid=gridlabel N=Np T=ft.T file=outfile

                # The search is the dominant cost by ~2 orders of magnitude, so it
                # runs on a task.  With the default concurrency of 1 this is the
                # same call the sequential driver made; with more, several points
                # search at once, each pinned to `maxthreads` threads.
                if nconc == 1
                    cands = search_one(ft, params, a, cache, backend;
                                       maxthreads = 0, outfile = outfile)
                    write_candidates(cands, outfile, a["threshold"])
                else
                    _demod_submit!(pending, caches, ft, params, a, backend, outfile,
                                   ft.T, _demod_maxthreads(nconc), nconc)
                end
            end
        end
        # Drain. Any error inside a task is rethrown here, so a failure is not
        # swallowed -- `wait` on a failed task raises rather than returning.
        _demod_drain!(pending)
    finally
        # Returns device memory to the driver; a no-op on the CPU backend.  In a
        # `finally` so an error mid-sweep still releases, as in `main`.
        release_backend!(backend)
    end
    return nothing
end

"""
    _demod_concurrency(a, backend) -> Int

How many grid points to search at once, and how many workspaces to build for it.

The whole point of the in-memory driver is that the observation is read **once**;
the same argument applies to the work, so the pipeline is parallelised *inside*
this process rather than by running one process per grid point (which would
re-read the `.dat` and re-pay Julia's start-up per point).  Concurrency beats
intra-point threading because a demodulation is serial -- ~25 ms during which
every worker would otherwise idle -- and because the search's chunk parallelism
flattens well below `nthreads()`.

Default: `nthreads()`, capped so that each search still gets at least one thread
and there are never more slots than points.  `--demod-concurrency 1` restores the
purely sequential driver.
"""
function _demod_concurrency(a, backend)
    nconc = Int(a["demod-concurrency"])
    nconc == 0 && (nconc = nthreads())
    nconc = max(1, min(nconc, nthreads()))
    backend isa CPUBackend || (nconc = 1)   # the device is one resource
    return nconc
end

# Threads each concurrent search may use.  With `nconc` slots and `nthreads()`
# threads there is no fixed division that is always right, so the caller may pin
# it; the default divides the pool, giving a single-threaded search per slot when
# the two are equal -- which is the configuration that scales linearly.
_demod_maxthreads(nconc) = nconc <= 1 ? 0 : max(1, nthreads() ÷ nconc)

"""
    _demod_submit!(pending, caches, ft, params, a, backend, outfile, T, maxthreads, nconc)

Take a cache from the pool and run one grid point's search on a task, blocking
while the pool is empty so that at most `nconc` searches are in flight.  The
cache is returned to the pool in a `finally`, so a failed point does not leak its
slot and stall the sweep.

`pending` is held to `nconc` entries by draining a *completed* task whenever it
reaches that size.  Without this the list would grow to one entry per grid point,
keeping every point's `FTTFile` — and therefore its amplitude array, `N` complex
words — alive until the end of the sweep, which is exactly the "peak memory
independent of the grid size" property this driver exists for.  A 792-point
sweep would pin ~0.5 GB that the sequential driver frees as it goes.
"""
function _demod_submit!(pending, caches, ft, params, a, backend, outfile, T,
                        maxthreads, nconc)
    # Drain first if the window is full: `take!` below cannot block on a slot
    # whose holder is only waiting in this list, so the two must be kept in step.
    length(pending) >= nconc && _demod_drain_one!(pending)
    c = take!(caches)
    t = Threads.@spawn begin
        try
            search_one(ft, params, a, c, backend; maxthreads = maxthreads, outfile = outfile)
        finally
            put!(caches, c)
        end
    end
    push!(pending, (t, outfile, T))
    return nothing
end

"""
    _demod_drain_one!(pending)

Collect the oldest in-flight grid point: wait for it, write its `.cohout`, and
drop it from `pending`.  `fetch` rather than `wait`: `wait(t::Task)` returns
`nothing` regardless of the task's value (it only raises on failure), so it would
hand `write_candidates` a `Nothing`.  Both raise on a failed task, so a broken
point aborts the sweep instead of silently producing no `.cohout`.

Waiting oldest-first keeps the window at `nconc` in-flight points and lets the
oldest `.cohout` land as soon as it is ready.  It can block on the oldest task
while newer ones have already finished, but with every task bounded by one
search that costs at most one search's tail.
"""
function _demod_drain_one!(pending)
    t, outfile, T = popfirst!(pending)
    write_candidates(fetch(t), outfile, T)
    return nothing
end

"""
    _demod_drain!(pending)

Wait for every remaining in-flight grid point and write each one's candidates.
Called once at the end of the sweep, after the per-point bounded drain above.
"""
function _demod_drain!(pending)
    while !isempty(pending)
        _demod_drain_one!(pending)
    end
    return nothing
end

# One grid row's remap parameters.  The row is the literal CSV text, so the
# conversion happens here and the filename still matches the CSV.
function _grid_kwargs(mode::Symbol, row)
    if mode == :ajs
        return (accel = parse(Float64, row[1]), jerk = parse(Float64, row[2]),
                snap = parse(Float64, row[3]))
    else
        return (pb = parse(Float64, row[1]), x = parse(Float64, row[2]),
                A_T = parse(Float64, row[3]))
    end
end
