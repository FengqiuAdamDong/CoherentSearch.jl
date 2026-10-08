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

**Sequential demodulation, threaded search.**  The search is the dominant cost
(measured ~2.4 s of a 2.7 s in-process point at `--nharms 16`) and it already
parallelises internally with `@spawn` over chunks, so overlapping demodulations
across grid points would only oversubscribe the threads the search is using.
`-t` controls the whole run.

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
    # One cache for every grid point: the harmonic plans key on `(params,
    # Nprof)` and the trial-grid phase tables on `r_lo`, none of which depends
    # on the file -- but `r_lo = lofreq*T'` DOES move, since each point's remap
    # lands on its own length, so the direct plans are rebuilt per point while
    # the per-thread workspaces are not.
    cache = SearchCache()
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

                a["outputfilenm"] = outfile
                cands = search_one(ft, params, a, cache, backend)
                write_candidates(cands, outfile, a["threshold"])
            end
        end
    finally
        # Returns device memory to the driver; a no-op on the CPU backend.  In a
        # `finally` so an error mid-sweep still releases, as in `main`.
        release_backend!(backend)
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
