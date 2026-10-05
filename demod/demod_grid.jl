#!/usr/bin/env julia
#
# Batch-demodulate one observation at every (accel, jerk, snap) point in a grid
# CSV, in ONE Julia process (so the resampler is compiled once, not per point).
#
# Usage:
#   julia --project=<CoherentSearch.jl> demod/demod_grid.jl IN.dat GRID.csv OUTDIR \
#       [--basename NAME]
#
# Each grid row i (1-based, matching the CSV header order accel,jerk,snap)
# writes OUTDIR/points/p%05d/NAME_demod.dat plus its .inf.  --basename defaults
# to the input file's stem.  The output .dat files are what realfft consumes.
include(joinpath(@__DIR__, "demod_dat.jl"))

function read_grid(csv_path)
    rows = NamedTuple{(:accel, :jerk, :snap),NTuple{3,Float64}}[]
    header = true
    for line in eachline(csv_path)
        if header
            header = false
            continue
        end
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) == 3 || error("$csv_path: expected 3 columns, got $(length(parts))")
        push!(rows, (parse(Float64, parts[1]), parse(Float64, parts[2]),
                     parse(Float64, parts[3])))
    end
    return rows
end

function main(args)
    length(args) >= 3 || error(
        "usage: demod_grid.jl IN.dat GRID.csv OUTDIR [--basename NAME]")
    input = args[1]
    grid_path = args[2]
    outdir = args[3]
    stem = splitext(Base.basename(input))[1]
    i = 4
    while i <= length(args)
        if args[i] == "--basename"
            i < length(args) || error("--basename needs a value")
            stem = args[i+1]
            i += 2
        else
            error("unknown option $(args[i])")
        end
    end

    rows = read_grid(grid_path)
    println("demod_grid: $(length(rows)) points from $grid_path")
    for (idx, r) in enumerate(rows)
        pdir = joinpath(outdir, "points", @sprintf("p%05d", idx))
        mkpath(pdir)
        out = joinpath(pdir, string(stem, "_demod.dat"))
        demod_file(input, out; accel = r.accel, jerk = r.jerk, snap = r.snap)
    end
    println("demod_grid: wrote $(length(rows)) demodulated series under ",
            joinpath(outdir, "points"))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
