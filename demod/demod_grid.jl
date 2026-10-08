#!/usr/bin/env julia
#
# Batch-demodulate one observation at every point in one or more grid CSVs, in
# ONE Julia process (so the resampler is compiled once, not per point).
#
# Usage:
#   julia --project=<CoherentSearch.jl> demod/demod_grid.jl IN.dat OUTDIR \
#       GRID1.csv [GRID2.csv ...] [--basename NAME]
#
# Any number of grid CSVs may be given (hybrid mode emits one per model); each
# is read independently and its own header selects the model:
#   accel,jerk,snap -> polynomial remap; each row writes
#     OUTDIR/NAME_demod_a{A}_j{J}_s{S}.dat + .inf
#   pb,x,at         -> circular orbit; x is a_p sin(i)/c [light-s], at is A_T
#     [rad] at the .inf epoch; each row writes
#     OUTDIR/NAME_demod_pb{Pb}_x{X}_at{A_T}.dat + .inf
#
# Literal CSV strings are used for filenames (not reformatted numbers), so the
# caller can reconstruct the filename from the grid text.  --basename defaults
# to the input file's stem.  The output .dat files are what realfft consumes.
#
# The grid reader and the filename rule live in the package
# (`CoherentSearch.read_demod_grid` / `demod_point_stem`), shared with the
# in-memory driver `bin/coherent_search.jl --demod-grid`: one definition, so a
# filename built here and one built there cannot drift apart.  `demod_file`
# itself stays in `demod_dat.jl` -- it is the on-disk wrapper the package does
# not need.
include(joinpath(@__DIR__, "demod_dat.jl"))

function main(args)
    length(args) >= 3 || error(
        "usage: demod_grid.jl IN.dat OUTDIR GRID1.csv [GRID2.csv ...] [--basename NAME]")
    input = args[1]
    outdir = args[2]
    stem = splitext(Base.basename(input))[1]
    grid_paths = String[]
    i = 3
    while i <= length(args)
        if args[i] == "--basename"
            i < length(args) || error("--basename needs a value")
            stem = args[i+1]
            i += 2
        else
            push!(grid_paths, args[i])
            i += 1
        end
    end
    isempty(grid_paths) && error("demod_grid.jl: no grid CSV given")

    total = 0
    for grid_path in grid_paths
        mode, rows = read_demod_grid(grid_path)
        println("demod_grid: $(length(rows)) $(mode) point(s) from $grid_path")
        for row in rows
            out = joinpath(outdir, demod_point_stem(stem, mode, row) * ".dat")
            if mode == :ajs
                demod_file(input, out; accel = parse(Float64, row[1]),
                           jerk = parse(Float64, row[2]), snap = parse(Float64, row[3]))
            else
                demod_file(input, out; pb = parse(Float64, row[1]),
                           x = parse(Float64, row[2]), A_T = parse(Float64, row[3]))
            end
        end
        total += length(rows)
    end
    println("demod_grid: wrote $total demodulated series under $outdir")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
