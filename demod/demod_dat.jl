#!/usr/bin/env julia
#
# Apply the exact accel/jerk/snap time-remap (`CoherentSearch.demodulate_series`,
# ported from FFA_stacking/demodulation/utils.py) to a PRESTO `.dat`/`.inf` pair,
# writing a new `.dat` + `.inf` pair that can be turned into a `.fft` with
# `realfft`.
#
# Usage:
#   julia --project=<CoherentSearch.jl> demod/demod_dat.jl IN.dat OUT.dat \
#       --accel A --jerk J --snap S [--v0 V0]
#
# IN.inf is read from beside IN.dat and OUT.inf is written beside OUT.dat.
# A/j/s are the injected-sign trial coefficients (positive = receding),
# defined at the anchor epoch = the midpoint of the input series.  With all
# three zero (and v0 = 0) the input is copied through unchanged.

using CoherentSearch
using Printf: @sprintf, @printf

function parse_inf(path)
    vals = Dict{String,String}()
    for line in eachline(path)
        occursin("=", line) || continue
        k, v = split(line, "=", limit = 2)
        vals[strip(k)] = strip(v)
    end
    N = parse(Int, vals["Number of bins in the time series"])
    dt = parse(Float64, vals["Width of each time series bin (sec)"])
    epoch = parse(Float64, vals["Epoch of observation (MJD)"])
    return N, dt, epoch
end

# Rewrite the .inf for the resampled series: new length and start epoch.  The
# remap invalidates the on/off break sample ranges, so any break pairs are
# dropped and the breaks flag forced to 0.  Key order and all other values are
# preserved from the input.
function write_inf(outpath, infpath, N, epoch)
    open(outpath, "w") do io
        for line in readlines(infpath)
            if !occursin("=", line)
                print(io, line, "\n")
                continue
            end
            k, _ = split(line, "=", limit = 2)
            key = strip(k)
            if key == "Number of bins in the time series"
                @printf(io, " %-38s =  %-11d\n", key, N)
            elseif key == "Epoch of observation (MJD)"
                @printf(io, " %-38s =  %05.15f\n", key, epoch)
            elseif key == "Any breaks in the data? (1 yes, 0 no)"
                @printf(io, " %-38s =  0\n", key)
            elseif occursin(r"^On/Off bin pair", key)
                # dropped: sample ranges no longer valid after the remap
            elseif startswith(key, "Data file name")
                @printf(io, " %-38s =  %s\n", key,
                        splitext(basename(outpath))[1])
            else
                print(io, line, "\n")
            end
        end
    end
    return outpath
end

"""
    demod_file(input, output; accel, jerk, snap, v0=0.0,
               pb=nothing, x=nothing, A_T=0.0, anchor="start",
               sin_i=nothing, companion_mass=1.4, pulsar_mass=1.4) -> (n_in, n_out)

Apply the remap to `input` and write `output` (+ beside it, an updated `.inf`).
Reads `input`'s `.inf` for N/dt/epoch.

Two mutually exclusive models:
  * accel/jerk/snap/v0 -- exact polynomial remap (the default);
  * circular orbit -- pass `pb` [days] and either `x` [light-seconds] or
    `sin_i` (+ `companion_mass`/`pulsar_mass` to derive x), with `A_T` [rad]
    the mean anomaly at `anchor` ("start" = the .inf epoch, pulsegen's
    default; "midpoint" = obs midpoint).
"""
function demod_file(input, output; accel = 0.0, jerk = 0.0, snap = 0.0, v0 = 0.0,
                    pb = nothing, x = nothing, A_T = 0.0, anchor = "start",
                    sin_i = nothing, companion_mass = 1.4, pulsar_mass = 1.4)
    infpath = replace(input, r"\.dat$" => ".inf")
    outinf = replace(output, r"\.dat$" => ".inf")
    N, dt, epoch = parse_inf(infpath)
    raw = Vector{Float32}(undef, N)
    open(input, "r") do io
        read!(io, raw)
    end
    length(raw) == N || error("$input holds $(length(raw)) floats, .inf says $N")

    ts_new, new_epoch = demodulate_series(raw, dt, epoch;
                                         accel = accel, jerk = jerk, snap = snap,
                                         v0 = v0, pb = pb, x = x, A_T = A_T,
                                         anchor = anchor, sin_i = sin_i,
                                         companion_mass = companion_mass,
                                         pulsar_mass = pulsar_mass)
    write(output, ts_new)
    write_inf(outinf, infpath, length(ts_new), new_epoch)
    return N, length(ts_new)
end

function demod_main(args)
    isempty(args) && error("usage: demod_dat.jl IN.dat OUT.dat " *
                           "--accel A --jerk J --snap S  |  --pb P --x X --a-t A")
    input = args[1]
    output = args[2]
    accel = 0.0; jerk = 0.0; snap = 0.0; v0 = 0.0
    pb = nothing; x = nothing; A_T = 0.0; anchor = "start"
    sin_i = nothing; companion_mass = 1.4; pulsar_mass = 1.4
    i = 3
    while i <= length(args)
        a = args[i]
        val = i < length(args) ? parse(Float64, args[i+1]) : error("$a needs a value")
        if a == "--accel" || a == "-a"
            accel = val
        elseif a == "--jerk" || a == "-j"
            jerk = val
        elseif a == "--snap" || a == "-s"
            snap = val
        elseif a == "--v0"
            v0 = val
        elseif a == "--pb"
            pb = val
        elseif a == "--x"
            x = val
        elseif a == "--a-t"
            A_T = val
        elseif a == "--sini"
            sin_i = val
        elseif a == "--companion-mass"
            companion_mass = val
        elseif a == "--pulsar-mass"
            pulsar_mass = val
        elseif a == "--a-t-anchor"
            anchor = args[i+1]
        else
            error("unknown option $a")
        end
        i += 2
    end

    N, nout = demod_file(input, output; accel = accel, jerk = jerk, snap = snap, v0 = v0,
                         pb = pb, x = x, A_T = A_T, anchor = anchor,
                         sin_i = sin_i, companion_mass = companion_mass,
                         pulsar_mass = pulsar_mass)
    if pb === nothing
        @printf("demod_dat: %s -> %s  (%d -> %d samples)  a=%g j=%g s=%g\n",
                input, output, N, nout, accel, jerk, snap)
    else
        @printf("demod_dat: %s -> %s  (%d -> %d samples)  pb=%g d x=%s A_T=%g (%s)\n",
                input, output, N, nout, pb,
                x === nothing ? "derived" : string(x), A_T, anchor)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && demod_main(ARGS)
