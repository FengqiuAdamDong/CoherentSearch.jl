#!/usr/bin/env julia
#
# Apply the exact accel/jerk/snap time-remap (`resample.jl`, ported from
# FFA_stacking/demodulation/utils.py) to a PRESTO `.dat`/`.inf` pair, writing a
# new `.dat` + `.inf` pair that can be turned into a `.fft` with `realfft`.
#
# Usage:
#   julia --project=<CoherentSearch.jl> demod/demod_dat.jl IN.dat OUT.dat \
#       --accel A --jerk J --snap S [--v0 V0]
#
# IN.inf is read from beside IN.dat and OUT.inf is written beside OUT.dat.
# A/j/s are the injected-sign trial coefficients (positive = receding),
# defined at the anchor epoch = the midpoint of the input series.  With all
# three zero (and v0 = 0) the input is copied through unchanged.

include(joinpath(@__DIR__, "resample.jl"))

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

function main(args)
    isempty(args) && error("usage: demod_dat.jl IN.dat OUT.dat --accel A --jerk J --snap S")
    input = args[1]
    output = args[2]
    infpath = replace(input, r"\.dat$" => ".inf")
    outinf = replace(output, r"\.dat$" => ".inf")
    accel = 0.0; jerk = 0.0; snap = 0.0; v0 = 0.0
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
        else
            error("unknown option $a")
        end
        i += 2
    end

    N, dt, epoch = parse_inf(infpath)
    raw = Vector{Float32}(undef, N)
    open(input, "r") do io
        read!(io, raw)
    end
    length(raw) == N || error("$input holds $(length(raw)) floats, .inf says $N")

    # Zero-mean before resampling.  The remap can leave a few sample holes
    # (uncovered indices), which scatter to zero; on data with a large DC
    # baseline those holes read as huge spikes and their comb swamps the
    # band.  The Python original never sees this because `load_data` calls
    # `TimeSeries.normalise` first -- this is the same de-meaning.
    m = sum(raw) / length(raw)
    raw .-= m

    # Anchor = midpoint of first/last sample, exactly as utils.anchor_mjd
    # (len(ts.data)*tsamp after the first sample, so the midpoint sits at N/2).
    reference_mjd = epoch + 0.5 * N * dt / 86400.0
    tstart_ind = round(Int, (epoch - reference_mjd) * 86400.0 / dt)

    ts_new, tstart_new = resample_ts_shift_snap(
        raw, accel, jerk, snap, dt, tstart_ind; v0 = v0)

    # PRESTO's realfft requires an even sample count; the remap can land on an
    # odd length, so drop the final sample (one bin ~ dt seconds).
    if isodd(length(ts_new))
        ts_new = ts_new[1:end-1]
    end

    new_epoch = reference_mjd + tstart_new * dt / 86400.0
    write(output, ts_new)
    write_inf(outinf, infpath, length(ts_new), new_epoch)
    @printf("demod_dat: %s -> %s  (%d -> %d samples)  a=%g j=%g s=%g\n",
            input, output, N, length(ts_new), accel, jerk, snap)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
