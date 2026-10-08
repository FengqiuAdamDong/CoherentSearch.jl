using Test
using CoherentSearch
using CoherentSearch: _lower_median!
using FFTW: FFTW
using Random
using Logging: with_logger, NullLogger

# ---------------------------------------------------------------------------
# External tools, probed once.  The PRESTO comparisons skip (with an @info, the
# test_fileio.jl / test_gpu.jl convention) when a tool is absent, so Pkg.test()
# stays green on a machine without PRESTO installed.
# ---------------------------------------------------------------------------
const REALFFT = get(ENV, "REALFFT", "/home/fadong/anaconda3/bin/realfft")
const REDNOISE = get(ENV, "REDNOISE", "/home/fadong/anaconda3/bin/rednoise")
const PYTHON_ = get(ENV, "COHERENT_PYTHON", "/home/fadong/anaconda3/bin/python")
# Probed by existence only: PRESTO's tools call `usage()` and exit non-zero with
# no arguments, so `success(run(...))` would report every working tool absent.
const HAS_REALFFT = isfile(REALFFT) && isexecutable(REALFFT)
const HAS_REDNOISE = isfile(REDNOISE) && isexecutable(REDNOISE)
const PULSEGEN = joinpath(@__DIR__, "..", "demod", "pulsegen_gr.py")
const HAS_PULSEGEN = isfile(PULSEGEN) && isfile(PYTHON_) &&
    success(pipeline(`$PYTHON_ $PULSEGEN -h`; stdout=devnull, stderr=devnull))

# `demod/demod_dat.jl` is the on-disk wrapper the in-memory path replaces; the
# testset below pins the two together, so it has to be loadable here.  Its
# `abspath(PROGRAM_FILE) == abspath(@__FILE__)` guard keeps its CLI out.
include(joinpath(@__DIR__, "..", "demod", "demod_dat.jl"))

# PRESTO's `rednoise`/`realfft` read a full `.inf` through `readinf`, which is
# strict about the field ORDER, not just their presence: it reads name,
# telescope, instrument, object, RA, DEC, observer, MJD, bary, N, dt, numonoff,
# then -- for a Radio band -- fov, dm, freq, freqband, num_chan, chan_wid,
# stopping with "end-of-file while looking for ..." the moment one is missing.
# "Beam diameter (arcsec)" is the fov line and is easy to leave out; without it
# every PRESTO comparison silently skips.  A minimal header, which looks
# perfectly reasonable, instead fails with "can not convert '<x>' to RA or DEC".
# `Data file name without suffix` must also name the file's own base, since
# `rednoise` uses it to find its input.  All three were measured the hard way.
function write_presto_inf(path, N, dt, epoch; name = first(splitext(basename(path))))
    write(path,
          " Data file name without suffix          =  $name\n" *
          " Telescope used                         =  GBT\n" *
          " Instrument used                        =  unset\n" *
          " Object being observed                  =  SYNTH\n" *
          " J2000 Right Ascension (hh:mm:ss.ssss)  =  17:45:39.9600\n" *
          " J2000 Declination     (dd:mm:ss.ssss)  =  -29:00:28.0800\n" *
          " Data observed by                       =  unset\n" *
          " Epoch of observation (MJD)             =  $(epoch)\n" *
          " Barycentered?           (1 yes, 0 no)  =  1\n" *
          " Number of bins in the time series      =  $N\n" *
          " Width of each time series bin (sec)    =  $dt\n" *
          " Any breaks in the data? (1 yes, 0 no)  =  0\n" *
          " Type of observation (EM band)          =  Radio\n" *
          " Beam diameter (arcsec)                 =  401\n" *
          " Dispersion measure (cm-3 pc)           =  0.0\n" *
          " Central freq of low channel (MHz)      =  1450.390625\n" *
          " Total bandwidth (MHz)                  =  800\n" *
          " Number of channels                     =  1024\n" *
          " Channel bandwidth (MHz)                =  0.78125\n" *
          " Data analyzed by                       =  unset\n")
    return path
end

# A synthetic time series with a Gaussian pulse train at a known frequency, so a
# test can assert something is recovered (and not merely written).
function synth_series(N; f = 1.0, dt = 0.0098304, snr = 30.0, width = 0.01, seed = 7)
    rng = MersenneTwister(seed)
    x = randn(rng, Float32, N) .* 1.0f0
    amp = snr * sqrt(N / 2)
    for i in 1:N
        t = (i - 1) * dt
        ph = mod(t * f, 1.0)
        d = min(ph, 1 - ph) * (1 / f)          # seconds from the nearest pulse peak
        x[i] += amp * exp(-0.5 * (d / width)^2)
    end
    return x
end

@testset "presto_fft_amps: PRESTO .fft layout" begin
    N = 512
    x = randn(MersenneTwister(3), Float32, N)
    plans = Dict{Int,Any}()
    a = presto_fft_amps(x, plans)

    @test length(a) == N ÷ 2
    @test length(plans) == 1                     # the plan is cached by length
    @test presto_fft_amps(x, plans) == a          # and reused, not rebuilt

    # DC and Nyquist packed into bin 1, computed longhand.  `≈` rather than `==`:
    # the longhand sum accumulates in a different order than the transform's.
    @test real(a[1]) ≈ sum(x)
    @test imag(a[1]) ≈ sum(Float32((-1)^(i + 1)) * x[i] for i in 1:N)
    @test imag(a[1]) == real(FFTW.rfft(x)[N ÷ 2 + 1])    # the transform's own Nyquist
    # The rest are the positive-frequency amplitudes verbatim.
    @test a[2:end] == FFTW.rfft(x)[2:(N ÷ 2)]

    # `realfft()` requires an even sample count and errors otherwise.
    @test_throws ErrorException presto_fft_amps(randn(Float32, 511), Dict{Int,Any}())
end

@testset "presto_invsqrt: the Quake reciprocal, not 1/sqrt" begin
    # Bit-exact against the shipped `libpresto.so:invsqrtf` (checked directly:
    # these three values and their bits), because `dered_engine` uses this and a
    # "tidier" 1/sqrt would silently rescale every S/N by ~1.001.
    @test reinterpret(UInt32, presto_invsqrt(1.0f0)) == 0x3f7f910f
    @test reinterpret(UInt32, presto_invsqrt(4.0f0)) == 0x3eff910f
    @test reinterpret(UInt32, presto_invsqrt(2.0f0)) == 0x3f34f95e
    @test presto_invsqrt(1.0f0) !== 1.0f0        # the classic 0.1% error
    # Within the estimate's own ~0.1%, over the range the engine uses.
    rng = MersenneTwister(11)
    worst = maximum(abs(presto_invsqrt(v) * sqrt(v) - 1) for v in rand(rng, Float32, 1000) .+ 0.01f0)
    @test worst < 5e-3
    # The lower order statistic, which is what the shipped `median` returns --
    # the mean of the two central values is 7% high on a log-distributed power
    # spectrum and would silently inflate every S/N.
    buf = Float32[6, 2, 4, 5, 1, 3]                # n = 6: lower is 3, not 3.5
    @test _lower_median!(copy(buf), 6) == 3.0f0
    @test _lower_median!(copy(Float32[1, 2, 3]), 3) == 2.0f0
    @test _lower_median!(copy(Float32[8, 7, 6, 5, 4, 3, 2, 1]), 8) == 4.0f0
end

@testset "presto_deredden!: properties" begin
    rng = MersenneTwister(5)
    H = 76817
    a = ComplexF32.(randn(rng, H), randn(rng, H))
    n = presto_deredden!(a, H * 2 * 0.0098304)
    @test n == H                                 # every bin written, DC included
    @test a[1] == ComplexF32(1, 0)               # DC pinned
    # The block medians are taken over power, then divided by sqrt of
    # median/log(2) -- so a unit-variance spectrum comes out with mean power
    # near 1.
    @test 0.9 < sum(abs2, a[2:end]) / (H - 1) < 1.1
    @test all(isfinite, a)
    # A short array still normalises rather than erroring.
    @test presto_deredden!(ComplexF32[1, 2, 3], 100.0) > 0
end

@testset "presto_deredden! vs PRESTO's rednoise" begin
    if !(HAS_REALFFT && HAS_REDNOISE)
        @info "Skipping the PRESTO rednoise comparison; binaries not found" REALFFT REDNOISE
    else
        N = 153600
        dt = 0.0098304
        x = randn(MersenneTwister(1234), Float32, N)
        a = presto_fft_amps(x, Dict{Int,Any}())
        H = N ÷ 2
        dir = mktempdir()
        name = "rn"
        write(joinpath(dir, "$name.fft"), a)
        write_presto_inf(joinpath(dir, "$name.inf"), N, dt, 54365.98291064203; name)
        presto_deredden!(a, N * dt)
        run(pipeline(Cmd(`$REDNOISE $name.fft`; dir); stdout=devnull, stderr=devnull))
        red = Vector{ComplexF32}(undef, H)
        open(joinpath(dir, name * "_red.fft"), "r") do io
            read!(io, red)
        end
        maxrel = maximum(abs.(a .- red) ./ abs.(red))
        # Measured 2.5e-7 for this port.  The tolerance is loose enough for
        # another host's libm in the medians and tight enough that the even-n
        # median convention (7% off, see `_lower_median!`) fails loudly -- which
        # is the whole reason this test exists.
        @test maxrel < 1e-5
    end
end

@testset "demodulate_series matches demod_dat.jl" begin
    # `demod/demod_dat.jl` is a thin wrapper over `demodulate_series` now, so
    # this pins that the wrapper's output IS the returned vector -- bit for bit,
    # which is the property the refactor could break.
    N = 1 << 14
    dt = 0.0098304
    epoch = 54365.98291064203
    x = synth_series(N; snr = 5.0, seed = 21)
    dir = mktempdir()
    dat = joinpath(dir, "d.dat")
    write(dat, x)
    write_presto_inf(joinpath(dir, "d.inf"), N, dt, epoch)

    cases = [(accel = -338.7, jerk = 0.2512, snap = 9.41e-4),
             (accel = 0.0, jerk = 0.0, snap = 0.0),
             (pb = 0.041666666666666664, x = 0.41361909, A_T = 0.7)]
    demod_dat = joinpath(@__DIR__, "..", "demod", "demod_dat.jl")
    for (i, kw) in enumerate(cases)
        out = joinpath(dir, "out$i.dat")
        demod_file(dat, out; kw...)
        written = Vector{Float32}(undef, filesize(out) ÷ 4)
        open(out, "r") do io
            read!(io, written)
        end
        series, _ = demodulate_series(x, dt, epoch; kw...)
        @test written == series
        @test iseven(length(series))             # realfft's requirement
    end
    # `demod_file` writes the `.inf` beside the `.dat`, with the new length.
    @test isfile(joinpath(dir, "out1.inf"))
    @test occursin("Number of bins in the time series",
                   read(joinpath(dir, "out3.inf"), String))
end

@testset "demodulate_series: anchor convention" begin
    # The anchor is load-bearing (a wrong one moved a recovered S/N from 40.8 to
    # 8.0).  `anchor = "start"` shifts the circular A_T from the observation
    # midpoint back to the .inf epoch by T/2, so the two anchors must give
    # genuinely different series, and "start" must recover a signal injected
    # with pulsegen's default anchor.
    N = 1 << 14
    dt = 0.0098304
    epoch = 5.5e4
    x = synth_series(N; f = 1.0, snr = 20.0, seed = 31)
    common = (pb = 0.041666666666666664, x = 0.41361909, A_T = 0.7)
    s_start, e_start = demodulate_series(x, dt, epoch; common..., anchor = "start")
    s_mid, e_mid = demodulate_series(x, dt, epoch; common..., anchor = "midpoint")
    @test s_start != s_mid
    # The new epoch follows the resampled start index, so it moves too.
    @test e_start !== epoch || e_mid !== epoch
    # And `sin_i` derives x through `projected_x`, matching the explicit value
    # pulsegen/nsns_grid use for this orbit.
    @test isapprox(projected_x(3600.0, 0.5, 1.4, 1.4), 0.4136190886666235; rtol = 1e-12)
    s_sini, _ = demodulate_series(x, dt, epoch;
                                  pb = common.pb, sin_i = 0.5, A_T = common.A_T)
    @test s_sini == s_start
end

@testset "read_demod_grid / demod_point_stem" begin
    dir = mktempdir()
    ajs = joinpath(dir, "ajs.csv")
    write(ajs, "accel,jerk,snap\n-338.7,0.2512,9.41e-4\n1,2,3\n")
    circ = joinpath(dir, "circ.csv")
    write(circ, "pb,x,at\n0.041666666666666664,0.41361909,0.7\n")
    @test read_demod_grid(ajs) == (:ajs, [("-338.7", "0.2512", "9.41e-4"), ("1", "2", "3")])
    @test read_demod_grid(circ) == (:circular, [("0.041666666666666664", "0.41361909", "0.7")])
    # Filenames come from the LITERAL CSV text, which is what lets the caller
    # reconstruct them from the grid file, and what `combine_cohout.py` matches.
    @test demod_point_stem("inj", :ajs, ("1", "2", "3")) == "inj_demod_a1_j2_s3"
    @test demod_point_stem("inj", :circular, ("0.5", "0.4", "0.7")) == "inj_demod_pb0.5_x0.4_at0.7"
    @test_throws SystemError read_demod_grid(joinpath(dir, "missing.csv"))
    bad = joinpath(dir, "bad.csv")
    write(bad, "alpha,beta\n1,2\n")
    @test_throws ErrorException read_demod_grid(bad)
end

@testset "in-memory chain recovers an injected pulsar" begin
    if !HAS_PULSEGEN
        @info "Skipping the inject/demod/search end-to-end test; pulsegen_gr.py or its Python not found" PULSEGEN
    else
        N = 1 << 15
        dt = 0.0098304
        epoch = 5.5e4
        dir = mktempdir()
        inj_inf = joinpath(dir, "inj.inf")
        write_presto_inf(inj_inf, N, dt, epoch; name = "inj")
        pb, x_lt, A_T = 0.041666666666666664, 0.5, 0.7
        # No `-real`: pulsegen then generates its own Gaussian noise background,
        # so the test needs no fixture -- the repo's .dat/.inf pairs are
        # gitignored and are not on every host.
        run(pipeline(`$PYTHON_ $PULSEGEN -inf $inj_inf -outdir $dir -outbasename inj
                      -pb $pb -e 0 -sin_i $x_lt -omega_peri 0 -A_T $A_T
                      -shapiro_r 0 -shapiro_s 0 -p0 1.0 -pdot 0 -snr 30
                      -pulse_width 0.01 -seed 7`;
                     stdout=devnull, stderr=devnull))

        raw = Vector{Float32}(undef, N)
        open(joinpath(dir, "inj.dat"), "r") do io
            read!(io, raw)
        end
        # Read the truth pulsegen actually used (a tiny scalar reader; no YAML
        # dependency for a flat file of `key: value`).
        truth = Dict{String,Float64}()
        for line in eachline(joinpath(dir, "inj_truth.yaml"))
            occursin(":", line) || continue
            k, v = split(line, ":", limit = 2)
            val = tryparse(Float64, strip(v))
            val === nothing || (truth[strip(k)] = val)
        end
        @test haskey(truth, "x_lt_s")

        series, new_epoch = demodulate_series(raw, dt, epoch;
                                              pb = truth["pb_days"], x = truth["x_lt_s"],
                                              A_T = truth["A_T_rad"])
        amps = presto_fft_amps(series, Dict{Int,Any}())
        presto_deredden!(amps, length(series) * dt)
        Np = length(series)
        inf = SimpleInf(inj_inf, "INJ", new_epoch, Np, dt, 0.0)
        ft = FFTFile("inj_demod", amps, inf, Np, Np * dt, 1.0 / (Np * dt),
                     true, true, real(amps[1]), imag(amps[1]))
        cands = search(ft, SearchParams(nharms = 16, threshold = 0.0);
                       lofreq = 0.5, hifreq = 2.0, blocksize = 512,
                       threshold = 0.0, progress = :none, wisdom = false, backend = CPUBackend())
        @test !isempty(cands)
        best = argmax(c -> c.metric, cands)
        df = 1.0 / ft.T
        @test abs(best.freq - 1.0) <= 3 * df
        @test best.metric >= 15

        # Second arm: the same demodulated series through PRESTO on disk must
        # give the same candidate.  This is the regression pin for "the whole
        # in-memory chain reproduces the file-based chain".
        if HAS_REALFFT && HAS_REDNOISE
            name = "ondisk"
            write(joinpath(dir, "$name.dat"), series)
            write_presto_inf(joinpath(dir, "$name.inf"), Np, dt, new_epoch; name)
            run(pipeline(Cmd(`$REALFFT $name.dat`; dir); stdout=devnull, stderr=devnull))
            run(pipeline(Cmd(`$REDNOISE $name.fft`; dir); stdout=devnull, stderr=devnull))
            ft_disk = FFTFile(joinpath(dir, name * "_red.fft"))
            c2 = search(ft_disk, SearchParams(nharms = 16, threshold = 0.0);
                        lofreq = 0.5, hifreq = 2.0, blocksize = 512,
                        threshold = 0.0, progress = :none, wisdom = false,
                        backend = CPUBackend())
            @test !isempty(c2)
            b2 = argmax(c -> c.metric, c2)
            @test abs(b2.freq - best.freq) <= 1 * df
            @test isapprox(b2.metric, best.metric; rtol = 0.01)
        else
            @info "Skipping the on-disk arm of the end-to-end test; PRESTO binaries not found"
        end
    end
end

@testset "CLI --demod-grid writes only .cohout" begin
    N = 1 << 14
    dt = 0.0098304
    epoch = 5.5e4
    dir = mktempdir()
    dat = joinpath(dir, "obs.dat")
    write(dat, synth_series(N; f = 1.0, snr = 25.0, seed = 41))
    write_presto_inf(joinpath(dir, "obs.inf"), N, dt, epoch; name = "obs")

    out = joinpath(dir, "out")
    grid = joinpath(dir, "grid.csv")
    write(grid, "pb,x,at\n0.041666666666666664,0.41361909,0.7\n")
    argv = [dat, "--demod-grid", grid, "--outdir", out, "--nharms", "8",
            "--threshold", "0.0", "--nowisdom", "--noprogress",
            "--lofreq", "0.5", "--hifreq", "2.0", "--blocksize", "64"]
    with_logger(NullLogger()) do
        CoherentSearch.main(argv)
    end
    # The requirement: the candidate file, and NOTHING else -- no .dat, no .inf,
    # no .fft, no <stem>_red.fft.
    @test readdir(out) == ["obs_demod_pb0.041666666666666664_x0.41361909_at0.7_red.cohout"]
    @test occursin("#Num", read(joinpath(out, only(readdir(out))), String))

    # A 2-model (hybrid) invocation: both CSVs in one call, both naming patterns.
    out2 = joinpath(dir, "out2")
    ajsgrid = joinpath(dir, "ajs.csv")
    write(ajsgrid, "accel,jerk,snap\n-338.7,0.2512,9.41e-4\n")
    with_logger(NullLogger()) do
        CoherentSearch.main([dat, "--demod-grid", grid, "--demod-grid", ajsgrid,
                             "--outdir", out2, "--nharms", "8", "--threshold", "0.0",
                             "--nowisdom", "--noprogress", "--lofreq", "0.5",
                             "--hifreq", "2.0", "--blocksize", "64"])
    end
    files = sort(readdir(out2))
    @test length(files) == 2
    @test any(f -> occursin(r"_demod_a-338\.7_j0\.2512_s9\.41e-4_red\.cohout$", f), files)
    @test any(f -> endswith(f, "_demod_pb0.041666666666666664_x0.41361909_at0.7_red.cohout"), files)
    @test all(f -> endswith(f, "_red.cohout"), files)

    # The naming contract: `demod/combine_cohout.py` must parse the output.
    combiner = joinpath(@__DIR__, "..", "demod", "combine_cohout.py")
    if isfile(PYTHON_)
        combined = joinpath(dir, "all.txt")
        # Put a row in each file, or the combiner has nothing to write.
        with_logger(NullLogger()) do
            CoherentSearch.main([dat, "--demod-grid", grid, "--demod-grid", ajsgrid,
                                 "--outdir", out2, "--nharms", "8", "--threshold", "0.0",
                                 "--nowisdom", "--noprogress", "--lofreq", "0.9",
                                 "--hifreq", "1.1", "--blocksize", "64"])
        end
        ok = success(pipeline(`$PYTHON_ $combiner $(joinpath(out2, "*.cohout")) -o $combined`;
                              stdout=devnull, stderr=devnull))
        @test ok
    else
        @info "Skipping the combine_cohout.py naming check; python not found" PYTHON_
    end
end


@testset "demod concurrency: same candidates, one .dat read" begin
    # The pipeline-parallel driver searches several grid points at once, each
    # with its own cache (workspaces are indexed by task index inside
    # `_search_region!`, so sharing one would be a data race) and pinned to its
    # own threads.  Concurrent and sequential must agree bit for bit -- the whole
    # reason the caches are per-slot rather than shared.
    N = 1 << 14
    dt = 0.0098304
    epoch = 5.5e4
    dir = mktempdir()
    dat = joinpath(dir, "obs.dat")
    write(dat, synth_series(N; f = 1.0, snr = 25.0, seed = 51))
    write_presto_inf(joinpath(dir, "obs.inf"), N, dt, epoch; name = "obs")

    # Four points, spanning two A_T values so the searches really are distinct.
    grid = joinpath(dir, "grid.csv")
    write(grid, "pb,x,at\n" *
                "0.041666666666666664,0.41361909,0.0\n" *
                "0.041666666666666664,0.41361909,1.0\n" *
                "0.041666666666666664,0.41361909,2.0\n" *
                "0.041666666666666664,0.41361909,3.0\n")

    base = [dat, "--demod-grid", grid, "--nharms", "8", "--threshold", "0.0",
            "--nowisdom", "--noprogress", "--lofreq", "0.5", "--hifreq", "2.0",
            "--blocksize", "64"]
    outs = (joinpath(dir, "seq"), joinpath(dir, "conc"))
    with_logger(NullLogger()) do
        CoherentSearch.main(vcat(base, ["--outdir", outs[1], "--demod-concurrency", "1"]))
        CoherentSearch.main(vcat(base, ["--outdir", outs[2], "--demod-concurrency", "4"]))
    end
    fs = sort(readdir(outs[1]))
    @test length(fs) == 4
    @test sort(readdir(outs[2])) == fs
    @test all(f -> endswith(f, "_red.cohout"), fs)
    for f in fs
        @test read(joinpath(outs[1], f), String) == read(joinpath(outs[2], f), String)
    end
    # And all four `.dat` reads happened once per invocation, not once per point:
    # the driver opens the .dat outside the grid loop, so this is structural --
    # pinned here by the absence of any per-point temporary file.  The candidate
    # files are the only things written.
    @test all(f -> endswith(f, ".cohout"), readdir(outs[2]))
end

@testset "CLI argument guards" begin
    dir = mktempdir()
    dat = joinpath(dir, "obs.dat")
    write(dat, zeros(Float32, 1024))
    write_presto_inf(joinpath(dir, "obs.inf"), 1024, 0.0098304, 5.5e4; name = "obs")
    grid = joinpath(dir, "grid.csv")
    write(grid, "pb,x,at\n0.041666666666666664,0.41361909,0.7\n")

    # A .dat without --demod-grid names the missing flag rather than failing
    # inside FFTFile, which strips 4 characters and looks for X.inf.
    err = try
        CoherentSearch.main([dat]); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("--demod-grid", err)

    # --demod-grid rejects the options that name a single output.
    base = [dat, "--demod-grid", grid, "--nharms", "8", "--nowisdom", "--noprogress"]
    for extra in (["-o", joinpath(dir, "x.txt")], ["--plot"], ["--plotstem", "x"])
        err = try
            with_logger(NullLogger()) do
                CoherentSearch.main([base; extra])
            end
            ""
        catch e
            sprint(showerror, e)
        end
        @test !isempty(err)
        @test occursin("--demod-grid", err)
    end
end
