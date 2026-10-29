using Test
using CoherentSearch
using Logging: with_logger, NullLogger

# The fixture is a real 1 h GBT observation (Terzan 5, dereddened):
# N = 343,348, dt = 0.01048576 s -> T = 3600.2647 s, Nyquist = 47.6837 Hz
# (171,674 bins).  A rung-k trial is scored only while k*f < Nyquist, so the
# clips fall at f = 47.68 (k=1), 23.84 (2), 15.89 (3), 11.92 (4), 9.537 (5),
# 7.947 Hz (6).
const DISK_RED = joinpath(@__DIR__, "disk_red.fft")

# (position in `spec.decimations`, 1-based index into that section) for a
# candidate: `#Harm`=H means k = nharms/H and f = f0 + (j-1)*df on that rung.
function spectrum_index(spec::Spectrum, nharms::Integer, ft::FFTFile, c)
    k = nharms ÷ c.nharm
    pos = findfirst(==(k), spec.decimations)
    pos === nothing && return nothing, 0
    return pos, round(Int, (c.freq - k * spec.r_lo / ft.T) / (k * spec.lodr / ft.T)) + 1
end

@testset "spectrum: every trial's S/N, exact" begin
    ft = FFTFile(DISK_RED)
    @test ft.T ≈ 343348 * 0.01048576
    @test ft.dereddened
    p = SearchParams(nharms = 8, decimations = [1, 2, 4], m = 16)
    s = Spectrum()
    cands = with_logger(NullLogger()) do
        search(ft, p; lofreq = 0.5, hifreq = 2.0, threshold = -Inf,
               remove = false, harm_remove = false, spectrum = s,
               wisdom = false, progress = :none)
    end
    # threshold=-Inf keeps every scored trial, and no rung clips in this band.
    @test s.total == length(cands) ÷ 3
    @test s.decimations == [1, 2, 4]
    @test length(s.vals) == 3
    @test all(v -> length(v) == s.total, s.vals)
    @test all(v -> !any(isnan, v), s.vals)
    @test all(v -> all(isfinite, v), s.vals)
    # Every candidate's S/N sits at the (rung, trial) its frequency names.
    @test all(cands) do c
        pos, j = spectrum_index(s, p.nharms, ft, c)
        pos !== nothing && 1 <= j <= s.total && s.vals[pos][j] == Float32(c.metric)
    end
    # ...and each section holds exactly the k=1 trial count.
    @test count(c -> c.nharm == 8, cands) == s.total
    @test count(c -> c.nharm == 4, cands) == s.total
    @test count(c -> c.nharm == 2, cands) == s.total
end

@testset "spectrum: Nyquist-clipped rungs are NaN past the prefix" begin
    ft = FFTFile(DISK_RED)
    p = SearchParams(nharms = 60, decimations = collect(1:6), m = 16)
    s = Spectrum()
    with_logger(NullLogger()) do
        search(ft, p; lofreq = 7.9, hifreq = 8.0, threshold = 8.0,
               remove = false, harm_remove = false, spectrum = s,
               wisdom = false, progress = :none)
    end
    @test length(s.vals) == 6
    # k=6 stops at 7.947 Hz, inside this band; k<=5 never clip here.
    valid6 = count(g -> 6 * (s.r_lo + g * s.lodr) < ft.N / 2, 0:(s.total - 1))
    @test 0 < valid6 < s.total
    @test count(isnan, s.vals[6]) == s.total - valid6
    @test isnan(s.vals[6][valid6 + 1])          # the prefix is contiguous
    @test all(isfinite, s.vals[6][1:valid6])
    @test all(k -> !any(isnan, s.vals[k]), 1:5)
    @test all(k -> all(isfinite, s.vals[k]), 1:5)
end

@testset "spectrum: chunk invariance" begin
    # The per-rung scored set must not depend on the chunking, NaN pattern
    # included (a chunk boundary must not drop or duplicate a trial).
    ft = FFTFile(DISK_RED)
    p = SearchParams(nharms = 60, decimations = collect(1:6), m = 16)
    runs = map((512, 4096)) do bs
        s = Spectrum()
        with_logger(NullLogger()) do
            search(ft, p; lofreq = 0.5, hifreq = 0.6, threshold = 8.0,
                   remove = false, harm_remove = false, spectrum = s,
                   blocksize = bs, wisdom = false, progress = :none)
        end
        s
    end
    @test runs[1].total == runs[2].total
    @test all(isequal(runs[1].vals[i], runs[2].vals[i]) for i in eachindex(runs[1].vals))
end

@testset "spectrum: file layout and determinism" begin
    ft = FFTFile(DISK_RED)
    p = SearchParams(nharms = 8, decimations = [1, 2, 4], m = 16)
    s = Spectrum()
    with_logger(NullLogger()) do
        search(ft, p; lofreq = 0.5, hifreq = 2.0, threshold = -Inf,
               remove = false, harm_remove = false, spectrum = s,
               wisdom = false, progress = :none)
    end
    dir = mktempdir()
    path = joinpath(dir, "out.spectrum")
    with_logger(NullLogger()) do
        write_spectrum(s, path, ft, p; threshold = -Inf)
    end
    @test filesize(path) == 8192 + 3 * s.total * 4
    hdr = String(read(path)[1:8192])
    for needle in ("#rung k=1 ", "#rung k=4 ", "P(H=1)", "NaN", "total=$(s.total)")
        @test occursin(needle, hdr)
    end
    offs = [parse(Int, m.captures[1]) for m in eachmatch(r"offset=(\d+)", hdr)]
    @test offs == [8192 + i * 4 * s.total for i in 0:2]
    vals = Vector{Float32}(undef, s.total)
    open(path, "r") do io
        seek(io, 8192)
        read!(io, vals)
    end
    @test vals == s.vals[1]
    # No timestamp or host information: a second write is byte-identical.
    path2 = joinpath(dir, "again.spectrum")
    with_logger(NullLogger()) do
        write_spectrum(s, path2, ft, p; threshold = -Inf)
    end
    @test read(path) == read(path2)
end
