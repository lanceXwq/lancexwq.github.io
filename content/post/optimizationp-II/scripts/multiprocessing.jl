# This file is the Julia source code regarding multiprocessing for my blog post at https://labpresse.com/code-optimization-in-scientific-research-part-ii/.

# Note that this example is very preliminary and the multiprocessing realization is obtained similarly to multithreading (by using SharedArrays).

# For a code that can actually run on a cluster, check Distributed.jl.

using Distributed
using SharedArrays
using BenchmarkTools

addprocs()
@everywhere using LinearAlgebra

function video_sim(xᵖ, yᵖ, x, y)
    F = size(x, 3)
    v = SharedArray(Array{eltype(x),3}(undef, length(xᵖ), length(yᵖ), F))
    𝑥 = SharedArray(x)
    𝑦 = SharedArray(y)
    @sync @distributed for f in 1:F
        v[:, :, f] = image_sim(xᵖ, yᵖ, view(𝑥, :, 1, f), view(𝑦, :, 1, f))
    end
    return v
end

@everywhere function image_sim(xᵖ, yᵖ, x, y)
    PSFˣ = exp.(-(xᵖ .- Transpose(x)) .^ 2)
    PSFʸ = exp.(-(y .- Transpose(yᵖ)) .^ 2)
    return PSFˣ * PSFʸ
end

@everywhere N = 20
@everywhere L = 20
@everywhere F = 100

x = L * rand(N, 1, F)
y = L * rand(N, 1, F)
@everywhere xᵖ = range(0, L, 256)
@everywhere yᵖ = range(0, L, 256)

@btime video_sim($xᵖ, $yᵖ, $x, $y)