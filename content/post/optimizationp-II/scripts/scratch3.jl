using LoopVectorization
using LinearAlgebra
using BenchmarkTools

N = 20
L = 20
F = 100

x = L * rand(N, 1, F)

y = L * rand(N, 1, F)

xᵖ = range(0, L, 256)

yᵖ = range(0, L, 256)

function video_sim(xᵖ, yᵖ, x, y)
    F = size(x, 3)
    v = Array{eltype(x),3}(undef, length(xᵖ), length(yᵖ), F)
    for f ∈ 1:F
        PSFˣ = exp.(-(xᵖ .- Transpose(view(x, :, 1, f))) .^ 2)
        PSFʸ = exp.(-(view(y, :, 1, f) .- Transpose(yᵖ)) .^ 2)
        v[:, :, f] = PSFˣ * PSFʸ
    end
    return v
end

@btime video_sim($xᵖ, $yᵖ, $x, $y)