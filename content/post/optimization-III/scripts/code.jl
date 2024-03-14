using LinearAlgebra
using BenchmarkTools
using Random

Random.seed!(1)

using CUDA
using Flux

using Test

function video_sim_v3(xᵖ, yᵖ, x, y)
    F = size(x, 2)
    v = Array{eltype(x),3}(undef, length(xᵖ), length(yᵖ), F)
    Threads.@threads for f in 1:F
        PSFˣ = exp.(-(xᵖ .- Transpose(view(x, :, f))) .^ 2)
        PSFʸ = exp.(-(view(y, :, f) .- Transpose(yᵖ)) .^ 2)
        v[:, :, f] = PSFˣ * PSFʸ
    end
    return v
end

function video_sim_GPU_v1(xᵖ, yᵖ, x, y)
    F = size(x, 2)
    v = Array{eltype(x),3}(undef, length(xᵖ), length(yᵖ), F)
    for f in 1:F
        PSFˣ = exp.(-(xᵖ .- Transpose(view(x, :, f))) .^ 2)
        PSFʸ = exp.(-(view(y, :, f) .- Transpose(yᵖ)) .^ 2)
        v[:, :, f] = PSFˣ * PSFʸ
    end
    return v
end

function video_sim_GPU_v2(xᵖ, yᵖ, x, y)
    PSFˣ = exp.(-(reshape(x, 1, size(x)...) .- xᵖ) .^ 2)
    PSFʸ = exp.(-(reshape(y, 1, size(y)...) .- yᵖ) .^ 2)
    return batched_mul(PSFˣ, batched_adjoint(PSFʸ))
end

N = 20
L = 20
F = 100

# Float64 version
x = L * rand(Float64, N, F)
y = L * rand(Float64, N, F)

xᵖ = Float64.(collect(range(0, L, 256)))
yᵖ = Float64.(collect(range(0, L, 256)))

# Float32 version
x = L * rand(Float32, N, F)
y = L * rand(Float32, N, F)

xᵖ = Float32.(collect(range(0, L, 256)))
yᵖ = Float32.(collect(range(0, L, 256)))

V₁ = @btime video_sim_v3($xᵖ, $yᵖ, $x, $y)
# video_sim_GPU_v1($xᵖ, $yᵖ, CuArray($x), CuArray($y))
V₂ = @btime video_sim_GPU_v2($xᵖ, $yᵖ, $x, $y)
V₃ = @btime CUDA.@sync video_sim_GPU_v2(CuArray($xᵖ), CuArray($yᵖ), CuArray($x), CuArray($y))

@test isequal(V₁, V₂) && isequal(V₂, V₃)
