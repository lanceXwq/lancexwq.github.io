using CairoMakie
using Distributions
using LinearAlgebra
using SpecialFunctions

x, y = collect(-3:0.01:3), collect(-3:0.01:3)
px, py = collect(-3:0.4:3), collect(-3:0.4:3)

μ = [0, 0]
Σ = I(2) ./ 2

d = MvNormal(μ, Σ)
d2 = MvNormal(μ .+ 1.3, Σ)

z = [pdf(d, [X, Y]) for X ∈ x, Y ∈ y]
z2 = [pdf(d2, [X, Y]) for X ∈ x, Y ∈ y]

fig = Figure(backgroundcolor=(:white, 1), resolution=(1000, 300))
ax = [
    Axis(fig[1, 1], aspect=1, backgroundcolor=:black, title="Point emitter"),
    Axis(fig[1, 2], aspect=1, backgroundcolor=:black, title="Gaussian PSF"),
    Axis(fig[1, 3], aspect=1, backgroundcolor=:black, title="Pixelated image"),
]

scatter!(ax[1], μ[1], μ[2], color=:white)
heatmap!(ax[2], x, y, z, colormap=:bone)

@views erfx = erf.(px[1:end-1], px[2:end]) ./ 2
@views erfy = erf.(py[1:end-1], py[2:end]) ./ 2
@views erfx2 = erf.(px[1:end-1] .- 1.3, px[2:end] .- 1.3) ./ 2
@views erfy2 = erf.(py[1:end-1] .- 1.3, py[2:end] .- 1.3) ./ 2

heatmap!(ax[3], px, py, erfx * Transpose(erfy), colormap=:bone)

hidedecorations!.(ax)

save("fig1.png", fig)
