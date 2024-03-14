using LinearAlgebra

using Random
using CairoMakie

function image_sim_v4(xᵖ, yᵖ, x, y)
    PSFˣ = exp.(-(xᵖ .- Transpose(x)) .^ 2)
    PSFʸ = exp.(-(y .- Transpose(yᵖ)) .^ 2)
    return PSFˣ * PSFʸ
end

Random.seed!(1)

x = 20 * rand(20)
y = 20 * rand(20)

xᵖ = range(0, 20, 256)
yᵖ = range(0, 20, 256)

I = image_sim_v4(xᵖ, yᵖ, x, y)

fig = Figure(resolution=(750, 300))
ax = [
    Axis(fig[1, 1], aspect=1, backgroundcolor=:black, title="Point emitter"),
    Axis(fig[1, 2], aspect=1, backgroundcolor=:black, title="Final image"),
]
scatter!(ax[1], x, y, color=:white, markersize=5)
limits!(ax[1], 0, 20, 0, 20)
heatmap!(ax[2], xᵖ, yᵖ, I, colormap=:bone)
hidedecorations!.(ax)
save("fig2.png", fig)