using CairoMakie
using ColorSchemes

f = Figure(resolution=(700, 200))
g = f[1, 1:2] = GridLayout()
ax = [Axis(g[1, 1]), Axis(g[1, 2])]

barplot!(ax[1], [0, 0, 0, 0], [0.3, 0.4, 0.1, 0.2], color=ColorSchemes.tab10[1:4], direction=:x, stack=1:4, width=0.1)

barplot!(ax[2], [0, 0, 0, 0], [0.4, 0.3, 0.2, 0.1], color=[ColorSchemes.tab10[2], ColorSchemes.tab10[1], ColorSchemes.tab10[4], ColorSchemes.tab10[3]], direction=:x, stack=1:4, width=0.1)

arrows!(ax[1], [0.45], [0.2], [0], [-0.05])
arrows!(ax[2], [0.45], [0.2], [0], [-0.05])

xlims!.(ax, 0, 1)
ylims!.(ax, 0, 0.3)
hideydecorations!.(ax)
hidexdecorations!.(ax, ticks=false, ticklabels=false)
hidespines!.(ax, :t, :r, :l)
colgap!(g, 50)

save("./230830 Gumbel Categorical/stick-breaking.png", f)