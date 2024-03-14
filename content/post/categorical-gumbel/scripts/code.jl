using LogExpFunctions
using GLMakie

function categorical_sampler1(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u
        c += p[i+=1]
    end
    return i
end

function categorical_sampler2(logp)
    i = 1
    c = logp[1]
    u = log(rand()) + logsumexp(logp)
    while c < u
        c = logaddexp(c, logp[i+=1])
    end
    return i
end

function categorical_sampler3(logp)
    x = -log.(-log.(rand(length(logp))))
    (~, n) = findmax(x .+ logp)
    return n
end

function categorical_sampler4(logp)
    x = -cloglog.(rand(length(logp)))
    (~, n) = findmax(x .+ logp)
    return n
end

logp = log.(rand(100) .* pi .* 1e3)
L = 10000
n1 = zeros(Int, L)
n2 = zeros(Int, L)
n3 = zeros(Int, L)

for i in 1:L
    n1[i] = categorical_sampler1(softmax(logp))
    n2[i] = categorical_sampler2(logp)
    n3[i] = categorical_sampler3(logp)
end

f = Figure()
ax = Axis(f[1, 1])
hist!(ax, n1)
hist!(ax, n2)
hist!(ax, n3)
f