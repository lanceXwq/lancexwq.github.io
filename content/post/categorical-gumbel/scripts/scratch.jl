using LogExpFunctions

function categorical_sampler1(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u
        c += p[i+=1]
    end
    return i
end

function categorical_sampler3(logp)
    x = -log.(-log.(rand(length(logp))))
    (~, n) = findmax(x .+ logp)
    return n
end

logp = -4 .* collect(0:5)
N = zeros(Int, 6, 2)

@time for i in 1:1_000_000_000
    N[categorical_sampler1(softmax(logp)), 1] += 1
    N[categorical_sampler3(logp), 2] += 1
end


