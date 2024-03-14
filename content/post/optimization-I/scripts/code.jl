### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 0aea22cf-564c-49bf-b470-1186dee9668f
using LinearAlgebra

# ╔═╡ 2dcf26c2-fc91-4a76-a4de-e36147e96657
using BenchmarkTools

# ╔═╡ 7ccc18e4-6946-4975-b70c-595827807df6
begin
    using Random
    Random.seed!(1)
end

# ╔═╡ 6091f9ea-2d98-11ee-3237-db65125aa388
md"
# Code Optimization Part I

In this file I provide the [Julia](http://www.julialang.org) source code for my [blog post](https://labpresse.com/code-optimization-in-scientific-research-part-i/). You can read this file directly but I highly recommend running it in an interactive fashion using [Pluto.jl](https://plutojl.org/)! 
"

# ╔═╡ 88b3549a-daf5-4793-972e-342843fab713
md"
## Load packages

To run this file, we first load some necessary packages. It is okay if you don't know how to add packages in Julia because Pluto will take care of it!

For linear algebra operation such as transpose, we need `LinearAlgebra`,
"

# ╔═╡ e72a6d6a-602d-42c6-b485-4bf3c9db9a8f
md"In order to benchmark the functions we write we need `BenchMarkTools`."

# ╔═╡ a5f84551-736b-4948-97d1-22074f580219
md"Finally, in order to provide reproducibility, we use `Random` and use 1 as the random number generation seed."

# ╔═╡ c0f2d3a6-9287-426d-823e-5455921a13f2
md"
## Variable preparation

Let's assume in the images we are simulating there are $N$ emitters in the region of interest defined by $[0, L]\times[0, L]$. To be consistent with blog post, we use
"

# ╔═╡ 8f5a8953-3e99-4a62-9c0e-eb95aa87de80
N = 20

# ╔═╡ 3fe28940-af10-4ebb-af6b-edadf717120a
L = 20

# ╔═╡ 1884b3d7-9edd-47fe-beb2-a15dbc972c1a
md"The functions described in my blog post (and defined below) each takes four arguments: `x`, `y`, `xᵖ`, and `yᵖ` such that:
- `x` and `y` are vectors storing all emitters' $x$-positions and $y$-positions, respectively.
- `xᵖ` and `yᵖ` are vectors storing all pixels' $x$-positions and $y$-positions, respectively.

Here, we assume emitters are uniformly scattered in the region of interest,"

# ╔═╡ deec64e2-117c-485a-866f-d2715e6bc101
x = L * rand(N)

# ╔═╡ 5a1f660b-8ca7-44df-9de6-705aae390ce3
y = L * rand(N)

# ╔═╡ 63e80403-4364-429b-9491-f3db9ef8cb60
md"Then, for `xᵖ` and `yᵖ`, we assume the area is monitored by $256\times256$ pixels of the same dimensions,"

# ╔═╡ e0e98d13-6762-479a-9d6d-a1a510b0f3b4
xᵖ = range(0, L, 256)

# ╔═╡ 067953bc-5628-4342-af4a-f2b44bcdaf9e
yᵖ = range(0, L, 256)

# ╔═╡ 9e43601a-a7ea-4c75-97db-fb6f4d2610e2
md"
## Function definitions

The function defined below are exactly the same as those in my blog post, but with more explanatory comments.
"

# ╔═╡ cb364b33-e653-463e-aa2d-6672af49b195
function image_sim_v1(xᵖ, yᵖ, x, y)
    # preallocate the memory space for PSF, which stores the contribution from each emitter to each pixel
    PSF = zeros(length(x), length(xᵖ), length(yᵖ))
    # a triple-nested for-loop that calculates each element of PSF, the n-loop is the out-most and the j-loop is the inner-most
    for n in eachindex(x), i in eachindex(xᵖ), j in eachindex(yᵖ)
        PSF[n, i, j] = exp(-(xᵖ[i] - x[n])^2 - (yᵖ[j] - y[n])^2)
    end
    # sum over all emitters to form a matrix (or a single image)
    return dropdims(sum(PSF, dims = 1), dims = 1)
end

# ╔═╡ e4d90b22-92bf-48c4-8ad9-6fcefcadb78d
function image_sim_v2(xᵖ, yᵖ, x, y)
    PSF = zeros(length(x), length(xᵖ), length(yᵖ))
    # a triple-nested for-loop but the order is switched
    for j in eachindex(yᵖ), i in eachindex(xᵖ), n in eachindex(x)
        PSF[n, i, j] = exp(-(xᵖ[i] - x[n])^2 - (yᵖ[j] - y[n])^2)
    end
    return dropdims(sum(PSF, dims = 1), dims = 1)
end

# ╔═╡ 0991be84-7fe0-41ac-b5f8-a6a3f4f5a1dd
function image_sim_v3(xᵖ, yᵖ, x, y)
    # only preallocate the memory space for one image
    PSF = zeros(length(xᵖ), length(yᵖ))
    for j in eachindex(yᵖ), i in eachindex(xᵖ), n in eachindex(x)
        # update each pixel by adding the contribution from the n-th emitter
        PSF[i, j] += exp(-(xᵖ[i] - x[n])^2 - (yᵖ[j] - y[n])^2)
    end
    return PSF
end

# ╔═╡ 36f58111-5478-4b3b-9bf7-0db78b8507ea
function image_sim_v4(xᵖ, yᵖ, x, y)
    # construct the matrix PSFˣ such that PSFˣ[i,j] = exp(-(xᵖ[i]-x[j])²)
    PSFˣ = exp.(-(xᵖ .- Transpose(x)) .^ 2)
    # construct the matrix PSFʸ such that PSFʸ[i,j] = exp(-(y[i]-yᵖ[j])²)
    PSFʸ = exp.(-(y .- Transpose(yᵖ)) .^ 2)
    # matrix umltiplication
    return PSFˣ * PSFʸ
end

# ╔═╡ 5b069b3b-57c8-4fa0-b686-78a9dab5599c
md"
## Sanity check

Before benchmarking these functions we must ensure they yield the same result. This can be done by:
"

# ╔═╡ edfa3a10-0004-4d35-a8f4-d467f2c1fb58
test_passed = begin
    I₁ = image_sim_v1(xᵖ, yᵖ, x, y)
    I₂ = image_sim_v2(xᵖ, yᵖ, x, y)
    I₃ = image_sim_v3(xᵖ, yᵖ, x, y)
    I₄ = image_sim_v4(xᵖ, yᵖ, x, y)
    isequal(I₁, I₂) && isequal(I₂, I₃) && isequal(I₃, I₄)
end

# ╔═╡ e2d568f3-3599-478b-bef0-d3bc55c5dbdd
md"Oh no! The pass has failed. Does this mean we have done something wrong? Not necessarily. As computers only have finite precision, it is expected that different algorithm may result in different outputs. We just need to make sure the difference is negligible. One way to do so is calculating the relative error regarding the actual value."

# ╔═╡ b85954de-bd3f-4dca-bf53-fc426bf89c54
largest_relative_error = max(
    maximum(abs.(I₁ .- I₂) ./ I₁),
    maximum(abs.(I₂ .- I₃) ./ I₂),
    maximum(abs.(I₃ .- I₃) ./ I₄),
)

# ╔═╡ daf80e8f-4fc1-4082-8cb4-07a558eda235
md"So we are safe!"

# ╔═╡ 3e4d3ff5-e4e1-48bd-b2bf-7c5f2c7c444c
md"
## Benchmarks
"

# ╔═╡ 08c22964-393c-4778-b7be-84e852550279
@btime image_sim_v1($xᵖ, $yᵖ, $x, $y); # semicolon is added to suppress output

# ╔═╡ a303e6ce-78ef-431e-919a-68013324ce98
@btime image_sim_v2($xᵖ, $yᵖ, $x, $y);

# ╔═╡ b14e8881-b28e-4973-a0ce-07a2730dd6a3
@btime image_sim_v3($xᵖ, $yᵖ, $x, $y);

# ╔═╡ 47c9b13e-2afd-45ea-9943-e382535fcf78
@btime image_sim_v4($xᵖ, $yᵖ, $x, $y);

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "0aa5d155aa584d7966880683c25d98a057bc58aa"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─6091f9ea-2d98-11ee-3237-db65125aa388
# ╟─88b3549a-daf5-4793-972e-342843fab713
# ╠═0aea22cf-564c-49bf-b470-1186dee9668f
# ╟─e72a6d6a-602d-42c6-b485-4bf3c9db9a8f
# ╠═2dcf26c2-fc91-4a76-a4de-e36147e96657
# ╟─a5f84551-736b-4948-97d1-22074f580219
# ╠═7ccc18e4-6946-4975-b70c-595827807df6
# ╟─c0f2d3a6-9287-426d-823e-5455921a13f2
# ╟─8f5a8953-3e99-4a62-9c0e-eb95aa87de80
# ╟─3fe28940-af10-4ebb-af6b-edadf717120a
# ╟─1884b3d7-9edd-47fe-beb2-a15dbc972c1a
# ╠═deec64e2-117c-485a-866f-d2715e6bc101
# ╠═5a1f660b-8ca7-44df-9de6-705aae390ce3
# ╟─63e80403-4364-429b-9491-f3db9ef8cb60
# ╠═e0e98d13-6762-479a-9d6d-a1a510b0f3b4
# ╠═067953bc-5628-4342-af4a-f2b44bcdaf9e
# ╟─9e43601a-a7ea-4c75-97db-fb6f4d2610e2
# ╠═cb364b33-e653-463e-aa2d-6672af49b195
# ╠═e4d90b22-92bf-48c4-8ad9-6fcefcadb78d
# ╠═0991be84-7fe0-41ac-b5f8-a6a3f4f5a1dd
# ╠═36f58111-5478-4b3b-9bf7-0db78b8507ea
# ╟─5b069b3b-57c8-4fa0-b686-78a9dab5599c
# ╠═edfa3a10-0004-4d35-a8f4-d467f2c1fb58
# ╟─e2d568f3-3599-478b-bef0-d3bc55c5dbdd
# ╠═b85954de-bd3f-4dca-bf53-fc426bf89c54
# ╟─daf80e8f-4fc1-4082-8cb4-07a558eda235
# ╟─3e4d3ff5-e4e1-48bd-b2bf-7c5f2c7c444c
# ╠═08c22964-393c-4778-b7be-84e852550279
# ╠═a303e6ce-78ef-431e-919a-68013324ce98
# ╠═b14e8881-b28e-4973-a0ce-07a2730dd6a3
# ╠═47c9b13e-2afd-45ea-9943-e382535fcf78
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
