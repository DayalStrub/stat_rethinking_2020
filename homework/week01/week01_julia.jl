# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Julia 1.6.4
#     language: julia
#     name: julia-1.6
# ---

using Turing
# using StatsPlots

W = 4
N = 15;

# +
delta = 100

p_grid = range(0.0, 1.0, length=delta)
# ## prior
prob_p = pdf.(Uniform(0, 1), p_grid)
# ## likelihood
prob_data = pdf.(Binomial.(N, p_grid), W)
# ## postrior
posterior = prob_data .* prob_p
posterior = posterior ./ sum(posterior);

# +
# plot(posterior)
# -


