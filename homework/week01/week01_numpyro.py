# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import altair as alt
import arviz as az
import jax.numpy as np  # jnp
from jax import random
import numpy as onp
import numpyro.distributions as dist
import pandas as pd

# %% [markdown] tags=[]
# ## 1.

# %% [markdown] tags=[]
# Suppose the globe tossing data (Chapter 2) had turned out to be 4 water in 15 tosses. Construct the posterior distribution, using grid approximation. Use the same flat prior as in the book.

# %%
W = 4
N = 15

# %% [markdown] tags=[]
# $$P ( p | W, N) \propto P ( W | p, N) P (p) $$

# %%
delta = 100

p_grid = np.linspace(0, 1, delta)
## prior
# prob_p = dist.Uniform(low=0.0, high=1.0).log_prob(p_grid)
prob_p = np.repeat(1, delta)
## likelihood
prob_data = np.exp(dist.Binomial(total_count=N, probs=p_grid).log_prob(W))
# more of a faff than scipy.stats pmf, but nicer
## postrior
posterior = prob_data * prob_p
posterior = posterior / sum(posterior)


# %%
def create_df(p_grid, prob_p, prob_data, posterior):
    dist_dict = {"prior": prob_p, "likelihood": prob_data, "posterior": posterior}

    df_list = []

    for name, distr in dist_dict.items():
        df_tmp = pd.DataFrame({"p": p_grid, "plausibility": distr, "distribution": name})
        df_list.append(df_tmp)

    df = pd.concat(df_list)

    return df

# %%
df = create_df(p_grid, prob_p, prob_data, posterior)

# %%
points = alt.Chart(df).mark_point().encode(x="p", y="plausibility", color="distribution")

lines = alt.Chart(df).mark_line().encode(x="p", y="plausibility", color="distribution")

lines + points

# %%
# TODO not great as grid approximation is too coarse
posterior_samples = random.choice(
    random.PRNGKey(0), a=p_grid, shape=(1_000_000,), replace=True, p=posterior
)

az.plot_posterior(onp.asarray(posterior_samples))

# %% [markdown]
# ## 2.

# %% [markdown]
# Start over in 1, but now use a prior that is zero below p = 0.5 and a constant above p = 0.5. This corresponds to prior information that a majority of the Earth’s surface is water. What difference does the better prior make?

# %%
# delta = 1_000

p_grid = np.linspace(0, 1, delta)
## prior
# prob_p = np.hstack([np.repeat(0.0, delta/2), np.repeat(2, delta/2)])
prob_p = 2 * (p_grid >= 0.5).astype(int)
## likelihood
prob_data = np.exp(
    dist.Binomial(total_count=N, probs=p_grid).log_prob(W)
)  # more of a faff than scipy.stats pmf
## postrior
posterior = prob_data * prob_p
posterior = posterior / sum(posterior)

# %%
df = create_df(p_grid, prob_p, prob_data, posterior)

# %%
points = alt.Chart(df).mark_point().encode(x="p", y="plausibility", color="distribution")

lines = alt.Chart(df).mark_line().encode(x="p", y="plausibility", color="distribution")

lines + points

# %% [markdown]
# ## 3.
#
# For the posterior distribution from 2, compute 89% percentile and HPDI intervals. Compare the widths of these intervals. Which is wider? Why? If you had only the information in the interval, what might you misunderstand about the shape of the posterior distribution?

# %%
posterior_samples = random.choice(
    random.PRNGKey(0), a=p_grid, shape=(1_000_000, 1), replace=True, p=posterior
)

# %%
np.percentile(posterior_samples, (5, 94))

# %%
az.hdi(onp.asarray(posterior_samples)[:, 0], hdi_prob=0.89)

# %%
