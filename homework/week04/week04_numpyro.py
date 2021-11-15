# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: py38
# ---

# %%
import arviz as az
import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
import pandas as pd

# %% [markdown]
# ## 1

# %%
islands = {
    1: np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    2: np.array([0.8, 0.1, 0.05, 0.025, 0.025]),
    3: np.array([0.05, 0.15, 0.7, 0.05, 0.05]),
}


# %%
def entropy(p):
    return - np.sum(p * np.log(p))


# %%
for i, p in islands.items():
    print(f"Island {i}: {entropy(p)}")


# %%
def divergence_kl(p, q):
    return np.sum(p * (np.log(p) - np.log(q)))


# %%
for i, q in islands.items():
    for j, p in islands.items():
        if j == i:
            continue
        print(f"Model {i}, Prediction {j}: {divergence_kl(p, q)}")

# %% [markdown]
# ## 2

# %%
df = pd.read_csv('../../data/happiness.csv')

df.sample(3)

# %%
df2 = df[df.age > 17].copy()  # only adults
df2["A"] = (df2.age - 18) / (65 - 18)


# %%
# df2["mid"] = d2.married

# %%
def model(married, A, happiness):
    a = numpyro.sample("a", dist.Normal(0, 1).expand([len(set(married))]))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a[married] + bA * A
    numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)

m6_9 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_9,
    optim.Adam(1),
    Trace_ELBO(),
    married=df2.married.values,
    A=df2.A.values,
    happiness=df2.happiness.values,
)
svi_result = svi.run(random.PRNGKey(0), 1000)
p6_9 = svi_result.params
post_69 = m6_9.sample_posterior(random.PRNGKey(1), p6_9, (1000,))

# %%
print_summary(post_69, 0.89, False)

# %%
az.summary(post_69)


# %%
def model(A, happiness):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bA * A
    numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)


m6_10 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m6_10, optim.Adam(1), Trace_ELBO(), A=df2.A.values, happiness=df2.happiness.values
)
svi_result = svi.run(random.PRNGKey(0), 1000)
p6_10 = svi_result.params
post_610 = m6_10.sample_posterior(random.PRNGKey(1), p6_10, (1000,))

print_summary(post_610, 0.89, False)

# %%
az.waic(post_610)

# %%
