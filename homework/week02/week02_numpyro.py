# -*- coding: utf-8 -*-
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
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO  # , init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
import pandas as pd

# %%
df = pd.read_csv("../../data/Howell1.csv", sep=";")

# %%
df.sample(5)

# %% [markdown]
# ### 1
#
# The weights listed below were recorded in the !Kung census, but heights were not recorded for these individuals. Provide predicted heights and 89% compatibility intervals for each of these individuals.

# %%
weights = [45, 40, 65, 31]

# %% [markdown]
# The weights are all adult weights, so we can restrict the dataset.

# %%
df_adults = df.loc[df["age"] >= 18, :]

# %%
len(df_adults.weight.values)

# %%
weight_mean = np.mean(df_adults.weight.values)
weight_mean


# %% [markdown]
# Model

# %%
def model(weight, height=None):
    weight_mean = np.mean(weight)
    # priors
    mu_0 = pyro.sample("mu_0", dist.Normal(178, 20))
    mu_1 = pyro.sample("mu_1", dist.Normal(0, 5))
    sigma = pyro.sample("sigma", dist.Uniform(0, 50))
    # likelihood
    mu = pyro.deterministic("mu", mu_0 + mu_1 * (weight - weight_mean))
    pyro.sample("height", dist.Normal(mu, sigma), obs=height)


# %% [markdown]
# Prior predictive

# %%
predictive = Predictive(model, num_samples=1_000)
height_pred_prior = predictive(random.PRNGKey(0), df_adults.weight.values)["height"]

# %%
height_pred_prior._value.shape

# %%
az.plot_dist(
    height_pred_prior._value,
    #     cumulative=True,
    label="height - prior predictive",
)
plt.show()

# %%
# az.plot_posterior(height_pred_prior._value)
# plt.show()

# %% [markdown]
# Inference

# %%
approximation = AutoLaplaceApproximation(model)

svi = SVI(
    model,
    approximation,
    optim.Adam(1),
    Trace_ELBO(),
    weight=df_adults.weight.values,
    height=df_adults.height.values,
)

svi_result = svi.run(random.PRNGKey(0), 2000)

# %%
params = svi_result.params
# params = svi.get_params()

# %%
samples = approximation.sample_posterior(random.PRNGKey(1), params, (1000,))
samples.pop("mu")
az.summary(samples)

# %% [markdown]
# Posterior predictive

# %%
predictive = Predictive(model, posterior_samples=samples)
height_pred_post = predictive(random.PRNGKey(0), df_adults.weight.values)["height"]

# %%
az.plot_posterior(height_pred_post._value)
plt.show()

# %%
az.plot_density(
    data=[df_adults.height.values, height_pred_prior._value, height_pred_post._value],
    data_labels=["data", "prior predictive", "posterior predictive"],
)
plt.show()

# %%
## alternative/manual approach - missing sigma

# mu_at_45 = samples["mu_0"] + samples["mu_1"] * (45 - weight_mean)

# az.plot_posterior(mu_at_45._value)
# plt.show()

# %% [markdown]
# Height for given values

# %%
predictive = Predictive(model, posterior_samples=samples)
height_pred = predictive(random.PRNGKey(0), np.array(weights))["height"]

# %%
# height_pred

# %%
df_pred = pd.DataFrame({"weight": weights})
df_pred["height_mean"] = height_pred._value.mean(axis=0)
df_pred["height_hdi"] = [az.hdi(s) for s in height_pred._value.T]
df_pred

# %% [markdown] tags=[]
# ### 2
#
# Model the relationship between height (cm) and the natural logarithm of weight (log-kg): log(weight). Use the entire Howell1 data frame, all 544 rows, adults and non-adults. Use any model type from Chapter 4 that you think useful: an ordinary linear regression, a polynomial or a spline. I recommend a plain linear regression, though. Plot the posterior predictions against the raw data

# %%

# %% [markdown]
# ### 3
#
# Plot the prior predictive distribution for the polynomial regression model in Chapter 4. You can modify the the code that plots the linear regression  prior predictive distribution. 20 or 30 parabolas from the prior should suffice to show where the prior probability resides. Can you modify the prior distributions of α, β1, and β2 so that the prior predictions stay within the biologically reasonable outcome space? That is to say: Do not try to fit the data by hand. But do try to keep the curves consistent with what you know about height and weight, before seeing these exact data.

# %%

# %%
