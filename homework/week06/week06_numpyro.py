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

import jax
import jax.numpy as jnp
from jax import random # , vmap
from jax.scipy.special import expit

import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
# import numpyro.optim as optim
# from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
# from numpyro.infer.autoguide import AutoLaplaceApproximation

import pandas as pd

# %%
jax.local_device_count()

# %% [markdown] tags=[]
# ## 1 
#
# What are the total and indirect causal effects of gender on grant awards? 
#
# Consider a mediation path (a pipe) through discipline. Draw the corresponding DAG and then use one or more binomial GLMs to answer the question. What is your causal interpretation? 
#
# If NWO’s goal is to equalize rates of funding between the genders, what type of intervention would be most effective?
#

# %%
df = pd.read_csv("../../data/NWOGrants.csv", sep=";")

# %%
df.info()

# %%
df.head(3)

# %%
df["gender_id"] = df["gender"].apply(lambda x: 0 if x == "m" else 1)
df["discipline_id"] = pd.factorize(df['discipline'])[0]

# %%
df.head(2)


# %% [markdown]
# ### Total influence - ignore discipline

# %%
def model_1(gender_id, applications, awards=None):
    a = numpyro.sample("a", dist.Normal(-1, 1).expand([2]))
    
    logit_p = a[gender_id]
    awards = numpyro.sample("awards", dist.Binomial(applications, logits=logit_p), obs=awards)
    
    p_awards = numpyro.deterministic("prob_award", awards / applications)
    
    a_diff = numpyro.deterministic("difference a", expit(a[0]) - expit(a[1]))


# %%
data_1 = {
    "gender_id": df["gender_id"].values,
    "applications": df["applications"].values,
    "awards": df["awards"].values,
}

mcmc_1 = MCMC(NUTS(model_1), num_warmup=500, num_samples=500, num_chains=4)
mcmc_1.run(random.PRNGKey(0), **data_1)

# %%
predictive = Predictive(model_1, num_samples=1_000)
pred_prior = predictive(random.PRNGKey(0), df["gender_id"].values, df["applications"].values)["prob_award"]

# %%
az.plot_posterior(pred_prior._value)

# %%
# # az.summary?

# %%
# mcmc_1.print_summary(0.89)
az.summary(mcmc_1, hdi_prob=0.89, var_names=["a", "difference a"])

# %%
az.plot_posterior(mcmc_1, var_names=["difference a"])


# %% [markdown]
# 3% is small but might be significant given low funding rates.

# %% [markdown]
# ### Direct influence

# %%
def model_2(gender_id, applications, discipline_id, awards=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([len(set(discipline_id))]) )    
    logit_p = a[gender_id] + b[discipline_id]
    awards = numpyro.sample("awards", dist.Binomial(applications, logits=logit_p), obs=awards)
    
    a_diff = numpyro.deterministic("difference a", expit(a[0]) - expit(a[1]))


# %%
data_2 = {
    "gender_id": df["gender_id"].values,
    "applications": df["applications"].values,
    "discipline_id": df["discipline_id"].values,
    "awards": df["awards"].values,
}

mcmc_2 = MCMC(NUTS(model_2), num_warmup=500, num_samples=500, num_chains=4)
mcmc_2.run(random.PRNGKey(0), **data_2)

# %%
az.summary(mcmc_2, hdi_prob=0.89)

# %%
# TODO this is wrong, as cannot inperpret the absolute difference as the impact
az.plot_posterior(mcmc_2, var_names=["difference a"])

# %%
# TODO plot posterior predictive
# for each dept and gender
# cf McElreath postcheck Fig 11.5 p.342

# mcmc_2.get_samples()

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## 2
#
# Suppose that the NWO Grants sample has an unobserved confound that influences both choice of discipline and the probability of an award. One example of such a confound could be the career stage of each applicant. Suppose that in some disciplines, junior scholars apply for most of the grants. In other disciplines, scholars from all career stages compete. As a result, career stage influences discipline as well as the probability of being awarded a grant. Add these influences to your DAG from Problem 1. What happens now when you condition on discipline? Does it provide an un-confounded estimate of the direct path from gender to an award? Why or why not? Justify your answer with the
# back-door criterion. Hint: This is structurally a lot like the grandparents-parentschildren-neighborhoods example from a previous week.
#
# If you have trouble thinking this though, try simulating fake data, assuming your DAG is true. Then analyze it using the model from Problem 1. What do you conclude? 
#
# __Is it possible for gender to have a real direct causal influence but for a regression conditioning on both gender and discipline to suggest zero influence?__

# %%

# %% [markdown]
# ## 3
#
# Use the available data to build one or more binomial GLMs of successful pirating attempts, using size and age as predictors. Consider any relevant interactions.

# %%
