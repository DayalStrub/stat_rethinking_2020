FROM jupyter/datascience-notebook

## tweak jupyter lab ----

RUN jupyter labextension install @jupyterlab/toc

RUN mamba install --quiet --yes jupytext && \
    jupyter labextension install jupyterlab-jupytext && \
    printf '\nc.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\nc.ContentsManager.default_jupytext_formats = "ipynb,py:percent,jl,auto"' >> /etc/jupyter/jupyter_notebook_config.py

## update python ----
## cf https://github.com/jupyter/docker-stacks/blob/master/datascience-notebook/Dockerfile

RUN mamba install --channel pytorch --quiet --yes \
    'black' \
    'jupytext' \
    'altair' \
    'arviz>=0.11' \
    'cpuonly' \
    'numpyro>=0.7' \
    'pygraphviz' \
    'pymc3>=3.11' \
    'pyro-ppl>=1.7' \
    'pytorch' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

## update julia -----

WORKDIR /notebooks
