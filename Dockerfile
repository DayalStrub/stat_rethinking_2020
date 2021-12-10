FROM jupyter/minimal-notebook

## tweak jupyter lab ----

# RUN jupyter labextension install @jupyterlab/toc

RUN mamba install --quiet --yes jupytext && \
    # jupyter labextension install jupyterlab-jupytext && \
    printf '\nc.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\nc.ContentsManager.default_jupytext_formats = "ipynb,py:percent,jl"' >> /etc/jupyter/jupyter_notebook_config.py

## update python and R ----

## NOTE seems like conda update only takes a list of requirements,
## and not an actual conda yaml

COPY python.yaml .
# RUN mamba update --name base --channel pytorch conda-forge --yes --file python.yaml

COPY r.yaml .
# RUN mamba update --name base --channel conda-forge --yes --file r.yaml
# RUN conda activate base && \
    # Rscript -e "remotes::install_github('rmcelreath/rethinking')"

## add julia -----

### copied form https://github.com/jupyter/docker-stacks/blob/master/datascience-notebook/Dockerfile

USER root

## Check https://julialang.org/downloads/
ARG julia_version="1.6.4"

## Julia dependencies
## install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_VERSION="${julia_version}"

WORKDIR /tmp

## hadolint ignore=SC2046
RUN set -x && \
    julia_arch=$(uname -m) && \
    julia_short_arch="${julia_arch}" && \
    if [ "${julia_short_arch}" == "x86_64" ]; then \
      julia_short_arch="x64"; \
    fi; \
    julia_installer="julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" && \
    julia_major_minor=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2) && \
    mkdir "/opt/julia-${JULIA_VERSION}" && \
    wget -q "https://julialang-s3.julialang.org/bin/linux/${julia_short_arch}/${julia_major_minor}/${julia_installer}" && \
    tar xzf "${julia_installer}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 && \
    rm "${julia_installer}" && \
    ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

## Show Julia where conda libraries are \
RUN mkdir /etc/julia && \
    echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl && \
    ## Create JULIA_PKGDIR \
    mkdir "${JULIA_PKGDIR}" && \
    chown "${NB_USER}" "${JULIA_PKGDIR}" && \
    fix-permissions "${JULIA_PKGDIR}"

### Add Julia packages -------

COPY Project.toml "${JULIA_PKGDIR}/environments/v1.6/"

RUN julia -e 'import Pkg; Pkg.activate()' && \
    julia -e 'import Pkg; Pkg.update()'
    # TODO check ENV persist
    # julia -e 'ENV["R_HOME"] = "/home/ubuntu/miniconda/envs/R/lib/R";' && \
    # julia -e 'ENV["PYTHON"] = "/home/ubuntu/miniconda/envs/py37/bin/python"; using Pkg; Pkg.build("PyCall");'

## add julia kernel --------

# USER ${NB_UID}

RUN julia -e 'using Pkg; Pkg.add("IJulia"); Pkg.precompile();' && \
    ## move kernelspec out of home \
    mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" && \
    chmod -R go+rx "${CONDA_DIR}/share/jupyter" && \
    rm -rf "${HOME}/.local" && \
    fix-permissions "${JULIA_PKGDIR}" "${CONDA_DIR}/share/jupyter"
 
## wrap up -------

USER ${NB_UID}

# WORKDIR "${HOME}"
WORKDIR /home/jovyan/notebooks
