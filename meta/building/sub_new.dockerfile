# This file will act as the build stage (stage 1) It also has version numbers
# Date 2025-02-13
FROM rocker/tidyverse:4.2.2 AS builder

WORKDIR /app

# Install system dependencies with versions
# curl and python3-pip are unpinned: their exact jammy versions were purged from the
# Ubuntu archive by later security updates, so an exact pin no longer resolves.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gdebi-core=0.9.5.7+nmu6 \
    libgl1=1.4.0-1 \
    libglx-mesa0=23.2.1-1ubuntu3.1~22.04.3 \
    libxt6=1:1.2.1-1 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install jupyter==1.0.0

# Install Quarto (1.4.557: first 1.4 line removed the compromised polyfill.io MathJax shim)
RUN curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.557/quarto-1.4.557-linux-amd64.deb \
    && gdebi -n quarto-1.4.557-linux-amd64.deb \
    && rm quarto-1.4.557-linux-amd64.deb

# Install R packages with version numbers
RUN R -e "install.packages('remotes')" && \
    R -e "remotes::install_version('renv', version = '0.15.5', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('stargazer', version = '5.2.3', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('xfun', version = '0.47', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('vctrs', version = '0.6.5', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('rmarkdown', version = '2.29', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('tidyverse', version = '2.0.0', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('knitr', version = '1.49', repos = 'https://cloud.r-project.org/')" && \
    R -e "remotes::install_version('IRkernel', version = '1.3.2', repos = 'https://cloud.r-project.org/')"

# Verify Quarto installation
RUN quarto check
