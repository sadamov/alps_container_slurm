FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN pip install --upgrade pip

# Dependencies
RUN pip install --no-cache-dir \
    "pandas<1.6.0dev0,>=1.3" \
    "fsspec==2024.2.0" \
    xarray \
    matplotlib \
    cartopy \
    pyproj \
    networkx \
    loguru \
    wandb \
    plotly \
    tueplots \
    jupyter-book \
    ipykernel \
    bokeh \
    numcodecs \
    pre-commit \
    pytest \
    pooch \
    gcsfs \
    spherical-geometry \
    jinja2 \
    dask[distributed] \
    "torch-geometric==2.3.1" \
    parse \
    pytorch-lightning \
    "dataclass-wizard<0.31.0" \
    sphinxcontrib-mermaid \
    isodate \
    semver \
    zarr \
    "dask==2024.1.1"

ARG CACHEBUST=11
RUN pip install --no-cache-dir --no-deps --force-reinstall \
        git+https://github.com/joeloskarsson/mllam-data-prep.git@arcdist_fix \
        git+https://github.com/joeloskarsson/weather-model-graphs.git@decoding_mask \
        git+https://github.com/sadamov/neural-lam.git@plot_boundaries
