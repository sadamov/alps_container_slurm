FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

# Copy constraints first
COPY constraints.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

# Install core dependencies with specific versions
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1

# Install project-specific packages
RUN pip install --no-cache-dir -c constraints.txt \
    git+https://github.com/joeloskarsson/mllam-data-prep.git@arcdist_fix \
    git+https://github.com/joeloskarsson/weather-model-graphs.git@decoding_mask \
    git+https://github.com/joeloskarsson/neural-lam-dev.git@boundary_forcing
