# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY comps /home/user/comps
WORKDIR /home/user/comps/mcp_web_retrievers

# Install all dependencies in pyproject.toml
RUN uv venv
# equivalent to source .venv/bin/activate
# setup with dependencies in pyproject.toml
ENV PATH="/home/user/comps/mcp_web_retrievers/.venv/bin:${PATH}"
RUN uv sync

ENTRYPOINT ["sleep", "infinity"]