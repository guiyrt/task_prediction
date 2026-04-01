# -- Builder Stage --
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy 
ENV UV_NO_DEV=1

WORKDIR /app

# Copy files needed to install dependencies
COPY pyproject.toml uv.lock README.md ./
COPY aware-protos/ ./aware-protos/

# Install dependencies (this layer will be cached)
RUN uv sync --frozen --no-install-project --no-editable

# Copy source code and install project
COPY src/ ./src
RUN uv sync --frozen --no-editable


# -- Runtime Stage --
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy the virtual environment
COPY --from=builder /app/.venv /app/.venv
# Set venv to PATH, so uv scripts entrypoints work
ENV PATH="/app/.venv/bin:$PATH"

# Copy models
COPY models/xgboost_hierarchical_v5/*.ubj models/xgboost_hierarchical_v5/*.json /app/model/

# Entrypoint
CMD ["task-pred", "serve"]