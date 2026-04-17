FROM ghcr.io/astral-sh/uv:bookworm AS base
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT}
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends git-lfs && \
    curl google.com

RUN --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

FROM base AS run
RUN --mount=type=bind,target=. \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable
ENV PATH=${UV_PROJECT_ENVIRONMENT}/bin:${PATH}

FROM base AS dev
ENV PRE_COMMIT_HOME=/pre-commit
RUN --mount=type=bind,source=.pre-commit-config.yaml,target=.pre-commit-config.yaml \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-groups --frozen --no-install-project && \
    git init . --quiet && \
    ls -l .git/ && \
    uv run --no-project pre-commit install-hooks && \
    rm -fr .git
