# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra encoder --no-install-project

COPY mil2het ./mil2het
COPY modules ./modules
COPY configs ./configs
COPY scripts ./scripts
COPY network_propagation.py ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra encoder --no-editable


FROM builder AS test

COPY tests ./tests

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group test --extra encoder --no-editable

RUN /opt/venv/bin/pytest -q --cpu-only


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 1000 mil2het \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin mil2het \
    && install -d -o mil2het -g mil2het /workspace

USER mil2het
WORKDIR /workspace

ENTRYPOINT ["/opt/venv/bin/mil2het"]
CMD ["--help"]
