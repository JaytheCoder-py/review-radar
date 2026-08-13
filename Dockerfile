# Two targets from one file: the nightly Cloud Run *job* and the read-only dashboard
# *service*. They share a layer, so the model of the world cannot drift between them.
#
#   docker build --target job     -t reviewradar-job .
#   docker build --target service -t reviewradar-web .

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Event ids are content hashes. Python randomises string hashing per process by
    # default, which would make the same conclusion hash differently on every run and
    # quietly destroy idempotency.
    PYTHONHASHSEED=0

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# Dependencies first, so a source change does not re-resolve the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra vertex --no-install-project

COPY src/ ./src/
COPY data/gold/ ./data/gold/
RUN uv sync --frozen --no-dev --extra vertex

# Never run as root. Cloud Run does not require it; nothing else should either.
RUN useradd --create-home --uid 10001 radar && chown -R radar:radar /app
USER radar

ENV PATH="/app/.venv/bin:$PATH"


# ----------------------------------------------------------------------------------
FROM base AS job
# A Cloud Run *job*: it runs to completion and exits. No request lifecycle, no port.
# Exits non-zero on a failure that is not per-filing, so Cloud Scheduler surfaces it
# rather than recording a successful run that did nothing.
ENTRYPOINT ["reviewradar"]
CMD ["ingest", "--help"]


# ----------------------------------------------------------------------------------
FROM base AS service
# The dashboard. Reads the event log and computes nothing.
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "reviewradar serve --db /data/events.duckdb --port ${PORT}"]
