# Production image.
#
#   docker build -f deploy/Dockerfile -t aqtp:latest .
#   docker run --rm -it --env-file .env -v "$PWD/data:/app/data" aqtp:latest aqtp status
#
# IST is baked in: every session boundary, expiry and square-off time in this
# system is expressed in Asia/Kolkata, and a container running in UTC would
# silently trade the wrong window.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata libgomp1 curl \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a code change does not invalidate the wheel cache.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ml,dashboard]"

COPY config ./config
COPY docs ./docs

RUN useradd --create-home --uid 10001 aqtp \
 && mkdir -p /app/data/journal /app/data/models /app/data/cache \
             /app/data/experiments /app/data/logs \
 && chown -R aqtp:aqtp /app
USER aqtp

VOLUME ["/app/data"]
EXPOSE 8787

# Fails the container health check when the config or credentials are wrong,
# which is exactly when it should not be receiving traffic or restarting quietly.
HEALTHCHECK --interval=60s --timeout=15s --start-period=45s --retries=3 \
  CMD aqtp config --validate || exit 1

# No --yes here: `docker run` is interactive by default and the typed
# confirmation is worth keeping. Override in compose for unattended runs.
ENTRYPOINT ["aqtp"]
CMD ["start"]
