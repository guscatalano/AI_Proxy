# AI Proxy container image.
#
#   docker build -t ai-proxy .
#   docker run --rm -p 8000:8000 -v ai-proxy-data:/data \
#     -e OLLAMA_URL=http://host.docker.internal:11434 \
#     --add-host host.docker.internal:host-gateway ai-proxy
#
# Or just: docker compose up  (see docker-compose.yml)

FROM python:3.12-slim

# Runtime config. State (SQLite DB, rules, generated images) lives on a volume so it
# survives container recreation. Bind to all interfaces so the mapped port is reachable.
ENV PROXY_HOST=0.0.0.0 \
    PROXY_PORT=8000 \
    PROXY_STATE_DIR=/data \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml MANIFEST.in README.md LICENSE requirements.txt ./
COPY ai_proxy ./ai_proxy
RUN pip install .

# Non-root runtime user; owns the state volume mountpoint.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data
USER appuser
VOLUME /data

EXPOSE 8000
CMD ["ai-proxy"]
