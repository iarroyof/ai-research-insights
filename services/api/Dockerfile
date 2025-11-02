# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONPATH=/app

# 1) Install deps with cache mount; this layer is reused until requirements.txt changes
COPY services/api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-warn-script-location --upgrade pip && \
    pip install --no-warn-script-location -r /tmp/requirements.txt

# 2) Code + config come from volumes at runtime (no rebuild for code edits)
#    - ./services/api -> /app
#    - ./config       -> /app/config
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
