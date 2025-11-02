# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=0
WORKDIR /init

COPY services/models-init/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY services/models-init/run.sh /init/run.sh
RUN chmod +x /init/run.sh

# Volumes mounted at runtime: /models, /openie6
CMD ["/bin/bash","-lc","/init/run.sh"]
