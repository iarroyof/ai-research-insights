# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

WORKDIR /app
ENV PYTHONPATH=/app/app

# 1) Install deps
COPY services/api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-warn-script-location --upgrade pip && \
    pip install --no-warn-script-location -r /tmp/requirements.txt

# Install NER models
RUN pip install --no-warn-script-location \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

# 2) Copy app code into image (for production)
COPY services/api /app
COPY config /app/config

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
