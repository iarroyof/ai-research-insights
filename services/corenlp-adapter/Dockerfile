FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=0 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY services/corenlp-adapter/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && pip install -r /tmp/requirements.txt
COPY services/corenlp-adapter/app.py /app/app.py
EXPOSE 9009
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","9009"]
