# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

COPY services/streamlit/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-warn-script-location --upgrade pip && \
    pip install --no-warn-script-location -r /tmp/requirements.txt

COPY services/streamlit/app.py /app/app.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
