import sys
from app.tasks.celery_app import celery_app

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    queues = ["cpu.default", "cpu.ingest"] if q == "cpu" else ["gpu.embed", "gpu.llm"]
    celery_app.worker_main(["worker", "-Q", ",".join(queues), "-l", "INFO"])
