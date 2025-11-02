from celery import Celery

celery_app = Celery(
    "ai_research_insights",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=4,
    task_acks_late=True,
    broker_transport_options={"visibility_timeout": 3600},
)
