import json
import logging

from starlette.responses import StreamingResponse

log = logging.getLogger(__name__)


async def sse_stream(generator, heartbeat=10, headers=None):
    async def event_publisher():
        try:
            async for chunk in generator:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "event: end\n\n"
        except Exception as exc:
            log.exception("SSE generator failed")
            payload = {
                "type": "error",
                "data": {
                    "message": "The chat stream stopped before completion. Please retry; the server logged the failure.",
                    "error_type": exc.__class__.__name__,
                },
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "event: end\n\n"

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers=headers,
    )
