import json
import logging
import asyncio

from starlette.responses import StreamingResponse

log = logging.getLogger(__name__)


async def sse_stream(generator, heartbeat=10, headers=None):
    async def event_publisher():
        pending = None
        try:
            iterator = generator.__aiter__()
            pending = asyncio.create_task(iterator.__anext__())
            while True:
                done, _ = await asyncio.wait({pending}, timeout=max(1, heartbeat))
                if not done:
                    yield ": heartbeat\n\n"
                    continue
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
                pending = asyncio.create_task(iterator.__anext__())
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
        finally:
            if pending is not None and not pending.done():
                pending.cancel()

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers=headers,
    )
