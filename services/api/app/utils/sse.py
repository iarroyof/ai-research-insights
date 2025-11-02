import json
from starlette.responses import StreamingResponse

async def sse_stream(generator, heartbeat=10, headers=None):
    async def event_publisher():
        try:
            async for chunk in generator:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "event: end\n\n"
        finally:
            pass
    
    return StreamingResponse(
        event_publisher(), 
        media_type="text/event-stream",
        headers=headers
    )
