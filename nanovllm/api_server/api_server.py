import argparse
from typing import AsyncGenerator
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import json

from nanovllm.utils.logger import init_logger
from nanovllm import SamplingParams
from nanovllm.utils.util import random_uuid
from nanovllm.engine.async_llm import AsyncLLM


# logger = init_logger(__name__)

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    # logger.info("Received a request.")
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(
        request_id, prompt=prompt, sampling_params=sampling_params
    )
    
    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for outputs in results_generator:
                text_output = outputs
                ret = {"text": text_output}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        async def abort_request() -> None:
            await engine.abort(request_id)

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_output = None
        async for outputs in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = outputs

        assert final_output is not None
        text_output = final_output
        ret = {"text": text_output}
        return JSONResponse(ret)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="/data3/lqc/models/qwen3-0.6b")
    args = parser.parse_args()
    
    print(f"Starting FastServe API server at http://{args.host}:{args.port}")
    engine = AsyncLLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )