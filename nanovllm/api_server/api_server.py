import argparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

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

    output_tokens = await engine.generate(request_id, prompt, sampling_params)

    response_data = {
        "text": "".join(output_tokens),
        "status": "success"
    }

    return JSONResponse(response_data)
    


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