import argparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from nanovllm.utils.logger import init_logger
from nanovllm import SamplingParams
from nanovllm.utils.util import random_uuid


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
    # 返回固定响应（模拟成功）
    return JSONResponse({
        "text": "This is a placeholder response from FastServe API server.",
        "status": "success"
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    print(f"Starting FastServe API server at http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )