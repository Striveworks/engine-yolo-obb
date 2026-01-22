import logging
import os
import uuid

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from starlette.responses import JSONResponse

from engine_yolo_obb.model_handler import ModelHandler

logger = logging.getLogger(__name__)


class Handler:
    def __init__(self, model_dir: str):
        self.model_handler = ModelHandler(model_dir)

    async def infer(self, request: Request) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_body = await request.json()
        inputs = request_body["inputs"]
        outputs = self.model_handler.handle(inputs)
        assert len(inputs) == len(outputs), f"{len(inputs)=} != {len(outputs)=}"
        response_body = {
            "model_name": "model",
            "model_version": "1.0.0",
            "id": request_id,
            "parameters": None,
            "outputs": [
                {
                    "name": input.get("name", None),
                    "shape": [1],
                    "datatype": "BYTES",
                    "parameters": None,
                    "data": [output],
                }
                for input, output in zip(inputs, outputs)
            ],
        }
        return JSONResponse(content=response_body)

    async def ready(self) -> JSONResponse:
        return JSONResponse(content={"status": "ready"})


def build_app(handler: Handler):
    router = APIRouter()
    router.add_api_route("/ready", handler.ready, methods=["GET"])
    router.add_api_route("/infer", handler.infer, methods=["POST"])
    app = FastAPI()
    app.include_router(router)
    return app


def main():
    model_dir = os.getenv("MODEL_DIR", "/mnt/models")
    logger.info(f"MODEL_DIR={model_dir!r}")
    handler = Handler(model_dir)
    app = build_app(handler)

    uvicorn.run(
        app,
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", 8080)),
        workers=int(os.getenv("WEB_CONCURRENCY", 1)),
        log_config=None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
