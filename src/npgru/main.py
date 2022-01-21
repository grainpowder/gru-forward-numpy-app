import uvicorn
from fastapi import FastAPI
import logging
import sys
from npgru.preprocessor import ModelFilePreparer, ModelFileDecompressor
from npgru.predictor import NumpyPredictor, TensorflowPredictor
from npgru.router import app_router

app = FastAPI(
    title="gru-forward-numpy-app",
    description="App to compare inference speed of ideated model and original Tensorflow model",
    version="0.1.1"
)

app.include_router(app_router)


@app.on_event("startup")
async def app_startup() -> None:
    logger = logging.Logger(name="start-up", level=logging.INFO)
    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(msg)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    logger.info("Prepare and decompress model files")
    ModelFilePreparer.execute()
    ModelFileDecompressor.execute()

    logger.info("Load NumpyPredictor")
    app.state.predictor_numpy = NumpyPredictor()

    logger.info("Load TensorflowPredictor")
    app.state.predictor_tensorflow = TensorflowPredictor()


if __name__ == "__main__":
    uvicorn.run(
        app="npgru.main:app",
        host="0.0.0.0",
        port=1234,
        reload=True,
        reload_dirs=["src"],
        workers=1
    )
