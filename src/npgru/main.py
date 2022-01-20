import uvicorn
from fastapi import FastAPI
from npgru.router.config import app_router

app = FastAPI(
    title="gru-forward-numpy-app",
    description="App to compare inference speed of ideated model and original Tensorflow model",
    version="0.1.1"
)

app.include_router(app_router)


if __name__ == "__main__":
    uvicorn.run(
        app="npgru.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["src"],
        workers=1
    )
