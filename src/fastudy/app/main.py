import uvicorn
from fastapi import FastAPI

from fastudy.app.configs.router import router

app = FastAPI()
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(
        app="main:app",
        host="127.0.0.1",
        port=5000,
        reload=True  # reload application when change in the projects is detected
    )
