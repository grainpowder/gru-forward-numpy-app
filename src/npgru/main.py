import uvicorn
from fastapi import FastAPI

from npgru.configs import app_config, router_config

config = app_config.get_config()

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

app.include_router(router_config.router)

if __name__ == "__main__":
    uvicorn.run(
        app="npgru.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["src"],
        workers=1
    )
