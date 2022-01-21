from fastapi import APIRouter
from npgru.router import ping, predict

ROUTER_PREFIX = "/npgru"
app_router = APIRouter(prefix=ROUTER_PREFIX)
app_router.include_router(ping.router)
app_router.include_router(predict.router)
