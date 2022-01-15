from npgru.routers import ping
from fastapi import APIRouter

# HTTP request form : HOST_IP:PORT/ROUTER_PREFIX/[ENDPOINT]
router = APIRouter()
ROUTER_PREFIX = "/npgru/api"

# methods in routers.ping will be described in swagger with "ping" tag
router.include_router(router=ping.router, prefix=ROUTER_PREFIX, tags=["ping"])
