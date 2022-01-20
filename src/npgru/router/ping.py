from fastapi import APIRouter

ROUTER_PREFIX = "/health"
router = APIRouter(prefix=ROUTER_PREFIX, tags=["health check"])


@router.get("/ping")
async def ping() -> str:
    return "pong"
