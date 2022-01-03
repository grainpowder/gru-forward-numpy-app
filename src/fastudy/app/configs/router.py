from fastapi import APIRouter

from fastudy.app.routers import ping

router = APIRouter()
router_prefix = "/fastudy/v1"

router.include_router(
    router=ping.router,
    prefix=router_prefix,
    tags=["ping"]  # to be displayed on swagger docs
)
