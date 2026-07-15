from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.deps import ensure_system, get_system, get_system_status
from api.routes.faq import router as faq_router
from api.routes.qa import router as qa_router
from base.logger import logger

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_system()
    try:
        yield
    finally:
        system = get_system()
        mysql_client = getattr(system, "mysql_client", None)
        close = getattr(mysql_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close MySQL client")


app = FastAPI(
    title="EduRAG 教育问答系统 API",
    description="基于 EducationQASystem 的问答与 FAQ 管理接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(qa_router)
app.include_router(faq_router)


@app.get("/health", tags=["meta"])
def health():
    state = get_system_status()
    return {"status": "ok" if state["ready"] else "degraded", **state}


# Serve the frontend. Mounted last so the explicit /api routes are matched first.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
