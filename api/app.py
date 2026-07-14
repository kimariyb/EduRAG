from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.deps import get_system_status
from api.routes.faq import router as faq_router
from api.routes.qa import router as qa_router

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="EduRAG 教育问答系统 API",
    description="基于 EducationQASystem 的问答与 FAQ 管理接口",
    version="1.0.0",
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
