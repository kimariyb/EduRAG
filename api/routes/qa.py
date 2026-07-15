from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from api.deps import get_system
from api.schemas import AskRequest, AskResponse, ClearResponse, HistoryResponse
from base.logger import logger

router = APIRouter(prefix="/api/qa", tags=["qa"])
log = logger.bind(module=__name__)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(jsonable_encoder(payload), ensure_ascii=False)}\n\n"


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, system=Depends(get_system)):
    if system is None:
        raise HTTPException(status_code=503, detail="问答系统未就绪")
    try:
        result = system.query(
            req.query, source_filter=req.source_filter, session_id=req.session_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Answer query failed")
        raise HTTPException(status_code=503, detail="问答后端不可用") from exc
    return AskResponse(
        session_id=result.session_id,
        source=result.source,
        answer=result.answer,
        history=result.history,
    )


@router.post("/ask/stream")
def ask_stream(req: AskRequest, system=Depends(get_system)):
    """Server-Sent Events stream: meta -> token* -> done | error."""
    if system is None:
        raise HTTPException(status_code=503, detail="问答系统未就绪")
    try:
        session_id, source, chunks = system.stream_query(
            req.query, source_filter=req.source_filter, session_id=req.session_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Answer stream setup failed")
        raise HTTPException(status_code=503, detail="问答后端不可用") from exc

    def event_stream() -> Iterator[str]:
        try:
            yield _sse({"type": "meta", "session_id": session_id, "source": source})
            for chunk in chunks:
                if chunk:
                    yield _sse({"type": "token", "content": chunk})
            history = system.get_session_history(session_id)
            yield _sse({"type": "done", "history": history})
        except Exception:  # noqa: BLE001
            log.exception("Answer stream failed")
            yield _sse({"type": "error", "message": "Answer generation failed."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def history(session_id: str, system=Depends(get_system)):
    if system is None:
        raise HTTPException(status_code=503, detail="问答系统未就绪")
    return HistoryResponse(
        session_id=session_id, history=system.get_session_history(session_id)
    )


@router.delete("/sessions/{session_id}", response_model=ClearResponse)
def clear(session_id: str, system=Depends(get_system)):
    if system is None:
        raise HTTPException(status_code=503, detail="问答系统未就绪")
    cleared = system.clear_session_history(session_id)
    return ClearResponse(session_id=session_id, cleared=bool(cleared))
