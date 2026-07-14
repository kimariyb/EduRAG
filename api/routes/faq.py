from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.deps import get_faq_backend, invalidate_faq_cache
from api.schemas import FAQCreate, FAQListResponse, FAQUpdate

router = APIRouter(prefix="/api/faq", tags=["faq"])


@router.get("", response_model=FAQListResponse)
def list_faqs(limit: int = 50, offset: int = 0):
    backend, _ = get_faq_backend()
    rows = backend.list_faqs(limit=limit, offset=offset)
    return FAQListResponse(items=rows)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_faq(payload: FAQCreate):
    backend, _ = get_faq_backend()
    faq_id = backend.insert_faq(
        question=payload.question, answer=payload.answer, subject=payload.subject
    )
    invalidate_faq_cache()
    return {"id": faq_id, "message": "created"}


@router.get("/{faq_id}")
def get_faq(faq_id: int):
    backend, _ = get_faq_backend()
    row = backend.get_faq(faq_id)
    if row is None:
        raise HTTPException(status_code=404, detail="FAQ 不存在")
    return row


@router.put("/{faq_id}")
def update_faq(faq_id: int, payload: FAQUpdate):
    backend, _ = get_faq_backend()
    try:
        backend.update_faq(
            faq_id,
            question=payload.question,
            answer=payload.answer,
            subject=payload.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    invalidate_faq_cache()
    return {"id": faq_id, "message": "updated"}


@router.delete("/{faq_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_faq(faq_id: int):
    backend, _ = get_faq_backend()
    backend.delete_faq(faq_id)
    invalidate_faq_cache()
    return None
