from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ConversationDetail,
    ConversationMessage,
    ConversationMeta,
    CreateConversationRequest,
)
from app.services.conversation_store import get_conversation_store

router = APIRouter()


@router.post("/conversations", response_model=ConversationMeta)
def create_conversation(body: CreateConversationRequest):
    store = get_conversation_store()
    conv = store.create_conversation(user_id=body.user_id, title=body.title)
    return ConversationMeta(**conv)


@router.get("/conversations", response_model=list[ConversationMeta])
def list_conversations(user_id: str = "local_user"):
    store = get_conversation_store()
    conversations = store.list_conversations(user_id=user_id)
    return [ConversationMeta(**c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str):
    store = get_conversation_store()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = store.get_messages(conversation_id)
    return ConversationDetail(
        conversation=ConversationMeta(**conv),
        messages=[ConversationMessage(**m) for m in messages],
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    store = get_conversation_store()
    deleted = store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id}
