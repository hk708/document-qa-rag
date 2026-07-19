from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore(ABC):
    @abstractmethod
    def create_conversation(self, user_id: str = "local_user", title: str | None = None) -> dict:
        pass

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> dict | None:
        pass

    @abstractmethod
    def list_conversations(self, user_id: str = "local_user") -> list[dict]:
        pass

    @abstractmethod
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list[dict]:
        pass

    @abstractmethod
    def update_summary(self, conversation_id: str, summary: str) -> None:
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        pass


class JsonConversationStore(ConversationStore):
    def __init__(self, base_dir: str | Path = "data/conversations"):
        self.base_dir = Path(base_dir)
        self.messages_dir = self.base_dir / "messages"
        self.conversations_file = self.base_dir / "conversations.json"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.messages_dir.mkdir(parents=True, exist_ok=True)

        if not self.conversations_file.exists():
            self._write_conversations([])

    def _read_conversations(self) -> list[dict]:
        try:
            raw = self.conversations_file.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write_conversations(self, conversations: list[dict]) -> None:
        self.conversations_file.write_text(
            json.dumps(conversations, indent=2),
            encoding="utf-8",
        )

    def _message_file(self, conversation_id: str) -> Path:
        return self.messages_dir / f"{conversation_id}.json"

    def _next_conversation_id(self, conversations: list[dict]) -> str:
        max_num = 0
        for conv in conversations:
            cid = conv.get("conversation_id", "")
            if cid.startswith("conv_"):
                try:
                    max_num = max(max_num, int(cid.split("_")[1]))
                except ValueError:
                    continue
        return f"conv_{max_num + 1:03d}"

    def _touch_updated_at(self, conversation_id: str) -> None:
        conversations = self._read_conversations()
        now = _utc_now_iso()
        for conv in conversations:
            if conv["conversation_id"] == conversation_id:
                conv["updated_at"] = now
                break
        self._write_conversations(conversations)

    def create_conversation(self, user_id: str = "local_user", title: str | None = None) -> dict:
        conversations = self._read_conversations()
        conversation_id = self._next_conversation_id(conversations)
        now = _utc_now_iso()

        conv = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title or "New Chat",
            "summary": "",
            "created_at": now,
            "updated_at": now,
        }

        conversations.append(conv)
        self._write_conversations(conversations)

        self._message_file(conversation_id).write_text("[]", encoding="utf-8")
        return conv

    def get_conversation(self, conversation_id: str) -> dict | None:
        for conv in self._read_conversations():
            if conv["conversation_id"] == conversation_id:
                return conv
        return None

    def list_conversations(self, user_id: str = "local_user") -> list[dict]:
        conversations = [c for c in self._read_conversations() if c.get("user_id") == user_id]
        return sorted(conversations, key=lambda c: c.get("updated_at", ""), reverse=True)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> dict:
        path = self._message_file(conversation_id)
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                messages = []
        except json.JSONDecodeError:
            messages = []

        message = {
            "role": role,
            "content": content,
            "timestamp": _utc_now_iso(),
        }
        if sources is not None:
            message["sources"] = sources

        messages.append(message)
        path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
        self._touch_updated_at(conversation_id)
        return message

    def get_messages(self, conversation_id: str) -> list[dict]:
        path = self._message_file(conversation_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def update_summary(self, conversation_id: str, summary: str) -> None:
        conversations = self._read_conversations()
        for conv in conversations:
            if conv["conversation_id"] == conversation_id:
                conv["summary"] = summary
                conv["updated_at"] = _utc_now_iso()
                break
        self._write_conversations(conversations)

    def delete_conversation(self, conversation_id: str) -> bool:
        conversations = self._read_conversations()
        original_len = len(conversations)
        conversations = [c for c in conversations if c["conversation_id"] != conversation_id]
        deleted = len(conversations) != original_len
        if not deleted:
            return False

        self._write_conversations(conversations)
        path = self._message_file(conversation_id)
        if path.exists():
            path.unlink()
        return True


_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = JsonConversationStore()
    return _store
