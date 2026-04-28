"""AutoGen 0.4+ ``Memory`` protocol implementation backed by LedgerMem.

The ``autogen_core.memory`` package defines a ``Memory`` protocol that the agent
runtime calls during conversation turns. Implementations must support
``add``, ``query``, ``update_context``, ``clear``, and ``close``.
"""

from __future__ import annotations

from typing import Any

from autogen_core import CancellationToken
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from ledgermem import AsyncLedgerMem, LedgerMem


def _to_text(content: MemoryContent | str) -> tuple[str, dict[str, Any]]:
    if isinstance(content, str):
        return content, {"mime_type": MemoryMimeType.TEXT.value}
    metadata = dict(content.metadata or {})
    metadata.setdefault("mime_type", str(content.mime_type))
    raw = content.content
    if isinstance(raw, (dict, list)):
        import json

        return json.dumps(raw, default=str), metadata
    return str(raw), metadata


class LedgerMemMemory(Memory):
    """An AutoGen ``Memory`` provider that persists into LedgerMem.

    Pass either a sync ``LedgerMem`` or an ``AsyncLedgerMem`` client. Sync
    clients are awaited via ``asyncio.to_thread`` to keep the agent loop
    non-blocking.
    """

    def __init__(
        self,
        client: LedgerMem | AsyncLedgerMem,
        *,
        top_k: int = 5,
        name: str = "ledgermem",
    ) -> None:
        self._client = client
        self._top_k = top_k
        self._name = name
        self._is_async = isinstance(client, AsyncLedgerMem)

    @property
    def name(self) -> str:
        return self._name

    async def _add(self, text: str, metadata: dict[str, Any]) -> Any:
        if self._is_async:
            return await self._client.add(text, metadata=metadata)
        import asyncio

        return await asyncio.to_thread(self._client.add, text, metadata=metadata)

    async def _search(self, query: str, limit: int) -> Any:
        if self._is_async:
            return await self._client.search(query, limit=limit)
        import asyncio

        return await asyncio.to_thread(self._client.search, query, limit=limit)

    async def _delete(self, memory_id: str) -> Any:
        if self._is_async:
            return await self._client.delete(memory_id)
        import asyncio

        return await asyncio.to_thread(self._client.delete, memory_id)

    async def _list(self, limit: int, cursor: str | None) -> Any:
        if self._is_async:
            return await self._client.list(limit=limit, cursor=cursor)
        import asyncio

        return await asyncio.to_thread(self._client.list, limit=limit, cursor=cursor)

    # --- Memory protocol --------------------------------------------------

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        text, metadata = _to_text(content)
        metadata.setdefault("source", "autogen")
        await self._add(text, metadata)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        query_str = query if isinstance(query, str) else _to_text(query)[0]
        limit = int(kwargs.get("limit", self._top_k))
        response = await self._search(query_str, limit)
        results: list[MemoryContent] = []
        for hit in getattr(response, "hits", []) or []:
            text = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = dict(getattr(hit, "metadata", {}) or {})
            score = getattr(hit, "score", None)
            if score is not None:
                metadata["score"] = score
            memory_id = getattr(hit, "id", None)
            if memory_id is not None:
                metadata["memory_id"] = memory_id
            results.append(MemoryContent(content=text, mime_type=MemoryMimeType.TEXT, metadata=metadata))
        return MemoryQueryResult(results=results)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        last_user = next(
            (msg for msg in reversed(messages) if getattr(msg, "source", None) == "user"),
            None,
        )
        if last_user is None or not getattr(last_user, "content", None):
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        query_text = (
            last_user.content if isinstance(last_user.content, str) else str(last_user.content)
        )
        results = await self.query(query_text)
        if results.results:
            joined = "\n".join(f"- {item.content}" for item in results.results)
            await model_context.add_message(
                SystemMessage(content=f"Relevant memories from LedgerMem:\n{joined}")
            )
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        # Collect every id BEFORE deleting. Cursors derived from a mutating
        # collection are unreliable — deleting in-place mid-pagination can
        # silently skip rows or loop forever depending on the backend.
        ids: list[str] = []
        cursor: str | None = None
        while True:
            page = await self._list(100, cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                memory_id = getattr(item, "id", None)
                if memory_id is not None:
                    ids.append(memory_id)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
        for memory_id in ids:
            await self._delete(memory_id)

    async def close(self) -> None:
        return None
