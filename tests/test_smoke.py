"""Smoke test: AutoGen memory provider talks to a mocked LedgerMem SDK."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock


def _install_fake_ledgermem() -> None:
    if "ledgermem" in sys.modules:
        return
    fake = types.ModuleType("ledgermem")

    class LedgerMem:
        def __init__(self, *a, **k):
            pass

        def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_1")

        def delete(self, memory_id):
            return None

        def list(self, limit=20, cursor=None):
            return types.SimpleNamespace(items=[], next_cursor=None)

    class AsyncLedgerMem:
        def __init__(self, *a, **k):
            pass

        async def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        async def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_1")

        async def delete(self, memory_id):
            return None

        async def list(self, limit=20, cursor=None):
            return types.SimpleNamespace(items=[], next_cursor=None)

    fake.LedgerMem = LedgerMem
    fake.AsyncLedgerMem = AsyncLedgerMem
    sys.modules["ledgermem"] = fake


_install_fake_ledgermem()

from autogen_core.memory import MemoryContent, MemoryMimeType  # noqa: E402
from ledgermem import LedgerMem  # noqa: E402
from ledgermem_autogen import LedgerMemMemory  # noqa: E402


def test_imports() -> None:
    assert LedgerMemMemory is not None


def test_add_and_query_via_sync_client() -> None:
    client = LedgerMem()
    client.add = MagicMock(return_value=None)
    hit = type("Hit", (), {"id": "m1", "content": "hello", "metadata": {}, "score": 0.5})()
    client.search = MagicMock(return_value=type("R", (), {"hits": [hit]})())

    memory = LedgerMemMemory(client=client, top_k=3)

    asyncio.run(memory.add(MemoryContent(content="hello", mime_type=MemoryMimeType.TEXT)))
    assert client.add.called

    result = asyncio.run(memory.query("hello"))
    assert len(result.results) == 1
    assert result.results[0].content == "hello"
    assert result.results[0].metadata["memory_id"] == "m1"
