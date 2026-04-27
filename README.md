# ledgermem-autogen

AutoGen 0.4+ `Memory` protocol implementation backed by [LedgerMem](https://github.com/ledgermem/ledgermem-python). Plug persistent semantic memory into any AutoGen agent.

## Install

```bash
pip install ledgermem-autogen
```

## Quickstart

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ledgermem import AsyncLedgerMem
from ledgermem_autogen import LedgerMemMemory


async def main() -> None:
    memory = LedgerMemMemory(
        client=AsyncLedgerMem(api_key="lm_...", workspace_id="ws_..."),
        top_k=5,
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        memory=[memory],
    )

    await agent.run(task="Remember that my favourite editor is Helix.")
    result = await agent.run(task="What is my favourite editor?")
    print(result.messages[-1].content)


asyncio.run(main())
```

## What it does

Before every model call, AutoGen invokes `memory.update_context(...)`. This adapter takes the latest user message, runs `LedgerMem.search(...)`, and injects the top-K hits as a `SystemMessage` so the model can see them. New facts get persisted via `memory.add(...)`.

## License

MIT — see [LICENSE](./LICENSE).
