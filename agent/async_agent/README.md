# Async Agent Documentation

The async agent module provides asynchronous versions of the Obsidian Agent, enabling concurrent processing and improved performance for batch operations.

## Features

- **Full Async Support**: Complete async/await implementation of the Agent class
- **Concurrent Execution**: Run multiple agents simultaneously 
- **Backward Compatible**: Works alongside existing sync implementation
- **Performance Benefits**: Significant speedup for batch processing tasks
- **Memory Isolation**: Each agent maintains its own isolated memory directory

## Installation

The async agent uses the same dependencies as the sync agent plus:
```bash
pip install aiofiles tqdm
```

## Basic Usage

### Synchronous Agent (Original)
```python
from agent.agent import Agent

# Create sync agent
agent = Agent(memory_path="my_memory")

# Chat synchronously
response = agent.chat("Create a file called 'notes.txt'")
print(response.reply)

# Save conversation
agent.save_conversation(log=True)
```

### Asynchronous Agent
```python
import asyncio
from agent.async_agent import AsyncAgent

async def main():
    # Create async agent
    agent = AsyncAgent(memory_path="my_async_memory")
    
    # Chat asynchronously
    response = await agent.chat("Create a file called 'async_notes.txt'")
    print(response.reply)
    
    # Save conversation
    await agent.save_conversation(log=True)

# Run async function
asyncio.run(main())
```

## Advanced Usage

### Concurrent Agents
```python
from agent.async_agent import AsyncAgent, run_agents_concurrently

async def concurrent_example():
    # Create multiple agents
    agents = [
        AsyncAgent(memory_path=f"agent_{i}_memory")
        for i in range(5)
    ]
    
    # Different tasks for each agent
    messages = [
        f"Create a summary for topic {i}"
        for i in range(5)
    ]
    
    # Run all agents concurrently
    responses = await run_agents_concurrently(agents, messages)
    
    for i, response in enumerate(responses):
        print(f"Agent {i}: {response.reply}")

asyncio.run(concurrent_example())
```

### Mixed Sync/Async Usage
```python
async def mixed_example():
    sync_agent = Agent(memory_path="sync_memory")
    async_agent = AsyncAgent(memory_path="async_memory")
    
    # Run sync agent in executor
    loop = asyncio.get_event_loop()
    sync_task = loop.run_in_executor(None, sync_agent.chat, "Sync task")
    
    # Run async agent normally
    async_response = await async_agent.chat("Async task")
    sync_response = await sync_task
    
    print(f"Sync: {sync_response.reply}")
    print(f"Async: {async_response.reply}")
```

### Batch Processing for SFT
```python
from agent.async_agent.async_sft_adapter import async_generate_conversations_batch

async def sft_batch_example():
    # Generate conversations for multiple personas concurrently
    results = await async_generate_conversations_batch(
        personas=persona_list,
        facts_per_persona=facts_lists,
        persona_model_factory=create_persona_model,
        num_turns=4,
        max_concurrent=10  # Limit concurrent conversations
    )
    
    success_rate = sum(results) / len(results)
    print(f"Success rate: {success_rate:.2%}")
```

## Performance Comparison

The async implementation provides significant performance improvements for parallel operations:

```python
# Synchronous: Process 10 agents sequentially
# Time: ~50 seconds

# Asynchronous: Process 10 agents concurrently  
# Time: ~8 seconds (6x speedup)
```

## API Reference

### AsyncAgent

```python
class AsyncAgent:
    def __init__(self, max_tool_turns: int = 4, memory_path: str = None)
    async def chat(self, message: str) -> AgentResponse
    async def save_conversation(self, log: bool = False)
```

### Helper Functions

```python
async def run_agents_concurrently(
    agents: list[AsyncAgent], 
    messages: list[str]
) -> list[AgentResponse]
```

### Async Model Functions

```python
async def get_model_response(
    messages: Optional[list[ChatMessage]] = None,
    message: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: str = OPENROUTER_STRONG_MODEL,
    schema: Optional[BaseModel] = None,
) -> Union[str, BaseModel]
```

## Testing

Run tests with pytest:
```bash
# Run all tests
python -m pytest agent/async_agent/test_agents.py -v

# Run only async tests
python -m pytest agent/async_agent/test_agents.py::TestAsyncAgent -v

# Run basic tests
python agent/async_agent/test_agents.py
```

## Examples

See `examples.py` for comprehensive examples including:
- Basic sync and async usage
- Concurrent agent operations
- Mixed sync/async patterns
- Batch processing
- Performance comparisons

## Best Practices

1. **Memory Path Management**: Always use unique memory paths for concurrent agents
2. **Rate Limiting**: Use semaphores to limit concurrent API calls
3. **Error Handling**: Wrap async operations in try/except blocks
4. **Resource Cleanup**: Always clean up memory directories after use
5. **Batch Size**: For optimal performance, batch 5-10 concurrent operations

## Migration Guide

To migrate from sync to async:

1. Import AsyncAgent instead of Agent
2. Add `async` to function definitions
3. Add `await` before agent operations
4. Use `asyncio.run()` to run async functions

```python
# Before (sync)
agent = Agent()
response = agent.chat("Hello")

# After (async)
agent = AsyncAgent()
response = await agent.chat("Hello")
```

## Limitations

- File operations still use sync I/O (wrapped in executors)
- Sandbox execution remains process-based (already non-blocking)
- Some tools may need async versions for optimal performance

## Future Improvements

- Native async file I/O with aiofiles
- Async versions of all tools
- WebSocket support for real-time communication
- Distributed agent coordination 