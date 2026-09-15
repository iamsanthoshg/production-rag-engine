# FastAPI Architectural Patterns & Best Practices

FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Core Architectural Components

### 1. Asynchronous Request Handling (ASGI)
FastAPI runs on top of Starlette and Uvicorn. When endpoints are defined using `async def`, FastAPI schedules their execution directly on the Python `asyncio` event loop.
- **Async Endpoints**: Should be used for non-blocking I/O operations such as querying external databases, calling third-party LLM APIs, or performing non-blocking file operations.
- **Sync Endpoints**: If an endpoint is declared as standard `def` (synchronous), FastAPI automatically offloads it to a separate worker thread pool (`anyio.to_thread`) to prevent blocking the main event loop.

### 2. Dependency Injection System
FastAPI features a hierarchical Dependency Injection (DI) system managed via `Depends()`.
- **Reusable Logic**: Common operations like database session lifecycle (`get_db`), user authentication, and rate limiting can be defined as dependency functions.
- **Sub-dependencies**: Dependencies can depend on other dependencies, building an execution tree.
- **Yield Dependencies**: Dependencies utilizing `yield` allow setup logic before the endpoint runs and teardown/cleanup logic after the response is sent (e.g., closing database connections).

```python
async def get_db_session():
    session = await SessionLocal()
    try:
        yield session
    finally:
        await session.close()
```

### 3. Lifespan Events
FastAPI uses ASGI lifespan state rather than legacy `startup` and `shutdown` event handlers:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models, warm caches, connect to DB
    yield
    # Shutdown: Flush buffers, close connection pools
```

### 4. Pydantic Data Validation
Request payloads and response serialization rely on Pydantic `BaseModel`. FastAPI automatically generates OpenAPI 3.0 documentation and validates incoming JSON against the model schema, rejecting malformed data with detailed 422 HTTP responses.
