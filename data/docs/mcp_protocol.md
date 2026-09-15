# Model Context Protocol (MCP) Architecture & Specification

The Model Context Protocol (MCP) is an open standard developed by Anthropic that standardizes how AI applications and agents connect to external data sources, developer tools, and operational environments.

## Architectural Architecture: Client-Server Topology

MCP separates components into:
- **MCP Host**: The coordinating application or agent (e.g., Claude Desktop, IDE extension, backend orchestrator).
- **MCP Client**: Maintains 1:1 connections with individual MCP servers, translating host intent into JSON-RPC messages.
- **MCP Server**: Lightweight process exposing specialized resources, prompts, or executable tools.

## Core Primitives

### 1. Tools
Tools allow LLMs to invoke actions or query dynamic state in the outside world (e.g., query database, execute shell command, create git branch).
- Registered with JSON Schema defining input parameters and return types.
- Executed on explicit client request following an LLM tool call decision.

### 2. Resources
Resources represent passive, readable data akin to file endpoints or database records.
- Identified via standardized URI schemes (e.g., `file:///workspace/repo`, `postgres://db/table`).
- Supports text or binary contents. Can push change notifications when data updates.

### 3. Prompts
Pre-defined, server-managed prompt templates designed to steer LLM conversations for specific tasks (e.g., code review workflow, root-cause bug diagnosis).

## Transports

MCP supports two foundational transport layers:
1. **Standard I/O (`stdio`)**: The MCP Host launches the server as a local subprocess and communicates via newline-delimited JSON-RPC over `stdin` / `stdout`. Fast, secure, and isolated.
2. **Server-Sent Events (`SSE`) over HTTP**: For distributed or remote servers. The client listens to server events via an open SSE stream and posts RPC requests over HTTP POST endpoints.
