# Chat Agent

A basic implementation of a conversational agent that can connect to MCP (Model Context Protocol) servers and use their tools.

## Features

- Discovers and connects to multiple MCP servers
- Maintains conversation history across interactions
- Agentic loop that handles tool calls and processes results
- Compatible with OpenAI API format (works with local LLMs via Ollama, LM Studio, etc.)
- Supports multiple MCP transport types: local Python files, stdio commands, and SSE URLs

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chat_agent

# Install dependencies
uv sync
```

## Configuration

Copy the example configuration file and edit it with your settings:

```bash
cp config.example.toml config.toml
```

### LLM Settings

```toml
[llm]
base_url = "http://localhost:11434/v1"  # Ollama, LM Studio, etc.
api_key = "ollama"
model = "qwen2.5"
```

### MCP Servers

```toml
[[mcp_servers]]
type = "stdio"
command = "uv"
args = ["run", "server.py"]
cwd = "/path/to/server"

[[mcp_servers]]
type = "sse"
url = "http://localhost:8000/sse"
```

## Usage

```bash
uv run chat_agent.py
```

The agent will:
1. Connect to all configured MCP servers
2. Discover available tools
3. Start an interactive chat session

### Commands

- Type your message and press Enter to chat
- `clear` - Reset conversation history
- `quit` or `exit` - Exit the agent

## Dependencies

- `fastmcp` - MCP client library
- `openai` - OpenAI API client (used for LLM communication)
