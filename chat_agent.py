# chat_agent.py — Conversational MCP Agent
#
# A chat agent that:
# 1. Discovers available MCP servers and their tools
# 2. Maintains conversation history
# 3. Decides when to use tools based on user needs
# 4. Returns tool results to the LLM for natural responses

import asyncio
import json
import re
import tomllib
from pathlib import Path
from openai import OpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def load_config():
    """Load configuration from config.toml."""
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Copy config.example.toml to config.toml and edit it with your settings."
        )
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_mcp_servers(config):
    """Build MCP server list from configuration."""
    servers = []
    for server in config.get("mcp_servers", []):
        server_type = server.get("type", "stdio")
        if server_type == "stdio":
            servers.append(StdioTransport(
                command=server["command"],
                args=server.get("args", []),
                cwd=server.get("cwd")
            ))
        elif server_type == "sse":
            servers.append(server["url"])
        elif server_type == "file":
            servers.append(server["path"])
    return servers


# Load configuration
config = load_config()
llm = OpenAI(
    base_url=config["llm"]["base_url"],
    api_key=config["llm"]["api_key"]
)
MODEL = config["llm"]["model"]
MCP_SERVERS = build_mcp_servers(config)


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output.
    
    Handles both:
    - <think>...</think> (proper tags)
    - ......</think> (missing opening tag)
    """
    if text is None:
        return ""
    # First try: proper <think>...</think> tags
    result = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Second try: everything before </think> if no opening tag
    result = re.sub(r'^.*?</think>\s*', '', result, flags=re.DOTALL)
    return result.strip()


class ChatAgent:
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.messages = []
        self._init_messages()
        self.tools = []
        self.mcp_clients = []

    def _init_messages(self):
        """Initialize or reset messages with system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def clear_history(self):
        """Clear conversation history while preserving system prompt."""
        self._init_messages()
    
    async def connect(self):
        """Connect to all MCP servers and discover their tools."""
        print("Discovering MCP servers and tools...\n")
        
        for server in MCP_SERVERS:
            # Server can be a string, StdioTransport, or other transport
            if isinstance(server, str):
                server_name = server
            elif isinstance(server, StdioTransport):
                server_name = f"{server.command} {' '.join(server.args)}"
            else:
                server_name = str(server)
            
            try:
                client = Client(server)
                await client.__aenter__()
                self.mcp_clients.append((server_name, client))
                
                # Get tools from this server
                server_tools = await client.list_tools()
                for t in server_tools:
                    self.tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "No description",
                            "parameters": t.inputSchema or {"type": "object", "properties": {}}
                        },
                        "_client": client  # Keep reference to call it later
                    })
                print(f"  [OK] {server_name}")
                for t in server_tools:
                    print(f"      - {t.name}")
            except Exception as e:
                print(f"  [FAIL] {server_name}: {e}")
        
        print(f"\nTotal: {len(self.tools)} tools from {len(self.mcp_clients)} servers\n")
    
    async def disconnect(self):
        """Disconnect from all MCP servers."""
        for server_name, client in self.mcp_clients:
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
    
    def _get_client_for_tool(self, tool_name: str):
        """Find which MCP client has this tool."""
        for tool in self.tools:
            if tool["function"]["name"] == tool_name:
                return tool["_client"]
        return None
    
    def _tools_for_api(self):
        """Return tools in API format (without internal _client reference)."""
        return [
            {"type": t["type"], "function": t["function"]} 
            for t in self.tools
        ]
    
    async def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Agentic loop: keep going until LLM gives final answer
        while True:
            # Call the LLM
            response = llm.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self._tools_for_api() if self.tools else None,
                tool_choice="auto" if self.tools else None
            )
            
            assistant_message = response.choices[0].message
            
            # Check if LLM wants to use tools
            if assistant_message.tool_calls:
                # Add assistant's tool request to history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    
                    print(f"  [TOOL] {tool_name}({json.dumps(tool_args)})")
                    
                    # Find the right MCP client and call the tool
                    client = self._get_client_for_tool(tool_name)
                    if client:
                        try:
                            result = await client.call_tool(tool_name, tool_args)
                            # Handle different result types
                            if hasattr(result, 'content'):
                                result_text = str(result.content)
                            else:
                                result_text = str(result)
                            print(f"     → {result_text[:80]}{'...' if len(result_text) > 80 else ''}")
                        except Exception as e:
                            result_text = f"Error: {e}"
                            print(f"     → Error: {e}")
                    else:
                        result_text = f"Error: Tool '{tool_name}' not found"
                    
                    # Add tool result to history
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })
                
                # Continue the loop - LLM will process tool results
                continue
            
            else:
                # No tool calls - we have a final response
                final_response = strip_thinking(assistant_message.content)
                self.messages.append({
                    "role": "assistant", 
                    "content": assistant_message.content  # Keep original in history
                })
                return final_response


async def main():
    system_prompt = config["llm"].get("system_prompt")
    agent = ChatAgent(system_prompt=system_prompt)
    await agent.connect()
    
    print("=" * 50)
    print("Chat Agent Ready!")
    print("Type 'quit' to exit, 'clear' to reset history")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
                
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            if user_input.lower() == 'clear':
                agent.clear_history()
                print("(conversation cleared)")
                continue
            if not user_input:
                continue
            
            response = await agent.chat(user_input)
            print(f"\nAssistant: {response}")
    
    finally:
        await agent.disconnect()
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
