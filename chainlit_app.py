# chainlit_app.py — Chainlit Conversational MCP Agent

import json
import re
import tomllib
from pathlib import Path

import chainlit as cl
from openai import OpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def load_config():
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Copy config.example.toml to config.toml and edit it with your settings."
        )

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_mcp_servers(config):
    servers = []

    for server in config.get("mcp_servers", []):
        server_type = server.get("type", "stdio")

        if server_type == "stdio":
            servers.append(
                StdioTransport(
                    command=server["command"],
                    args=server.get("args", []),
                    cwd=server.get("cwd"),
                    env=server.get("env"),
                )
            )

        elif server_type == "sse":
            servers.append(server["url"])

        elif server_type == "file":
            servers.append(server["path"])

    return servers


config = load_config()

llm = OpenAI(
    base_url=config["llm"]["base_url"],
    api_key=config["llm"]["api_key"],
)

MODEL = config["llm"]["model"]
MCP_SERVERS = build_mcp_servers(config)


def strip_thinking(text: str) -> str:
    if text is None:
        return ""

    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)

    return result.strip()


class ChatAgent:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt
        self.messages = []
        self._init_messages()
        self.tools = []
        self.mcp_clients = []

    def _init_messages(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )

    def clear_history(self):
        self._init_messages()

    async def connect(self):
        discovered = []

        for server in MCP_SERVERS:
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

                server_tools = await client.list_tools()

                for t in server_tools:
                    self.tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "No description",
                                "parameters": t.inputSchema
                                or {"type": "object", "properties": {}},
                            },
                            "_client": client,
                        }
                    )

                discovered.append(
                    {
                        "server": server_name,
                        "ok": True,
                        "tools": [t.name for t in server_tools],
                    }
                )

            except Exception as e:
                discovered.append(
                    {
                        "server": server_name,
                        "ok": False,
                        "error": str(e),
                        "tools": [],
                    }
                )

        return discovered

    async def disconnect(self):
        for _, client in self.mcp_clients:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass

    def _get_client_for_tool(self, tool_name: str):
        for tool in self.tools:
            if tool["function"]["name"] == tool_name:
                return tool["_client"]
        return None

    def _tools_for_api(self):
        return [
            {"type": t["type"], "function": t["function"]}
            for t in self.tools
        ]

    async def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = llm.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self._tools_for_api() if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name

                    try:
                        tool_args = (
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        )
                    except json.JSONDecodeError as e:
                        result_text = f"Error: invalid JSON arguments: {e}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text,
                            }
                        )
                        continue

                    async with cl.Step(name=f"Tool: {tool_name}") as step:
                        step.input = json.dumps(tool_args, indent=2)

                        client = self._get_client_for_tool(tool_name)

                        if client:
                            try:
                                result = await client.call_tool(tool_name, tool_args)

                                if hasattr(result, "content"):
                                    result_text = str(result.content)
                                else:
                                    result_text = str(result)

                            except Exception as e:
                                result_text = f"Error: {e}"
                        else:
                            result_text = f"Error: Tool '{tool_name}' not found"

                        step.output = result_text[:4000]

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )

                continue

            final_response = strip_thinking(assistant_message.content)

            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                }
            )

            return final_response


@cl.on_chat_start
async def on_chat_start():
    system_prompt = config["llm"].get("system_prompt")

    agent = ChatAgent(system_prompt=system_prompt)

    msg = cl.Message(content="Discovering MCP servers and tools...")
    await msg.send()

    discovered = await agent.connect()

    cl.user_session.set("agent", agent)

    lines = ["# Chat Agent Ready", ""]

    ok_count = 0
    tool_count = 0

    for item in discovered:
        if item["ok"]:
            ok_count += 1
            tool_count += len(item["tools"])
            lines.append(f"✅ **{item['server']}**")
            for tool in item["tools"]:
                lines.append(f"- `{tool}`")
        else:
            lines.append(f"❌ **{item['server']}**")
            lines.append(f"- Error: `{item['error']}`")

        lines.append("")

    lines.append(f"**Total:** {tool_count} tools from {ok_count} MCP server(s).")
    lines.append("")
    lines.append("Type `/clear` to reset the conversation history.")

    msg.content = "\n".join(lines)
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    agent: ChatAgent | None = cl.user_session.get("agent")

    if agent is None:
        await cl.Message(
            content="No agent session found. Refresh the page to start a new session."
        ).send()
        return

    user_text = message.content.strip()

    if user_text.lower() in {"/clear", "clear"}:
        agent.clear_history()
        await cl.Message(content="Conversation history cleared. Humanity gets another chance.").send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        response = await agent.chat(user_text)
        response_msg.content = response
        await response_msg.update()

    except Exception as e:
        response_msg.content = f"Error while processing message:\n\n```text\n{e}\n```"
        await response_msg.update()


@cl.on_chat_end
async def on_chat_end():
    agent: ChatAgent | None = cl.user_session.get("agent")

    if agent is not None:
        await agent.disconnect()
