# chainlit_app.py — Streaming Chainlit Conversational MCP Agent
#
# PATCH (connection handling): MCP clients are no longer held open for the
# whole chat session. Each operation (tool discovery, tool call) runs inside a
# short `async with client:` block, so fastmcp (re)establishes the session each
# time. A dropped/idle connection now self-heals on the next call instead of
# leaving a dead session that only a full restart could recover.

import asyncio
import json
import re
import tomllib
from pathlib import Path

import chainlit as cl
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import boto3
from botocore.config import Config




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

llm = AsyncOpenAI(
    base_url=config["llm"]["base_url"],
    api_key=config["llm"]["api_key"],
)

LLM_STREAM_TIMEOUT_SECONDS = 300
MCP_DISCOVERY_TIMEOUT_SECONDS = 60
MCP_TOOL_TIMEOUT_SECONDS = 240
S3_FETCH_TIMEOUT_SECONDS = 30
REFRESH_WINDOW_MESSAGE = "resonate:refresh"

MODEL = config["llm"]["model"]
MCP_SERVERS = build_mcp_servers(config)

_storage_cfg = config["storage"]
_s3 = boto3.client(
    "s3",
    endpoint_url=_storage_cfg["endpoint_url"],
    aws_access_key_id=_storage_cfg["access_key_id"],
    aws_secret_access_key=_storage_cfg["secret_access_key"],
    region_name=_storage_cfg.get("region", "eu-central-1"),
    config=Config(s3={"addressing_style": "path"}),
)
S3_BUCKET = _storage_cfg["bucket"]


def fetch_image(image_id: str) -> tuple[bytes, str]:
    obj = _s3.get_object(Bucket=S3_BUCKET, Key=image_id)
    return obj["Body"].read(), obj.get("ContentType", "application/octet-stream")


LOCAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "display_image",
            "description": (
                "Display a stored image to the user in the conversation, by its "
                "image_id (the SHA-256 returned by ingest_image or stored in MongoDB). "
                "Call this whenever the user should actually see an image you found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "SHA-256 key of the stored image."},
                    "caption": {"type": "string", "description": "Optional short caption."},
                },
                "required": ["image_id"],
            },
        },
    }
]


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
        #self.tools = []
        # modified to include the dismplay_image tool as a local tool available to the agent, in addition to any tools discovered from MCP servers
        self.tools = list(LOCAL_TOOL_SCHEMAS)
        self.local_tools = {"display_image": self._display_image}
        self.mcp_clients = []
        self._init_messages()

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
                # Build the client but DO NOT hold the context open.
                # Entering `async with client:` per operation lets fastmcp
                # (re)establish the session each time, so an idle/dropped
                # connection self-heals on the next call rather than leaving
                # a dead session that only an agent restart can fix.
                client = Client(server)

                async with asyncio.timeout(MCP_DISCOVERY_TIMEOUT_SECONDS):
                    async with client:
                        server_tools = await client.list_tools()

                self.mcp_clients.append((server_name, client))

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
                if isinstance(e, TimeoutError):
                    error_text = f"Timed out after {MCP_DISCOVERY_TIMEOUT_SECONDS}s during MCP tool discovery."
                else:
                    error_text = str(e)
                discovered.append(
                    {
                        "server": server_name,
                        "ok": False,
                        "error": error_text,
                        "tools": [],
                    }
                )

        return discovered

    async def disconnect(self):
        # No-op: clients are only entered for the duration of each operation
        # (see connect / chat), never held open, so there is nothing to close.
        self.mcp_clients.clear()

    def _get_client_for_tool(self, tool_name: str):
        for tool in self.tools:
            if tool["function"]["name"] == tool_name:
                return tool.get("_client")
        return None
    
    async def _display_image(self, args: dict) -> str:
        image_id = args.get("image_id")
        caption = args.get("caption") or ""
        if not image_id:
            return "Error: image_id is required."
        try:
            async with asyncio.timeout(S3_FETCH_TIMEOUT_SECONDS):
                data, _ctype = await asyncio.to_thread(fetch_image, image_id)
        except TimeoutError:
            return f"Error: timed out after {S3_FETCH_TIMEOUT_SECONDS}s while fetching image {image_id}."
        except Exception as e:
            return f"Error: could not fetch image {image_id}: {e}"
        img = cl.Image(name=image_id, content=data, display="inline")
        await cl.Message(content=caption, elements=[img]).send()
        return f"Displayed image {image_id} to the user."

    def _tools_for_api(self):
        return [
            {"type": t["type"], "function": t["function"]}
            for t in self.tools
        ]

    async def chat(self, user_message: str, response_msg: cl.Message) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict] = {}

            try:
                async with asyncio.timeout(LLM_STREAM_TIMEOUT_SECONDS):
                    stream = await llm.chat.completions.create(
                        model=MODEL,
                        messages=self.messages,
                        tools=self._tools_for_api() if self.tools else None,
                        tool_choice="auto" if self.tools else None,
                        stream=True,
                    )

                    async for chunk in stream:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        if delta.content:
                            content_parts.append(delta.content)

                            # We stream raw content for responsiveness.
                            # If the model emits <think>...</think>, it will be cleaned after completion.
                            await response_msg.stream_token(delta.content)

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                index = tc.index

                                if index not in tool_calls_by_index:
                                    tool_calls_by_index[index] = {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": "",
                                        },
                                    }

                                if tc.id:
                                    tool_calls_by_index[index]["id"] = tc.id

                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_by_index[index]["function"]["name"] += tc.function.name

                                    if tc.function.arguments:
                                        tool_calls_by_index[index]["function"]["arguments"] += tc.function.arguments
            except TimeoutError as e:
                raise RuntimeError(
                    f"Timed out after {LLM_STREAM_TIMEOUT_SECONDS}s while waiting for the model response."
                ) from e

            full_content = "".join(content_parts)
            tool_calls = list(tool_calls_by_index.values())

            if tool_calls:
                # Clean any partial streamed content before showing tool execution.
                response_msg.content = strip_thinking(full_content)
                await response_msg.update()

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": tool_calls,
                    }
                )

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]

                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError as e:
                        result_text = f"Error: invalid JSON arguments: {e}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": result_text,
                            }
                        )
                        continue
                    
                    if tool_name in self.local_tools:
                        try:
                            result_text = await self.local_tools[tool_name](tool_args)
                        except Exception as e:
                            result_text = f"Error: {e}"
                        self.messages.append(
                            {"role": "tool", "tool_call_id": tool_call["id"], "content": result_text}
                        )
                        continue
                    
                    async with cl.Step(name=f"Tool: {tool_name}") as step:
                        step.input = json.dumps(tool_args, indent=2)

                        client = self._get_client_for_tool(tool_name)

                        if client:
                            try:
                                # Open the session just for this call. fastmcp
                                # reconnects on context entry, so if the previous
                                # session was dropped (idle timeout, gateway
                                # session invalidation, ~60s SSE termination,
                                # etc.) this call transparently re-establishes it.
                                async with asyncio.timeout(MCP_TOOL_TIMEOUT_SECONDS):
                                    async with client:
                                        result = await client.call_tool(tool_name, tool_args)

                                if hasattr(result, "content"):
                                    result_text = str(result.content)
                                else:
                                    result_text = str(result)

                            except TimeoutError:
                                result_text = f"Error: tool '{tool_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS}s."
                            except Exception as e:
                                result_text = f"Error: {e}"
                        else:
                            result_text = f"Error: Tool '{tool_name}' not found"

                        step.output = result_text[:4000]

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_text,
                        }
                    )

                # New response message for the final answer after tool calls.
                response_msg = cl.Message(content="")
                await response_msg.send()

                continue

            final_response = strip_thinking(full_content)

            response_msg.content = final_response
            await response_msg.update()

            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_content,
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
        await cl.Message(
            content="Conversation history cleared. Humanity gets another chance."
        ).send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        await agent.chat(user_text, response_msg=response_msg)
        await cl.send_window_message(REFRESH_WINDOW_MESSAGE)

    except Exception as e:
        error_text = str(e) or e.__class__.__name__
        response_msg.content = f"Error while processing message:\n\n```text\n{error_text}\n```"
        await response_msg.update()


@cl.on_chat_end
async def on_chat_end():
    agent: ChatAgent | None = cl.user_session.get("agent")

    if agent is not None:
        await agent.disconnect()
