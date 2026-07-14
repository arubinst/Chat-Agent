# chainlit_app.py — Streaming Chainlit Conversational MCP Agent
#
# PATCH (connection handling): MCP clients are no longer held open for the
# whole chat session. Each operation (tool discovery, tool call) runs inside a
# short `async with client:` block, so fastmcp (re)establishes the session each
# time. A dropped/idle connection now self-heals on the next call instead of
# leaving a dead session that only a full restart could recover.

import asyncio
import html
import io
import ipaddress
import json
import re
import socket
import tomllib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from datetime import datetime, timezone

import chainlit as cl
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import (
    StdioTransport,
    StreamableHttpTransport,
)

import boto3
from botocore.config import Config
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PdfImage, Paragraph, SimpleDocTemplate, Spacer




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
        elif server_type == "streamable-http":
            servers.append(
                StreamableHttpTransport(
                    url=server["url"],
                    headers={
                        "Authorization": (
                            f"Bearer {os.environ['MCP_GATEWAY_AUTH_TOKEN']}"
                        ),
                    },
                )
            )
        elif server_type == "sse":
            servers.append(server["url"])
        elif server_type == "file":
            servers.append(server["path"])
        else:
            raise ValueError(
                f"Unsupported MCP server type: {server_type}"
            )
    return servers


config = load_config()

LLM_STREAM_TIMEOUT_SECONDS = 300
ANTHROPIC_MAX_TOKENS = 4096
MCP_DISCOVERY_TIMEOUT_SECONDS = 60
MCP_TOOL_TIMEOUT_SECONDS = 240
S3_FETCH_TIMEOUT_SECONDS = 30
REFRESH_WINDOW_MESSAGE = "resonate:refresh"
PDF_IMAGE_MAX_BYTES = 10 * 1024 * 1024
PDF_IMAGE_MAX_PIXELS = (1400, 1400)
REMOTE_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", re.IGNORECASE)

MODEL = config["llm"]["model"]
DEFAULT_LLM_PROVIDER = config["llm"].get("provider", "openai-compatible")
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


def validate_llm_endpoint(endpoint: str) -> str:
    """Validate and normalize an OpenAI-compatible API base URL."""
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Endpoint must be a complete HTTP(S) URL, for example "
            "https://api.example.com/v1."
        )

    return normalized


def restore_config_defaults(agent: "ChatAgent"):
    """Restore all runtime LLM settings, including the non-displayed config key."""
    agent.update_llm_connection(
        config["llm"]["base_url"],
        config["llm"]["api_key"],
        config["llm"]["model"],
        DEFAULT_LLM_PROVIDER,
    )


def _is_public_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (OSError, ValueError):
        return False


def fetch_remote_image(url: str) -> bytes:
    """Download one public image for PDF embedding, with SSRF and size limits."""
    if not _is_public_image_url(url):
        raise ValueError("Image URL is not a public HTTP(S) address.")

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    response = build_opener(NoRedirect).open(
        Request(url, headers={"User-Agent": "Chat-Agent PDF exporter"}), timeout=10
    )
    content_type = response.headers.get_content_type()
    if not content_type.startswith("image/"):
        raise ValueError("URL did not return an image.")
    data = response.read(PDF_IMAGE_MAX_BYTES + 1)
    if len(data) > PDF_IMAGE_MAX_BYTES:
        raise ValueError("Image exceeds the 10 MB export limit.")
    return data


def optimize_pdf_image(data: bytes) -> io.BytesIO:
    """Convert images to page-friendly JPEG data embedded within the PDF."""
    with PILImage.open(io.BytesIO(data)) as image:
        image.thumbnail(PDF_IMAGE_MAX_PIXELS)
        if image.mode in {"RGBA", "LA", "P"}:
            background = PILImage.new("RGB", image.size, "white")
            if image.mode != "P":
                background.paste(image, mask=image.getchannel("A"))
            else:
                background.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
        output.seek(0)
        return output


def _pdf_image(data: bytes, available_width: float):
    image_data = optimize_pdf_image(data)
    with PILImage.open(image_data) as image:
        width, height = image.size
    scale = min(available_width / width, 5 * inch / height, 1)
    return PdfImage(io.BytesIO(image_data.getvalue()), width=width * scale, height=height * scale)


def _pdf_paragraphs(text: str, style: ParagraphStyle):
    text = REMOTE_IMAGE_PATTERN.sub("", text)
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", escaped)
    return [Paragraph(part.replace("\n", "<br/>") or " ", style) for part in escaped.split("\n\n")]


def build_chat_pdf(entries: list[dict], mode: str) -> bytes:
    """Build an in-memory PDF; it is never stored in the application's S3 bucket."""
    if mode == "summary":
        assistant_entries = [entry for entry in entries if entry["kind"] == "assistant"]
        if not assistant_entries:
            raise ValueError("There is no completed assistant response to export.")
        entries = [assistant_entries[-1]]
    elif not entries:
        raise ValueError("There is no conversation to export.")

    output = io.BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ExportBody", parent=styles["BodyText"], leading=15, spaceAfter=8)
    tool = ParagraphStyle("Tool", parent=body, textColor=colors.HexColor("#555555"), leftIndent=12)
    story = [Paragraph("Chat Agent Export", styles["Title"])]
    story.append(Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 16))
    for entry in entries:
        kind = entry["kind"]
        if kind == "tool":
            story.append(Paragraph(f"<b>Used Tool:</b> {html.escape(entry['name'])}", tool))
            continue
        heading = "You" if kind == "user" else "Assistant"
        story.append(Paragraph(heading, styles["Heading2"]))
        story.extend(_pdf_paragraphs(entry["content"], body))
        image_sources = [("remote", url, "") for url in REMOTE_IMAGE_PATTERN.findall(entry["content"])]
        image_sources.extend(("s3", image["image_id"], image["caption"]) for image in entry.get("images", []))
        for source_type, source, caption in image_sources:
            try:
                data = fetch_image(source)[0] if source_type == "s3" else fetch_remote_image(source)
                story.append(_pdf_image(data, document.width))
                if caption:
                    story.append(Paragraph(html.escape(caption), styles["Italic"]))
            except Exception as exc:
                label = caption or source
                story.append(Paragraph(f"[Image unavailable: {html.escape(label)}]", tool))
        story.append(Spacer(1, 12))
    document.build(story)
    return output.getvalue()


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
    def __init__(
        self,
        system_prompt: str | None = None,
        base_url: str = config["llm"]["base_url"],
        api_key: str = config["llm"]["api_key"],
        model: str = MODEL,
        provider: str = DEFAULT_LLM_PROVIDER,
    ):
        if provider not in {"openai-compatible", "anthropic"}:
            raise ValueError(
                "llm.provider must be either 'openai-compatible' or 'anthropic'."
            )
        self.system_prompt = system_prompt
        self.provider = provider
        self.base_url = validate_llm_endpoint(base_url)
        self.api_key = api_key
        self.model = model
        self.llm = self._build_llm_client()
        self.messages = []
        self.anthropic_messages = []
        #self.tools = []
        # modified to include the dismplay_image tool as a local tool available to the agent, in addition to any tools discovered from MCP servers
        self.tools = list(LOCAL_TOOL_SCHEMAS)
        self.local_tools = {"display_image": self._display_image}
        self.mcp_clients = []
        self.export_entries: list[dict] = []
        self._current_response_images: list[dict] = []
        self._init_messages()

    def _init_messages(self):
        self.messages = []
        self.anthropic_messages = []
        if self.system_prompt:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )

    def clear_history(self):
        self._init_messages()
        self.export_entries.clear()
        self._current_response_images.clear()

    def _build_llm_client(self):
        if self.provider == "anthropic":
            return AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def update_llm_connection(self, base_url: str, api_key: str, model: str, provider: str):
        """Replace this session's LLM settings without changing shared config."""
        model = model.strip()
        if not model:
            raise ValueError("Model must not be empty.")
        if provider not in {"openai-compatible", "anthropic"}:
            raise ValueError("Unsupported LLM provider.")

        self.base_url = validate_llm_endpoint(base_url)
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.llm = self._build_llm_client()
        self.clear_history()

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
        self._current_response_images.append({"image_id": image_id, "caption": caption})
        return f"Displayed image {image_id} to the user."

    def _tools_for_api(self):
        return [
            {"type": t["type"], "function": t["function"]}
            for t in self.tools
        ]

    def _tools_for_anthropic(self):
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in self.tools
        ]

    async def _anthropic_response(self, response_msg: cl.Message):
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        assistant_blocks: list[dict] = []
        async with asyncio.timeout(LLM_STREAM_TIMEOUT_SECONDS):
            async with self.llm.messages.stream(
                model=self.model,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                system=self.system_prompt or "",
                messages=self.anthropic_messages,
                tools=self._tools_for_anthropic() if self.tools else [],
                tool_choice={"type": "auto"} if self.tools else None,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_calls[event.index] = {
                                "id": block.id,
                                "type": "function",
                                "function": {"name": block.name, "arguments": ""},
                            }
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            content_parts.append(delta.text)
                            await response_msg.stream_token(delta.text)
                        elif delta.type == "input_json_delta":
                            tool_calls[event.index]["function"]["arguments"] += delta.partial_json
        for index in sorted(tool_calls):
            call = tool_calls[index]
            raw_args = call["function"]["arguments"] or "{}"
            try:
                tool_input = json.loads(raw_args)
            except json.JSONDecodeError:
                tool_input = {}
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": tool_input,
                }
            )
        if content_parts:
            assistant_blocks.insert(0, {"type": "text", "text": "".join(content_parts)})
        return "".join(content_parts), list(tool_calls.values()), assistant_blocks

    async def _chat_anthropic(self, response_msg: cl.Message) -> str:
        while True:
            try:
                full_content, tool_calls, assistant_blocks = await self._anthropic_response(response_msg)
            except TimeoutError as e:
                raise RuntimeError(
                    f"Timed out after {LLM_STREAM_TIMEOUT_SECONDS}s while waiting for the model response."
                ) from e

            if tool_calls:
                response_msg.content = full_content
                await response_msg.update()
                self.anthropic_messages.append({"role": "assistant", "content": assistant_blocks})
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError as e:
                        result_text = f"Error: invalid JSON arguments: {e}"
                    else:
                        if tool_name in self.local_tools:
                            try:
                                result_text = await self.local_tools[tool_name](tool_args)
                            except Exception as e:
                                result_text = f"Error: {e}"
                        else:
                            async with cl.Step(name=f"Tool: {tool_name}") as step:
                                step.input = json.dumps(tool_args, indent=2)
                                client = self._get_client_for_tool(tool_name)
                                if client:
                                    try:
                                        async with asyncio.timeout(MCP_TOOL_TIMEOUT_SECONDS):
                                            async with client:
                                                result = await client.call_tool(tool_name, tool_args)
                                        result_text = str(result.content) if hasattr(result, "content") else str(result)
                                    except TimeoutError:
                                        result_text = f"Error: tool '{tool_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS}s."
                                    except Exception as e:
                                        result_text = f"Error: {e}"
                                else:
                                    result_text = f"Error: Tool '{tool_name}' not found"
                                step.output = result_text[:4000]
                            self.export_entries.append({"kind": "tool", "name": tool_name})
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": tool_call["id"], "content": result_text}
                    )
                self.anthropic_messages.append({"role": "user", "content": tool_results})
                response_msg = cl.Message(content="")
                await response_msg.send()
                continue

            response_msg.content = full_content
            await response_msg.update()
            self.anthropic_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": full_content}]}
            )
            self.export_entries.append(
                {"kind": "assistant", "content": full_content, "images": list(self._current_response_images)}
            )
            return full_content

    async def chat(self, user_message: str, response_msg: cl.Message) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self.anthropic_messages.append({"role": "user", "content": user_message})
        self.export_entries.append({"kind": "user", "content": user_message})
        self._current_response_images = []

        if self.provider == "anthropic":
            return await self._chat_anthropic(response_msg)

        while True:
            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict] = {}

            try:
                async with asyncio.timeout(LLM_STREAM_TIMEOUT_SECONDS):
                    stream = await self.llm.chat.completions.create(
                        model=self.model,
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

                    self.export_entries.append({"kind": "tool", "name": tool_name})

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
            self.export_entries.append(
                {
                    "kind": "assistant",
                    "content": final_response,
                    "images": list(self._current_response_images),
                }
            )

            return final_response


async def send_pdf_export(agent: ChatAgent, mode: str):
    pdf_data = await asyncio.to_thread(build_chat_pdf, agent.export_entries, mode)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    export_file = cl.File(
        name=f"chat-{mode}-{timestamp}.pdf",
        content=pdf_data,
        mime="application/pdf",
        display="inline",
    )
    await cl.Message(
        content=f"Your {mode}-only PDF export is ready.", elements=[export_file]
    ).send()


@cl.on_chat_start
async def on_chat_start():
    system_prompt = config["llm"].get("system_prompt")
    agent = ChatAgent(
        system_prompt=system_prompt,
        provider=DEFAULT_LLM_PROVIDER,
    )
    cl.user_session.set("agent", agent)

    settings = cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="llm_provider",
                label="LLM provider",
                initial=agent.provider,
                items={
                    "openai-compatible": "OpenAI-compatible",
                    "anthropic": "Anthropic / Claude",
                },
                description="Select Anthropic / Claude only for the native Anthropic Messages API.",
            ),
            cl.input_widget.TextInput(
                id="llm_endpoint",
                label="LLM endpoint",
                initial=agent.base_url,
                description=(
                    "Base URL used for this browser session. For native Anthropic, "
                    "use https://api.anthropic.com (without /v1)."
                ),
                placeholder="https://api.example.com/v1",
            ),
            cl.input_widget.TextInput(
                id="llm_api_key",
                label="LLM API key",
                initial="",
                description=(
                    "Enter a key to replace the current session key. Leave blank "
                    "to keep the configured key. This field is not prefilled."
                ),
                placeholder="Leave blank to retain the current key",
            ),
            cl.input_widget.TextInput(
                id="llm_model",
                label="Model",
                initial=agent.model,
                description="Model name used for this browser session.",
                placeholder="gemma4:latest",
            ),
        ]
    )
    await settings.send()

    msg = cl.Message(content="Discovering MCP servers and tools...")
    await msg.send()

    discovered = await agent.connect()

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
    lines.append("Use `/export-pdf summary` or `/export-pdf full` to download a PDF.")

    msg.content = "\n".join(lines)
    await msg.update()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    agent: ChatAgent | None = cl.user_session.get("agent")

    if agent is None:
        await cl.Message(
            content="No agent session found. Refresh the page to start a new session."
        ).send()
        return

    endpoint = str(settings.get("llm_endpoint", ""))
    submitted_key = str(settings.get("llm_api_key", ""))
    api_key = submitted_key.strip() or agent.api_key
    model = str(settings.get("llm_model", ""))
    provider = str(settings.get("llm_provider", ""))

    try:
        is_config_reset = (
            provider == DEFAULT_LLM_PROVIDER
            and validate_llm_endpoint(endpoint) == validate_llm_endpoint(config["llm"]["base_url"])
            and model.strip() == config["llm"]["model"]
            and not submitted_key.strip()
        )
        if is_config_reset:
            restore_config_defaults(agent)
        else:
            agent.update_llm_connection(endpoint, api_key, model, provider)
    except (TypeError, ValueError) as e:
        await cl.Message(content=f"Could not update LLM connection: {e}").send()
        return

    await cl.Message(
        content=(
            "LLM configuration updated for this session. Conversation history was "
            "cleared; the API key is not displayed or saved to config.toml."
        )
    ).send()


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

    export_match = re.fullmatch(r"/export-pdf(?:\s+(summary|full))?", user_text, re.IGNORECASE)
    if export_match:
        mode = (export_match.group(1) or "full").lower()
        try:
            await send_pdf_export(agent, mode)
        except Exception as e:
            await cl.Message(content=f"Could not create PDF export: {e}").send()
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
