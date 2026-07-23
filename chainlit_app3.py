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
import mistune
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import (
    StdioTransport,
    StreamableHttpTransport,
)

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as PdfImage,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from resonate_images import read_resonate_overlay




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
LLM_TIMEOUT_DECISION_SECONDS = 600
ANTHROPIC_MAX_TOKENS = 4096
MCP_DISCOVERY_TIMEOUT_SECONDS = 60
MCP_TOOL_TIMEOUT_SECONDS = 240
# Some serving stacks (e.g. vLLM's GLM tool-call parser) drop streamed tool
# calls whose arguments are empty: streaming deltas are only emitted while
# argument tokens arrive, and a call with `{}` arguments produces none. Any
# tool without required parameters can be legally called with `{}`, so those
# get one required pad parameter to guarantee argument tokens; it is stripped
# again before the real MCP call.
ZERO_ARG_PAD_PARAM = "call_reason"
ZERO_ARG_PAD_SCHEMA = {
    "type": "string",
    "description": "One short sentence explaining why you are calling this tool.",
}
# Set DEBUG_LLM_STREAM=1 to log one line per streamed chunk to stdout.
DEBUG_LLM_STREAM = os.environ.get("DEBUG_LLM_STREAM", "") not in {"", "0", "false"}


def _debug_stream(message: str):
    if DEBUG_LLM_STREAM:
        print(f"[llm-stream {datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}] {message}", flush=True)


REFRESH_WINDOW_MESSAGE = "resonate:refresh"
PDF_IMAGE_MAX_BYTES = 10 * 1024 * 1024
PDF_IMAGE_MAX_PIXELS = (1400, 1400)
RESONATE_IMAGES_DIR = os.environ.get("RESONATE_IMAGES_DIR")
RESONATE_PUBLIC_BASE_URL = os.environ.get("RESONATE_PUBLIC_BASE_URL")
REMOTE_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", re.IGNORECASE)
MARKDOWN_AST = mistune.create_markdown(renderer="ast", plugins=["table"])


class LLMResponseTimeout(RuntimeError):
    """Raised when one model-response attempt exceeds its allowed duration."""

# GLM-5.x on vLLM honors only "high" (balanced) and treats every other value
# as "max" (deepest, the default); "none" disables thinking entirely via
# chat_template_kwargs. "low"/"medium" are offered for non-GLM endpoints.
REASONING_EFFORT_CHOICES = {"default", "none", "low", "medium", "high", "max"}

MODEL = config["llm"]["model"]
DEFAULT_LLM_PROVIDER = config["llm"].get("provider", "openai-compatible")
DEFAULT_REASONING_EFFORT = config["llm"].get("reasoning_effort", "default")
SHOW_MCP_TOOL_LIST = config.get("display", {}).get("show_mcp_tool_list", False)
MCP_SERVERS = build_mcp_servers(config)

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


def normalize_llm_provider(provider: str) -> str:
    aliases = {
        "openai-compatible": "openai-compatible",
        "anthropic": "anthropic",
        "anthropic / claude": "anthropic",
    }
    normalized = aliases.get(provider.strip().lower())
    if normalized is None:
        raise ValueError("Unsupported LLM provider.")
    return normalized


def normalize_reasoning_effort(value: str) -> str:
    normalized = (value or "").strip().lower() or "default"
    if normalized not in REASONING_EFFORT_CHOICES:
        raise ValueError(
            "Reasoning effort must be one of: "
            + ", ".join(sorted(REASONING_EFFORT_CHOICES))
        )
    return normalized


def normalize_llm_endpoint(endpoint: str, provider: str) -> str:
    normalized = validate_llm_endpoint(endpoint)
    parsed = urlparse(normalized)
    # The Anthropic SDK appends /v1/messages itself. Accept the API endpoint
    # commonly pasted from Anthropic's documentation as a convenience.
    if (
        provider == "anthropic"
        and parsed.hostname == "api.anthropic.com"
        and parsed.path.rstrip("/") == "/v1/messages"
    ):
        return f"{parsed.scheme}://{parsed.netloc}"
    return normalized


def restore_config_defaults(agent: "ChatAgent"):
    """Restore all runtime LLM settings, including the non-displayed config key."""
    agent.update_llm_connection(
        config["llm"]["base_url"],
        config["llm"]["api_key"],
        config["llm"]["model"],
        DEFAULT_LLM_PROVIDER,
        DEFAULT_REASONING_EFFORT,
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
    """Read a trusted local overlay or download one public image for PDF export."""
    resonate_overlay = read_resonate_overlay(
        url,
        public_base_url=RESONATE_PUBLIC_BASE_URL,
        images_dir=RESONATE_IMAGES_DIR,
        max_bytes=PDF_IMAGE_MAX_BYTES,
    )
    if resonate_overlay is not None:
        return resonate_overlay

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


def _pdf_inline(children: list[dict] | None) -> str:
    """Convert Mistune inline tokens to the small XML subset ReportLab supports."""
    parts = []
    for token in children or []:
        token_type = token["type"]
        if token_type == "text":
            parts.append(html.escape(token.get("raw", "")))
        elif token_type == "strong":
            parts.append(f"<b>{_pdf_inline(token.get('children'))}</b>")
        elif token_type == "emphasis":
            parts.append(f"<i>{_pdf_inline(token.get('children'))}</i>")
        elif token_type == "codespan":
            parts.append(f"<font name='Courier'>{html.escape(token.get('raw', ''))}</font>")
        elif token_type == "link":
            url = html.escape(token.get("attrs", {}).get("url", ""), quote=True)
            label = _pdf_inline(token.get("children"))
            parts.append(f'<a href="{url}" color="#1a5fb4"><u>{label}</u></a>')
        elif token_type == "image":
            # The image itself is embedded separately after the text content.
            continue
        elif token_type in {"softbreak", "linebreak"}:
            parts.append("<br/>")
        else:
            parts.append(_pdf_inline(token.get("children")))
    return "".join(parts)


def _pdf_list(token: dict, styles: dict, depth: int = 0):
    ordered = token.get("attrs", {}).get("ordered", False)
    items = []
    for item in token.get("children", []):
        content = []
        for child in item.get("children", []):
            if child["type"] in {"block_text", "paragraph"}:
                content.append(Paragraph(_pdf_inline(child.get("children")) or " ", styles["body"]))
            elif child["type"] == "list":
                content.append(_pdf_list(child, styles, depth + 1))
        items.append(ListItem(content or [Paragraph(" ", styles["body"])], leftIndent=12))
    options = {
        "bulletType": "1" if ordered else "bullet",
        "leftIndent": 18 + depth * 12,
        "bulletFontName": "Helvetica",
        "bulletFontSize": 9,
        "spaceAfter": 8,
    }
    if ordered:
        options["start"] = str(token.get("attrs", {}).get("start", 1))
    return ListFlowable(items, **options)


def _pdf_table(token: dict, styles: dict, available_width: float):
    rows = []
    header_rows = 0
    for section in token.get("children", []):
        if section["type"] == "table_head":
            row_tokens = [section]
            header_rows = 1
        else:
            row_tokens = section.get("children", [])
        for row in row_tokens:
            cells = row.get("children", [])
            rows.append([
                Paragraph(_pdf_inline(cell.get("children")) or " ", styles["table_header"] if cell.get("attrs", {}).get("head") else styles["table_cell"])
                for cell in cells
            ])
    column_count = max((len(row) for row in rows), default=1)
    for row in rows:
        row.extend([Paragraph(" ", styles["table_cell"])] * (column_count - len(row)))
    table = Table(
        rows,
        colWidths=[available_width / column_count] * column_count,
        repeatRows=header_rows,
        hAlign="LEFT",
        splitByRow=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d2df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _pdf_markdown(text: str, styles: dict, available_width: float):
    """Render common Markdown blocks as native ReportLab flowables."""
    text = REMOTE_IMAGE_PATTERN.sub("", text)
    flowables = []
    for token in MARKDOWN_AST(text):
        token_type = token["type"]
        if token_type in {"blank_line"}:
            continue
        if token_type == "heading":
            level = min(token.get("attrs", {}).get("level", 3), 3)
            flowables.append(Paragraph(_pdf_inline(token.get("children")) or " ", styles[f"heading_{level}"]))
        elif token_type == "paragraph":
            flowables.append(Paragraph(_pdf_inline(token.get("children")) or " ", styles["body"]))
        elif token_type == "list":
            flowables.append(_pdf_list(token, styles))
        elif token_type == "table":
            flowables.append(_pdf_table(token, styles, available_width))
            flowables.append(Spacer(1, 8))
        elif token_type == "block_code":
            flowables.append(Preformatted(token.get("raw", ""), styles["code_block"]))
            flowables.append(Spacer(1, 8))
        elif token_type == "thematic_break":
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aab7c4"), spaceBefore=8, spaceAfter=12))
        elif token_type == "block_quote":
            quote = _pdf_inline(token.get("children"))
            flowables.append(Paragraph(quote or " ", styles["quote"]))
    return flowables


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
    body = ParagraphStyle("ExportBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=15, spaceAfter=8)
    tool = ParagraphStyle("Tool", parent=body, textColor=colors.HexColor("#555555"), leftIndent=12)
    markdown_styles = {
        "body": body,
        "heading_1": ParagraphStyle("MarkdownH1", parent=styles["Heading1"], fontSize=18, leading=23, spaceBefore=16, spaceAfter=8),
        "heading_2": ParagraphStyle("MarkdownH2", parent=styles["Heading2"], fontSize=14, leading=18, spaceBefore=14, spaceAfter=6),
        "heading_3": ParagraphStyle("MarkdownH3", parent=styles["Heading3"], fontSize=11.5, leading=15, spaceBefore=12, spaceAfter=5),
        "table_header": ParagraphStyle("TableHeader", parent=body, fontName="Helvetica-Bold", fontSize=8.5, leading=11, spaceAfter=0),
        "table_cell": ParagraphStyle("TableCell", parent=body, fontSize=8.5, leading=11, spaceAfter=0),
        "code_block": ParagraphStyle("CodeBlock", parent=body, fontName="Courier", fontSize=8, leading=10, leftIndent=8, rightIndent=8, backColor=colors.HexColor("#f3f5f7"), borderColor=colors.HexColor("#d8dee5"), borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=4),
        "quote": ParagraphStyle("Quote", parent=body, leftIndent=14, borderColor=colors.HexColor("#7c9bbd"), borderWidth=2, borderPadding=8, textColor=colors.HexColor("#455565")),
    }
    story = [Paragraph("Chat Agent Export", styles["Title"])]
    story.append(Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 16))
    entry_index = 0
    while entry_index < len(entries):
        entry = entries[entry_index]
        kind = entry["kind"]
        if kind == "tool":
            tool_names = []
            while entry_index < len(entries) and entries[entry_index]["kind"] == "tool":
                tool_names.append(entries[entry_index]["name"])
                entry_index += 1
            story.append(Paragraph(
                f"<b>Tools used ({len(tool_names)}):</b> {html.escape(', '.join(tool_names))}", tool
            ))
            continue
        heading = "You" if kind == "user" else "Assistant"
        story.append(Paragraph(heading, styles["Heading2"]))
        story.extend(_pdf_markdown(entry["content"], markdown_styles, document.width))
        image_sources = [(url, "") for url in REMOTE_IMAGE_PATTERN.findall(entry["content"])]
        for source, caption in image_sources:
            try:
                data = fetch_remote_image(source)
                story.append(_pdf_image(data, document.width))
                if caption:
                    story.append(Paragraph(html.escape(caption), styles["Italic"]))
            except Exception as exc:
                label = caption or source
                story.append(Paragraph(f"[Image unavailable: {html.escape(label)}]", tool))
        story.append(Spacer(1, 12))
        entry_index += 1
    document.build(story)
    return output.getvalue()


def pad_zero_arg_schema(parameters: dict | None) -> tuple[dict, bool]:
    """Return (schema, was_padded); pads schemas with no required parameters."""
    parameters = parameters or {"type": "object", "properties": {}}
    if parameters.get("required"):
        return parameters, False
    properties = parameters.get("properties") or {}
    if ZERO_ARG_PAD_PARAM in properties:
        return parameters, False
    padded = dict(parameters)
    padded["properties"] = {**properties, ZERO_ARG_PAD_PARAM: ZERO_ARG_PAD_SCHEMA}
    padded["required"] = [ZERO_ARG_PAD_PARAM]
    return padded, True


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
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ):
        provider = normalize_llm_provider(provider)
        self.system_prompt = system_prompt
        self.provider = provider
        self.base_url = normalize_llm_endpoint(base_url, self.provider)
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.llm = self._build_llm_client()
        self.messages = []
        self.anthropic_messages = []
        self.tools = []
        self.padded_tools: set[str] = set()
        self.local_tools = {}
        self.mcp_clients = []
        self.export_entries: list[dict] = []
        self._active_response_msg: cl.Message | None = None
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

    def _build_llm_client(self):
        if self.provider == "anthropic":
            return AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def update_llm_connection(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str,
        reasoning_effort: str = "default",
    ):
        """Replace this session's LLM settings without changing shared config."""
        model = model.strip()
        if not model:
            raise ValueError("Model must not be empty.")
        provider = normalize_llm_provider(provider)
        reasoning_effort = normalize_reasoning_effort(reasoning_effort)

        self.base_url = normalize_llm_endpoint(base_url, provider)
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
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
                    parameters, was_padded = pad_zero_arg_schema(t.inputSchema)
                    if was_padded:
                        self.padded_tools.add(t.name)
                    self.tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "No description",
                                "parameters": parameters,
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
    
    def _tools_for_api(self):
        return [
            {"type": t["type"], "function": t["function"]}
            for t in self.tools
        ]

    def _reasoning_request_kwargs(self) -> dict:
        """Extra request fields controlling reasoning on OpenAI-compatible endpoints.

        GLM-5.x on vLLM reads chat_template_kwargs (enable_thinking, and
        reasoning_effort where only "high" lowers effort — anything else means
        "max"); other endpoints read the top-level reasoning_effort field, so
        both are sent. The Anthropic path is intentionally left untouched.
        """
        effort = self.reasoning_effort
        if self.provider == "anthropic" or effort == "default":
            return {}
        if effort == "none":
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        return {
            "extra_body": {
                "reasoning_effort": effort,
                "chat_template_kwargs": {"reasoning_effort": effort},
            }
        }

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
                raise LLMResponseTimeout(
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
                        if tool_name in self.padded_tools:
                            tool_args.pop(ZERO_ARG_PAD_PARAM, None)
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
                self._active_response_msg = response_msg
                continue

            response_msg.content = full_content
            await response_msg.update()
            self.anthropic_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": full_content}]}
            )
            self.export_entries.append({"kind": "assistant", "content": full_content})
            return full_content

    async def chat(
        self,
        user_message: str | None,
        response_msg: cl.Message,
        *,
        resume: bool = False,
    ) -> str:
        """Run one agent turn, optionally resuming a timed-out model attempt.

        A resumed attempt deliberately does not add another user message: the
        original prompt, any tool calls, and their results are already in the
        session history.
        """
        if not resume:
            if user_message is None:
                raise ValueError("user_message is required for a new chat turn.")
            self.messages.append({"role": "user", "content": user_message})
            self.anthropic_messages.append({"role": "user", "content": user_message})
            self.export_entries.append({"kind": "user", "content": user_message})

        self._active_response_msg = response_msg

        if self.provider == "anthropic":
            return await self._chat_anthropic(response_msg)

        while True:
            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict] = {}

            try:
                _debug_stream(
                    "request: "
                    + ", ".join(f"{m['role']}" for m in self.messages)
                    + f" | {len(self.tools)} tools"
                )
                async with asyncio.timeout(LLM_STREAM_TIMEOUT_SECONDS):
                    stream = await self.llm.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=self._tools_for_api() if self.tools else None,
                        tool_choice="auto" if self.tools else None,
                        stream=True,
                        **self._reasoning_request_kwargs(),
                    )
                    _debug_stream("stream opened, waiting for first chunk")

                    async for chunk in stream:
                        if not chunk.choices:
                            _debug_stream("chunk: no choices")
                            continue

                        delta = chunk.choices[0].delta
                        _debug_stream(
                            f"chunk: content={len(delta.content) if delta.content else 0}ch"
                            f" reasoning={len(getattr(delta, 'reasoning_content', None) or '')}ch"
                            f" tool_calls={len(delta.tool_calls) if delta.tool_calls else 0}"
                            f" finish={chunk.choices[0].finish_reason}"
                        )

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
                raise LLMResponseTimeout(
                    f"Timed out after {LLM_STREAM_TIMEOUT_SECONDS}s while waiting for the model response."
                ) from e

            full_content = "".join(content_parts)
            tool_calls = list(tool_calls_by_index.values())
            _debug_stream(
                f"stream done: {len(full_content)}ch content, {len(tool_calls)} tool call(s)"
            )

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

                    if tool_name in self.padded_tools:
                        tool_args.pop(ZERO_ARG_PAD_PARAM, None)

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
                self._active_response_msg = response_msg

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
            self.export_entries.append({"kind": "assistant", "content": final_response})

            return final_response

    async def continue_chat(self, response_msg: cl.Message) -> str:
        """Retry the last model turn after an LLM stream timeout."""
        return await self.chat(None, response_msg, resume=True)


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


def build_llm_settings(agent: ChatAgent) -> cl.ChatSettings:
    return cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="llm_provider",
                label="LLM provider",
                initial_value=agent.provider,
                items={
                    "OpenAI-compatible": "openai-compatible",
                    "Anthropic / Claude": "anthropic",
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
            cl.input_widget.Select(
                id="llm_reasoning_effort",
                label="Reasoning effort",
                initial_value=agent.reasoning_effort,
                items={
                    "Default (endpoint decides)": "default",
                    "Off — no thinking": "none",
                    "Low (non-GLM endpoints only)": "low",
                    "Medium (non-GLM endpoints only)": "medium",
                    "High (GLM: balanced)": "high",
                    "Max (GLM: deepest, its default)": "max",
                },
                description=(
                    "OpenAI-compatible endpoints only; ignored for Anthropic. "
                    "GLM-5.x honors Off, High (balanced) and Max (deepest); it "
                    "treats Low/Medium as Max."
                ),
            ),
        ]
    )


async def ask_to_continue_after_llm_timeout(
    agent: ChatAgent, fallback_message: cl.Message
) -> bool:
    """Let the user choose whether to retry a timed-out model response."""
    status_message = agent._active_response_msg or fallback_message
    status_message.content = (
        f"The model did not respond within {LLM_STREAM_TIMEOUT_SECONDS // 60} minutes. "
        "Work completed so far has been kept."
    )
    await status_message.update()

    choice = await cl.AskActionMessage(
        content=(
            "The model may be busy. Would you like to try again for another "
            f"{LLM_STREAM_TIMEOUT_SECONDS // 60} minutes?"
        ),
        actions=[
            cl.Action(
                name="continue_llm_wait",
                payload={},
                label="Continue waiting",
                tooltip="Retry the model using the work already completed.",
            ),
            cl.Action(
                name="stop_llm_wait",
                payload={},
                label="Stop for now",
                tooltip="Keep the current session and resume later with 'go on'.",
            ),
        ],
        timeout=LLM_TIMEOUT_DECISION_SECONDS,
    ).send()
    return choice is not None and choice["name"] == "continue_llm_wait"


@cl.on_chat_start
async def on_chat_start():
    system_prompt = config["llm"].get("system_prompt")
    agent = ChatAgent(
        system_prompt=system_prompt,
        provider=DEFAULT_LLM_PROVIDER,
    )
    cl.user_session.set("agent", agent)

    settings = build_llm_settings(agent)
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
            if SHOW_MCP_TOOL_LIST:
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

    # Chainlit returns None for empty text inputs; `or ""` keeps str(None)
    # from turning into the literal string "None".
    endpoint = str(settings.get("llm_endpoint") or "")
    submitted_key = str(settings.get("llm_api_key") or "")
    api_key = submitted_key.strip() or agent.api_key
    model = str(settings.get("llm_model") or "")
    provider = str(settings.get("llm_provider") or "")
    reasoning_effort = str(settings.get("llm_reasoning_effort") or "default")

    try:
        provider = normalize_llm_provider(provider)
        reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        normalized_endpoint = normalize_llm_endpoint(endpoint, provider)
        is_config_reset = (
            provider == DEFAULT_LLM_PROVIDER
            and normalized_endpoint
            == normalize_llm_endpoint(config["llm"]["base_url"], DEFAULT_LLM_PROVIDER)
            and model.strip() == config["llm"]["model"]
            and not submitted_key.strip()
            and reasoning_effort == normalize_reasoning_effort(DEFAULT_REASONING_EFFORT)
        )
        if is_config_reset:
            restore_config_defaults(agent)
            await build_llm_settings(agent).send()
        else:
            agent.update_llm_connection(endpoint, api_key, model, provider, reasoning_effort)
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
        resume = False
        while True:
            try:
                if resume:
                    await agent.continue_chat(response_msg=response_msg)
                else:
                    await agent.chat(user_text, response_msg=response_msg)
                await cl.send_window_message(REFRESH_WINDOW_MESSAGE)
                return
            except LLMResponseTimeout:
                if not await ask_to_continue_after_llm_timeout(agent, response_msg):
                    await cl.Message(
                        content=(
                            "Stopped for now. Send **go on** when you want to retry "
                            "from the work already completed."
                        )
                    ).send()
                    return

                response_msg = cl.Message(content="")
                await response_msg.send()
                resume = True

    except Exception as e:
        error_text = str(e) or e.__class__.__name__
        response_msg.content = f"Error while processing message:\n\n```text\n{error_text}\n```"
        await response_msg.update()


@cl.on_chat_end
async def on_chat_end():
    agent: ChatAgent | None = cl.user_session.get("agent")

    if agent is not None:
        await agent.disconnect()
