"""
resonate_chainlit_langgraph_mvp.py

MVP architecture for RESONATE:
- Chainlit UI as operator cockpit
- LangGraph as orchestration engine
- OpenAI-compatible LLM client
- MCP tools exposed to specialized worker nodes

Run:
    uv add chainlit langgraph langchain-core openai fastmcp
    uv run chainlit run resonate_chainlit_langgraph_mvp.py -w

Expected config.toml shape is compatible with your current agent:

[llm]
base_url = "..."
api_key = "..."
model = "..."
system_prompt = "..."

[[mcp_servers]]
type = "stdio"
command = "uv"
args = ["run", "server.py"]
cwd = "/path/to/server"

"""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Literal, TypedDict

import chainlit as cl
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langgraph.graph import StateGraph, END
from openai import OpenAI


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Copy config.example.toml to config.toml and edit it with your settings."
        )

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_mcp_servers(config: dict[str, Any]) -> list[Any]:
    servers: list[Any] = []

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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def strip_thinking(text: str | None) -> str:
    if not text:
        return ""

    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
    return result.strip()


def compact_json(data: Any, max_len: int = 4000) -> str:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    except TypeError:
        text = str(data)

    if len(text) > max_len:
        return text[:max_len] + "\n... [truncated]"
    return text


# -----------------------------------------------------------------------------
# MCP Tool Registry
# -----------------------------------------------------------------------------


class MCPToolRegistry:
    """Connects to MCP servers and allows calling tools by name."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.clients: list[tuple[str, Client]] = []

    async def connect(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []

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
                self.clients.append((server_name, client))

                server_tools = await client.list_tools()
                tool_names: list[str] = []

                for tool in server_tools:
                    tool_names.append(tool.name)
                    self.tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description or "No description",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "client": client,
                        "server": server_name,
                    }

                discovered.append(
                    {
                        "server": server_name,
                        "ok": True,
                        "tools": tool_names,
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

    async def disconnect(self) -> None:
        for _, client in self.clients:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass

    def list_tool_names(self) -> list[str]:
        return sorted(self.tools.keys())

    def tools_matching(self, keywords: list[str]) -> list[str]:
        """Simple initial router: select tools by name/description keywords."""
        selected: list[str] = []
        lowered_keywords = [k.lower() for k in keywords]

        for name, meta in self.tools.items():
            haystack = f"{name} {meta.get('description', '')}".lower()
            if any(k in haystack for k in lowered_keywords):
                selected.append(name)

        return sorted(selected)

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return f"Error: tool '{tool_name}' not found"

        client: Client = self.tools[tool_name]["client"]

        try:
            result = await client.call_tool(tool_name, args)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as e:
            return f"Error while calling tool '{tool_name}': {e}"


# -----------------------------------------------------------------------------
# LangGraph State
# -----------------------------------------------------------------------------


class EvidenceItem(TypedDict, total=False):
    source: str
    tool: str
    query: str
    content: str
    metadata: dict[str, Any]


class ResonanceState(TypedDict, total=False):
    user_request: str
    plan: str
    selected_collectors: list[str]
    gdelt_evidence: list[EvidenceItem]
    social_evidence: list[EvidenceItem]
    image_evidence: list[EvidenceItem]
    stored_records: list[str]
    assessment: str
    final_answer: str
    errors: list[str]


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------


def ask_llm(messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
    response = llm.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return strip_thinking(response.choices[0].message.content)


async def run_worker_agent(
    *,
    name: str,
    system_prompt: str,
    task: str,
    allowed_tool_names: list[str],
    registry: MCPToolRegistry,
    max_iterations: int = 8,
) -> str:
    """
    Runs a small OpenAI-compatible tool-calling agent restricted to selected MCP tools.
    """

    tools = []

    for tool_name in allowed_tool_names:
        if tool_name not in registry.tools:
            continue

        meta = registry.tools[tool_name]

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": meta["description"],
                    "parameters": meta["parameters"],
                },
            }
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]

    async with cl.Step(name=name) as step:
        step.input = {
            "task": task,
            "allowed_tools": allowed_tool_names,
        }

        for _ in range(max_iterations):
            response = llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
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
                    raw_args = tool_call.function.arguments or "{}"

                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        result_text = f"Error: invalid JSON arguments: {e}"
                    else:
                        async with cl.Step(name=f"{name} → Tool: {tool_name}") as tool_step:
                            tool_step.input = json.dumps(tool_args, indent=2, ensure_ascii=False)

                            result_text = await registry.call_tool(tool_name, tool_args)

                            tool_step.output = result_text[:4000]

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )

                continue

            final = strip_thinking(assistant_message.content)

            step.output = final
            return final

        final = "Worker stopped because it reached the maximum number of tool iterations."
        step.output = final
        return final

# -----------------------------------------------------------------------------
# Graph Nodes
# -----------------------------------------------------------------------------


async def planner_node(state: ResonanceState) -> ResonanceState:
    user_request = state["user_request"]

    async with cl.Step(name="Planner") as step:
        step.input = user_request

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are the RESONATE orchestrator. Build a concise investigation plan. "
                    "Select relevant collectors from: gdelt, social, image. "
                    "Return JSON with keys: plan, selected_collectors."
                ),
            },
            {"role": "user", "content": user_request},
        ]

        raw = ask_llm(prompt)
        step.output = raw

        try:
            parsed = json.loads(raw)
            plan = parsed.get("plan", raw)
            selected = parsed.get("selected_collectors", [])
        except json.JSONDecodeError:
            plan = raw
            lowered = user_request.lower()
            selected = []
            if any(k in lowered for k in ["news", "gdelt", "media", "article"]):
                selected.append("gdelt")
            if any(k in lowered for k in ["x", "twitter", "telegram", "social"]):
                selected.append("social")
            if any(k in lowered for k in ["image", "satellite", "sentinel", "photo"]):
                selected.append("image")
            if not selected:
                selected = ["gdelt", "social"]

        return {
            **state,
            "plan": plan,
            "selected_collectors": selected,
        }


async def gdelt_collector_node(state: ResonanceState) -> ResonanceState:
    registry: MCPToolRegistry = cl.user_session.get("tool_registry")
    user_request = state["user_request"]

    if "gdelt" not in state.get("selected_collectors", []):
        return {**state, "gdelt_evidence": []}

    async with cl.Step(name="GDELT Agent") as step:
        tools = registry.tools_matching(["gdelt"])
        step.input = compact_json({"available_tools": tools, "request": user_request})

        if not tools:
            msg = "No GDELT MCP tools found."
            step.output = msg
            return {**state, "gdelt_evidence": [], "errors": state.get("errors", []) + [msg]}

        # MVP: call the first matching GDELT tool with a generic query argument.
        # You will likely customize this once we know your exact tool schemas.
        tool_name = tools[0]
        #result = await registry.call_tool(tool_name, {"query": user_request})
        result = await run_worker_agent(
            name="GDELT Agent",
            task=f"Collect relevant GDELT evidence for: {user_request}",
            allowed_tool_names=tools,
            registry=registry,
        )

        evidence: EvidenceItem = {
            "source": "gdelt",
            "tool": tool_name,
            "query": user_request,
            "content": result,
            "metadata": {},
        }

        step.output = result[:4000]
        return {**state, "gdelt_evidence": [evidence]}


async def social_collector_node(state: ResonanceState) -> ResonanceState:
    registry: MCPToolRegistry = cl.user_session.get("tool_registry")
    user_request = state["user_request"]

    if "social" not in state.get("selected_collectors", []):
        return {**state, "social_evidence": []}

    async with cl.Step(name="Social Scraping Agent") as step:
        tools = registry.tools_matching(["twitter", "x", "telegram", "social"])
        step.input = compact_json({"available_tools": tools, "request": user_request})

        if not tools:
            msg = "No social scraping MCP tools found."
            step.output = msg
            return {**state, "social_evidence": [], "errors": state.get("errors", []) + [msg]}

        evidence_items: list[EvidenceItem] = []

        for tool_name in tools[:3]:
            result = await registry.call_tool(tool_name, {"query": user_request})
            evidence_items.append(
                {
                    "source": "social",
                    "tool": tool_name,
                    "query": user_request,
                    "content": result,
                    "metadata": {},
                }
            )

        step.output = compact_json(evidence_items)
        return {**state, "social_evidence": evidence_items}


async def image_analysis_node(state: ResonanceState) -> ResonanceState:
    registry: MCPToolRegistry = cl.user_session.get("tool_registry")
    user_request = state["user_request"]

    if "image" not in state.get("selected_collectors", []):
        return {**state, "image_evidence": []}

    async with cl.Step(name="Image / Satellite Agent") as step:
        tools = registry.tools_matching(["sentinel", "satellite", "image", "vision"])
        step.input = compact_json({"available_tools": tools, "request": user_request})

        if not tools:
            msg = "No image/satellite MCP tools found."
            step.output = msg
            return {**state, "image_evidence": [], "errors": state.get("errors", []) + [msg]}

        evidence_items: list[EvidenceItem] = []

        for tool_name in tools[:2]:
            result = await registry.call_tool(tool_name, {"query": user_request})
            evidence_items.append(
                {
                    "source": "image",
                    "tool": tool_name,
                    "query": user_request,
                    "content": result,
                    "metadata": {},
                }
            )

        step.output = compact_json(evidence_items)
        return {**state, "image_evidence": evidence_items}


async def database_writer_node(state: ResonanceState) -> ResonanceState:
    registry: MCPToolRegistry = cl.user_session.get("tool_registry")

    async with cl.Step(name="Database Agent") as step:
        all_evidence = (
            state.get("gdelt_evidence", [])
            + state.get("social_evidence", [])
            + state.get("image_evidence", [])
        )

        db_tools = registry.tools_matching(["database", "postgres", "save", "store", "insert"])
        step.input = compact_json({"available_tools": db_tools, "evidence_count": len(all_evidence)})

        if not db_tools:
            msg = "No database/storage MCP tools found. Evidence kept in memory only."
            step.output = msg
            return {**state, "stored_records": [], "errors": state.get("errors", []) + [msg]}

        stored_records: list[str] = []
        tool_name = db_tools[0]

        for item in all_evidence:
            result = await registry.call_tool(
                tool_name,
                {
                    "record": item,
                },
            )
            stored_records.append(result)

        step.output = compact_json(stored_records)
        return {**state, "stored_records": stored_records}


async def assessment_node(state: ResonanceState) -> ResonanceState:
    async with cl.Step(name="Evidence Assessment") as step:
        evidence_summary = {
            "gdelt": state.get("gdelt_evidence", []),
            "social": state.get("social_evidence", []),
            "image": state.get("image_evidence", []),
            "errors": state.get("errors", []),
        }

        step.input = compact_json(evidence_summary)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an OSINT evidence evaluator. Assess the collected evidence. "
                    "Be concise. Identify corroboration, contradictions, gaps, and confidence."
                ),
            },
            {
                "role": "user",
                "content": compact_json(
                    {
                        "request": state["user_request"],
                        "plan": state.get("plan"),
                        "evidence": evidence_summary,
                    },
                    max_len=12000,
                ),
            },
        ]

        assessment = ask_llm(prompt)
        step.output = assessment
        return {**state, "assessment": assessment}


async def final_synthesis_node(state: ResonanceState) -> ResonanceState:
    async with cl.Step(name="Final Synthesis") as step:
        step.input = state.get("assessment", "")

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are RESONATE, an OSINT investigation assistant. "
                    "Give the user a clear final answer. Include what was checked, what was found, "
                    "uncertainties, and next recommended actions. Do not overclaim."
                ),
            },
            {
                "role": "user",
                "content": compact_json(
                    {
                        "request": state["user_request"],
                        "plan": state.get("plan"),
                        "assessment": state.get("assessment"),
                        "errors": state.get("errors", []),
                    },
                    max_len=12000,
                ),
            },
        ]

        final = ask_llm(prompt)
        step.output = final
        return {**state, "final_answer": final}


# -----------------------------------------------------------------------------
# Graph definition
# -----------------------------------------------------------------------------


def build_resonate_graph():
    graph = StateGraph(ResonanceState)

    graph.add_node("planner", planner_node)
    graph.add_node("gdelt_collector", gdelt_collector_node)
    graph.add_node("social_collector", social_collector_node)
    graph.add_node("image_analysis", image_analysis_node)
    graph.add_node("database_writer", database_writer_node)
    graph.add_node("assessment", assessment_node)
    graph.add_node("final_synthesis", final_synthesis_node)

    graph.set_entry_point("planner")

    # MVP sequential version. Later we can parallelize collectors.
    graph.add_edge("planner", "gdelt_collector")
    graph.add_edge("gdelt_collector", "social_collector")
    graph.add_edge("social_collector", "image_analysis")
    graph.add_edge("image_analysis", "database_writer")
    graph.add_edge("database_writer", "assessment")
    graph.add_edge("assessment", "final_synthesis")
    graph.add_edge("final_synthesis", END)

    return graph.compile()


# -----------------------------------------------------------------------------
# Chainlit UI
# -----------------------------------------------------------------------------


@cl.on_chat_start
async def on_chat_start():
    registry = MCPToolRegistry()

    msg = cl.Message(content="Discovering MCP servers and tools...")
    await msg.send()

    discovered = await registry.connect()
    resonate_graph = build_resonate_graph()

    cl.user_session.set("tool_registry", registry)
    cl.user_session.set("resonate_graph", resonate_graph)

    lines = ["# RESONATE Ready", ""]

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
    lines.append("Send an investigation request to start a RESONATE run.")

    msg.content = "\n".join(lines)
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("resonate_graph")

    if graph is None:
        await cl.Message(content="No RESONATE graph found. Refresh the page.").send()
        return

    user_request = message.content.strip()

    if not user_request:
        return

    if user_request.lower() in {"/diagram", "diagram"}:
        try:
            mermaid = graph.get_graph().draw_mermaid()
            await cl.Message(content=f"```mermaid\n{mermaid}\n```").send()
        except Exception as e:
            await cl.Message(content=f"Could not generate graph diagram: `{e}`").send()
        return

    initial_state: ResonanceState = {
        "user_request": user_request,
        "errors": [],
    }

    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        final_state = await graph.ainvoke(initial_state)
        response_msg.content = final_state.get("final_answer", "No final answer generated.")
        await response_msg.update()
    except Exception as e:
        response_msg.content = f"Error during RESONATE run:\n\n```text\n{e}\n```"
        await response_msg.update()


@cl.on_chat_end
async def on_chat_end():
    registry: MCPToolRegistry | None = cl.user_session.get("tool_registry")
    if registry is not None:
        await registry.disconnect()
