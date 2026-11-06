"""Export the app's tool declarations (defined once, for Gemini, in gemini.build_tools) as plain JSON schema
specs for the local brain, tagged with whether a tool returns information to read back or just does something."""
from __future__ import annotations

from typing import Any

# Tools whose result the model must read back to the user. Everything else is an action.
INFO_TOOLS = {"calendar_events", "tasks_list", "gmail_check"}


def _json_schema(s: Any) -> dict:
    d = s.model_dump(exclude_none=True, mode="json") if hasattr(s, "model_dump") else dict(s)
    return _lower(d)


def _lower(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _lower(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _lower(v)
        elif k in ("type", "description", "required", "enum", "minimum", "maximum", "format", "nullable"):
            out[k] = v
    out.setdefault("type", "object")
    return out


def robot_tool_specs() -> list[dict]:
    from .gemini import build_tools
    specs = []
    for tool in build_tools(enable_search=False):
        for decl in tool.function_declarations or []:
            params = _json_schema(decl.parameters) if decl.parameters else {"type": "object", "properties": {}}
            params.setdefault("properties", {})
            specs.append({"name": decl.name, "description": decl.description or "", "parameters": params,
                          "kind": "info" if decl.name in INFO_TOOLS else "action"})
    return specs
