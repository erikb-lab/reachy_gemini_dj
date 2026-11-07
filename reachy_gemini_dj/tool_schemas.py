"""Export the app's tool declarations (defined once, for Gemini, in gemini.build_tools) as plain JSON schema
specs for the local brain, tagged with whether a tool returns information to read back or just does something."""
from __future__ import annotations

from typing import Any

# Tools whose result the model must read back to the user. Everything else is an action.
INFO_TOOLS = {"calendar_events", "tasks_list", "gmail_check", "scan_qr", "frc_next_match", "frc_event_status",
              "frc_team_info", "frc_rankings", "list_memories", "list_timers"}


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


GOOGLE_TOOL_SERVICE = {"calendar_events": "calendar", "calendar_add": "calendar", "tasks_list": "tasks",
                       "tasks_add": "tasks", "tasks_complete": "tasks", "gmail_check": "gmail"}


def robot_tool_specs(google_services=None, frc_enabled: bool | None = None) -> list[dict]:
    """Tool list sent to the brain. Every tool costs prompt tokens (~100 each, ~22 tokens/s to read on a laptop
    CPU), so pass google_services / frc_enabled to leave out tools that cannot work yet. None = include all."""
    from .gemini import build_tools
    specs = []
    for tool in build_tools(enable_search=False):
        for decl in tool.function_declarations or []:
            svc = GOOGLE_TOOL_SERVICE.get(decl.name)
            if google_services is not None and svc and svc not in google_services:
                continue
            if frc_enabled is False and decl.name.startswith("frc_"):
                continue
            params = _json_schema(decl.parameters) if decl.parameters else {"type": "object", "properties": {}}
            params.setdefault("properties", {})
            specs.append({"name": decl.name, "description": decl.description or "", "parameters": params,
                          "kind": "info" if decl.name in INFO_TOOLS else "action"})
    return specs
