"""Typed tool surface bound to the LLM.

Five tools, deliberately scope-limited to what the paper experiment
needs. Each has a Pydantic input schema, a typed output dataclass, and
a thin wrapper that calls the harvested implementation. Wrappers
default to MOCK behavior; live ROS execution is enabled by the
`ros.enabled` config field (step 5 lands the real bridge).

Risk tier is recorded per tool for downstream metrics. v1 treats all
five as `low`; the field exists to support a future high-stakes
extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

RiskTier = Literal["low", "high"]


# ---------------------------------------------------------------------------
# Input schemas (what the LLM emits) — Pydantic for tool-call validation.
# ---------------------------------------------------------------------------


class MoveForwardInput(BaseModel):
    distance_m: float = Field(..., description="meters to drive forward; positive only", ge=0.0)


class RotateInput(BaseModel):
    angle_deg: float = Field(..., description="degrees to rotate in place; positive = right")


class LookAroundInput(BaseModel):
    pass  # no args


class ReportInput(BaseModel):
    message: str = Field(..., description="verbal message; no robot motion")


class DeferInput(BaseModel):
    reason: str = Field(..., description="why the agent is holding")


# ---------------------------------------------------------------------------
# Output dataclass (what the dispatch node feeds back into context).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    risk_tier: RiskTier
    ok: bool
    detail: str                      # human-readable summary, fed back to LLM
    terminates_turn: bool            # True for `report` and `defer`


# ---------------------------------------------------------------------------
# Wrappers — mock implementations. Step 5 swaps in roslibpy under
# `live=True`. Tests pass live=False; the experimental harness runs
# with live=False (mock) by default.
# ---------------------------------------------------------------------------


def move_forward(args: MoveForwardInput, *, live: bool = False) -> ToolResult:
    if live:
        from uagent.ros.bridge import RosBridge  # lazy
        RosBridge().drive_forward(args.distance_m)
    return ToolResult(
        tool_name="move_forward",
        risk_tier="low",
        ok=True,
        detail=f"moved {args.distance_m:.2f} m forward" + ("" if live else " [mock]"),
        terminates_turn=False,
    )


def rotate(args: RotateInput, *, live: bool = False) -> ToolResult:
    if live:
        from uagent.ros.bridge import RosBridge
        RosBridge().rotate_in_place(args.angle_deg)
    return ToolResult(
        tool_name="rotate",
        risk_tier="low",
        ok=True,
        detail=f"rotated {args.angle_deg:.1f} deg" + ("" if live else " [mock]"),
        terminates_turn=False,
    )


def look_around(args: LookAroundInput, *, live: bool = False) -> ToolResult:
    if live:
        from uagent.ros.bridge import RosBridge
        RosBridge().capture_panorama()
        detail = "captured panorama"
    else:
        detail = "panorama not available in experiment mode [mock]"
    return ToolResult(
        tool_name="look_around",
        risk_tier="low",
        ok=True,
        detail=detail,
        terminates_turn=False,
    )


def report(args: ReportInput, *, live: bool = False) -> ToolResult:
    return ToolResult(
        tool_name="report",
        risk_tier="low",
        ok=True,
        detail=f"reported: {args.message}",
        terminates_turn=True,
    )


def defer(args: DeferInput, *, live: bool = False) -> ToolResult:
    return ToolResult(
        tool_name="defer",
        risk_tier="low",
        ok=True,
        detail=f"deferred: {args.reason}",
        terminates_turn=True,
    )


# ---------------------------------------------------------------------------
# Tool registry — used by runtime.py to bind tools to the LLM and route
# dispatch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk_tier: RiskTier
    input_schema: type[BaseModel]
    handler: Callable[..., ToolResult]
    terminates_turn: bool


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "move_forward": ToolSpec("move_forward", "low", MoveForwardInput, move_forward, False),
    "rotate":       ToolSpec("rotate",       "low", RotateInput,      rotate,       False),
    "look_around":  ToolSpec("look_around",  "low", LookAroundInput,  look_around,  False),
    "report":       ToolSpec("report",       "low", ReportInput,      report,       True),
    "defer":        ToolSpec("defer",        "low", DeferInput,       defer,        True),
}


# ---------------------------------------------------------------------------
# LLM-side tool surface — StructuredTool objects with the friendly names
# the system prompt advertises (`move_forward`, etc.). The LLM emits these
# names; runtime.dispatch looks them up in TOOL_REGISTRY by string match.
#
# These shells are not callable — actual dispatch routes through
# TOOL_REGISTRY[name].handler. Binding LLM tools to a usable function is
# required by langchain's StructuredTool API even though we never invoke
# through it.
# ---------------------------------------------------------------------------


def _bound_tool_unreachable(**kwargs: Any) -> str:
    raise RuntimeError(
        "LLM-bound tool stub invoked directly; dispatch must go through TOOL_REGISTRY"
    )


def _llm_tool(name: str, description: str, schema: type[BaseModel]) -> StructuredTool:
    return StructuredTool.from_function(
        func=_bound_tool_unreachable,
        name=name,
        description=description,
        args_schema=schema,
    )


LLM_TOOLS: list[StructuredTool] = [
    _llm_tool("move_forward", "Drive forward by `distance_m` meters. Positive only.", MoveForwardInput),
    _llm_tool("rotate",       "Rotate in place by `angle_deg` degrees. Positive = right.", RotateInput),
    _llm_tool("look_around",  "Capture a panoramic view to re-sense the environment.", LookAroundInput),
    _llm_tool("report",       "Speak verbally; no robot motion. Ends the turn.", ReportInput),
    _llm_tool("defer",        "Explicit HOLD — defer the decision. Ends the turn.", DeferInput),
]
