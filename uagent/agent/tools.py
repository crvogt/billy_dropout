"""Typed tool surface bound to the LLM.

Five tools, deliberately scope-limited to what the paper experiment
needs. Each has a Pydantic input schema, a typed output dataclass, and a
thin wrapper that calls the harvested implementation. Wrappers default
to MOCK behavior; live ROS execution is enabled by the `ros.enabled`
config field (step 5).

Risk tier is recorded per tool for downstream metrics. v1 treats all
five as `low`; the field exists to support a future high-stakes
extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

RiskTier = Literal["low", "high"]


# ---------------------------------------------------------------------------
# Input schemas (what the LLM emits) — Pydantic for tool-call validation.
# ---------------------------------------------------------------------------


class MoveForwardInput(BaseModel):
    distance_m: float = Field(..., description="meters to drive forward; positive only")


class RotateInput(BaseModel):
    angle_deg: float = Field(..., description="degrees to rotate in place; positive = right")


class LookAroundInput(BaseModel):
    pass  # no args


class ReportInput(BaseModel):
    message: str = Field(..., description="verbal message; no robot motion")


class DeferInput(BaseModel):
    reason: str = Field(..., description="why the agent is holding")


# ---------------------------------------------------------------------------
# Output dataclasses (what the dispatch node feeds back into context).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    risk_tier: RiskTier
    ok: bool
    detail: str                      # human-readable summary, fed back to LLM
    terminates_turn: bool            # True for `report` and `defer`


# ---------------------------------------------------------------------------
# Wrappers — mock implementations for step 4. Step 5 swaps in roslibpy.
# ---------------------------------------------------------------------------


def move_forward(args: MoveForwardInput, *, live: bool = False) -> ToolResult:
    """Drive forward by args.distance_m meters."""
    raise NotImplementedError("agent runtime is implemented in step 4")


def rotate(args: RotateInput, *, live: bool = False) -> ToolResult:
    """Rotate in place by args.angle_deg degrees."""
    raise NotImplementedError


def look_around(args: LookAroundInput, *, live: bool = False) -> ToolResult:
    """Capture a panoramic view (re-sense)."""
    raise NotImplementedError


def report(args: ReportInput, *, live: bool = False) -> ToolResult:
    """Verbal-only action; ends the turn."""
    raise NotImplementedError


def defer(args: DeferInput, *, live: bool = False) -> ToolResult:
    """Explicit HOLD; ends the turn."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool registry — used by runtime.py to bind tools to the LLM and route
# dispatch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk_tier: RiskTier
    input_schema: type[BaseModel]
    handler: object                  # Callable[[BaseModel, *, bool], ToolResult]
    terminates_turn: bool


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "move_forward": ToolSpec("move_forward", "low", MoveForwardInput, move_forward, False),
    "rotate":       ToolSpec("rotate",       "low", RotateInput,       rotate,       False),
    "look_around":  ToolSpec("look_around",  "low", LookAroundInput,  look_around,  False),
    "report":       ToolSpec("report",       "low", ReportInput,       report,       True),
    "defer":        ToolSpec("defer",        "low", DeferInput,        defer,        True),
}
