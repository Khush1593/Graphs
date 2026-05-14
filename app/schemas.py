"""Pydantic response schemas for LLM-structured outputs.

The pipeline stages that depend on LLM-emitted JSON (dataset understanding,
clarification interview, goal proposals, dashboard plan) now validate the
model's response against these schemas instead of regex-cleaning a markdown
string. Anything the LLM emits that doesn't fit the schema raises
``pydantic.ValidationError`` — callers catch it and either retry or fall back
to deterministic logic, rather than silently corrupting downstream state.

Models use ``extra="ignore"`` so a chatty model that adds an unexpected key
(e.g. ``"reasoning"``) is accepted instead of crashing; structural correctness
is what we care about. Where a field is genuinely free-form (e.g. clarification
question defaults can be ``str`` OR ``list[str]`` OR ``""``) we keep it ``Any``
and let the downstream engine validate the shape it actually needs.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    """Base for every LLM-output model. ``extra="ignore"`` keeps us robust to
    chatty models; ``str_strip_whitespace`` normalizes accidental leading
    spaces in JSON strings."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Stage 3 — Dataset Understanding
# ---------------------------------------------------------------------------


class DerivedMetricOpportunity(_Permissive):
    name: str
    formula: str
    why_useful: str


class TemporalFeatures(_Permissive):
    has_time: bool = False
    time_column: Optional[str] = None
    suggested_granularity: Optional[str] = None
    seasonality_candidates: List[str] = Field(default_factory=list)


class UnderstandingResponse(_Permissive):
    dataset_description: str
    business_domain_guess: str
    primary_entity: str
    derived_metric_opportunities: List[DerivedMetricOpportunity] = Field(default_factory=list)
    temporal_features: TemporalFeatures = Field(default_factory=TemporalFeatures)
    data_quality_notes: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


# ---------------------------------------------------------------------------
# Stage 4 — Clarification Interview
# ---------------------------------------------------------------------------


ClarificationKey = Literal[
    "primary_goal",
    "focus_metric",
    "focus_dimensions",
    "time_granularity",
    "comparison_window",
    "decision_to_make",
    "known_events",
    "audience",
]


class ClarificationQuestion(_Permissive):
    id: str
    key: ClarificationKey
    question: str
    type: Literal["single_select", "multi_select", "free_text"]
    options: List[str] = Field(default_factory=list)
    # ``default`` is intentionally permissive: single_select expects a str,
    # multi_select a list[str], free_text an empty string. Downstream
    # clarification_engine performs shape-specific validation.
    default: Any = ""
    why_asked: str = ""


class ClarificationQuestionsResponse(_Permissive):
    questions: List[ClarificationQuestion]


# ---------------------------------------------------------------------------
# Stage 5 — Goals
# ---------------------------------------------------------------------------


class Goal(_Permissive):
    id: str
    title: str
    metric: str
    dimension: Optional[str] = None
    direction: Literal["increase", "decrease", "stabilize"]
    target_pct: Optional[float] = None
    priority: Literal["high", "medium", "low"]
    time_horizon: Literal["daily", "weekly", "monthly", "quarterly"]
    rationale: str
    active: bool = True


class GoalsResponse(_Permissive):
    goals: List[Goal]


# ---------------------------------------------------------------------------
# Stage 8 — Dashboard Plan
# ---------------------------------------------------------------------------


class DashboardItemPlan(_Permissive):
    kind: Literal["kpi", "line", "bar", "hist"]
    title: str
    sql: str


class DashboardPlanResponse(_Permissive):
    items: List[DashboardItemPlan]
