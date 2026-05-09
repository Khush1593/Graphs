from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.data_engine import SchemaInfo


@dataclass
class Context:
    schema: SchemaInfo
    semantic_hints: dict
    understanding: dict
    user_clarifications: dict
    user_goals: list[dict]
    metadata: dict = field(default_factory=dict)

    @property
    def understanding_summary(self) -> dict:
        return self.understanding.get("summary", {}) if self.understanding else {}

    @property
    def measure_columns(self) -> list[str]:
        return [
            m.get("name")
            for m in (self.understanding_summary.get("measure_columns") or [])
            if m.get("name")
        ]

    @property
    def dimension_columns(self) -> list[str]:
        return [
            d.get("name")
            for d in (self.understanding_summary.get("dimension_columns") or [])
            if d.get("name")
        ]

    @property
    def identifier_columns(self) -> list[str]:
        return list(self.understanding_summary.get("identifier_columns") or [])

    @property
    def time_column(self) -> str | None:
        return (self.understanding_summary.get("time_scope") or {}).get("time_column")

    @property
    def primary_goal_text(self) -> str | None:
        return (self.user_clarifications or {}).get("primary_goal")

    @property
    def focus_metric(self) -> str | None:
        clarif = (self.user_clarifications or {}).get("focus_metric")
        if clarif and clarif in self.measure_columns:
            return clarif
        return self.understanding_summary.get("primary_metric")

    @property
    def focus_dimensions(self) -> list[str]:
        clarif = (self.user_clarifications or {}).get("focus_dimensions") or []
        return [d for d in clarif if d in self.dimension_columns]

    @property
    def time_granularity(self) -> str | None:
        clarif = (self.user_clarifications or {}).get("time_granularity")
        if clarif:
            return clarif
        return (self.understanding_summary.get("time_scope") or {}).get("granularity_guess")

    @property
    def comparison_window(self) -> str | None:
        return (self.user_clarifications or {}).get("comparison_window")

    @property
    def active_goals(self) -> list[dict]:
        return [g for g in (self.user_goals or []) if g.get("active", True)]

    def to_dict(self) -> dict:
        return {
            "schema": {
                "columns": list(self.schema.columns),
                "dtypes": dict(self.schema.dtypes),
                "date_column": self.schema.date_column,
                "numeric_column": self.schema.numeric_column,
                "category_column": self.schema.category_column,
            },
            "semantic_hints": self.semantic_hints,
            "understanding_summary": self.understanding_summary,
            "user_clarifications": self.user_clarifications,
            "user_goals": self.user_goals,
            "active_goal_count": len(self.active_goals),
            "metadata": self.metadata,
            "derived": {
                "focus_metric": self.focus_metric,
                "focus_dimensions": self.focus_dimensions,
                "time_column": self.time_column,
                "time_granularity": self.time_granularity,
                "comparison_window": self.comparison_window,
                "primary_goal_text": self.primary_goal_text,
                "measure_columns": self.measure_columns,
                "dimension_columns": self.dimension_columns,
                "identifier_columns": self.identifier_columns,
            },
        }

    def to_prompt_block(self) -> dict:
        clarif = self.user_clarifications or {}
        business_context = {
            k: v
            for k, v in {
                "decision_to_make": (clarif.get("decision_to_make") or "").strip() or None,
                "known_events": (clarif.get("known_events") or "").strip() or None,
                "audience": (clarif.get("audience") or "").strip() or None,
            }.items()
            if v
        }

        return {
            "primary_goal": self.primary_goal_text,
            "focus_metric": self.focus_metric,
            "focus_dimensions": self.focus_dimensions,
            "time_granularity": self.time_granularity,
            "comparison_window": self.comparison_window,
            "business_context": business_context,
            "active_goals": [
                {
                    "id": g.get("id"),
                    "title": g.get("title"),
                    "metric": g.get("metric"),
                    "dimension": g.get("dimension"),
                    "direction": g.get("direction"),
                    "target_pct": g.get("target_pct"),
                    "priority": g.get("priority"),
                    "rationale": g.get("rationale"),
                }
                for g in self.active_goals
            ],
            "available_measures": self.measure_columns,
            "available_dimensions": self.dimension_columns,
            "identifier_columns": self.identifier_columns,
            "time_column": self.time_column,
            "business_domain": self.understanding_summary.get("business_domain_guess"),
            "primary_entity": self.understanding_summary.get("primary_entity"),
        }


def build_context(
    schema: SchemaInfo,
    semantic_hints: dict | None,
    understanding: dict | None,
    user_clarifications: dict | None,
    user_goals: list[dict] | None,
    debug_logger=None,
) -> Context:
    ctx = Context(
        schema=schema,
        semantic_hints=semantic_hints or {},
        understanding=understanding or {},
        user_clarifications=user_clarifications or {},
        user_goals=list(user_goals or []),
        metadata={
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "understanding_source": (understanding or {}).get("source"),
            "understanding_provider": (understanding or {}).get("provider"),
            "understanding_model": (understanding or {}).get("model"),
            "user_goal_count": len(user_goals or []),
            "active_goal_count": len([g for g in (user_goals or []) if g.get("active", True)]),
        },
    )

    if debug_logger:
        debug_logger.log_event("context_built", ctx.to_dict())

    return ctx


def goal_alignment_score(
    signal_metric: str | None,
    signal_dimension: str | None,
    signal_direction: str | None,
    goals: list[dict],
) -> tuple[int, list[str]]:
    if not goals:
        return 0, []

    matched_ids: list[str] = []
    score = 0
    for g in goals:
        local_score = 0
        if g.get("metric") and signal_metric and g["metric"] == signal_metric:
            local_score += 30
        if g.get("dimension") and signal_dimension and g["dimension"] == signal_dimension:
            local_score += 20
        if (
            g.get("direction")
            and signal_direction
            and g["direction"] == signal_direction
        ):
            local_score += 10

        priority_bonus = {"high": 5, "medium": 2, "low": 0}.get(g.get("priority", "medium"), 2)

        if local_score > 0:
            matched_ids.append(g.get("id") or g.get("title", ""))
            score += local_score + priority_bonus

    return min(score, 60), matched_ids
