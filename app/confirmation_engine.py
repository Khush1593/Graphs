from __future__ import annotations

from typing import Any

from app.config import settings
from app.context_engine import Context
from app.llm_engine import generate_business_confirmation, resolve_active_model


REQUIRED_KEYS = {
    "headline",
    "data_summary",
    "user_intent_summary",
    "goals_recap",
    "what_will_be_built",
    "open_questions",
    "confidence",
}


def build_confirmation_input(context: Context) -> dict:
    summary = context.understanding_summary
    time_scope = summary.get("time_scope") or {}

    return {
        "data_profile": {
            "row_count": (summary.get("data_quality_signals") or {}).get("row_count"),
            "primary_entity": summary.get("primary_entity"),
            "business_domain_guess": summary.get("business_domain_guess"),
            "primary_metric": summary.get("primary_metric"),
            "time_column": time_scope.get("time_column"),
            "time_start": time_scope.get("start"),
            "time_end": time_scope.get("end"),
            "time_granularity_guess": time_scope.get("granularity_guess"),
        },
        "context_prompt_block": context.to_prompt_block(),
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
            for g in context.active_goals
        ],
        "data_quality_notes": summary.get("data_quality_notes") or [],
        "suggested_kpis": summary.get("suggested_kpis") or [],
    }


def _direction_phrase(direction: str | None) -> str:
    return {
        "increase": "grow",
        "decrease": "reduce",
        "stabilize": "stabilize",
    }.get(direction or "", "track")


def _goal_to_plain_english(g: dict) -> str:
    metric = g.get("metric") or "the metric"
    dim = g.get("dimension")
    direction = _direction_phrase(g.get("direction"))
    target = g.get("target_pct")
    target_phrase = f" by {target:g}%" if isinstance(target, (int, float)) and target else ""
    if dim:
        return f"{direction.capitalize()} {metric}{target_phrase} across {dim}."
    return f"{direction.capitalize()} {metric}{target_phrase} overall."


def generate_deterministic_confirmation(context: Context) -> dict:
    summary = context.understanding_summary
    time_scope = summary.get("time_scope") or {}
    row_count = (summary.get("data_quality_signals") or {}).get("row_count")
    domain = summary.get("business_domain_guess") or "unknown business domain"
    entity = summary.get("primary_entity") or "row"
    primary_metric = context.focus_metric or summary.get("primary_metric") or "the primary metric"
    primary_goal = context.primary_goal_text or "no primary goal stated"
    focus_dims = context.focus_dimensions or []
    granularity = context.time_granularity or "weekly"

    headline = f"This looks like a {domain} dataset where each {entity} carries a {primary_metric} value."

    if time_scope.get("start") and time_scope.get("end"):
        data_summary = (
            f"Loaded {row_count} {entity}s spanning {time_scope['start']} to {time_scope['end']} "
            f"(granularity guess: {granularity})."
        )
    else:
        data_summary = f"Loaded {row_count} {entity}s with no clear time column."

    if focus_dims:
        intent = (
            f"You said your primary goal is to '{primary_goal}', anchored on {primary_metric}, "
            f"sliced by {', '.join(focus_dims)}, monitored {granularity}."
        )
    else:
        intent = (
            f"You said your primary goal is to '{primary_goal}', anchored on {primary_metric}, "
            f"monitored {granularity}."
        )

    clarif = context.user_clarifications or {}
    decision = (clarif.get("decision_to_make") or "").strip()
    known_events = (clarif.get("known_events") or "").strip()
    audience = (clarif.get("audience") or "").strip()
    if decision:
        intent += f" Decision you are working on: \"{decision}\"."
    if audience:
        intent += f" Dashboard audience: {audience}."

    goals_recap = [
        {"title": g.get("title", ""), "plain_english": _goal_to_plain_english(g)}
        for g in context.active_goals
    ]

    what_will_be_built = (
        f"I'll build a dashboard centered on {primary_metric}, with KPIs and trend charts at "
        f"{granularity} granularity. For each active goal I'll add a chart that visualizes its "
        "metric and dimension. Insights will prioritize signals that align with your goals; "
        "Q&A will use these focus areas as defaults when your question is ambiguous."
    )

    open_questions: list[str] = []
    if not context.active_goals:
        open_questions.append("No active goals — confirm whether you want to skip goal tracking.")
    if (summary.get("data_quality_signals") or {}).get("small_sample_warning"):
        open_questions.append(
            f"Sample size is small ({row_count} rows); some trends may not be statistically robust."
        )
    if not focus_dims:
        open_questions.append("No focus dimensions selected — dashboard will use the most discriminating dimension by default.")
    if not decision:
        open_questions.append("No decision_to_make captured — insights will be prioritized by goal alignment alone.")
    if not known_events:
        open_questions.append("No known_events captured — anomalies in this period may be flagged that you already know about.")
    if not audience:
        open_questions.append("No audience captured — narrative will use a generic business tone.")
    if known_events:
        open_questions.append(f"Known events you flagged: \"{known_events}\". Anomaly engine will treat these as expected, not flagged.")

    return {
        "headline": headline,
        "data_summary": data_summary,
        "user_intent_summary": intent,
        "goals_recap": goals_recap,
        "what_will_be_built": what_will_be_built,
        "open_questions": open_questions,
        "confidence": "low",
    }


def _validate_confirmation(raw: Any, fallback: dict) -> tuple[dict, list[str]]:
    issues: list[str] = []
    if not isinstance(raw, dict):
        issues.append("not_a_dict")
        return fallback, issues

    merged = dict(fallback)
    for key in REQUIRED_KEYS:
        if key not in raw:
            issues.append(f"missing_key:{key}")
            continue
        value = raw[key]
        if value is None:
            issues.append(f"null_value:{key}")
            continue
        if isinstance(value, str) and not value.strip():
            issues.append(f"empty_string:{key}")
            continue
        if key == "goals_recap" and not isinstance(value, list):
            issues.append("goals_recap_not_list")
            continue
        if key == "open_questions" and not isinstance(value, list):
            issues.append("open_questions_not_list")
            continue
        merged[key] = value

    return merged, issues


def generate_confirmation_summary(
    context: Context,
    debug_logger=None,
) -> dict:
    inp = build_confirmation_input(context)
    fallback = generate_deterministic_confirmation(context)

    if debug_logger:
        debug_logger.log_event(
            "confirmation_input_built",
            {
                "input": inp,
                "fallback_summary": fallback,
            },
        )

    provider = settings.llm_provider
    model_name = resolve_active_model()

    try:
        if debug_logger:
            debug_logger.log_event(
                "confirmation_llm_called",
                {"provider": provider, "model": model_name},
            )

        raw = generate_business_confirmation(inp)

        if debug_logger:
            debug_logger.log_event(
                "confirmation_llm_parsed",
                {"provider": provider, "model": model_name, "parsed": raw},
            )

        merged, issues = _validate_confirmation(raw, fallback)

        result = {
            "source": "llm",
            "provider": provider,
            "model": model_name,
            "summary": merged,
            "input": inp,
            "fallback_baseline": fallback,
            "validation_issues": issues,
            "error": None,
        }

        if debug_logger:
            debug_logger.log_event(
                "confirmation_final",
                {
                    "source": result["source"],
                    "validation_issues": issues,
                    "summary": merged,
                },
            )
        return result

    except Exception as exc:
        if debug_logger:
            debug_logger.log_event(
                "confirmation_fallback_used",
                {
                    "provider": provider,
                    "model": model_name,
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )

        result = {
            "source": "deterministic",
            "provider": provider,
            "model": model_name,
            "summary": fallback,
            "input": inp,
            "fallback_baseline": fallback,
            "validation_issues": [],
            "error": str(exc),
        }
        if debug_logger:
            debug_logger.log_event(
                "confirmation_final",
                {
                    "source": result["source"],
                    "summary": fallback,
                },
            )
        return result
