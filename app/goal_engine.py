from __future__ import annotations

from typing import Any

from app.config import settings
from app.llm_engine import generate_goals, resolve_active_model


ALLOWED_DIRECTIONS = {"increase", "decrease", "stabilize"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
ALLOWED_HORIZONS = {"daily", "weekly", "monthly", "quarterly"}

MIN_GOALS = 3
MAX_GOALS = 5

TARGET_PCT_MIN = -100.0
TARGET_PCT_MAX = 1000.0


def build_goal_input(understanding_summary: dict, user_clarifications: dict) -> dict:
    summary = understanding_summary or {}
    clarif = user_clarifications or {}
    measures = [m.get("name") for m in (summary.get("measure_columns") or []) if m.get("name")]
    dimensions = [d.get("name") for d in (summary.get("dimension_columns") or []) if d.get("name")]
    identifiers = list(summary.get("identifier_columns") or [])
    time_scope = summary.get("time_scope") or {}
    derived = summary.get("derived_metric_opportunities") or []

    return {
        "business_domain_guess": summary.get("business_domain_guess"),
        "primary_entity": summary.get("primary_entity"),
        "primary_metric": summary.get("primary_metric"),
        "measure_columns": measures,
        "dimension_columns": dimensions,
        "identifier_columns": identifiers,
        "time_column": time_scope.get("time_column"),
        "time_granularity_guess": time_scope.get("granularity_guess"),
        "derived_metric_opportunities": [
            {"name": d.get("name"), "formula": d.get("formula"), "why_useful": d.get("why_useful")}
            for d in derived
        ],
        "user_clarifications": {
            "primary_goal": clarif.get("primary_goal"),
            "focus_metric": clarif.get("focus_metric"),
            "focus_dimensions": clarif.get("focus_dimensions") or [],
            "time_granularity": clarif.get("time_granularity"),
            "comparison_window": clarif.get("comparison_window"),
        },
    }


def _coerce_target_pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < TARGET_PCT_MIN:
        num = TARGET_PCT_MIN
    if num > TARGET_PCT_MAX:
        num = TARGET_PCT_MAX
    return round(num, 2)


def _validate_goal(
    raw: Any,
    measures: list[str],
    dimensions: list[str],
    identifiers: list[str],
    time_column: str | None,
    default_horizon: str | None,
) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_a_dict"

    gid = str(raw.get("id", "")).strip()
    title = str(raw.get("title", "")).strip()
    metric = raw.get("metric")
    dimension = raw.get("dimension")
    direction = str(raw.get("direction", "")).strip().lower()
    priority = str(raw.get("priority", "medium")).strip().lower()
    horizon = str(raw.get("time_horizon", "")).strip().lower()
    rationale = str(raw.get("rationale", "")).strip()
    target_pct = _coerce_target_pct(raw.get("target_pct"))
    active = raw.get("active", True)

    if not gid or not title:
        return None, "missing_id_or_title"

    if direction not in ALLOWED_DIRECTIONS:
        return None, f"invalid_direction:{direction}"

    if not isinstance(metric, str) or metric not in measures:
        return None, "metric_not_in_measures"

    if metric in identifiers or metric == time_column:
        return None, "metric_is_identifier_or_time"

    if dimension is not None and dimension != "":
        if not isinstance(dimension, str) or dimension not in dimensions:
            return None, "dimension_not_in_dimensions"
        if dimension in identifiers or dimension == time_column:
            return None, "dimension_is_identifier_or_time"
    else:
        dimension = None

    if priority not in ALLOWED_PRIORITIES:
        priority = "medium"
    if horizon not in ALLOWED_HORIZONS:
        horizon = default_horizon if default_horizon in ALLOWED_HORIZONS else "weekly"

    if not isinstance(active, bool):
        active = bool(active) if active is not None else True

    return (
        {
            "id": gid,
            "title": title,
            "metric": metric,
            "dimension": dimension,
            "direction": direction,
            "target_pct": target_pct,
            "priority": priority,
            "time_horizon": horizon,
            "rationale": rationale or "Derived from user clarifications and dataset understanding.",
            "source": "llm",
            "active": active,
        },
        None,
    )


def generate_deterministic_goals(
    understanding_summary: dict,
    user_clarifications: dict,
) -> list[dict]:
    inp = build_goal_input(understanding_summary, user_clarifications)
    measures = inp["measure_columns"]
    dimensions = inp["dimension_columns"]
    primary_metric = inp["user_clarifications"].get("focus_metric") or inp["primary_metric"]
    focus_dims = inp["user_clarifications"].get("focus_dimensions") or []
    horizon = inp["user_clarifications"].get("time_granularity") or inp["time_granularity_guess"] or "weekly"
    primary_goal_text = inp["user_clarifications"].get("primary_goal") or ""

    goals: list[dict] = []
    if primary_metric and primary_metric in measures:
        goals.append(
            {
                "id": "g_primary_growth",
                "title": f"Improve {primary_metric} aligned to '{primary_goal_text}'"
                if primary_goal_text
                else f"Grow {primary_metric}",
                "metric": primary_metric,
                "dimension": None,
                "direction": "increase",
                "target_pct": 10.0,
                "priority": "high",
                "time_horizon": horizon,
                "rationale": f"Anchored to user-selected primary_goal ('{primary_goal_text}') and focus_metric '{primary_metric}'.",
                "source": "deterministic",
                "active": True,
            }
        )

    if primary_metric and focus_dims:
        first_dim = next((d for d in focus_dims if d in dimensions), None)
        if first_dim:
            goals.append(
                {
                    "id": "g_focus_dimension_share",
                    "title": f"Diversify {primary_metric} across {first_dim}",
                    "metric": primary_metric,
                    "dimension": first_dim,
                    "direction": "stabilize",
                    "target_pct": None,
                    "priority": "medium",
                    "time_horizon": horizon,
                    "rationale": f"User flagged '{first_dim}' as a focus dimension; reduce single-segment dependency.",
                    "source": "deterministic",
                    "active": True,
                }
            )

    if primary_metric and dimensions:
        weakest_target_dim = next(
            (d for d in dimensions if d not in focus_dims), dimensions[0]
        )
        goals.append(
            {
                "id": "g_segment_recovery",
                "title": f"Reverse decline in weakest {weakest_target_dim} segments",
                "metric": primary_metric,
                "dimension": weakest_target_dim,
                "direction": "increase",
                "target_pct": 5.0,
                "priority": "medium",
                "time_horizon": horizon,
                "rationale": f"Lift declining {weakest_target_dim} segments to protect overall {primary_metric} growth.",
                "source": "deterministic",
                "active": True,
            }
        )

    return goals[:MAX_GOALS]


def _signature_key(goal: dict) -> tuple[str, str | None, str]:
    return (goal.get("metric"), goal.get("dimension"), goal.get("direction"))


def _merge_goals(
    llm_goals: list[dict],
    fallback_goals: list[dict],
) -> tuple[list[dict], list[str]]:
    merged: list[dict] = []
    seen: set[tuple[str, str | None, str]] = set()
    used_ids: list[str] = []

    for g in llm_goals:
        key = _signature_key(g)
        if key in seen:
            continue
        seen.add(key)
        merged.append(g)
        used_ids.append(g["id"])
        if len(merged) >= MAX_GOALS:
            return merged, used_ids

    for g in fallback_goals:
        key = _signature_key(g)
        if key in seen:
            continue
        seen.add(key)
        merged.append(g)
        used_ids.append(g["id"])
        if len(merged) >= MAX_GOALS:
            break

    return merged, used_ids


def generate_goals_for_user(
    understanding_summary: dict,
    user_clarifications: dict,
    debug_logger=None,
) -> dict:
    inp = build_goal_input(understanding_summary, user_clarifications)
    fallback = generate_deterministic_goals(understanding_summary, user_clarifications)

    if debug_logger:
        debug_logger.log_event(
            "goals_input_built",
            {
                "input": inp,
                "fallback_goal_count": len(fallback),
                "fallback_ids": [g["id"] for g in fallback],
            },
        )

    provider = settings.llm_provider
    model_name = resolve_active_model()

    try:
        if debug_logger:
            debug_logger.log_event(
                "goals_llm_called",
                {"provider": provider, "model": model_name},
            )

        raw_goals = generate_goals(inp)

        if debug_logger:
            debug_logger.log_event(
                "goals_llm_parsed",
                {
                    "provider": provider,
                    "model": model_name,
                    "raw_count": len(raw_goals),
                    "raw_goals": raw_goals,
                },
            )

        accepted: list[dict] = []
        rejections: list[dict] = []
        default_horizon = inp["user_clarifications"].get("time_granularity") or inp["time_granularity_guess"]
        for raw in raw_goals:
            cleaned, reason = _validate_goal(
                raw,
                measures=inp["measure_columns"],
                dimensions=inp["dimension_columns"],
                identifiers=inp["identifier_columns"],
                time_column=inp["time_column"],
                default_horizon=default_horizon,
            )
            if cleaned:
                accepted.append(cleaned)
            else:
                rejections.append({"raw": raw, "reason": reason})

        if debug_logger:
            debug_logger.log_event(
                "goals_validation",
                {
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejections),
                    "rejections": rejections,
                },
            )

        if len(accepted) < MIN_GOALS and debug_logger:
            debug_logger.log_event(
                "goals_topup_with_fallback",
                {
                    "reason": "llm_yielded_fewer_than_min_goals",
                    "accepted_count": len(accepted),
                    "min_required": MIN_GOALS,
                },
            )

        merged, used_ids = _merge_goals(accepted, fallback)

        result = {
            "source": "llm" if accepted else "deterministic",
            "provider": provider,
            "model": model_name,
            "goals": merged,
            "used_ids": used_ids,
            "input": inp,
            "fallback": fallback,
            "validation_rejections": rejections,
            "error": None,
        }

        if debug_logger:
            debug_logger.log_event(
                "goals_final",
                {
                    "source": result["source"],
                    "goal_count": len(merged),
                    "used_ids": used_ids,
                    "goals": merged,
                },
            )
        return result

    except Exception as exc:
        if debug_logger:
            debug_logger.log_event(
                "goals_fallback_used",
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
            "goals": fallback,
            "used_ids": [g["id"] for g in fallback],
            "input": inp,
            "fallback": fallback,
            "validation_rejections": [],
            "error": str(exc),
        }

        if debug_logger:
            debug_logger.log_event(
                "goals_final",
                {
                    "source": result["source"],
                    "goal_count": len(fallback),
                    "used_ids": result["used_ids"],
                    "goals": fallback,
                },
            )
        return result


def normalize_user_goals(
    edited_rows: list[dict],
    measures: list[str],
    dimensions: list[str],
    identifiers: list[str],
    time_column: str | None,
    default_horizon: str | None,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[str, str | None, str]] = set()

    for idx, row in enumerate(edited_rows):
        if not isinstance(row, dict):
            rejected.append({"row_index": idx, "row": row, "reason": "not_a_dict"})
            continue
        candidate = dict(row)
        if not candidate.get("id"):
            candidate["id"] = f"g_user_{idx}"
        if "source" not in candidate or not str(candidate.get("source") or "").strip():
            candidate["source"] = "user_added"

        cleaned, reason = _validate_goal(
            candidate,
            measures=measures,
            dimensions=dimensions,
            identifiers=identifiers,
            time_column=time_column,
            default_horizon=default_horizon,
        )
        if not cleaned:
            rejected.append({"row_index": idx, "row": row, "reason": reason})
            continue

        if str(candidate.get("source")).strip() in {"deterministic", "user_added"}:
            cleaned["source"] = candidate["source"]

        sig = _signature_key(cleaned)
        if sig in seen:
            rejected.append({"row_index": idx, "row": row, "reason": "duplicate_metric_dim_direction"})
            continue
        seen.add(sig)
        accepted.append(cleaned)

    return accepted, rejected
