from __future__ import annotations

from typing import Any

from app.config import settings
from app.llm_engine import generate_clarifications, resolve_active_model


ALLOWED_TYPES = {"single_select", "multi_select", "free_text"}
MAX_QUESTIONS = 8
MIN_QUESTIONS = 4
EXPECTED_KEYS = {
    "primary_goal",
    "focus_metric",
    "focus_dimensions",
    "time_granularity",
    "comparison_window",
    "decision_to_make",
    "known_events",
    "audience",
}


def build_clarification_input(understanding_summary: dict) -> dict:
    summary = understanding_summary or {}
    measures = [m.get("name") for m in (summary.get("measure_columns") or []) if m.get("name")]
    dimensions = [d.get("name") for d in (summary.get("dimension_columns") or []) if d.get("name")]
    identifiers = list(summary.get("identifier_columns") or [])
    time_scope = summary.get("time_scope") or {}

    return {
        "business_domain_guess": summary.get("business_domain_guess"),
        "primary_entity": summary.get("primary_entity"),
        "primary_metric": summary.get("primary_metric"),
        "measure_columns": measures,
        "dimension_columns": dimensions,
        "identifier_columns": identifiers,
        "time_column": time_scope.get("time_column"),
        "time_granularity_guess": time_scope.get("granularity_guess"),
        "has_time": bool(time_scope.get("has_time")),
    }


def _validate_question(
    raw: Any,
    measures: list[str],
    dimensions: list[str],
    time_column: str | None,
    identifiers: list[str],
) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_a_dict"

    qid = str(raw.get("id", "")).strip()
    key = str(raw.get("key", "")).strip()
    text = str(raw.get("question", "")).strip()
    qtype = str(raw.get("type", "")).strip().lower()
    options = raw.get("options")
    default = raw.get("default")
    why = str(raw.get("why_asked", "")).strip()

    if not qid or not key or not text or qtype not in ALLOWED_TYPES:
        return None, "missing_or_invalid_basic_fields"

    if qtype in {"single_select", "multi_select"}:
        if not isinstance(options, list) or not options:
            return None, "missing_options"
        options = [str(o).strip() for o in options if str(o).strip()]
        if not options:
            return None, "empty_options_after_clean"

        if key == "focus_metric":
            options = [o for o in options if o in measures]
            if not options:
                return None, "focus_metric_options_not_in_measures"
        if key == "focus_dimensions":
            blocked = set(identifiers)
            if time_column:
                blocked.add(time_column)
            options = [o for o in options if o in dimensions and o not in blocked]
            if not options:
                return None, "focus_dimensions_options_not_in_dimensions"

        if qtype == "single_select":
            if not isinstance(default, str) or default not in options:
                default = options[0]
        else:
            if not isinstance(default, list):
                default = [options[0]]
            else:
                default = [str(d).strip() for d in default if str(d).strip() in options]
                if not default:
                    default = [options[0]]
    else:
        options = []
        default = "" if not isinstance(default, str) else default

    return (
        {
            "id": qid,
            "key": key,
            "question": text,
            "type": qtype,
            "options": options,
            "default": default,
            "why_asked": why or "Helps tailor downstream goals, dashboard, and insights.",
            "source": "llm",
        },
        None,
    )


def generate_deterministic_questions(understanding_summary: dict) -> list[dict]:
    inp = build_clarification_input(understanding_summary)
    measures = inp["measure_columns"]
    dimensions = inp["dimension_columns"]
    primary_metric = inp["primary_metric"]
    granularity_guess = inp["time_granularity_guess"]
    has_time = inp["has_time"]

    questions: list[dict] = []

    domain_options = ["Grow the primary metric", "Improve efficiency", "Reduce concentration risk", "Other"]
    questions.append(
        {
            "id": "q_primary_goal",
            "key": "primary_goal",
            "question": "What is your primary business goal for this dataset?",
            "type": "single_select",
            "options": domain_options,
            "default": domain_options[0],
            "why_asked": "Drives goal generation, dashboard prioritization, and which insights to surface.",
            "source": "deterministic",
        }
    )

    if measures:
        default_metric = primary_metric if primary_metric in measures else measures[0]
        questions.append(
            {
                "id": "q_focus_metric",
                "key": "focus_metric",
                "question": "Which metric matters most?",
                "type": "single_select",
                "options": measures,
                "default": default_metric,
                "why_asked": "All KPIs and trends will be anchored to this metric.",
                "source": "deterministic",
            }
        )

    if dimensions:
        questions.append(
            {
                "id": "q_focus_dimensions",
                "key": "focus_dimensions",
                "question": "Which breakdowns matter most? (pick 1-2)",
                "type": "multi_select",
                "options": dimensions,
                "default": [dimensions[0]],
                "why_asked": "Determines which dimension breakdowns we surface in charts and insights.",
                "source": "deterministic",
            }
        )

    if has_time:
        granularity_options = ["daily", "weekly", "monthly", "quarterly"]
        default_granularity = granularity_guess if granularity_guess in granularity_options else "weekly"
        questions.append(
            {
                "id": "q_time_granularity",
                "key": "time_granularity",
                "question": "What time granularity do you want to monitor?",
                "type": "single_select",
                "options": granularity_options,
                "default": default_granularity,
                "why_asked": "Controls trend chart bucketing and the period we use for growth comparisons.",
                "source": "deterministic",
            }
        )

    comparison_options = [
        "previous period",
        "same period last year",
        "6-period rolling average",
    ]
    questions.append(
        {
            "id": "q_comparison_window",
            "key": "comparison_window",
            "question": "What comparison baseline do you want to use?",
            "type": "single_select",
            "options": comparison_options,
            "default": comparison_options[2],
            "why_asked": "Shapes growth signals, benchmark context, and how we frame regressions.",
            "source": "deterministic",
        }
    )

    questions.append(
        {
            "id": "q_decision_to_make",
            "key": "decision_to_make",
            "question": "In one sentence, what business decision are you trying to make from this data?",
            "type": "free_text",
            "options": [],
            "default": "",
            "why_asked": (
                "Lets the AI prioritize signals and frame insights toward the decision you are working on. "
                "Optional — leave blank if not yet clear."
            ),
            "source": "deterministic",
        }
    )
    questions.append(
        {
            "id": "q_known_events",
            "key": "known_events",
            "question": "Any known events or anomalies in this period? (campaigns, supply issues, system changes — leave blank if none)",
            "type": "free_text",
            "options": [],
            "default": "",
            "why_asked": (
                "Without this, the engine can flag normal events as anomalies. With it, the narrator "
                "can frame them in business terms and avoid false alerts."
            ),
            "source": "deterministic",
        }
    )
    questions.append(
        {
            "id": "q_audience",
            "key": "audience",
            "question": "Who will look at this dashboard? (e.g. CEO, sales team, operations)",
            "type": "free_text",
            "options": [],
            "default": "",
            "why_asked": (
                "Tailors the language and level of detail in dashboard titles and insight bullets."
            ),
            "source": "deterministic",
        }
    )

    return questions[:MAX_QUESTIONS]


def _merge_questions(
    llm_questions: list[dict],
    fallback_questions: list[dict],
) -> tuple[list[dict], list[str]]:
    seen_keys: set[str] = set()
    merged: list[dict] = []
    used_keys: list[str] = []

    for q in llm_questions:
        key = q.get("key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(q)
        used_keys.append(key)
        if len(merged) >= MAX_QUESTIONS:
            return merged, used_keys

    for q in fallback_questions:
        key = q.get("key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(q)
        used_keys.append(key)
        if len(merged) >= MAX_QUESTIONS:
            break

    return merged, used_keys


def generate_clarification_questions(
    understanding_summary: dict,
    debug_logger=None,
) -> dict:
    inp = build_clarification_input(understanding_summary)
    fallback = generate_deterministic_questions(understanding_summary)

    if debug_logger:
        debug_logger.log_event(
            "clarifications_input_built",
            {
                "input": inp,
                "fallback_question_count": len(fallback),
                "fallback_keys": [q["key"] for q in fallback],
            },
        )

    provider = settings.llm_provider
    model_name = resolve_active_model()

    try:
        if debug_logger:
            debug_logger.log_event(
                "clarifications_llm_called",
                {"provider": provider, "model": model_name},
            )

        raw_questions = generate_clarifications(inp)

        if debug_logger:
            debug_logger.log_event(
                "clarifications_llm_parsed",
                {
                    "provider": provider,
                    "model": model_name,
                    "raw_count": len(raw_questions),
                    "raw_questions": raw_questions,
                },
            )

        accepted: list[dict] = []
        rejections: list[dict] = []
        for raw in raw_questions:
            cleaned, reason = _validate_question(
                raw,
                measures=inp["measure_columns"],
                dimensions=inp["dimension_columns"],
                time_column=inp["time_column"],
                identifiers=inp["identifier_columns"],
            )
            if cleaned:
                accepted.append(cleaned)
            else:
                rejections.append({"raw": raw, "reason": reason})

        if debug_logger:
            debug_logger.log_event(
                "clarifications_validation",
                {
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejections),
                    "rejections": rejections,
                },
            )

        if len(accepted) < MIN_QUESTIONS:
            if debug_logger:
                debug_logger.log_event(
                    "clarifications_topup_with_fallback",
                    {
                        "reason": "llm_yielded_fewer_than_min_questions",
                        "accepted_count": len(accepted),
                        "min_required": MIN_QUESTIONS,
                    },
                )

        merged, used_keys = _merge_questions(accepted, fallback)

        result = {
            "source": "llm" if accepted else "deterministic",
            "provider": provider,
            "model": model_name,
            "questions": merged,
            "used_keys": used_keys,
            "input": inp,
            "fallback": fallback,
            "validation_rejections": rejections,
            "error": None,
        }

        if debug_logger:
            debug_logger.log_event(
                "clarifications_final",
                {
                    "source": result["source"],
                    "question_count": len(merged),
                    "used_keys": used_keys,
                    "questions": merged,
                },
            )
        return result

    except Exception as exc:
        if debug_logger:
            debug_logger.log_event(
                "clarifications_fallback_used",
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
            "questions": fallback,
            "used_keys": [q["key"] for q in fallback],
            "input": inp,
            "fallback": fallback,
            "validation_rejections": [],
            "error": str(exc),
        }
        if debug_logger:
            debug_logger.log_event(
                "clarifications_final",
                {
                    "source": result["source"],
                    "question_count": len(fallback),
                    "used_keys": result["used_keys"],
                    "questions": fallback,
                },
            )
        return result


def normalize_user_answers(
    questions: list[dict],
    raw_answers: dict[str, Any],
) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for q in questions:
        key = q["key"]
        qtype = q["type"]
        value = raw_answers.get(key, q.get("default"))
        if qtype == "multi_select":
            if not isinstance(value, list):
                value = [value] if value else []
            allowed = set(q.get("options") or [])
            value = [v for v in value if v in allowed] or list(q.get("default") or [])
        elif qtype == "single_select":
            allowed = set(q.get("options") or [])
            if value not in allowed:
                value = q.get("default")
        else:
            value = str(value or "").strip()
        clean[key] = value
    return clean
