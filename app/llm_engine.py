from __future__ import annotations

import json
import re
import time
from typing import List

import google.generativeai as genai
from groq import Groq
from openai import OpenAI

from app.config import settings


LLM_TIMEOUT_SECONDS = 120
LLM_MAX_ATTEMPTS = 2  # initial attempt + 1 retry on transient errors
LLM_BACKOFF_BASE_SECONDS = 1.5

_TRANSIENT_ERROR_KEYWORDS = (
    "timed out",
    "timeout",
    "connection",
    "connect timeout",
    "read timeout",
    "rate limit",
    "ratelimit",
    "too many requests",
    "503",
    "502",
    "504",
    "temporarily unavailable",
    "service unavailable",
)


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(keyword in msg for keyword in _TRANSIENT_ERROR_KEYWORDS)


def _build_prompt(
    columns: List[str],
    question: str,
    semantic_hints: dict | None = None,
    context_block: dict | None = None,
    column_types: dict | None = None,
) -> str:
    semantic_section = ""
    if semantic_hints:
        semantic_section = (
            "Business context hints:\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
        )

    column_type_section = ""
    if column_types:
        identifiers = ", ".join(column_types.get("identifier_columns") or []) or "none"
        time_col = column_types.get("time_column") or "none"
        measures = []
        for m in (column_types.get("measure_columns") or []):
            agg = "SUM" if m.get("aggregation_hint") == "sum" else "AVG"
            measures.append(f"{m['name']} ({agg})")
        dims = ", ".join(
            d.get("name", "") for d in (column_types.get("dimension_columns") or [])
        )
        column_type_section = (
            "Column classification (follow these rules STRICTLY):\n"
            f"- IDENTIFIERS — never aggregate, never GROUP BY: {identifiers}\n"
            f"- MEASURES — always wrap in aggregate function shown: {', '.join(measures) or 'none'}\n"
            f"- DIMENSIONS — safe to GROUP BY, filter, slice: {dims or 'none'}\n"
            f"- TIME COLUMN — use for date bucketing or ORDER BY time: {time_col}\n\n"
        )

    context_section = ""
    if context_block:
        context_section = (
            "User-confirmed context (PRIORITIZE OVER HINTS):\n"
            f"{json.dumps(context_block, ensure_ascii=True, default=str)}\n\n"
            "When the user's question is ambiguous, default to filtering or grouping by "
            "focus_dimensions and aggregating focus_metric. Honor active_goals when relevant.\n\n"
        )

    return (
        "You are a SQL expert writing queries for a DuckDB table named 'data'.\n\n"
        f"Available columns: {', '.join(columns)}\n\n"
        f"{column_type_section}"
        f"{semantic_section}"
        f"{context_section}"
        f"Generate a SQL query for:\n\"{question}\"\n\n"
        "Rules:\n"
        "- Only SELECT queries — no INSERT, UPDATE, DELETE, DROP, or CREATE\n"
        "- Use ONLY the columns listed above\n"
        "- Return SQL only — no explanation\n"
        "- Respect column classification above: never GROUP BY an IDENTIFIER column\n"
        "- When aggregating a MEASURE, use the indicated function (SUM or AVG)\n"
        "- For trend questions, GROUP BY the TIME COLUMN and ORDER BY it\n"
        "- For top-N questions, use ORDER BY aggregate DESC LIMIT N\n"
    )


def _normalize_sql(output: str) -> str:
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("sql", "", 1).strip()
    return text.split(";")[0].strip() + ";"


def _generate_text_once(prompt: str) -> str:
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        client = OpenAI(api_key=settings.openai_api_key, timeout=LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing.")
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(
            prompt,
            request_options={"timeout": LLM_TIMEOUT_SECONDS},
        )
        return response.text or ""

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing.")
        client = Groq(api_key=settings.groq_api_key, timeout=LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    raise ValueError("Unsupported LLM_PROVIDER. Use openai, gemini, or groq.")


def _generate_text(prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            return _generate_text_once(prompt)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc):
                raise
            is_last_attempt = attempt + 1 >= LLM_MAX_ATTEMPTS
            if is_last_attempt:
                break
            time.sleep(LLM_BACKOFF_BASE_SECONDS ** (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _extract_json_array(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse chart plan JSON array from LLM output.")
    return match.group(0)


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse JSON object from LLM output.")
    return match.group(0)


def resolve_active_model() -> str:
    provider = settings.llm_provider
    if provider == "openai":
        return settings.openai_model
    if provider == "gemini":
        return settings.gemini_model
    if provider == "groq":
        return settings.groq_model
    return "unknown"


def generate_business_confirmation(input_data: dict) -> dict:
    prompt = (
        "You are a senior business analyst presenting a confirmation recap to a user "
        "before building their BI dashboard. Your job is to summarize the assembled "
        "context (data understanding, user clarifications, business goals) so the user "
        "can VALIDATE whether the system understood them correctly.\n\n"
        f"Assembled context:\n{json.dumps(input_data, ensure_ascii=True, default=str)}\n\n"
        "Output JSON only — no markdown fences, no commentary. All output in English.\n\n"
        "Required JSON keys:\n"
        "- headline: one short sentence describing the business in plain English.\n"
        "- data_summary: 1-2 sentences about what data was loaded (rows, primary entity, time scope).\n"
        "- user_intent_summary: 1-2 sentences about the user's stated focus_metric, focus_dimensions, "
        "primary_goal, and time_granularity.\n"
        "- goals_recap: array of {title, plain_english}. One entry per active_goal in the input. "
        "plain_english should rephrase the goal in conversational language.\n"
        "- what_will_be_built: 2-3 sentences describing what the dashboard, insights, and Q&A "
        "will deliver based on this context.\n"
        "- open_questions: array (0-3 items) of short strings — anything you're uncertain about "
        "or that the user might want to clarify. Empty array is fine.\n"
        "- confidence: \"high\" | \"medium\" | \"low\" — your confidence that the assembled "
        "context will produce useful output.\n\n"
        "Hard rules:\n"
        "- Use ONLY information present in the input. Never invent metrics, columns, or goals.\n"
        "- Reference real column names and the user's actual goal titles.\n"
        "- If active_goals is empty, say so explicitly in user_intent_summary.\n"
        "- Keep each text field concise — this is a recap, not a report.\n"
    )

    raw = _generate_text(prompt)
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response for business confirmation.")

    json_text = _extract_json_object(raw)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Business confirmation output is not a JSON object.")
    return parsed


def generate_goals(input_data: dict) -> list[dict]:
    prompt = (
        "You are a senior business analyst translating a user's intent into a small set of "
        "operational, measurable BUSINESS GOALS that will drive a BI dashboard.\n\n"
        "You receive:\n"
        "- An understanding object (columns, measures, dimensions, time scope, "
        "derived-metric opportunities, business domain)\n"
        "- The user's clarification answers (primary_goal, focus_metric, focus_dimensions, "
        "time_granularity, comparison_window)\n\n"
        f"Inputs:\n{json.dumps(input_data, ensure_ascii=True, default=str)}\n\n"
        "Produce 3 to 5 STRUCTURED goals that downstream code can use directly to plan charts "
        "and prioritize insights. Output JSON ARRAY only — no markdown fences, no commentary. "
        "All in English.\n\n"
        "Each goal object MUST have these keys:\n"
        "- id: short stable string (e.g. \"g_revenue_growth\").\n"
        "- title: short user-facing goal text (under 100 chars).\n"
        "- metric: column name from understanding.measure_columns. NEVER invent.\n"
        "- dimension: column name from understanding.dimension_columns OR null. NEVER invent.\n"
        "- direction: one of \"increase\", \"decrease\", \"stabilize\".\n"
        "- target_pct: numeric percentage target (positive number) or null if no concrete target.\n"
        "- priority: one of \"high\", \"medium\", \"low\".\n"
        "- time_horizon: one of \"daily\", \"weekly\", \"monthly\", \"quarterly\". "
        "Default to user_clarifications.time_granularity when uncertain.\n"
        "- rationale: one sentence connecting this goal to either user_clarifications.primary_goal, "
        "the focus_metric, focus_dimensions, OR a derived_metric_opportunity from understanding.\n"
        "- active: boolean. Default true.\n\n"
        "Hard rules:\n"
        "- 3 to 5 goals. No duplicates (do not repeat the same metric+dimension+direction combo).\n"
        "- The first goal MUST align with user_clarifications.primary_goal.\n"
        "- At least one goal SHOULD reference a focus_dimension when one is selected.\n"
        "- If user_clarifications.primary_goal mentions \"margin\" but no margin column exists, "
        "phrase the goal as a proxy (e.g. focus on price-quantity trade-offs) and explain in rationale.\n"
        "- Use understanding.derived_metric_opportunities to inspire goals beyond raw totals "
        "(e.g. AOV, revenue per customer, concentration share).\n"
        "- Never reference identifier_columns or the time_column as a metric or dimension.\n"
    )

    raw = _generate_text(prompt)
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response for goals.")

    json_text = _extract_json_array(raw)
    parsed = json.loads(json_text)
    if not isinstance(parsed, list):
        raise ValueError("Goals output is not a JSON array.")
    return parsed


def generate_clarifications(input_data: dict) -> list[dict]:
    prompt = (
        "You are a senior business analyst preparing a short clarification interview "
        "with the user before building a dashboard.\n\n"
        "Produce up to 8 SHORT, dataset-specific questions split into two parts:\n"
        "  PART A — 5 STRUCTURED dashboard-config questions (single/multi select):\n"
        "    primary_goal, focus_metric, focus_dimensions, time_granularity, comparison_window.\n"
        "  PART B — 3 FREE-TEXT business-context questions (optional for the user):\n"
        "    decision_to_make, known_events, audience.\n\n"
        "You will receive a structured 'understanding' object with the columns, measures, "
        "dimensions, time scope, and business domain already classified. Use ONLY these.\n\n"
        f"Understanding:\n{json.dumps(input_data, ensure_ascii=True, default=str)}\n\n"
        "Output JSON ARRAY only (no markdown fences, no commentary). All in English.\n\n"
        "Each question object MUST have these keys:\n"
        "- id: short stable string id (e.g. \"q_primary_goal\").\n"
        "- key: snake_case key. Use EXACTLY one of: primary_goal, focus_metric, focus_dimensions, "
        "time_granularity, comparison_window, decision_to_make, known_events, audience.\n"
        "- question: short, user-facing question text. You MAY personalize the wording.\n"
        "- type: one of \"single_select\", \"multi_select\", \"free_text\".\n"
        "- options: array of strings. REQUIRED for single_select and multi_select. "
        "For free_text use an empty array.\n"
        "- default: For single_select, must be one of options. For multi_select, an array (subset of options). "
        "For free_text, use an empty string.\n"
        "- why_asked: one sentence explaining why this question matters for the dashboard or insights.\n\n"
        "Hard rules for PART A (structured):\n"
        "- For focus_metric options use ONLY names from understanding.measure_columns.\n"
        "- For focus_dimensions options use ONLY names from understanding.dimension_columns "
        "(exclude identifier_columns and the time_column).\n"
        "- Default focus_metric = understanding.primary_metric when present.\n"
        "- Default time_granularity = understanding.time_granularity_guess when present.\n"
        "- Tailor primary_goal options to the inferred business domain "
        "(e.g. retail: \"Grow revenue\", \"Improve margins\", \"Reduce concentration risk\", \"Expand into new regions\").\n\n"
        "Hard rules for PART B (free-text business context):\n"
        "- Use type=\"free_text\", options=[], default=\"\".\n"
        "- Frame the question to invite a 1-2 sentence answer.\n"
        "- These questions are OPTIONAL for the user — make that clear in the wording or why_asked.\n"
        "- decision_to_make: ask for the business decision the user wants to make from this data.\n"
        "- known_events: ask about anomalies, campaigns, supply changes, system migrations the user knows about.\n"
        "- audience: ask who will look at the dashboard.\n\n"
        "General rules:\n"
        "- 7 to 8 questions total (5 PART A + 2 to 3 PART B). No duplicates of keys.\n"
        "- Never invent column names. Every column referenced must appear in the understanding.\n"
    )

    raw = _generate_text(prompt)
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response for clarification questions.")

    json_text = _extract_json_array(raw)
    parsed = json.loads(json_text)
    if not isinstance(parsed, list):
        raise ValueError("Clarification questions output is not a JSON array.")
    return parsed


def generate_dataset_understanding(input_data: dict) -> dict:
    prompt = (
        "You are a senior data analyst producing a STRUCTURAL understanding of a dataset for "
        "internal AI consumption. Your output is NOT shown to end users — it is consumed by "
        "downstream pipeline stages (clarification questions, goals, dashboards, insights) "
        "that each generate their own user-facing output.\n\n"
        "The input profile already includes deterministic FACTS (column classification, time scope, "
        "data quality signals, derived-metric formula matches). Trust them — do not reclassify "
        "columns, do not infer the time scope, do not change identifier/measure/dimension "
        "assignments.\n\n"
        "Your job: layer BUSINESS INTERPRETATION on top of these facts so downstream stages "
        "have the context they need. Do NOT generate KPIs, dashboards, or business questions "
        "— other stages handle that.\n\n"
        "Output JSON only. No markdown fences, no commentary. All output in English.\n\n"
        f"Dataset profile:\n{json.dumps(input_data, ensure_ascii=True, default=str)}\n\n"
        "Required JSON keys:\n"
        "- dataset_description: 1-2 plain English sentences. Name the entity, the primary metric, "
        "and the key dimensions. This drives the natural-language framing in later stages.\n"
        "- business_domain_guess: short label (e.g. \"retail sales\", \"web analytics\", \"finance\", \"healthcare ops\"). "
        "Drives clarification-question option tailoring and narrator framing.\n"
        "- primary_entity: what each row represents (e.g. \"a sales transaction\"). Used by the "
        "confirmation step and the narrator.\n"
        "- derived_metric_opportunities: array (0-5 items) of {name, formula, why_useful}. "
        "Suggest derivations the deterministic engine did NOT already detect (it already finds "
        "formula matches like revenue = price * quantity). Focus on ratio metrics and cohort-style "
        "derivations: AOV, revenue per customer, repeat-purchase rate, units per order. These "
        "feed the goal generator.\n"
        "- temporal_features: object with keys has_time, time_column, suggested_granularity, "
        "seasonality_candidates (array, e.g. [\"day_of_week\", \"month\", \"quarter\"]). "
        "ONLY seasonality_candidates is yours to fill — has_time, time_column, and "
        "suggested_granularity are overridden by deterministic values.\n"
        "- data_quality_notes: array of short observations (consistency checks, suspicious ranges, "
        "sample-size warnings). Surfaces in the confirmation step's open-questions list.\n"
        "- confidence: \"high\" | \"medium\" | \"low\" — your confidence in the structural interpretation.\n\n"
        "Hard rules:\n"
        "- NEVER invent column names. Every column referenced must appear in profile.columns.\n"
        "- NEVER list identifier columns as measures or dimensions.\n"
        "- NEVER use the time column as a slicing dimension.\n"
        "- Keep all text fields concise — this is structured input for other AI stages, not a report.\n"
    )

    raw = _generate_text(prompt)
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response for dataset understanding.")

    json_text = _extract_json_object(raw)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Dataset understanding output is not a JSON object.")
    return parsed


def generate_sql(
    question: str,
    columns: List[str],
    semantic_hints: dict | None = None,
    context_block: dict | None = None,
    column_types: dict | None = None,
) -> str:
    prompt = _build_prompt(
        columns,
        question,
        semantic_hints=semantic_hints,
        context_block=context_block,
        column_types=column_types,
    )
    text = _generate_text(prompt)
    return _normalize_sql(text)


def suggest_dashboard_plan(
    columns: List[str],
    date_column: str | None,
    numeric_column: str | None,
    category_column: str | None,
    semantic_hints: dict | None = None,
    context=None,
    max_items: int = 5,
) -> List[dict]:
    semantic_block = ""
    if semantic_hints:
        semantic_block = (
            "Business context hints (high priority):\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
        )

    context_block = ""
    if context is not None:
        context_payload = context.to_prompt_block()
        context_block = (
            "User-confirmed context (HIGHEST priority — override hints when in conflict):\n"
            f"{json.dumps(context_payload, ensure_ascii=True, default=str)}\n\n"
            "Build the dashboard around focus_metric, focus_dimensions, and the active_goals. "
            "Each active goal should map to at least one chart that helps the user track it. "
            "Prefer a chart that lets the user see whether the goal's direction "
            "(increase/decrease/stabilize) is being met.\n\n"
        )

    prompt = (
        "You are an expert BI dashboard planner.\n\n"
        "Table name: data\n"
        f"Columns: {', '.join(columns)}\n"
        f"Detected date column: {date_column}\n"
        f"Detected numeric column: {numeric_column}\n"
        f"Detected category column: {category_column}\n\n"
        f"{semantic_block}"
        f"{context_block}"
        "Return ONLY a JSON array (no explanation) with up to "
        f"{max_items} objects.\n"
        "Each object MUST have keys: kind, title, sql.\n"
        "Allowed kind values: kpi, line, bar, hist.\n"
        "Rules:\n"
        "- SQL must be a single SELECT statement from table data\n"
        "- Use only provided columns\n"
        "- For line chart SQL aliases must be: dt, value\n"
        "- For bar chart SQL aliases must be: category, total\n"
        "- For histogram SQL alias must be: value\n"
        "- For KPI return one numeric value\n"
        "- Prioritize useful diversity instead of repetitive charts\n"
    )

    raw_text = _generate_text(prompt)
    json_text = _extract_json_array(raw_text)
    parsed = json.loads(json_text)

    if not isinstance(parsed, list):
        raise ValueError("Dashboard plan should be a JSON list.")
    return parsed


def generate_insight_narrative(
    signals: dict,
    semantic_hints: dict | None = None,
    context_block: dict | None = None,
) -> str:
    hints_block = ""
    if semantic_hints:
        hints_block = (
            "Semantic hints:\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
        )

    context_section = ""
    if context_block:
        context_section = (
            "User-confirmed context (anchor every insight to these goals when relevant):\n"
            f"{json.dumps(context_block, ensure_ascii=True, default=str)}\n\n"
            "Framing rules:\n"
            "- When a signal advances an active_goal (same direction or improvement toward it), "
            "frame it as \"advances goal 'X'\".\n"
            "- When a signal moves OPPOSITE to an active_goal direction, frame it as \"threatens "
            "goal 'X'\" — even if the absolute change looks positive in raw terms. For example, if "
            "a goal direction is 'decrease' and the metric grew, that growth is BAD news for the user.\n"
            "- For 'stabilize' goals, treat any large swing (in either direction) as a threat.\n"
            "- The signals dict includes goal_consensus_direction — use it to disambiguate framing "
            "when the signal's wanted_direction differs from the raw movement.\n"
            "- Anchor anomaly bullets explicitly: name which goal the anomaly threatens (or note "
            "if it advances one).\n"
            "- Time-period terminology: use the label from signals.growth.primary_period_label "
            "(e.g. WoW, MoM, QoQ) instead of always saying \"MoM\". The time_granularity field "
            "tells you the user's chosen monitoring cadence.\n"
            "- Benchmark phrasing: signals.benchmarks.method tells you whether the comparison is "
            "vs the previous period, year-over-year, or a 6-period rolling average. Frame the "
            "delta_vs_baseline_pct accordingly (e.g. \"down 5% versus the same period last year\" "
            "vs \"down 5% versus the rolling 6-period baseline\").\n"
            "Business-context rules (when business_context fields are non-empty):\n"
            "- If business_context.audience is set, tailor language to that audience "
            "(e.g. CEO -> high-level + dollar amounts; ops team -> operational details).\n"
            "- If business_context.decision_to_make is set, prioritize bullets that directly inform "
            "that decision; reorder if needed.\n"
            "- If business_context.known_events is set, do NOT flag those events as anomalies. "
            "Reference them as known context (e.g. \"as expected from the campaign on March 15\").\n\n"
        )

    prompt = (
        "You are a business analyst.\n"
        "Based ONLY on the structured signals below, generate practical actionable insights.\n"
        "Do not invent any metrics or values not present in signals.\n"
        "Use short bullet points (6-8 bullets).\n"
        "Each bullet should be specific and business-oriented.\n"
        "Prioritize output order: high, medium, low.\n"
        "For each key issue include a concrete next action.\n"
        "Translate anomalies into plain business language with date examples.\n"
        "Include 1-2 cross-signal narratives combining growth, contributors, and risk.\n"
        "Use benchmark context from signals when available to explain whether performance is strong or weak.\n"
        "Prefix each bullet with a priority tag: [HIGH], [MEDIUM], or [LOW].\n\n"
        f"{hints_block}"
        f"{context_section}"
        "Structured signals:\n"
        f"{json.dumps(signals, ensure_ascii=True)}\n"
    )

    text = _generate_text(prompt).strip()
    if not text:
        raise ValueError("Empty LLM response for insight narrative.")
    return text
