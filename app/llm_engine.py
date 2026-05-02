from __future__ import annotations

import json
import re
from typing import List

import google.generativeai as genai
from groq import Groq
from openai import OpenAI

from app.config import settings


def _build_prompt(columns: List[str], question: str, semantic_hints: dict | None = None) -> str:
    semantic_block = ""
    if semantic_hints:
        semantic_block = (
            "Business context hints:\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
        )

    return (
        "You are a SQL expert.\n\n"
        "Table name: data\n"
        f"Columns: {', '.join(columns)}\n\n"
        f"{semantic_block}"
        f"Generate a SQL query for:\n\"{question}\"\n\n"
        "Rules:\n"
        "- Only SELECT queries\n"
        "- No modification queries\n"
        "- Use only given columns\n"
        "- Return only SQL\n"
    )


def _normalize_sql(output: str) -> str:
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("sql", "", 1).strip()
    return text.split(";")[0].strip() + ";"


def _generate_text(prompt: str) -> str:
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        client = OpenAI(api_key=settings.openai_api_key)
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
        response = model.generate_content(prompt)
        return response.text or ""

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing.")
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    raise ValueError("Unsupported LLM_PROVIDER. Use openai, gemini, or groq.")


def _extract_json_array(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse chart plan JSON array from LLM output.")
    return match.group(0)


def generate_sql(question: str, columns: List[str], semantic_hints: dict | None = None) -> str:
    prompt = _build_prompt(columns, question, semantic_hints=semantic_hints)
    text = _generate_text(prompt)
    return _normalize_sql(text)


def suggest_dashboard_plan(
    columns: List[str],
    date_column: str | None,
    numeric_column: str | None,
    category_column: str | None,
    semantic_hints: dict | None = None,
    max_items: int = 5,
) -> List[dict]:
    semantic_block = ""
    if semantic_hints:
        semantic_block = (
            "Business context hints (high priority):\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
        )

    prompt = (
        "You are an expert BI dashboard planner.\n\n"
        "Table name: data\n"
        f"Columns: {', '.join(columns)}\n"
        f"Detected date column: {date_column}\n"
        f"Detected numeric column: {numeric_column}\n"
        f"Detected category column: {category_column}\n\n"
        f"{semantic_block}"
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


def generate_insight_narrative(signals: dict, semantic_hints: dict | None = None) -> str:
    hints_block = ""
    if semantic_hints:
        hints_block = (
            "Semantic hints:\n"
            f"{json.dumps(semantic_hints, ensure_ascii=True)}\n\n"
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
        "Structured signals:\n"
        f"{json.dumps(signals, ensure_ascii=True)}\n"
    )

    text = _generate_text(prompt).strip()
    if not text:
        raise ValueError("Empty LLM response for insight narrative.")
    return text
