from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.data_engine import SchemaInfo
from app.llm_engine import suggest_dashboard_plan
from app.sql_safety import is_safe_query


@dataclass
class DashboardItem:
    kind: str
    title: str
    sql: str


ALLOWED_KINDS = {"kpi", "line", "bar", "hist"}


def _rule_based_dashboard_plan(schema: SchemaInfo, semantic_hints: dict | None = None) -> List[DashboardItem]:
    items: List[DashboardItem] = []
    metric_col = schema.numeric_column
    date_col = schema.date_column
    category_col = schema.category_column

    if semantic_hints:
        hinted_metric = semantic_hints.get("primary_metric")
        hinted_time = semantic_hints.get("time_column")
        hinted_dims = semantic_hints.get("dimensions") or []

        if isinstance(hinted_metric, str) and hinted_metric in schema.columns:
            metric_col = hinted_metric
        if isinstance(hinted_time, str) and hinted_time in schema.columns:
            date_col = hinted_time
        if hinted_dims and isinstance(hinted_dims, list):
            first_dimension = next((d for d in hinted_dims if isinstance(d, str) and d in schema.columns), None)
            if first_dimension:
                category_col = first_dimension

    if metric_col:
        items.append(
            DashboardItem(
                kind="kpi",
                title=f"Total {metric_col}",
                sql=f"SELECT SUM(\"{metric_col}\") AS total_value FROM data",
            )
        )
        items.append(
            DashboardItem(
                kind="kpi",
                title="Transaction Count",
                sql="SELECT COUNT(*) AS total_count FROM data",
            )
        )

    if date_col and metric_col:
        items.append(
            DashboardItem(
                kind="line",
                title=f"{metric_col} Over Time",
                sql=(
                    f"SELECT \"{date_col}\" AS dt, "
                    f"SUM(\"{metric_col}\") AS value "
                    f"FROM data GROUP BY 1 ORDER BY 1"
                ),
            )
        )

    if category_col and metric_col:
        items.append(
            DashboardItem(
                kind="bar",
                title=f"Top 5 {category_col} by {metric_col}",
                sql=(
                    f"SELECT \"{category_col}\" AS category, "
                    f"SUM(\"{metric_col}\") AS total "
                    f"FROM data GROUP BY 1 ORDER BY total DESC LIMIT 5"
                ),
            )
        )

    if metric_col:
        items.append(
            DashboardItem(
                kind="hist",
                title=f"Distribution of {metric_col}",
                sql=f"SELECT \"{metric_col}\" AS value FROM data",
            )
        )

    return items[:5]


def _to_dashboard_item(raw: dict) -> tuple[DashboardItem | None, str | None]:
    kind = str(raw.get("kind", "")).strip().lower()
    title = str(raw.get("title", "")).strip()
    sql = str(raw.get("sql", "")).strip()

    if kind not in ALLOWED_KINDS or not title or not sql:
        return None, "missing_or_invalid_fields"
    if not is_safe_query(sql):
        return None, "unsafe_sql"
    return DashboardItem(kind=kind, title=title, sql=sql), None


def _merge_unique(primary: List[DashboardItem], fallback: List[DashboardItem], max_items: int = 5) -> List[DashboardItem]:
    merged: List[DashboardItem] = []
    seen_sql = set()

    for item in primary + fallback:
        key = item.sql.strip().lower()
        if key in seen_sql:
            continue
        merged.append(item)
        seen_sql.add(key)
        if len(merged) >= max_items:
            break

    return merged


def build_dashboard_plan(
    schema: SchemaInfo,
    use_ai: bool = True,
    return_debug: bool = False,
    semantic_hints: dict | None = None,
):
    fallback_items = _rule_based_dashboard_plan(schema, semantic_hints=semantic_hints)
    debug_info = {
        "planner_mode": "rule_based_only" if not use_ai else "ai_assisted_with_fallback",
        "ai_enabled": use_ai,
        "semantic_hints": semantic_hints,
        "ai_attempted": False,
        "ai_success": False,
        "ai_error": None,
        "ai_suggested_count": 0,
        "ai_accepted_count": 0,
        "ai_rejected_items": [],
        "fallback_item_count": len(fallback_items),
        "final_item_count": 0,
        "final_items": [],
    }

    if not use_ai:
        debug_info["final_item_count"] = len(fallback_items)
        debug_info["final_items"] = [item.__dict__ for item in fallback_items]
        if return_debug:
            return fallback_items, debug_info
        return fallback_items

    try:
        debug_info["ai_attempted"] = True
        suggested = suggest_dashboard_plan(
            columns=schema.columns,
            date_column=schema.date_column,
            numeric_column=schema.numeric_column,
            category_column=schema.category_column,
            semantic_hints=semantic_hints,
            max_items=5,
        )
        debug_info["ai_suggested_count"] = len(suggested)

        ai_items = []
        for raw in suggested:
            if isinstance(raw, dict):
                item, reason = _to_dashboard_item(raw)
                if item:
                    ai_items.append(item)
                else:
                    debug_info["ai_rejected_items"].append({"item": raw, "reason": reason})
            else:
                debug_info["ai_rejected_items"].append({"item": raw, "reason": "not_a_dict"})

        debug_info["ai_accepted_count"] = len(ai_items)
        debug_info["ai_success"] = len(ai_items) > 0

        if not ai_items:
            debug_info["final_item_count"] = len(fallback_items)
            debug_info["final_items"] = [item.__dict__ for item in fallback_items]
            if return_debug:
                return fallback_items, debug_info
            return fallback_items

        final_items = _merge_unique(ai_items, fallback_items, max_items=5)
        debug_info["final_item_count"] = len(final_items)
        debug_info["final_items"] = [item.__dict__ for item in final_items]
        if return_debug:
            return final_items, debug_info
        return final_items
    except Exception as exc:
        debug_info["ai_error"] = str(exc)
        debug_info["final_item_count"] = len(fallback_items)
        debug_info["final_items"] = [item.__dict__ for item in fallback_items]
        if return_debug:
            return fallback_items, debug_info
        return fallback_items
