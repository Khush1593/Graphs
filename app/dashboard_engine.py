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
    source: str = "rule"
    priority_score: int = 0
    goal_id: str | None = None


ALLOWED_KINDS = {"kpi", "line", "bar", "hist"}
PRIORITY_TO_SCORE = {"high": 90, "medium": 60, "low": 30}


def _resolve_columns(
    schema: SchemaInfo,
    semantic_hints: dict | None,
    context=None,
) -> tuple[str | None, str | None, str | None, list[str]]:
    metric_col = schema.numeric_column
    date_col = schema.date_column
    category_col = schema.category_column
    focus_dimensions: list[str] = []

    if semantic_hints:
        hinted_metric = semantic_hints.get("primary_metric")
        hinted_time = semantic_hints.get("time_column")
        hinted_dims = semantic_hints.get("dimensions") or []

        if isinstance(hinted_metric, str) and hinted_metric in schema.columns:
            metric_col = hinted_metric
        if isinstance(hinted_time, str) and hinted_time in schema.columns:
            date_col = hinted_time
        if hinted_dims and isinstance(hinted_dims, list):
            first_dimension = next(
                (d for d in hinted_dims if isinstance(d, str) and d in schema.columns),
                None,
            )
            if first_dimension:
                category_col = first_dimension

    if context is not None:
        ctx_metric = context.focus_metric
        if ctx_metric and ctx_metric in schema.columns:
            metric_col = ctx_metric
        ctx_dims = [d for d in context.focus_dimensions if d in schema.columns]
        if ctx_dims:
            focus_dimensions = ctx_dims
            category_col = ctx_dims[0]
        ctx_time = context.time_column
        if ctx_time and ctx_time in schema.columns:
            date_col = ctx_time

    return metric_col, date_col, category_col, focus_dimensions


def _rule_based_dashboard_plan(
    schema: SchemaInfo,
    semantic_hints: dict | None = None,
    context=None,
) -> List[DashboardItem]:
    items: List[DashboardItem] = []
    metric_col, date_col, category_col, focus_dimensions = _resolve_columns(
        schema, semantic_hints, context
    )

    if metric_col:
        items.append(
            DashboardItem(
                kind="kpi",
                title=f"Total {metric_col}",
                sql=f"SELECT SUM(\"{metric_col}\") AS total_value FROM data",
                source="rule",
                priority_score=80,
            )
        )
        items.append(
            DashboardItem(
                kind="kpi",
                title="Transaction Count",
                sql="SELECT COUNT(*) AS total_count FROM data",
                source="rule",
                priority_score=40,
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
                source="rule",
                priority_score=70,
            )
        )

    dims_for_bar = focus_dimensions or ([category_col] if category_col else [])
    for dim in dims_for_bar[:2]:
        if dim and metric_col:
            items.append(
                DashboardItem(
                    kind="bar",
                    title=f"Top 5 {dim} by {metric_col}",
                    sql=(
                        f"SELECT \"{dim}\" AS category, "
                        f"SUM(\"{metric_col}\") AS total "
                        f"FROM data GROUP BY 1 ORDER BY total DESC LIMIT 5"
                    ),
                    source="rule",
                    priority_score=60,
                )
            )

    if metric_col:
        items.append(
            DashboardItem(
                kind="hist",
                title=f"Distribution of {metric_col}",
                sql=f"SELECT \"{metric_col}\" AS value FROM data",
                source="rule",
                priority_score=30,
            )
        )

    return items


def _goal_driven_items(schema: SchemaInfo, context) -> List[DashboardItem]:
    items: List[DashboardItem] = []
    if context is None:
        return items

    time_col = context.time_column if context.time_column in schema.columns else None

    for g in context.active_goals:
        metric = g.get("metric")
        dim = g.get("dimension")
        priority = g.get("priority", "medium")
        gid = g.get("id")
        score = PRIORITY_TO_SCORE.get(priority, 60)

        if not metric or metric not in schema.columns:
            continue

        if dim and dim in schema.columns:
            items.append(
                DashboardItem(
                    kind="bar",
                    title=f"[Goal] {g.get('title', metric)} — by {dim}",
                    sql=(
                        f"SELECT \"{dim}\" AS category, "
                        f"SUM(\"{metric}\") AS total "
                        f"FROM data GROUP BY 1 ORDER BY total DESC LIMIT 10"
                    ),
                    source="goal",
                    priority_score=score,
                    goal_id=gid,
                )
            )
        elif time_col:
            items.append(
                DashboardItem(
                    kind="line",
                    title=f"[Goal] {g.get('title', metric)} — over time",
                    sql=(
                        f"SELECT \"{time_col}\" AS dt, "
                        f"SUM(\"{metric}\") AS value "
                        f"FROM data GROUP BY 1 ORDER BY 1"
                    ),
                    source="goal",
                    priority_score=score,
                    goal_id=gid,
                )
            )

    return items


def _to_dashboard_item(raw: dict) -> tuple[DashboardItem | None, str | None]:
    kind = str(raw.get("kind", "")).strip().lower()
    title = str(raw.get("title", "")).strip()
    sql = str(raw.get("sql", "")).strip()

    if kind not in ALLOWED_KINDS or not title or not sql:
        return None, "missing_or_invalid_fields"
    if not is_safe_query(sql):
        return None, "unsafe_sql"
    return (
        DashboardItem(kind=kind, title=title, sql=sql, source="ai", priority_score=50),
        None,
    )


def _merge_unique(
    primary: List[DashboardItem],
    fallback: List[DashboardItem],
    max_items: int = 8,
) -> List[DashboardItem]:
    merged: List[DashboardItem] = []
    seen_sql = set()

    combined = sorted(primary + fallback, key=lambda i: i.priority_score, reverse=True)

    for item in combined:
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
    context=None,
    debug_logger=None,
):
    fallback_items = _rule_based_dashboard_plan(schema, semantic_hints=semantic_hints, context=context)
    goal_items = _goal_driven_items(schema, context)

    debug_info = {
        "planner_mode": "rule_based_only" if not use_ai else "ai_assisted_with_fallback",
        "ai_enabled": use_ai,
        "semantic_hints": semantic_hints,
        "context_present": context is not None,
        "context_focus_metric": context.focus_metric if context else None,
        "context_focus_dimensions": context.focus_dimensions if context else [],
        "context_active_goal_count": len(context.active_goals) if context else 0,
        "ai_attempted": False,
        "ai_success": False,
        "ai_error": None,
        "ai_suggested_count": 0,
        "ai_accepted_count": 0,
        "ai_rejected_items": [],
        "fallback_item_count": len(fallback_items),
        "goal_item_count": len(goal_items),
        "final_item_count": 0,
        "final_items": [],
    }

    if debug_logger:
        debug_logger.log_event(
            "dashboard_planner_context_received",
            {
                "context_focus_metric": context.focus_metric if context else None,
                "context_focus_dimensions": context.focus_dimensions if context else [],
                "context_time_column": context.time_column if context else None,
                "context_active_goals": [
                    {
                        "id": g.get("id"),
                        "metric": g.get("metric"),
                        "dimension": g.get("dimension"),
                        "priority": g.get("priority"),
                    }
                    for g in (context.active_goals if context else [])
                ],
                "rule_based_items": [
                    {"kind": i.kind, "title": i.title, "source": i.source, "priority": i.priority_score}
                    for i in fallback_items
                ],
                "goal_driven_items": [
                    {"kind": i.kind, "title": i.title, "goal_id": i.goal_id, "priority": i.priority_score}
                    for i in goal_items
                ],
            },
        )

    if not use_ai:
        merged = _merge_unique(goal_items, fallback_items, max_items=8)
        debug_info["final_item_count"] = len(merged)
        debug_info["final_items"] = [item.__dict__ for item in merged]
        if return_debug:
            return merged, debug_info
        return merged

    try:
        debug_info["ai_attempted"] = True
        suggested = suggest_dashboard_plan(
            columns=schema.columns,
            date_column=schema.date_column,
            numeric_column=schema.numeric_column,
            category_column=schema.category_column,
            semantic_hints=semantic_hints,
            context=context,
            max_items=5,
        )
        debug_info["ai_suggested_count"] = len(suggested)

        ai_items: List[DashboardItem] = []
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
            merged = _merge_unique(goal_items, fallback_items, max_items=8)
            debug_info["final_item_count"] = len(merged)
            debug_info["final_items"] = [item.__dict__ for item in merged]
            if return_debug:
                return merged, debug_info
            return merged

        final_items = _merge_unique(goal_items + ai_items, fallback_items, max_items=8)
        debug_info["final_item_count"] = len(final_items)
        debug_info["final_items"] = [item.__dict__ for item in final_items]
        if return_debug:
            return final_items, debug_info
        return final_items
    except Exception as exc:
        debug_info["ai_error"] = str(exc)
        merged = _merge_unique(goal_items, fallback_items, max_items=8)
        debug_info["final_item_count"] = len(merged)
        debug_info["final_items"] = [item.__dict__ for item in merged]
        if return_debug:
            return merged, debug_info
        return merged
