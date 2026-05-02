from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import settings
from app.dashboard_engine import build_dashboard_plan
from app.data_engine import create_connection, detect_schema, execute_select_query, load_csv_to_duckdb
from app.debug_logger import DebugRunLogger
from app.insight_engine import generate_deterministic_insight_text, generate_insight_signals_v2
from app.llm_engine import generate_insight_narrative, generate_sql
from app.semantic_engine import infer_semantic_hints
from app.sql_safety import is_safe_query


def _priority_badge(priority: str) -> str:
    if priority == "high":
        return "🔴 High"
    if priority == "medium":
        return "🟡 Medium"
    return "🟢 Low"


st.set_page_config(page_title="AI BI Agent MVP", layout="wide")
st.title("AI Business Intelligence Agent (MVP)")
st.caption("CSV upload -> auto dashboard -> NL Q&A -> safe SQL -> insight")

if "con" not in st.session_state:
    st.session_state.con = create_connection()
if "schema" not in st.session_state:
    st.session_state.schema = None
if "df" not in st.session_state:
    st.session_state.df = None
if "semantic_hints" not in st.session_state:
    st.session_state.semantic_hints = None
if "debug_logger" not in st.session_state:
    st.session_state.debug_logger = DebugRunLogger(base_dir="debug")
    st.session_state.debug_logger.log_event(
        "run_started",
        {
            "llm_provider": settings.llm_provider,
            "dashboard_ai_enabled": settings.dashboard_ai_enabled,
        },
    )
if "last_dashboard_signature" not in st.session_state:
    st.session_state.last_dashboard_signature = None

st.sidebar.header("LLM Configuration")
st.sidebar.write(f"Provider from .env: **{settings.llm_provider}**")
st.sidebar.write("Supported: openai, gemini, groq")
st.sidebar.write(
    f"Dashboard planner: **{'AI-assisted + fallback' if settings.dashboard_ai_enabled else 'Rule-based only'}**"
)
st.sidebar.write(f"Debug log file: **{st.session_state.debug_logger.file_path}**")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv_to_duckdb(st.session_state.con, uploaded_file)
        schema = detect_schema(df)
        semantic_hints = infer_semantic_hints(df, schema)
        st.session_state.df = df
        st.session_state.schema = schema
        st.session_state.semantic_hints = semantic_hints.to_dict()
        st.session_state.debug_logger.log_event(
            "csv_loaded",
            {
                "file_name": uploaded_file.name,
                "rows": len(df),
                "columns": list(df.columns),
                "schema": {
                    "date_column": schema.date_column,
                    "numeric_column": schema.numeric_column,
                    "category_column": schema.category_column,
                },
                "semantic_hints": semantic_hints.to_dict(),
            },
        )
        st.success("CSV loaded successfully into DuckDB table 'data'.")
    except Exception as exc:
        st.session_state.debug_logger.log_event("csv_load_error", {"error": str(exc)})
        st.error(f"File load failed: {exc}")

if st.session_state.schema and st.session_state.df is not None:
    schema = st.session_state.schema
    df = st.session_state.df
    semantic_hints = st.session_state.semantic_hints

    st.subheader("Detected Schema")
    st.write(
        {
            "date_column": schema.date_column,
            "numeric_column": schema.numeric_column,
            "category_column": schema.category_column,
            "columns": schema.columns,
        }
    )
    st.subheader("Semantic Hints")
    st.write(semantic_hints)

    st.subheader("Auto Dashboard")
    dashboard_plan, dashboard_debug = build_dashboard_plan(
        schema,
        use_ai=settings.dashboard_ai_enabled,
        return_debug=True,
        semantic_hints=semantic_hints,
    )

    dashboard_signature = "|".join([f"{item.kind}:{item.sql}" for item in dashboard_plan])
    if st.session_state.last_dashboard_signature != dashboard_signature:
        st.session_state.debug_logger.log_event("dashboard_planning", dashboard_debug)
        st.session_state.last_dashboard_signature = dashboard_signature

    kpi_cols = st.columns(2)
    kpi_cursor = 0

    for item in dashboard_plan:
        result = execute_select_query(st.session_state.con, item.sql)
        st.session_state.debug_logger.log_event(
            "dashboard_query_executed",
            {
                "kind": item.kind,
                "title": item.title,
                "sql": item.sql,
                "row_count": len(result),
                "columns": list(result.columns),
            },
        )

        if item.kind == "kpi":
            value = result.iloc[0, 0] if not result.empty else None
            kpi_cols[kpi_cursor % 2].metric(item.title, value=f"{value}")
            kpi_cursor += 1
        elif item.kind == "line" and not result.empty:
            result["dt"] = pd.to_datetime(result["dt"], errors="coerce")
            fig = px.line(result, x="dt", y="value", title=item.title)
            st.plotly_chart(fig, use_container_width=True)
        elif item.kind == "bar" and not result.empty:
            fig = px.bar(result, x="category", y="total", title=item.title)
            st.plotly_chart(fig, use_container_width=True)
        elif item.kind == "hist" and not result.empty:
            fig = px.histogram(result, x="value", nbins=30, title=item.title)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ask a Question")
    question = st.text_input("Example: Top 5 products by revenue")

    if st.button("Run Question") and question.strip():
        try:
            sql = generate_sql(question, schema.columns, semantic_hints=semantic_hints)
            st.code(sql, language="sql")
            st.session_state.debug_logger.log_event(
                "qa_sql_generated",
                {"question": question, "sql": sql, "semantic_hints": semantic_hints},
            )

            if not is_safe_query(sql):
                st.session_state.debug_logger.log_event(
                    "qa_sql_blocked",
                    {"question": question, "sql": sql, "reason": "unsafe_query"},
                )
                st.error("Generated SQL blocked by safety layer.")
            else:
                answer_df = execute_select_query(st.session_state.con, sql)
                st.session_state.debug_logger.log_event(
                    "qa_sql_executed",
                    {
                        "question": question,
                        "sql": sql,
                        "row_count": len(answer_df),
                        "columns": list(answer_df.columns),
                    },
                )
                st.dataframe(answer_df, use_container_width=True)
        except Exception as exc:
            st.session_state.debug_logger.log_event(
                "qa_error",
                {"question": question, "error": str(exc)},
            )
            st.error(f"Q&A failed: {exc}")

    st.subheader("Business Insight")
    insight_signals = generate_insight_signals_v2(df, schema, semantic_hints=semantic_hints)
    deterministic_text = generate_deterministic_insight_text(insight_signals)

    narrative_text = deterministic_text
    narrative_source = "deterministic"
    narrative_error = None

    try:
        narrative_text = generate_insight_narrative(insight_signals, semantic_hints=semantic_hints)
        narrative_source = "llm"
    except Exception as exc:
        narrative_error = str(exc)

    st.session_state.debug_logger.log_event(
        "insight_v2_signals",
        {
            "signals": insight_signals,
            "deterministic_text": deterministic_text,
            "narrative_source": narrative_source,
            "narrative_error": narrative_error,
        },
    )

    st.info(narrative_text)

    priority_items = insight_signals.get("priority_insights", [])
    if priority_items:
        st.markdown("### Prioritized Insights")
        for item in priority_items[:5]:
            st.markdown(
                f"- **{_priority_badge(item.get('priority', 'low'))}** {item.get('message', '')}  "
                f"Action: {item.get('recommendation', '')}"
            )

    action_items = insight_signals.get("action_items", [])
    if action_items:
        st.markdown("### Recommended Actions")
        for action in action_items[:5]:
            st.markdown(
                f"- **{_priority_badge(action.get('priority', 'low'))}** {action.get('recommendation', '')}"
            )

    with st.expander("Insight Signals (v2)"):
        st.json(insight_signals)

    st.session_state.debug_logger.log_event(
        "insight_generated",
        {
            "insight": narrative_text,
            "source": narrative_source,
        },
    )
else:
    st.info("Upload a CSV to start.")
