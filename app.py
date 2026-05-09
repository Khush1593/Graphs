from __future__ import annotations

import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.clarification_engine import (
    generate_clarification_questions,
    normalize_user_answers,
)
from app.config import settings
from app.confirmation_engine import generate_confirmation_summary
from app.context_engine import build_context
from app.dashboard_engine import build_dashboard_plan
from app.data_engine import create_connection, detect_schema, execute_select_query, load_csv_to_duckdb
from app.debug_logger import DebugRunLogger
from app.goal_engine import (
    generate_goals_for_user,
    normalize_user_goals,
)
from app.insight_engine import generate_deterministic_insight_text, generate_insight_signals_v2
from app.llm_engine import generate_insight_narrative, generate_sql
from app.persistence import ProjectStore, schema_from_dict, schema_to_dict
from app.semantic_engine import infer_semantic_hints
from app.sql_safety import is_safe_query
from app.understanding_engine import generate_understanding


STEP_LABELS = [
    "1. Upload Data",
    "2. AI Understanding",
    "3. Clarification Questions",
    "4. Business Goals",
    "5. AI Confirmation",
    "6. Dashboard, Insights & Q&A",
]


class _FileLikeBytes:
    """Mimics the parts of Streamlit's UploadedFile that load_csv_to_duckdb uses."""

    def __init__(self, data: bytes, name: str) -> None:
        self._data = data
        self.name = name
        self.size = len(data)

    def getbuffer(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def _priority_badge(priority: str) -> str:
    if priority == "high":
        return "🔴 High"
    if priority == "medium":
        return "🟡 Medium"
    return "🟢 Low"


def _save_project_state() -> None:
    pid = st.session_state.get("active_project_id")
    if not pid:
        return
    store: ProjectStore = st.session_state.project_store

    if st.session_state.upload_signature is not None:
        store.save_artifact(pid, "upload_signature", {"signature": st.session_state.upload_signature})
    if st.session_state.schema is not None:
        store.save_artifact(pid, "schema", schema_to_dict(st.session_state.schema))
    if st.session_state.semantic_hints is not None:
        store.save_artifact(pid, "semantic_hints", st.session_state.semantic_hints)
    if st.session_state.understanding is not None:
        store.save_artifact(pid, "understanding", st.session_state.understanding)
    if st.session_state.clarifications is not None:
        store.save_artifact(pid, "clarifications", st.session_state.clarifications)
    if st.session_state.user_clarifications is not None:
        store.save_artifact(pid, "user_clarifications", st.session_state.user_clarifications)
    if st.session_state.goals is not None:
        store.save_artifact(pid, "goals", st.session_state.goals)
    if st.session_state.user_goals is not None:
        store.save_artifact(pid, "user_goals", st.session_state.user_goals)
    if st.session_state.confirmation is not None:
        store.save_artifact(pid, "confirmation", st.session_state.confirmation)
    store.save_artifact(
        pid,
        "confirmation_accepted",
        {"accepted": bool(st.session_state.confirmation_accepted)},
    )

    store.update_meta(
        pid,
        current_step=st.session_state.current_step,
        debug_log_path=st.session_state.debug_logger.file_path,
    )


def _load_project_state(project_id: str) -> bool:
    store: ProjectStore = st.session_state.project_store
    meta = store.get_meta(project_id)
    if meta is None:
        return False

    raw = store.load_raw_data(project_id)
    if raw is None:
        # Project has no CSV yet — just register it as active and reset the pipeline.
        st.session_state.active_project_id = project_id
        _reset_pipeline_state()
        return True

    file_bytes, filename = raw
    fake_upload = _FileLikeBytes(file_bytes, filename)
    df = load_csv_to_duckdb(st.session_state.con, fake_upload)

    # Prefer saved schema/hints if present (they may include user adjustments later);
    # otherwise recompute deterministically.
    schema_artifact = store.load_artifact(project_id, "schema")
    schema = schema_from_dict(schema_artifact) if schema_artifact else detect_schema(df)
    hints_artifact = store.load_artifact(project_id, "semantic_hints")
    semantic_hints = hints_artifact if hints_artifact is not None else infer_semantic_hints(df, schema).to_dict()

    upload_sig_artifact = store.load_artifact(project_id, "upload_signature") or {}
    upload_signature = upload_sig_artifact.get("signature") or f"{filename}|{len(file_bytes)}"

    st.session_state.df = df
    st.session_state.schema = schema
    st.session_state.semantic_hints = semantic_hints
    st.session_state.upload_signature = upload_signature

    st.session_state.understanding = store.load_artifact(project_id, "understanding")
    st.session_state.understanding_signature = (
        f"{upload_signature}|{','.join(df.columns)}" if st.session_state.understanding else None
    )
    st.session_state.clarifications = store.load_artifact(project_id, "clarifications")
    st.session_state.clarifications_signature = (
        st.session_state.understanding_signature if st.session_state.clarifications else None
    )
    st.session_state.user_clarifications = store.load_artifact(project_id, "user_clarifications")

    st.session_state.goals = store.load_artifact(project_id, "goals")
    if st.session_state.goals and st.session_state.user_clarifications:
        st.session_state.goals_signature = (
            f"{st.session_state.understanding_signature}|"
            f"{json.dumps(st.session_state.user_clarifications, sort_keys=True, ensure_ascii=True)}"
        )
    else:
        st.session_state.goals_signature = None
    st.session_state.user_goals = store.load_artifact(project_id, "user_goals")

    st.session_state.confirmation = store.load_artifact(project_id, "confirmation")
    if st.session_state.confirmation and st.session_state.user_goals is not None:
        st.session_state.confirmation_signature = (
            f"{st.session_state.goals_signature}|"
            f"{json.dumps(st.session_state.user_goals, sort_keys=True, ensure_ascii=True, default=str)}"
        )
    else:
        st.session_state.confirmation_signature = None

    accepted_artifact = store.load_artifact(project_id, "confirmation_accepted") or {}
    st.session_state.confirmation_accepted = bool(accepted_artifact.get("accepted"))

    st.session_state.last_dashboard_signature = None
    st.session_state.current_step = max(1, int(meta.current_step or 1))
    st.session_state.active_project_id = project_id

    st.session_state.debug_logger.log_event(
        "project_loaded",
        {
            "project_id": project_id,
            "name": meta.name,
            "current_step": st.session_state.current_step,
            "filename": filename,
            "rows": int(len(df)),
            "artifact_summary": {
                name: store.has_artifact(project_id, name)
                for name in (
                    "understanding",
                    "clarifications",
                    "user_clarifications",
                    "goals",
                    "user_goals",
                    "confirmation",
                )
            },
        },
    )
    return True


def _advance_to(step: int) -> None:
    if step > st.session_state.current_step:
        st.session_state.debug_logger.log_event(
            "step_advanced",
            {"from": st.session_state.current_step, "to": step},
        )
        st.session_state.current_step = step
    _save_project_state()
    st.rerun()


def _reset_pipeline_state() -> None:
    st.session_state.understanding = None
    st.session_state.understanding_signature = None
    st.session_state.clarifications = None
    st.session_state.clarifications_signature = None
    st.session_state.user_clarifications = None
    st.session_state.goals = None
    st.session_state.goals_signature = None
    st.session_state.user_goals = None
    st.session_state.confirmation = None
    st.session_state.confirmation_signature = None
    st.session_state.confirmation_accepted = False
    st.session_state.last_dashboard_signature = None
    st.session_state.current_step = 1


st.set_page_config(page_title="AI BI Agent MVP", layout="wide")
st.title("AI Business Intelligence Agent (MVP)")
st.caption("Step-wise execution: each stage runs only when you click Next.")

if "project_store" not in st.session_state:
    st.session_state.project_store = ProjectStore()
if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None
if "con" not in st.session_state:
    st.session_state.con = create_connection()
if "schema" not in st.session_state:
    st.session_state.schema = None
if "df" not in st.session_state:
    st.session_state.df = None
if "semantic_hints" not in st.session_state:
    st.session_state.semantic_hints = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "understanding" not in st.session_state:
    st.session_state.understanding = None
if "understanding_signature" not in st.session_state:
    st.session_state.understanding_signature = None
if "clarifications" not in st.session_state:
    st.session_state.clarifications = None
if "clarifications_signature" not in st.session_state:
    st.session_state.clarifications_signature = None
if "user_clarifications" not in st.session_state:
    st.session_state.user_clarifications = None
if "goals" not in st.session_state:
    st.session_state.goals = None
if "goals_signature" not in st.session_state:
    st.session_state.goals_signature = None
if "user_goals" not in st.session_state:
    st.session_state.user_goals = None
if "confirmation" not in st.session_state:
    st.session_state.confirmation = None
if "confirmation_signature" not in st.session_state:
    st.session_state.confirmation_signature = None
if "confirmation_accepted" not in st.session_state:
    st.session_state.confirmation_accepted = False
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

current_step = st.session_state.current_step

st.sidebar.header("Project")
projects = st.session_state.project_store.list_projects()
project_labels = ["(no project — create or pick one)"] + [
    f"{p.name}  ·  step {p.current_step}  ·  {p.last_modified}" for p in projects
]
project_label_to_id = {label: pid for label, pid in zip(project_labels[1:], [p.project_id for p in projects])}

active_pid = st.session_state.active_project_id
default_idx = 0
if active_pid:
    for i, p in enumerate(projects):
        if p.project_id == active_pid:
            default_idx = i + 1
            break

selected_label = st.sidebar.selectbox(
    "Active project",
    project_labels,
    index=default_idx,
    key="project_select",
)

if selected_label != project_labels[0]:
    selected_pid = project_label_to_id[selected_label]
    if selected_pid != active_pid:
        if _load_project_state(selected_pid):
            st.rerun()

active_meta = (
    st.session_state.project_store.get_meta(active_pid) if active_pid else None
)

if active_meta:
    rename_value = st.sidebar.text_input(
        "Rename project", value=active_meta.name, key=f"rename_{active_meta.project_id}"
    )
    cols_a, cols_b = st.sidebar.columns(2)
    if cols_a.button("Save name", key=f"rename_btn_{active_meta.project_id}"):
        if rename_value.strip() and rename_value.strip() != active_meta.name:
            st.session_state.project_store.rename_project(active_meta.project_id, rename_value)
            st.rerun()
    if cols_b.button("Delete", key=f"delete_btn_{active_meta.project_id}"):
        st.session_state.project_store.delete_project(active_meta.project_id)
        st.session_state.active_project_id = None
        _reset_pipeline_state()
        st.rerun()

with st.sidebar.expander("Create new project"):
    new_name = st.text_input("Name (optional)", key="new_project_name")
    if st.button("Create", key="create_project_btn"):
        meta = st.session_state.project_store.create_project(name=new_name or None)
        st.session_state.active_project_id = meta.project_id
        _reset_pipeline_state()
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.header("Pipeline Progress")
for idx, label in enumerate(STEP_LABELS, start=1):
    if idx < current_step:
        st.sidebar.markdown(f"✅ {label}")
    elif idx == current_step:
        st.sidebar.markdown(f"🟡 **{label}** (current)")
    else:
        st.sidebar.markdown(f"⚪ {label}")

st.sidebar.markdown("---")
st.sidebar.header("LLM Configuration")
st.sidebar.write(f"Provider from .env: **{settings.llm_provider}**")
st.sidebar.write("Supported: openai, gemini, groq")
st.sidebar.write(
    f"Dashboard planner: **{'AI-assisted + fallback' if settings.dashboard_ai_enabled else 'Rule-based only'}**"
)
st.sidebar.write(f"Debug log file: **{st.session_state.debug_logger.file_path}**")


# =====================================================================
# STEP 1: Upload Data
# =====================================================================
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        upload_signature = f"{uploaded_file.name}|{uploaded_file.size}"
        if st.session_state.upload_signature != upload_signature:
            uploaded_bytes = uploaded_file.getbuffer().tobytes() if hasattr(uploaded_file.getbuffer(), "tobytes") else bytes(uploaded_file.getbuffer())
            df = load_csv_to_duckdb(st.session_state.con, uploaded_file)
            schema = detect_schema(df)
            semantic_hints = infer_semantic_hints(df, schema)
            st.session_state.df = df
            st.session_state.schema = schema
            st.session_state.semantic_hints = semantic_hints.to_dict()
            st.session_state.upload_signature = upload_signature
            _reset_pipeline_state()

            if not st.session_state.active_project_id:
                base_name = uploaded_file.name.rsplit(".", 1)[0]
                meta = st.session_state.project_store.create_project(name=base_name)
                st.session_state.active_project_id = meta.project_id

            pid = st.session_state.active_project_id
            st.session_state.project_store.save_raw_data(
                pid, uploaded_bytes, uploaded_file.name
            )
            st.session_state.project_store.update_meta(
                pid,
                upload_filename=uploaded_file.name,
                upload_size=uploaded_file.size,
                debug_log_path=st.session_state.debug_logger.file_path,
            )

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
                    "project_id": pid,
                },
            )
            _save_project_state()
            st.success(
                f"Loaded {uploaded_file.name} ({len(df)} rows). Pipeline reset to Step 1. "
                f"Auto-saved into project."
            )
    except Exception as exc:
        st.session_state.debug_logger.log_event("csv_load_error", {"error": str(exc)})
        st.error(f"File load failed: {exc}")

if st.session_state.df is None or st.session_state.schema is None:
    st.info("Upload a CSV to start.")
    st.stop()

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

with st.expander("Data preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

if current_step < 2:
    if st.button("Next: Generate AI Understanding ➜", type="primary"):
        _advance_to(2)
    st.stop()


# =====================================================================
# STEP 2: AI Understanding (Stage 3)
# =====================================================================
st.header("Step 2: AI Understanding (Stage 3)")
st.caption(
    "**For internal AI use, not user-facing output.** This is the structural read of your data "
    "that downstream stages (clarifications, goals, dashboard, insights) consume. KPIs, business "
    "questions, and reports are generated by later stages — not here."
)

understanding_signature = (
    f"{st.session_state.upload_signature}|{','.join(df.columns)}"
)
if st.session_state.understanding_signature != understanding_signature:
    st.session_state.debug_logger.log_event(
        "understanding_started",
        {"signature": understanding_signature},
    )
    with st.spinner("Calling LLM for dataset understanding..."):
        understanding = generate_understanding(
            df=df,
            schema=schema,
            semantic_hints=semantic_hints,
            debug_logger=st.session_state.debug_logger,
        )
    st.session_state.understanding = understanding
    st.session_state.understanding_signature = understanding_signature
    _save_project_state()

understanding = st.session_state.understanding
summary = understanding.get("summary", {})

st.caption(
    f"Source: {understanding.get('source')} | "
    f"Provider: {understanding.get('provider')} | "
    f"Model: {understanding.get('model')} | "
    f"Confidence: {summary.get('confidence', 'unknown')}"
)
if understanding.get("error"):
    st.warning(
        f"LLM call failed, using deterministic fallback. Reason: {understanding['error']}\n\n"
        "Click below to retry the LLM call. Common causes: model busy, network blip, "
        "or rate limit. The system already retried once with backoff before falling back."
    )
    if st.button("🔁 Regenerate AI Understanding from LLM", key="regen_understanding"):
        st.session_state.understanding = None
        st.session_state.understanding_signature = None
        st.rerun()
st.markdown(f"**Description:** {summary.get('dataset_description', 'N/A')}")
st.markdown(f"**Business domain:** {summary.get('business_domain_guess', 'N/A')}")
st.markdown(f"**Each row represents:** {summary.get('primary_entity', 'N/A')}")
st.markdown(f"**Primary metric:** {summary.get('primary_metric', 'N/A')}")

time_scope = summary.get("time_scope") or {}
st.markdown(
    f"**Time scope:** {time_scope.get('start', 'N/A')} → {time_scope.get('end', 'N/A')} "
    f"(span {time_scope.get('span_days', 'N/A')} days, granularity: "
    f"{time_scope.get('granularity_guess', 'N/A')})"
)

identifiers = summary.get("identifier_columns", []) or []
if identifiers:
    st.markdown(f"**Identifier columns (do not aggregate):** {', '.join(identifiers)}")

measures = summary.get("measure_columns", []) or []
if measures:
    st.markdown("**Measure columns:**")
    for m in measures:
        st.markdown(
            f"- `{m.get('name')}` — role: {m.get('role')}, agg: {m.get('aggregation_hint')}"
        )

dimensions = summary.get("dimension_columns", []) or []
if dimensions:
    st.markdown("**Dimension columns:**")
    for d in dimensions:
        charts = ", ".join(d.get("suitable_chart_types", []) or [])
        st.markdown(
            f"- `{d.get('name')}` — cardinality: {d.get('cardinality_bucket')}, "
            f"chart types: {charts}"
        )

derived = summary.get("derived_metric_opportunities", []) or []
if derived:
    st.markdown("**Derived metric opportunities (passed to goal generator):**")
    for d in derived:
        source_tag = f" _(source: {d.get('source')})_" if d.get("source") else ""
        st.markdown(
            f"- **{d.get('name')}** = `{d.get('formula')}` — {d.get('why_useful', '')}{source_tag}"
        )

notes = summary.get("data_quality_notes", []) or []
if notes:
    st.markdown("**Data quality notes (surfaced in confirmation step):**")
    for note in notes:
        st.markdown(f"- {note}")

with st.expander("Understanding (full structured object)"):
    st.json(understanding)

if current_step < 3:
    if st.button("Next: Generate Clarification Questions ➜", type="primary"):
        _advance_to(3)
    st.stop()


# =====================================================================
# STEP 3: Clarification Questions (Stage 4)
# =====================================================================
st.header("Step 3: Clarification Questions (Stage 4)")

if st.session_state.clarifications_signature != understanding_signature:
    st.session_state.debug_logger.log_event(
        "clarifications_started",
        {"signature": understanding_signature},
    )
    with st.spinner("Calling LLM for clarification questions..."):
        clarifications = generate_clarification_questions(
            understanding_summary=summary,
            debug_logger=st.session_state.debug_logger,
        )
    st.session_state.clarifications = clarifications
    st.session_state.clarifications_signature = understanding_signature
    _save_project_state()

clarifications = st.session_state.clarifications
questions = clarifications.get("questions", [])

st.caption(
    f"Source: {clarifications.get('source')} | "
    f"Provider: {clarifications.get('provider')} | "
    f"Model: {clarifications.get('model')} | "
    f"Question count: {len(questions)}"
)
if clarifications.get("error"):
    st.warning(
        f"LLM call failed, using deterministic questions. Reason: {clarifications['error']}\n\n"
        "Click below to retry the LLM call. The system already retried once with backoff "
        "before falling back to the deterministic question set."
    )
    if st.button("🔁 Regenerate Clarification Questions from LLM", key="regen_clarifications"):
        st.session_state.clarifications = None
        st.session_state.clarifications_signature = None
        st.rerun()

with st.form("clarification_form"):
    raw_answers: dict = {}
    for q in questions:
        key = q["key"]
        qtype = q["type"]
        widget_key = f"clarif_{q['id']}"
        st.markdown(f"**{q['question']}**")
        if q.get("why_asked"):
            st.caption(q["why_asked"])

        if qtype == "single_select":
            opts = q.get("options") or []
            default = q.get("default")
            default_index = opts.index(default) if default in opts else 0
            raw_answers[key] = st.radio(
                label=q["question"],
                options=opts,
                index=default_index,
                key=widget_key,
                label_visibility="collapsed",
            )
        elif qtype == "multi_select":
            opts = q.get("options") or []
            default = q.get("default") or []
            raw_answers[key] = st.multiselect(
                label=q["question"],
                options=opts,
                default=[d for d in default if d in opts],
                key=widget_key,
                label_visibility="collapsed",
            )
        else:
            default_val = q.get("default") or ""
            raw_answers[key] = st.text_input(
                label=q["question"],
                value=default_val,
                key=widget_key,
                label_visibility="collapsed",
            )

    confirmed = st.form_submit_button("Confirm answers and continue ➜", type="primary")

if confirmed:
    normalized = normalize_user_answers(questions, raw_answers)
    st.session_state.user_clarifications = normalized
    st.session_state.debug_logger.log_event(
        "clarifications_user_answered",
        {
            "answers": normalized,
            "question_keys": [q["key"] for q in questions],
        },
    )
    _save_project_state()
    st.success("Answers saved.")
    _advance_to(4)

if st.session_state.user_clarifications:
    st.markdown("**Your saved answers:**")
    st.json(st.session_state.user_clarifications)

with st.expander("Clarifications (full structured object)"):
    st.json(clarifications)

if current_step < 4 or not st.session_state.user_clarifications:
    if st.session_state.user_clarifications and current_step < 4:
        if st.button("Next: Generate Business Goals ➜", type="primary"):
            _advance_to(4)
    st.stop()


# =====================================================================
# STEP 4: Business Goals (Stage 5)
# =====================================================================
st.header("Step 4: Business Goals (Stage 5)")

clarif_payload = st.session_state.user_clarifications
goals_signature = (
    f"{understanding_signature}|"
    f"{json.dumps(clarif_payload, sort_keys=True, ensure_ascii=True)}"
)

if st.session_state.goals_signature != goals_signature:
    st.session_state.debug_logger.log_event(
        "goals_started",
        {"signature": goals_signature},
    )
    with st.spinner("Calling LLM for goal generation..."):
        goals_result = generate_goals_for_user(
            understanding_summary=summary,
            user_clarifications=clarif_payload,
            debug_logger=st.session_state.debug_logger,
        )
    st.session_state.goals = goals_result
    st.session_state.goals_signature = goals_signature
    st.session_state.user_goals = None
    _save_project_state()

goals_result = st.session_state.goals
st.caption(
    f"Source: {goals_result.get('source')} | "
    f"Provider: {goals_result.get('provider')} | "
    f"Model: {goals_result.get('model')} | "
    f"Goal count: {len(goals_result.get('goals', []))}"
)
if goals_result.get("error"):
    st.warning(
        f"LLM call failed, using deterministic fallback. Reason: {goals_result['error']}\n\n"
        "Click below to retry the LLM call. The system already retried once with backoff "
        "before falling back to deterministic goals."
    )
    if st.button("🔁 Regenerate Goals from LLM", key="regen_goals"):
        st.session_state.goals = None
        st.session_state.goals_signature = None
        st.session_state.user_goals = None
        st.rerun()

measure_names = [
    m.get("name") for m in (summary.get("measure_columns") or []) if m.get("name")
]
dimension_names = [
    d.get("name") for d in (summary.get("dimension_columns") or []) if d.get("name")
]
identifier_names = list(summary.get("identifier_columns") or [])
time_column = (summary.get("time_scope") or {}).get("time_column")
default_horizon = clarif_payload.get("time_granularity") or (
    summary.get("time_scope") or {}
).get("granularity_guess")

editable_seed = st.session_state.user_goals or goals_result.get("goals", [])

editor_columns = [
    "active",
    "priority",
    "title",
    "metric",
    "dimension",
    "direction",
    "target_pct",
    "time_horizon",
    "rationale",
    "id",
    "source",
]

editor_rows: list[dict] = []
for g in editable_seed:
    row = {col: g.get(col) for col in editor_columns}
    editor_rows.append(row)

edited = st.data_editor(
    editor_rows,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "active": st.column_config.CheckboxColumn("Active", default=True),
        "priority": st.column_config.SelectboxColumn(
            "Priority", options=["high", "medium", "low"], default="medium"
        ),
        "metric": st.column_config.SelectboxColumn(
            "Metric", options=measure_names, required=True
        ),
        "dimension": st.column_config.SelectboxColumn(
            "Dimension (optional)", options=[None, *dimension_names]
        ),
        "direction": st.column_config.SelectboxColumn(
            "Direction", options=["increase", "decrease", "stabilize"], required=True
        ),
        "target_pct": st.column_config.NumberColumn(
            "Target %", min_value=-100.0, max_value=1000.0, step=1.0
        ),
        "time_horizon": st.column_config.SelectboxColumn(
            "Horizon", options=["daily", "weekly", "monthly", "quarterly"]
        ),
        "title": st.column_config.TextColumn("Goal title", required=True),
        "rationale": st.column_config.TextColumn("Rationale"),
        "id": st.column_config.TextColumn("ID", disabled=True),
        "source": st.column_config.TextColumn("Source", disabled=True),
    },
    key=f"goals_editor_{goals_signature}",
)

save_goals = st.button("Save goals and continue ➜", type="primary")
if save_goals:
    accepted_goals, rejected_goals = normalize_user_goals(
        edited_rows=edited if isinstance(edited, list) else list(edited),
        measures=measure_names,
        dimensions=dimension_names,
        identifiers=identifier_names,
        time_column=time_column,
        default_horizon=default_horizon,
    )
    st.session_state.user_goals = accepted_goals
    st.session_state.debug_logger.log_event(
        "goals_user_saved",
        {
            "accepted_count": len(accepted_goals),
            "rejected_count": len(rejected_goals),
            "accepted": accepted_goals,
            "rejections": rejected_goals,
        },
    )
    if rejected_goals:
        st.warning(
            f"Saved {len(accepted_goals)} goals. "
            f"{len(rejected_goals)} row(s) rejected: "
            + "; ".join(f"row {r['row_index']}: {r['reason']}" for r in rejected_goals)
        )
    if accepted_goals:
        _save_project_state()
        st.success(f"Saved {len(accepted_goals)} goals.")
        _advance_to(5)

if st.session_state.user_goals is not None:
    st.markdown(f"**Saved goals ({len(st.session_state.user_goals)}):**")
    st.json(st.session_state.user_goals)

with st.expander("Goals (full structured object)"):
    st.json(goals_result)

if current_step < 5 or not st.session_state.user_goals:
    if st.session_state.user_goals and current_step < 5:
        if st.button("Next: Generate AI Confirmation ➜", type="primary"):
            _advance_to(5)
    st.stop()


# =====================================================================
# STEP 5: AI Confirmation (Stage 7)
# =====================================================================
st.header("Step 5: AI Confirmation (Stage 7)")

context = build_context(
    schema=schema,
    semantic_hints=semantic_hints,
    understanding=st.session_state.understanding,
    user_clarifications=st.session_state.user_clarifications,
    user_goals=st.session_state.user_goals,
    debug_logger=st.session_state.debug_logger,
)

context_prompt_block = context.to_prompt_block()

confirmation_signature = (
    f"{st.session_state.goals_signature}|"
    f"{json.dumps(st.session_state.user_goals or [], sort_keys=True, ensure_ascii=True, default=str)}"
)

if st.session_state.confirmation_signature != confirmation_signature:
    st.session_state.debug_logger.log_event(
        "confirmation_started",
        {"signature": confirmation_signature},
    )
    with st.spinner("Calling LLM to recap your assembled context..."):
        confirmation = generate_confirmation_summary(
            context=context,
            debug_logger=st.session_state.debug_logger,
        )
    st.session_state.confirmation = confirmation
    st.session_state.confirmation_signature = confirmation_signature
    st.session_state.confirmation_accepted = False
    _save_project_state()

confirmation = st.session_state.confirmation
conf_summary = confirmation.get("summary", {})

st.caption(
    f"Source: {confirmation.get('source')} | "
    f"Provider: {confirmation.get('provider')} | "
    f"Model: {confirmation.get('model')} | "
    f"Confidence: {conf_summary.get('confidence', 'unknown')}"
)
if confirmation.get("error"):
    st.warning(
        f"LLM call failed, using deterministic recap. Reason: {confirmation['error']}\n\n"
        "Click below to retry the LLM call. The system already retried once with backoff "
        "before falling back to the deterministic recap."
    )
    if st.button("🔁 Regenerate Confirmation from LLM", key="regen_confirmation"):
        st.session_state.confirmation = None
        st.session_state.confirmation_signature = None
        st.session_state.confirmation_accepted = False
        st.rerun()

st.markdown(f"### {conf_summary.get('headline', '')}")
st.markdown(f"**Data summary:** {conf_summary.get('data_summary', '')}")
st.markdown(f"**Your stated intent:** {conf_summary.get('user_intent_summary', '')}")

goals_recap = conf_summary.get("goals_recap") or []
if goals_recap:
    st.markdown("**Goals we'll track:**")
    for g in goals_recap:
        st.markdown(f"- **{g.get('title', '')}** — {g.get('plain_english', '')}")

st.markdown(f"**What I'll build:** {conf_summary.get('what_will_be_built', '')}")

open_questions = conf_summary.get("open_questions") or []
if open_questions:
    st.markdown("**Open questions / things to verify:**")
    for q in open_questions:
        st.markdown(f"- ⚠️ {q}")

with st.expander("Confirmation (full structured object)"):
    st.json(confirmation)

if not st.session_state.confirmation_accepted:
    if st.button("Confirm and build dashboard ➜", type="primary"):
        st.session_state.confirmation_accepted = True
        st.session_state.debug_logger.log_event(
            "confirmation_user_accepted",
            {
                "confirmation_signature": confirmation_signature,
                "summary": conf_summary,
            },
        )
        _save_project_state()
        _advance_to(6)
else:
    st.success("Context confirmed. Scroll down for Step 6 (Dashboard, Insights & Q&A).")

if current_step < 6 or not st.session_state.confirmation_accepted:
    st.stop()


# =====================================================================
# STEP 6: Context-aware Dashboard, Insights & Q&A (Stages 8 + 10 + 11 + 12)
# =====================================================================
st.header("Step 6: Dashboard, Insights & Q&A (context-aware)")

with st.expander("Active context (Stage 6 — drives all stages below)"):
    st.json(context.to_dict())

st.subheader("Auto Dashboard (goal-aware)")
dashboard_plan, dashboard_debug = build_dashboard_plan(
    schema,
    use_ai=settings.dashboard_ai_enabled,
    return_debug=True,
    semantic_hints=semantic_hints,
    context=context,
    debug_logger=st.session_state.debug_logger,
)

dashboard_signature = "|".join([f"{item.kind}:{item.sql}" for item in dashboard_plan])
if st.session_state.last_dashboard_signature != dashboard_signature:
    st.session_state.debug_logger.log_event("dashboard_planning", dashboard_debug)
    st.session_state.last_dashboard_signature = dashboard_signature

kpi_cols = st.columns(2)
kpi_cursor = 0

for item in dashboard_plan:
    try:
        result = execute_select_query(st.session_state.con, item.sql)
    except Exception as sql_exc:
        st.session_state.debug_logger.log_event(
            "dashboard_query_failed",
            {
                "kind": item.kind,
                "title": item.title,
                "sql": item.sql,
                "source": getattr(item, "source", None),
                "error": str(sql_exc),
            },
        )
        st.caption(f"⚠️ Chart '{item.title}' skipped — SQL error: {sql_exc}")
        continue

    st.session_state.debug_logger.log_event(
        "dashboard_query_executed",
        {
            "kind": item.kind,
            "title": item.title,
            "sql": item.sql,
            "source": getattr(item, "source", None),
            "priority_score": getattr(item, "priority_score", None),
            "goal_id": getattr(item, "goal_id", None),
            "row_count": len(result),
            "columns": list(result.columns),
        },
    )

    try:
        if item.kind == "kpi":
            value = result.iloc[0, 0] if not result.empty else None
            kpi_cols[kpi_cursor % 2].metric(item.title, value=f"{value}")
            kpi_cursor += 1
        elif item.kind == "line" and not result.empty:
            if "dt" not in result.columns or "value" not in result.columns:
                st.caption(f"⚠️ '{item.title}' skipped — expected columns dt, value, got: {list(result.columns)}")
                continue
            result["dt"] = pd.to_datetime(result["dt"], errors="coerce")
            fig = px.line(result, x="dt", y="value", title=item.title)
            st.plotly_chart(fig, use_container_width=True)
        elif item.kind == "bar" and not result.empty:
            if "category" not in result.columns or "total" not in result.columns:
                st.caption(f"⚠️ '{item.title}' skipped — expected columns category, total, got: {list(result.columns)}")
                continue
            fig = px.bar(result, x="category", y="total", title=item.title)
            st.plotly_chart(fig, use_container_width=True)
        elif item.kind == "hist" and not result.empty:
            if "value" not in result.columns:
                st.caption(f"⚠️ '{item.title}' skipped — expected column value, got: {list(result.columns)}")
                continue
            fig = px.histogram(result, x="value", nbins=30, title=item.title)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as render_exc:
        st.session_state.debug_logger.log_event(
            "dashboard_render_failed",
            {"title": item.title, "kind": item.kind, "error": str(render_exc)},
        )
        st.caption(f"⚠️ '{item.title}' failed to render: {render_exc}")

st.subheader("Ask a Question (goal-aware)")
question = st.text_input("Example: Top 5 products by revenue")

if st.button("Run Question") and question.strip():
    try:
        st.session_state.debug_logger.log_event(
            "qa_sql_context_received",
            {
                "question": question,
                "context_focus_metric": context.focus_metric,
                "context_focus_dimensions": context.focus_dimensions,
                "context_active_goal_count": len(context.active_goals),
            },
        )
        understanding_summary = (st.session_state.understanding or {}).get("summary", {})
        sql = generate_sql(
            question,
            schema.columns,
            semantic_hints=semantic_hints,
            context_block=context_prompt_block,
            column_types={
                "identifier_columns": understanding_summary.get("identifier_columns") or [],
                "measure_columns": understanding_summary.get("measure_columns") or [],
                "dimension_columns": understanding_summary.get("dimension_columns") or [],
                "time_column": context.time_column,
            },
        )
        st.code(sql, language="sql")
        st.session_state.debug_logger.log_event(
            "qa_sql_generated",
            {
                "question": question,
                "sql": sql,
                "semantic_hints": semantic_hints,
                "context_used": True,
            },
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

st.subheader("Business Insight (goal-aligned)")
st.session_state.debug_logger.log_event(
    "insight_signals_context_received",
    {
        "context_focus_metric": context.focus_metric,
        "context_focus_dimensions": context.focus_dimensions,
        "context_active_goals": [
            {"id": g.get("id"), "metric": g.get("metric"), "dimension": g.get("dimension"), "direction": g.get("direction")}
            for g in context.active_goals
        ],
    },
)
insight_signals = generate_insight_signals_v2(
    df,
    schema,
    semantic_hints=semantic_hints,
    context=context,
)
deterministic_text = generate_deterministic_insight_text(insight_signals)

narrative_text = deterministic_text
narrative_source = "deterministic"
narrative_error = None

try:
    st.session_state.debug_logger.log_event(
        "insight_narrative_context_received",
        {
            "context_primary_goal": context.primary_goal_text,
            "context_active_goal_count": len(context.active_goals),
            "context_focus_metric": context.focus_metric,
            "context_focus_dimensions": context.focus_dimensions,
        },
    )
    narrative_text = generate_insight_narrative(
        insight_signals,
        semantic_hints=semantic_hints,
        context_block=context_prompt_block,
    )
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
        "goal_alignment_applied": insight_signals.get("goal_alignment_applied"),
        "active_goal_count": insight_signals.get("active_goal_count"),
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
