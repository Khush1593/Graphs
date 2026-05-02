# AI Business Intelligence Agent MVP

A step-by-step MVP that lets you:
- Upload a CSV
- Auto-generate a dashboard
- Ask natural language questions (LLM to SQL)
- Get safe SQL execution results
- See a basic business insight

## Run locally

1. Create and activate a Python virtual environment
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set API key(s)
4. Run app:
   - `streamlit run app.py`

## LLM providers

Set `LLM_PROVIDER` in `.env` to one of:
- `openai`
- `gemini`
- `groq`

The app supports all three providers.

## Optional AI Dashboard Planning

Set `DASHBOARD_AI_ENABLED=true` (default) to let the LLM suggest useful chart SQL.
If LLM suggestion fails or returns invalid content, the app automatically falls back to rule-based dashboard logic.

## Semantic Hint Layer (Business Context)

The app now infers business context from your dataset and passes it into LLM prompts:

```json
{
   "primary_metric": "revenue",
   "time_column": "order_date",
   "dimensions": ["region", "category"],
   "metric_candidates": ["revenue", "profit", "amount"]
}
```

This improves chart relevance and SQL generation quality by prioritizing true business metrics and dimensions.

## Insight Engine v2

Business insights now use a two-layer approach:

1. Deterministic signal engine (no LLM dependency)
- Growth signals: MoM, WoW, trend slope
- Top contributors: top 3 dimension contributors with share %
- Declining segments: negative growth segments vs previous period
- Concentration risk: top-share dependency risk
- Outliers/anomalies: z-score based anomaly count

2. LLM insight narrator
- LLM receives only structured signals + semantic hints
- LLM does not analyze raw rows directly
- If LLM fails, app falls back to deterministic narrative automatically

You can inspect the exact structured signals in the UI under "Insight Signals (v2)" and in debug logs.

## Insight Engine v2.1 (Actionable Layer)

The insight system now adds business decision support:

- Action recommendations per signal (growth, decline, concentration, anomalies)
- Priority tagging (`high`, `medium`, `low`) with UI badges
- Anomaly explanations with concrete date-level events (spike/drop examples)
- Cross-signal reasoning narratives (for example growth driver + dependency risk)
- Benchmark context (current period vs 6-period baseline)

UI now displays:

- Prioritized Insights
- Recommended Actions
- Full structured signals in the expander

## Debug Logs (Per Run)

- A `debug` folder is used for logs.
- Every new Streamlit session run creates a new file: `debug/run_YYYYMMDD_HHMMSS_<id>.md`
- Logs are written in easy-to-read Markdown with JSON blocks.
- Captured events include:
   - run start/config
   - CSV load + detected schema
   - dashboard planning (AI accepted/rejected/fallback)
   - dashboard SQL execution stats
   - Q&A SQL generation/safety/execution
   - final insight text

## Try these Questions:

- “Top 5 products by revenue”
- “Total revenue by region”
- “Sales trend over time”
- “Which category generates most revenue?”