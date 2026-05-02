# 🧠 AI BI Copilot — Product Flow & Design Blueprint

This document defines the complete end-to-end product flow for the AI-powered Business Intelligence Copilot.  
It is intended for designers, frontend developers, and product teams to understand how the system should behave and be structured.

---

# 🎯 Product Overview

## Goal
Transform raw data into:
- Clear business understanding
- Actionable insights
- Interactive dashboards

## Core Flow
User connects data → AI understands → clarifies → defines goals → builds dashboard → enables Q&A

---

# 🟣 1. Data Source Connection

## Purpose
Allow users to easily connect their data.

## UI Components
- Upload CSV
- Connect Database (PostgreSQL, MySQL, etc.)
- (Optional future) Google Sheets

## UX Notes
- Keep it simple and fast
- Provide sample dataset option

## System Actions
- Load data into DuckDB
- Trigger schema detection

---

# 🟣 2. AI Data Analysis (Processing State)

## Purpose
Show that AI is analyzing the dataset.

## UI
Loader with steps:
- Reading data
- Detecting structure
- Understanding patterns

## Backend Actions
- Schema detection
- Semantic inference:
  - primary_metric
  - time_column
  - dimensions
- Data profiling

---

# 🟣 3. Metadata Preview

## Purpose
Show AI understanding before proceeding.

## UI Components
- Primary Metric
- Time Column
- Key Dimensions
- Dataset Summary:
  - number of rows
  - number of columns
  - date range

## Goal
User should confirm:
> “AI understood my data correctly”

---

# 🟣 4. AI Clarification Questions

## Purpose
Remove ambiguity and personalize analysis.

## UI Options
- Chat-style interface
- Form-style inputs

## Example Questions
- Which metric matters most?
- What is your main goal? (growth, cost reduction, etc.)
- Any specific area of focus?
- Preferred time granularity?

## UX Rules
- Max 4–5 questions
- Allow skip option
- Pre-fill suggestions

## System Impact
- Updates semantic layer
- Improves insights and charts

---

# 🟣 5. Goal Generation

## Purpose
Define business goals based on:
- Dataset
- User answers

## UI
Card-based goals:
- Increase revenue
- Reduce dependency
- Improve weak segments

## UX Features
- Editable goals
- Accept / modify / add custom

---

# 🟣 6. Context Document (System Brain)

## Purpose
Create a single source of truth for the system.

## UI Layout
Scrollable structured panel with:

### Sections
1. Data Summary  
2. Semantic Layer  
3. Business Goals  
4. Key Metrics  
5. Suggested Analysis Areas  

## Note
This powers:
- Chart generation
- Insights
- Q&A system

---

# 🟣 7. AI Understanding Confirmation

## Purpose
Ensure correctness before dashboard generation.

## UI
Summary text:
> “Here’s what I understand about your business…”

Includes:
- Primary metric
- Goals
- Focus areas

## Actions
- Confirm
- Edit

---

# 🟣 8. Chart Planning (AI + Rules)

## Purpose
Decide what charts and KPIs to generate.

## Logic
- AI-assisted planning
- Deterministic fallback

## Inputs
- Semantic hints
- Business goals
- Dataset schema

## Output
Structured JSON config

---

# 🟣 9. Chart Configuration (Internal)

## Example Structure
```json
{
  "charts": [
    {
      "type": "line",
      "title": "Revenue Over Time",
      "sql": "...",
      "priority": "high"
    }
  ],
  "kpis": [
    {
      "title": "Total Revenue",
      "value_sql": "..."
    }
  ]
}

```

## Purpose

- Drives frontend rendering
- Ensures consistency

---

# 🟣 10. Dashboard View

## Purpose
Main analytics interface

## Layout

### Top Section
- KPI Cards (3–5)

### Middle Section
Charts:
- Line chart (trend)
- Bar chart (categories)
- Histogram (distribution)

### Optional Side Panel
- Insights summary

---

# 🟣 11. Business Insights Panel

## Purpose
Display AI-generated insights

## Structure

🔴 Key Insights

🟡 Supporting Insights

🟢 Observations

## Each Insight Includes
- Description
- Supporting data
- Suggested action

---

# 🟣 12. Chat Interface (SQL RAG)

## Purpose
Allow interactive analysis

## UI
- Chat input
- Response table or chart

## Example Queries
- Why did revenue drop?
- Top categories last month

## Backend Flow
- Natural language → SQL
- SQL → DuckDB
- Results → UI

---

# 🟣 13. Debug / Transparency Layer (Optional)

## Purpose
Increase trust and debuggability

## Shows
- Semantic hints
- Generated SQL
- Insight signals

---

# 🟣 14. Full User Flow

Connect Data
  ↓
AI Analysis
  ↓
Clarification Questions
  ↓
Goal Setting
  ↓
Context Document
  ↓
Confirmation
  ↓
Chart Planning
  ↓
Dashboard + Insights
  ↓
Chat Exploration

---

# 🟣 15. Design Principles

1. Clarity over complexity

Keep UI simple and understandable.

2. Progressive disclosure

Reveal information step-by-step.

3. Trust building

Show what AI is doing.

4. Editable AI

User should always have control.

5. Fast feedback

Avoid long waiting states.