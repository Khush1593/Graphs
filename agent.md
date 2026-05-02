# 🚀 AI Business Intelligence Agent (MVP + Future-Ready Specification)

---

# 1. 📌 PRODUCT OVERVIEW

## 1.1 Vision
Build an AI-powered Business Intelligence system that:
- Connects to business data
- Automatically generates dashboards
- Allows natural language querying (Q&A)
- Provides actionable business insights

This system should evolve into:
> "ChatGPT for Business Data + AI Analyst"

---

## 1.2 MVP Objective (STRICT)

Build a **working system** where a user can:
1. Upload a CSV dataset
2. Automatically see a dashboard (charts + KPIs)
3. Ask questions in natural language
4. Get correct answers via SQL execution
5. See at least one meaningful business insight

---

## 1.3 NON-GOALS (Do NOT build in MVP)
- No authentication system
- No multi-user support
- No complex DB connectors (CSV only)
- No vector DB / RAG system
- No real-time streaming
- No heavy ML models

---

# 2. 🏗️ SYSTEM ARCHITECTURE

## 2.1 High-Level Flow

User → Upload CSV → Process Data → Generate Dashboard  
User → Ask Question → LLM → SQL → Execute → Return Result  
System → Generate Insight → Display  

---

## 2.2 Architecture

Frontend: Streamlit  
Backend: Python (single app)  
Database: DuckDB (in-memory or file-based)  
LLM: OpenAI API  

---

# 3. 🧩 MODULES (IMPLEMENTATION DETAILS)

---

# MODULE 1: DATA INGESTION

## Goal:
Accept CSV and load into queryable database

## Requirements:
- Accept `.csv` file upload
- Validate file
- Load into DuckDB as table `data`

## Implementation:
```python
import duckdb
con = duckdb.connect()
con.execute("CREATE TABLE data AS SELECT * FROM 'uploaded_file.csv'") 

Output:
Table data ready for queries
Extract schema:
column names
inferred data types

MODULE 2: SCHEMA ANALYSIS (BASIC)
Goal:

Understand dataset structure for dashboard + SQL generation

Tasks:
Identify:
date column (if exists)
numeric columns (revenue, price, etc.)
categorical columns (product, region)
Heuristic Rules:
Date column → contains "date" or datetime type
Numeric → int/float
Category → string with repeated values
Output:
{
  "date_column": "order_date",
  "numeric_column": "revenue",
  "category_column": "product"
}
MODULE 3: AUTO DASHBOARD GENERATION
Goal:

Generate 3–5 meaningful charts automatically

Charts (MANDATORY):
1. KPI
Total revenue (SUM)
2. Time Series
Revenue over time
3. Top Categories
Top 5 categories by revenue
4. Optional
Count of transactions
Distribution histogram
SQL Examples:
KPI
SELECT SUM(revenue) AS total_revenue FROM data
Time Series
SELECT order_date, SUM(revenue)
FROM data
GROUP BY order_date
ORDER BY order_date
Top Categories
SELECT product, SUM(revenue) as total
FROM data
GROUP BY product
ORDER BY total DESC
LIMIT 5
Output Format (Internal JSON)
[
  {
    "type": "kpi",
    "title": "Total Revenue",
    "sql": "SELECT SUM(revenue) FROM data"
  },
  {
    "type": "line",
    "title": "Revenue Over Time",
    "sql": "SELECT order_date, SUM(revenue) FROM data GROUP BY order_date"
  }
]
MODULE 4: SQL EXECUTION ENGINE
Goal:

Execute SQL safely on DuckDB

Requirements:
Execute SELECT queries only
Return result as DataFrame
MODULE 5: NATURAL LANGUAGE Q&A (TEXT → SQL)
Goal:

Convert user question into SQL and execute

Flow:
User input question
Send to LLM with schema
Receive SQL
Validate SQL
Execute SQL
Return result
Prompt Template:
You are a SQL expert.

Table name: data
Columns: {column_list}

Generate a SQL query for:
"{user_question}"

Rules:
- Only SELECT queries
- No modification queries
- Use only given columns
- Return only SQL
MODULE 6: SQL SAFETY LAYER
Goal:

Prevent harmful queries

Block keywords:
DELETE
UPDATE
DROP
ALTER
INSERT
Implementation:
def is_safe_query(sql):
    forbidden = ["DELETE", "UPDATE", "DROP", "ALTER", "INSERT"]
    return not any(word in sql.upper() for word in forbidden)
MODULE 7: INSIGHT ENGINE (BASIC BUT IMPORTANT)
Goal:

Provide at least 1 business insight

Logic:
Compare last period vs previous period
Example:
Monthly revenue comparison
Output:

"Sales dropped by 15% last month compared to previous month."

Implementation:
Use SQL to compute aggregates
Use Python logic to compare
MODULE 8: UI (STREAMLIT)
Components:
File uploader
Dashboard section (charts)
Chat input box
Results display (table + chart)
Insight section

Flow:
Upload CSV
Show dashboard
Ask question
Show result
Show insight
4. 📊 DATA FLOW
Upload CSV
Load into DuckDB
Extract schema
Generate dashboard JSON
Execute SQL for charts
Render charts
User asks question
LLM generates SQL
Validate SQL
Execute
Return result
Generate insight
5. 🎯 MVP QUALITY REQUIREMENTS
MUST HAVE:
No crashes
Correct SQL execution
Relevant charts
At least 1 useful insight
Response time < 5 seconds
ACCEPTABLE LIMITATIONS:
SQL may fail sometimes
UI can be basic
Insights can be simple
6. 🚀 FUTURE ROADMAP (IMPORTANT FOR AI AGENT CONTEXT)
Phase 2:
Database connectors (PostgreSQL, MySQL)
Better schema understanding
Multiple dashboards
Phase 3:
SQL RAG (schema + query memory)
Vector database
Context-aware querying
Phase 4:
AI business recommendations
Anomaly detection
Forecasting
Phase 5:
Multi-user SaaS platform
Authentication
Role-based dashboards
7. 🧠 DESIGN PRINCIPLES
Keep logic simple
Avoid overengineering
Prefer deterministic logic over AI when possible
Use AI only for:
SQL generation
Natural language explanation
8. ⚠️ COMMON FAILURE CASES
Wrong column mapping
Invalid SQL from LLM
Missing date column
Empty dataset
Handling:
Add fallback messages
Validate before execution
9. 🏁 FINAL DELIVERABLE

A working app where:

User uploads CSV
Dashboard appears automatically
User asks question → gets answer
Insight is displayed
10. 💡 DEMO SCRIPT
Upload dataset
Show charts instantly
Ask:
"Top 5 products by revenue"
Show result
Show insight:
"Sales dropped last month"
🔥 FINAL INSTRUCTION FOR AI AGENT

Build this system incrementally:

Step 1: CSV → DuckDB
Step 2: Schema detection
Step 3: Dashboard SQL + charts
Step 4: Q&A (LLM → SQL)
Step 5: SQL safety
Step 6: Insight generation
Step 7: UI integration

DO NOT SKIP STEPS
DO NOT ADD EXTRA FEATURES

Focus on:

Correctness
Simplicity
Working demo
