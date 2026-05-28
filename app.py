import streamlit as st
import google.generativeai as genai
import pandas as pd
import sqlite3
import json
import os
import random
from typing import Any
from datetime import date, timedelta
from pypdf import PdfReader
from io import BytesIO


# ------------------------------------------------------------------ #
#  PROMPTS                                                             #
# ------------------------------------------------------------------ #
PROMPTS = {
    "T1": {
        "Prompt 1 — SQL Query Optimization": {
            "desc": "Optimize a slow 500M-row Snowflake fact table query.",
            "prompt": """
CONTEXT:
I am a data engineer working with a large fact table (~500M rows) called orders_fact in Snowflake.
The query below is taking over 10 minutes:

    SELECT c.customer_name, p.product_category,
           SUM(o.sale_amount) AS total_sales, COUNT(o.order_id) AS order_count
    FROM orders_fact o
    JOIN customers_dim c ON o.customer_id = c.customer_id
    JOIN products_dim  p ON o.product_id  = p.product_id
    WHERE o.order_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY c.customer_name, p.product_category
    ORDER BY total_sales DESC;

ROLE: Act as a senior Snowflake performance-tuning expert with 10+ years of experience.

TASK: Optimize the SQL query for maximum performance on Snowflake.

CONSTRAINTS:
- Focus ONLY on performance improvements (do not change business logic)
- Use Snowflake-specific features (clustering keys, result cache, materialized views)
- Assume tables are not currently clustered
- Output must be valid Snowflake SQL

FORMAT:
1. Optimized SQL with inline comments explaining each change
2. Bullet list of optimization techniques applied
3. Estimated performance impact per technique

EXAMPLES:
Good: Adding a WHERE filter on a clustered column to enable micro-partition pruning.
Bad: Changing SUM() to approximation — changes business logic.
""",
        },
        "Prompt 2 — Spark Pipeline Debugging": {
            "desc": "Debug a PySpark ETL job failing with 'No space left on device' on AWS EMR.",
            "prompt": """
CONTEXT:
I am building an Apache Spark ETL pipeline in PySpark that reads raw JSON logs from S3,
transforms them, and writes Parquet files back to S3. It fails intermittently with:

    org.apache.spark.SparkException: Task failed while writing rows.
    Caused by: java.io.IOException: No space left on device

Cluster: AWS EMR with 5 worker nodes (m5.2xlarge, 32 GB RAM each).
Input data: ~200 GB compressed JSON per run.

ROLE: Act as a senior data engineer specializing in Apache Spark and distributed systems on AWS.

TASK: Diagnose the root cause and provide a step-by-step remediation plan with PySpark code snippets.

CONSTRAINTS:
- Solutions must be in PySpark (Python API)
- Do not suggest hardware upgrades unless all software solutions are exhausted
- Order solutions by ease of implementation (quick wins first)
- Focus on local disk — S3 bucket size is unlimited

FORMAT:
### Root Cause Analysis
### Remediation Steps (ordered by priority, each with code snippet)
### Long-term Prevention

EXAMPLES:
Quick win: spark.conf.set("spark.sql.shuffle.partitions", "400") to reduce shuffle spill.
""",
        },
        "Prompt 3 — Data Quality Checks": {
            "desc": "Generate SQL data quality checks for a Snowflake transactions table.",
            "prompt": """
CONTEXT:
I am a data engineer responsible for a daily ingestion pipeline loading customer transaction
data into Snowflake. Target table schema:

    transactions (
        transaction_id   VARCHAR(36)  NOT NULL,
        customer_id      INTEGER      NOT NULL,
        amount           DECIMAL(12,2),
        currency         VARCHAR(3),
        transaction_date TIMESTAMP,
        status           VARCHAR(20),   -- values: completed, pending, failed
        merchant_name    VARCHAR(255)
    )

ROLE: Act as a data quality engineer experienced with dbt, Great Expectations, and SQL validation.

TASK: Generate a comprehensive set of data quality checks for the transactions table.

CONSTRAINTS:
- Plain SQL compatible with Snowflake
- Cover: nullability, uniqueness, referential integrity, value ranges, format, freshness
- Each check returns 0 rows when data is CLEAN (fail-fast pattern)
- Include severity: CRITICAL or WARNING for each check

FORMAT:
JSON array where each element has:
{"check_name": "...", "severity": "CRITICAL|WARNING", "description": "...", "sql": "..."}

EXAMPLES:
{"check_name": "null_transaction_id", "severity": "CRITICAL",
 "description": "transaction_id must never be null",
 "sql": "SELECT transaction_id FROM transactions WHERE transaction_id IS NULL"}
""",
        },
    },
    "T2": """
CONTEXT: You are processing e-commerce user activity logs for a data pipeline.
ROLE: Act as a data analyst extracting structured insights from raw activity logs.
TASK: Analyze the following user activity and return a structured JSON summary.

USER ACTIVITY LOGS:
{logs}

CONSTRAINTS:
- Return ONLY valid JSON — no markdown, no explanation, no code fences
- All numeric fields must be numbers (not strings)
- The insights array must contain exactly 3 meaningful business insights

FORMAT — return exactly this structure:
{{
  "summary": "<one-sentence summary>",
  "total_users": <integer>,
  "purchasing_users": <integer>,
  "total_revenue": <number>,
  "insights": ["<insight 1>", "<insight 2>", "<insight 3>"]
}}
""",
    "T3": """
CONTEXT: I am a data engineer who needs synthetic test data for pipeline testing.
ROLE: Act as a synthetic data generation expert.
TASK: Generate {n_rows} additional realistic rows following the EXACT same schema, column names, and value ranges as the sample data.

SCHEMA (column name -> data type): {schema}
SAMPLE ROWS: {sample_rows}

CONSTRAINTS:
- Use the exact same column names as in the schema
- Infer realistic value ranges, formats, and distributions from the sample rows
- Generate varied, non-duplicate values that are consistent with the apparent data domain
- Preserve data types: numbers as numbers, strings as strings, dates as date strings
- Return ONLY a valid JSON array with no markdown, no explanation, no code fences

FORMAT:
[{{ "<col1>": <val1>, "<col2>": <val2>, ... }}]
""",
    "T4": """
CONTEXT: You are a document Q&A assistant.
ROLE: Answer questions based ONLY on the provided document.
TASK: Answer the question below using only information from the document.

DOCUMENT:
\"\"\"{document}\"\"\"

QUESTION: {question}

CONSTRAINTS:
- If the answer is not in the document, say exactly:
  "This information is not available in the provided document."
- Be concise and direct.
- Do not hallucinate facts not present in the document.
""",
    "T5": """
CONTEXT: You are a SQL expert for a SQLite3 database. Today's date is {today_str}.
ROLE: Translate natural-language business questions into precise SQLite3 SQL queries.

SCHEMA:
- customer(customer_id INTEGER PK, name TEXT, email TEXT, join_date TEXT)
- sales(sale_id INTEGER PK, customer_id INTEGER FK, product TEXT, amount REAL, sale_date TEXT)

TASK: Translate this question into a valid SQLite3 SQL query:
"{nl_query}"

CONSTRAINTS:
- Output ONLY the raw SQL — no markdown, no explanation, no code fences
- Use SQLite3 date syntax: date('now'), date('now', '-3 days')
- Join tables whenever customer name is needed in output
- For "last N days" use: sale_date >= date('now', '-N days')
""",
}

# ------------------------------------------------------------------ #
#  PAGE CONFIG                                                         #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="GenAI Assignment — Module 7",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  GEMINI SETUP                                                        #
# ------------------------------------------------------------------ #
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("🤖 GenAI Assignment")
    st.caption("Module 7 — Nagarro")
    st.divider()

    api_key_input = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY,
        type="password",
        placeholder="Paste your Gemini key here",
        help="Get a free key at https://aistudio.google.com/app/api-keys",
    )

    if api_key_input:
        configure_fn = getattr(genai, "configure", None)
        model_cls = getattr(genai, "GenerativeModel", None)
        if callable(configure_fn) and callable(model_cls):
            configure_fn(api_key=api_key_input)
            model: Any = model_cls("gemini-2.5-flash-lite")
            st.success("API key set ✓", icon="✅")
        else:
            st.error("Installed Gemini SDK does not expose configure()/GenerativeModel().")
            model = None
    else:
        st.warning("Enter your Gemini API key to begin.", icon="🔑")
        model = None

    st.divider()
    task = st.radio(
        "Select Task",
        [
            "🏠 Overview",
            "T1 — Prompt Engineering",
            "T2 — Chat + JSON Output",
            "T3 — Data Augmentation",
            "T4 — Document Q&A",
            "T5 — NL → SQL",
        ],
    )


def call_gemini(prompt: str) -> str:
    """Call Gemini and strip any markdown fences from response."""
    if model is None:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith(("json", "sql")):
            text = text[text.index("\n"):]
        text = text.strip()
    return text


# ================================================================== #
#  OVERVIEW                                                            #
# ================================================================== #
if task == "🏠 Overview":
    st.title("Generative AI & AI Agenting — Module 7")
    st.markdown(
        "All **5 assignment tasks** in a single Streamlit app, powered by **Gemini 1.5 Flash**."
    )

    cols = st.columns(2)
    tasks_info = [
        ("T1", "Prompt Engineering", "3 structured DE prompts with Context, Role, Task, Constraints, Format & Examples."),
        ("T2", "Chat + JSON Output", "Send user activity logs → Gemini returns structured JSON (summary, revenue, insights)."),
        ("T3", "Data Augmentation", "Read a CSV (5 rows) → Gemini generates N synthetic rows matching the same schema."),
        ("T4", "Document Q&A", "Pass a Data Engineering document → ask factual, inferential, and out-of-scope questions."),
        ("T5", "NL → SQL", "Type a plain-English question → Gemini generates SQLite3 SQL → executed on a live DB."),
    ]
    for i, (tag, title, desc) in enumerate(tasks_info):
        with cols[i % 2]:
            st.info(f"**{tag} — {title}**\n\n{desc}")

    st.divider()
    st.markdown("**Tech Stack:** Python · Streamlit · Gemini 1.5 Flash · SQLite3 · Pandas · Azure App Service")


# ================================================================== #
#  TASK 1 — PROMPT ENGINEERING                                         #
# ================================================================== #
elif task == "T1 — Prompt Engineering":
    st.title("Task 1 — Prompt Engineering")
    st.markdown(
        "Three well-structured prompts for data engineering use cases. "
        "Each includes **Context · Role · Task · Constraints · Format · Examples**."
    )

    t1_prompts = PROMPTS["T1"]
    selected = st.selectbox("Choose a prompt", list(t1_prompts.keys()))
    st.info(f"**Use case:** {t1_prompts[selected]['desc']}")

    with st.expander("View full prompt sent to Gemini"):
        st.code(t1_prompts[selected]["prompt"], language="markdown")

    if st.button("▶ Run Prompt", type="primary"):
        with st.spinner("Waiting for Gemini..."):
            result = call_gemini(t1_prompts[selected]["prompt"])
        st.subheader("Gemini Response")
        st.markdown(result)


# ================================================================== #
#  TASK 2 — CHAT + JSON OUTPUT                                         #
# ================================================================== #
elif task == "T2 — Chat + JSON Output":
    st.title("Task 2 — Chat with LLM & Format Output")
    st.markdown(
        "Send user activity logs to Gemini and receive a **structured JSON** response "
        "with summary, revenue stats, and business insights."
    )

    default_logs = (
        "- User A logged in and purchased a laptop worth $1200\n"
        "- User B logged in but did not make any purchase\n"
        "- User C purchased a phone worth $800"
    )
    logs = st.text_area("User Activity Logs", value=default_logs, height=150)

    if st.button("▶ Analyze with Gemini", type="primary"):
        prompt = PROMPTS["T2"].format(logs=logs)
        with st.spinner("Analyzing with Gemini..."):
            raw = call_gemini(prompt)

        try:
            data = json.loads(raw)

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Users", data["total_users"])
            c2.metric("Purchasing Users", data["purchasing_users"])
            c3.metric("Total Revenue", f"${data['total_revenue']:,.2f}")
            conv = round(data["purchasing_users"] / data["total_users"] * 100)
            c4.metric("Conversion Rate", f"{conv}%")

            st.subheader("Summary")
            st.success(data["summary"])

            st.subheader("Business Insights")
            for i, insight in enumerate(data["insights"], 1):
                st.markdown(f"**{i}.** {insight}")

            st.subheader("Raw JSON")
            st.json(data)

        except json.JSONDecodeError:
            st.error("Could not parse JSON from Gemini. Raw response:")
            st.code(raw)


# ================================================================== #
#  TASK 3 — DATA AUGMENTATION                                          #
# ================================================================== #
elif task == "T3 — Data Augmentation":
    st.title("Task 3 — Data Generation & Augmentation")
    st.markdown(
        "Upload any CSV file and ask Gemini to generate more synthetic rows "
        "matching its schema, distributions, and value ranges."
    )

    DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_data/transactions.csv")
    uploaded_csv = st.file_uploader("Upload a CSV file (optional — uses sample transactions CSV if not provided)", type="csv")

    if uploaded_csv is not None:
        df_original = pd.read_csv(uploaded_csv)
    else:
        df_original = pd.read_csv(DEFAULT_CSV_PATH)

    st.subheader(f"Dataset — {len(df_original)} rows, {len(df_original.columns)} columns")
    st.dataframe(df_original, use_container_width=True)

    st.divider()
    n_rows = st.slider("Number of synthetic rows to generate", 5, 30, 10)

    if st.button("▶ Generate via Gemini", type="primary"):
        schema = {col: str(df_original[col].dtype) for col in df_original.columns}
        sample_rows = df_original.to_dict(orient="records")

        prompt = PROMPTS["T3"].format(
            n_rows=n_rows,
            schema=json.dumps(schema),
            sample_rows=json.dumps(sample_rows),
        )
        with st.spinner(f"Generating {n_rows} synthetic rows..."):
            raw = call_gemini(prompt)

        try:
            new_rows = json.loads(raw)
            df_new = pd.DataFrame(new_rows)
            df_augmented = pd.concat([df_original, df_new], ignore_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Original Rows", len(df_original))
            c2.metric("Synthetic Rows", len(df_new))
            c3.metric("Total Rows", len(df_augmented))

            st.subheader("Synthetic Rows Generated by Gemini")
            st.dataframe(df_new, use_container_width=True)

            st.subheader("Full Augmented Dataset")
            st.dataframe(df_augmented, use_container_width=True)

            csv_out = df_augmented.to_csv(index=False)
            st.download_button(
                "⬇ Download Augmented CSV",
                data=csv_out,
                file_name="transactions_augmented.csv",
                mime="text/csv",
            )

        except json.JSONDecodeError:
            st.error("Could not parse JSON. Raw response:")
            st.code(raw)


# ================================================================== #
#  TASK 4 — DOCUMENT Q&A                                               #
# ================================================================== #
elif task == "T4 — Document Q&A":
    st.title("Task 4 — Document Q&A")
    st.markdown("Upload a PDF and ask questions about its contents.")

    # ---- PDF Upload ----
    uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_pdf is not None:
        reader = PdfReader(BytesIO(uploaded_pdf.read()))
        DOCUMENT = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.success(f"PDF loaded: {uploaded_pdf.name} ({len(reader.pages)} page(s))")

        with st.expander("📄 View extracted text", expanded=False):
            st.code(DOCUMENT, language="text")

        st.subheader("Ask a Question")
        question = st.text_area(
            "Your question",
            height=100,
            placeholder="Type your question about the document...",
        )

        if st.button("▶ Ask Gemini", type="primary") and question.strip():
            prompt = PROMPTS["T4"].format(document=DOCUMENT, question=question)
            with st.spinner("Reading document and generating answer..."):
                answer = call_gemini(prompt)

            st.subheader("Gemini's Answer")
            st.markdown(answer)
    else:
        st.info("Please upload a PDF to get started.")


# ================================================================== #
#  TASK 5 — NATURAL LANGUAGE → SQL                                     #
# ================================================================== #
elif task == "T5 — NL → SQL":
    st.title("Task 5 — Natural Language → SQL")
    st.markdown(
        "Type a plain-English business question → Gemini generates a **SQLite3 SQL query** "
        "→ executed on a live database → results displayed."
    )

    DB_PATH = os.path.join(os.path.dirname(__file__), "sales.db")

    # ---- Build DB ----
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS customer (
        customer_id INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        email       TEXT NOT NULL,
        join_date   TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sales (
        sale_id     INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        product     TEXT NOT NULL,
        amount      REAL NOT NULL,
        sale_date   TEXT NOT NULL,
        FOREIGN KEY(customer_id) REFERENCES customer(customer_id)
    );
    """

    def ensure_db():
        if os.path.exists(DB_PATH):
            return
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(SCHEMA_SQL)
        customers = [
            (1, "Alice Johnson",  "alice@example.com",  "2023-06-01"),
            (2, "Bob Smith",      "bob@example.com",    "2023-07-15"),
            (3, "Carol White",    "carol@example.com",  "2023-08-20"),
            (4, "David Brown",    "david@example.com",  "2023-09-05"),
            (5, "Eva Martinez",   "eva@example.com",    "2024-01-10"),
        ]
        conn.executemany("INSERT INTO customer VALUES (?,?,?,?)", customers)
        today = date.today()
        products = ["Laptop", "Phone", "Tablet", "Headphones", "Smartwatch", "Monitor", "Keyboard"]
        sales, sid = [], 1
        for cid in range(1, 6):
            for _ in range(random.randint(2, 4)):
                d = (today - timedelta(days=random.randint(0, 2))).strftime("%Y-%m-%d")
                sales.append((sid, cid, random.choice(products), round(random.uniform(100, 2500), 2), d))
                sid += 1
            for _ in range(random.randint(3, 5)):
                d = (today - timedelta(days=random.randint(7, 90))).strftime("%Y-%m-%d")
                sales.append((sid, cid, random.choice(products), round(random.uniform(50, 1500), 2), d))
                sid += 1
        conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?)", sales)
        conn.commit()
        conn.close()

    ensure_db()

    # Show DB contents
    with st.expander("🗄️ View Database", expanded=False):
        conn = sqlite3.connect(DB_PATH)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**customer** table")
            st.dataframe(pd.read_sql("SELECT * FROM customer", conn), use_container_width=True)
        with c2:
            st.markdown("**sales** table (last 10)")
            st.dataframe(
                pd.read_sql("SELECT s.*, c.name FROM sales s JOIN customer c ON s.customer_id=c.customer_id ORDER BY sale_date DESC LIMIT 10", conn),
                use_container_width=True,
            )
        conn.close()

    st.subheader("Ask a Business Question")

    sample_nl = {
        "— Pick a sample question —": "",
        "Highest sales per customer (last 3 days)": "Show the highest sale amount done by each customer in the last 3 days",
        "Top spending customer overall": "Which customer has spent the most money overall?",
        "Products sold recently": "List all products sold in the last 3 days with customer name and amount",
        "Sales count per product": "How many sales were made per product across all time?",
        "Inactive customers": "Show customers who made no purchases in the last 30 days",
    }

    selected_nl = st.selectbox("Sample questions", list(sample_nl.keys()))
    nl_query = st.text_input(
        "Your question in plain English",
        value=sample_nl[selected_nl],
        placeholder="e.g. Which customer has the highest total sales this month?",
    )

    if st.button("▶ Generate & Run SQL", type="primary") and nl_query.strip():
        today_str = date.today().strftime("%Y-%m-%d")

        sql_prompt = PROMPTS["T5"].format(today_str=today_str, nl_query=nl_query)
        with st.spinner("Generating SQL with Gemini..."):
            sql = call_gemini(sql_prompt).strip().rstrip(";") + ";"

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        # Execute
        try:
            conn = sqlite3.connect(DB_PATH)
            df_result = pd.read_sql(sql, conn)
            conn.close()

            st.subheader("Query Results")
            if df_result.empty:
                st.warning("No results found for the query.")
            else:
                st.success(f"{len(df_result)} row(s) returned")
                st.dataframe(df_result, use_container_width=True)

        except Exception as e:
            st.error(f"SQL execution error: {e}")
            st.info("The generated SQL may have a syntax issue. Try rephrasing your question.")

    if st.button("🔄 Reset Database"):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        ensure_db()
        st.success("Database reset with fresh data!")
        st.rerun()