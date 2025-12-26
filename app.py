import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from datetime import datetime
import traceback

st.set_page_config(page_title="Virtual Data Analyst Assistant", layout="wide")

# streamlit/secrets.toml:
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL_ANSWER = "deepseek-chat"      


MODEL_CHART = "deepseek-chat"        


st.title("📊 Virtual Data Analyst Assistant")
st.write("Upload your CSV or Excel file and ask anything about the data!")

uploaded_file = st.file_uploader("Upload your data file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload a CSV or Excel file to get started.")
    st.stop()


try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("🧾 Data Preview")
st.dataframe(df.head())

st.subheader("💬 Ask a Question About Your Data")
user_question = st.text_input("Type your question...")

if not user_question:
    st.stop()

sample_df = df.head(8).iloc[:, :20]  
sample_data = sample_df.to_string(index=False)

prompt = f"""
You are a professional data analyst.
Use the dataset sample to answer the user's question clearly.

DATA SAMPLE:
{sample_data}

USER QUESTION:
{user_question}

Answer (be clear, structured, and practical):
""".strip()

with st.spinner("Analysing..."):
    try:
        response = client.chat.completions.create(
            model=MODEL_ANSWER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        ai_reply = response.choices[0].message.content.strip()
        st.success("🧠 Insight")
        st.write(ai_reply)
    except Exception:
        st.error("DeepSeek request failed.")
        st.text(traceback.format_exc())
        st.stop()

chart_prompt = f"""
Based on the data sample and the user's question, suggest ONE suitable chart.

DATA SAMPLE:
{sample_data}

USER QUESTION:
{user_question}

Respond in EXACTLY this format (3 lines only):
Chart Type: bar|line|pie
X-axis: <exact_column_name>
Y-axis: <exact_column_name>
""".strip()

chart_reply = None
try:
    chart_response = client.chat.completions.create(
        model=MODEL_CHART,
        messages=[{"role": "user", "content": chart_prompt}],
        max_tokens=120,
        temperature=0.0,
    )
    chart_reply = chart_response.choices[0].message.content.strip()
    st.info("📊 Chart Suggestion\n\n" + chart_reply)
except Exception:
    st.warning("Chart suggestion failed.")
    st.text(traceback.format_exc())

if chart_reply:
    try:
        lines = [ln.strip() for ln in chart_reply.splitlines() if ln.strip()]
        if len(lines) < 3:
            raise ValueError("Chart response was not in the expected 3-line format.")

        chart_type = lines[0].split(":", 1)[1].strip().lower()
        x_col = lines[1].split(":", 1)[1].strip()
        y_col = lines[2].split(":", 1)[1].strip()

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Suggested columns not found. X='{x_col}' Y='{y_col}'. "
                f"Available columns: {', '.join(map(str, df.columns[:30]))}"
                + (" ..." if len(df.columns) > 30 else "")
            )

        if chart_type in {"bar", "pie"}:
            y_numeric = pd.to_numeric(df[y_col], errors="coerce")
            if y_numeric.isna().all():
                raise ValueError(f"Y-axis column '{y_col}' is not numeric and cannot be aggregated.")
            tmp = df.copy()
            tmp[y_col] = y_numeric

        fig, ax = plt.subplots()

        if chart_type == "bar":
            tmp.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)
            ax.set_xlabel(x_col)
            ax.set_ylabel(f"Sum of {y_col}")
        elif chart_type == "line":
           
            y_numeric = pd.to_numeric(df[y_col], errors="coerce")
            if y_numeric.isna().all():
                raise ValueError(f"Y-axis column '{y_col}' is not numeric and cannot be plotted as a line.")
            tmp = df.copy()
            tmp[y_col] = y_numeric
            tmp = tmp.sort_values(by=x_col)
            tmp.plot(x=x_col, y=y_col, kind="line", ax=ax)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        elif chart_type == "pie":
            tmp.groupby(x_col)[y_col].sum().plot(kind="pie", ax=ax, ylabel="")
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not generate chart: {e}")

st.markdown("### 🗳️ Was this answer helpful?")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Yes"):
        st.success("Thanks for your feedback!")
with col2:
    if st.button("👎 No"):
        st.warning("Thanks — we'll use this to improve.")


if st.button("📥 Download AI Answer"):
    filename = f"ai_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    content = (
        "User Question:\n"
        + user_question
        + "\n\nAI Answer:\n"
        + ai_reply
        + "\n\nChart Suggestion:\n"
        + (chart_reply or "N/A")
        + "\n"
    )

    st.download_button(
        "Download Insight (.txt)",
        data=content.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )
