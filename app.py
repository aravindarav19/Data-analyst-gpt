import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from together import Together
from datetime import datetime
import traceback

# ---------------- Page Config ----------------
st.set_page_config(page_title="Virtual Data Analyst Assistant", layout="wide")

# ---------------- Together Client ----------------
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
client = Together(api_key=TOGETHER_API_KEY)

# ✅ DeepSeek model (recommended default)
model = "deepseek-ai/DeepSeek-V3.1"
# Alternative (reasoning-heavy):
# model = "deepseek-ai/DeepSeek-R1"

# ---------------- UI ----------------
st.title("📊 Virtual Data Analyst Assistant")
st.write("Upload your CSV or Excel file and ask anything about the data!")

uploaded_file = st.file_uploader(
    "Upload your data file (.csv or .xlsx)", type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # ---------------- Load Data ----------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("🧾 Data Preview")
    st.dataframe(df.head())

    st.subheader("💬 Ask a Question About Your Data")
    user_question = st.text_input("Type your question...")

    if user_question:
        # Keep sample small to avoid token issues
        sample_df = df.head(5).iloc[:, :15]
        sample_data = sample_df.to_string(index=False)

        # ---------------- Main Analysis Prompt ----------------
        prompt = f"""
You are a professional data analyst.
Analyse the dataset sample and answer the user's question clearly and concisely.

DATA SAMPLE:
{sample_data}

USER QUESTION:
{user_question}

Answer:
"""

        with st.spinner("Analyzing..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.2,
                )
                ai_reply = response.choices[0].message.content.strip()
                st.success("🧠 Insight")
                st.write(ai_reply)

            except Exception as e:
                st.error("AI request failed.")
                st.text(traceback.format_exc())
                st.stop()

        # ---------------- Chart Suggestion Prompt ----------------
        chart_prompt = f"""
Based on the data sample and the user's question, suggest a chart.

DATA SAMPLE:
{sample_data}

USER QUESTION:
{user_question}

Respond in EXACTLY this format:
Chart Type: bar / line / pie
X-axis: column_name
Y-axis: column_name
"""

        try:
            chart_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": chart_prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            chart_reply = chart_response.choices[0].message.content.strip()
            st.info("📊 Chart Suggestion\n\n" + chart_reply)

        except Exception as e:
            st.warning("Chart suggestion failed.")
            st.text(traceback.format_exc())
            chart_reply = None

        # ---------------- Chart Rendering ----------------
        if chart_reply:
            try:
                lines = chart_reply.splitlines()
                chart_type = lines[0].split(":")[1].strip().lower()
                x_col = lines[1].split(":")[1].strip()
                y_col = lines[2].split(":")[1].strip()

                fig, ax = plt.subplots()

                if chart_type == "bar":
                    df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)
                elif chart_type == "line":
                    df.sort_values(x_col).plot(x=x_col, y=y_col, kind="line", ax=ax)
                elif chart_type == "pie":
                    df.groupby(x_col)[y_col].sum().plot(kind="pie", ax=ax, ylabel="")
                else:
                    raise ValueError("Unsupported chart type")

                st.pyplot(fig)

            except Exception as e:
                st.warning(f"⚠️ Could not generate chart: {e}")

        # ---------------- Feedback ----------------
        st.markdown("### 🗳️ Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.warning("Thanks — we'll use this to improve.")

        # ---------------- Download ----------------
        if st.button("📥 Download AI Answer"):
            filename = f"ai_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write("User Question:\n")
                f.write(user_question + "\n\n")
                f.write("AI Answer:\n")
                f.write(ai_reply + "\n\n")
                f.write("Chart Suggestion:\n")
                f.write(chart_reply or "N/A")

            with open(filename, "rb") as f:
                st.download_button(
                    "Download Insight (.txt)", data=f, file_name=filename
                )

else:
    st.info("👈 Please upload a CSV or Excel file to get started.")
