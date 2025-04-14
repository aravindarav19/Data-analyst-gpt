import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from together import Together
from datetime import datetime

# Set page config
st.set_page_config(page_title="AI Data Analyst Assistant", layout="wide")

# Load Together AI API Key
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
client = Together(api_key=TOGETHER_API_KEY)
model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Title
st.title("📊 AI Data Analyst Assistant")
st.write("Upload your CSV and ask anything about the data!")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("💬 Ask a Question About Your Data")
    user_question = st.text_input("Type your question...")

    if user_question:
        sample_data = df.head(10).to_string(index=False)

        # Build AI prompt for answering the question
        prompt = f"""You are a data analyst assistant. Analyze the dataset and answer the following user question clearly.
        
        DATA SAMPLE:
        {sample_data}

        USER QUESTION:
        {user_question}

        Your answer:"""

        with st.spinner("Analyzing..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_reply = response.choices[0].message.content.strip()
            st.success("🧠 Insight")
            st.write(ai_reply)

        # Ask for chart recommendation
        chart_prompt = f"""Based on the data sample below, and the user's question, suggest a chart type and which columns to use.

        DATA SAMPLE:
        {sample_data}

        USER QUESTION:
        {user_question}

        Respond in this format ONLY:
        Chart Type: <bar/pie/line>
        X-axis: <column_name>
        Y-axis: <column_name>
        """

        chart_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": chart_prompt}]
        )

        chart_reply = chart_response.choices[0].message.content.strip()
        st.info("📊 Chart Suggestion:\n" + chart_reply)

        # Try to parse and plot the chart
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

            st.pyplot(fig)

        except Exception as e:
            st.warning("⚠️ Couldn't generate chart. Reason: " + str(e))

        # Feedback
        st.markdown("### 🗳️ Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.warning("Thanks — we'll use this to improve.")

        # Download insight
        if st.button("📥 Download AI Answer"):
            filename = f"ai_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write("User Question:\n")
                f.write(user_question + "\n\n")
                f.write("AI Answer:\n")
                f.write(ai_reply + "\n\n")
                f.write("Chart Suggestion:\n")
                f.write(chart_reply)
            with open(filename, "rb") as f:
                st.download_button("Download Insight (.txt)", data=f, file_name=filename)

else:
    st.info("👈 Please upload a CSV file to get started.")
