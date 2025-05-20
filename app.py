import streamlit as st
import pandas as pd
import time
from openai import OpenAI

st.set_page_config(page_title="Powerdata.ai", layout="wide")

st.markdown("""
# 👋 Welcome to Powerdata.ai
Your all-in-one workspace to analyze, visualize, and model data — using just natural language.
""")

with st.spinner("Loading your AI workspace..."):
    time.sleep(1)

st.success("Ready! Choose your task below.")

# ✅ Load sample data if no upload
if 'df' not in st.session_state:
    sales_df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=10, freq="M"),
        "Region": ["North", "South", "East", "West"] * 2 + ["North", "South"],
        "Product": ["A", "B", "C", "D"] * 2 + ["A", "C"],
        "Sales": [1200, 950, 780, 1430, 1130, 970, 810, 1540, 1180, 990]
    })

    iris_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_df = pd.read_csv(iris_url)
    iris_df.columns = [c.replace(" ", "_") for c in iris_df.columns]
    iris_df["source"] = "iris_sample"

    df = pd.concat([sales_df, iris_df], ignore_index=True, sort=False).fillna(0)
    st.session_state.df = df

# ✅ Task selector
task = st.selectbox("Choose a task", [
    "AI Chat with Data",
    "Explore Data",
    "Summary Stats",
    "Visualization",
    "Machine Learning"
])

# ✅ Task handlers
if task == "AI Chat with Data":
    st.markdown("## 💬 Ask Questions About Your Data")
    user_prompt = st.text_input("Ask something about your data:", "What does the data say about sales?")

    if user_prompt:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        try:
            with st.spinner("Generating AI response..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": f"{user_prompt}\n\nHere is a sample of the uploaded data:\n{st.session_state.df.head(10).to_string(index=False)}"}
                    ]
                )
                reply = response.choices[0].message.content
                st.markdown(f"**AI Response:**\n{reply}")
        except Exception as e:
            st.error("⚠️ Error: " + str(e))

    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample Dataset", csv, file_name="demo_data.csv")

elif task == "Explore Data":
    st.dataframe(st.session_state.df.head())

elif task == "Summary Stats":
    st.write("## 📊 Summary Statistics")
    st.dataframe(st.session_state.df.describe(include='all'))

elif task == "Visualization":
    st.write("## 📈 Create a Chart")
    num_cols = st.session_state.df.select_dtypes("number").columns.tolist()
    x_axis = st.selectbox("X-axis", st.session_state.df.columns)
    y_axis = st.selectbox("Y-axis", num_cols)
    chart_type = st.radio("Chart Type", ["Line", "Bar", "Area"])
    if chart_type == "Line":
        st.line_chart(st.session_state.df[[x_axis, y_axis]])
    elif chart_type == "Bar":
        st.bar_chart(st.session_state.df[[x_axis, y_axis]])
    else:
        st.area_chart(st.session_state.df[[x_axis, y_axis]])

elif task == "Machine Learning":
    st.write("## 🤖 ML Module (coming soon)")
