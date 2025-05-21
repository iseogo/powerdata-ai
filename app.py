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
    "Dashboards",
    "Text Mining",
    "Forecasting",
    "Time Series",
    "Descriptive Statistics",
    "Supervised Learning",
    "Unsupervised Learning",
    "Semi-Supervised Learning",
    "Reinforcement Learning",
    "Machine Learning"
])

# ✅ Automated data cleaning
if st.checkbox("🧹 Auto-clean the dataset"):
    df_clean = st.session_state.df.copy()

    # 1. Standardize column names
    df_clean.columns = [c.strip().replace(" ", "_").lower() for c in df_clean.columns]

    # 2. Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # 3. Fill missing values
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])

    # 4. Convert date columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except:
                continue

    # 5. Trim whitespace in object columns
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].str.strip()

    # 6. Drop constant columns
    nunique = df_clean.nunique()
    const_cols = nunique[nunique == 1].index
    df_clean.drop(columns=const_cols, inplace=True)

    # 7. (Optional) Replace invalid datatypes if needed (basic)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = df_clean[col].astype(float)
            except:
                continue

    st.session_state.df = df_clean
    st.success("✅ Full cleaning complete: renamed cols, removed dups, filled nulls, trimmed, converted, dropped constants")

# ✅ Task handlers
if task == "AI Chat with Data":
    st.markdown("## 💬 AI Chat Sidebar")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 💭 Prompt")
        user_prompt = st.text_area("Type or speak a question:", "What does the data say about sales?", height=100)
        st.markdown("""
        <script>
        function startDictation() {
            if (window.hasOwnProperty('webkitSpeechRecognition')) {
                var recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = "en-US";
                recognition.start();
                recognition.onresult = function(e) {
                    const input = document.querySelector('textarea');
                    if (input) {
                        input.value = e.results[0][0].transcript;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    recognition.stop();
                };
                recognition.onerror = function(e) {
                    recognition.stop();
                }
            }
        }
        </script>
        <button onclick="startDictation()">🎤 Speak</button>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧠 AI Response")
        if user_prompt:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            try:
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful data analyst."},
                            {"role": "user", "content": f"{user_prompt}\n\nHere is a sample of the uploaded data:\n{st.session_state.df.head(10).to_string(index=False)}"}
                        ]
                    )
                    reply = response.choices[0].message.content
                    st.markdown(f"**Answer:**
{reply}")
            except Exception as e:
                st.error("⚠️ Error: " + str(e))




""", unsafe_allow_html=True)
""", unsafe_allow_html=True)


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

elif task == "Dashboards":
    st.write("## 📊 Interactive Dashboard")
    st.dataframe(st.session_state.df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(st.session_state.df.select_dtypes("number"))
    with col2:
        st.bar_chart(st.session_state.df.select_dtypes("number"))

elif task == "Text Mining":
    st.write("## 🧠 Text Mining")
    if 'text' in st.session_state.df.columns:
        st.text_area("Sample Text Column", value='
'.join(st.session_state.df['text'].astype(str).head()), height=150)
    else:
        st.info("No 'text' column found in dataset.")

elif task == "Forecasting":
    st.write("## 🔮 Forecasting")
    st.line_chart(st.session_state.df.select_dtypes("number"))

elif task == "Time Series":
    st.write("## ⏱️ Time Series Analysis")
    st.line_chart(st.session_state.df.select_dtypes("number"))

elif task == "Descriptive Statistics":
    st.write("## 📌 Descriptive Statistics")
    st.dataframe(st.session_state.df.describe())

elif task == "Supervised Learning":
    st.write("## 🎯 Supervised Learning")
    st.code("Train/test split, classification or regression model setup goes here")

elif task == "Unsupervised Learning":
    st.write("## 🧩 Unsupervised Learning")
    st.code("Clustering, PCA, and dimensionality reduction here")

elif task == "Semi-Supervised Learning":
    st.write("## 🧠 Semi-Supervised Learning")
    st.code("Label propagation or self-training logic here")

elif task == "Reinforcement Learning":
    st.write("## 🕹️ Reinforcement Learning")
    st.code("Simulation environment, rewards, and feedback loop logic here")

elif task == "Machine Learning":
    st.write("## 🤖 ML Module")
    st.write("Combine supervised and unsupervised model workflows here.")
