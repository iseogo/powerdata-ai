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

# ✅ Exportable Report Generator
if st.button("📝 Generate Full Report"):
    df = st.session_state.df
    report_sections = [
        "# 📊 Powerdata.ai Automated Report",
        f"""## Shape
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}""",
        f"""## Columns
- {', '.join(df.columns[:10])}""",
        f"""## Sample Preview
{df.head(5).to_markdown(index=False)}""",
        f"""## Summary Statistics
{df.describe(include='all').to_markdown()}"""
    ]}
- Columns: {df.shape[1]}""",
        f"""## Columns
- {', '.join(df.columns[:10])}""",
        f"""## Sample Preview
{df.head(5).to_markdown(index=False)}""",
        f"""## Summary Statistics
{df.describe(include='all').to_markdown()}"""
    ]}
- Columns: {df.shape[1]}",
        f"## Columns
- {', '.join(df.columns[:10])}",
        f"## Sample Preview
{df.head(5).to_markdown(index=False)}",
        f"## Summary Statistics
{df.describe(include='all').to_markdown()}"
    ]}
- Columns: {df.shape[1]}",
        f"## Columns
- {', '.join(df.columns[:10])}",
        "## Sample Preview
" + df.head(5).to_markdown(index=False),
        "## Summary Statistics
" + df.describe(include='all').to_markdown()
    ]
    full_report = "

".join(report_sections)
    st.markdown(full_report)
    st.download_button("📄 Download Report as TXT", full_report.encode(), file_name="powerdata_full_report.txt")

    # PDF and Word export (basic text for now)
    from io import BytesIO
    from docx import Document
    from fpdf import FPDF

    # Word
    doc = Document()
    doc.add_heading("Powerdata.ai Report", 0)
    doc.add_paragraph("Generated report summary:")
    for section in report_sections:
        doc.add_paragraph(section)
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    st.download_button("📄 Download Report as Word", doc_io, file_name="powerdata_report.docx")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Powerdata.ai Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    for section in report_sections:
        pdf.multi_cell(0, 10, section)
    pdf_io = BytesIO()
    pdf.output(pdf_io)
    pdf_io.seek(0)
    st.download_button("📄 Download Report as PDF", pdf_io, file_name="powerdata_report.pdf")

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

# ✅ Data Analysis Workflow (Download → Clean → Explore → Report)
if st.checkbox("📥 Run Full Data Analysis Workflow"):
    st.write("### Step 1: Preview Data")
    st.dataframe(st.session_state.df.head())

    st.write("### Step 2: Auto-cleaning")
    cleaned = st.session_state.df.copy()
    cleaned.columns = [c.strip().replace(" ", "_").lower() for c in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    if cleaned.isnull().sum().sum() > 0:
        cleaned = cleaned.fillna(cleaned.mode().iloc[0])
    for col in cleaned.select_dtypes(include='object').columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
    st.session_state.df = cleaned
    st.success("Cleaned data ready.")

    st.write("### Step 3: Summary Statistics")
    st.dataframe(cleaned.describe(include='all'))

    st.write("### Step 4: Auto-generated Report")
    report = f"""
    ## 🧾 Auto Report
    - Rows: {cleaned.shape[0]}
    - Columns: {cleaned.shape[1]}
    - Missing values: {cleaned.isnull().sum().sum()}
    - Top columns: {cleaned.columns[:5].tolist()}
    - Sample:
    {cleaned.head(3).to_markdown(index=False)}
    """
    st.markdown(report)
    st.download_button("📄 Download Report", report.encode("utf-8"), file_name="data_report.txt")


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
                    st.markdown(f"**Answer:** {reply}")
            except Exception as e:
                st.error("⚠️ Error: " + str(e))






    st.download_button("📥 Download Sample Dataset", st.session_state.df.to_csv(index=False).encode('utf-8'), file_name="demo_data.csv")

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
        st.text_area("Sample Text Column", value='\n'.join(st.session_state.df['text'].astype(str).head().tolist()), height=150)
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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    df = st.session_state.df
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for classification")
    else:
        target = st.selectbox("🎯 Target (Categorical)", cat_cols)
        features = st.multiselect("🔢 Features (Numeric)", num_cols, default=num_cols[:-1])

        if features and target:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.write("### ✅ Classification Report")
            st.text(classification_report(y_test, y_pred))
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

elif task == "Unsupervised Learning":
    st.write("## 🧩 Unsupervised Learning")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    df = st.session_state.df
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for clustering")
    else:
        scaled = StandardScaler().fit_transform(df[num_cols])
        sil_scores_all = []
        opt_k = 2
        best_score = -1
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            lbls = km.fit_predict(scaled)
            score = silhouette_score(scaled, lbls)
            sil_scores_all.append((k, score))
            if score > best_score:
                opt_k = k
                best_score = score

        sil_df = pd.DataFrame(sil_scores_all, columns=["k", "Silhouette Score"])
        st.line_chart(sil_df.set_index("k"))

        n_clusters = st.number_input("Suggested Optimal Clusters (auto-selected):", min_value=2, max_value=10, value=opt_k)



elif task == "Semi-Supervised Learning":
    st.write("## 🧠 Semi-Supervised Learning")
    st.code("Label propagation or self-training logic here")

elif task == "Reinforcement Learning":
    st.write("## 🕹️ Reinforcement Learning")
    st.code("Simulation environment, rewards, and feedback loop logic here")

elif task == "Machine Learning":
    st.write("## 🤖 Machine Learning Workflow")
    st.write("Select features and target to build a simple model:")
    df = st.session_state.df
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns to run ML")
    else:
        target = st.selectbox("🎯 Target Variable", num_cols)
        features = st.multiselect("🔢 Features", [col for col in num_cols if col != target], default=[col for col in num_cols if col != target])

        if features and target:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("### 🧠 Model Evaluation")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
            st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))
