import streamlit as st
import time

st.markdown("""
# 👋 Welcome to Powerdata.ai
Your all-in-one workspace to analyze, visualize, and model data — using just natural language.
""")

with st.spinner("Loading your AI workspace..."):
    time.sleep(1)

st.success("Ready! Choose your task below.")

# ✅ Inject sample sales + iris data if none uploaded
import pandas as pd
if 'df' not in globals():
    df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=10, freq="M"),
        "Region": ["North", "South", "East", "West"] * 2 + ["North", "South"],
        "Product": ["A", "B", "C", "D"] * 2 + ["A", "C"],
        "Sales": [1200, 950, 780, 1430, 1130, 970, 810, 1540, 1180, 990]
    })

    iris_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_df = pd.read_csv(iris_url)
    iris_df.columns = [c.replace(" ", "_") for c in iris_df.columns]
    iris_df["source"] = "iris_sample"
    df = pd.concat([df, iris_df.reset_index(drop=True)], ignore_index=True, sort=False).fillna(0)

st.markdown("## 🗣️ Ask Questions or Make a Request")
st.markdown("Use the text box below, or click the microphone icon to speak.")

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
            document.getElementById('voiceInput').value = e.results[0][0].transcript;
            recognition.stop();
        };

        recognition.onerror = function(e) {
            recognition.stop();
        }
    }
}
</script>
<input id='voiceInput' name='voiceInput' placeholder='Click mic or type here...' style='width: 80%; height: 30px;'>
<button onclick='startDictation()'>🎤</button>
""", unsafe_allow_html=True)

# Text or Voice Input Box

# ✅ Updated OpenAI ChatCompletion block
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

user_prompt = st.text_input("Ask something about your data (type or speak):", "What does the data say about sales?")
if user_prompt:
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    try:
        with st.spinner("Generating AI response..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst. If a dataset is uploaded, use it to answer the user's request."},
                    {"role": "user", "content": f"{user_prompt}\n\nHere is a sample of the uploaded data:\n{df.head(10).to_string(index=False) if 'df' in locals() else 'No dataset uploaded.'}"}
                ]
            )
            reply = response.choices[0].message.content if response.choices else 'No response received.'
            st.session_state.qa_history.append((user_prompt, reply))
            st.markdown(f"**AI Response:**\n{reply}")
    except Exception as e:
        st.error("⚠️ Error: " + str(e))
    if 'df' not in locals():
        st.warning("⚠️ No dataset uploaded yet. AI responses may be generic.")
    else:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Your Uploaded Dataset", csv_data, file_name="uploaded_dataset.csv")
