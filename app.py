import streamlit as st
import time

st.markdown("""
# 👋 Welcome to Powerdata.ai
Your all-in-one workspace to analyze, visualize, and model data — using just natural language.
""")

with st.spinner("Loading your AI workspace..."):
    time.sleep(1)

st.success("Ready! Choose your task below.")

# Text or Voice Input Box

# ✅ Updated OpenAI ChatCompletion block
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

user_prompt = st.text_input("Ask something about your data (type or speak):", "What does the data say about sales?")
if user_prompt:
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    try:
        with st.spinner("Generating AI response..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst. If a dataset is uploaded, use it to answer the user's request."},
                    {"role": "user", "content": f"{user_prompt}\n\nHere is a sample of the uploaded data:\n{df.head(10).to_string(index=False) if 'df' in locals() else 'No dataset uploaded.'}"}
                ]
            )
            reply = response.choices[0].message.content
            st.session_state.qa_history.append((user_prompt, reply))
            st.markdown(f"**AI Response:**\n{reply}")
    except Exception as e:
        st.error(f⚠️ Error: {str(e)}")
    if 'df' not in locals():
        st.warning("⚠️ No dataset uploaded yet. AI responses may be generic.")
    else:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Your Uploaded Dataset", csv_data, file_name="uploaded_dataset.csv")
