import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Powerdata.ai – Ask Your Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of your data:")
    st.dataframe(df.head())

    # Text input for natural language questions
    user_query = st.text_input("Ask a question about your data:")

    if user_query:
        with st.spinner("Thinking..."):
            prompt = f"You are a data analyst. Analyze this dataset and answer: {user_query}. Here's the data:\n{df.head(10).to_string()}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You help users understand and visualize data."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()

        st.write("### Answer:")
        st.write(answer)

        # Optional: simple example to generate a chart (hardcoded for now)
        if "top" in user_query.lower() and "products" in user_query.lower():
            try:
                top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
                fig, ax = plt.subplots()
                top_products.plot(kind='bar', ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.write("(Chart generation failed – check column names)")
