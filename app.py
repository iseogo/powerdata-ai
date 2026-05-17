"""
Power Data AI - Enhanced Streamlit Application (Updated for OpenAI 1.0+)
By Issaka Seogo | Seogo Global Impact

Empowering data-driven decisions through intelligent analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from openai import OpenAI
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Power Data AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand colors
BRAND_COLORS = {
    'blue': '#003399',
    'gold': '#FFD700',
    'white': '#FFFFFF',
    'light_blue': '#E6F0FF',
    'dark_gold': '#B8960A'
}

# Custom CSS for branding
st.markdown(f"""
<style>
    /* Main branding */
    .main {{
        background-color: {BRAND_COLORS['white']};
    }}
    
    /* Header styling */
    .header {{
        background: linear-gradient(135deg, {BRAND_COLORS['blue']} 0%, {BRAND_COLORS['dark_gold']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }}
    
    .header h1 {{
        color: {BRAND_COLORS['gold']};
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .header p {{
        color: {BRAND_COLORS['white']};
        font-size: 1.2rem;
        font-style: italic;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {BRAND_COLORS['gold']};
        color: {BRAND_COLORS['blue']};
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {BRAND_COLORS['dark_gold']};
        transform: scale(1.05);
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: {BRAND_COLORS['light_blue']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {BRAND_COLORS['gold']};
        margin: 1rem 0;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        color: {BRAND_COLORS['blue']};
        font-style: italic;
        border-top: 2px solid {BRAND_COLORS['gold']};
        margin-top: 3rem;
    }}
</style>
""", unsafe_allow_html=True)

# Language translations
TRANSLATIONS = {
    'en': {
        'title': 'Power Data AI',
        'tagline': 'Turning your data into direction.',
        'upload': 'Upload Your Data',
        'analyze': 'Analyze',
        'question': 'Ask a Question About Your Data',
        'insights': 'Key Insights',
        'visualizations': 'Visualizations',
        'download': 'Download Report',
        'sample_data': 'Try Sample Data',
        'mode': 'Analysis Mode',
        'quick': 'Quick Insights',
        'deep': 'Deep Analysis',
        'custom': 'Custom Query',
        'about': 'About Power Data AI',
        'creator': 'Created by Issaka Seogo | Seogo Global Impact',
        'mission': 'Empowering leaders, analysts, and entrepreneurs through data-driven clarity.'
    },
    'fr': {
        'title': 'Power Data AI',
        'tagline': 'Transformez vos données en direction.',
        'upload': 'Téléchargez vos données',
        'analyze': 'Analyser',
        'question': 'Posez une question sur vos données',
        'insights': 'Informations clés',
        'visualizations': 'Visualisations',
        'download': 'Télécharger le rapport',
        'sample_data': 'Essayer des données exemple',
        'mode': "Mode d'analyse",
        'quick': 'Aperçu rapide',
        'deep': 'Analyse approfondie',
        'custom': 'Requête personnalisée',
        'about': 'À propos de Power Data AI',
        'creator': 'Créé par Issaka Seogo | Seogo Global Impact',
        'mission': 'Donner aux dirigeants, analystes et entrepreneurs une clarté basée sur les données.'
    }
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def get_text(key):
    """Get translated text"""
    return TRANSLATIONS[st.session_state.language][key]

def analyze_with_ai(df, question, mode='quick'):
    """Use OpenAI to analyze data and answer questions"""
    try:
        api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
        if not api_key:
            return "⚠️ OpenAI API key not configured. Please add it to Streamlit secrets."
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare data context
        data_summary = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample': df.head(5).to_dict(),
            'describe': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Create prompt based on mode
        if mode == 'quick':
            prompt = f"""Analyze this dataset and answer the question concisely.

Dataset info: {json.dumps(data_summary, indent=2)}

Question: {question}

Provide a clear, actionable answer in 2-3 sentences. Focus on what matters for business decisions."""
        
        elif mode == 'deep':
            prompt = f"""As Power Data AI, provide a comprehensive analysis.

Dataset info: {json.dumps(data_summary, indent=2)}

Question: {question}

Provide:
1. Overview - What the data shows
2. Key Insights - 3-5 bullet points
3. Why It Matters - Business interpretation
4. Recommended Actions - Next steps

Be warm, intelligent, and empowering."""
        
        else:  # custom
            prompt = f"""Dataset info: {json.dumps(data_summary, indent=2)}

User query: {question}"""
        
        # Use new OpenAI API format
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Power Data AI, a warm and intelligent data coach who helps people make confident, data-informed decisions. You explain concepts simply and always provide actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"⚠️ Analysis error: {str(e)}"

def create_auto_visualizations(df):
    """Automatically generate relevant visualizations"""
    charts = []
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Distribution of first numeric column
    if len(numeric_cols) > 0:
        fig = px.histogram(
            df, 
            x=numeric_cols[0],
            title=f'Distribution of {numeric_cols[0]}',
            color_discrete_sequence=[BRAND_COLORS['blue']]
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        charts.append(('distribution', fig))
    
    # 2. Top categories if categorical column exists
    if len(categorical_cols) > 0:
        value_counts = df[categorical_cols[0]].value_counts().head(10)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Top 10 {categorical_cols[0]}',
            color=value_counts.values,
            color_continuous_scale=[[0, BRAND_COLORS['blue']], [1, BRAND_COLORS['gold']]]
        )
        fig.update_layout(
            xaxis_title=categorical_cols[0],
            yaxis_title='Count',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        charts.append(('categories', fig))
    
    # 3. Correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale=[[0, BRAND_COLORS['blue']], [0.5, 'white'], [1, BRAND_COLORS['gold']]],
            aspect='auto'
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        charts.append(('correlation', fig))
    
    # 4. Scatter plot if 2+ numeric columns
    if len(numeric_cols) >= 2:
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f'{numeric_cols[1]} vs {numeric_cols[0]}',
            color=df[categorical_cols[0]] if len(categorical_cols) > 0 else None,
            color_discrete_sequence=[BRAND_COLORS['blue'], BRAND_COLORS['gold'], BRAND_COLORS['dark_gold']]
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        charts.append(('scatter', fig))
    
    return charts

def load_sample_data(dataset_name):
    """Load sample datasets"""
    if dataset_name == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        return df
    
    elif dataset_name == "Sample Sales":
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        return pd.DataFrame({
            'date': dates,
            'product': np.random.choice(['Product A', 'Product B', 'Product C'], 100),
            'sales': np.random.randint(1000, 10000, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'profit_margin': np.random.uniform(0.1, 0.4, 100)
        })
    
    return None

# Main App
def main():
    # Header
    st.markdown(f"""
    <div class="header">
        <h1>{get_text('title')}</h1>
        <p>{get_text('tagline')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Language selector
        st.session_state.language = st.selectbox(
            "🌍 Language / Langue",
            options=['en', 'fr'],
            format_func=lambda x: '🇬🇧 English' if x == 'en' else '🇫🇷 Français',
            index=0 if st.session_state.language == 'en' else 1
        )
        
        st.markdown("---")
        
        # Analysis mode
        analysis_mode = st.radio(
            get_text('mode'),
            ['quick', 'deep', 'custom'],
            format_func=lambda x: get_text(x)
        )
        
        st.markdown("---")
        
        # Sample data
        st.subheader(get_text('sample_data'))
        if st.button("📊 Iris Dataset"):
            st.session_state.df = load_sample_data("Iris")
            st.success("✅ Iris dataset loaded!")
        
        if st.button("💰 Sample Sales"):
            st.session_state.df = load_sample_data("Sample Sales")
            st.success("✅ Sales dataset loaded!")
        
        st.markdown("---")
        
        # About
        with st.expander(get_text('about')):
            st.markdown(f"""
            **{get_text('mission')}**
            
            {get_text('creator')}
            
            🌐 [seogoglobalimpacts.com](https://seogoglobalimpacts.com)
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(get_text('upload'))
        uploaded_file = st.file_uploader(
            "CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file to begin analysis"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.df is not None:
            st.metric("📊 Rows", f"{len(st.session_state.df):,}")
            st.metric("📋 Columns", len(st.session_state.df.columns))
    
    # Data preview
    if st.session_state.df is not None:
        st.markdown("---")
        
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Question interface
        st.markdown("---")
        st.subheader(get_text('question'))
        
        question = st.text_area(
            "Your question",
            placeholder="e.g., What are the top 5 products by sales? / Quels sont les 5 meilleurs produits par ventes?",
            height=100
        )
        
        if st.button(get_text('analyze'), type="primary"):
            if question:
                with st.spinner("🤔 Analyzing your data..."):
                    answer = analyze_with_ai(st.session_state.df, question, analysis_mode)
                    
                    # Display answer
                    st.markdown("### 💡 " + get_text('insights'))
                    st.markdown(f"""
                    <div class="metric-card">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'question': question,
                        'answer': answer
                    })
            else:
                st.warning("⚠️ Please enter a question")
        
        # Auto-visualizations
        st.markdown("---")
        st.subheader(get_text('visualizations'))
        
        with st.spinner("📊 Generating visualizations..."):
            charts = create_auto_visualizations(st.session_state.df)
            
            if charts:
                for i, (chart_type, fig) in enumerate(charts):
                    with st.container():
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome state
        st.info(f"""
        👋 **Welcome to Power Data AI!**
        
        Upload your data file or try a sample dataset to get started.
        
        Power Data AI helps you:
        - 📊 Understand your data instantly
        - 💡 Get AI-powered insights
        - 📈 Visualize trends automatically
        - 🎯 Make confident decisions
        
        No coding required. Just ask questions in plain English or French.
        """)
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <strong>Power Data AI</strong> – {get_text('tagline')}<br>
        By <strong>Issaka Seogo</strong> | Seogo Global Impact Ltd. Co.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
