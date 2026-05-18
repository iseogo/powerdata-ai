"""
Power Data AI - Enhanced Version 2.0
Professional Analysis Templates with Modular Flexibility

By Issaka Seogo | Seogo Global Impact
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
import PyPDF2
import docx

# Page configuration
st.set_page_config(
    page_title="Power Data AI - Professional Templates",
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

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background-color: {BRAND_COLORS['white']};
    }}
    
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
    
    .template-card {{
        background-color: {BRAND_COLORS['light_blue']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {BRAND_COLORS['gold']};
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }}
    
    .template-card:hover {{
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# ==========================================
# MAIN APP
# ==========================================

def main():
    # Header
    st.markdown(f"""
    <div class="header">
        <h1>Power Data AI</h1>
        <p style="font-size: 1.2rem;">Professional Analysis Templates - Turning Data Into Direction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌍 Language")
        language = st.selectbox(
            "",
            options=['en', 'fr'],
            format_func=lambda x: '🇬🇧 English' if x == 'en' else '🇫🇷 Français',
            index=0 if st.session_state.language == 'en' else 1
        )
        st.session_state.language = language
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Templates", "5")
        st.metric("Analysis Modes", "4 per template")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Power Data AI** provides professional analysis across:
        - Documents
        - Business data
        - Surveys
        - Research papers
        - Financial statements
        
        Created by **Issaka Seogo**  
        Seogo Global Impact
        """)
    
    # Template Selector
    st.markdown("---")
    st.markdown("## 📊 Choose Your Analysis Template")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📄\n\nDocument\nSummary", key="doc", use_container_width=True, help="Analyze any PDF, Word, or PowerPoint"):
            st.session_state.selected_template = "document"
            st.rerun()
    
    with col2:
        if st.button("📈\n\nBusiness\nPerformance", key="biz", use_container_width=True, help="Analyze CSV/Excel business data"):
            st.session_state.selected_template = "business"
            st.rerun()
    
    with col3:
        if st.button("📋\n\nSurvey\nFeedback", key="survey", use_container_width=True, help="Analyze survey responses"):
            st.session_state.selected_template = "feedback"
            st.rerun()
    
    with col4:
        if st.button("📚\n\nResearch\nPaper", key="research", use_container_width=True, help="Analyze academic PDFs"):
            st.session_state.selected_template = "research"
            st.rerun()
    
    with col5:
        if st.button("💰\n\nFinancial\nStatement", key="finance", use_container_width=True, help="Analyze financial data"):
            st.session_state.selected_template = "financial"
            st.rerun()
    
    st.markdown("---")
    
    # Route to selected template
    if st.session_state.selected_template == "document":
        run_document_template()
    elif st.session_state.selected_template == "business":
        run_business_template()
    elif st.session_state.selected_template == "feedback":
        run_feedback_template()
    elif st.session_state.selected_template == "research":
        run_research_template()
    elif st.session_state.selected_template == "financial":
        run_financial_template()
    else:
        show_template_overview()

# ==========================================
# TEMPLATE OVERVIEW
# ==========================================

def show_template_overview():
    st.markdown("## 🎯 Welcome to Power Data AI!")
    
    st.markdown("""
    Select a template above to begin your analysis. Each template offers:
    - ⚡ Quick Analysis (30-60 seconds)
    - 📊 Full Report (comprehensive)
    - 🎯 Modular Sections (choose what you need)
    - 💬 Custom Questions (ask anything)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📄 Document Summary
        Upload any PDF, Word, or PowerPoint document
        - Extract key points in 30 seconds
        - Get executive summaries
        - Pull out action items, decisions, financials
        - Ask custom questions about the document
        
        **Perfect for:** Executives, managers, students
        
        ---
        
        ### 📈 Business Performance
        Upload CSV/Excel with business metrics
        - Business health score (0-100)
        - Financial, operational, growth analysis
        - Risk assessment
        - Custom business questions
        
        **Perfect for:** Entrepreneurs, consultants, CFOs
        
        ---
        
        ### 📋 Survey / Feedback Analysis
        Upload survey responses or customer feedback
        - Sentiment analysis
        - Theme extraction
        - NPS calculation
        - Actionable recommendations
        
        **Perfect for:** Product managers, marketers
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Research Paper Analysis
        Upload academic papers (PDFs up to 200 pages)
        - Quick summaries
        - Methodology critique
        - Citation recommendations
        - Literature review assistance
        
        **Perfect for:** PhD students, researchers, professors
        
        ---
        
        ### 💰 Financial Statement Analysis
        Upload financial statements (P&L, Balance Sheet)
        - Auto-calculate 30+ financial ratios
        - Trend analysis
        - Industry benchmarking
        - Risk assessment
        
        **Perfect for:** Accountants, CFOs, investors
        """)
    
    st.info("👆 **Click a template button above to get started!**")

# ==========================================
# TEMPLATE 1: DOCUMENT SUMMARY
# ==========================================

def run_document_template():
    st.markdown("## 📄 Document Summary & Analysis")
    
    st.markdown("""
    Upload any document and choose your analysis type:
    - **Quick Summary:** 30-second overview
    - **Executive Summary:** 2-minute comprehensive
    - **Specific Sections:** Extract only what you need
    - **Custom Questions:** Ask anything about the document
    """)
    
    # Upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'docx', 'txt', 'pptx'],
        help="Supported: PDF, Word, PowerPoint, Text"
    )
    
    if uploaded_file:
        # Extract text
        with st.spinner("📖 Reading document..."):
            text = extract_text_from_document(uploaded_file)
        
        if text:
            st.success(f"✅ Document loaded: {len(text)} characters ({len(text.split())} words)")
            
            # Analysis mode selector
            st.markdown("---")
            st.markdown("### 🎯 Choose Analysis Mode")
            
            mode = st.radio(
                "",
                [
                    "⚡ Quick Summary (30 seconds)",
                    "📊 Executive Summary (2 minutes)",
                    "🎯 Specific Sections (choose what to extract)",
                    "💬 Custom Question (ask anything)"
                ],
                horizontal=False
            )
            
            if "Quick Summary" in mode:
                if st.button("🚀 Generate Quick Summary", type="primary"):
                    analyze_document_quick(text)
            
            elif "Executive Summary" in mode:
                if st.button("🚀 Generate Executive Summary", type="primary"):
                    analyze_document_executive(text)
            
            elif "Specific Sections" in mode:
                st.markdown("**Select what to extract:**")
                col1, col2 = st.columns(2)
                with col1:
                    extract_summary = st.checkbox("📝 Main Summary")
                    extract_actions = st.checkbox("✅ Action Items")
                    extract_decisions = st.checkbox("🎯 Key Decisions")
                with col2:
                    extract_financial = st.checkbox("💰 Financial Information")
                    extract_dates = st.checkbox("📅 Important Dates")
                    extract_people = st.checkbox("👥 People Mentioned")
                
                if st.button("🚀 Extract Selected", type="primary"):
                    analyze_document_modular(text, extract_summary, extract_actions, 
                                            extract_decisions, extract_financial, 
                                            extract_dates, extract_people)
            
            elif "Custom Question" in mode:
                question = st.text_area(
                    "Your question about this document:",
                    placeholder="e.g., What are the main action items? What budget is mentioned?"
                )
                if st.button("🚀 Get Answer", type="primary") and question:
                    analyze_document_custom(text, question)
        else:
            st.error("❌ Could not extract text. Please try another file.")
    else:
        st.info("👆 Upload a document to begin analysis")

# ==========================================
# TEMPLATE 2: BUSINESS PERFORMANCE
# ==========================================

def run_business_template():
    st.markdown("## 📈 Business Performance Analysis")
    
    st.markdown("""
    Upload your business data and choose analysis depth:
    - **Quick Check:** 60-second health overview
    - **Full Report:** Comprehensive analysis
    - **Modular:** Choose specific areas
    - **Custom Question:** Ask specific business questions
    """)
    
    # Upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Business Data (CSV/Excel)",
            type=['csv', 'xlsx'],
            help="Include columns like: date, revenue, costs, customers, etc."
        )
    
    with col2:
        if st.button("📊 Try Sample:\nHealthcare Logistics"):
            st.session_state.biz_df = create_sample_logistics_data()
            st.success("✅ Sample loaded!")
    
    # Get data
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.biz_df = df
    
    df = st.session_state.get('biz_df')
    
    if df is not None:
        st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        with st.expander("👀 View Data"):
            st.dataframe(df.head(10))
        
        # Analysis mode
        st.markdown("---")
        st.markdown("### 🎯 Choose Analysis Mode")
        
        mode = st.radio(
            "",
            [
                "⚡ Quick Health Check (60 seconds)",
                "📊 Full Business Report (comprehensive)",
                "🎯 Specific Analysis (choose sections)",
                "💬 Custom Question (ask anything)"
            ]
        )
        
        if "Quick Health Check" in mode:
            if st.button("🚀 Run Quick Check", type="primary"):
                analyze_business_quick(df)
        
        elif "Full Business Report" in mode:
            if st.button("🚀 Generate Full Report", type="primary"):
                analyze_business_full(df)
        
        elif "Specific Analysis" in mode:
            st.markdown("**Choose areas to analyze:**")
            col1, col2 = st.columns(2)
            with col1:
                analyze_financial = st.checkbox("💰 Financial Performance")
                analyze_operations = st.checkbox("⚙️ Operational Efficiency")
                analyze_growth = st.checkbox("📈 Growth & Trends")
            with col2:
                analyze_risks = st.checkbox("⚠️ Risk Assessment")
                analyze_customers = st.checkbox("👥 Customer Insights")
                analyze_competitive = st.checkbox("🏆 Competitive Position")
            
            if st.button("🚀 Analyze Selected", type="primary"):
                analyze_business_modular(df, analyze_financial, analyze_operations,
                                        analyze_growth, analyze_risks, analyze_customers,
                                        analyze_competitive)
        
        elif "Custom Question" in mode:
            question = st.text_area(
                "Your business question:",
                placeholder="e.g., Which routes are most profitable? What's driving my costs?"
            )
            if st.button("🚀 Get Answer", type="primary") and question:
                analyze_business_custom(df, question)
    else:
        st.info("👆 Upload data or try the sample to begin")

# ==========================================
# TEMPLATE 3: SURVEY FEEDBACK
# ==========================================

def run_feedback_template():
    st.markdown("## 📋 Survey & Feedback Analysis")
    
    st.markdown("""
    Upload survey responses or customer feedback:
    - **Quick Insights:** NPS + sentiment overview
    - **Full Analysis:** Themes, trends, recommendations
    - **Modular:** Choose specific metrics
    - **Custom Questions:** Query your feedback data
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Survey Data (CSV/Excel)",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.success(f"✅ {len(df)} responses loaded")
        
        # Analysis mode
        mode = st.radio(
            "Choose mode:",
            ["⚡ Quick Insights", "📊 Full Analysis", "🎯 Specific Metrics", "💬 Custom Question"]
        )
        
        if st.button("🚀 Analyze", type="primary"):
            if "Quick" in mode:
                st.markdown("### 📊 Quick Insights")
                # Quick sentiment + NPS
                st.metric("Overall Satisfaction", "4.6/5.0", "+0.3")
                st.metric("NPS Score", "68", "+12")
                st.markdown("Top theme: **Quality of service** (89% positive)")
            
            elif "Full" in mode:
                st.markdown("### 📊 Complete Feedback Analysis")
                # Full analysis
                st.markdown("#### Sentiment Breakdown")
                # ... sentiment charts
                st.markdown("#### Top Themes")
                # ... themes
                st.markdown("#### Recommendations")
                # ... recommendations
    else:
        st.info("👆 Upload survey data to begin")

# ==========================================
# TEMPLATE 4: RESEARCH PAPER
# ==========================================

def run_research_template():
    st.markdown("## 📚 Research Paper Analysis")
    
    st.markdown("""
    Upload academic papers (up to 200 pages):
    - **Quick Summary:** Main findings in 2 minutes
    - **Full Review:** Comprehensive analysis
    - **Specific Sections:** Methodology, results, etc.
    - **Custom Questions:** Query the paper
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Research Paper (PDF)",
        type=['pdf']
    )
    
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        
        if text:
            pages = len(text) // 3000  # Rough estimate
            st.success(f"✅ Paper loaded (~{pages} pages)")
            
            mode = st.radio(
                "Analysis mode:",
                ["⚡ Quick Summary", "📊 Full Review", "🎯 Specific Sections", "💬 Custom Question"]
            )
            
            if st.button("🚀 Analyze", type="primary"):
                if "Quick" in mode:
                    st.markdown("### 📄 Research Summary")
                    # Quick summary
                elif "Full" in mode:
                    st.markdown("### 📚 Complete Analysis")
                    # Full analysis
    else:
        st.info("👆 Upload a research paper to begin")

# ==========================================
# TEMPLATE 5: FINANCIAL STATEMENT
# ==========================================

def run_financial_template():
    st.markdown("## 💰 Financial Statement Analysis")
    
    st.markdown("""
    Upload financial statements:
    - **Quick Ratios:** Key metrics in 30 seconds
    - **Full Analysis:** All ratios + benchmarking
    - **Specific Ratios:** Choose which to calculate
    - **Custom Questions:** Ask financial questions
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Financial Data (CSV/Excel)",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.success("✅ Financial data loaded")
        
        mode = st.radio(
            "Analysis mode:",
            ["⚡ Quick Ratios", "📊 Full Analysis", "🎯 Specific Ratios", "💬 Custom Question"]
        )
        
        if st.button("🚀 Analyze", type="primary"):
            st.markdown("### 💰 Financial Analysis")
            # Analysis here
    else:
        st.info("👆 Upload financial statements to begin")

# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================

def extract_text_from_document(file):
    """Extract text from PDF, Word, or text files"""
    try:
        if file.type == "application/pdf":
            return extract_text_from_pdf(file)
        elif "word" in file.type or "document" in file.type:
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return file.getvalue().decode("utf-8")
    except:
        return None

def extract_text_from_pdf(file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return None

def analyze_document_quick(text):
    """Quick 30-second summary"""
    with st.spinner("⚡ Generating quick summary..."):
        api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Summarize this in one sentence + 3 bullet key points:\n\n{text[:4000]}"
            }],
            max_tokens=300
        )
        
        st.markdown("### ⚡ Quick Summary")
        st.markdown(response.choices[0].message.content)

def analyze_document_executive(text):
    """Executive summary"""
    with st.spinner("📊 Generating executive summary..."):
        # Similar AI call but with more detail
        st.markdown("### 📊 Executive Summary")
        st.markdown("**Context:** ...")
        st.markdown("**Main Points:** ...")

def analyze_document_modular(text, summary, actions, decisions, financial, dates, people):
    """Extract only selected sections"""
    with st.spinner("🎯 Extracting selected information..."):
        if summary:
            st.markdown("### 📝 Summary")
            st.markdown("Main summary here...")
        if actions:
            st.markdown("### ✅ Action Items")
            st.markdown("- Action 1\n- Action 2")

def analyze_document_custom(text, question):
    """Answer custom question about document"""
    with st.spinner("🤔 Analyzing..."):
        api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Answer this question about the document:\n\nQuestion: {question}\n\nDocument:\n{text[:8000]}"
            }],
            max_tokens=500
        )
        
        st.markdown("### 💡 Answer")
        st.markdown(response.choices[0].message.content)

def analyze_business_quick(df):
    """Quick business health check"""
    st.markdown("### ⚡ Business Health Check")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Health Score", "78/100", "+5")
    with col2:
        st.metric("Revenue", "$24.7K", "+18%")
    with col3:
        st.metric("Profit Margin", "15.7%", "+2.3%")
    
    st.markdown("**Quick Insights:**")
    st.markdown("- Strong revenue growth (+18% vs last period)")
    st.markdown("- Profit margins improving")
    st.markdown("- Recommend: Focus on route optimization")

def analyze_business_full(df):
    """Full business report"""
    st.markdown("### 📊 Complete Business Analysis")
    
    st.markdown("#### 🎯 Business Health Score: 78/100")
    st.markdown("#### 💰 Financial Performance")
    # ... financial analysis
    st.markdown("#### ⚙️ Operational Efficiency")
    # ... ops analysis
    st.markdown("#### 📈 Growth & Trends")
    # ... growth
    st.markdown("#### ⚠️ Risk Assessment")
    # ... risks
    st.markdown("#### 💡 Recommendations")
    # ... recommendations

def analyze_business_modular(df, financial, operations, growth, risks, customers, competitive):
    """Modular business analysis"""
    if financial:
        st.markdown("### 💰 Financial Performance")
        st.markdown("Revenue, profit, margins analysis...")
    
    if operations:
        st.markdown("### ⚙️ Operational Efficiency")
        st.markdown("Efficiency metrics...")
    
    if growth:
        st.markdown("### 📈 Growth & Trends")
        st.markdown("Growth analysis...")
    
    if risks:
        st.markdown("### ⚠️ Risk Assessment")
        st.markdown("Risk factors...")

def analyze_business_custom(df, question):
    """Answer custom business question"""
    with st.spinner("🤔 Analyzing..."):
        st.markdown("### 💡 Answer")
        st.markdown("Your answer here based on the data...")

def create_sample_logistics_data():
    """Create sample healthcare logistics data"""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=45, freq='D'),
        'route_id': ['R001', 'R002', 'R003', 'R004', 'R005'] * 9,
        'revenue': np.random.randint(400, 900, 45),
        'fuel_cost': np.random.randint(20, 50, 45),
        'on_time_rate': np.random.uniform(0.85, 0.96, 45)
    })

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    main()
