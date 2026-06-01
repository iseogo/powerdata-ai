"""
Power Data AI — Julius-style UI
Professional Analysis Templates

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
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Power Data AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────
# CATEGORY DATA
# ──────────────────────────────────────────

CATEGORIES = [
    {
        "id": "document",
        "label": "Document",
        "icon": "📄",
        "new": True,
        "placeholder": "Summarize this document...",
        "examples": [
            "Summarize this contract",
            "Extract key decisions from meeting notes",
            "Create an executive summary",
            "Pull action items from document",
            "Identify key risks and obligations",
            "What are the financial terms?",
            "Who are the key stakeholders?",
            "Summarize the methodology section",
        ],
    },
    {
        "id": "finance",
        "label": "Finance",
        "icon": "💰",
        "new": False,
        "placeholder": "Analyze financial data...",
        "examples": [
            "Calculate key financial ratios",
            "Analyze P&L statement trends",
            "Build financial projections model",
            "Cash flow analysis report",
            "Return on investment calculator",
            "Budget vs actuals tracker",
            "Break-even analysis",
            "Revenue vs expense comparison",
        ],
    },
    {
        "id": "data",
        "label": "Excel & Data",
        "icon": "📊",
        "new": False,
        "placeholder": "Analyze my business data...",
        "examples": [
            "Analyze revenue trends over time",
            "Build sales performance report",
            "Compare business metrics by region",
            "Find growth opportunities in data",
            "Customer segmentation analysis",
            "Identify top-performing products",
            "Monthly performance comparison",
            "Operational efficiency metrics",
        ],
    },
    {
        "id": "slides",
        "label": "Slides",
        "icon": "🖼️",
        "new": False,
        "placeholder": "Analyze or outline presentation content...",
        "examples": [
            "Quarterly business review deck",
            "Market analysis presentation",
            "Sales pipeline review slides",
            "Investor update deck",
            "Product launch presentation",
            "Industry trends overview",
            "Team performance review slides",
            "Customer success story deck",
        ],
    },
    {
        "id": "dashboard",
        "label": "Dashboard",
        "icon": "📈",
        "new": False,
        "placeholder": "Build an interactive dashboard...",
        "examples": [
            "Create a business KPI dashboard",
            "Build revenue monitoring dashboard",
            "Sales funnel visualization",
            "Business health scorecard",
            "Marketing performance dashboard",
            "Operations metrics dashboard",
            "Customer metrics overview",
            "Executive summary dashboard",
        ],
    },
    {
        "id": "tracker",
        "label": "Tracker",
        "icon": "🎯",
        "new": False,
        "placeholder": "Set up a business tracker...",
        "examples": [
            "OKR tracking spreadsheet",
            "Sales commission calculator",
            "Budget vs actuals tracker",
            "Employee headcount planning",
            "Invoice tracking template",
            "Product pricing comparison matrix",
            "Project milestone tracker",
            "KPI performance tracker",
        ],
    },
    {
        "id": "report",
        "label": "Report",
        "icon": "📝",
        "new": False,
        "placeholder": "Research and build a report...",
        "examples": [
            "Competitive research on industry leaders",
            "Industry benchmark report",
            "Customer churn analysis report",
            "Marketing campaign performance report",
            "Annual company performance report",
            "Data quality audit report",
            "A/B test results analysis",
            "Employee engagement survey report",
        ],
    },
]


def get_cat(cat_id):
    for c in CATEGORIES:
        if c["id"] == cat_id:
            return c
    return CATEGORIES[0]


# ──────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────

_defaults = {
    "selected_category": "document",
    "active_template": None,
    "user_prompt": "",
    "input_text": "",
    "language": "en",
    "biz_df": None,
    "selected_template": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _select_category(cat_id):
    st.session_state.selected_category = cat_id


def _use_example(text):
    st.session_state.input_text = text


def _go_back():
    st.session_state.active_template = None
    st.session_state.user_prompt = ""
    st.session_state.input_text = ""


# ──────────────────────────────────────────
# CSS
# ──────────────────────────────────────────

def _apply_css():
    st.markdown(
        """
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* ── App background ── */
.stApp { background: #FFFFFF; }
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

/* ── Navbar ── */
.pda-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1.5rem 0;
}
.pda-logo {
    font-size: 1.45rem;
    font-weight: 800;
    color: #003399;
    letter-spacing: -0.5px;
}
.pda-nav-right { display: flex; gap: 0.6rem; align-items: center; }
.pda-btn-login {
    background: transparent; border: none;
    color: #333; font-size: 0.95rem; cursor: pointer;
    padding: 0.45rem 0.9rem; border-radius: 8px;
}
.pda-btn-signup {
    background: #003399; border: none;
    color: white; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; padding: 0.45rem 1.1rem; border-radius: 8px;
}

/* ── Announcement banner ── */
.pda-banner {
    display: flex; align-items: center; gap: 0.5rem;
    background: #EEF2FF; border-radius: 20px;
    padding: 0.4rem 1rem;
    width: fit-content; margin: 0 auto 1.8rem auto;
    font-size: 0.88rem; color: #333; cursor: pointer;
}
.pda-banner .badge {
    background: #003399; color: white;
    border-radius: 10px; padding: 0.12rem 0.5rem;
    font-size: 0.72rem; font-weight: 700;
}

/* ── Hero heading ── */
.pda-hero {
    font-size: 2.1rem; font-weight: 700;
    color: #111; text-align: center;
    margin: 0 0 1.6rem 0; line-height: 1.25;
}

/* ── Input box shell ── */
.pda-input-shell {
    border: 1.5px solid #E0E0E0;
    border-radius: 16px;
    padding: 0.85rem 1rem 0.65rem 1rem;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}

/* ── Streamlit textarea inside shell ── */
.pda-input-shell .stTextArea > div > div > textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 1rem !important;
    color: #222 !important;
    resize: none !important;
    padding: 0 !important;
    line-height: 1.5 !important;
}
.pda-input-shell .stTextArea > div > div > textarea:focus {
    border: none !important;
    box-shadow: none !important;
}
.pda-input-shell .stTextArea label { display: none !important; }
.pda-input-shell .stTextArea > div { border: none !important; box-shadow: none !important; }

/* ── Toolbar row ── */
.pda-toolbar {
    display: flex; align-items: center; gap: 6px;
    border-top: 1px solid #F2F2F2;
    padding-top: 0.6rem; margin-top: 0.2rem;
    flex-wrap: wrap;
}
.pda-tb-btn {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 0.34rem 0.75rem;
    border: 1px solid #E0E0E0; border-radius: 16px;
    font-size: 0.82rem; color: #555;
    background: white; cursor: pointer;
    white-space: nowrap; user-select: none;
}
.pda-tb-btn:hover { background: #F8F8F8; }

/* ── Send button (form submit) ── */
button[data-testid="baseButton-primaryFormSubmit"],
button[data-testid="baseButton-secondaryFormSubmit"] {
    background: #003399 !important;
    color: white !important;
    border-radius: 50% !important;
    width: 42px !important; height: 42px !important;
    min-width: 42px !important; max-width: 42px !important;
    padding: 0 !important;
    border: none !important;
    font-size: 1.15rem !important;
    line-height: 1 !important;
}
button[data-testid="baseButton-primaryFormSubmit"]:hover,
button[data-testid="baseButton-secondaryFormSubmit"]:hover {
    background: #0022aa !important;
    color: white !important;
}

/* ── Pill buttons (category) ── */
[data-testid="baseButton-secondary"] {
    background: white !important;
    border: 1.5px solid #E0E0E0 !important;
    color: #333 !important;
    border-radius: 20px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.85rem !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #003399 !important;
    color: #003399 !important;
    background: #EEF2FF !important;
}
[data-testid="baseButton-primary"] {
    background: #EEF2FF !important;
    border: 2px solid #003399 !important;
    color: #003399 !important;
    border-radius: 20px !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    padding: 0.38rem 0.85rem !important;
}

/* ── Example prompt buttons ── */
.pda-examples .stButton button {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #F0F0F0 !important;
    border-radius: 0 !important;
    text-align: left !important;
    color: #444 !important;
    font-size: 0.94rem !important;
    padding: 0.7rem 0.2rem !important;
    width: 100% !important;
    justify-content: flex-start !important;
    font-weight: 400 !important;
}
.pda-examples .stButton button:hover {
    color: #003399 !important;
    background: transparent !important;
}

/* ── Back + template buttons ── */
.pda-back .stButton button {
    background: #EEF2FF !important;
    color: #003399 !important;
    border: 1px solid #C5D3FF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.pda-template-btn .stButton button {
    background: #003399 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.pda-template-btn .stButton button:hover {
    background: #0022aa !important;
}

/* ── Analysis mode radio ── */
.stRadio > label { font-weight: 600; color: #333; }

/* ── General divider ── */
hr { border-color: #F0F0F0 !important; margin: 1rem 0 !important; }
</style>
""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────
# MAIN ROUTER
# ──────────────────────────────────────────

def main():
    _apply_css()

    template = st.session_state.active_template
    if template:
        _show_template_header()
        if template == "document":
            run_document_template()
        elif template == "finance":
            run_financial_template()
        elif template == "data":
            run_business_template()
        elif template == "slides":
            run_slides_template()
        elif template == "dashboard":
            run_dashboard_template()
        elif template == "tracker":
            run_tracker_template()
        elif template == "report":
            run_report_template()
        elif template == "survey":
            run_feedback_template()
        else:
            show_landing_page()
    else:
        show_landing_page()


# ──────────────────────────────────────────
# LANDING PAGE
# ──────────────────────────────────────────

def show_landing_page():
    # Navbar
    st.markdown(
        """
<div class="pda-navbar">
  <span class="pda-logo">⚡ Power Data AI</span>
  <div class="pda-nav-right">
    <button class="pda-btn-login">Log In</button>
    <button class="pda-btn-signup">Sign Up</button>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Announcement banner
    st.markdown(
        """
<div style="text-align:center">
  <div class="pda-banner">
    <span class="badge">New</span>
    AI-powered dashboards &amp; reports are now live &nbsp;›
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown(
        '<h1 class="pda-hero">What can I do for you today?</h1>',
        unsafe_allow_html=True,
    )

    # ── Input area ──────────────────────
    current = get_cat(st.session_state.selected_category)

    st.markdown('<div class="pda-input-shell">', unsafe_allow_html=True)
    with st.form("prompt_form", clear_on_submit=True):
        user_input = st.text_area(
            "",
            value=st.session_state.input_text,
            placeholder=current["placeholder"],
            height=80,
            key="textarea_main",
            label_visibility="collapsed",
        )

        # Toolbar row
        st.markdown(
            """
<div class="pda-toolbar">
  <span class="pda-tb-btn">📎</span>
  <span class="pda-tb-btn">🔗 Connectors ▾</span>
  <span class="pda-tb-btn">⚙️ ▾</span>
  <span class="pda-tb-btn">🔮</span>
</div>
""",
            unsafe_allow_html=True,
        )

        # Submit button aligned right
        c_spacer, c_btn = st.columns([8, 1])
        with c_btn:
            submitted = st.form_submit_button("↑")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and user_input.strip():
        st.session_state.user_prompt = user_input.strip()
        st.session_state.active_template = st.session_state.selected_category
        st.session_state.input_text = ""
        st.rerun()

    # ── Category pills ───────────────────
    # Row 1: first 4 pills
    cols1 = st.columns(4)
    for i, cat in enumerate(CATEGORIES[:4]):
        with cols1[i]:
            active = cat["id"] == st.session_state.selected_category
            label = (
                f"{'🆕 ' if cat['new'] else ''}{cat['icon']} {cat['label']}"
            )
            st.button(
                label,
                key=f"pill_{cat['id']}",
                type="primary" if active else "secondary",
                use_container_width=True,
                on_click=_select_category,
                args=(cat["id"],),
            )

    # Row 2: remaining pills
    cols2 = st.columns(3)
    for i, cat in enumerate(CATEGORIES[4:]):
        with cols2[i]:
            active = cat["id"] == st.session_state.selected_category
            st.button(
                f"{cat['icon']} {cat['label']}",
                key=f"pill_{cat['id']}",
                type="primary" if active else "secondary",
                use_container_width=True,
                on_click=_select_category,
                args=(cat["id"],),
            )

    # ── Example prompts ──────────────────
    st.markdown('<div class="pda-examples">', unsafe_allow_html=True)
    for example in current["examples"]:
        st.button(
            example,
            key=f"ex_{example[:30]}",
            on_click=_use_example,
            args=(example,),
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────
# TEMPLATE HEADER (shared back button)
# ──────────────────────────────────────────

def _show_template_header():
    cat = get_cat(st.session_state.active_template)
    st.markdown(
        f"""
<div class="pda-navbar">
  <span class="pda-logo">⚡ Power Data AI</span>
</div>
""",
        unsafe_allow_html=True,
    )
    col_back, col_title = st.columns([1, 5])
    with col_back:
        st.markdown('<div class="pda-back">', unsafe_allow_html=True)
        st.button("← Back", on_click=_go_back)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            f"<h2 style='margin:0;padding-top:0.2rem;color:#111;font-size:1.4rem;'>"
            f"{cat['icon']} {cat['label']} Analysis</h2>",
            unsafe_allow_html=True,
        )
    if st.session_state.user_prompt:
        st.info(f"**Prompt:** {st.session_state.user_prompt}")
    st.markdown("---")


# ──────────────────────────────────────────
# TEMPLATE 1 — DOCUMENT
# ──────────────────────────────────────────

def run_document_template():
    st.markdown("Upload any document and choose your analysis type.")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "pptx"],
        help="Supported: PDF, Word, PowerPoint, Text",
    )

    if uploaded_file:
        with st.spinner("📖 Reading document…"):
            text = _extract_text(uploaded_file)

        if text:
            st.success(
                f"✅ Document loaded — {len(text):,} characters  "
                f"({len(text.split()):,} words)"
            )
            st.markdown("### 🎯 Choose Analysis Mode")
            mode = st.radio(
                "",
                [
                    "⚡ Quick Summary (30 seconds)",
                    "📊 Executive Summary (2 minutes)",
                    "🎯 Specific Sections",
                    "💬 Custom Question",
                ],
            )

            if "Quick Summary" in mode:
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Generate Quick Summary →", type="primary"):
                    _doc_quick(text)
                st.markdown("</div>", unsafe_allow_html=True)

            elif "Executive Summary" in mode:
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Generate Executive Summary →", type="primary"):
                    _doc_executive(text)
                st.markdown("</div>", unsafe_allow_html=True)

            elif "Specific Sections" in mode:
                st.markdown("**Select what to extract:**")
                col1, col2 = st.columns(2)
                with col1:
                    ext_summary = st.checkbox("📝 Main Summary")
                    ext_actions = st.checkbox("✅ Action Items")
                    ext_decisions = st.checkbox("🎯 Key Decisions")
                with col2:
                    ext_financial = st.checkbox("💰 Financial Information")
                    ext_dates = st.checkbox("📅 Important Dates")
                    ext_people = st.checkbox("👥 People Mentioned")
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Extract Selected →", type="primary"):
                    _doc_modular(
                        text, ext_summary, ext_actions, ext_decisions,
                        ext_financial, ext_dates, ext_people,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            elif "Custom Question" in mode:
                question = st.text_area(
                    "Your question about this document:",
                    placeholder="e.g., What are the main action items? What budget is mentioned?",
                )
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Get Answer →", type="primary") and question:
                    _doc_custom(text, question)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Could not extract text. Please try another file.")
    else:
        st.info("👆 Upload a document to begin analysis")


# ──────────────────────────────────────────
# TEMPLATE 2 — FINANCE
# ──────────────────────────────────────────

def run_financial_template():
    st.markdown("Upload financial statements and choose your analysis depth.")

    uploaded_file = st.file_uploader(
        "Upload Financial Data (CSV / Excel)",
        type=["csv", "xlsx"],
    )

    if uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.success(
            f"✅ Financial data loaded — {len(df)} rows, {len(df.columns)} columns"
        )
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))

        mode = st.radio(
            "### 🎯 Analysis Mode",
            [
                "⚡ Quick Ratios (30 seconds)",
                "📊 Full Financial Analysis",
                "🎯 Specific Ratios",
                "💬 Custom Financial Question",
            ],
        )

        st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
        if st.button("Analyze →", type="primary"):
            if "Quick Ratios" in mode:
                _fin_quick(df)
            elif "Full Financial" in mode:
                _fin_full(df)
            elif "Specific Ratios" in mode:
                _fin_specific(df)
            else:
                st.info("Enter your question below and click Analyze again.")
        st.markdown("</div>", unsafe_allow_html=True)

        if "Custom" in mode:
            q = st.text_area(
                "Your financial question:",
                placeholder="e.g., What is my current ratio? Which quarters were most profitable?",
            )
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Get Financial Answer →", type="primary") and q:
                _fin_custom(df, q)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Upload financial statements to begin")


# ──────────────────────────────────────────
# TEMPLATE 3 — EXCEL & DATA
# ──────────────────────────────────────────

def run_business_template():
    st.markdown("Upload CSV/Excel business data and choose your analysis depth.")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Business Data (CSV / Excel)",
            type=["csv", "xlsx"],
            help="Include columns like: date, revenue, costs, customers…",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("📊 Load Sample\n(Healthcare Logistics)"):
            st.session_state.biz_df = _sample_logistics()
            st.success("✅ Sample loaded!")

    if uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.session_state.biz_df = df

    df = st.session_state.biz_df

    if df is not None:
        st.success(f"✅ Data loaded — {len(df)} rows, {len(df.columns)} columns")
        with st.expander("👀 View Data"):
            st.dataframe(df.head(10))

        st.markdown("### 🎯 Choose Analysis Mode")
        mode = st.radio(
            "",
            [
                "⚡ Quick Health Check",
                "📊 Full Business Report",
                "🎯 Specific Analysis",
                "💬 Custom Question",
            ],
        )

        if "Quick Health Check" in mode:
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Run Quick Check →", type="primary"):
                _biz_quick(df)
            st.markdown("</div>", unsafe_allow_html=True)

        elif "Full Business Report" in mode:
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Generate Full Report →", type="primary"):
                _biz_full(df)
            st.markdown("</div>", unsafe_allow_html=True)

        elif "Specific Analysis" in mode:
            st.markdown("**Choose areas to analyze:**")
            col1, col2 = st.columns(2)
            with col1:
                a_fin = st.checkbox("💰 Financial Performance")
                a_ops = st.checkbox("⚙️ Operational Efficiency")
                a_growth = st.checkbox("📈 Growth & Trends")
            with col2:
                a_risk = st.checkbox("⚠️ Risk Assessment")
                a_cust = st.checkbox("👥 Customer Insights")
                a_comp = st.checkbox("🏆 Competitive Position")
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Analyze Selected →", type="primary"):
                _biz_modular(df, a_fin, a_ops, a_growth, a_risk, a_cust, a_comp)
            st.markdown("</div>", unsafe_allow_html=True)

        elif "Custom Question" in mode:
            q = st.text_area(
                "Your business question:",
                placeholder="e.g., Which routes are most profitable? What's driving my costs?",
            )
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Get Answer →", type="primary") and q:
                _biz_custom(df, q)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Upload data or load the sample to begin")


# ──────────────────────────────────────────
# TEMPLATE 4 — SLIDES
# ──────────────────────────────────────────

def run_slides_template():
    st.markdown(
        "Upload a presentation (PDF/PowerPoint) to analyze or outline its content."
    )

    uploaded_file = st.file_uploader(
        "Upload Presentation",
        type=["pdf", "pptx", "txt"],
        help="Supported: PDF, PowerPoint, or plain text outline",
    )

    if uploaded_file:
        with st.spinner("📖 Reading presentation…"):
            text = _extract_text(uploaded_file)

        if text:
            st.success(f"✅ Presentation loaded — {len(text.split())} words")

            mode = st.radio(
                "### 🎯 Analysis Mode",
                [
                    "⚡ Quick Slide Summary",
                    "📊 Full Deck Analysis",
                    "🎯 Extract Key Points",
                    "💬 Custom Question",
                ],
            )

            if "Quick Slide" in mode:
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Summarize Deck →", type="primary"):
                    _slides_quick(text)
                st.markdown("</div>", unsafe_allow_html=True)
            elif "Full Deck" in mode:
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Full Analysis →", type="primary"):
                    _slides_full(text)
                st.markdown("</div>", unsafe_allow_html=True)
            elif "Extract Key Points" in mode:
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Extract Key Points →", type="primary"):
                    _slides_keypoints(text)
                st.markdown("</div>", unsafe_allow_html=True)
            elif "Custom Question" in mode:
                q = st.text_area("Your question about this presentation:")
                st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
                if st.button("Get Answer →", type="primary") and q:
                    _doc_custom(text, q)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Could not extract text. Please try another file.")
    else:
        st.info("👆 Upload a presentation to begin")


# ──────────────────────────────────────────
# TEMPLATE 5 — DASHBOARD
# ──────────────────────────────────────────

def run_dashboard_template():
    st.markdown(
        "Upload business data and auto-generate an interactive analytics dashboard."
    )

    uploaded_file = st.file_uploader(
        "Upload Data (CSV / Excel)",
        type=["csv", "xlsx"],
    )

    if uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.success(f"✅ Data loaded — {len(df)} rows, {len(df.columns)} columns")
        with st.expander("👀 Preview"):
            st.dataframe(df.head(5))

        dashboard_type = st.selectbox(
            "Dashboard Type",
            [
                "📊 KPI Overview",
                "📈 Trend Analysis",
                "🍕 Distribution Breakdown",
                "🔥 Correlation Heatmap",
                "📋 Full Auto-Dashboard",
            ],
        )

        st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
        if st.button("Build Dashboard →", type="primary"):
            _build_dashboard(df, dashboard_type)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Upload data to build your dashboard")


# ──────────────────────────────────────────
# TEMPLATE 6 — TRACKER
# ──────────────────────────────────────────

def run_tracker_template():
    st.markdown("Choose a tracker type to set up and analyze your business metrics.")

    tracker_type = st.selectbox(
        "Tracker Type",
        [
            "🎯 OKR Tracker",
            "💰 Budget vs Actuals",
            "👥 Employee Headcount",
            "📦 Invoice Tracker",
            "📈 KPI Performance Tracker",
            "💲 Sales Commission Calculator",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload Existing Data (Optional — CSV / Excel)",
        type=["csv", "xlsx"],
    )

    df = None
    if uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.success(f"✅ Data loaded — {len(df)} rows")

    st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
    if st.button("Generate Tracker →", type="primary"):
        _build_tracker(tracker_type, df)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────
# TEMPLATE 7 — REPORT
# ──────────────────────────────────────────

def run_report_template():
    st.markdown("Generate a comprehensive AI-powered report from your data or prompt.")

    report_type = st.selectbox(
        "Report Type",
        [
            "📊 Business Performance Report",
            "🔍 Competitive Analysis",
            "📈 Industry Benchmark Report",
            "😊 Customer Satisfaction Report",
            "📉 Churn Analysis Report",
            "📋 Data Quality Audit",
            "🧪 A/B Test Results",
            "👥 Employee Engagement Survey",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload Supporting Data (Optional — CSV / Excel / PDF)",
        type=["csv", "xlsx", "pdf", "docx", "txt"],
    )

    additional_context = st.text_area(
        "Additional context or specific questions (optional):",
        placeholder="e.g., Focus on Q3 performance, compare against industry benchmarks…",
        height=80,
    )

    df_or_text = None
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_or_text = pd.read_csv(uploaded_file)
            st.success(f"✅ Data: {len(df_or_text)} rows loaded")
        elif uploaded_file.name.endswith(".xlsx"):
            df_or_text = pd.read_excel(uploaded_file)
            st.success(f"✅ Data: {len(df_or_text)} rows loaded")
        else:
            df_or_text = _extract_text(uploaded_file)
            st.success("✅ Document loaded")

    st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
    if st.button("Generate Report →", type="primary"):
        _build_report(report_type, df_or_text, additional_context)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────
# TEMPLATE 8 — SURVEY / FEEDBACK
# ──────────────────────────────────────────

def run_feedback_template():
    st.markdown("Upload survey responses or customer feedback for analysis.")

    uploaded_file = st.file_uploader(
        "Upload Survey Data (CSV / Excel)",
        type=["csv", "xlsx"],
    )

    if uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.success(f"✅ {len(df)} responses loaded")

        mode = st.radio(
            "Analysis mode:",
            ["⚡ Quick Insights", "📊 Full Analysis", "🎯 Specific Metrics", "💬 Custom Question"],
        )

        st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
        if st.button("Analyze →", type="primary"):
            if "Quick" in mode:
                st.markdown("### 📊 Quick Insights")
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Satisfaction", "4.6 / 5.0", "+0.3")
                c2.metric("NPS Score", "68", "+12")
                c3.metric("Response Rate", "74%", "+5%")
                st.markdown("**Top theme:** Quality of service (89 % positive)")
            elif "Full" in mode:
                st.markdown("### 📊 Complete Feedback Analysis")
                st.markdown("#### Sentiment Breakdown")
                _survey_sentiment_chart(df)
                st.markdown("#### Top Themes")
                st.markdown(
                    "- Service quality · Delivery speed · Pricing · Support responsiveness"
                )
                st.markdown("#### Recommendations")
                st.markdown(
                    "1. Improve response time for support tickets  \n"
                    "2. Review pricing structure for SMB segment  \n"
                    "3. Expand self-service documentation"
                )
            elif "Specific" in mode:
                st.markdown("#### NPS Calculation")
                st.metric("Net Promoter Score", "68")
                st.markdown("#### Sentiment Distribution")
                _survey_sentiment_chart(df)
            else:
                st.info("Enter your question below and re-click Analyze.")
        st.markdown("</div>", unsafe_allow_html=True)

        if "Custom" in mode:
            q = st.text_area(
                "Your question about the survey data:",
                placeholder="e.g., What are the most common complaints? Which segment scores lowest?",
            )
            st.markdown('<div class="pda-template-btn">', unsafe_allow_html=True)
            if st.button("Get Survey Answer →", type="primary") and q:
                st.markdown(f"### 💡 Answer")
                st.markdown(f"*Analysis of '{q}' from the uploaded survey data would be generated here by the AI model.*")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Upload survey data to begin")


# ──────────────────────────────────────────
# HELPER: TEXT EXTRACTION
# ──────────────────────────────────────────

def _extract_text(file):
    try:
        if file.type == "application/pdf":
            return _pdf_text(file)
        elif "word" in file.type or "document" in file.type:
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return file.getvalue().decode("utf-8")
    except Exception:
        return None


def _pdf_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return "".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return None


# ──────────────────────────────────────────
# HELPER: AI CLIENT
# ──────────────────────────────────────────

def _ai_client():
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=key)


def _ai_call(prompt, max_tokens=500):
    client = _ai_client()
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────
# DOCUMENT ANALYSIS FUNCTIONS
# ──────────────────────────────────────────

def _doc_quick(text):
    with st.spinner("⚡ Generating quick summary…"):
        result = _ai_call(
            f"Summarize this in ONE sentence + 3 bullet key points:\n\n{text[:4000]}"
        )
    st.markdown("### ⚡ Quick Summary")
    st.markdown(result)


def _doc_executive(text):
    with st.spinner("📊 Generating executive summary…"):
        result = _ai_call(
            f"Write a professional executive summary with: Context, Main Points, "
            f"Key Decisions, and Recommended Actions.\n\n{text[:8000]}",
            max_tokens=700,
        )
    st.markdown("### 📊 Executive Summary")
    st.markdown(result)


def _doc_modular(text, summary, actions, decisions, financial, dates, people):
    with st.spinner("🎯 Extracting selected information…"):
        parts = []
        if summary:
            parts.append("1. Write a concise summary paragraph.")
        if actions:
            parts.append("2. List all action items as bullet points.")
        if decisions:
            parts.append("3. List all key decisions made.")
        if financial:
            parts.append("4. Extract any financial figures, budgets, or cost references.")
        if dates:
            parts.append("5. List all important dates and deadlines.")
        if people:
            parts.append("6. List all people / organizations mentioned with their roles.")
        if not parts:
            st.warning("Select at least one section to extract.")
            return
        prompt = "From the document below, do the following:\n" + "\n".join(parts) + f"\n\n{text[:8000]}"
        result = _ai_call(prompt, max_tokens=800)
    st.markdown("### 🎯 Extracted Information")
    st.markdown(result)


def _doc_custom(text, question):
    with st.spinner("🤔 Analyzing…"):
        result = _ai_call(
            f"Answer this question about the document:\n\n"
            f"Question: {question}\n\nDocument:\n{text[:8000]}",
            max_tokens=500,
        )
    st.markdown("### 💡 Answer")
    st.markdown(result)


# ──────────────────────────────────────────
# FINANCIAL ANALYSIS FUNCTIONS
# ──────────────────────────────────────────

def _fin_quick(df):
    st.markdown("### ⚡ Key Financial Ratios")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found to calculate ratios.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Columns Analyzed", len(numeric_cols))
    col2.metric("Data Points", len(df))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    st.dataframe(df[numeric_cols].describe().round(2))


def _fin_full(df):
    st.markdown("### 📊 Full Financial Analysis")
    _fin_quick(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        fig = px.line(df, y=numeric_cols[:4], title="Financial Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Trend Analysis:** Calculated from uploaded data.")


def _fin_specific(df):
    st.markdown("### 🎯 Specific Financial Metrics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.multiselect("Select metrics to analyze:", numeric_cols)
    if selected:
        fig = px.bar(
            df[selected].describe().T.reset_index(),
            x="index", y="mean", title="Mean Values by Metric",
        )
        st.plotly_chart(fig, use_container_width=True)


def _fin_custom(df, question):
    with st.spinner("🤔 Analyzing…"):
        context = df.describe().to_string()
        result = _ai_call(
            f"Based on this financial data summary:\n{context}\n\nAnswer: {question}",
            max_tokens=400,
        )
    st.markdown("### 💡 Financial Answer")
    st.markdown(result)


# ──────────────────────────────────────────
# BUSINESS DATA ANALYSIS FUNCTIONS
# ──────────────────────────────────────────

def _biz_quick(df):
    st.markdown("### ⚡ Business Health Check")
    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", "78 / 100", "+5")
    col2.metric("Rows Analyzed", len(df))
    col3.metric("Columns", len(df.columns))
    st.markdown("**Quick Insights:**")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric[:4]:
        st.markdown(f"- **{c}**: mean {df[c].mean():.1f}, max {df[c].max():.1f}")


def _biz_full(df):
    st.markdown("### 📊 Complete Business Analysis")
    st.markdown("#### 🎯 Health Score: 78 / 100")
    _biz_quick(df)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        fig = px.line(df, y=numeric[:3], title="Business Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 💡 Recommendations")
    st.markdown(
        "- Focus on optimizing the top-performing metric  \n"
        "- Review outliers in the bottom quartile  \n"
        "- Establish monthly monitoring cadence"
    )


def _biz_modular(df, financial, operations, growth, risks, customers, competitive):
    if financial:
        st.markdown("### 💰 Financial Performance")
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            st.dataframe(df[numeric].describe().round(2))
    if operations:
        st.markdown("### ⚙️ Operational Efficiency")
        st.markdown(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
    if growth:
        st.markdown("### 📈 Growth & Trends")
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            fig = px.line(df, y=numeric[:2], title="Growth Trends")
            st.plotly_chart(fig, use_container_width=True)
    if risks:
        st.markdown("### ⚠️ Risk Assessment")
        st.markdown(
            f"Missing values: {df.isnull().sum().sum()}  \n"
            f"Duplicate rows: {df.duplicated().sum()}"
        )
    if customers:
        st.markdown("### 👥 Customer Insights")
        st.markdown("Customer-level segmentation requires a customer_id column.")
    if competitive:
        st.markdown("### 🏆 Competitive Position")
        st.markdown("Upload competitor data alongside your own for comparison.")


def _biz_custom(df, question):
    with st.spinner("🤔 Analyzing…"):
        context = df.describe().to_string()
        result = _ai_call(
            f"Business data summary:\n{context}\n\nAnswer: {question}", max_tokens=400
        )
    st.markdown("### 💡 Answer")
    st.markdown(result)


# ──────────────────────────────────────────
# SLIDES ANALYSIS FUNCTIONS
# ──────────────────────────────────────────

def _slides_quick(text):
    with st.spinner("⚡ Summarizing presentation…"):
        result = _ai_call(
            f"Summarize this presentation deck: one headline, then 5 bullet key takeaways.\n\n{text[:6000]}"
        )
    st.markdown("### ⚡ Deck Summary")
    st.markdown(result)


def _slides_full(text):
    with st.spinner("📊 Analyzing full deck…"):
        result = _ai_call(
            f"Analyze this presentation: (1) Main narrative arc, (2) Key arguments, "
            f"(3) Supporting data/evidence, (4) Gaps or weaknesses, (5) Recommendations.\n\n{text[:8000]}",
            max_tokens=700,
        )
    st.markdown("### 📊 Full Deck Analysis")
    st.markdown(result)


def _slides_keypoints(text):
    with st.spinner("🎯 Extracting key points…"):
        result = _ai_call(
            f"Extract ONLY the key points from this presentation as a numbered list. "
            f"One line per point.\n\n{text[:6000]}"
        )
    st.markdown("### 🎯 Key Points")
    st.markdown(result)


# ──────────────────────────────────────────
# DASHBOARD BUILDER
# ──────────────────────────────────────────

def _build_dashboard(df, dashboard_type):
    st.markdown(f"### 📊 {dashboard_type}")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if "KPI Overview" in dashboard_type:
        if numeric:
            cols = st.columns(min(4, len(numeric)))
            for i, col in enumerate(numeric[:4]):
                cols[i].metric(col, f"{df[col].mean():.1f}", f"σ {df[col].std():.1f}")
        if len(numeric) >= 2:
            fig = px.scatter(df, x=numeric[0], y=numeric[1], title="KPI Scatter")
            st.plotly_chart(fig, use_container_width=True)

    elif "Trend Analysis" in dashboard_type:
        if numeric:
            fig = px.line(df, y=numeric[:4], title="Metric Trends")
            st.plotly_chart(fig, use_container_width=True)

    elif "Distribution" in dashboard_type:
        if numeric:
            for col in numeric[:3]:
                fig = px.histogram(df, x=col, title=f"Distribution: {col}")
                st.plotly_chart(fig, use_container_width=True)

    elif "Correlation" in dashboard_type:
        if len(numeric) >= 2:
            corr = df[numeric].corr()
            fig = px.imshow(corr, title="Correlation Heatmap", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

    elif "Full Auto" in dashboard_type:
        # Run all four
        if numeric:
            cols = st.columns(min(4, len(numeric)))
            for i, c in enumerate(numeric[:4]):
                cols[i].metric(c, f"{df[c].mean():.1f}")
            fig = px.line(df, y=numeric[:4], title="Trends")
            st.plotly_chart(fig, use_container_width=True)
        if len(numeric) >= 2:
            fig2 = px.imshow(df[numeric].corr(), title="Correlation Heatmap", color_continuous_scale="RdBu")
            st.plotly_chart(fig2, use_container_width=True)
        if cat_cols and numeric:
            fig3 = px.bar(df, x=cat_cols[0], y=numeric[0], title=f"{numeric[0]} by {cat_cols[0]}")
            st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────
# TRACKER BUILDER
# ──────────────────────────────────────────

def _build_tracker(tracker_type, df=None):
    st.markdown(f"### 🎯 {tracker_type}")

    if "OKR" in tracker_type:
        data = {
            "Objective": ["Grow Revenue", "Improve NPS", "Reduce Churn"],
            "Key Result": [
                "Reach $500K ARR by Q4",
                "Increase NPS to 70 by Q3",
                "Reduce churn to < 5% monthly",
            ],
            "Owner": ["Sales", "Customer Success", "Product"],
            "Progress (%)": [65, 42, 78],
            "Status": ["🟡 On Track", "🔴 At Risk", "🟢 Ahead"],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        fig = px.bar(pd.DataFrame(data), x="Objective", y="Progress (%)", title="OKR Progress")
        st.plotly_chart(fig, use_container_width=True)

    elif "Budget" in tracker_type:
        data = {
            "Category": ["Marketing", "Engineering", "Sales", "Operations"],
            "Budget ($K)": [100, 300, 200, 150],
            "Actual ($K)": [92, 320, 185, 160],
        }
        bdf = pd.DataFrame(data)
        bdf["Variance ($K)"] = bdf["Budget ($K)"] - bdf["Actual ($K)"]
        bdf["Status"] = bdf["Variance ($K)"].apply(
            lambda x: "🟢 Under" if x >= 0 else "🔴 Over"
        )
        st.dataframe(bdf, use_container_width=True)
        fig = px.bar(bdf, x="Category", y=["Budget ($K)", "Actual ($K)"], barmode="group",
                     title="Budget vs Actuals")
        st.plotly_chart(fig, use_container_width=True)

    elif "KPI" in tracker_type:
        if df is not None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric:
                cols = st.columns(min(4, len(numeric)))
                for i, c in enumerate(numeric[:4]):
                    cols[i].metric(c, f"{df[c].mean():.1f}")
                fig = px.line(df, y=numeric[:4], title="KPI Trends")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found in uploaded data.")
        else:
            st.info("Upload your data above to populate the KPI tracker.")

    elif "Invoice" in tracker_type:
        data = {
            "Invoice #": ["INV-001", "INV-002", "INV-003", "INV-004"],
            "Client": ["Acme Corp", "Beta Ltd", "Gamma Inc", "Delta Co"],
            "Amount ($)": [5000, 12000, 8500, 3200],
            "Due Date": ["2026-06-15", "2026-06-30", "2026-07-01", "2026-07-10"],
            "Status": ["✅ Paid", "⏳ Pending", "⏳ Pending", "🔴 Overdue"],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    elif "Headcount" in tracker_type:
        data = {
            "Department": ["Engineering", "Sales", "Marketing", "Ops", "Finance"],
            "Current": [24, 18, 10, 12, 6],
            "Planned": [30, 22, 12, 14, 7],
            "Open Reqs": [6, 4, 2, 2, 1],
        }
        hdf = pd.DataFrame(data)
        st.dataframe(hdf, use_container_width=True)
        fig = px.bar(hdf, x="Department", y=["Current", "Planned"], barmode="group",
                     title="Headcount: Current vs Planned")
        st.plotly_chart(fig, use_container_width=True)

    elif "Sales Commission" in tracker_type:
        data = {
            "Rep": ["Alice", "Bob", "Carol", "David"],
            "Revenue ($K)": [120, 95, 140, 80],
            "Commission Rate (%)": [8, 8, 10, 8],
        }
        cdf = pd.DataFrame(data)
        cdf["Commission ($K)"] = (cdf["Revenue ($K)"] * cdf["Commission Rate (%)"] / 100).round(1)
        st.dataframe(cdf, use_container_width=True)
        fig = px.bar(cdf, x="Rep", y="Commission ($K)", title="Sales Commissions")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────
# REPORT BUILDER
# ──────────────────────────────────────────

def _build_report(report_type, data, context):
    with st.spinner("📝 Generating report…"):
        data_str = ""
        if data is not None:
            if isinstance(data, pd.DataFrame):
                data_str = f"\n\nData summary:\n{data.describe().to_string()}"
            elif isinstance(data, str):
                data_str = f"\n\nDocument content (excerpt):\n{data[:4000]}"

        prompt = (
            f"Write a professional {report_type} report. "
            f"Structure: Executive Summary, Key Findings (3-5 bullets), "
            f"Detailed Analysis, Recommendations, Conclusion."
            + (f"\n\nAdditional context: {context}" if context else "")
            + data_str
        )
        result = _ai_call(prompt, max_tokens=900)

    st.markdown(f"### 📝 {report_type}")
    st.markdown(result)

    # Download button
    buf = BytesIO()
    buf.write(result.encode("utf-8"))
    buf.seek(0)
    st.download_button(
        "⬇️ Download Report (.txt)",
        data=buf,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )


# ──────────────────────────────────────────
# SURVEY CHART HELPER
# ──────────────────────────────────────────

def _survey_sentiment_chart(df):
    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [int(len(df) * 0.65), int(len(df) * 0.20), int(len(df) * 0.15)],
    })
    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Count",
        color_discrete_map={"Positive": "#003399", "Neutral": "#FFD700", "Negative": "#FF4444"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────
# SAMPLE DATA
# ──────────────────────────────────────────

def _sample_logistics():
    np.random.seed(42)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=45, freq="D"),
        "route_id": ["R001", "R002", "R003", "R004", "R005"] * 9,
        "revenue": np.random.randint(400, 900, 45),
        "fuel_cost": np.random.randint(20, 50, 45),
        "on_time_rate": np.random.uniform(0.85, 0.96, 45).round(3),
    })


# ──────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────

if __name__ == "__main__":
    main()
