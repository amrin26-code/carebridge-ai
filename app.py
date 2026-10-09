import copy
import io
import json
import re
from pathlib import Path
from datetime import date, datetime, time, timedelta
from xml.sax.saxutils import escape

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# =========================================================
# CARETRAIL — single-file Streamlit health dashboard
# =========================================================
st.set_page_config(
    page_title="CareTrail | Health Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULTS = {
    "theme": "Light",
    "visits": [],
    "notes": [],
    "vitals": [],
    "symptoms": [],
    "medications": [],
    "documents": [],
    "classifier_result": None,
    "visit_comparison": None,
    "active_profile": "Personal Records",
    "ai_demo_mode": True,
    "ai_history": [],
    "generated_pdf": None,
    "page": "Overview",
    "quick_nav": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)

# ------------------------- Theme -------------------------
THEMES = {
    "Light": {
        "bg": "#F3F7FA",
        "panel": "#FFFFFF",
        "panel_alt": "#E8F1F5",
        "text": "#153047",
        "muted": "#506779",
        "border": "#CFDDE5",
        "accent": "#087F8C",
        "accent_hover": "#066773",
        "accent_text": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "input_text": "#153047",
        "sidebar": "#E7F0F5",
        "sidebar_text": "#153047",
        "plot": "plotly_white",
    },
    "Dark": {
        "bg": "#101A2B",
        "panel": "#19283D",
        "panel_alt": "#22364D",
        "text": "#F2F7FA",
        "muted": "#C1D0DC",
        "border": "#3A5067",
        "accent": "#38B8BE",
        "accent_hover": "#5BD0D2",
        "accent_text": "#08232B",
        "input_bg": "#203249",
        "input_text": "#F2F7FA",
        "sidebar": "#0B1423",
        "sidebar_text": "#F2F7FA",
        "plot": "plotly_dark",
    },
}
C = THEMES.get(st.session_state.theme, THEMES["Light"])
is_dark = st.session_state.theme == "Dark"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {"dark" if is_dark else "light"}; }}
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
        background-color: {C["bg"]} !important;
        color: {C["text"]} !important;
    }}
    [data-testid="stHeader"] {{ background: {C["bg"]} !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {C["sidebar"]} !important;
        color: {C["sidebar_text"]} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {C["sidebar_text"]} !important; }}
    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp h6, .stApp strong, .stMarkdown, .stMarkdown p,
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        color: {C["text"]} !important;
    }}
    [data-testid="stCaptionContainer"] p, .stCaption, small {{
        color: {C["muted"]} !important;
    }}
    [data-testid="stMetric"] {{
        background: {C["panel"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        box-shadow: 0 2px 8px rgba(20, 50, 70, 0.04);
    }}
    [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {C["border"]} !important;
        border-radius: 12px !important;
    }}
    .stApp input, .stApp textarea,
    .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="textarea"] textarea {{
        background: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
        -webkit-text-fill-color: {C["input_text"]} !important;
        caret-color: {C["input_text"]} !important;
        border-color: {C["border"]} !important;
    }}
    .stApp input::placeholder, .stApp textarea::placeholder {{
        color: {C["muted"]} !important;
        -webkit-text-fill-color: {C["muted"]} !important;
        opacity: 1 !important;
    }}
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="popover"], .stApp [data-baseweb="menu"] {{
        background: {C["input_bg"]} !important;
        border-color: {C["border"]} !important;
        color: {C["input_text"]} !important;
    }}
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] input,
    [data-baseweb="popover"] *, [data-baseweb="menu"] * {{
        color: {C["input_text"]} !important;
    }}
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
        background: {C["panel"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 9px !important;
        transition: border-color .15s ease, background .15s ease;
    }}
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {{
        border-color: {C["accent"]} !important;
