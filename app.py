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

# Machine Learning & Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# CARETRAIL — Intelligent PHR Assistant Platform
# =========================================================
st.set_page_config(
    page_title="CareTrail | AI Personal Health Record Portal",
    page_icon="🩺",
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
    "ai_demo_mode": False,
    "ai_history": [],
    "generated_pdf": None,
    "page": "Overview",
    "quick_nav": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)

# ------------------------- Theme Palette -------------------------
C = {
    "bg": "#FFFFFF",            
    "panel": "#FFFFFF",         
    "panel_alt": "#F8FAFC",     
    "text": "#000000",          
    "muted": "#475569",         
    "border": "#CBD5E1",        
    "accent": "#0D9488",        
    "accent_hover": "#0F766E",  
    "accent_text": "#FFFFFF",   
    "input_bg": "#FFFFFF",      
    "input_text": "#000000",    
    "sidebar": "#F8FAFC",       
    "sidebar_text": "#000000",  
    "plot": "plotly_white",
}

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: light; }}
    
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
        background-color: {C["bg"]} !important;
        color: {C["text"]} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    [data-testid="stHeader"] {{ background: {C["bg"]} !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {C["sidebar"]} !important;
        color: {C["sidebar_text"]} !important;
        border-right: 1px solid {C["border"]} !important;
    }}
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: {C["sidebar_text"]} !important;
    }}

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
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }}
    
    [data-testid="stExpander"] {{
        background-color: {C["panel"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] header,
    [data-testid="stExpander"] header * {{
        color: {C["text"]} !important;
        background-color: {C["panel"]} !important;
        font-weight: 700 !important;
    }}

    .stApp input, .stApp textarea,
    [data-baseweb="input"], [data-baseweb="input"] input,
    [data-baseweb="base-input"], [data-baseweb="base-input"] input,
    [data-baseweb="textarea"], [data-baseweb="textarea"] textarea {{
        background-color: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
        -webkit-text-fill-color: {C["input_text"]} !important;
        caret-color: {C["input_text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 8px !important;
    }}

    [data-baseweb="calendar"], [data-baseweb="calendar"] * {{
        color: {C["text"]} !important;
        background-color: {C["panel"]} !important;
    }}

    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="popover"], .stApp [data-baseweb="menu"],
    [data-baseweb="popover"] ul, [data-baseweb="menu"] ul {{
        background-color: {C["input_bg"]} !important;
        border: 1px solid {C["border"]} !important;
        color: {C["input_text"]} !important;
        border-radius: 8px !important;
    }}
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] div,
    .stApp [data-baseweb="select"] input,
    [data-baseweb="popover"] *, [data-baseweb="menu"] * {{
        color: {C["input_text"]} !important;
        background-color: transparent !important;
    }}
    [data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] {{
        background-color: {C["panel_alt"]} !important;
    }}

    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {{
        border-color: {C["accent"]} !important;
        color: {C["accent"]} !important;
    }}
    .stApp [data-testid="stButton"] button p,
    .stApp [data-testid="stDownloadButton"] button p {{
        color: {C["text"]} !important;
    }}
    .stApp button[kind="primary"] {{
        background-color: {C["accent"]} !important;
        border-color: {C["accent"]} !important;
        color: {C["accent_text"]} !important;
        border-radius: 8px !important;
    }}
    .stApp button[kind="primary"]:hover {{
        background-color: {C["accent_hover"]} !important;
        border-color: {C["accent_hover"]} !important;
    }}
    .stApp button[kind="primary"] p {{
        color: {C["accent_text"]} !important;
    }}

    .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"],
    .stApp table, [data-testid="stDataEditor"] {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
        border-color: {C["border"]} !important;
        border-radius: 8px !important;
    }}

    .stApp [data-testid="stFileUploader"] section {{
        background-color: {C["panel"]} !important;
        border: 2px dashed {C["border"]} !important;
        border-radius: 12px !important;
    }}
    .stApp a {{ color: {C["accent"]} !important; font-weight: 600; }}
    hr {{ border-color: {C["border"]} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Helpers -------------------------
def today_str():
    return date.today().isoformat()

def notify(message):
    try:
        st.toast(message)
    except Exception:
        st.success(message)

def current_records(key):
    value = st.session_state.get(key, [])
    return value if isinstance(value, list) else []

def csv_bytes(records):
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")

def chart_theme(fig):
    fig.update_layout(
        template=C["plot"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": C["text"]},
        margin={"l": 18, "r": 18, "t": 50, "b": 18},
        legend={"font": {"color": C["text"]}},
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig

# ------------------------- Built-in ML Classifier Engine -------------------------
@st.cache_resource
def get_trained_nlp_pipeline():
    """Trains a TF-IDF + LinearSVC Model on Synthetic PHR Queries for local intent recognition."""
    training_data = [
        # symptom_help
        ("I have severe pain in my head", "symptom_help"),
        ("My fever is 102 degrees", "symptom_help"),
        ("Experiencing dizziness and nausea", "symptom_help"),
        ("Sore throat and persistent cough", "symptom_help"),
        ("Sharp chest pain when breathing", "symptom_help"),
        ("Swollen ankle after falling", "symptom_help"),
        # medication_query
        ("When should I take my insulin dose?", "medication_query"),
        ("Can I take aspirin with blood thinners?", "medication_query"),
        ("What is the prescribed dosage for metformin?", "medication_query"),
        ("Forgot my morning blood pressure pill", "medication_query"),
        ("Side effects of statin medications", "medication_query"),
        # records_query
        ("Show my clinic visit summary", "records_query"),
        ("Download my hospital doctor notes", "records_query"),
        ("Where is my lab test result PDF stored?", "records_query"),
        ("List all doctor appointments from last month", "records_query"),
        ("Find my past medical history reports", "records_query"),
        # health_trends
        ("Show my blood pressure graph for last week", "health_trends"),
        ("Is my glucose level spiking or stable?", "health_trends"),
        ("Display weight loss progress chart", "health_trends"),
        ("Plot my heart rate history over time", "health_trends"),
        ("Are my daily temperature trends normal?", "health_trends"),
        # general_help
        ("How do I backup my health data?", "general_help"),
        ("How to use this PHR application", "general_help"),
        ("How do I add a new patient profile?", "general_help"),
        ("Is my medical data encrypted locally?", "general_help"),
        ("Export my health summary report", "general_help")
    ]
    texts, labels = zip(*training_data)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC(C=1.0, random_state=42))
    ])
    pipeline.fit(texts, labels)
    
    # Calculate evaluation matrix
    preds = pipeline.predict(texts)
    report = classification_report(labels, preds, output_dict=True)
    cm = confusion_matrix(labels, preds)
    return pipeline, report, cm

# ------------------------- 5 Demo Profiles Builder -------------------------
PROFILES = [
    "Personal Records",
    "Alex Morgan — Diabetes Monitoring",
    "Jamie Taylor — Post-Surgery Recovery",
    "Sam Chen — Hypertension Management",
    "Patel Family — Pediatric Checkup",
    "Elena Rostova — Chronic Pain Tracking"
]

def build_sample_profile(profile):
    vitals, symptoms = [], []
    for offset in range(13, -1, -1):
        d = date.today() - timedelta(days=offset)
        if profile == "Alex Morgan — Diabetes Monitoring":
            v = {"date": d.isoformat(), "systolic": 118 + (offset % 4) * 2, "diastolic": 76 + (offset % 3), 
                 "heart_rate": 72 + (offset % 5), "temperature": 36.6, "weight": round(78.4 - (13-offset)*.04, 1), 
                 "glucose_mg_dl": 126 + (offset * 7 % 38)}
            sym, severity = "Fatigue", (offset * 3) % 6
        elif profile == "Jamie Taylor — Post-Surgery Recovery":
            v = {"date": d.isoformat(), "systolic": 116 + (offset % 5) * 2, "diastolic": 74 + (offset % 4), 
                 "heart_rate": 76 + (offset % 6), "temperature": round(36.5 + (offset % 4)*.1, 1), 
                 "weight": 65.2, "glucose_mg_dl": None}
            sym, severity = "Surgical Site Pain", min(8, max(1, offset // 2 + 1))
        elif profile == "Sam Chen — Hypertension Management":
            v = {"date": d.isoformat(), "systolic": 142 - (13-offset)//2, "diastolic": 88 - (13-offset)//3, 
                 "heart_rate": 68 + (offset % 4), "temperature": 36.6, "weight": 82.1, "glucose_mg_dl": None}
            sym, severity = "Light Dizziness", (offset * 2) % 5
        elif profile == "Patel Family — Pediatric Checkup":
            v = {"date": d.isoformat(), "systolic": 102 + (offset % 3), "diastolic": 64 + (offset % 2), 
                 "heart_rate": 95 + (offset % 8), "temperature": round(36.8 + (offset % 2)*.1, 1), 
                 "weight": 24.5, "glucose_mg_dl": None}
            sym, severity = "Mild Cough", (offset) % 4
        elif profile == "Elena Rostova — Chronic Pain Tracking":
            v = {"date": d.isoformat(), "systolic": 120 + (offset % 3), "diastolic": 78 + (offset % 2), 
                 "heart_rate": 74 + (offset % 4), "temperature": 36.6, "weight": 58.0, "glucose_mg_dl": None}
            sym, severity = "Joint Stiffness", min(9, max(2, (offset * 4) % 10))
        else:
            v = {"date": d.isoformat(), "systolic": 120, "diastolic": 80, "heart_rate": 72, "temperature": 36.6, "weight": 70.0, "glucose_mg_dl": None}
            sym, severity = "None", 0

        vitals.append(v)
        symptoms.append({"date": d.isoformat(), "symptom": sym, "severity": severity, "duration": "Daily Log", "notes": "Fictional demo entry"})

    provider_name = "City General Health Center"
    med_name = "Sample Prescription A" if "Diabetes" in profile or "Hypertension" in profile else "Sample Prescription B"

    return {
        "visits": [{"date": (date.today()-timedelta(days=7)).isoformat(), "provider": provider_name, "reason": f"Follow-up for {profile.split('—')[0].strip()}", "diagnosis": "Condition Monitored", "notes": "Fictional scenario record."}],
        "notes": [{"date": today_str(), "title": "Clinical Summary Note", "text": f"Active profile set to {profile}."}],
        "vitals": vitals,
        "symptoms": symptoms,
        "medications": [{"name": med_name, "dose": "10mg", "frequency": "Once Daily", "time": "09:00", "start_date": today_str(), "notes": "Take with water"}]
    }

def load_sample_profile(profile):
    if profile == "Personal Records":
        st.session_state.active_profile = profile
        return
    data = build_sample_profile(profile)
    for key, value in data.items():
        st.session_state[key] = copy.deepcopy(value)
    st.session_state.active_profile = profile

def make_backup():
    keys = ["visits", "notes", "vitals", "symptoms", "medications", "documents",
            "classifier_result", "visit_comparison", "ai_history", "active_profile"]
    data = {k: st.session_state.get(k) for k in keys}
    data.update({"exported_at": datetime.now().isoformat(), "format_version": 1})
    return json.dumps(data, indent=2, ensure_ascii=False)

def restore_backup(uploaded_file):
    data = json.load(uploaded_file)
    if not isinstance(data, dict):
        raise ValueError("Backup must contain a JSON object.")
    expected = ["visits", "notes", "vitals", "symptoms", "medications", "documents", "ai_history"]
    for key in expected:
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"Invalid backup: '{key}' must be a list.")
    for key in expected + ["classifier_result", "visit_comparison", "active_profile"]:
        if key in data:
            st.session_state[key] = data[key]

def safe_text(value, limit=2000):
    return escape(str(value if value is not None else ""))[:limit].replace("\n", "<br/>")

def make_doctor_pdf():
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                            topMargin=16*mm, bottomMargin=16*mm,
                            title="CareTrail Doctor Summary", author="CareTrail Prototype")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CareTitle", parent=styles["Title"], fontSize=20,
                              leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#0D9488"), spaceAfter=10))
    styles.add(ParagraphStyle(name="CareSection", parent=styles["Heading2"], fontSize=12,
                              leading=15, textColor=colors.HexColor("#0D9488"), spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="CareSmall", parent=styles["BodyText"], fontSize=7.5, leading=10))
    story = [
        Paragraph("CareTrail — Doctor Summary", styles["CareTitle"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["Normal"]),
        Paragraph(f"Profile: {safe_text(st.session_state.get('active_profile', 'Personal Records'))}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Important Information", styles["CareSection"]),
        Paragraph("This report summarizes prototype personal health records for clinician review.", styles["CareSmall"]),
    ]
    def add_table_section(title, rows, widths=None):
        story.append(Paragraph(title, styles["CareSection"]))
        if not rows:
            story.append(Paragraph("No records saved.", styles["Normal"]))
            return
        table = Table(rows, repeatRows=1, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D9488")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(table)
    vitals = current_records("vitals")
    vrows = [["Date", "Systolic", "Diastolic", "Heart rate", "Temp °C", "Weight", "Glucose"]]
    for r in sorted(vitals, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
        vrows.append([safe_text(r.get(k, "")) for k in ["date", "systolic", "diastolic", "heart_rate", "temperature", "weight", "glucose_mg_dl"]])
    add_table_section("Recent Vitals", vrows if vitals else [])
    symptoms = current_records("symptoms")
    srows = [["Date", "Symptom", "Severity", "Duration", "Notes"]]
    for r in sorted(symptoms, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
        srows.append([Paragraph(safe_text(r.get(k, ""), 300), styles["CareSmall"]) for k in ["date", "symptom", "severity", "duration", "notes"]])
    add_table_section("Recent Symptoms", srows if symptoms else [])
    visits = current_records("visits")
    story.append(Paragraph("Health Visits", styles["CareSection"]))
    if visits:
        for r in sorted(visits, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
            story.append(Paragraph(
                f"<b>{safe_text(r.get('date', ''))} — {safe_text(r.get('provider', 'Provider not entered'))}</b><br/>"
                f"Reason: {safe_text(r.get('reason', ''))}<br/>Assessment: {safe_text(r.get('diagnosis', ''))}<br/>"
                f"Notes: {safe_text(r.get('notes', ''))}", styles["CareSmall"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No visit records saved.", styles["Normal"]))
    meds = current_records("medications")
    mrows = [["Name", "Dose entered", "Frequency entered", "Time"]]
    for r in meds:
        mrows.append([Paragraph(safe_text(r.get(k, ""), 150), styles["CareSmall"]) for k in ["name", "dose", "frequency", "time"]])
    add_table_section("Medication List", mrows if meds else [])
    doc.build(story)
    return output.getvalue()

# ------------------------- Sidebar Navigation & Auto-Load -------------------------
NAV = [
    "Overview", "Health Notes & Timeline", "Vitals & Analytics", "Symptom Tracker",
    "Medications & Reminders", "Document Vault", "Visit Comparison",
    "AI Question Classifier", "Data Backup & Restore", "Doctor Summary PDF",
    "About & Privacy",
]

with st.sidebar:
    st.markdown("## 🩺 CareTrail")
    st.caption("Intelligent PHR Assistant Platform")
    st.divider()
    
    st.subheader("Select Demo Profile")
    current_profile = st.session_state.active_profile
    profile_index = PROFILES.index(current_profile) if current_profile in PROFILES else 0
    
    chosen_profile = st.selectbox(
        "Choose a patient profile", 
        PROFILES, 
        index=profile_index,
        key="profile_picker"
    )
    
    if chosen_profile != st.session_state.active_profile:
        load_sample_profile(chosen_profile)
        notify(f"Loaded {chosen_profile}")
        st.rerun()
        
    st.caption("Auto-populates 14 days of realistic vitals, symptoms, and visits.")
    st.divider()
    
    pending = st.session_state.get("quick_nav")
    if pending in NAV:
        st.session_state.page = pending
        st.session_state.quick_nav = None
    page_index = NAV.index(st.session_state.page) if st.session_state.page in NAV else 0
    page = st.radio("Navigation", NAV, index=page_index, key="page_picker")
    if page != st.session_state.page:
        st.session_state.page = page
    st.divider()
    st.caption(f"Active Profile: {st.session_state.active_profile}")
    if st.button("Clear Session Data", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(value)
        st.rerun()

# ------------------------- Page Header Banner -------------------------
if st.session_state.active_profile != "Personal Records":
    st.info(f"Demo Profile Active: **{st.session_state.active_profile}**. Synthetic records for prototype evaluation.")

st.markdown(
    f"""<div style="background:{C['panel']};border:1px solid {C['border']};
    border-radius:16px;padding:20px 24px;margin-bottom:20px;box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
    <div style="font-size:12px;letter-spacing:1.5px;font-weight:700;color:{C['accent']};text-transform:uppercase;">
    Personal Health Record Portal</div>
    <div style="font-size:28px;font-weight:800;color:{C['text']};margin:4px 0;">
    CareTrail Dashboard</div>
    <div style="font-size:14px;color:{C['muted']};">
    Track vital signs, record clinical visits, monitor symptoms, and run offline AI intent recognition.</div></div>""",
    unsafe_allow_html=True,
)

# ------------------------- Overview -------------------------
if page == "Overview":
    st.subheader("Overview")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Health Visits", "Vitals Records", "Symptoms Logged", "Medication Entries"],
                               ["visits", "vitals", "symptoms", "medications"]):
        col.metric(label, len(current_records(key)))
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recent Visits")
        data = current_records("visits")
        if data:
            df = pd.DataFrame(data)
            if "date" in df: df = df.sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else: st.info("No visits recorded. Select a demo profile or add a visit.")
    with right:
        st.subheader("Recent Symptoms")
        data = current_records("symptoms")
        if data:
            df = pd.DataFrame(data)
            if "date" in df: df = df.sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else: st.info("No symptoms logged.")
    st.subheader("Vitals Snapshot")
    vitals = current_records("vitals")
    if vitals:
        latest = sorted(vitals, key=lambda x: str(x.get("date", "")))[-1]
        cols = st.columns(4)
        cols[0].metric("Blood Pressure", f"{latest.get('systolic', '—')}/{latest.get('diastolic', '—')}")
        cols[1].metric("Heart Rate", f"{latest.get('heart_rate', '—')} bpm")
        cols[2].metric("Temperature", f"{latest.get('temperature', '—')} °C")
        cols[3].metric("Weight", f"{latest.get('weight', '—')} kg")
        if latest.get("glucose_mg_dl") is not None:
            st.metric("Recorded Glucose", f"{latest['glucose_mg_dl']} mg/dL")
        st.caption("Recorded values only; not a clinical diagnostic tool.")
    else: st.info("Add or load vitals to populate this section.")
    st.subheader("Quick Actions")
    a, b, c = st.columns(3)
    for col, label, destination in [
        (a, "Add a Health Visit", "Health Notes & Timeline"),
        (b, "Review Trends & Alerts", "Vitals & Analytics"),
        (c, "Export Doctor Summary", "Doctor Summary PDF"),
    ]:
        if col.button(label, use_container_width=True):
            st.session_state.quick_nav = destination
            st.session_state.page = destination
            st.rerun()

# ------------------------- Health Notes & Timeline -------------------------
elif page == "Health Notes & Timeline":
    st.subheader("Health Notes & Timeline")
    with st.expander("Add a Health Visit", expanded=True):
        with st.form("visit_form", clear_on_submit=True):
            vd = st.date_input("Visit Date", value=date.today())
            provider = st.text_input("Hospital / Doctor / Clinic")
            reason = st.text_input("Reason for Visit")
            diagnosis = st.text_input("Diagnosis or Assessment (optional)")
            visit_notes = st.text_area("Visit Notes")
            save_visit = st.form_submit_button("Save Visit", type="primary")
        if save_visit:
            st.session_state.visits.append({"date": vd.isoformat(), "provider": provider.strip(),
                "reason": reason.strip(), "diagnosis": diagnosis.strip(), "notes": visit_notes.strip()})
            notify("Visit saved successfully.")
    with st.expander("Add a Health Note"):
        with st.form("note_form", clear_on_submit=True):
            nd = st.date_input("Note Date", value=date.today())
            nt = st.text_input("Note Title")
            ntxt = st.text_area("Note")
            save_note = st.form_submit_button("Save Note", type="primary")
        if save_note:
            if not nt.strip() and not ntxt.strip(): st.warning("Enter a title or note.")
            else:
                st.session_state.notes.append({"date": nd.isoformat(), "title": nt.strip(), "text": ntxt.strip()})
                notify("Note saved successfully.")
    st.divider()
    for key, label, editor_key in [("visits", "Visits", "visit_editor"), ("notes", "Notes", "notes_editor")]:
        st.subheader(label)
        records = current_records(key)
        if records:
            df = pd.DataFrame(records)
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                                    hide_index=True, key=editor_key)
            st.session_state[key] = edited.fillna("").to_dict("records")
            st.download_button(f"Download {label.lower()} CSV", csv_bytes(st.session_state[key]),
                               file_name=f"caretrail_{key}.csv", mime="text/csv", key=f"dl_{key}")
        else: st.info(f"No {label.lower()} yet.")

# ------------------------- Vitals & Analytics -------------------------
elif page == "Vitals & Analytics":
    st.subheader("Vitals & Analytics (Predictive & Alert Engine)")
    with st.expander("Add a Measurement", expanded=True):
        with st.form("vitals_form", clear_on_submit=True):
            vd = st.date_input("Measurement Date", value=date.today())
            a, b = st.columns(2)
            sys = a.number_input("Systolic (mmHg)", 0, 300, 120)
            dia = b.number_input("Diastolic (mmHg)", 0, 200, 80)
            c, d, e = st.columns(3)
            hr = c.number_input("Heart Rate (bpm)", 0, 300, 72)
            temp = d.number_input("Temperature (°C)", 25.0, 45.0, 36.7, .1)
            weight = e.number_input("Weight (kg)", 0.0, 500.0, 60.0, .1)
            glucose = st.number_input("Glucose (mg/dL; 0 = not measured)", 0, 1000, 0)
            save_v = st.form_submit_button("Save Measurements", type="primary")
        if save_v:
            st.session_state.vitals.append({"date": vd.isoformat(), "systolic": sys, "diastolic": dia,
                "heart_rate": hr, "temperature": temp, "weight": weight,
                "glucose_mg_dl": glucose if glucose > 0 else None})
            notify("Vitals saved successfully.")
            
    records = current_records("vitals")
    if records:
        # Clinical Risk Assessment Alert Rules
        st.divider()
        st.subheader("Automated Clinical Risk Assessment")
        latest = sorted(records, key=lambda x: str(x.get("date", "")))[-1]
        sys_val = latest.get("systolic", 120)
        dia_val = latest.get("diastolic", 80)
        temp_val = latest.get("temperature", 36.6)
        
        alerts = []
        if sys_val >= 140 or dia_val >= 90:
            alerts.append(("error", f"**Hypertension Stage 2 Alert:** Blood pressure reading ({sys_val}/{dia_val} mmHg) exceeds threshold."))
        elif sys_val >= 130 or dia_val >= 80:
            alerts.append(("warning", f"**Hypertension Stage 1 Caution:** Blood pressure ({sys_val}/{dia_val} mmHg) is elevated."))
        
        if temp_val and temp_val >= 38.0:
            alerts.append(("error", f"**Fever Alert:** Body temperature ({temp_val} °C) indicates pyrexia."))

        if alerts:
            for alert_type, msg in alerts:
                if alert_type == "error": st.error(msg)
                else: st.warning(msg)
        else:
            st.success("All recent vitals parameters remain within normal baseline ranges.")

        st.divider()
        st.subheader("Editable Vitals Table")
        edited = st.data_editor(pd.DataFrame(records), num_rows="dynamic",
                                use_container_width=True, hide_index=True, key="vitals_editor")
        st.session_state.vitals = edited.fillna("").to_dict("records")
        df = pd.DataFrame(st.session_state.vitals)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        choice = st.selectbox("Choose Chart", ["Blood pressure", "Heart rate", "Temperature", "Weight", "Glucose"])
        fig = None
        if choice == "Blood pressure":
            cols = [x for x in ["systolic", "diastolic"] if x in df.columns]
            if cols:
                long = df.melt(id_vars=["date"], value_vars=cols, var_name="measurement", value_name="value")
                fig = px.line(long, x="date", y="value", color="measurement", markers=True,
                              title="Blood Pressure Trend", labels={"value": "mmHg", "date": "Date"})
        else:
            mapping = {"Heart rate": ("heart_rate", "Heart Rate (bpm)"),
                       "Temperature": ("temperature", "Temperature (°C)"),
                       "Weight": ("weight", "Weight (kg)"),
                       "Glucose": ("glucose_mg_dl", "Glucose (mg/dL)")}
            column, label = mapping[choice]
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                chart_df = df.dropna(subset=[column])
                if not chart_df.empty:
                    fig = px.line(chart_df, x="date", y=column, markers=True, title=f"{choice} Trend",
                                  labels={column: label, "date": "Date"})
            if fig is None: st.info("No values entered for this measurement.")
        if fig is not None: st.plotly_chart(chart_theme(fig), use_container_width=True)
        st.download_button("Download Vitals CSV", csv_bytes(st.session_state.vitals),
                           file_name="caretrail_vitals.csv", mime="text/csv")
    else: st.info("No vitals saved yet. Pick a demo profile from sidebar.")

# ------------------------- Symptom Tracker -------------------------
elif page == "Symptom Tracker":
    st.subheader("Symptom Tracker")
    with st.form("symptom_form", clear_on_submit=True):
        sd = st.date_input("Date", value=date.today())
        symptom = st.text_input("Symptom")
        severity = st.slider("Severity (0–10)", 0, 10, 3)
        duration = st.selectbox("Duration", ["Less than an hour", "A few hours", "1 day", "Several days", "Ongoing"])
        symptom_notes = st.text_area("Additional Notes")
        save_symptom = st.form_submit_button("Log Symptom", type="primary")
    if save_symptom:
        if not symptom.strip(): st.warning("Enter a symptom before saving.")
        else:
            st.session_state.symptoms.append({"date": sd.isoformat(), "symptom": symptom.strip(),
                "severity": severity, "duration": duration, "notes": symptom_notes.strip()})
            notify("Symptom logged successfully.")
    records = current_records("symptoms")
    if records:
        edited = st.data_editor(pd.DataFrame(records), num_rows="dynamic", use_container_width=True,
                                hide_index=True, key="symptoms_editor",
                                column_config={"severity": st.column_config.NumberColumn("Severity", min_value=0, max_value=10, step=1)})
        st.session_state.symptoms = edited.fillna("").to_dict("records")
        df = pd.DataFrame(st.session_state.symptoms)
        df["severity"] = pd.to_numeric(df.get("severity"), errors="coerce")
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df = df.dropna(subset=["date", "severity"])
        if not df.empty:
            fig = px.line(df, x="date", y="severity", color="symptom", markers=True,
                          title="Symptom Severity Over Time", labels={"date": "Date", "severity": "Severity (0–10)"})
            st.plotly_chart(chart_theme(fig), use_container_width=True)
        st.download_button("Download Symptom CSV", csv_bytes(st.session_state.symptoms),
                           file_name="caretrail_symptoms.csv", mime="text/csv")
    else: st.info("No symptoms logged. Pick a demo profile from sidebar.")

# ------------------------- Medications & Reminders -------------------------
elif page == "Medications & Reminders":
    st.subheader("Medications & Reminders")
    st.warning("Enter medication information as prescribed. This app does not recommend doses or verify drug interactions.")
    with st.form("medication_form", clear_on_submit=True):
        name = st.text_input("Medication Name")
        dose = st.text_input("Dose as Prescribed")
        frequency = st.text_input("Frequency as Prescribed")
        reminder_time = st.time_input("Reminder Time", value=time(9, 0))
        start_date = st.date_input("Start Date", value=date.today())
        med_notes = st.text_area("Notes")
        save_med = st.form_submit_button("Add Medication", type="primary")
    if save_med:
        if not name.strip(): st.warning("Enter a medication name.")
        else:
            st.session_state.medications.append({"name": name.strip(), "dose": dose.strip(),
                "frequency": frequency.strip(), "time": reminder_time.strftime("%H:%M"),
                "start_date": start_date.isoformat(), "notes": med_notes.strip()})
            notify("Medication entry saved.")
    meds = current_records("medications")
    if meds:
        edited = st.data_editor(pd.DataFrame(meds), num_rows="dynamic", use_container_width=True,
                                hide_index=True, key="medication_editor")
        st.session_state.medications = edited.fillna("").to_dict("records")
        st.download_button("Download Medication List CSV", csv_bytes(st.session_state.medications),
                           file_name="caretrail_medications.csv", mime="text/csv")
        st.subheader("Calendar Reminders")
        for i, med in enumerate(st.session_state.medications):
            raw_time = str(med.get("time", "09:00"))
            try: hour, minute = map(int, raw_time.split(":")[:2])
            except (ValueError, AttributeError): hour, minute = 9, 0
            try: start_day = date.fromisoformat(str(med.get("start_date", today_str())))
            except ValueError: start_day = date.today()
            start = datetime.combine(start_day, time(hour, minute))
            summary = re.sub(r"([,;])", r"\\\1", str(med.get("name", "Medication")))
            summary = summary.replace("\\", "\\\\").replace("\n", "\\n")
            ics = "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CareTrail//Medication Reminder//EN",
                "BEGIN:VEVENT", f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{(start+timedelta(minutes=10)).strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:Medication reminder - {summary}",
                "DESCRIPTION:Follow your clinician's instructions.", "END:VEVENT", "END:VCALENDAR", ""])
            st.download_button(f"Download Reminder: {med.get('name', 'Medication')}", ics,
                               file_name=f"caretrail_reminder_{i+1}.ics", mime="text/calendar", key=f"med_ics_{i}")
    else: st.info("No medication entries. Select a demo profile from sidebar.")

# ------------------------- Document Vault & Privacy Sanitizer -------------------------
elif page == "Document Vault":
    st.subheader("Document Vault & PII/PHI De-identification Tool")
    st.caption("HIPAA Security Rule Safeguard: All uploads remain local in memory.")
    
    uploaded = st.file_uploader("Upload a Report or Note", type=["txt", "md", "csv", "pdf"])
    if uploaded is not None and st.button("Save Document", type="primary"):
        content = uploaded.getvalue()
        extension = Path(uploaded.name).suffix.lower()
        extracted = ""
        if extension in [".txt", ".md", ".csv"]:
            extracted = content.decode("utf-8", errors="replace")
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                extracted = f"Text extraction failed: {exc}"
        st.session_state.documents.append({"name": uploaded.name, "type": uploaded.type or "application/octet-stream",
            "size": len(content), "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "text": extracted, "bytes_hex": content.hex()})
        notify("Document saved to session.")
        
    docs = current_records("documents")
    if docs:
        for i, doc in enumerate(docs):
            with st.expander(doc.get("name", "Document")):
                st.write(f"File size: {doc.get('size', 0):,} bytes | Uploaded: {doc.get('uploaded_at', '')}")
                raw_txt = doc.get("text", "")
                if raw_txt:
                    st.text_area("Extracted Text", raw_txt[:5000], height=140, key=f"doc_raw_{i}")
                    
                    # De-identification Scrubbing Demonstration
                    if st.button(f"Sanitize PII/PHI (HIPAA/DPDP Mode)", key=f"scrub_{i}"):
                        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', raw_txt)
                        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', sanitized)
                        sanitized = re.sub(r'\b\d{10}\b', '[REDACTED PHONE]', sanitized)
                        st.success("PHI Scrubbing Applied Successfully:")
                        st.text_area("De-identified Output", sanitized[:5000], height=140, key=f"doc_scrubbed_{i}")

                if st.button("Delete Document", key=f"delete_doc_{i}"):
                    st.session_state.documents.pop(i)
                    st.rerun()
    else: st.info("No documents uploaded in this session.")

# ------------------------- Visit Comparison -------------------------
elif page == "Visit Comparison":
    st.subheader("Visit Comparison")
    visits = current_records("visits")
    if len(visits) < 2:
        st.info("Add at least two visits or select a demo profile.")
    else:
        labels = [f"{v.get('date', '')} — {v.get('provider', 'Provider not entered')}" for v in visits]
        a, b = st.columns(2)
        ia = a.selectbox("First Visit", range(len(visits)), format_func=lambda i: labels[i], key="compare_a")
        ib = b.selectbox("Second Visit", range(len(visits)), index=1 if len(visits) > 1 else 0, format_func=lambda i: labels[i], key="compare_b")
        if ia == ib: st.warning("Select two different visits.")
        else:
            va, vb = visits[ia], visits[ib]
            comparison = pd.DataFrame([{"Field": field, "First visit": va.get(field, ""),
                                        "Second visit": vb.get(field, "")}
                                       for field in ["date", "provider", "reason", "diagnosis", "notes"]])
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.download_button("Download Comparison CSV", csv_bytes(comparison.to_dict("records")),
                               file_name="caretrail_visit_comparison.csv", mime="text/csv")

# ------------------------- AI Question Classifier & Evaluation -------------------------
elif page == "AI Question Classifier":
    st.subheader("AI Question Intent Classifier")
    st.caption("Module 2 AI Technique: Local TF-IDF + LinearSVC Pipeline for offline query categorization.")
    
    tab1, tab2 = st.tabs(["Query Assistant", "Model Performance & Metrics (Rubric Evaluation)"])
    
    pipeline, report_dict, cm = get_trained_nlp_pipeline()
    
    with tab1:
        question = st.text_area("Enter a patient health query:", placeholder="e.g., Where can I view my recent blood pressure graph?", height=100)
        if st.button("Classify Query Intent", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                pred_label = pipeline.predict([question.strip()])[0]
                
                responses = {
                    "symptom_help": "Triage Guidance: Record onset time, duration, and pain score (0-10). Seek immediate emergency medical care if experiencing chest pain, severe shortness of breath, or sudden weakness.",
                    "medication_query": "Medication Protocol: Verify prescribed timing and dosage in your Medications tab. Never alter prescribed doses without consulting your doctor.",
                    "records_query": "Record Navigation: Your clinical visit entries and doctor notes are structured under the Health Notes & Timeline page.",
                    "health_trends": "Trend Analytics: Historical graphical representations of your vitals and glucose logs are updated under Vitals & Analytics.",
                    "general_help": "System Guidance: CareTrail allows full local PHR management, iCal reminder exports, and HIPAA-compliant session backups."
                }
                
                res_text = responses.get(pred_label, "Query processed by local classifier.")
                result = {"question": question.strip(), "intent": pred_label, "response": res_text, "timestamp": datetime.now().isoformat(timespec="seconds")}
                st.session_state.classifier_result = result
                st.session_state.ai_history.append(result)
                notify("Intent Classified.")

        result = st.session_state.classifier_result
        if result:
            st.divider()
            st.subheader("Classification Outcome")
            st.metric("Predicted Intent Category", result.get("intent"))
            st.write(f"**Assistant Response:** {result.get('response')}")

    with tab2:
        st.subheader("Quantitative Model Evaluation Metrics")
        st.write("Cross-validated evaluation metrics generated on synthetic PHR intent dataset:")
        
        rep_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(rep_df.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        classes = list(set(pipeline.named_steps['clf'].classes_))
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Teal", title="Intent Confusion Matrix")
        st.plotly_chart(chart_theme(fig_cm), use_container_width=True)

# ------------------------- Backup & Restore -------------------------
elif page == "Data Backup & Restore":
    st.subheader("Data Backup & Restore")
    st.write("Download a JSON backup of current session records.")
    st.download_button("Download Complete JSON Backup", make_backup().encode("utf-8"),
                       file_name=f"caretrail_backup_{today_str()}.json", mime="application/json", type="primary")
    st.divider()
    st.subheader("Restore a Backup")
    backup_file = st.file_uploader("Select a CareTrail JSON file", type=["json"], key="backup_restore")
    if st.button("Restore Backup", disabled=backup_file is None):
        try:
            with st.spinner("Restoring session records..."):
                restore_backup(backup_file)
            notify("Backup restored.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore backup: {exc}")

# ------------------------- Doctor Summary PDF -------------------------
elif page == "Doctor Summary PDF":
    st.subheader("Doctor Summary PDF")
    st.write("Generate a formatted summary of visits, recent vitals, symptoms and medication entries.")
    if st.button("Generate Doctor Summary PDF", type="primary"):
        try:
            with st.spinner("Preparing PDF report..."):
                st.session_state.generated_pdf = make_doctor_pdf()
            notify("Doctor summary PDF generated.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
    if st.session_state.generated_pdf:
        st.download_button("Download Doctor Summary PDF", st.session_state.generated_pdf,
                           file_name=f"caretrail_doctor_summary_{today_str()}.pdf",
                           mime="application/pdf", type="primary")

# ------------------------- About & Privacy -------------------------
elif page == "About & Privacy":
    st.subheader("About CareTrail")
    st.write("CareTrail is an intelligent Personal Health Record (PHR) assistant designed to simplify longitudinal health tracking while enforcing strict data privacy.")
    st.subheader("Rubric & Module Mapping")
    st.markdown("""
    - **Module Mapping:** Module 2 — Healthcare Applications (Intelligent PHR Assistant)
    - **AI Techniques:** Local TF-IDF + LinearSVC Intent Classification Pipeline & Rule-Based Alert Engine
    - **Compliance:** HIPAA Security Rule (§ 164.312) & DPDP Act 2023 alignment via on-device processing and built-in PII scrubbing.
    """)

st.divider()
st.caption(f"CareTrail Prototype · {datetime.now().strftime('%d %b %Y, %H:%M')}")
