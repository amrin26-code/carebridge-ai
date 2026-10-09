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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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
    "theme": "Dark",
    "visits": [],
    "notes": [],
    "vitals": [],
    "symptoms": [],
    "medications": [],
    "documents": [],
    "classifier_result": None,
    "visit_comparison": None,
    "active_profile": "Personal Records",
    "custom_profiles": {},
    "profile_records": {},
    "ai_history": [],
    "generated_pdf": None,
    "page": "Overview",
    "quick_nav": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)

PROFILE_DATA_KEYS = [
    "visits",
    "notes",
    "vitals",
    "symptoms",
    "medications",
    "documents",
]

if not isinstance(st.session_state.get("profile_records"), dict):
    st.session_state.profile_records = {}

if "Personal Records" not in st.session_state.profile_records:
    st.session_state.profile_records["Personal Records"] = {
        key: copy.deepcopy(st.session_state[key])
        for key in PROFILE_DATA_KEYS
    }

# ------------------------- Theme Palette System -------------------------
THEMES = {
    "Light": {
        "bg": "#FFFFFF",
        "panel": "#F8FAFC",
        "panel_alt": "#F1F5F9",
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
    },
    "Dark": {
        "bg": "#000000",
        "panel": "#111827",
        "panel_alt": "#1F2937",
        "text": "#F9FAFB",
        "muted": "#9CA3AF",
        "border": "#334155",
        "accent": "#14B8A6",
        "accent_hover": "#2DD4BF",
        "accent_text": "#042F2E",
        "input_bg": "#1F2937",
        "input_text": "#F9FAFB",
        "sidebar": "#090D16",
        "sidebar_text": "#F9FAFB",
        "plot": "plotly_dark",
    },
}

C = THEMES.get(st.session_state.theme, THEMES["Dark"])
is_dark = st.session_state.theme == "Dark"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {"dark" if is_dark else "light"}; }}
    
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
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
    .stApp input::placeholder, .stApp textarea::placeholder {{
        color: {C["muted"]} !important;
        -webkit-text-fill-color: {C["muted"]} !important;
        opacity: 0.8 !important;
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

# ------------------------- Helpers & Validation Engines -------------------------
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
    if not records:
        return pd.DataFrame().to_csv(index=False).encode("utf-8-sig")
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

def validate_dataframe(df, required_columns, context="Dataset"):
    if df is None or df.empty:
        st.warning(f"⚠️ {context} contains no valid records.")
        return False
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Input validation failed for {context}: Missing column(s): {', '.join(missing_cols)}")
        return False
    return True

def safe_text(value, limit=2000):
    return escape(str(value if value is not None else ""))[:limit].replace("\n", "<br/>")

# ------------------------- Built-in ML Classifier Engine -------------------------
@st.cache_resource
def get_trained_nlp_pipeline():
    training_dataset = [
        ("I have severe pain in my head", "symptom_help"),
        ("My fever is 102 degrees", "symptom_help"),
        ("Experiencing dizziness and nausea", "symptom_help"),
        ("Sore throat and persistent cough", "symptom_help"),
        ("Sharp chest pain when breathing", "symptom_help"),
        ("Swollen ankle after falling", "symptom_help"),
        ("Stomach cramps and acid reflux", "symptom_help"),
        ("Shortness of breath after climbing stairs", "symptom_help"),
        ("When should I take my insulin dose?", "medication_query"),
        ("Can I take aspirin with blood thinners?", "medication_query"),
        ("What is the prescribed dosage for metformin?", "medication_query"),
        ("Forgot my morning blood pressure pill", "medication_query"),
        ("Side effects of statin medications", "medication_query"),
        ("Am I supposed to take antibiotics with food?", "medication_query"),
        ("Can I skip a missed dose of lisinopril?", "medication_query"),
        ("Show my clinic visit summary", "records_query"),
        ("Download my hospital doctor notes", "records_query"),
        ("Where is my lab test result PDF stored?", "records_query"),
        ("List all doctor appointments from last month", "records_query"),
        ("Find my past medical history reports", "records_query"),
        ("View uploaded clinical records", "records_query"),
        ("Show my blood pressure graph for last week", "health_trends"),
        ("Is my glucose level spiking or stable?", "health_trends"),
        ("Display weight loss progress chart", "health_trends"),
        ("Plot my heart rate history over time", "health_trends"),
        ("Are my daily temperature trends normal?", "health_trends"),
        ("Chart my systolic readings over 30 days", "health_trends"),
        ("How do I backup my health data?", "general_help"),
        ("How to use this PHR application", "general_help"),
        ("How do I add a new patient profile?", "general_help"),
        ("Is my medical data encrypted locally?", "general_help"),
        ("Export my health summary report", "general_help"),
        ("Where are my privacy settings configured?", "general_help"),
        ("How do I restore my session JSON backup?", "general_help")
    ]
    texts, labels = zip(*training_dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC(C=1.0, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    test_preds = pipeline.predict(X_test)
    report = classification_report(y_test, test_preds, output_dict=True)
    cm = confusion_matrix(y_test, test_preds)
    test_acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_acc": test_acc,
        "macro_f1": macro_f1
    }
    return pipeline, report, cm, split_info, list(pipeline.named_steps['clf'].classes_)

# ------------------------- Profiles & Synthetic Demo Engine -------------------------
BUILTIN_PROFILES = [
    "Personal Records",
    "Alex Morgan — Diabetes Monitoring",
    "Jamie Taylor — Post-Surgery Recovery",
    "Sam Chen — Hypertension Management",
    "Patel Family — Pediatric Checkup",
    "Elena Rostova — Chronic Pain Tracking"
]

def load_synthetic_demo_comparison():
    rec_a = {
        "date": "2026-09-15", "provider": "St. Jude Clinic",
        "reason": "Routine Checkup", "diagnosis": "Hypertension Stage 1",
        "medication": "Lisinopril", "dosage": "10mg", "notes": "Patient reports mild dizziness."
    }
    rec_b = {
        "date": "2026-10-01", "provider": "City Central Hospital",
        "reason": "Follow-up Visit", "diagnosis": "Hypertension Controlled",
        "medication": "Lisinopril", "dosage": "20mg", "notes": "Dosage adjusted. Dizziness resolved."
    }
    st.session_state.visits = [
        {"date": rec_a["date"], "provider": rec_a["provider"], "reason": rec_a["reason"], "diagnosis": rec_a["diagnosis"], "notes": f"Medication: {rec_a['medication']} {rec_a['dosage']}. {rec_a['notes']}"},
        {"date": rec_b["date"], "provider": rec_b["provider"], "reason": rec_b["reason"], "diagnosis": rec_b["diagnosis"], "notes": f"Medication: {rec_b['medication']} {rec_b['dosage']}. {rec_b['notes']}"}
    ]
    st.session_state.medications = [
        {"name": "Lisinopril", "dose": "10mg", "frequency": "Once Daily", "time": "08:00", "start_date": "2026-09-15", "notes": "Record A entry"},
        {"name": "Lisinopril", "dose": "20mg", "frequency": "Once Daily", "time": "08:00", "start_date": "2026-10-01", "notes": "Record B entry (Dosage increase)"}
    ]
    st.session_state.vitals = [
        {"date": "2026-09-15", "systolic": 142, "diastolic": 90, "heart_rate": 78, "temperature": 36.7, "weight": 81.5, "glucose_mg_dl": 105},
        {"date": "2026-10-01", "systolic": 124, "diastolic": 82, "heart_rate": 72, "temperature": 36.6, "weight": 80.8, "glucose_mg_dl": 98}
    ]
    st.session_state.symptoms = [
        {"date": "2026-09-15", "symptom": "Dizziness", "severity": 6, "duration": "A few hours", "notes": "Reported during doctor checkup"},
        {"date": "2026-10-01", "symptom": "Dizziness", "severity": 0, "duration": "Ongoing", "notes": "Symptom resolved after dosage change"}
    ]
    st.session_state.active_profile = "Synthetic Patient Comparison Demo"
    st.session_state.profile_records["Synthetic Patient Comparison Demo"] = {
        key: copy.deepcopy(st.session_state.get(key, []))
        for key in PROFILE_DATA_KEYS
    }

def build_sample_profile(profile):
    vitals, symptoms = [], []
    for offset in range(13, -1, -1):
        d = date.today() - timedelta(days=offset)
        
        if profile == "Alex Morgan — Diabetes Monitoring":
            glucose_val = 126 + (offset * 7 % 38)
            v = {"date": d.isoformat(), "systolic": 118 + (offset % 4) * 2, "diastolic": 76 + (offset % 3), 
                 "heart_rate": 72 + (offset % 5), "temperature": 36.6, "weight": round(78.4 - (13-offset)*.04, 1), 
                 "glucose_mg_dl": glucose_val}
            sym, severity = "Fatigue", (offset * 3) % 6

        elif profile == "Jamie Taylor — Post-Surgery Recovery":
            glucose_val = 95 + (offset % 5) * 3
            v = {"date": d.isoformat(), "systolic": 116 + (offset % 5) * 2, "diastolic": 74 + (offset % 4), 
                 "heart_rate": 76 + (offset % 6), "temperature": round(36.5 + (offset % 4)*.1, 1), 
                 "weight": 65.2, "glucose_mg_dl": glucose_val}
            sym, severity = "Surgical Site Pain", min(8, max(1, offset // 2 + 1))

        elif profile == "Sam Chen — Hypertension Management":
            glucose_val = 102 + (offset * 3 % 15)
            v = {"date": d.isoformat(), "systolic": 142 - (13-offset)//2, "diastolic": 88 - (13-offset)//3, 
                 "heart_rate": 68 + (offset % 4), "temperature": 36.6, "weight": 82.1, "glucose_mg_dl": glucose_val}
            sym, severity = "Light Dizziness", (offset * 2) % 5

        elif profile == "Patel Family — Pediatric Checkup":
            glucose_val = 88 + (offset % 4) * 2
            v = {"date": d.isoformat(), "systolic": 102 + (offset % 3), "diastolic": 64 + (offset % 2), 
                 "heart_rate": 95 + (offset % 8), "temperature": round(36.8 + (offset % 2)*.1, 1), 
                 "weight": 24.5, "glucose_mg_dl": glucose_val}
            sym, severity = "Mild Cough", (offset) % 4

        elif profile == "Elena Rostova — Chronic Pain Tracking":
            glucose_val = 98 + (offset * 2 % 12)
            v = {"date": d.isoformat(), "systolic": 120 + (offset % 3), "diastolic": 78 + (offset % 2), 
                 "heart_rate": 74 + (offset % 4), "temperature": 36.6, "weight": 58.0, "glucose_mg_dl": glucose_val}
            sym, severity = "Joint Stiffness", min(9, max(2, (offset * 4) % 10))

        else:
            glucose_val = 95 + (offset % 6)
            v = {"date": d.isoformat(), "systolic": 120, "diastolic": 80, "heart_rate": 72, "temperature": 36.6, "weight": 70.0, "glucose_mg_dl": glucose_val}
            sym, severity = "None", 0

        vitals.append(v)
        symptoms.append({"date": d.isoformat(), "symptom": sym, "severity": severity, "duration": "Daily Log", "notes": f"Entry for patient profile: {profile}"})

    provider_name = "City General Health Center"
    med_name = f"Prescription ({profile.split('—')[0].strip()})"

    return {
        "visits": [{"date": (date.today()-timedelta(days=7)).isoformat(), "provider": provider_name, "reason": f"Follow-up for {profile.split('—')[0].strip()}", "diagnosis": "Condition Monitored", "notes": f"Scenario record for {profile}"}],
        "notes": [{"date": today_str(), "title": "Clinical Summary Note", "text": f"Active profile set to {profile}."}],
        "vitals": vitals,
        "symptoms": symptoms,
        "medications": [{"name": med_name, "dose": "10mg", "frequency": "Once Daily", "time": "09:00", "start_date": today_str(), "notes": "Take with water"}],
        "documents": []
    }

def load_sample_profile(profile):
    """Save the current profile and load records belonging to the selected profile."""
    profile_records = st.session_state.profile_records
    current_profile = st.session_state.active_profile
    # Save the currently active profile before switching.
    profile_records[current_profile] = {
        key: copy.deepcopy(st.session_state.get(key, []))
        for key in PROFILE_DATA_KEYS
    }
    # Load previously saved records for the selected profile.
    if profile in profile_records:
        data = copy.deepcopy(profile_records[profile])
    # Load a custom profile for its first use.
    elif profile in st.session_state.custom_profiles:
        data = copy.deepcopy(st.session_state.custom_profiles[profile])
    # Load a built-in synthetic profile for its first use.
    elif profile in BUILTIN_PROFILES and profile != "Personal Records":
        data = build_sample_profile(profile)
    # Personal Records starts with its own saved records.
    elif profile == "Personal Records":
        data = {
            key: [] for key in PROFILE_DATA_KEYS
        }
    else:
        st.error("The selected patient profile could not be loaded.")
        return
    for key in PROFILE_DATA_KEYS:
        st.session_state[key] = copy.deepcopy(data.get(key, []))
    st.session_state.active_profile = profile
    # Save a snapshot of the newly selected profile.
    profile_records[profile] = {
        key: copy.deepcopy(st.session_state[key])
        for key in PROFILE_DATA_KEYS
    }

def make_backup():
    keys = [
        "visits",
        "notes",
        "vitals",
        "symptoms",
        "medications",
        "documents",
        "classifier_result",
        "visit_comparison",
        "ai_history",
        "active_profile",
        "custom_profiles",
        "profile_records",
        "theme",
    ]
    data = {
        key: copy.deepcopy(st.session_state.get(key))
        for key in keys
    }
    data.update({
        "exported_at": datetime.now().isoformat(),
        "format_version": 2,
    })
    return json.dumps(data, indent=2, ensure_ascii=False)

def restore_backup(uploaded_file):
    """Validate a backup before modifying the current session."""
    if uploaded_file is None:
        st.error("Backup file is missing.")
        return False
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        st.error("The uploaded file is not valid UTF-8 JSON.")
        return False
    except Exception as exc:
        st.error(f"Could not read backup: {exc}")
        return False
    if not isinstance(data, dict):
        st.error("The backup must contain a JSON object.")
        return False
    list_keys = [
        "visits",
        "notes",
        "vitals",
        "symptoms",
        "medications",
        "documents",
        "ai_history",
    ]
    # Validate all record lists before changing session state.
    for key in list_keys:
        if key in data:
            if not isinstance(data[key], list):
                st.error(f"Invalid backup: '{key}' must be a list.")
                return False
            if not all(isinstance(item, dict) for item in data[key]):
                st.error(
                    f"Invalid backup: every item in '{key}' must be an object."
                )
                return False
    for key in ["custom_profiles", "profile_records"]:
        if key in data and not isinstance(data[key], dict):
            st.error(f"Invalid backup: '{key}' must be an object.")
            return False
    if "active_profile" in data and not isinstance(data["active_profile"], str):
        st.error("Invalid backup: active_profile must be text.")
        return False
    if "theme" in data and data["theme"] not in THEMES:
        st.error("Invalid backup: unsupported theme.")
        return False
    if "classifier_result" in data:
        value = data["classifier_result"]
        if value is not None and not isinstance(value, dict):
            st.error("Invalid backup: classifier_result must be an object or null.")
            return False
    if "visit_comparison" in data:
        value = data["visit_comparison"]
        if value is not None and not isinstance(value, dict):
            st.error("Invalid backup: visit_comparison must be an object or null.")
            return False
    # Validate saved profile records, if supplied.
    profile_records = data.get("profile_records", {})
    if profile_records:
        for profile_name, records in profile_records.items():
            if not isinstance(profile_name, str) or not isinstance(records, dict):
                st.error("Invalid profile data in backup.")
                return False
            for key in PROFILE_DATA_KEYS:
                if key in records:
                    if not isinstance(records[key], list):
                        st.error(
                            f"Invalid records for '{profile_name}': "
                            f"'{key}' must be a list."
                        )
                        return False
                    if not all(isinstance(item, dict) for item in records[key]):
                        st.error(
                            f"Invalid records for '{profile_name}': "
                            f"'{key}' contains an invalid entry."
                        )
                        return False
    # Apply the validated backup.
    restore_keys = list_keys + [
        "classifier_result",
        "visit_comparison",
        "active_profile",
        "custom_profiles",
        "profile_records",
        "theme",
    ]
    for key in restore_keys:
        if key in data:
            st.session_state[key] = copy.deepcopy(data[key])
    # Support older backups without profile_records.
    active_profile = st.session_state.active_profile
    if active_profile not in st.session_state.profile_records:
        st.session_state.profile_records[active_profile] = {
            key: copy.deepcopy(st.session_state.get(key, []))
            for key in PROFILE_DATA_KEYS
        }
    return True

# ------------------------- PDF Summary Generator -------------------------
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
    
    current_patient = st.session_state.get('active_profile', 'Personal Records')
    story = [
        Paragraph("CareTrail — Doctor Summary", styles["CareTitle"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["Normal"]),
        Paragraph(f"<b>Patient Profile:</b> {safe_text(current_patient)}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Important Information", styles["CareSection"]),
        Paragraph(f"This personalized medical summary contains structured records specifically for patient: <b>{safe_text(current_patient)}</b>.", styles["CareSmall"]),
    ]
    def add_table_section(title, rows, widths=None):
        story.append(Paragraph(title, styles["CareSection"]))
        if not rows or len(rows) <= 1:
            story.append(Paragraph("No records saved for this profile.", styles["Normal"]))
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
        vrows.append([safe_text(r.get(k, "Not Specified")) for k in ["date", "systolic", "diastolic", "heart_rate", "temperature", "weight", "glucose_mg_dl"]])
    add_table_section("Recent Vitals", vrows)

    symptoms = current_records("symptoms")
    srows = [["Date", "Symptom", "Severity", "Duration", "Notes"]]
    for r in sorted(symptoms, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
        srows.append([Paragraph(safe_text(r.get(k, "Not Specified"), 300), styles["CareSmall"]) for k in ["date", "symptom", "severity", "duration", "notes"]])
    add_table_section("Recent Symptoms", srows)

    visits = current_records("visits")
    story.append(Paragraph("Health Visits", styles["CareSection"]))
    if visits:
        for r in sorted(visits, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
            story.append(Paragraph(
                f"<b>{safe_text(r.get('date', ''))} — {safe_text(r.get('provider', 'Provider not entered'))}</b><br/>"
                f"Reason: {safe_text(r.get('reason', 'Not Specified'))}<br/>Assessment: {safe_text(r.get('diagnosis', 'Not Specified'))}<br/>"
                f"Notes: {safe_text(r.get('notes', 'Not Specified'))}", styles["CareSmall"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No visit records saved.", styles["Normal"]))

    meds = current_records("medications")
    mrows = [["Name", "Dose entered", "Frequency entered", "Time"]]
    for r in meds:
        mrows.append([Paragraph(safe_text(r.get(k, "Not Specified"), 150), styles["CareSmall"]) for k in ["name", "dose", "frequency", "time"]])
    add_table_section("Medication List", mrows)

    doc.build(story)
    return output.getvalue()

# ------------------------- Sidebar Navigation & Controls -------------------------
NAV = [
    "Overview", "Health Notes & Timeline", "Vitals & Analytics", "Symptom Tracker",
    "Medications & Reminders", "Document Vault", "Visit Comparison",
    "AI Question Classifier", "Data Backup & Restore", "Doctor Summary PDF",
    "About & Privacy",
]

with st.sidebar:
    st.markdown("## 🩺 CareTrail")
    st.caption("AI Personal Health Record Portal")
    st.divider()
    
    # Theme Switcher
    chosen_theme = st.radio("Appearance Theme", ["Dark", "Light"],
                            index=0 if st.session_state.theme == "Dark" else 1,
                            horizontal=True, key="theme_picker")
    if chosen_theme != st.session_state.theme:
        st.session_state.theme = chosen_theme
        st.rerun()
    st.divider()

    # ONE-CLICK SYNTHETIC DEMO BUTTON
    if st.button("⚡ Load Sample Patient Demo", type="primary", use_container_width=True):
        load_synthetic_demo_comparison()
        st.session_state.page = "Visit Comparison"
        notify("Loaded Synthetic Patient Demo Records!")
        st.rerun()
        
    st.divider()
    
    all_profiles = BUILTIN_PROFILES + list(st.session_state.custom_profiles.keys())
    if st.session_state.active_profile not in all_profiles:
        all_profiles.append(st.session_state.active_profile)
    current_profile = st.session_state.active_profile
    profile_index = all_profiles.index(current_profile)
    
    st.subheader("Select Patient Profile")
    chosen_profile = st.selectbox(
        "Choose profile to test", 
        all_profiles, 
        index=profile_index,
        key="profile_picker"
    )
    
    if chosen_profile != st.session_state.active_profile:
        load_sample_profile(chosen_profile)
        notify(f"Switched to {chosen_profile}")
        st.rerun()
        
    st.caption("Switch between demo patient scenarios or custom profiles.")
    
    # Custom Profile Creation Form
    with st.expander("➕ Add New Patient Profile"):
        with st.form("new_patient_form", clear_on_submit=True):
            p_name = st.text_input("Patient Full Name")
            p_condition = st.text_input("Primary Condition", placeholder="e.g. Asthma, High Cholesterol")
            p_sys = st.number_input("Baseline Systolic (mmHg)", 50, 250, 120)
            p_dia = st.number_input("Baseline Diastolic (mmHg)", 30, 150, 80)
            p_hr = st.number_input("Baseline Heart Rate (bpm)", 40, 200, 72)
            p_glucose = st.number_input("Glucose (mg/dL; 0 = N/A)", 0, 500, 100)
            
            create_btn = st.form_submit_button("Create Patient Profile", type="primary")
            
        if create_btn:
            if not p_name.strip():
                st.warning("Please enter a patient name.")
            else:
                formatted_profile_name = f"{p_name.strip()} — {p_condition.strip() or 'General Checkup'}"
                new_profile_data = {
                    "visits": [{"date": today_str(), "provider": "General Health Clinic", "reason": "Initial Patient Registration", "diagnosis": "Baseline Evaluation", "notes": f"New test patient registered: {p_condition}"}],
                    "notes": [{"date": today_str(), "title": "Patient Setup Note", "text": "Created profile for testing and tracking."}],
                    "vitals": [{"date": today_str(), "systolic": p_sys, "diastolic": p_dia, "heart_rate": p_hr, "temperature": 36.6, "weight": 70.0, "glucose_mg_dl": p_glucose if p_glucose > 0 else None}],
                    "symptoms": [],
                    "medications": [],
                    "documents": []
                }
                st.session_state.custom_profiles[formatted_profile_name] = new_profile_data
                load_sample_profile(formatted_profile_name)
                notify(f"Created Profile: {formatted_profile_name}")
                st.rerun()

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

# ------------------------- Page Header Banner & Privacy Panel -------------------------
if st.session_state.active_profile != "Personal Records":
    st.info(f"Active Patient Profile: **{st.session_state.active_profile}**")

st.markdown(
    f"""<div style="background:{C['panel']};border:1px solid {C['border']};
    border-radius:16px;padding:20px 24px;margin-bottom:15px;box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
    <div style="font-size:12px;letter-spacing:1.5px;font-weight:700;color:{C['accent']};text-transform:uppercase;">
    Personal Health Record Portal</div>
    <div style="font-size:28px;font-weight:800;color:{C['text']};margin:4px 0;">
    CareTrail Dashboard</div>
    <div style="font-size:14px;color:{C['muted']};">
    Longitudinal PHR tracking, discrepancy comparison analysis, and local NLP intent classification.</div></div>""",
    unsafe_allow_html=True,
)

# PRIVACY & SAFETY PANEL
with st.expander("🛡️ Privacy, Safety & Technical Limitations Panel", expanded=False):
    st.markdown("""- **Synthetic Data Usage:** Built-in demonstration profiles contain fictional health records.
- **Non-Diagnostic Role:** CareTrail is an educational record-management prototype. It does not diagnose conditions, prescribe treatment, or recommend automated dosage changes.
- **Data Handling:** Records are maintained in Streamlit session state on the application server while the session is active. Session data may not persist after the session ends. Do not enter real patient identifiers or confidential medical information.
- **Document Handling:** Uploaded files and extracted text may be processed by the hosted application. Do not upload identifiable patient records.
- **Professional Verification:** Record discrepancies and threshold alerts require appropriate professional review.
- **Clinical Limitations:** The alerts and classifier are demonstrations, not validated clinical decision-support tools.""")

# ------------------------- Overview -------------------------
if page == "Overview":
    st.subheader("Overview")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Health Visits", "Vitals Records", "Symptoms Logged", "Medication Entries"],
                               ["visits", "vitals", "symptoms", "medications"]):
        col.metric(label, len(current_records(key)))
        
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Quick Synthetic Demonstration")
        st.write("Load two pre-configured synthetic medical records to immediately test match detection, discrepancy handling, and export tools.")
        if st.button("⚡ Load Sample Synthetic Records", type="primary"):
            load_synthetic_demo_comparison()
            st.session_state.page = "Visit Comparison"
            st.rerun()
            
    with c2:
        st.subheader("Recent Visits Snapshot")
        data = current_records("visits")
        if data:
            df = pd.DataFrame(data)
            if "date" in df: df = df.sort_values("date", ascending=False)
            st.dataframe(df.head(4), use_container_width=True, hide_index=True)
        else:
            st.info("No visit records saved in current profile.")

    st.subheader("Vitals Overview")
    vitals = current_records("vitals")
    if vitals:
        latest = sorted(vitals, key=lambda x: str(x.get("date", "")))[-1]
        cols = st.columns(4)
        cols[0].metric("Blood Pressure", f"{latest.get('systolic', '—')}/{latest.get('diastolic', '—')}")
        cols[1].metric("Heart Rate", f"{latest.get('heart_rate', '—')} bpm")
        cols[2].metric("Temperature", f"{latest.get('temperature', '—')} °C")
        cols[3].metric("Weight", f"{latest.get('weight', '—')} kg")
        if latest.get("glucose_mg_dl") is not None:
            st.metric("Glucose Reading", f"{latest['glucose_mg_dl']} mg/dL")
    else: 
        st.info("Add or load vitals data to populate metrics.")

# ------------------------- Health Notes & Timeline -------------------------
elif page == "Health Notes & Timeline":
    st.subheader("Health Notes & Timeline")
    with st.expander("Add a Health Visit Entry", expanded=True):
        with st.form("visit_form", clear_on_submit=True):
            vd = st.date_input("Visit Date", value=date.today())
            provider = st.text_input("Hospital / Doctor / Clinic")
            reason = st.text_input("Reason for Visit")
            diagnosis = st.text_input("Diagnosis or Assessment")
            visit_notes = st.text_area("Visit Notes")
            save_visit = st.form_submit_button("Save Visit Record", type="primary")
            
        if save_visit:
            if not provider.strip():
                st.error("❌ Input Validation Error: Provider name is required.")
            else:
                st.session_state.visits.append({
                    "date": vd.isoformat(), 
                    "provider": provider.strip(),
                    "reason": reason.strip() or "Not Specified", 
                    "diagnosis": diagnosis.strip() or "Not Specified", 
                    "notes": visit_notes.strip() or "Not Specified"
                })
                notify("Visit entry saved successfully.")
                st.rerun()

    with st.expander("Add a Personal Health Note"):
        with st.form("note_form", clear_on_submit=True):
            nd = st.date_input("Note Date", value=date.today())
            nt = st.text_input("Note Title")
            ntxt = st.text_area("Note Content")
            save_note = st.form_submit_button("Save Note", type="primary")
        if save_note:
            if not nt.strip() and not ntxt.strip(): 
                st.warning("Please enter a title or note content.")
            else:
                st.session_state.notes.append({"date": nd.isoformat(), "title": nt.strip() or "Untitled", "text": ntxt.strip() or "Not Specified"})
                notify("Note saved.")
                st.rerun()

    st.divider()
    for key, label, editor_key in [("visits", "Visits", "visit_editor"), ("notes", "Notes", "notes_editor")]:
        st.subheader(f"Saved {label}")
        records = current_records(key)
        if records:
            df = pd.DataFrame(records)
            if validate_dataframe(df, ["date"], context=label):
                edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True, key=editor_key)
                st.session_state[key] = edited.fillna("Not Specified").to_dict("records")
                st.download_button(f"Download {label.lower()} CSV", csv_bytes(st.session_state[key]), file_name=f"caretrail_{key}.csv", mime="text/csv", key=f"dl_{key}")
        else: 
            st.info(f"No {label.lower()} recorded yet.")

# ------------------------- Vitals & Analytics -------------------------
elif page == "Vitals & Analytics":
    st.subheader("Vitals & Analytics (Predictive & Alert Engine)")
    with st.expander("Log New Measurement", expanded=True):
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
            st.session_state.vitals.append({
                "date": vd.isoformat(), "systolic": sys, "diastolic": dia,
                "heart_rate": hr, "temperature": temp, "weight": weight,
                "glucose_mg_dl": glucose if glucose > 0 else None
            })
            notify("Vitals recorded successfully.")
            st.rerun()
            
    records = current_records("vitals")
    if records:
        st.divider()
        st.subheader("Automated Risk Assessment Alerts")
        
        latest = max(
            records,
            key=lambda record: str(record.get("date", "")),
        )

        def numeric_value(record, key):
            value = pd.to_numeric(record.get(key), errors="coerce")
            return None if pd.isna(value) else float(value)

        sys_val = numeric_value(latest, "systolic")
        dia_val = numeric_value(latest, "diastolic")
        temp_val = numeric_value(latest, "temperature")

        alerts = []
        if sys_val is not None and dia_val is not None:
            if sys_val >= 140 or dia_val >= 90:
                alerts.append((
                    "error",
                    f"**Blood Pressure Alert:** Recorded blood pressure "
                    f"({sys_val:g}/{dia_val:g} mmHg) meets this prototype's "
                    "high-reading alert threshold."
                ))
            elif sys_val >= 130 or dia_val >= 80:
                alerts.append((
                    "warning",
                    f"**Blood Pressure Caution:** Recorded blood pressure "
                    f"({sys_val:g}/{dia_val:g} mmHg) meets this prototype's "
                    "elevated-reading caution threshold."
                ))

        if temp_val is not None and temp_val >= 38.0:
            alerts.append((
                "error",
                f"**Temperature Alert:** Recorded temperature "
                f"({temp_val:g} °C) meets this prototype's fever threshold."
            ))

        if alerts:
            for alert_type, message in alerts:
                if alert_type == "error":
                    st.error(message)
                else:
                    st.warning(message)
        else:
            st.info(
                "No implemented blood-pressure or temperature alert threshold "
                "was triggered by the available latest reading. This does not "
                "mean all measurements are normal or rule out a medical problem."
            )

        st.divider()
        st.subheader("Interactive Vitals Logs & Plotly Visualizations")
        df = pd.DataFrame(records)
        if validate_dataframe(df, ["date"], context="Vitals Data"):
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True, key="vitals_editor")
            st.session_state.vitals = edited.fillna("").to_dict("records")
            
            df_chart = pd.DataFrame(st.session_state.vitals)
            df_chart["date"] = pd.to_datetime(df_chart.get("date"), errors="coerce")
            df_chart = df_chart.dropna(subset=["date"]).sort_values("date")
            
            choice = st.selectbox("Select Visual Trend", ["Blood pressure", "Heart rate", "Temperature", "Weight", "Glucose"])
            fig = None
            if choice == "Blood pressure":
                cols = [x for x in ["systolic", "diastolic"] if x in df_chart.columns]
                if cols:
                    long = df_chart.melt(id_vars=["date"], value_vars=cols, var_name="measurement", value_name="value")
                    fig = px.line(long, x="date", y="value", color="measurement", markers=True, title="Blood Pressure Trend", labels={"value": "mmHg", "date": "Date"})
            else:
                mapping = {"Heart rate": ("heart_rate", "Heart Rate (bpm)"),
                           "Temperature": ("temperature", "Temperature (°C)"),
                           "Weight": ("weight", "Weight (kg)"),
                           "Glucose": ("glucose_mg_dl", "Glucose (mg/dL)")}
                column, label = mapping[choice]
                if column in df_chart.columns:
                    df_chart[column] = pd.to_numeric(df_chart[column], errors="coerce")
                    chart_df = df_chart.dropna(subset=[column])
                    if not chart_df.empty:
                        fig = px.line(chart_df, x="date", y=column, markers=True, title=f"{choice} Trend", labels={column: label, "date": "Date"})
            if fig is not None: 
                st.plotly_chart(chart_theme(fig), use_container_width=True)
            st.download_button("Download Vitals CSV", csv_bytes(st.session_state.vitals), file_name="caretrail_vitals.csv", mime="text/csv")
    else: 
        st.info("No vitals recorded yet. Pick or create a patient profile from the sidebar.")

# ------------------------- Symptom Tracker -------------------------
elif page == "Symptom Tracker":
    st.subheader("Symptom Tracker")
    with st.form("symptom_form", clear_on_submit=True):
        sd = st.date_input("Date", value=date.today())
        symptom = st.text_input("Symptom Description")
        severity = st.slider("Severity Rating (0–10)", 0, 10, 3)
        duration = st.selectbox("Duration", ["Less than an hour", "A few hours", "1 day", "Several days", "Ongoing"])
        symptom_notes = st.text_area("Additional Symptom Details")
        save_symptom = st.form_submit_button("Log Symptom", type="primary")
        
    if save_symptom:
        if not symptom.strip(): 
            st.warning("Please enter a symptom description before saving.")
        else:
            st.session_state.symptoms.append({
                "date": sd.isoformat(), "symptom": symptom.strip(),
                "severity": severity, "duration": duration, "notes": symptom_notes.strip() or "Not Specified"
            })
            notify("Symptom logged successfully.")
            st.rerun()
            
    records = current_records("symptoms")
    if records:
        df = pd.DataFrame(records)
        if validate_dataframe(df, ["symptom"], context="Symptoms Log"):
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True, key="symptoms_editor",
                                    column_config={"severity": st.column_config.NumberColumn("Severity", min_value=0, max_value=10, step=1)})
            st.session_state.symptoms = edited.fillna("Not Specified").to_dict("records")
            
            df_chart = pd.DataFrame(st.session_state.symptoms)
            df_chart["severity"] = pd.to_numeric(df_chart.get("severity"), errors="coerce")
            df_chart["date"] = pd.to_datetime(df_chart.get("date"), errors="coerce")
            df_chart = df_chart.dropna(subset=["date", "severity"])
            if not df_chart.empty:
                fig = px.line(df_chart, x="date", y="severity", color="symptom", markers=True,
                              title="Symptom Severity Over Time", labels={"date": "Date", "severity": "Severity (0–10)"})
                st.plotly_chart(chart_theme(fig), use_container_width=True)
            st.download_button("Download Symptom CSV", csv_bytes(st.session_state.symptoms), file_name="caretrail_symptoms.csv", mime="text/csv")
    else: 
        st.info("No symptoms logged yet.")

# ------------------------- Medications & Reminders -------------------------
elif page == "Medications & Reminders":
    st.subheader("Medications & Reminders")
    st.info("⚠️ Record medication details as prescribed. Omitted or missing fields are labeled 'Not Specified' rather than treated as discontinuations.")
    
    with st.form("medication_form", clear_on_submit=True):
        name = st.text_input("Medication Name")
        dose = st.text_input("Dose as Prescribed", placeholder="e.g. 10mg, 500mcg")
        frequency = st.text_input("Frequency", placeholder="e.g. Once Daily, Twice Daily")
        reminder_time = st.time_input("Reminder Time", value=time(9, 0))
        start_date = st.date_input("Start Date", value=date.today())
        med_notes = st.text_area("Instructions & Notes")
        save_med = st.form_submit_button("Add Medication Entry", type="primary")
        
    if save_med:
        if not name.strip(): 
            st.error("❌ Input Validation Error: Medication name is required.")
        else:
            st.session_state.medications.append({
                "name": name.strip(), 
                "dose": dose.strip() or "Not Specified",
                "frequency": frequency.strip() or "Not Specified", 
                "time": reminder_time.strftime("%H:%M"),
                "start_date": start_date.isoformat(), 
                "notes": med_notes.strip() or "Not Specified"
            })
            notify("Medication entry added.")
            st.rerun()
            
    meds = current_records("medications")
    if meds:
        df = pd.DataFrame(meds).fillna("Not Specified")
        if validate_dataframe(df, ["name"], context="Medication List"):
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True, key="medication_editor")
            st.session_state.medications = edited.fillna("Not Specified").to_dict("records")
            st.download_button("Download Medication List CSV", csv_bytes(st.session_state.medications), file_name="caretrail_medications.csv", mime="text/csv")
            
            st.subheader("iCal Calendar Reminder Export")
            for i, med in enumerate(st.session_state.medications):
                raw_time = str(med.get("time", "09:00"))
                try: 
                    hour, minute = map(int, raw_time.split(":")[:2])
                except (ValueError, AttributeError): 
                    hour, minute = 9, 0

                s_val = med.get("start_date", today_str())
                if isinstance(s_val, (date, datetime)):
                    start_day = s_val
                else:
                    try: 
                        start_day = date.fromisoformat(str(s_val))
                    except (ValueError, TypeError): 
                        start_day = date.today()

                start = datetime.combine(start_day, time(hour, minute))
                summary = re.sub(r"([,;])", r"\\\1", str(med.get("name", "Medication")))
                summary = summary.replace("\\", "\\\\").replace("\n", "\\n")
                ics = "\r\n".join([
                    "BEGIN:VCALENDAR", 
                    "VERSION:2.0", 
                    "PRODID:-//CareTrail//Medication Reminder//EN",
                    "BEGIN:VEVENT", 
                    f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                    f"DTEND:{(start+timedelta(minutes=10)).strftime('%Y%m%dT%H%M%S')}",
                    f"SUMMARY:Medication reminder - {summary}",
                    "DESCRIPTION:Take as directed by your clinician.", 
                    "END:VEVENT", 
                    "END:VCALENDAR", 
                    ""
                ])
                st.download_button(f"Download Reminder: {med.get('name', 'Medication')}", ics,
                                   file_name=f"caretrail_reminder_{i+1}.ics", mime="text/calendar", key=f"med_ics_{i}")
    else: 
        st.info("No medication records found.")

# ------------------------- Document Vault & Privacy Scrubbing -------------------------
elif page == "Document Vault":
    st.subheader("Document Vault & PII/PHI De-identification Scrubbing Tool")
    st.caption(
        "Uploaded documents may be processed on the hosted application server. "
        "Use fictional or de-identified data only."
    )
    st.warning(
        "Identifier scrubbing uses limited pattern matching for selected identifiers. "
        "It may miss names, addresses, dates of birth, medical record numbers, "
        "and other identifying information. Review the output manually; "
        "do not treat it as guaranteed de-identification."
    )
    
    uploaded = st.file_uploader("Upload Clinical Record or Report", type=["txt", "md", "csv", "pdf"])
    if uploaded is not None:
        if uploaded.size == 0:
            st.error("❌ Input Validation Error: Uploaded file is empty.")
        elif st.button("Save Document to Session", type="primary"):
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
                    extracted = f"PDF text extraction failed: {exc}"
            st.session_state.documents.append({
                "name": uploaded.name,
                "type": uploaded.type or "application/octet-stream",
                "size": len(content),
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                "text": extracted,
            })
            notify("Document added securely.")
            st.rerun()
            
    docs = current_records("documents")
    if docs:
        for i, doc in enumerate(docs):
            with st.expander(f"📄 {doc.get('name', 'Document')}"):
                st.write(f"Size: {doc.get('size', 0):,} bytes | Uploaded: {doc.get('uploaded_at', '')}")
                raw_txt = doc.get("text", "")
                if raw_txt:
                    st.text_area("Raw Extracted Content", raw_txt[:5000], height=140, key=f"doc_raw_{i}")
                    
                    if st.button(f"Scrub PII/PHI Identifier Tags", key=f"scrub_{i}"):
                        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', raw_txt)
                        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', sanitized)
                        sanitized = re.sub(r'\b\d{10}\b', '[REDACTED PHONE]', sanitized)
                        st.success("PHI Scrubbing Applied:")
                        st.text_area("De-identified Document", sanitized[:5000], height=140, key=f"doc_scrubbed_{i}")

                if st.button("Remove Document", key=f"delete_doc_{i}"):
                    st.session_state.documents.pop(i)
                    st.rerun()
    else: 
        st.info("No documents currently stored.")

# ------------------------- Visit Comparison -------------------------
elif page == "Visit Comparison":
    st.subheader("Patient Record Comparison & Discrepancy Analyzer")
    visits = current_records("visits")
    
    if len(visits) < 2:
        st.warning("⚠️ At least two visit records are required to run comparison.")
        if st.button("⚡ Click Here to Load Synthetic Patient Comparison Demo"):
            load_synthetic_demo_comparison()
            st.rerun()
    else:
        labels = [f"{v.get('date', 'N/A')} — {v.get('provider', 'Not Specified')}" for v in visits]
        col1, col2 = st.columns(2)
        idx_a = col1.selectbox("Select First Record (Record A)", range(len(visits)), format_func=lambda i: labels[i], index=0)
        idx_b = col2.selectbox("Select Second Record (Record B)", range(len(visits)), format_func=lambda i: labels[i], index=min(1, len(visits)-1))
        
        if idx_a == idx_b:
            st.warning("Please select two distinct visit records to analyze discrepancies.")
        else:
            rec_a, rec_b = visits[idx_a], visits[idx_b]
            
            comparison_data = []
            discrepancies = []
            all_keys = sorted(list(set(list(rec_a.keys()) + list(rec_b.keys()))))
            
            for k in all_keys:
                val_a = rec_a.get(k, "Not Specified")
                val_b = rec_b.get(k, "Not Specified")
                status = "Matching" if val_a == val_b else "Discrepancy Detected"
                if status != "Matching":
                    discrepancies.append(k)
                comparison_data.append({"Field": k, "Record A": val_a, "Record B": val_b, "Comparison Status": status})
                
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.subheader("Discrepancy Summary")
            if discrepancies:
                st.warning(f"Detected {len(discrepancies)} field discrepancy(ies): {', '.join(discrepancies)}")
            else:
                st.success("All fields between Record A and Record B match identically.")
                
            st.subheader("Export Comparison Report")
            r_col1, r_col2 = st.columns(2)
            
            csv_rep = comp_df.to_csv(index=False).encode("utf-8-sig")
            r_col1.download_button("📥 Download Report (CSV)", csv_rep, file_name=f"caretrail_comparison_{today_str()}.csv", mime="text/csv")
            
            txt_report = f"""CARETRAIL HEALTH RECORD COMPARISON REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Profile: {st.session_state.active_profile}

RECORD A: {labels[idx_a]}
RECORD B: {labels[idx_b]}

FIELD BREAKDOWN:
{comp_df.to_string(index=False)}

DISCREPANCIES DETECTED: {len(discrepancies)}
DISCREPANT FIELDS: {', '.join(discrepancies) if discrepancies else 'None'}

CLINICAL DISCLAIMER: This automated record comparison report is generated for tracking and verification purposes. Any dosage or treatment changes require review and confirmation by a licensed healthcare professional.
"""
            r_col2.download_button("📄 Download Report (TXT)", txt_report.encode("utf-8"), file_name=f"caretrail_comparison_{today_str()}.txt", mime="text/plain")

# ------------------------- AI Question Classifier & Evaluation -------------------------
elif page == "AI Question Classifier":
    st.subheader("AI Question Intent Classifier")
    st.caption("Module 2 Technique: Local TF-IDF Vectorizer + LinearSVC Machine Learning Pipeline.")
    
    pipeline, report_dict, cm, split_info, classes = get_trained_nlp_pipeline()
    
    t1, t2 = st.tabs([
        "Interactive Query Assistant",
        "📊 AI Model Performance & Metrics (Rubric Evaluation)",
    ])
    
    with t1:
        question = st.text_area("Enter a patient health query:", placeholder="e.g., Where can I view my recent blood pressure graph?", height=100)
        if st.button("Classify Query Intent", type="primary"):
            if not question.strip():
                st.warning("Please enter a query.")
            else:
                pred_label = pipeline.predict([question.strip()])[0]
                
                responses = {
                    "symptom_help": "Triage Guidance: Log onset time, duration, and severity score. Seek emergency medical care immediately if experiencing severe chest pain or shortness of breath.",
                    "medication_query": "Medication Protocol: Verify prescribed schedule under Medications & Reminders. Never alter prescribed dosages without consulting your clinician.",
                    "records_query": "Record Navigation: Clinical visit summaries and doctor notes are located under Health Notes & Timeline.",
                    "health_trends": "Trend Analytics: Graphical representations of vitals and glucose logs are displayed under Vitals & Analytics.",
                    "general_help": "System Guidance: CareTrail supports local PHR management, iCal reminder exports, and session backups."
                }
                
                res_text = responses.get(pred_label, "Query processed by local classifier.")
                result = {"question": question.strip(), "intent": pred_label, "response": res_text, "timestamp": datetime.now().isoformat(timespec="seconds")}
                st.session_state.classifier_result = result
                st.session_state.ai_history.append(result)
                notify("Query Classified.")

        result = st.session_state.classifier_result
        if result:
            st.divider()
            st.subheader("Classification Outcome")
            st.metric("Predicted Intent Category", result.get("intent"))
            st.write(f"**Assistant Response:** {result.get('response')}")

    with t2:
        st.markdown("### AI Intent Classifier Test-Set Metrics")
        st.caption(
            "Metrics are calculated on a stratified holdout test set. "
            "The dataset is small and manually constructed, so the scores are "
            "illustrative and should not be interpreted as evidence of clinical "
            "accuracy or real-world patient-query performance."
        )
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{split_info['test_acc']*100:.1f}%")
        m2.metric("Macro F1-Score", f"{split_info['macro_f1']:.3f}")
        m3.metric("Train Set Samples", split_info['train_size'])
        m4.metric("Test Set Samples (Unseen)", split_info['test_size'])
        
        st.divider()
        st.subheader("Detailed Classification Report")
        rep_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(rep_df, use_container_width=True)
        
        st.subheader("Unseen Test-Set Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Teal", title="Intent Confusion Matrix (Test Set)")
        st.plotly_chart(chart_theme(fig_cm), use_container_width=True)

# ------------------------- Backup & Restore -------------------------
elif page == "Data Backup & Restore":
    st.subheader("Data Backup & Restore")
    st.write("Export a complete JSON backup of current session records.")
    st.download_button("Download Session Backup (JSON)", make_backup().encode("utf-8"), file_name=f"caretrail_backup_{today_str()}.json", mime="application/json", type="primary")
    st.divider()
    st.subheader("Restore Backup")
    backup_file = st.file_uploader("Select CareTrail JSON Backup File", type=["json"], key="backup_restore")
    if st.button(
        "Restore Session Backup Data",
        disabled=backup_file is None,
    ):
        if restore_backup(backup_file):
            notify("Session data restored successfully.")
            st.rerun()

# ------------------------- Doctor Summary PDF -------------------------
elif page == "Doctor Summary PDF":
    st.subheader("Doctor Summary PDF Export")
    st.write(f"Generate a PDF summary of clinical visits, vitals logs, symptoms, and medication records for **{st.session_state.active_profile}**.")
    if st.button("Generate Doctor Summary PDF", type="primary"):
        try:
            with st.spinner("Generating document..."):
                st.session_state.generated_pdf = make_doctor_pdf()
            notify("PDF ready.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
            
    if st.session_state.generated_pdf:
        st.download_button("Download PDF Document", st.session_state.generated_pdf, file_name=f"caretrail_doctor_summary_{today_str()}.pdf", mime="application/pdf", type="primary")

# ------------------------- About & Privacy -------------------------
elif page == "About & Privacy":
    st.subheader("About CareTrail")
    st.write(
        "CareTrail is an educational Personal Health Record (PHR) prototype "
        "for organizing health records, comparing visits, visualizing selected "
        "measurements, and demonstrating text-intent classification."
    )
    st.subheader("Rubric & Module Mapping")
    st.markdown("""- **Module Mapping:** Module 2 — Healthcare Applications (Intelligent PHR Assistant).
- **AI Techniques:** TF-IDF text features with a LinearSVC intent classifier, evaluated using a stratified holdout split.
- **Rule-Based Component:** Selected blood-pressure and temperature threshold alerts.
- **Data Limitations:** The classifier uses a small, manually constructed dataset. Its evaluation does not establish clinical validity.
- **Privacy and Compliance:** CareTrail is a student prototype. It has not been independently assessed or certified as HIPAA-compliant and should not be used to store identifiable patient records.""")

st.divider()
st.caption(f"CareTrail Prototype · Active Profile: {st.session_state.active_profile} · Theme: {st.session_state.theme}")
