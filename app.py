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
    
    /* Core App Containers */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
        background-color: {C["bg"]} !important;
        color: {C["text"]} !important;
    }}
    
    /* Headers & Navigation Sidebar */
    [data-testid="stHeader"] {{ background: {C["bg"]} !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {C["sidebar"]} !important;
        color: {C["sidebar_text"]} !important;
    }}
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: {C["sidebar_text"]} !important;
    }}

    /* Global Typography Fixes */
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

    /* Metrics & Custom Panels */
    [data-testid="stMetric"] {{
        background: {C["panel"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        box-shadow: 0 2px 8px rgba(20, 50, 70, 0.04);
    }}
    [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {C["panel"]} !important;
        border-color: {C["border"]} !important;
        border-radius: 12px !important;
    }}

    /* Form Inputs, Text Areas, & Placeholders */
    .stApp input, .stApp textarea,
    .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="textarea"] textarea {{
        background-color: {C["input_bg"]} !important;
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

    /* Dropdowns, Selectboxes, & BaseWeb Popover Overrides */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="popover"], .stApp [data-baseweb="menu"],
    [data-baseweb="popover"] ul, [data-baseweb="menu"] ul {{
        background-color: {C["input_bg"]} !important;
        border-color: {C["border"]} !important;
        color: {C["input_text"]} !important;
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

    /* Radio Controls & Sliders */
    .stApp [data-testid="stRadioButtonGroup"] p,
    .stApp [data-testid="stRadioButtonGroup"] label {{
        color: {C["text"]} !important;
    }}

    /* Buttons */
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 9px !important;
        transition: border-color .15s ease, background .15s ease;
    }}
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {{
        border-color: {C["accent"]} !important;
    }}
    .stApp [data-testid="stButton"] button p,
    .stApp [data-testid="stDownloadButton"] button p {{
        color: {C["text"]} !important;
    }}
    .stApp button[kind="primary"] {{
        background-color: {C["accent"]} !important;
        border-color: {C["accent"]} !important;
        color: {C["accent_text"]} !important;
    }}
    .stApp button[kind="primary"] p {{
        color: {C["accent_text"]} !important;
    }}

    /* Tabs & Data Tables */
    .stApp [data-testid="stTabs"] button {{ color: {C["text"]} !important; }}
    .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"],
    .stApp table, [data-testid="stDataEditor"] {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
        border-color: {C["border"]} !important;
    }}

    /* File Uploader & Alerts */
    .stApp [data-testid="stFileUploader"] section {{
        background-color: {C["panel"]} !important;
        border: 1px dashed {C["border"]} !important;
        border-radius: 12px !important;
    }}
    .stApp [data-testid="stAlert"] {{
        border-radius: 10px !important;
    }}

    .stApp a {{ color: {C["accent"]} !important; }}
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
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font={"color": C["text"]},
        margin={"l": 18, "r": 18, "t": 50, "b": 18},
        legend={"font": {"color": C["text"]}},
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig

def load_model():
    candidates = [
        Path("intelliphr_intent_model.joblib"),
        Path("models/intelliphr_intent_model.joblib"),
        Path("model/intelliphr_intent_model.joblib"),
        Path("artifacts/intelliphr_intent_model.joblib"),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return joblib.load(path), str(path), None
            except Exception as exc:
                return None, str(path), str(exc)
    return None, None, "Model file not found. Add intelliphr_intent_model.joblib to the repository root or a models/ folder."

def build_sample_profile(profile):
    vitals, symptoms = [], []
    for offset in range(13, -1, -1):
        d = date.today() - timedelta(days=offset)
        if profile == "Alex Morgan — Diabetes Monitoring":
            v = {"date": d.isoformat(), "systolic": 118 + (offset % 4) * 2,
                 "diastolic": 76 + (offset % 3), "heart_rate": 72 + (offset % 5),
                 "temperature": round(36.6 + (offset % 3) * .1, 1),
                 "weight": round(78.4 - (13-offset) * .04, 1),
                 "glucose_mg_dl": 126 + (offset * 7 % 38)}
            sym, severity = "Fatigue", (offset * 3) % 6
        elif profile == "Jamie Taylor — Post-Surgery Recovery":
            v = {"date": d.isoformat(), "systolic": 116 + (offset % 5) * 2,
                 "diastolic": 74 + (offset % 4), "heart_rate": 76 + (offset % 6),
                 "temperature": round(36.5 + (offset % 4) * .1, 1),
                 "weight": round(65.2 - (13-offset) * .02, 1),
                 "glucose_mg_dl": None}
            sym, severity = "Pain", min(8, max(1, offset // 2 + 1))
        else:
            v = {"date": d.isoformat(), "systolic": 118 + (offset % 4) * 2,
                 "diastolic": 75 + (offset % 3), "heart_rate": 70 + (offset % 6),
                 "temperature": round(36.6 + (offset % 3) * .1, 1),
                 "weight": round(62.0 - (13-offset) * .02, 1),
                 "glucose_mg_dl": None}
            sym, severity = "Headache", offset % 4
        vitals.append(v)
        symptoms.append({"date": d.isoformat(), "symptom": sym, "severity": severity,
                         "duration": "Sample daily log",
                         "notes": "Fictional demonstration data"})
    if profile == "Alex Morgan — Diabetes Monitoring":
        reason, diagnosis, med_name = "Routine diabetes follow-up", "Fictional diabetes monitoring scenario", "Sample medication A"
    elif profile == "Jamie Taylor — Post-Surgery Recovery":
        reason, diagnosis, med_name = "Recovery follow-up", "Fictional post-surgery recovery scenario", "Sample medication B"
    else:
        reason, diagnosis, med_name = "Routine health review", "Fictional general wellness scenario", None
    medications = []
    if med_name:
        medications = [{"name": med_name, "dose": "Example only — not a prescription",
                        "frequency": "As directed in this fictional scenario", "time": "09:00",
                        "start_date": today_str(),
                        "notes": "Fictional demo entry. Do not use for treatment."}]
    return {
        "visits": [{"date": (date.today()-timedelta(days=7)).isoformat(),
                    "provider": "Sample Community Clinic", "reason": reason,
                    "diagnosis": diagnosis,
                    "notes": "Fictional demonstration data only."}],
        "notes": [{"date": today_str(), "title": "Sample care note",
                   "text": "Fictional scenario data for demonstrating CareTrail."}],
        "vitals": vitals, "symptoms": symptoms, "medications": medications,
        "documents": [], "classifier_result": None, "visit_comparison": None,
        "ai_history": [], "generated_pdf": None,
    }

def load_sample_profile(profile):
    for key, value in build_sample_profile(profile).items():
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
    expected = ["visits", "notes", "vitals", "symptoms", "medications",
                "documents", "ai_history"]
    for key in expected:
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"Invalid backup: '{key}' must be a list.")
    for key in expected + ["classifier_result", "visit_comparison", "active_profile"]:
        if key in data:
            st.session_state[key] = data[key]

def safe_text(value, limit=2000):
    """Escape user-supplied content before putting it into ReportLab Paragraphs."""
    return escape(str(value if value is not None else ""))[:limit].replace("\n", "<br/>")

def make_doctor_pdf():
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                            topMargin=16*mm, bottomMargin=16*mm,
                            title="CareTrail Doctor Summary", author="CareTrail Prototype")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CareTitle", parent=styles["Title"], fontSize=20,
                              leading=24, alignment=TA_CENTER,
                              textColor=colors.HexColor("#087F8C"), spaceAfter=10))
    styles.add(ParagraphStyle(name="CareSection", parent=styles["Heading2"], fontSize=12,
                              leading=15, textColor=colors.HexColor("#087F8C"),
                              spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="CareSmall", parent=styles["BodyText"],
                              fontSize=7.5, leading=10))
    story = [
        Paragraph("CareTrail — Doctor Summary", styles["CareTitle"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["Normal"]),
        Paragraph(f"Profile: {safe_text(st.session_state.get('active_profile', 'Personal Records'))}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Important information", styles["CareSection"]),
        Paragraph("This report summarizes prototype records. Entries may be fictional or user-entered and are not independently verified. This is not a diagnosis or prescription.", styles["CareSmall"]),
    ]
    def add_table_section(title, rows, widths=None):
        story.append(Paragraph(title, styles["CareSection"]))
        if not rows:
            story.append(Paragraph("No records saved.", styles["Normal"]))
            return
        table = Table(rows, repeatRows=1, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087F8C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CFDDE5")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F7F8")]),
        ]))
        story.append(table)
    vitals = current_records("vitals")
    vrows = [["Date", "Systolic", "Diastolic", "Heart rate", "Temp °C", "Weight", "Glucose"]]
    for r in sorted(vitals, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
        vrows.append([safe_text(r.get(k, "")) for k in
                      ["date", "systolic", "diastolic", "heart_rate", "temperature", "weight", "glucose_mg_dl"]])
    add_table_section("Recent vitals", vrows if vitals else [])
    symptoms = current_records("symptoms")
    srows = [["Date", "Symptom", "Severity", "Duration", "Notes"]]
    for r in sorted(symptoms, key=lambda x: str(x.get("date", "")), reverse=True)[:10]:
        srows.append([Paragraph(safe_text(r.get(k, ""), 300), styles["CareSmall"])
                      for k in ["date", "symptom", "severity", "duration", "notes"]])
    add_table_section("Recent symptoms", srows if symptoms else [])
    visits = current_records("visits")
    story.append(Paragraph("Health visits", styles["CareSection"]))
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
        mrows.append([Paragraph(safe_text(r.get(k, ""), 150), styles["CareSmall"])
                      for k in ["name", "dose", "frequency", "time"]])
    add_table_section("Medication list", mrows if meds else [])
    story += [Spacer(1, 10), Paragraph(
        "For clinician review only. Review all details before sharing. Do not use this report alone for diagnosis or treatment decisions.",
        styles["CareSmall"])]
    doc.build(story)
    return output.getvalue()

# ------------------------- Sidebar -------------------------
NAV = [
    "Overview", "Health Notes & Timeline", "Vitals & Analytics", "Symptom Tracker",
    "Medications & Reminders", "Document Vault", "Visit Comparison",
    "AI Question Classifier", "Data Backup & Restore", "Doctor Summary PDF",
    "About & Privacy",
]
with st.sidebar:
    st.markdown("## CareTrail")
    st.caption("Personal health information dashboard")
    st.divider()
    chosen_theme = st.radio("Appearance", ["Light", "Dark"],
                            index=0 if st.session_state.theme == "Light" else 1,
                            horizontal=True, key="theme_picker")
    if chosen_theme != st.session_state.theme:
        st.session_state.theme = chosen_theme
        st.rerun()
    st.divider()
    st.subheader("Demo patient")
    profiles = ["Personal Records", "Alex Morgan — Diabetes Monitoring",
                "Jamie Taylor — Post-Surgery Recovery", "Sample General Wellness"]
    current_profile = st.session_state.active_profile
    profile_index = profiles.index(current_profile) if current_profile in profiles else 0
    chosen_profile = st.selectbox("Select a fictional profile", profiles, index=profile_index,
                                  key="profile_picker")
    if st.button("Load Sample Patient Data", type="primary", use_container_width=True):
        if chosen_profile == "Personal Records":
            st.session_state.active_profile = chosen_profile
            notify("Personal Records selected.")
        else:
            load_sample_profile(chosen_profile)
            notify("Sample patient data loaded.")
        st.rerun()
    st.caption("Loading sample data replaces current records. Download a backup first if needed.")
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
    st.caption(f"Active profile: {st.session_state.active_profile}")
    if st.button("Clear Session Data", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(value)
        st.rerun()

# ------------------------- Page header -------------------------
st.title("CareTrail")
st.caption("Your health information, organized in one place.")
if st.session_state.active_profile != "Personal Records":
    st.info(f"Demo profile active: **{st.session_state.active_profile}**. All sample patient details are fictional.")
st.markdown(
    f"""<div style="background:{C['panel']};border:1px solid {C['border']};
    border-radius:16px;padding:18px 20px;margin-bottom:16px;">
    <div style="font-size:11px;letter-spacing:1.2px;font-weight:700;color:{C['accent']};">
    PERSONAL HEALTH DASHBOARD</div>
    <div style="font-size:25px;font-weight:750;color:{C['text']};margin:4px 0;">
    Your health, organized.</div>
    <div style="font-size:14px;color:{C['muted']};">
    Record information, review trends, and prepare for appointments.</div></div>""",
    unsafe_allow_html=True,
)

# ------------------------- Overview -------------------------
if page == "Overview":
    st.subheader("Overview")
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Health visits", "Vitals records", "Symptoms logged", "Medication entries"],
                               ["visits", "vitals", "symptoms", "medications"]):
        col.metric(label, len(current_records(key)))
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recent visits")
        data = current_records("visits")
        if data:
            df = pd.DataFrame(data)
            if "date" in df: df = df.sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else: st.info("No visits recorded. Load sample data or add a visit.")
    with right:
        st.subheader("Recent symptoms")
        data = current_records("symptoms")
        if data:
            df = pd.DataFrame(data)
            if "date" in df: df = df.sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else: st.info("No symptoms logged.")
    st.subheader("Vitals snapshot")
    vitals = current_records("vitals")
    if vitals:
        latest = sorted(vitals, key=lambda x: str(x.get("date", "")))[-1]
        cols = st.columns(4)
        cols[0].metric("Blood pressure", f"{latest.get('systolic', '—')}/{latest.get('diastolic', '—')}")
        cols[1].metric("Heart rate", f"{latest.get('heart_rate', '—')} bpm")
        cols[2].metric("Temperature", f"{latest.get('temperature', '—')} °C")
        cols[3].metric("Weight", f"{latest.get('weight', '—')} kg")
        if latest.get("glucose_mg_dl") is not None:
            st.metric("Recorded glucose", f"{latest['glucose_mg_dl']} mg/dL")
        st.caption("Recorded values only; not a diagnosis or clinical interpretation.")
    else: st.info("Add or load vitals to populate this section.")
    st.subheader("Quick actions")
    a, b, c = st.columns(3)
    for col, label, destination in [
        (a, "Add a health visit", "Health Notes & Timeline"),
        (b, "Review trends", "Vitals & Analytics"),
        (c, "Export doctor summary", "Doctor Summary PDF"),
    ]:
        if col.button(label, use_container_width=True):
            st.session_state.quick_nav = destination
            st.session_state.page = destination
            st.rerun()

# ------------------------- Health notes & timeline -------------------------
elif page == "Health Notes & Timeline":
    st.subheader("Health Notes & Timeline")
    with st.expander("Add a health visit", expanded=True):
        with st.form("visit_form", clear_on_submit=True):
            vd = st.date_input("Visit date", value=date.today())
            provider = st.text_input("Hospital / doctor / clinic")
            reason = st.text_input("Reason for visit")
            diagnosis = st.text_input("Diagnosis or assessment (optional)")
            visit_notes = st.text_area("Visit notes")
            save_visit = st.form_submit_button("Save visit", type="primary")
        if save_visit:
            st.session_state.visits.append({"date": vd.isoformat(), "provider": provider.strip(),
                "reason": reason.strip(), "diagnosis": diagnosis.strip(), "notes": visit_notes.strip()})
            notify("Visit saved successfully.")
    with st.expander("Add a health note"):
        with st.form("note_form", clear_on_submit=True):
            nd = st.date_input("Note date", value=date.today())
            nt = st.text_input("Note title")
            ntxt = st.text_area("Note")
            save_note = st.form_submit_button("Save note", type="primary")
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

# ------------------------- Vitals & analytics -------------------------
elif page == "Vitals & Analytics":
    st.subheader("Vitals & Analytics")
    with st.expander("Add a measurement", expanded=True):
        with st.form("vitals_form", clear_on_submit=True):
            vd = st.date_input("Measurement date", value=date.today())
            a, b = st.columns(2)
            sys = a.number_input("Systolic (mmHg)", 0, 300, 120)
            dia = b.number_input("Diastolic (mmHg)", 0, 200, 80)
            c, d, e = st.columns(3)
            hr = c.number_input("Heart rate (bpm)", 0, 300, 72)
            temp = d.number_input("Temperature (°C)", 25.0, 45.0, 36.7, .1)
            weight = e.number_input("Weight (kg)", 0.0, 500.0, 60.0, .1)
            glucose = st.number_input("Glucose (mg/dL; 0 = not measured)", 0, 1000, 0)
            save_v = st.form_submit_button("Save measurements", type="primary")
        if save_v:
            st.session_state.vitals.append({"date": vd.isoformat(), "systolic": sys, "diastolic": dia,
                "heart_rate": hr, "temperature": temp, "weight": weight,
                "glucose_mg_dl": glucose if glucose > 0 else None})
            notify("Vitals saved successfully.")
    records = current_records("vitals")
    if records:
        st.divider()
        st.subheader("Editable vitals table")
        edited = st.data_editor(pd.DataFrame(records), num_rows="dynamic",
                                use_container_width=True, hide_index=True, key="vitals_editor")
        st.session_state.vitals = edited.fillna("").to_dict("records")
        df = pd.DataFrame(st.session_state.vitals)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        choice = st.selectbox("Choose chart", ["Blood pressure", "Heart rate", "Temperature", "Weight", "Glucose"])
        fig = None
        if choice == "Blood pressure":
            cols = [x for x in ["systolic", "diastolic"] if x in df.columns]
            if cols:
                long = df.melt(id_vars=["date"], value_vars=cols, var_name="measurement", value_name="value")
                fig = px.line(long, x="date", y="value", color="measurement", markers=True,
                              title="Blood pressure trend", labels={"value": "mmHg", "date": "Date"})
        else:
            mapping = {"Heart rate": ("heart_rate", "Heart rate (bpm)"),
                       "Temperature": ("temperature", "Temperature (°C)"),
                       "Weight": ("weight", "Weight (kg)"),
                       "Glucose": ("glucose_mg_dl", "Glucose (mg/dL)")}
            column, label = mapping[choice]
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                chart_df = df.dropna(subset=[column])
                if not chart_df.empty:
                    fig = px.line(chart_df, x="date", y=column, markers=True, title=f"{choice} trend",
                                  labels={column: label, "date": "Date"})
            if fig is None: st.info("No values entered for this measurement.")
        if fig is not None: st.plotly_chart(chart_theme(fig), use_container_width=True)
        st.download_button("Download vitals CSV", csv_bytes(st.session_state.vitals),
                           file_name="caretrail_vitals.csv", mime="text/csv")
    else: st.info("No vitals saved yet. Load a sample profile to see 14-day trends.")
    st.caption("Graphs display recorded values and are not diagnostic tools.")

# ------------------------- Symptom tracker -------------------------
elif page == "Symptom Tracker":
    st.subheader("Symptom Tracker")
    with st.form("symptom_form", clear_on_submit=True):
        sd = st.date_input("Date", value=date.today())
        symptom = st.text_input("Symptom")
        severity = st.slider("Severity (0–10)", 0, 10, 3)
        duration = st.selectbox("Duration", ["Less than an hour", "A few hours", "1 day", "Several days", "Ongoing"])
        symptom_notes = st.text_area("Additional notes")
        save_symptom = st.form_submit_button("Log symptom", type="primary")
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
                          title="Symptom severity over time", labels={"date": "Date", "severity": "Severity (0–10)"})
            st.plotly_chart(chart_theme(fig), use_container_width=True)
        st.download_button("Download symptom CSV", csv_bytes(st.session_state.symptoms),
                           file_name="caretrail_symptoms.csv", mime="text/csv")
    else: st.info("No symptoms logged. Load a sample patient to see example trends.")
    st.caption("For severe or rapidly worsening symptoms, seek medical care.")

# ------------------------- Medications & reminders -------------------------
elif page == "Medications & Reminders":
    st.subheader("Medications & Reminders")
    st.warning("Enter medication information as prescribed. This app does not recommend doses or verify drug interactions.")
    with st.form("medication_form", clear_on_submit=True):
        name = st.text_input("Medication name")
        dose = st.text_input("Dose as prescribed")
        frequency = st.text_input("Frequency as prescribed")
        reminder_time = st.time_input("Reminder time", value=time(9, 0))
        start_date = st.date_input("Start date", value=date.today())
        med_notes = st.text_area("Notes")
        save_med = st.form_submit_button("Add medication", type="primary")
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
        st.download_button("Download medication list CSV", csv_bytes(st.session_state.medications),
                           file_name="caretrail_medications.csv", mime="text/csv")
        st.subheader("Calendar reminders")
        for i, med in enumerate(st.session_state.medications):
            raw_time = str(med.get("time", "09:00"))
            try: hour, minute = map(int, raw_time.split(":")[:2])
            except (ValueError, AttributeError): hour, minute = 9, 0
            try: start_day = date.fromisoformat(str(med.get("start_date", today_str())))
            except ValueError: start_day = date.today()
            start = datetime.combine(start_day, time(hour, minute))
            # Escape ICS special characters.
            summary = re.sub(r"([,;])", r"\\\1", str(med.get("name", "Medication")))
            summary = summary.replace("\\", "\\\\").replace("\n", "\\n")
            ics = "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CareTrail//Medication Reminder//EN",
                "BEGIN:VEVENT", f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{(start+timedelta(minutes=10)).strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:Medication reminder - {summary}",
                "DESCRIPTION:Follow your clinician's instructions.", "END:VEVENT", "END:VCALENDAR", ""])
            st.download_button(f"Download reminder: {med.get('name', 'Medication')}", ics,
                               file_name=f"caretrail_reminder_{i+1}.ics", mime="text/calendar", key=f"med_ics_{i}")
    else: st.info("No medication entries. Load a sample profile to see a demo entry.")

# ------------------------- Document vault -------------------------
elif page == "Document Vault":
    st.subheader("Document Vault")
    st.caption("Files are stored in this app session. Avoid uploading sensitive medical documents to a public demo.")
    uploaded = st.file_uploader("Upload a report or note", type=["txt", "md", "csv", "pdf"])
    if uploaded is not None and st.button("Save document", type="primary"):
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
        notify("Document saved to this session.")
    docs = current_records("documents")
    if docs:
        for i, doc in enumerate(docs):
            with st.expander(doc.get("name", "Document")):
                st.write(f"File size: {doc.get('size', 0):,} bytes")
                st.write(f"Uploaded: {doc.get('uploaded_at', '')}")
                if doc.get("text"):
                    st.text_area("Extracted text", doc["text"][:20000], height=180, key=f"document_text_{i}")
                try:
                    st.download_button("Download original", bytes.fromhex(doc.get("bytes_hex", "")),
                                       file_name=doc.get("name", "document"),
                                       mime=doc.get("type", "application/octet-stream"), key=f"download_doc_{i}")
                except ValueError:
                    st.warning("Original file bytes could not be restored.")
                if st.button("Delete document", key=f"delete_doc_{i}"):
                    st.session_state.documents.pop(i)
                    st.rerun()
    else: st.info("No documents uploaded in this session.")

# ------------------------- Visit comparison -------------------------
elif page == "Visit Comparison":
    st.subheader("Visit Comparison")
    visits = current_records("visits")
    if len(visits) < 2:
        st.info("Add at least two visits or load a sample profile and add another visit.")
    else:
        labels = [f"{v.get('date', '')} — {v.get('provider', 'Provider not entered')}" for v in visits]
        a, b = st.columns(2)
        ia = a.selectbox("First visit", range(len(visits)), format_func=lambda i: labels[i], key="compare_a")
        ib = b.selectbox("Second visit", range(len(visits)), index=1, format_func=lambda i: labels[i], key="compare_b")
        if ia == ib: st.warning("Select two different visits.")
        else:
            va, vb = visits[ia], visits[ib]
            comparison = pd.DataFrame([{"Field": field, "First visit": va.get(field, ""),
                                        "Second visit": vb.get(field, "")}
                                       for field in ["date", "provider", "reason", "diagnosis", "notes"]])
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.download_button("Download comparison CSV", csv_bytes(comparison.to_dict("records")),
                               file_name="caretrail_visit_comparison.csv", mime="text/csv")

# ------------------------- AI classifier -------------------------
elif page == "AI Question Classifier":
    st.subheader("AI Question Classifier")
    st.toggle("Offline AI Demo Mode", key="ai_demo_mode",
              help="Uses predefined responses. No external LLM API is called.")
    if st.session_state.ai_demo_mode:
        st.info("Offline demo mode uses predefined examples, not a language model.")
    else:
        st.info("The local intent classifier can run if its model file is available. No external LLM is connected.")
    question = st.text_area("Enter a health-related question",
                            placeholder="Example: How do I organize my health records?", height=110)
    if st.button("Analyze question", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif st.session_state.ai_demo_mode:
            q = question.lower()
            if any(w in q for w in ["symptom", "pain", "headache", "fever"]):
                intent, response = "symptom_help", "Demo response: Record the symptom, when it started, its severity, and associated changes. This does not assess urgency or diagnose."
            elif any(w in q for w in ["medicine", "medication", "dose", "drug"]):
                intent, response = "medication_query", "Demo response: Check instructions from your clinician or pharmacist. This prototype does not verify doses or drug interactions."
            elif any(w in q for w in ["record", "report", "document", "visit"]):
                intent, response = "records_query", "Demo response: Use Health Notes & Timeline and Document Vault to organize information in this session."
            elif any(w in q for w in ["trend", "history", "change", "chart"]):
                intent, response = "health_trends", "Demo response: Open Vitals & Analytics to view trends from measurements saved in this session."
            else:
                intent, response = "general_help", "Demo response: CareTrail organizes health notes, visits, symptoms, measurements and medication entries."
            result = {"question": question.strip(), "intent": intent, "response": response,
                      "mode": "Offline mock", "timestamp": datetime.now().isoformat(timespec="seconds")}
            st.session_state.classifier_result = result
            st.session_state.ai_history.append(result)
            notify("Demo response generated.")
        else:
            with st.spinner("Loading the trained classifier..."):
                model, path, error = load_model()
            if model is None:
                st.error("The trained classifier could not be loaded.")
                st.code(error or "Model unavailable.")
            else:
                try:
                    prediction = str(model.predict([question.strip()])[0])
                    scores = {}
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba([question.strip()])[0]
                        classes = getattr(model, "classes_", None)
                        if classes is None and hasattr(model, "named_steps"):
                            final_step = list(model.named_steps.values())[-1]
                            classes = getattr(final_step, "classes_", None)
                        if classes is not None:
                            scores = {str(k): float(v) for k, v in zip(classes, probabilities)}
                    result = {"question": question.strip(), "intent": prediction,
                              "response": "The model classified text intent only; it does not diagnose or provide treatment.",
                              "mode": f"Trained classifier ({path})", "scores": scores,
                              "timestamp": datetime.now().isoformat(timespec="seconds")}
                    st.session_state.classifier_result = result
                    st.session_state.ai_history.append(result)
                    notify("Question classified.")
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")
    result = st.session_state.classifier_result
    if result:
        st.divider()
        st.subheader("Result")
        st.metric("Predicted intent", result.get("intent", "Unknown"))
        st.write(result.get("response", ""))
        st.caption(f"Mode: {result.get('mode', 'Unknown')}")
        scores = result.get("scores", {})
        if scores:
            df = pd.DataFrame([{"Intent": k, "Probability": v} for k, v in scores.items()]).sort_values("Probability", ascending=False)
            fig = px.bar(df, x="Intent", y="Probability", title="Classifier scores", range_y=[0, 1])
            st.plotly_chart(chart_theme(fig), use_container_width=True)
    with st.expander("Previous classifier/demo queries"):
        if st.session_state.ai_history:
            st.dataframe(pd.DataFrame(st.session_state.ai_history).drop(columns=["scores"], errors="ignore"),
                         use_container_width=True, hide_index=True)
        else: st.caption("No queries yet.")

# ------------------------- Backup and restore -------------------------
elif page == "Data Backup & Restore":
    st.subheader("Data Backup & Restore")
    st.write("Download a JSON backup of current session records. It may include personal notes and uploaded document contents.")
    st.download_button("Download complete JSON backup", make_backup().encode("utf-8"),
                       file_name=f"caretrail_backup_{today_str()}.json", mime="application/json", type="primary")
    st.divider()
    st.subheader("Restore a backup")
    backup_file = st.file_uploader("Select a CareTrail JSON file", type=["json"], key="backup_restore")
    if st.button("Restore backup", disabled=backup_file is None):
        try:
            with st.spinner("Restoring session records..."):
                restore_backup(backup_file)
            notify("Backup restored.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore backup: {exc}")

# ------------------------- Doctor summary PDF -------------------------
elif page == "Doctor Summary PDF":
    st.subheader("Doctor Summary PDF")
    st.write("Generate a formatted summary of visits, recent vitals, symptoms and medication entries.")
    st.warning("Review the report before sharing. It may contain fictional demo values or unverified user-entered data.")
    if st.button("Generate Doctor Summary PDF", type="primary"):
        try:
            with st.spinner("Preparing the PDF report..."):
                st.session_state.generated_pdf = make_doctor_pdf()
            notify("Doctor summary PDF generated.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
    if st.session_state.generated_pdf:
        st.download_button("Download Doctor Summary PDF", st.session_state.generated_pdf,
                           file_name=f"caretrail_doctor_summary_{today_str()}.pdf",
                           mime="application/pdf", type="primary")
    st.caption("The report is an organizational summary, not a clinical interpretation.")

# ------------------------- About & privacy -------------------------
elif page == "About & Privacy":
    st.subheader("About CareTrail")
    st.write("CareTrail is a healthcare information prototype for organizing records, tracking measurements, visualizing trends and classifying text intent.")
    st.subheader("Current capabilities")
    st.markdown("""
    - Fictional sample profiles with 14 days of demonstration data
    - Health visit and note tracking
    - Editable vitals and symptom tables with interactive charts
    - Medication list and calendar reminder export
    - Document upload, text extraction and original-file download
    - Offline mock responses and optional trained intent classifier
    - JSON backup/restore and CSV exports
    - Doctor Summary PDF export
    - Light and Dark themes
    """)
    st.subheader("Limitations")
    st.markdown("""
    - Not a diagnostic or treatment system.
    - Mock responses are predefined, not generated by an LLM.
    - No real drug-interaction service, hospital record, wearable or pharmacy integration is connected.
    - Session data may be lost when the app restarts.
    - This prototype should not be represented as a production-secure medical-record system or as independently verified HIPAA compliance.
    """)
    st.subheader("Privacy")
    st.write("Use fictional or de-identified data for public demonstrations. Protect downloaded backups and avoid storing sensitive medical information in this prototype.")

st.divider()
st.caption(f"CareTrail prototype · {st.session_state.theme} mode · {datetime.now().strftime('%d %b %Y, %H:%M')}")
