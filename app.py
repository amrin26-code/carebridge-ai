
import copy
import io
import json
from pathlib import Path
from datetime import date, datetime, time, timedelta

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# PDF generation: add reportlab to requirements.txt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareTrail | Health Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. SESSION STATE — SAFE INITIALIZATION
# =========================================================

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
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default)


# =========================================================
# 3. THEME
# =========================================================

theme_choice = st.session_state.theme

if theme_choice == "Dark":
    C = {
        "bg": "#101827",
        "panel": "#192538",
        "panel_alt": "#24344A",
        "text": "#F1F5F9",
        "muted": "#CBD5E1",
        "border": "#3A4A60",
        "primary": "#60A5FA",
        "primary_text": "#081426",
        "input_bg": "#223249",
        "input_text": "#FFFFFF",
        "sidebar": "#0B1220",
        "chart": "plotly_dark",
    }
else:
    C = {
        "bg": "#F3F6FB",
        "panel": "#FFFFFF",
        "panel_alt": "#E9EFF7",
        "text": "#172033",
        "muted": "#475569",
        "border": "#CFD8E6",
        "primary": "#1769AA",
        "primary_text": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "input_text": "#172033",
        "sidebar": "#16243A",
        "chart": "plotly_white",
    }

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"dark" if theme_choice == "Dark" else "light"};
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background: {C["bg"]} !important;
        color: {C["text"]} !important;
    }}

    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6, .stApp strong,
    .stMarkdown, .stMarkdown p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: {C["text"]} !important;
    }}

    [data-testid="stSidebar"] {{
        background: {C["sidebar"]} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
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

    .stApp [data-baseweb="select"] > div {{
        background: {C["input_bg"]} !important;
        border-color: {C["border"]} !important;
    }}
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] input {{
        color: {C["input_text"]} !important;
        -webkit-text-fill-color: {C["input_text"]} !important;
    }}

    [data-baseweb="popover"], [data-baseweb="menu"],
    [data-baseweb="popover"] li, [data-baseweb="menu"] li {{
        background: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
    }}

    [data-testid="stMetric"] {{
        background: {C["panel"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"] {{
        border-color: {C["border"]} !important;
    }}

    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
        background: {C["panel"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 9px !important;
    }}
    .stApp [data-testid="stButton"] button p,
    .stApp [data-testid="stDownloadButton"] button p {{
        color: {C["text"]} !important;
    }}

    .stApp button[kind="primary"] {{
        background: {C["primary"]} !important;
        color: {C["primary_text"]} !important;
        border-color: {C["primary"]} !important;
    }}
    .stApp button[kind="primary"] p {{
        color: {C["primary_text"]} !important;
    }}

    .stApp [data-testid="stTabs"] button {{
        color: {C["text"]} !important;
    }}

    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"],
    .stApp table {{
        background: {C["panel"]} !important;
        color: {C["text"]} !important;
    }}

    .stApp [data-testid="stFileUploader"] section {{
        background: {C["panel"]} !important;
        border: 1px dashed {C["border"]} !important;
    }}

    .stApp a {{ color: {C["primary"]} !important; }}
    hr {{ border-color: {C["border"]} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. GENERAL HELPERS
# =========================================================

def today_str():
    return date.today().isoformat()


def notify(message):
    """Use a toast where supported, with a visible fallback."""
    try:
        st.toast(message)
    except Exception:
        st.success(message)


def chart_theme(fig):
    fig.update_layout(
        template=C["chart"],
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig


def load_model():
    paths = [
        Path("intelliphr_intent_model.joblib"),
        Path("models/intelliphr_intent_model.joblib"),
        Path("model/intelliphr_intent_model.joblib"),
        Path("artifacts/intelliphr_intent_model.joblib"),
    ]
    for path in paths:
        if path.is_file():
            try:
                return joblib.load(path), str(path), None
            except Exception as exc:
                return None, str(path), str(exc)
    return None, None, "Model file not found in the expected locations."


def csv_bytes(records):
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8")


def current_records(key):
    records = st.session_state.get(key, [])
    return records if isinstance(records, list) else []


# =========================================================
# 5. SAMPLE PATIENT DATA
# =========================================================

def build_sample_profile(profile_name):
    """Generate deterministic fictional records for a smooth demo."""
    records_vitals = []
    records_symptoms = []

    for day_offset in range(13, -1, -1):
        d = date.today() - timedelta(days=day_offset)

        if profile_name == "Alex Morgan — Diabetes Monitoring":
            vitals = {
                "date": d.isoformat(),
                "systolic": 118 + (day_offset % 4) * 2,
                "diastolic": 76 + (day_offset % 3),
                "heart_rate": 72 + (day_offset % 5),
                "temperature": 36.6 + (day_offset % 3) * 0.1,
                "weight": round(78.4 - (13 - day_offset) * 0.04, 1),
                "glucose_mg_dl": 126 + (day_offset * 7 % 38),
            }
            symptom_name = "Fatigue"
            symptom_severity = (day_offset * 3) % 5

        elif profile_name == "Jamie Taylor — Post-Surgery Recovery":
            vitals = {
                "date": d.isoformat(),
                "systolic": 116 + (day_offset % 5) * 2,
                "diastolic": 74 + (day_offset % 4),
                "heart_rate": 76 + (day_offset % 6),
                "temperature": 36.5 + (day_offset % 4) * 0.1,
                "weight": round(65.2 - (13 - day_offset) * 0.02, 1),
                "glucose_mg_dl": None,
            }
            symptom_name = "Pain"
            symptom_severity = min(8, max(1, day_offset // 2 + 1))

        else:
            vitals = {
                "date": d.isoformat(),
                "systolic": 118 + (day_offset % 4) * 2,
                "diastolic": 75 + (day_offset % 3),
                "heart_rate": 70 + (day_offset % 6),
                "temperature": 36.6 + (day_offset % 3) * 0.1,
                "weight": round(62.0 - (13 - day_offset) * 0.02, 1),
                "glucose_mg_dl": None,
            }
            symptom_name = "Headache"
            symptom_severity = day_offset % 4

        records_vitals.append(vitals)
        records_symptoms.append({
            "date": d.isoformat(),
            "symptom": symptom_name,
            "severity": symptom_severity,
            "duration": "Sample daily log",
            "notes": "Fictional demonstration data",
        })

    if profile_name == "Alex Morgan — Diabetes Monitoring":
        diagnosis = "Fictional diabetes monitoring scenario"
        reason = "Routine diabetes follow-up"
        medications = [{
            "name": "Sample medication A",
            "dose": "Example dose — not a prescription",
            "frequency": "As directed in this fictional scenario",
            "time": "08:00",
            "start_date": today_str(),
            "notes": "Fictional demo entry. Do not use for treatment.",
        }]
    elif profile_name == "Jamie Taylor — Post-Surgery Recovery":
        diagnosis = "Fictional post-surgery recovery scenario"
        reason = "Recovery follow-up"
        medications = [{
            "name": "Sample medication B",
            "dose": "Example dose — not a prescription",
            "frequency": "As directed in this fictional scenario",
            "time": "09:00",
            "start_date": today_str(),
            "notes": "Fictional demo entry. Do not use for treatment.",
        }]
    else:
        diagnosis = "Fictional general wellness scenario"
        reason = "Routine health review"
        medications = []

    visits = [{
        "date": (date.today() - timedelta(days=7)).isoformat(),
        "provider": "Sample Community Clinic",
        "reason": reason,
        "diagnosis": diagnosis,
        "notes": "All patient details are fictional and for demonstration only.",
    }]

    notes = [{
        "date": today_str(),
        "title": "Sample care note",
        "text": "Fictional scenario data for demonstrating CareTrail features.",
    }]

    return {
        "visits": visits,
        "notes": notes,
        "vitals": records_vitals,
        "symptoms": records_symptoms,
        "medications": medications,
        "documents": [],
        "classifier_result": None,
        "visit_comparison": None,
        "ai_history": [],
    }


def load_sample_profile(profile_name):
    data = build_sample_profile(profile_name)
    for key, value in data.items():
        st.session_state[key] = copy.deepcopy(value)
    st.session_state.active_profile = profile_name


# =========================================================
# 6. BACKUP / RESTORE
# =========================================================

def make_backup():
    keys = [
        "visits", "notes", "vitals", "symptoms",
        "medications", "documents", "classifier_result",
        "visit_comparison", "ai_history", "active_profile",
    ]
    payload = {key: st.session_state.get(key) for key in keys}
    payload["exported_at"] = datetime.now().isoformat()
    payload["format_version"] = 1
    return json.dumps(payload, indent=2, ensure_ascii=False)


def restore_backup(uploaded):
    data = json.load(uploaded)
    for key in [
        "visits", "notes", "vitals", "symptoms",
        "medications", "documents", "classifier_result",
        "visit_comparison", "ai_history", "active_profile",
    ]:
        if key in data:
            st.session_state[key] = data[key]


# =========================================================
# 7. DOCTOR SUMMARY PDF
# =========================================================

def make_doctor_pdf():
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="CareTrail Doctor Summary",
        author="CareTrail Prototype",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CareTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1769AA"),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1769AA"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SmallBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    ))

    story = []
    story.append(Paragraph("CareTrail — Doctor Summary", styles["CareTitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"],
    ))
    story.append(Paragraph(
        f"Profile: {st.session_state.get('active_profile', 'Personal Records')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Important information", styles["SectionTitle"]))
    story.append(Paragraph(
        "This document summarizes information stored in a prototype. "
        "It may contain fictional sample data or user-entered information. "
        "It is not a diagnosis, prescription, or verified medical record.",
        styles["SmallBody"],
    ))

    # Latest vitals
    story.append(Paragraph("Recent Vitals", styles["SectionTitle"]))
    vitals = current_records("vitals")

    if vitals:
        df = pd.DataFrame(vitals)
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False)
        cols = [
            c for c in [
                "date", "systolic", "diastolic", "heart_rate",
                "temperature", "weight", "glucose_mg_dl"
            ] if c in df.columns
        ]
        table_data = [[str(c).replace("_", " ").title() for c in cols]]
        for _, row in df.head(10).iterrows():
            table_data.append([
                "" if pd.isna(row[c]) else str(row[c])
                for c in cols
            ])

        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD6E2")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F2F6FA")]),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No vitals records saved.", styles["Normal"]))

    # Symptoms
    story.append(Paragraph("Recent Symptoms", styles["SectionTitle"]))
    symptoms = current_records("symptoms")
    if symptoms:
        symptom_df = pd.DataFrame(symptoms)
        cols = [c for c in ["date", "symptom", "severity", "duration", "notes"]
                if c in symptom_df.columns]
        table_data = [[str(c).title() for c in cols]]
        for _, row in symptom_df.sort_values(
            "date", ascending=False
        ).head(10).iterrows():
            table_data.append([
                Paragraph(
                    str("" if pd.isna(row[c]) else row[c])[:250],
                    styles["SmallBody"],
                )
                for c in cols
            ])

        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD6E2")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No symptom entries saved.", styles["Normal"]))

    # Visits
    story.append(Paragraph("Health Visits", styles["SectionTitle"]))
    visits = current_records("visits")
    if visits:
        for visit in sorted(
            visits, key=lambda x: x.get("date", ""), reverse=True
        )[:10]:
            story.append(Paragraph(
                f"<b>{visit.get('date', '')} — "
                f"{visit.get('provider', 'Provider not entered')}</b>",
                styles["SmallBody"],
            ))
            story.append(Paragraph(
                f"Reason: {visit.get('reason', '')}<br/>"
                f"Assessment: {visit.get('diagnosis', '')}<br/>"
                f"Notes: {visit.get('notes', '')}",
                styles["SmallBody"],
            ))
            story.append(Spacer(1, 5))
    else:
        story.append(Paragraph("No visit records saved.", styles["Normal"]))

    # Medications
    story.append(Paragraph("Medication List", styles["SectionTitle"]))
    medications = current_records("medications")
    if medications:
        med_rows = [["Name", "Dose entered", "Frequency entered", "Time"]]
        for med in medications:
            med_rows.append([
                str(med.get("name", ""))[:100],
                str(med.get("dose", ""))[:100],
                str(med.get("frequency", ""))[:100],
                str(med.get("time", "")),
            ])
        table = Table(med_rows, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD6E2")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No medication entries saved.", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "For clinician review only. Values and medication details have not "
        "been independently verified. Do not use this report alone to make "
        "diagnosis or treatment decisions.",
        styles["SmallBody"],
    ))

    doc.build(story)
    return output.getvalue()


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("# CareTrail")
    st.caption("Personal health information dashboard")
    st.divider()

    selected_theme = st.radio(
        "Appearance",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="theme_picker",
    )

    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.divider()
    st.subheader("Demo patient")

    profile_options = [
        "Personal Records",
        "Alex Morgan — Diabetes Monitoring",
        "Jamie Taylor — Post-Surgery Recovery",
        "Sample General Wellness",
    ]

    selected_profile = st.selectbox(
        "Select a fictional profile",
        profile_options,
        index=profile_options.index(st.session_state.active_profile)
        if st.session_state.active_profile in profile_options else 0,
        key="profile_picker",
    )

    if st.button("Load Sample Patient Data", type="primary",
                 use_container_width=True):
        if selected_profile == "Personal Records":
            st.session_state.active_profile = selected_profile
            notify("Personal Records selected.")
        else:
            load_sample_profile(selected_profile)
            notify("Sample patient data loaded.")
        st.rerun()

    st.caption(
        "Loading a sample profile replaces the current session records. "
        "Export a backup first if you need to preserve them."
    )

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Health Notes & Timeline",
            "Vitals & Analytics",
            "Symptom Tracker",
            "Medications & Reminders",
            "Document Vault",
            "Visit Comparison",
            "AI Question Classifier",
            "Data Backup & Restore",
            "Doctor Summary PDF",
            "About & Privacy",
        ],
    )

    st.divider()
    st.caption(f"Active profile: {st.session_state.active_profile}")

    if st.button("Clear Session Data", use_container_width=True):
        for key, default in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(default)
        st.rerun()


# =========================================================
# 9. HEADER
# =========================================================

st.title("CareTrail")
st.caption("Your health information, organized in one place.")

if st.session_state.active_profile != "Personal Records":
    st.info(
        f"Demo profile active: **{st.session_state.active_profile}**. "
        "All sample patient details are fictional."
    )

st.markdown(
    f"""
    <div style="
        background:{C['panel']};
        border:1px solid {C['border']};
        border-radius:14px;
        padding:18px 20px;
        margin-bottom:16px;">
        <div style="font-size:12px;color:{C['muted']};">
            PERSONAL HEALTH DASHBOARD
        </div>
        <div style="font-size:25px;font-weight:700;color:{C['text']};">
            Your health, organized.
        </div>
        <div style="font-size:14px;color:{C['muted']};">
            Record information, review trends, and prepare for appointments.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 10. OVERVIEW
# =========================================================

if page == "Overview":
    st.subheader("Overview")

    a, b, c, d = st.columns(4)
    a.metric("Health visits", len(current_records("visits")))
    b.metric("Vitals records", len(current_records("vitals")))
    c.metric("Symptoms logged", len(current_records("symptoms")))
    d.metric("Medication entries", len(current_records("medications")))

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Recent visits")
        visits = current_records("visits")
        if visits:
            df = pd.DataFrame(visits).sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else:
            st.info("No visits recorded. Load sample data or add a visit.")

    with right:
        st.subheader("Recent symptoms")
        symptoms = current_records("symptoms")
        if symptoms:
            df = pd.DataFrame(symptoms).sort_values("date", ascending=False)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        else:
            st.info("No symptoms logged.")

    st.subheader("Vitals snapshot")
    vitals = current_records("vitals")

    if vitals:
        latest = sorted(vitals, key=lambda x: x.get("date", ""))[-1]
        a, b, c, d = st.columns(4)
        a.metric("Blood pressure",
                 f"{latest.get('systolic', '—')}/{latest.get('diastolic', '—')}")
        b.metric("Heart rate", f"{latest.get('heart_rate', '—')} bpm")
        c.metric("Temperature", f"{latest.get('temperature', '—')} °C")
        d.metric("Weight", f"{latest.get('weight', '—')} kg")

        if latest.get("glucose_mg_dl") is not None:
            st.metric("Sample glucose", f"{latest['glucose_mg_dl']} mg/dL")

        st.caption("Recorded values only; not a diagnosis or clinical interpretation.")
    else:
        st.info("Add or load vitals to populate this section.")

    st.subheader("Quick actions")
    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button("Add a health visit", use_container_width=True):
            st.session_state.quick_nav = "Health Notes & Timeline"
            st.rerun()
    with q2:
        if st.button("Review trends", use_container_width=True):
            st.session_state.quick_nav = "Vitals & Analytics"
            st.rerun()
    with q3:
        if st.button("Export doctor summary", use_container_width=True):
            st.session_state.quick_nav = "Doctor Summary PDF"
            st.rerun()


# =========================================================
# 11. HEALTH NOTES AND TIMELINE
# =========================================================

elif page == "Health Notes & Timeline":
    st.subheader("Health Notes & Timeline")

    with st.expander("Add a health visit", expanded=True):
        with st.form("visit_form", clear_on_submit=True):
            visit_date = st.date_input("Visit date", value=date.today())
            provider = st.text_input("Hospital / doctor / clinic")
            reason = st.text_input("Reason for visit")
            diagnosis = st.text_input("Diagnosis or assessment (optional)")
            notes = st.text_area("Visit notes")
            submitted = st.form_submit_button("Save visit", type="primary")

        if submitted:
            st.session_state.visits.append({
                "date": visit_date.isoformat(),
                "provider": provider.strip(),
                "reason": reason.strip(),
                "diagnosis": diagnosis.strip(),
                "notes": notes.strip(),
            })
            notify("Visit saved successfully.")

    with st.expander("Add a health note"):
        with st.form("note_form", clear_on_submit=True):
            note_date = st.date_input("Note date", value=date.today())
            title = st.text_input("Note title")
            note_text = st.text_area("Note")
            save_note = st.form_submit_button("Save note", type="primary")

        if save_note:
            if not title.strip() and not note_text.strip():
                st.warning("Enter a title or note.")
            else:
                st.session_state.notes.append({
                    "date": note_date.isoformat(),
                    "title": title.strip(),
                    "text": note_text.strip(),
                })
                notify("Note saved successfully.")

    st.divider()
    st.subheader("Visits")

    visits = current_records("visits")
    if visits:
        df = pd.DataFrame(visits)
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="visit_editor",
        )
        st.session_state.visits = edited.fillna("").to_dict("records")
        st.download_button(
            "Download visits CSV",
            csv_bytes(st.session_state.visits),
            file_name="caretrail_visits.csv",
            mime="text/csv",
        )
    else:
        st.info("No visits yet.")

    st.subheader("Notes")
    notes_list = current_records("notes")
    if notes_list:
        notes_df = pd.DataFrame(notes_list)
        edited_notes = st.data_editor(
            notes_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="notes_editor",
        )
        st.session_state.notes = edited_notes.fillna("").to_dict("records")
    else:
        st.info("No notes yet.")


# =========================================================
# 12. VITALS AND ANALYTICS
# =========================================================

elif page == "Vitals & Analytics":
    st.subheader("Vitals & Analytics")

    with st.expander("Add a measurement", expanded=True):
        with st.form("vitals_form", clear_on_submit=True):
            record_date = st.date_input("Measurement date", value=date.today())
            a, b = st.columns(2)
            systolic = a.number_input("Systolic (mmHg)", 0, 300, 120)
            diastolic = b.number_input("Diastolic (mmHg)", 0, 200, 80)
            c, d, e = st.columns(3)
            heart_rate = c.number_input("Heart rate (bpm)", 0, 300, 72)
            temperature = d.number_input("Temperature (°C)", 25.0, 45.0, 36.7, 0.1)
            weight = e.number_input("Weight (kg)", 0.0, 500.0, 60.0, 0.1)
            glucose = st.number_input(
                "Glucose (mg/dL; optional, enter 0 if not measured)",
                min_value=0, max_value=1000, value=0,
            )
            save_vitals = st.form_submit_button("Save measurements", type="primary")

        if save_vitals:
            st.session_state.vitals.append({
                "date": record_date.isoformat(),
                "systolic": systolic,
                "diastolic": diastolic,
                "heart_rate": heart_rate,
                "temperature": temperature,
                "weight": weight,
                "glucose_mg_dl": glucose if glucose > 0 else None,
            })
            notify("Vitals saved successfully.")

    vitals = current_records("vitals")

    if vitals:
        st.divider()
        st.subheader("Editable vitals table")

        vitals_df = pd.DataFrame(vitals)
        edited_vitals = st.data_editor(
            vitals_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="vitals_editor",
        )
        st.session_state.vitals = edited_vitals.fillna(
            {"date": "", "systolic": 0, "diastolic": 0,
             "heart_rate": 0, "temperature": 0, "weight": 0}
        ).to_dict("records")

        df = pd.DataFrame(st.session_state.vitals)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        measurement = st.selectbox(
            "Choose chart",
            ["Blood pressure", "Heart rate", "Temperature", "Weight", "Glucose"],
        )

        if measurement == "Blood pressure":
            chart_df = df.melt(
                id_vars=["date"],
                value_vars=["systolic", "diastolic"],
                var_name="measurement",
                value_name="value",
            )
            fig = px.line(
                chart_df, x="date", y="value", color="measurement",
                markers=True, title="Blood pressure trend",
                labels={"date": "Date", "value": "mmHg"},
            )
        else:
            mapping = {
                "Heart rate": ("heart_rate", "Heart rate (bpm)"),
                "Temperature": ("temperature", "Temperature (°C)"),
                "Weight": ("weight", "Weight (kg)"),
                "Glucose": ("glucose_mg_dl", "Glucose (mg/dL)"),
            }
            column, label = mapping[measurement]
            if column in df.columns and df[column].notna().any():
                fig = px.line(
                    df.dropna(subset=[column]),
                    x="date", y=column, markers=True,
                    title=f"{measurement} trend",
                    labels={column: label, "date": "Date"},
                )
            else:
                fig = None
                st.info("No values entered for this measurement.")

        if fig is not None:
            st.plotly_chart(chart_theme(fig), use_container_width=True)

        st.download_button(
            "Download vitals CSV",
            csv_bytes(st.session_state.vitals),
            file_name="caretrail_vitals.csv",
            mime="text/csv",
        )
    else:
        st.info("No vitals saved yet. Load a sample profile to see 14-day trends.")

    st.caption("Graphs display recorded values and are not diagnostic tools.")


# =========================================================
# 13. SYMPTOM TRACKER
# =========================================================

elif page == "Symptom Tracker":
    st.subheader("Symptom Tracker")

    with st.form("symptom_form", clear_on_submit=True):
        symptom_date = st.date_input("Date", value=date.today())
        symptom_name = st.text_input("Symptom")
        severity = st.slider("Severity (0–10)", 0, 10, 3)
        duration = st.selectbox(
            "Duration",
            ["Less than an hour", "A few hours", "1 day",
             "Several days", "Ongoing"],
        )
        symptom_notes = st.text_area("Additional notes")
        save_symptom = st.form_submit_button("Log symptom", type="primary")

    if save_symptom:
        if not symptom_name.strip():
            st.warning("Enter a symptom before saving.")
        else:
            st.session_state.symptoms.append({
                "date": symptom_date.isoformat(),
                "symptom": symptom_name.strip(),
                "severity": severity,
                "duration": duration,
                "notes": symptom_notes.strip(),
            })
            notify("Symptom logged successfully.")

    symptoms = current_records("symptoms")

    if symptoms:
        st.divider()
        st.subheader("Editable symptom table")
        symptom_df = pd.DataFrame(symptoms)
        edited = st.data_editor(
            symptom_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="symptoms_editor",
            column_config={
                "severity": st.column_config.NumberColumn(
                    "Severity", min_value=0, max_value=10, step=1
                )
            },
        )
        st.session_state.symptoms = edited.fillna("").to_dict("records")

        df = pd.DataFrame(st.session_state.symptoms)
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "severity"])

        if not df.empty:
            fig = px.line(
                df, x="date", y="severity", color="symptom",
                markers=True, title="Symptom severity over time",
                labels={"date": "Date", "severity": "Severity (0–10)"},
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

        st.download_button(
            "Download symptom CSV",
            csv_bytes(st.session_state.symptoms),
            file_name="caretrail_symptoms.csv",
            mime="text/csv",
        )
    else:
        st.info("No symptoms logged. Load a sample patient to see example trends.")

    st.caption("For severe or rapidly worsening symptoms, seek medical care.")


# =========================================================
# 14. MEDICATIONS AND REMINDERS
# =========================================================

elif page == "Medications & Reminders":
    st.subheader("Medications & Reminders")
    st.warning(
        "Enter medication information as prescribed. This app does not "
        "recommend doses or verify drug interactions."
    )

    with st.form("medication_form", clear_on_submit=True):
        name = st.text_input("Medication name")
        dose = st.text_input("Dose as prescribed")
        frequency = st.text_input("Frequency as prescribed")
        reminder_time = st.time_input("Reminder time", value=time(9, 0))
        start_date = st.date_input("Start date", value=date.today())
        med_notes = st.text_area("Notes")
        save_med = st.form_submit_button("Add medication", type="primary")

    if save_med:
        if not name.strip():
            st.warning("Enter a medication name.")
        else:
            st.session_state.medications.append({
                "name": name.strip(),
                "dose": dose.strip(),
                "frequency": frequency.strip(),
                "time": reminder_time.strftime("%H:%M"),
                "start_date": start_date.isoformat(),
                "notes": med_notes.strip(),
            })
            notify("Medication entry saved.")

    medications = current_records("medications")
    if medications:
        st.divider()
        med_df = pd.DataFrame(medications)
        edited_meds = st.data_editor(
            med_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="medication_editor",
        )
        st.session_state.medications = edited_meds.fillna("").to_dict("records")

        st.download_button(
            "Download medication list CSV",
            csv_bytes(st.session_state.medications),
            file_name="caretrail_medications.csv",
            mime="text/csv",
        )

        for i, med in enumerate(st.session_state.medications):
            reminder_time = med.get("time", "09:00")
            try:
                hour, minute = map(int, reminder_time.split(":"))
            except (ValueError, AttributeError):
                hour, minute = 9, 0

            start = datetime.combine(
                date.fromisoformat(med.get("start_date", today_str())),
                time(hour, minute),
            )
            end = start + timedelta(minutes=10)

            ics = "\r\n".join([
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "PRODID:-//CareTrail//Medication Reminder//EN",
                "BEGIN:VEVENT",
                f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:Medication reminder - {med.get('name', 'Medication')}",
                "DESCRIPTION:Follow your clinician's instructions.",
                "END:VEVENT",
                "END:VCALENDAR",
            ])

            st.download_button(
                f"Download calendar reminder: {med.get('name', 'Medication')}",
                ics,
                file_name=f"caretrail_reminder_{i + 1}.ics",
                mime="text/calendar",
                key=f"med_ics_{i}",
            )
    else:
        st.info("No medication entries. Load a sample profile to see a demo entry.")


# =========================================================
# 15. DOCUMENT VAULT
# =========================================================

elif page == "Document Vault":
    st.subheader("Document Vault")
    st.caption(
        "Files are kept in the current session only. Avoid uploading sensitive "
        "medical documents to a public demonstration."
    )

    uploaded = st.file_uploader(
        "Upload a report or note",
        type=["txt", "md", "csv", "pdf"],
    )

    if uploaded is not None and st.button("Save document", type="primary"):
        content = uploaded.getvalue()
        extracted = ""
        extension = Path(uploaded.name).suffix.lower()

        if extension in [".txt", ".md", ".csv"]:
            extracted = content.decode("utf-8", errors="replace")
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                extracted = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except Exception as exc:
                extracted = f"Text extraction failed: {exc}"

        st.session_state.documents.append({
            "name": uploaded.name,
            "type": uploaded.type or "application/octet-stream",
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "text": extracted,
            "bytes_hex": content.hex(),
        })
        notify("Document saved to this session.")

    documents = current_records("documents")
    if documents:
        for index, doc in enumerate(documents):
            with st.expander(doc.get("name", "Document")):
                st.write(f"File size: {doc.get('size', 0):,} bytes")
                st.write(f"Uploaded: {doc.get('uploaded_at', '')}")
                if doc.get("text"):
                    st.text_area(
                        "Extracted text",
                        doc["text"][:20000],
                        height=180,
                        key=f"document_text_{index}",
                    )

                try:
                    original = bytes.fromhex(doc.get("bytes_hex", ""))
                    st.download_button(
                        "Download original",
                        original,
                        file_name=doc.get("name", "document"),
                        mime=doc.get("type", "application/octet-stream"),
                        key=f"download_document_{index}",
                    )
                except ValueError:
                    st.warning("Original file bytes could not be restored.")

                if st.button("Delete document", key=f"delete_document_{index}"):
                    st.session_state.documents.pop(index)
                    st.rerun()
    else:
        st.info("No documents uploaded in this session.")


# =========================================================
# 16. VISIT COMPARISON
# =========================================================

elif page == "Visit Comparison":
    st.subheader("Visit Comparison")
    visits = current_records("visits")

    if len(visits) < 2:
        st.info("Add at least two visits or load a sample profile and add another visit.")
    else:
        labels = [
            f"{v.get('date', '')} — {v.get('provider', 'Provider not entered')}"
            for v in visits
        ]
        a, b = st.columns(2)
        index_a = a.selectbox("First visit", range(len(visits)),
                              format_func=lambda i: labels[i], key="compare_a")
        index_b = b.selectbox("Second visit", range(len(visits)),
                              index=1, format_func=lambda i: labels[i],
                              key="compare_b")

        if index_a == index_b:
            st.warning("Select two different visits.")
        else:
            visit_a, visit_b = visits[index_a], visits[index_b]
            comparison = pd.DataFrame([
                {"Field": field, "First visit": visit_a.get(field, ""),
                 "Second visit": visit_b.get(field, "")}
                for field in ["date", "provider", "reason", "diagnosis", "notes"]
            ])
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.download_button(
                "Download comparison CSV",
                csv_bytes(comparison.to_dict("records")),
                file_name="caretrail_visit_comparison.csv",
                mime="text/csv",
            )


# =========================================================
# 17. AI CLASSIFIER AND OFFLINE DEMO MODE
# =========================================================

elif page == "AI Question Classifier":
    st.subheader("AI Question Classifier")

    st.toggle(
        "Offline AI Demo Mode",
        key="ai_demo_mode",
        help="Uses predefined responses. No external LLM API is called.",
    )

    if st.session_state.ai_demo_mode:
        st.info(
            "Offline demo mode is active. Responses below are predefined "
            "examples and are not generated by a language model."
        )
    else:
        st.info(
            "Your trained intent classifier can still work locally. "
            "No OpenAI or Claude API is connected in this version."
        )

    question = st.text_area(
        "Enter a health-related question",
        placeholder="Example: How do I organize my health records?",
        height=110,
    )

    if st.button("Analyze question", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif st.session_state.ai_demo_mode:
            q = question.lower()

            if any(word in q for word in ["symptom", "pain", "headache", "fever"]):
                intent = "symptom_help"
                response = (
                    "Demo response: Record the symptom, when it started, "
                    "its severity, and any associated changes. This example "
                    "does not assess medical urgency or provide a diagnosis."
                )
            elif any(word in q for word in ["medicine", "medication", "dose", "drug"]):
                intent = "medication_query"
                response = (
                    "Demo response: Check the medication instructions from "
                    "your clinician or pharmacist. This prototype does not "
                    "verify doses or drug interactions."
                )
            elif any(word in q for word in ["record", "report", "document", "visit"]):
                intent = "records_query"
                response = (
                    "Demo response: You can use Health Notes & Timeline and "
                    "Document Vault to organize records in the current session."
                )
            elif any(word in q for word in ["trend", "history", "change", "chart"]):
                intent = "health_trends"
                response = (
                    "Demo response: Open Vitals & Analytics to view trends "
                    "from the measurements saved in this session."
                )
            else:
                intent = "general_help"
                response = (
                    "Demo response: CareTrail helps organize health notes, "
                    "visits, symptoms, measurements and medication entries."
                )

            result = {
                "question": question.strip(),
                "intent": intent,
                "response": response,
                "mode": "Offline mock",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
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
                            last_step = list(model.named_steps.values())[-1]
                            classes = getattr(last_step, "classes_", None)
                        if classes is not None:
                            scores = {
                                str(cls): float(prob)
                                for cls, prob in zip(classes, probabilities)
                            }

                    result = {
                        "question": question.strip(),
                        "intent": prediction,
                        "response": (
                            "The trained model classified the intent of your text. "
                            "It does not provide a diagnosis or treatment plan."
                        ),
                        "mode": f"Trained classifier ({path})",
                        "scores": scores,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
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

        if result.get("scores"):
            scores_df = pd.DataFrame([
                {"Intent": key, "Probability": value}
                for key, value in result["scores"].items()
            ]).sort_values("Probability", ascending=False)

            fig = px.bar(
                scores_df, x="Intent", y="Probability",
                title="Classifier scores", range_y=[0, 1],
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)

    with st.expander("Previous classifier/demo queries"):
        history = st.session_state.ai_history
        if history:
            st.dataframe(
                pd.DataFrame(history).drop(columns=["scores"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No queries yet.")


# =========================================================
# 18. BACKUP AND RESTORE
# =========================================================

elif page == "Data Backup & Restore":
    st.subheader("Data Backup & Restore")
    st.write(
        "Download a JSON backup of current session records. "
        "The file may include personal notes and uploaded document contents."
    )

    st.download_button(
        "Download complete JSON backup",
        make_backup().encode("utf-8"),
        file_name=f"caretrail_backup_{today_str()}.json",
        mime="application/json",
        type="primary",
    )

    st.divider()
    st.subheader("Restore a backup")

    backup_file = st.file_uploader(
        "Select a CareTrail JSON file",
        type=["json"],
        key="backup_restore",
    )

    if st.button("Restore backup", disabled=backup_file is None):
        try:
            with st.spinner("Restoring session records..."):
                restore_backup(backup_file)
            notify("Backup restored.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore the backup: {exc}")


# =========================================================
# 19. DOCTOR SUMMARY PDF
# =========================================================

elif page == "Doctor Summary PDF":
    st.subheader("Doctor Summary PDF")

    st.write(
        "Generate a formatted summary of saved visits, recent vitals, "
        "symptoms and medication entries."
    )

    st.warning(
        "Review the report before sharing it. The report may contain fictional "
        "demo values or unverified user-entered data."
    )

    if st.button("Generate Doctor Summary PDF", type="primary"):
        try:
            with st.spinner("Preparing the PDF report..."):
                pdf_bytes = make_doctor_pdf()
            st.session_state.generated_pdf = pdf_bytes
            notify("Doctor summary PDF generated.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    if st.session_state.get("generated_pdf"):
        st.download_button(
            "Download Doctor Summary PDF",
            data=st.session_state.generated_pdf,
            file_name=f"caretrail_doctor_summary_{today_str()}.pdf",
            mime="application/pdf",
            type="primary",
        )

    st.caption(
        "The report is an organizational summary, not a clinical interpretation."
    )


# =========================================================
# 20. ABOUT AND PRIVACY
# =========================================================

elif page == "About & Privacy":
    st.subheader("About CareTrail")

    st.write(
        "CareTrail is a healthcare information prototype for demonstrating "
        "health record organization, tracking, visualization and text-intent "
        "classification."
    )

    st.subheader("Current capabilities")
    st.markdown(
        """
        - Fictional sample patient profiles and 14-day demonstration data
        - Health visit and note tracking
        - Editable vitals and symptom tables
        - Interactive charts
        - Medication list and calendar reminder export
        - Document upload and basic text extraction
        - Offline mock AI responses and optional trained intent classification
        - JSON backup and restore
        - Doctor Summary PDF export
        - Light and Dark themes
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - This is not a diagnostic or treatment system.
        - Mock AI responses are predefined examples, not LLM-generated advice.
        - No real-time drug interaction service is connected.
        - No hospital record, wearable, or pharmacy integration is enabled.
        - Session data may be lost when the app restarts.
        - This prototype should not be represented as a production-secure
          medical records system or as independently verified HIPAA compliance.
        """
    )

    st.subheader("Privacy")
    st.write(
        "Use fictional or de-identified data for public demonstrations. "
        "Protect downloaded backups and avoid storing sensitive medical "
        "information in this prototype."
    )


# =========================================================
# 21. FOOTER
# =========================================================

st.divider()
st.caption(
    f"CareTrail prototype · {theme_choice} mode · "
    f"{datetime.now().strftime('%d %b %Y, %H:%M')}"
)
