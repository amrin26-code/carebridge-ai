
import io
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path
from datetime import date, datetime, time, timedelta
from typing import Any


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareTrail | Health Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. THEME SETTINGS
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme_choice = st.session_state.get("theme", "Light")

if theme_choice == "Dark":
    C = {
        "bg": "#101827",
        "panel": "#192538",
        "panel_alt": "#223249",
        "text": "#F1F5F9",
        "muted": "#CBD5E1",
        "border": "#3A4A60",
        "primary": "#60A5FA",
        "button_text": "#071426",
        "input_bg": "#223249",
        "input_text": "#FFFFFF",
        "sidebar": "#0B1220",
        "success_bg": "#12352D",
        "warning_bg": "#423317",
        "danger_bg": "#451F29",
        "chart_template": "plotly_dark",
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
        "button_text": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "input_text": "#172033",
        "sidebar": "#16243A",
        "success_bg": "#E4F5EB",
        "warning_bg": "#FFF4D6",
        "danger_bg": "#FCE8E8",
        "chart_template": "plotly_white",
    }

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"dark" if theme_choice == "Dark" else "light"};
    }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background-color: {C["bg"]} !important;
        color: {C["text"]} !important;
    }}

    /* Main text */
    .stApp p, .stApp span, .stApp label,
    .stApp li, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5,
    .stApp h6, .stApp strong, .stApp em,
    .stApp small, .stMarkdown, .stMarkdown p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: {C["text"]} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {C["sidebar"]} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{
        background-color: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
        -webkit-text-fill-color: {C["input_text"]} !important;
    }}

    /* Inputs and text areas */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="textarea"] textarea {{
        background-color: {C["input_bg"]} !important;
        color: {C["input_text"]} !important;
        -webkit-text-fill-color: {C["input_text"]} !important;
        caret-color: {C["input_text"]} !important;
        border-color: {C["border"]} !important;
    }}

    .stApp input::placeholder,
    .stApp textarea::placeholder {{
        color: {C["muted"]} !important;
        -webkit-text-fill-color: {C["muted"]} !important;
        opacity: 1 !important;
    }}

    /* Dropdowns */
    .stApp [data-baseweb="select"] > div {{
        background-color: {C["input_bg"]} !important;
        border-color: {C["border"]} !important;
    }}
    .stApp [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {{
        color: {C["input_text"]} !important;
    }}
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li {{
        background-color: {C["input_bg"]} !important;
    }}

    /* Cards and containers */
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"],
    [data-testid="stMetric"],
    [data-testid="stAlert"] {{
        border-color: {C["border"]} !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {C["panel"]} !important;
        border-radius: 12px !important;
    }}

    [data-testid="stMetric"] {{
        background-color: {C["panel"]} !important;
        padding: 16px !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 12px !important;
    }}

    /* Buttons */
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 9px !important;
    }}
    .stApp [data-testid="stButton"] button p,
    .stApp [data-testid="stDownloadButton"] button p {{
        color: {C["text"]} !important;
    }}

    .stApp button[kind="primary"] {{
        background-color: {C["primary"]} !important;
        color: {C["button_text"]} !important;
        border-color: {C["primary"]} !important;
    }}
    .stApp button[kind="primary"] p {{
        color: {C["button_text"]} !important;
    }}

    /* Tabs */
    .stApp [data-testid="stTabs"] button {{
        color: {C["text"]} !important;
    }}

    /* Dataframes */
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"],
    .stApp table {{
        background-color: {C["panel"]} !important;
        color: {C["text"]} !important;
    }}

    /* File uploader */
    .stApp [data-testid="stFileUploader"] section {{
        background-color: {C["panel"]} !important;
        border: 1px dashed {C["border"]} !important;
    }}

    /* Links */
    .stApp a {{
        color: {C["primary"]} !important;
    }}

    /* Disabled fields */
    .stApp input:disabled,
    .stApp textarea:disabled {{
        background-color: {C["panel_alt"]} !important;
        color: {C["muted"]} !important;
        -webkit-text-fill-color: {C["muted"]} !important;
    }}

    /* Dividers */
    hr {{
        border-color: {C["border"]} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. SESSION STATE
# =========================================================

DEFAULTS = {
    "visits": [],
    "notes": [],
    "vitals": [],
    "symptoms": [],
    "medications": [],
    "documents": [],
    "classifier_result": None,
    "visit_comparison": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default.copy()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def today_string():
    return date.today().isoformat()


def safe_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return date.today()


def load_intent_model():
    """Search common locations for the saved intent classifier."""
    locations = [
        Path("intelliphr_intent_model.joblib"),
        Path("models/intelliphr_intent_model.joblib"),
        Path("model/intelliphr_intent_model.joblib"),
        Path("artifacts/intelliphr_intent_model.joblib"),
    ]

    for location in locations:
        if location.exists():
            try:
                return joblib.load(location), str(location), None
            except Exception as exc:
                return None, str(location), str(exc)

    return None, None, "Model file was not found in the expected locations."


def add_demo_data():
    if st.session_state.visits:
        return False

    st.session_state.visits = [
        {
            "date": (date.today() - timedelta(days=30)).isoformat(),
            "provider": "Demo General Physician",
            "reason": "Routine health review",
            "diagnosis": "Demo entry - not a real diagnosis",
            "notes": "Example record for testing the dashboard.",
        },
        {
            "date": (date.today() - timedelta(days=7)).isoformat(),
            "provider": "Demo Clinic",
            "reason": "Follow-up visit",
            "diagnosis": "Demo follow-up",
            "notes": "Example follow-up record.",
        },
    ]

    st.session_state.notes = [
        {
            "date": today_string(),
            "title": "Welcome to CareTrail",
            "text": "This is sample information. Replace it with your own records.",
        }
    ]

    st.session_state.vitals = [
        {
            "date": (date.today() - timedelta(days=14)).isoformat(),
            "systolic": 118, "diastolic": 76,
            "heart_rate": 72, "temperature": 36.7,
            "weight": 62.0,
        },
        {
            "date": (date.today() - timedelta(days=7)).isoformat(),
            "systolic": 120, "diastolic": 78,
            "heart_rate": 74, "temperature": 36.8,
            "weight": 61.8,
        },
        {
            "date": today_string(),
            "systolic": 117, "diastolic": 75,
            "heart_rate": 71, "temperature": 36.6,
            "weight": 61.7,
        },
    ]

    st.session_state.symptoms = [
        {
            "date": today_string(),
            "symptom": "Headache",
            "severity": 2,
            "duration": "A few hours",
            "notes": "Demo entry",
        }
    ]

    st.session_state.medications = [
        {
            "name": "Demo medication",
            "dose": "As prescribed",
            "frequency": "As directed",
            "time": "09:00",
            "start_date": today_string(),
            "notes": "Example only. Not a treatment recommendation.",
        }
    ]

    return True


def export_backup():
    keys = list(DEFAULTS.keys())
    data = {
        key: st.session_state.get(key, [])
        for key in keys
    }
    data["exported_at"] = datetime.now().isoformat()
    return json.dumps(data, indent=2, ensure_ascii=False)


def restore_backup(uploaded_file):
    data = json.load(uploaded_file)

    for key in DEFAULTS:
        if key in data and isinstance(data[key], list):
            st.session_state[key] = data[key]

    return True


def make_ics(medication):
    """Create a simple calendar event file for a medication reminder."""
    reminder_time = medication.get("time", "09:00")
    try:
        hour, minute = map(int, reminder_time.split(":"))
    except (ValueError, AttributeError):
        hour, minute = 9, 0

    start_day = safe_date(medication.get("start_date", today_string()))
    start_dt = datetime.combine(start_day, time(hour, minute))
    end_dt = start_dt + timedelta(minutes=10)

    def fmt(dt):
        return dt.strftime("%Y%m%dT%H%M%S")

    summary = medication.get("name", "Medication reminder")
    dose = medication.get("dose", "")
    safe_summary = summary.replace(",", "\\,").replace(";", "\\;")

    content = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//CareTrail//Medication Reminder//EN",
        "BEGIN:VEVENT",
        f"DTSTART:{fmt(start_dt)}",
        f"DTEND:{fmt(end_dt)}",
        f"SUMMARY:Medication reminder - {safe_summary}",
        f"DESCRIPTION:Dose: {dose}. Follow your prescriber's instructions.",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(content)


def apply_chart_theme(fig):
    fig.update_layout(
        template=C["chart_template"],
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(font=dict(color=C["text"])),
    )
    fig.update_xaxes(
        gridcolor=C["border"],
        zerolinecolor=C["border"],
        tickfont=dict(color=C["text"]),
        title_font=dict(color=C["text"]),
    )
    fig.update_yaxes(
        gridcolor=C["border"],
        zerolinecolor=C["border"],
        tickfont=dict(color=C["text"]),
        title_font=dict(color=C["text"]),
    )
    return fig


def show_empty(message):
    st.info(message)


# =========================================================
# 5. SIDEBAR AND THEME SWITCH
# =========================================================

with st.sidebar:
    st.markdown("# CareTrail")
    st.caption("Your personal health information dashboard")
    st.divider()

    theme_choice = st.radio(
        "Appearance",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="theme_selector",
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

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
            "About & Privacy",
        ],
        label_visibility="visible",
    )

    st.divider()

    if st.button("Load Demo Data", use_container_width=True):
        added = add_demo_data()
        if added:
            st.success("Demo data loaded.")
        else:
            st.warning("Records already exist. Clear them first if you want to reload.")

    if st.button("Clear All Temporary Data", use_container_width=True):
        for key in DEFAULTS:
            st.session_state[key] = []
        st.session_state.classifier_result = None
        st.session_state.visit_comparison = None
        st.rerun()

    st.caption("Demo entries are examples, not real medical records.")


# =========================================================
# 6. PAGE HEADER
# =========================================================

st.title("CareTrail")
st.caption("Organize your health information in one place.")

st.markdown(
    f"""
    <div style="
        background:{C['panel']};
        border:1px solid {C['border']};
        border-radius:14px;
        padding:16px 20px;
        margin-bottom:18px;">
        <div style="font-size:13px;color:{C['muted']};">
            PERSONAL HEALTH DASHBOARD
        </div>
        <div style="font-size:24px;font-weight:700;color:{C['text']};">
            Your health, organized.
        </div>
        <div style="font-size:14px;color:{C['muted']};">
            Track entries, review trends, and prepare for appointments.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. OVERVIEW
# =========================================================

if page == "Overview":
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Health visits", len(st.session_state.visits))
    col2.metric("Vitals records", len(st.session_state.vitals))
    col3.metric("Symptoms logged", len(st.session_state.symptoms))
    col4.metric("Medications listed", len(st.session_state.medications))

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Recent health visits")

        if st.session_state.visits:
            visits_df = pd.DataFrame(st.session_state.visits)
            if "date" in visits_df.columns:
                visits_df = visits_df.sort_values("date", ascending=False)
            st.dataframe(
                visits_df.head(5),
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_empty("No visits added yet. Add a visit under Health Notes & Timeline.")

    with right:
        st.subheader("Recent symptoms")

        if st.session_state.symptoms:
            symptoms_df = pd.DataFrame(st.session_state.symptoms)
            st.dataframe(
                symptoms_df.sort_values("date", ascending=False).head(5),
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_empty("Your symptom log is empty.")

    st.subheader("Vitals snapshot")

    if st.session_state.vitals:
        latest = sorted(
            st.session_state.vitals,
            key=lambda item: item.get("date", ""),
        )[-1]

        a, b, c, d = st.columns(4)
        a.metric("Blood pressure", f"{latest.get('systolic', '—')}/{latest.get('diastolic', '—')}")
        b.metric("Heart rate", f"{latest.get('heart_rate', '—')} bpm")
        c.metric("Temperature", f"{latest.get('temperature', '—')} °C")
        d.metric("Weight", f"{latest.get('weight', '—')} kg")

        st.caption(
            "These are saved entries, not a medical assessment. "
            "Interpret measurements with a qualified clinician."
        )
    else:
        show_empty("Add a vitals entry to see your latest measurements.")


# =========================================================
# 8. HEALTH NOTES AND TIMELINE
# =========================================================

elif page == "Health Notes & Timeline":
    st.subheader("Health Notes & Timeline")

    with st.expander("Add a health visit", expanded=True):
        with st.form("visit_form", clear_on_submit=True):
            visit_date = st.date_input("Visit date", value=date.today())
            provider = st.text_input("Hospital / doctor / clinic")
            reason = st.text_input("Reason for visit")
            diagnosis = st.text_input("Diagnosis or assessment (optional)")
            visit_notes = st.text_area("Visit notes")
            save_visit = st.form_submit_button("Save visit", type="primary")

            if save_visit:
                st.session_state.visits.append({
                    "date": visit_date.isoformat(),
                    "provider": provider.strip(),
                    "reason": reason.strip(),
                    "diagnosis": diagnosis.strip(),
                    "notes": visit_notes.strip(),
                })
                st.success("Visit saved.")

    with st.expander("Add a health note"):
        with st.form("note_form", clear_on_submit=True):
            note_date = st.date_input("Note date", value=date.today())
            note_title = st.text_input("Note title")
            note_text = st.text_area("Your note")
            save_note = st.form_submit_button("Save note", type="primary")

            if save_note:
                if not note_title.strip() and not note_text.strip():
                    st.warning("Enter a title or note first.")
                else:
                    st.session_state.notes.append({
                        "date": note_date.isoformat(),
                        "title": note_title.strip(),
                        "text": note_text.strip(),
                    })
                    st.success("Note saved.")

    st.divider()
    st.subheader("Visit timeline")

    if st.session_state.visits:
        visits_df = pd.DataFrame(st.session_state.visits)
        visits_df = visits_df.sort_values("date", ascending=False)
        st.dataframe(visits_df, use_container_width=True, hide_index=True)

        visit_csv = visits_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export visits as CSV",
            visit_csv,
            file_name="caretrail_visits.csv",
            mime="text/csv",
        )
    else:
        show_empty("No visits recorded yet.")

    st.subheader("Saved notes")

    if st.session_state.notes:
        for index, note in enumerate(reversed(st.session_state.notes)):
            with st.expander(
                f"{note.get('date', '')} — {note.get('title', 'Untitled note')}"
            ):
                st.write(note.get("text", ""))
    else:
        show_empty("No notes saved yet.")


# =========================================================
# 9. VITALS AND ANALYTICS
# =========================================================

elif page == "Vitals & Analytics":
    st.subheader("Vitals & Analytics")

    with st.expander("Add a vitals record", expanded=True):
        with st.form("vitals_form", clear_on_submit=True):
            record_date = st.date_input("Measurement date", value=date.today())

            col1, col2 = st.columns(2)
            systolic = col1.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=0, max_value=300, value=120,
            )
            diastolic = col2.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=0, max_value=200, value=80,
            )

            col3, col4, col5 = st.columns(3)
            heart_rate = col3.number_input(
                "Heart rate (bpm)", min_value=0, max_value=300, value=72
            )
            temperature = col4.number_input(
                "Temperature (°C)", min_value=25.0, max_value=45.0,
                value=36.7, step=0.1,
            )
            weight = col5.number_input(
                "Weight (kg)", min_value=0.0, max_value=500.0,
                value=60.0, step=0.1,
            )

            save_vitals = st.form_submit_button("Save vitals", type="primary")

            if save_vitals:
                st.session_state.vitals.append({
                    "date": record_date.isoformat(),
                    "systolic": systolic,
                    "diastolic": diastolic,
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "weight": weight,
                })
                st.success("Vitals saved.")

    if st.session_state.vitals:
        df = pd.DataFrame(st.session_state.vitals)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        st.divider()
        st.subheader("Trends over time")

        metric = st.selectbox(
            "Choose a measurement",
            [
                "Blood pressure",
                "Heart rate",
                "Temperature",
                "Weight",
            ],
        )

        if metric == "Blood pressure":
            chart_df = df.melt(
                id_vars=["date"],
                value_vars=["systolic", "diastolic"],
                var_name="measurement",
                value_name="value",
            )
            fig = px.line(
                chart_df, x="date", y="value", color="measurement",
                markers=True, title="Blood pressure over time",
                labels={"date": "Date", "value": "mmHg"},
            )
        else:
            mapping = {
                "Heart rate": ("heart_rate", "Heart rate (bpm)"),
                "Temperature": ("temperature", "Temperature (°C)"),
                "Weight": ("weight", "Weight (kg)"),
            }
            column, label = mapping[metric]
            fig = px.line(
                df, x="date", y=column, markers=True,
                title=f"{metric} over time",
                labels={column: label, "date": "Date"},
            )

        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        st.subheader("All vitals records")
        display_df = df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download vitals CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="caretrail_vitals.csv",
            mime="text/csv",
        )
    else:
        show_empty("Add measurements to start viewing your trends.")

    st.caption(
        "Charts show recorded values only. They do not diagnose disease "
        "or replace professional interpretation."
    )


# =========================================================
# 10. SYMPTOM TRACKER
# =========================================================

elif page == "Symptom Tracker":
    st.subheader("Symptom Tracker")

    with st.form("symptom_form", clear_on_submit=True):
        symptom_date = st.date_input("Date", value=date.today())
        symptom_name = st.text_input("Symptom")
        severity = st.slider("Severity (0 = none, 10 = worst)", 0, 10, 3)
        duration = st.selectbox(
            "Duration",
            ["Less than an hour", "A few hours", "1 day", "Several days", "Ongoing"],
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
                st.success("Symptom logged.")

    if st.session_state.symptoms:
        st.divider()
        symptom_df = pd.DataFrame(st.session_state.symptoms)
        symptom_df["date"] = pd.to_datetime(symptom_df["date"])

        fig = px.scatter(
            symptom_df,
            x="date",
            y="severity",
            color="symptom",
            size="severity",
            hover_data=["duration", "notes"],
            title="Symptom severity over time",
            labels={"date": "Date", "severity": "Severity"},
        )
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        st.dataframe(
            symptom_df.sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download symptom log CSV",
            symptom_df.to_csv(index=False).encode("utf-8"),
            file_name="caretrail_symptoms.csv",
            mime="text/csv",
        )
    else:
        show_empty("No symptoms logged yet.")

    st.caption(
        "Seek urgent medical attention for severe or rapidly worsening symptoms."
    )


# =========================================================
# 11. MEDICATIONS AND REMINDERS
# =========================================================

elif page == "Medications & Reminders":
    st.subheader("Medications & Reminders")

    st.warning(
        "Enter medication details exactly as directed by your clinician or pharmacist. "
        "CareTrail does not recommend doses or check drug interactions."
    )

    with st.form("medication_form", clear_on_submit=True):
        med_name = st.text_input("Medication name")
        med_dose = st.text_input("Dose as prescribed")
        med_frequency = st.text_input("Frequency (as prescribed)")
        med_time = st.time_input("Reminder time", value=time(9, 0))
        med_start = st.date_input("Start date", value=date.today())
        med_notes = st.text_area("Notes")
        save_med = st.form_submit_button("Add medication", type="primary")

        if save_med:
            if not med_name.strip():
                st.warning("Enter a medication name.")
            else:
                st.session_state.medications.append({
                    "name": med_name.strip(),
                    "dose": med_dose.strip(),
                    "frequency": med_frequency.strip(),
                    "time": med_time.strftime("%H:%M"),
                    "start_date": med_start.isoformat(),
                    "notes": med_notes.strip(),
                })
                st.success("Medication entry saved.")

    st.divider()
    st.subheader("Medication list")

    if st.session_state.medications:
        for index, med in enumerate(st.session_state.medications):
            with st.container(border=True):
                st.markdown(f"**{med.get('name', 'Medication')}**")
                st.write(f"Dose: {med.get('dose', 'Not entered')}")
                st.write(f"Frequency: {med.get('frequency', 'Not entered')}")
                st.write(f"Reminder time: {med.get('time', '09:00')}")
                st.write(f"Start date: {med.get('start_date', '')}")
                if med.get("notes"):
                    st.write(med["notes"])

                ics_content = make_ics(med)
                st.download_button(
                    "Download calendar reminder",
                    data=ics_content,
                    file_name=f"caretrail_reminder_{index + 1}.ics",
                    mime="text/calendar",
                    key=f"ics_{index}",
                )

                if st.button("Remove entry", key=f"remove_med_{index}"):
                    st.session_state.medications.pop(index)
                    st.rerun()
    else:
        show_empty("No medication entries yet.")


# =========================================================
# 12. DOCUMENT VAULT
# =========================================================

elif page == "Document Vault":
    st.subheader("Document Vault")

    st.write(
        "Upload reports or notes for this active app session. "
        "Text extraction is available for supported file types."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["txt", "csv", "md", "pdf"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        if st.button("Save document to session", type="primary"):
            file_bytes = uploaded.getvalue()

            extracted_text = ""
            file_ext = Path(uploaded.name).suffix.lower()

            if file_ext in [".txt", ".md", ".csv"]:
                extracted_text = file_bytes.decode("utf-8", errors="replace")
            elif file_ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(file_bytes))
                    extracted_text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except Exception as exc:
                    extracted_text = f"PDF extraction failed: {exc}"

            st.session_state.documents.append({
                "name": uploaded.name,
                "type": uploaded.type or "unknown",
                "size": len(file_bytes),
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                "text": extracted_text,
                "bytes_hex": file_bytes.hex(),
            })
            st.success("Document saved for this session.")

    st.divider()
    st.subheader("Saved documents")

    if st.session_state.documents:
        for index, doc in enumerate(st.session_state.documents):
            with st.expander(doc.get("name", "Document")):
                st.write(f"Type: {doc.get('type', 'Unknown')}")
                st.write(f"Size: {doc.get('size', 0):,} bytes")
                st.write(f"Added: {doc.get('uploaded_at', '')}")

                extracted = doc.get("text", "")
                if extracted:
                    st.text_area(
                        "Extracted text",
                        value=extracted[:20000],
                        height=180,
                        key=f"doc_text_{index}",
                    )
                else:
                    st.info("No text was extracted from this file.")

                try:
                    raw_bytes = bytes.fromhex(doc.get("bytes_hex", ""))
                    st.download_button(
                        "Download original file",
                        data=raw_bytes,
                        file_name=doc.get("name", "document"),
                        mime=doc.get("type", "application/octet-stream"),
                        key=f"doc_download_{index}",
                    )
                except ValueError:
                    st.error("Original file data is unavailable.")

                if st.button("Remove document", key=f"remove_doc_{index}"):
                    st.session_state.documents.pop(index)
                    st.rerun()
    else:
        show_empty("No documents saved in this session.")

    st.caption(
        "Session storage is temporary and may be lost when the app restarts. "
        "Avoid uploading sensitive medical records to a public demo."
    )


# =========================================================
# 13. VISIT COMPARISON
# =========================================================

elif page == "Visit Comparison":
    st.subheader("Visit Comparison")

    visits = st.session_state.visits

    if len(visits) < 2:
        show_empty("Add at least two visits to compare them.")
    else:
        visit_labels = [
            f"{v.get('date', '')} — {v.get('provider', 'Provider not entered')}"
            for v in visits
        ]

        col1, col2 = st.columns(2)
        first_index = col1.selectbox(
            "First visit",
            range(len(visits)),
            format_func=lambda i: visit_labels[i],
            key="visit_a",
        )
        second_index = col2.selectbox(
            "Second visit",
            range(len(visits)),
            index=1 if len(visits) > 1 else 0,
            format_func=lambda i: visit_labels[i],
            key="visit_b",
        )

        if first_index == second_index:
            st.warning("Choose two different visits.")
        else:
            a = visits[first_index]
            b = visits[second_index]

            comparison = pd.DataFrame([
                {"Field": "Date", "First visit": a.get("date", ""), "Second visit": b.get("date", "")},
                {"Field": "Provider", "First visit": a.get("provider", ""), "Second visit": b.get("provider", "")},
                {"Field": "Reason", "First visit": a.get("reason", ""), "Second visit": b.get("reason", "")},
                {"Field": "Diagnosis / assessment", "First visit": a.get("diagnosis", ""), "Second visit": b.get("diagnosis", "")},
                {"Field": "Notes", "First visit": a.get("notes", ""), "Second visit": b.get("notes", "")},
            ])

            st.dataframe(comparison, use_container_width=True, hide_index=True)

            st.download_button(
                "Download comparison CSV",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name="caretrail_visit_comparison.csv",
                mime="text/csv",
            )


# =========================================================
# 14. AI QUESTION CLASSIFIER
# =========================================================

elif page == "AI Question Classifier":
    st.subheader("AI Question Classifier")

    st.write(
        "Classifies the type of health-related question using your saved "
        "TF-IDF and Logistic Regression model. It does not diagnose illness "
        "or provide medical advice."
    )

    model, model_path, model_error = load_intent_model()

    if model is not None:
        st.success(f"Intent model loaded from `{model_path}`.")
    else:
        st.warning(
            "The saved intent model could not be loaded. "
            "Check that `intelliphr_intent_model.joblib` is committed to GitHub."
        )
        with st.expander("Model loading details"):
            st.code(model_error or "Unknown model-loading issue")

    question = st.text_area(
        "Enter a question",
        placeholder="Example: How can I organize my previous health records?",
        height=120,
    )

    if st.button("Classify question", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif model is None:
            st.error("The classifier model is unavailable.")
        else:
            try:
                prediction = model.predict([question.strip()])[0]
                result = {
                    "question": question.strip(),
                    "intent": str(prediction),
                }

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([question.strip()])[0]
                    classes = getattr(model, "classes_", None)

                    if classes is None and hasattr(model, "named_steps"):
                        final_step = list(model.named_steps.values())[-1]
                        classes = getattr(final_step, "classes_", None)

                    if classes is not None:
                        result["scores"] = {
                            str(cls): float(prob)
                            for cls, prob in zip(classes, probabilities)
                        }

                st.session_state.classifier_result = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.classifier_result

    if result:
        st.divider()
        st.subheader("Classification result")
        st.metric("Predicted category", result.get("intent", "Unknown"))
        st.write("**Question:**", result.get("question", ""))

        if result.get("scores"):
            scores_df = pd.DataFrame([
                {"Category": key, "Probability": value}
                for key, value in result["scores"].items()
            ]).sort_values("Probability", ascending=False)

            fig = px.bar(
                scores_df,
                x="Category",
                y="Probability",
                title="Model class probabilities",
                range_y=[0, 1],
            )
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        st.caption(
            "Model scores reflect the trained classifier, not clinical certainty. "
            "Do not use this feature for diagnosis, triage, or treatment decisions."
        )


# =========================================================
# 15. DATA BACKUP AND RESTORE
# =========================================================

elif page == "Data Backup & Restore":
    st.subheader("Data Backup & Restore")

    st.write(
        "Export your current session data to a JSON file. "
        "The backup may contain notes and document contents, so store it securely."
    )

    backup_text = export_backup()

    st.download_button(
        "Download complete JSON backup",
        data=backup_text.encode("utf-8"),
        file_name=f"caretrail_backup_{today_string()}.json",
        mime="application/json",
        type="primary",
    )

    st.divider()
    st.subheader("Restore from JSON backup")

    backup_file = st.file_uploader(
        "Choose a CareTrail JSON backup",
        type=["json"],
        key="restore_file",
    )

    st.warning(
        "Restoring replaces the current session lists with the lists in the backup. "
        "Download a backup first if you want to keep your current entries."
    )

    if st.button("Restore backup", disabled=backup_file is None):
        try:
            restore_backup(backup_file)
            st.success("Backup restored successfully.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore backup: {exc}")


# =========================================================
# 16. ABOUT AND PRIVACY
# =========================================================

elif page == "About & Privacy":
    st.subheader("About CareTrail")

    st.markdown(
        """
        CareTrail is a prototype for organizing personal health information.
        It brings together health notes, visit history, symptom logs, vitals,
        medication entries, and a basic text-intent classifier.
        """
    )

    st.subheader("Important limitations")

    st.markdown(
        """
        - CareTrail is not a medical device or a diagnostic system.
        - The classifier categorizes text; it does not assess medical risk.
        - No real-time drug interaction database is connected.
        - No wearable, hospital, or electronic health record integration is enabled.
        - Calendar files provide reminders but do not verify medication instructions.
        - Records kept in Streamlit session state may disappear after a restart.
        - Uploaded documents and exported backups can contain sensitive information.
        """
    )

    st.subheader("Privacy recommendations")

    st.markdown(
        """
        - Use demo or de-identified data while testing.
        - Do not upload passwords, identification numbers, or records you do not
          have permission to store.
        - Keep downloaded backups in a secure location.
        - Do not treat this prototype as a secure medical records system.
        """
    )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    f"CareTrail prototype · {theme_choice} mode · "
    f"Last rendered {datetime.now().strftime('%d %b %Y, %H:%M')}"
)
