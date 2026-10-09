
import html
import re
from datetime import date, timedelta

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareTrail | Health Companion",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. GLOBAL CSS — READABLE COLOURS THROUGHOUT THE APP
# =========================================================

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #F3F6FA !important;
        color: #1E293B !important;
    }

    /* Main page text */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricDelta"] {
        color: #1E293B !important;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #10243A !important;
        border-right: 1px solid #263E56;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li {
        color: #FFFFFF !important;
    }

    /* Text inputs and dropdowns */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border-color: #CBD5E1 !important;
    }

    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    /* Buttons */
    .stApp .stButton > button,
    .stApp .stDownloadButton > button {
        background-color: #087E8B !important;
        color: #FFFFFF !important;
        border: 1px solid #087E8B !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        min-height: 42px;
    }

    .stApp .stButton > button *,
    .stApp .stDownloadButton > button * {
        color: #FFFFFF !important;
    }

    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover {
        background-color: #066773 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE5EF !important;
        padding: 18px !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(15, 35, 60, 0.04);
    }

    [data-testid="stMetric"] * {
        color: #1E293B !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E7EDF5 !important;
        border-radius: 10px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #087E8B !important;
    }

    /* Dataframes, expanders and alerts */
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid #DCE5EF !important;
        border-radius: 10px !important;
    }

    [data-testid="stAlert"] p {
        color: #1E293B !important;
    }

    /* Custom cards */
    .hero {
        background: linear-gradient(120deg, #10243A, #174B64);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .hero h1, .hero p, .hero span {
        color: #FFFFFF !important;
    }

    .hero h1 {
        font-size: 32px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #8DE0D6 !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 700;
        color: #10243A !important;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    .muted {
        color: #64748B !important;
        font-size: 14px;
        line-height: 1.6;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 14px;
        padding: 19px;
        min-height: 145px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        color: #10243A !important;
        font-size: 17px;
        margin-top: 10px;
    }

    .info-card p {
        color: #526277 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .pill {
        display: inline-block;
        padding: 5px 10px;
        background-color: #DDF5F1;
        color: #086B69 !important;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .record-card {
        background-color: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-left: 4px solid #087E8B;
        border-radius: 10px;
        padding: 14px 17px;
        margin-bottom: 12px;
    }

    .record-card p {
        color: #334155 !important;
    }

    .footer {
        border-top: 1px solid #DCE5EF;
        margin-top: 35px;
        padding-top: 15px;
        color: #64748B !important;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. SESSION STATE
# =========================================================

DEFAULTS = {
    "visits": [],
    "measurements": [],
    "demo_loaded": False,
    "comparison_report": None,
    "medication_report": None,
    "classifier_result": None,
    "classifier_question": "",
    "visit_a_notes": "",
    "visit_b_notes": "",
    "meds_a": "",
    "meds_b": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def show_hero(title, subtitle, eyebrow="CARETRAIL HEALTH COMPANION"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, description=None):
    st.markdown(
        f'<div class="section-heading">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<div class="muted">{html.escape(description)}</div>',
            unsafe_allow_html=True,
        )


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def split_entries(text):
    return [
        item.strip()
        for item in re.split(r"[\n,;]+", str(text))
        if item.strip()
    ]


def visits_dataframe():
    columns = ["Date", "Visit Type", "Notes", "Medications", "Follow-up"]
    return pd.DataFrame(st.session_state.visits, columns=columns)


def measurements_dataframe():
    columns = [
        "Date", "Weight (kg)", "Systolic BP",
        "Diastolic BP", "Heart rate"
    ]
    return pd.DataFrame(st.session_state.measurements, columns=columns)


def add_demo_data():
    """Populate every section with synthetic demonstration data."""

    today = date.today()

    st.session_state.visits = [
        {
            "Date": str(today - timedelta(days=30)),
            "Visit Type": "General check-up",
            "Notes": "DEMO: Routine check-up recorded.",
            "Medications": "Demo Medicine A; Demo Supplement B",
            "Follow-up": "Review recorded measurements",
        },
        {
            "Date": str(today - timedelta(days=21)),
            "Visit Type": "Laboratory test",
            "Notes": "DEMO: Laboratory visit record.",
            "Medications": "Demo Medicine A",
            "Follow-up": "Review example report with clinician",
        },
        {
            "Date": str(today - timedelta(days=14)),
            "Visit Type": "Follow-up",
            "Notes": "DEMO: Follow-up appointment.",
            "Medications": "Demo Medicine A; Demo Medicine C",
            "Follow-up": "Next review in two weeks",
        },
        {
            "Date": str(today - timedelta(days=7)),
            "Visit Type": "General check-up",
            "Notes": "DEMO: Routine check-up recorded.",
            "Medications": "Demo Medicine A",
            "Follow-up": "Continue planned follow-up",
        },
        {
            "Date": str(today),
            "Visit Type": "Specialist appointment",
            "Notes": "DEMO: Example specialist visit.",
            "Medications": "Demo Medicine A; Demo Medicine D",
            "Follow-up": "Example follow-up appointment",
        },
    ]

    st.session_state.measurements = [
        {
            "Date": str(today - timedelta(days=30)),
            "Weight (kg)": 65.0,
            "Systolic BP": 120,
            "Diastolic BP": 80,
            "Heart rate": 74,
        },
        {
            "Date": str(today - timedelta(days=21)),
            "Weight (kg)": 64.8,
            "Systolic BP": 122,
            "Diastolic BP": 81,
            "Heart rate": 76,
        },
        {
            "Date": str(today - timedelta(days=14)),
            "Weight (kg)": 64.6,
            "Systolic BP": 119,
            "Diastolic BP": 79,
            "Heart rate": 73,
        },
        {
            "Date": str(today - timedelta(days=7)),
            "Weight (kg)": 64.5,
            "Systolic BP": 118,
            "Diastolic BP": 79,
            "Heart rate": 72,
        },
        {
            "Date": str(today),
            "Weight (kg)": 64.4,
            "Systolic BP": 121,
            "Diastolic BP": 80,
            "Heart rate": 75,
        },
    ]

    st.session_state.visit_a_notes = (
        "Routine check-up\n"
        "Blood pressure recorded\n"
        "Follow-up planned\n"
        "Laboratory report requested"
    )

    st.session_state.visit_b_notes = (
        "Routine check-up\n"
        "Blood pressure recorded\n"
        "Follow-up appointment scheduled"
    )

    st.session_state.meds_a = (
        "Demo Medicine A\n"
        "Demo Supplement B\n"
        "Demo Medicine C"
    )

    st.session_state.meds_b = (
        "Demo Medicine A\n"
        "Demo Medicine C\n"
        "Demo Medicine D"
    )

    st.session_state.classifier_question = (
        "Help me understand my medical records."
    )

    st.session_state.comparison_report = None
    st.session_state.medication_report = None
    st.session_state.classifier_result = None
    st.session_state.demo_loaded = True


def clear_all_data():
    st.session_state.visits = []
    st.session_state.measurements = []
    st.session_state.demo_loaded = False
    st.session_state.comparison_report = None
    st.session_state.medication_report = None
    st.session_state.classifier_result = None
    st.session_state.classifier_question = ""
    st.session_state.visit_a_notes = ""
    st.session_state.visit_b_notes = ""
    st.session_state.meds_a = ""
    st.session_state.meds_b = ""


def compare_text_lists(text_a, text_b):
    list_a = split_entries(text_a)
    list_b = split_entries(text_b)

    # Preserve the original spelling while comparing case-insensitively.
    map_a = {normalize(x): x for x in list_a}
    map_b = {normalize(x): x for x in list_b}

    common = sorted(map_a.keys() & map_b.keys())
    only_a = sorted(map_a.keys() - map_b.keys())
    only_b = sorted(map_b.keys() - map_a.keys())

    report_rows = []

    for key in common:
        report_rows.append({"Category": "Matching", "Entry": map_a[key]})
    for key in only_a:
        report_rows.append({"Category": "Only in A", "Entry": map_a[key]})
    for key in only_b:
        report_rows.append({"Category": "Only in B", "Entry": map_b[key]})

    return (
        [map_a[x] for x in common],
        [map_a[x] for x in only_a],
        [map_b[x] for x in only_b],
        pd.DataFrame(report_rows, columns=["Category", "Entry"]),
    )


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 22px 0;">
            <div style="font-size: 13px; letter-spacing: 2px;
                        color: #8DE0D6 !important; font-weight: 700;">
                CARETRAIL
            </div>
            <div style="font-size: 24px; font-weight: 700;
                        color: #FFFFFF !important; margin-top: 5px;">
                Health Companion
            </div>
            <div style="font-size: 12px; color: #C5D3E0 !important;
                        margin-top: 7px;">
                Your health information, organised.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Visit Comparison",
            "Medical Timeline",
            "Medication Review",
            "Health Trends",
            "AI Question Classifier",
            "About CareTrail",
        ],
    )

    st.markdown("---")
    st.markdown("**DEMO CONTROLS**")

    if st.button("Load Demo for All Checks", use_container_width=True):
        add_demo_data()
        st.rerun()

    if st.button("Clear All Session Data", use_container_width=True):
        clear_all_data()
        st.rerun()

    if st.session_state.demo_loaded:
        st.success("Synthetic demo data loaded.")
    else:
        st.caption("Load demo data to explore every section.")

    st.markdown("---")
    st.caption("Academic prototype")
    st.caption("Not for clinical decision-making.")


# =========================================================
# 6. DASHBOARD
# =========================================================

if page == "Dashboard":
    show_hero(
        "Your health, in one place.",
        "Organise visit notes, compare records, review medication entries, "
        "track measurements, and explore the AI classifier.",
    )

    visits = visits_dataframe()
    measurements = measurements_dataframe()

    medication_count = sum(
        len(split_entries(record.get("Medications", "")))
        for record in st.session_state.visits
    )

    try:
        joblib.load("intelliphr_intent_model.joblib")
        model_status = "Available"
    except Exception:
        model_status = "Not found"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visit records", len(visits))
    c2.metric("Measurements", len(measurements))
    c3.metric("Medication entries", medication_count)
    c4.metric("AI model", model_status)

    st.markdown("")
    section_title("Explore CareTrail", "Choose a module to explore.")

    cards = [
        (
            "01 · RECORDS",
            "Visit Comparison",
            "Compare two sets of notes and find matching or different entries.",
        ),
        (
            "02 · HISTORY",
            "Medical Timeline",
            "Add dated visits, review your records, and export a CSV.",
        ),
        (
            "03 · MEDICATIONS",
            "Medication Review",
            "Compare two medication lists using text matching.",
        ),
        (
            "04 · MONITORING",
            "Health Trends",
            "Record sample measurements and explore charts over time.",
        ),
        (
            "05 · MACHINE LEARNING",
            "AI Question Classifier",
            "Predict the intent category of a typed question.",
        ),
        (
            "06 · PROJECT",
            "About CareTrail",
            "Explore the technologies, features, and limitations.",
        ),
    ]

    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, card in zip(columns, cards[start:start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <span class="pill">{html.escape(card[0])}</span>
                        <h3>{html.escape(card[1])}</h3>
                        <p>{html.escape(card[2])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    section_title("Recent visit records")

    if visits.empty:
        st.info(
            "No visit records yet. Use 'Load Demo for All Checks' "
            "in the sidebar to populate the app."
        )
    else:
        st.dataframe(
            visits.sort_values("Date", ascending=False).head(5),
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.demo_loaded:
        st.info(
            "Demo mode is active. All example records and measurements "
            "are synthetic and are provided only to demonstrate the app."
        )

    st.warning(
        "CareTrail is an academic prototype. It does not diagnose diseases, "
        "recommend treatments, or replace a healthcare professional."
    )


# =========================================================
# 7. VISIT COMPARISON
# =========================================================

elif page == "Visit Comparison":
    show_hero(
        "Visit Comparison",
        "Compare two sets of visit notes and identify text differences.",
    )

    section_title(
        "Enter visit details",
        "Separate each entry with a new line, comma, or semicolon.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visit A")
        date_a = st.date_input("Visit A date", value=date.today(), key="visit_date_a")
        text_a = st.text_area(
            "Visit A notes",
            key="visit_a_notes",
            height=210,
            placeholder="Enter notes or load the demo data.",
        )

    with col2:
        st.subheader("Visit B")
        date_b = st.date_input("Visit B date", value=date.today(), key="visit_date_b")
        text_b = st.text_area(
            "Visit B notes",
            key="visit_b_notes",
            height=210,
            placeholder="Enter notes or load the demo data.",
        )

    if st.button("Compare Visit Notes", type="primary"):
        common, only_a, only_b, report = compare_text_lists(text_a, text_b)

        st.session_state.comparison_report = report

        c1, c2, c3 = st.columns(3)
        c1.metric("Matching entries", len(common))
        c2.metric("Only in Visit A", len(only_a))
        c3.metric("Only in Visit B", len(only_b))

        left, right = st.columns(2)

        with left:
            section_title("Matching in both visits")
            if common:
                for item in common:
                    st.success(item)
            else:
                st.info("No matching entries found.")

            section_title("Only in Visit A")
            if only_a:
                for item in only_a:
                    st.write("•", item)
            else:
                st.caption("No unique entries.")

        with right:
            section_title("Only in Visit B")
            if only_b:
                for item in only_b:
                    st.write("•", item)
            else:
                st.caption("No unique entries.")

    if st.session_state.comparison_report is not None:
        report = st.session_state.comparison_report.copy()
        report.insert(0, "Visit B date", str(date_b))
        report.insert(0, "Visit A date", str(date_a))

        st.download_button(
            "Download Visit Comparison (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="caretrail_visit_comparison.csv",
            mime="text/csv",
        )

    st.caption(
        "This is exact text comparison, not clinical interpretation. "
        "Similar wording may be treated as different entries."
    )


# =========================================================
# 8. MEDICAL TIMELINE
# =========================================================

elif page == "Medical Timeline":
    show_hero(
        "Medical Timeline",
        "Keep dated visit notes together and export your records.",
    )

    section_title("Add a visit record")

    with st.form("visit_form", clear_on_submit=True):
        visit_date = st.date_input("Visit date", value=date.today())

        visit_type = st.selectbox(
            "Visit type",
            [
                "General check-up",
                "Follow-up",
                "Laboratory test",
                "Specialist appointment",
                "Other",
            ],
        )

        notes = st.text_area("Visit notes")
        medications = st.text_area("Medication entries")
        follow_up = st.text_input("Follow-up notes")

        submitted = st.form_submit_button("Save Visit Record")

    if submitted:
        if not any([notes.strip(), medications.strip(), follow_up.strip()]):
            st.error("Enter at least one detail before saving.")
        else:
            st.session_state.visits.append(
                {
                    "Date": str(visit_date),
                    "Visit Type": visit_type,
                    "Notes": notes.strip(),
                    "Medications": medications.strip(),
                    "Follow-up": follow_up.strip(),
                }
            )
            st.success("Visit record added to this session.")
            st.rerun()

    st.markdown("")
    section_title("Visit history")

    visits = visits_dataframe()

    if visits.empty:
        st.info("No visit records yet. Load demo data or add a record above.")
    else:
        visits = visits.sort_values("Date", ascending=False)

        st.dataframe(
            visits,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export Medical Timeline (CSV)",
            data=visits.to_csv(index=False).encode("utf-8"),
            file_name="caretrail_medical_timeline.csv",
            mime="text/csv",
        )

        section_title("Timeline view")

        for _, row in visits.iterrows():
            with st.expander(f"{row['Date']} — {row['Visit Type']}"):
                st.write("**Notes:**", row["Notes"] or "None recorded")
                st.write("**Medication entries:**", row["Medications"] or "None recorded")
                st.write("**Follow-up:**", row["Follow-up"] or "None recorded")

    st.caption(
        "Records are stored temporarily in session memory, not a permanent database."
    )


# =========================================================
# 9. MEDICATION REVIEW
# =========================================================

elif page == "Medication Review":
    show_hero(
        "Medication Review",
        "Compare two medication lists to identify matching and different entries.",
    )

    st.info(
        "This tool compares text only. It does not verify prescriptions, "
        "dosages, interactions, medication safety, or suitability."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Medication List A")
        meds_a = st.text_area(
            "Enter list A",
            key="meds_a",
            height=190,
            placeholder="Enter medication entries or load demo data.",
        )

    with col2:
        st.subheader("Medication List B")
        meds_b = st.text_area(
            "Enter list B",
            key="meds_b",
            height=190,
            placeholder="Enter medication entries or load demo data.",
        )

    if st.button("Compare Medication Lists", type="primary"):
        common, only_a, only_b, report = compare_text_lists(meds_a, meds_b)

        st.session_state.medication_report = report

        c1, c2, c3 = st.columns(3)
        c1.metric("Matching entries", len(common))
        c2.metric("Only in list A", len(only_a))
        c3.metric("Only in list B", len(only_b))

        left, right = st.columns(2)

        with left:
            section_title("Matching entries")
            if common:
                for item in common:
                    st.success(item)
            else:
                st.info("No exact matches.")

            section_title("Only in list A")
            if only_a:
                for item in only_a:
                    st.write("•", item)
            else:
                st.caption("No unique entries.")

        with right:
            section_title("Only in list B")
            if only_b:
                for item in only_b:
                    st.write("•", item)
            else:
                st.caption("No unique entries.")

    if st.session_state.medication_report is not None:
        st.download_button(
            "Download Medication Comparison (CSV)",
            data=st.session_state.medication_report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="caretrail_medication_comparison.csv",
            mime="text/csv",
        )


# =========================================================
# 10. HEALTH TRENDS
# =========================================================

elif page == "Health Trends":
    show_hero(
        "Health Trends",
        "Record measurements and view numerical changes over time.",
    )

    st.info(
        "Charts show recorded values only. They do not determine whether "
        "a measurement is safe or diagnose a health condition."
    )

    section_title("Add a measurement")

    with st.form("measurement_form", clear_on_submit=True):
        measurement_date = st.date_input(
            "Measurement date",
            value=date.today(),
        )

        col1, col2 = st.columns(2)

        with col1:
            weight = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=0.1,
                help="Use 0 if unavailable.",
            )

            systolic = st.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=0,
                max_value=350,
                value=0,
                help="Use 0 if unavailable.",
            )

            diastolic = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=0,
                max_value=250,
                value=0,
                help="Use 0 if unavailable.",
            )

        with col2:
            heart_rate = st.number_input(
                "Heart rate (beats/min)",
                min_value=0,
                max_value=300,
                value=0,
                help="Use 0 if unavailable.",
            )

        measurement_submitted = st.form_submit_button("Save Measurement")

    if measurement_submitted:
        if all(x == 0 for x in [weight, systolic, diastolic, heart_rate]):
            st.error("Enter at least one measurement greater than zero.")
        else:
            st.session_state.measurements.append(
                {
                    "Date": str(measurement_date),
                    "Weight (kg)": weight if weight > 0 else None,
                    "Systolic BP": systolic if systolic > 0 else None,
                    "Diastolic BP": diastolic if diastolic > 0 else None,
                    "Heart rate": heart_rate if heart_rate > 0 else None,
                }
            )
            st.success("Measurement added to this session.")
            st.rerun()

    st.markdown("")
    section_title("Measurement history")

    measurements = measurements_dataframe()

    if measurements.empty:
        st.info("No measurements recorded. Load demo data or add measurements above.")
    else:
        measurements = measurements.sort_values("Date")

        st.dataframe(
            measurements,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export Health Measurements (CSV)",
            data=measurements.to_csv(index=False).encode("utf-8"),
            file_name="caretrail_health_measurements.csv",
            mime="text/csv",
        )

        chart_options = {
            "Weight (kg)": "Weight (kg)",
            "Systolic blood pressure": "Systolic BP",
            "Diastolic blood pressure": "Diastolic BP",
            "Heart rate": "Heart rate",
        }

        selected_chart = st.selectbox(
            "Select a measurement for the chart",
            list(chart_options.keys()),
        )

        column = chart_options[selected_chart]
        chart_data = measurements[["Date", column]].copy()
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
        chart_data["Date"] = pd.to_datetime(chart_data["Date"], errors="coerce")
        chart_data = chart_data.dropna(subset=["Date", column])

        if not chart_data.empty:
            chart_data = chart_data.sort_values("Date").set_index("Date")
            st.line_chart(chart_data, y=column)
        else:
            st.warning("No valid values available for this chart.")


# =========================================================
# 11. AI QUESTION CLASSIFIER
# =========================================================

elif page == "AI Question Classifier":
    show_hero(
        "AI Question Classifier",
        "Test the trained machine-learning model using your own questions "
        "or built-in examples.",
        eyebrow="MACHINE LEARNING MODULE",
    )

    st.markdown(
        """
        <div class="info-card">
            <span class="pill">HOW IT WORKS</span>
            <h3>Text classification</h3>
            <p>The model predicts an intent category from the question.
            It does not generate a medical answer or diagnose a condition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = {
        "Symptom help": "I have a headache and feel unwell.",
        "Medication query": "I have a question about my medication.",
        "Records query": "Help me understand my medical records.",
        "Health trends": "How can I review my health measurements?",
        "General help": "What features does this application provide?",
    }

    selected_example = st.selectbox(
        "Try a demo question",
        ["Choose a question"] + list(examples.keys()),
    )

    if st.button("Use Demo Question"):
        if selected_example != "Choose a question":
            st.session_state.classifier_question = examples[selected_example]
            st.rerun()

    question = st.text_area(
        "Enter a question to classify",
        key="classifier_question",
        placeholder="Type a question or choose a demo question above.",
        height=130,
    )

    try:
        model = joblib.load("intelliphr_intent_model.joblib")
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)

    if model is None:
        st.error(
            "The trained model could not be loaded. Check that "
            "intelliphr_intent_model.joblib is present in the repository "
            "and that the required packages are installed."
        )
        with st.expander("Technical details"):
            st.code(model_error or "Unknown error")
    else:
        st.success("Trained model loaded successfully.")

        if st.button("Classify Question", type="primary"):
            if not question.strip():
                st.warning("Enter a question before classifying.")
            else:
                try:
                    prediction = model.predict([question])[0]
                    result = {
                        "question": question,
                        "prediction": str(prediction),
                    }

                    if hasattr(model, "predict_proba"):
                        scores = model.predict_proba([question])[0]
                        classes = getattr(
                            model,
                            "classes_",
                            range(len(scores)),
                        )

                        result["scores"] = pd.DataFrame(
                            {
                                "Intent": [str(x) for x in classes],
                                "Model score (%)": [
                                    round(float(x) * 100, 2)
                                    for x in scores
                                ],
                            }
                        ).sort_values("Model score (%)", ascending=False)

                    st.session_state.classifier_result = result

                except Exception as exc:
                    st.session_state.classifier_result = None
                    st.error("Classification failed.")
                    st.code(str(exc))

        result = st.session_state.classifier_result

        if result is not None:
            st.markdown("")
            section_title("Classification result")

            st.write("**Question:**", result["question"])
            st.metric("Predicted intent", result["prediction"])

            if "scores" in result:
                st.markdown("**Scores across available categories**")
                st.dataframe(
                    result["scores"],
                    use_container_width=True,
                    hide_index=True,
                )

                chart_data = result["scores"].set_index("Intent")
                st.bar_chart(chart_data["Model score (%)"])

                st.caption(
                    "Scores are model outputs, not medically validated "
                    "probabilities or diagnostic confidence."
                )

            st.warning(
                "The classifier predicts intent only. It does not provide "
                "medical advice or establish a diagnosis."
            )


# =========================================================
# 12. ABOUT CARETRAIL
# =========================================================

elif page == "About CareTrail":
    show_hero(
        "About CareTrail",
        "An academic prototype exploring health-information organisation "
        "and machine-learning-based text classification.",
    )

    section_title("Project overview")

    st.write(
        "CareTrail brings together sample visit records, text comparisons, "
        "measurement charts, CSV exports, and a machine-learning classifier "
        "in a single Streamlit application."
    )

    section_title("Features included")

    features = pd.DataFrame(
        [
            {
                "Module": "Dashboard",
                "Function": "Overview and summary metrics",
            },
            {
                "Module": "Visit Comparison",
                "Function": "Compare visit notes and export results",
            },
            {
                "Module": "Medical Timeline",
                "Function": "Store session visit entries and export CSV",
            },
            {
                "Module": "Medication Review",
                "Function": "Compare medication-list text entries",
            },
            {
                "Module": "Health Trends",
                "Function": "Record measurements, chart values and export CSV",
            },
            {
                "Module": "AI Question Classifier",
                "Function": "Predict question-intent categories",
            },
        ]
    )

    st.dataframe(features, use_container_width=True, hide_index=True)

    section_title("Technologies")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Streamlit</h3>
                <p>Interactive Python web application framework.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Scikit-learn</h3>
                <p>Machine-learning tools used by the text classifier.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Pandas</h3>
                <p>Tabular data handling, organisation and CSV exports.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Limitations and responsible use")

    st.markdown(
        """
        - This is an academic demonstration, not a clinical system.
        - Classifier results depend on the training data and can be incorrect.
        - Model scores are not diagnostic confidence.
        - Text comparison does not assess medical meaning or medication safety.
        - Trend charts display recorded measurements without clinical interpretation.
        - Session data is temporary and is not a permanent medical record.
        - Do not enter identifiable patient information or sensitive medical records.
        """
    )

    st.warning(
        "Do not use CareTrail to make diagnosis, treatment, or medication "
        "decisions. Seek advice from a qualified healthcare professional."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <strong>CareTrail</strong> · Academic healthcare information prototype
        <br>
        For educational demonstration only. Not a substitute for professional
        medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
