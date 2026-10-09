
import streamlit as st
import joblib
import pandas as pd
import re
import html
from datetime import date, timedelta

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CareTrail | Healthcare Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "intelliphr_intent_model.joblib"

# =========================================================
# DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --navy: #17324D;
    --teal: #087F8C;
    --pale: #E8F5F5;
    --bg: #F5F8FB;
    --border: #E0E8EF;
    --muted: #718096;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp { background: var(--bg); color: var(--navy); }

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
}

.hero {
    background: linear-gradient(115deg, #17324D, #20556A 65%, #087F8C);
    padding: 30px;
    border-radius: 19px;
    color: white;
    margin-bottom: 23px;
}

.hero h1 {
    color: white !important;
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
}

.hero p { color: #DFECEF; margin-bottom: 0; }

.eyebrow {
    color: #A9E5E4;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.heading {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: var(--navy);
    margin-bottom: 4px;
}

.subheading {
    color: var(--muted);
    margin-bottom: 20px;
    font-size: 14px;
}

.metric {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    min-height: 110px;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.7px;
    font-weight: 700;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    color: var(--navy);
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 21px;
    margin-bottom: 15px;
}

.panel-title {
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 7px;
}

.panel-caption {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 12px;
}

.pill {
    display: inline-block;
    background: var(--pale);
    color: var(--teal);
    border-radius: 30px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}

.stButton > button, .stDownloadButton > button {
    background: var(--teal);
    color: white;
    border: 1px solid var(--teal);
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #066974;
    color: white;
    border-color: #066974;
}

.stTextArea textarea, .stTextInput input {
    border-radius: 9px;
    background: white;
}

[data-testid="stAlert"] { border-radius: 11px; }

.footer {
    text-align: center;
    color: #7B8A9A;
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 35px;
    font-size: 12px;
}

@media(max-width: 700px) {
    .hero { padding: 22px; }
    .hero h1 { font-size: 27px; }
    .panel { padding: 15px; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


# =========================================================
# SESSION DATA
# =========================================================
if "visits" not in st.session_state:
    st.session_state.visits = []

if "measurements" not in st.session_state:
    st.session_state.measurements = pd.DataFrame({
        "Date": pd.Series(dtype="object"),
        "Weight (kg)": pd.Series(dtype="float"),
        "Systolic BP": pd.Series(dtype="float"),
        "Diastolic BP": pd.Series(dtype="float"),
        "Heart rate": pd.Series(dtype="float"),
    })

if "comparison_report" not in st.session_state:
    st.session_state.comparison_report = None

if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False


# =========================================================
# HELPERS
# =========================================================
def safe(value):
    return html.escape(str(value))


def entries(text):
    return [x.strip() for x in text.splitlines() if x.strip()]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def compare_lists(old, new):
    old_map = {normalize(x): x for x in old}
    new_map = {normalize(x): x for x in new}

    added = [new_map[k] for k in new_map if k not in old_map]
    removed = [old_map[k] for k in old_map if k not in new_map]
    same = [new_map[k] for k in new_map if k in old_map]

    return added, removed, same


def metric(label, value, caption=""):
    st.markdown(f"""
    <div class="metric">
      <div class="metric-label">{safe(label)}</div>
      <div class="metric-value">{safe(value)}</div>
      <div style="font-size:12px;color:#718096;margin-top:4px">
        {safe(caption)}
      </div>
    </div>
    """, unsafe_allow_html=True)


def panel_title(title, caption=""):
    st.markdown(
        f'<div class="panel-title">{safe(title)}</div>',
        unsafe_allow_html=True
    )
    if caption:
        st.markdown(
            f'<div class="panel-caption">{safe(caption)}</div>',
            unsafe_allow_html=True
        )


def show_entries(title, items, empty="No entries recorded."):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title(title)

    if items:
        for item in items:
            st.markdown(
                f'<div style="padding:9px 0;border-bottom:1px solid #EDF1F5">'
                f'{safe(item)}</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption(empty)

    st.markdown('</div>', unsafe_allow_html=True)


def report_text(title, content):
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"{content}\n\n"
        "DISCLAIMER\n"
        "CareTrail is an academic prototype. Information entered is not "
        "independently verified. Text differences do not establish clinical "
        "significance. Do not use this report to diagnose a condition or "
        "change treatment. Consult a qualified healthcare professional.\n"
    )


def add_demo_data():
    if st.session_state.demo_loaded:
        return

    today = date.today()
    st.session_state.visits = [
        {
            "date": today - timedelta(days=60),
            "title": "Previous demonstration visit",
            "notes": "Synthetic example: routine follow-up.",
            "medications": ["Example Medicine A: 1 tablet daily"],
            "followup": ["Example: return for scheduled review"],
        },
        {
            "date": today - timedelta(days=15),
            "title": "Recent demonstration visit",
            "notes": "Synthetic example: review of recorded information.",
            "medications": [
                "Example Medicine A: 1 tablet daily",
                "Example Medicine B: as recorded in example"
            ],
            "followup": [
                "Example: return for scheduled review",
                "Example: bring previous test report"
            ],
        },
    ]

    st.session_state.measurements = pd.DataFrame({
        "Date": [
            today - timedelta(days=60),
            today - timedelta(days=45),
            today - timedelta(days=30),
            today - timedelta(days=15),
        ],
        "Weight (kg)": [64.0, 63.8, 64.2, 63.9],
        "Systolic BP": [120, 122, 119, 121],
        "Diastolic BP": [78, 79, 77, 78],
        "Heart rate": [72, 74, 71, 73],
    })

    st.session_state.demo_loaded = True


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div style="font-size:30px">🩺</div>
      <div style="font-family:Manrope,sans-serif;font-size:25px;
                  font-weight:800;color:#17324D">CareTrail</div>
      <div style="font-size:12px;color:#718096">
        Healthcare Information Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
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
        label_visibility="visible"
    )

    st.markdown("---")

    if model is not None:
        st.success("NLP model loaded")
    else:
        st.warning("NLP model unavailable")

    st.caption("Academic prototype")
    st.caption("Use synthetic data only.")


# =========================================================
# GLOBAL HEADER
# =========================================================
st.markdown("""
<div class="hero">
  <div class="eyebrow">PATIENT INFORMATION • ANALYTICS • NLP</div>
  <h1>CareTrail</h1>
  <p>Organize visit information, review recorded differences,
  and explore health data in one workspace.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown('<div class="heading">Dashboard overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Your workspace for visit records, '
        'measurements and AI-assisted question classification.</div>',
        unsafe_allow_html=True
    )

    visits = st.session_state.visits
    measurements = st.session_state.measurements

    a, b, c, d = st.columns(4)
    with a:
        metric("VISIT RECORDS", len(visits), "Saved in this session")
    with b:
        metric("MEASUREMENTS", len(measurements), "Recorded data points")
    with c:
        metric("AI CLASSIFIER", "Ready" if model else "Unavailable",
               "Trained NLP model")
    with d:
        metric("REPORTS", "Exportable", "Text and CSV downloads")

    st.write("")

    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Welcome to CareTrail",
                    "Choose a tool to begin working.")

        st.markdown("""
        <span class="pill">Visit comparison</span>
        <span class="pill">Medical timeline</span>
        <span class="pill">Medication review</span>
        <span class="pill">Health trends</span>
        <span class="pill">AI classifier</span>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="color:#718096;line-height:1.8">
        CareTrail is a prototype for organizing information entered for
        different visits. It highlights text differences and visualizes
        recorded measurements. Its AI component predicts question categories;
        it does not answer medical questions or provide a diagnosis.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Quick start")

        st.write("**New to CareTrail?**")
        st.write("Load clearly labelled synthetic example data to explore the features.")

        if st.button("Load demo data", use_container_width=True):
            add_demo_data()
            st.success("Synthetic demo data loaded.")
            st.rerun()

        if st.button("Clear session data", use_container_width=True):
            st.session_state.visits = []
            st.session_state.measurements = pd.DataFrame({
                "Date": pd.Series(dtype="object"),
                "Weight (kg)": pd.Series(dtype="float"),
                "Systolic BP": pd.Series(dtype="float"),
                "Diastolic BP": pd.Series(dtype="float"),
                "Heart rate": pd.Series(dtype="float"),
            })
            st.session_state.comparison_report = None
            st.session_state.demo_loaded = False
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Recent visit records")

    if visits:
        recent = sorted(visits, key=lambda x: x["date"], reverse=True)
        st.dataframe(
            pd.DataFrame([
                {
                    "Date": v["date"],
                    "Visit": v["title"],
                    "Notes": v["notes"],
                    "Medication entries": len(v["medications"]),
                }
                for v in recent
            ]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No visit records yet. Use Medical Timeline to add one.")

    st.markdown("### Workspace tools")
    tool_cols = st.columns(3)

    tools = [
        ("Compare visits", "Identify differences between two records."),
        ("Review medications", "Compare medication text entries."),
        ("Explore health trends", "Plot recorded measurements over time."),
    ]

    for col, (title, description) in zip(tool_cols, tools):
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title(title)
            st.caption(description)
            st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# VISIT COMPARISON
# =========================================================
elif page == "Visit Comparison":
    st.markdown('<div class="heading">Visit comparison</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Compare medication entries and '
        'follow-up instructions from two visits.</div>',
        unsafe_allow_html=True
    )

    with st.form("compare_form"):
        col1, col2 = st.columns(2)

        with col1:
            old_date = st.date_input("Previous visit date",
                                     value=date.today() - timedelta(days=30))
            old_meds_text = st.text_area(
                "Previous medication entries",
                placeholder="One entry per line",
                height=140
            )
            old_follow_text = st.text_area(
                "Previous follow-up instructions",
                placeholder="One instruction per line",
                height=110
            )

        with col2:
            new_date = st.date_input("Current visit date",
                                     value=date.today())
            new_meds_text = st.text_area(
                "Current medication entries",
                placeholder="One entry per line",
                height=140
            )
            new_follow_text = st.text_area(
                "Current follow-up instructions",
                placeholder="One instruction per line",
                height=110
            )

        compare = st.form_submit_button(
            "Compare records", use_container_width=True
        )

    if compare:
        old_meds, new_meds = entries(old_meds_text), entries(new_meds_text)
        old_follow = entries(old_follow_text)
        new_follow = entries(new_follow_text)

        med_diff = compare_lists(old_meds, new_meds)
        follow_diff = compare_lists(old_follow, new_follow)

        content = [
            f"Previous visit date: {old_date}",
            f"Current visit date: {new_date}",
            "",
            "MEDICATIONS",
        ]

        for title, items in zip(
            ["New entries", "No longer listed", "Exact matches"], med_diff
        ):
            content += [f"\n{title}:"] + (items or ["None identified"])

        content.append("\nFOLLOW-UP INSTRUCTIONS")
        for title, items in zip(
            ["New entries", "No longer listed", "Exact matches"], follow_diff
        ):
            content += [f"\n{title}:"] + (items or ["None identified"])

        st.session_state.comparison_report = report_text(
            "CARETRAIL VISIT COMPARISON", "\n".join(content)
        )

        x, y, z = st.columns(3)
        with x:
            metric("NEW MEDICATION ENTRIES", len(med_diff[0]))
        with y:
            metric("NO LONGER LISTED", len(med_diff[1]))
        with z:
            metric("EXACT MATCHES", len(med_diff[2]))

        st.markdown("### Medication differences")
        left, right = st.columns(2)
        with left:
            show_entries("New entries", med_diff[0])
            show_entries("No longer listed", med_diff[1])
        with right:
            show_entries("Exact matches", med_diff[2])

        st.markdown("### Follow-up differences")
        left, right = st.columns(2)
        with left:
            show_entries("New instructions", follow_diff[0])
        with right:
            show_entries("No longer listed", follow_diff[1])
            show_entries("Exact matches", follow_diff[2])

    if st.session_state.comparison_report:
        st.download_button(
            "Download comparison report",
            st.session_state.comparison_report,
            file_name="CareTrail_Comparison.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.info(
        "Text comparison only: a missing entry does not prove that a "
        "medicine was stopped. Verify all differences with a clinician."
    )


# =========================================================
# MEDICAL TIMELINE
# =========================================================
elif page == "Medical Timeline":
    st.markdown('<div class="heading">Medical timeline</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Create a chronological record of visit '
        'notes. Records are held in the current app session only.</div>',
        unsafe_allow_html=True
    )

    with st.form("add_visit_form"):
        visit_date = st.date_input("Visit date", value=date.today())
        visit_title = st.text_input("Visit title",
                                    placeholder="Example: Routine follow-up")
        visit_notes = st.text_area("Visit notes",
                                   placeholder="Enter a short note",
                                   height=110)
        visit_meds = st.text_area(
            "Medication entries",
            placeholder="One entry per line",
            height=100
        )
        visit_follow = st.text_area(
            "Follow-up instructions",
            placeholder="One instruction per line",
            height=100
        )
        save_visit = st.form_submit_button(
            "Add visit record", use_container_width=True
        )

    if save_visit:
        if not visit_title.strip():
            st.error("Please enter a visit title.")
        else:
            st.session_state.visits.append({
                "date": visit_date,
                "title": visit_title.strip(),
                "notes": visit_notes.strip(),
                "medications": entries(visit_meds),
                "followup": entries(visit_follow),
            })
            st.success("Visit record added to this session.")
            st.rerun()

    st.markdown("### Visit history")

    if st.session_state.visits:
        ordered = sorted(st.session_state.visits,
                         key=lambda x: x["date"], reverse=True)

        for i, visit in enumerate(ordered):
            with st.expander(
                f"{visit['date']} — {visit['title']}", expanded=(i == 0)
            ):
                st.write("**Notes**")
                st.write(visit["notes"] or "No notes entered.")
                st.write("**Medication entries**")
                for item in visit["medications"]:
                    st.write(f"- {item}")
                if not visit["medications"]:
                    st.caption("No medication entries.")

                st.write("**Follow-up instructions**")
                for item in visit["followup"]:
                    st.write(f"- {item}")
                if not visit["followup"]:
                    st.caption("No follow-up instructions.")

        timeline_df = pd.DataFrame([
            {
                "Date": v["date"],
                "Visit": v["title"],
                "Notes": v["notes"],
                "Medications": " | ".join(v["medications"]),
                "Follow-up": " | ".join(v["followup"]),
            }
            for v in ordered
        ])

        st.download_button(
            "Download medical timeline (CSV)",
            timeline_df.to_csv(index=False),
            file_name="CareTrail_Timeline.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No visit records saved. Add a record above or load demo data from Dashboard.")


# =========================================================
# MEDICATION REVIEW
# =========================================================
elif page == "Medication Review":
    st.markdown('<div class="heading">Medication review</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Compare medication text entries and '
        'prepare a review list. This tool does not validate prescriptions.</div>',
        unsafe_allow_html=True
    )

    old_text = st.text_area(
        "Previous medication list",
        placeholder="One medication entry per line",
        height=170
    )
    new_text = st.text_area(
        "Current medication list",
        placeholder="One medication entry per line",
        height=170
    )

    if st.button("Review medication differences", use_container_width=True):
        result = compare_lists(entries(old_text), entries(new_text))
        labels = ["New entries", "No longer listed", "Exact matches"]

        for label, items in zip(labels, result):
            show_entries(label, items)

        medication_report = report_text(
            "CARETRAIL MEDICATION TEXT REVIEW",
            "\n".join(
                [f"\n{label}:"] + (items or ["None identified"])
                for label, items in zip(labels, result)
            )
        )

        st.download_button(
            "Download medication review",
            medication_report,
            file_name="CareTrail_Medication_Review.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.warning(
        "This comparison cannot determine whether a medication is safe, "
        "appropriate, discontinued, or interacting with another medication. "
        "Confirm medication questions with a qualified healthcare professional."
    )


# =========================================================
# HEALTH TRENDS
# =========================================================
elif page == "Health Trends":
    st.markdown('<div class="heading">Health trends</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Record numeric measurements and visualize '
        'their values over time. The charts describe the entered data only.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title("Add a measurement",
                "Leave fields blank when a measurement was not recorded.")

    with st.form("measurement_form"):
        measure_date = st.date_input("Measurement date", value=date.today())

        a, b = st.columns(2)
        with a:
            weight = st.number_input(
                "Weight (kg)", min_value=0.0, max_value=500.0,
                value=None, step=0.1, format="%.1f"
            )
            systolic = st.number_input(
                "Systolic blood pressure", min_value=0, max_value=350,
                value=None, step=1
            )
            diastolic = st.number_input(
                "Diastolic blood pressure", min_value=0, max_value=250,
                value=None, step=1
            )

        with b:
            heart_rate = st.number_input(
                "Heart rate (beats/min)", min_value=0, max_value=300,
                value=None, step=1
            )

        add_measurement = st.form_submit_button(
            "Save measurement", use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if add_measurement:
        if all(v is None for v in [weight, systolic, diastolic, heart_rate]):
            st.error("Enter at least one measurement.")
        else:
            new_row = {
                "Date": measure_date,
                "Weight (kg)": weight,
                "Systolic BP": systolic,
                "Diastolic BP": diastolic,
                "Heart rate": heart_rate,
            }
            st.session_state.measurements = pd.concat(
                [
                    st.session_state.measurements,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )
            st.success("Measurement saved for this session.")
            st.rerun()

    data = st.session_state.measurements.copy()

    if not data.empty:
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")

        st.markdown("### Measurement charts")

        chart_columns = [
            ("Weight (kg)", "Weight over time"),
            ("Systolic BP", "Systolic blood pressure"),
            ("Diastolic BP", "Diastolic blood pressure"),
            ("Heart rate", "Heart rate over time"),
        ]

        for i in range(0, len(chart_columns), 2):
            cols = st.columns(2)
            for col, (column, title) in zip(
                cols, chart_columns[i:i + 2]
            ):
                with col:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    panel_title(title)
                    chart_data = data[["Date", column]].dropna()
                    if not chart_data.empty:
                        st.line_chart(
                            chart_data.set_index("Date"),
                            y=column,
                            use_container_width=True
                        )
                    else:
                        st.caption("No values recorded for this measurement.")
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Recorded measurements")
        st.dataframe(data, use_container_width=True, hide_index=True)

        st.download_button(
            "Download measurements (CSV)",
            data.to_csv(index=False),
            file_name="CareTrail_Health_Trends.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No measurements recorded yet. Add data above or load demo data from Dashboard.")

    st.warning(
        "Charts are descriptive only. They do not establish a diagnosis, "
        "identify medical emergencies, or recommend treatment."
    )


# =========================================================
# AI QUESTION CLASSIFIER
# =========================================================
elif page == "AI Question Classifier":
    st.markdown('<div class="heading">AI question classifier</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Predict the intent category of a question '
        'using the saved TF-IDF and Logistic Regression pipeline.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.3, 0.7])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Enter a question")
        question = st.text_area(
            "Question",
            placeholder="Example: How can I understand the information in my records?",
            height=150
        )
        classify = st.button("Classify question", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Model categories")
        for label in [
            "Symptom help",
            "Medication query",
            "Records query",
            "Health trends",
            "General help",
        ]:
            st.markdown(
                f'<span class="pill">{label}</span>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if classify:
        if not question.strip():
            st.warning("Please enter a question.")
        elif model is None:
            st.error("The trained model could not be loaded.")
            with st.expander("Technical details"):
                st.code(model_error or "Unknown model error")
        else:
            try:
                prediction = model.predict([question.strip()])[0]
                label_map = {
                    "symptom_help": "Symptom-related question",
                    "medication_query": "Medication-related question",
                    "records_query": "Medical records question",
                    "health_trends": "Health trends question",
                    "general_help": "General health assistance",
                }
                friendly = label_map.get(
                    str(prediction),
                    str(prediction).replace("_", " ").title()
                )

                st.markdown("### Classification result")
                st.markdown(f"""
                <div class="hero">
                  <div class="eyebrow">PREDICTED CATEGORY</div>
                  <h1 style="font-size:28px">{safe(friendly)}</h1>
                  <p>Model label: {safe(prediction)}</p>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([question.strip()])[0]
                    classes = model.classes_

                    st.markdown("### Model scores")
                    for category, score in sorted(
                        zip(classes, probabilities),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        name = label_map.get(
                            str(category),
                            str(category).replace("_", " ").title()
                        )
                        st.write(f"**{name}** — {score:.1%}")
                        st.progress(float(score))

                st.warning(
                    "This is a research classifier trained on synthetic "
                    "examples. It does not diagnose conditions, assess urgency, "
                    "or provide medical advice."
                )

            except Exception as exc:
                st.error("Question classification failed.")
                with st.expander("Technical details"):
                    st.code(str(exc))


# =========================================================
# ABOUT
# =========================================================
elif page == "About CareTrail":
    st.markdown('<div class="heading">About CareTrail</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">An academic prototype demonstrating '
        'health-information organization, comparison and NLP classification.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Project features")
        st.write("- Visit record comparison")
        st.write("- Chronological medical timeline")
        st.write("- Medication text review")
        st.write("- Measurement trend charts")
        st.write("- NLP-based question classification")
        st.write("- Downloadable reports")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("Technology")
        st.write("**Frontend:** Streamlit")
        st.write("**NLP:** TF-IDF and Logistic Regression")
        st.write("**Model loading:** Joblib")
        st.write("**Data visualization:** Pandas and Streamlit charts")
        st.write("**Deployment:** Streamlit Community Cloud")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title("Limitations and responsible use")
    st.write(
        "CareTrail is a student research prototype, not a clinical "
        "decision-support system. The classifier was trained on synthetic "
        "examples and has not been clinically validated. Measurements and "
        "record comparisons are not interpreted as medical advice."
    )
    st.write(
        "Session data may be lost when the session restarts. This version "
        "does not provide secure persistent patient storage, user accounts, "
        "or hospital-system integration. Do not enter identifiable patient data."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
  <strong style="color:#17324D">CareTrail</strong>
  · Healthcare Information & AI Prototype<br>
  Synthetic data recommended · Not for diagnosis or treatment decisions
</div>
""", unsafe_allow_html=True)
