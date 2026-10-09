
from pathlib import Path
import io
import re

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# CARETRAIL — PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CareTrail Health Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
        color: #172b4d;
    }

    [data-testid="stHeader"] {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #10243a;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, p, label {
        color: #172b4d;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e8f0;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #087f8c;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #066875;
        color: white;
        border: none;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 18px;
        border: 1px solid #e1e8f0;
        border-radius: 12px;
    }

    .hero {
        background: linear-gradient(120deg, #10243a, #087f8c);
        padding: 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "visits": [],
    "measurements": [],
    "comparison_report": None,
    "medication_report": None,
    "classifier_result": None,
    "classifier_question": "",
    "visit_a_notes": "",
    "visit_b_notes": "",
    "meds_a": "",
    "meds_b": "",
    "demo_loaded": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# DEMO DATA
# --------------------------------------------------

def add_demo_data():
    st.session_state.visits = [
        {
            "Date": "2026-10-01",
            "Visit Type": "Routine check-up",
            "Notes": "Routine follow-up visit. Blood pressure recorded.",
            "Provider": "Demo Provider",
        },
        {
            "Date": "2026-10-04",
            "Visit Type": "Laboratory review",
            "Notes": "Laboratory report reviewed during follow-up.",
            "Provider": "Demo Provider",
        },
        {
            "Date": "2026-10-07",
            "Visit Type": "Follow-up",
            "Notes": "Follow-up appointment scheduled.",
            "Provider": "Demo Provider",
        },
    ]

    st.session_state.measurements = [
        {"Date": "2026-10-01", "Measurement": "Sample metric A", "Value": 72.0},
        {"Date": "2026-10-03", "Measurement": "Sample metric A", "Value": 74.0},
        {"Date": "2026-10-05", "Measurement": "Sample metric A", "Value": 73.0},
        {"Date": "2026-10-07", "Measurement": "Sample metric A", "Value": 75.0},
        {"Date": "2026-10-09", "Measurement": "Sample metric A", "Value": 74.0},
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
        "Medicine A, 1 tablet daily\n"
        "Medicine B, as prescribed"
    )

    st.session_state.meds_b = (
        "Medicine A, 1 tablet daily\n"
        "Medicine C, as prescribed"
    )

    st.session_state.classifier_question = (
        "How can I understand the symptoms I am experiencing?"
    )

    st.session_state.comparison_report = None
    st.session_state.medication_report = None
    st.session_state.classifier_result = None
    st.session_state.demo_loaded = True


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_intent_model():
    """
    Find the model relative to this script, not the current
    working directory. Also check common model subfolders.
    """
    base_dir = Path(__file__).resolve().parent

    candidate_paths = [
        base_dir / "intelliphr_intent_model.joblib",
        base_dir / "models" / "intelliphr_intent_model.joblib",
        base_dir / "model" / "intelliphr_intent_model.joblib",
        base_dir / "artifacts" / "intelliphr_intent_model.joblib",
    ]

    for path in candidate_paths:
        if path.is_file():
            return joblib.load(path), str(path)

    checked = "\n".join(str(path) for path in candidate_paths)

    raise FileNotFoundError(
        "Could not find intelliphr_intent_model.joblib. "
        "Locations checked:\n" + checked
    )


def split_entries(text):
    """Split user-entered notes on newlines, commas, or semicolons."""
    return [
        item.strip()
        for item in re.split(r"[\n,;]+", text)
        if item.strip()
    ]


def make_csv_download(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🩺 CareTrail")
    st.caption("Health Companion")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Visit Comparison",
            "Medical Timeline",
            "Medication Review",
            "Health Trends",
            "AI Question Classifier",
            "About CareTrail",
        ],
        key="navigation",
    )

    st.divider()

    st.markdown("### Demo controls")

    if st.button("Load Demo for All Checks", use_container_width=True):
        add_demo_data()
        st.rerun()

    if st.button("Clear All Session Data", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

    st.divider()
    st.caption("Academic healthcare information prototype")


# --------------------------------------------------
# COMMON HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>CareTrail Health Companion</h1>
        <p>
        Organize visit notes, compare medication lists,
        review sample trends, and explore AI-based question classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 1. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.header("Dashboard")
    st.write("Your overview of the CareTrail prototype.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Visit Records", len(st.session_state.visits))

    with col2:
        st.metric("Trend Data Points", len(st.session_state.measurements))

    with col3:
        st.metric(
            "Demo Data",
            "Loaded" if st.session_state.demo_loaded else "Not loaded",
        )

    st.subheader("Explore CareTrail")

    cards = [
        (
            "Visit Comparison",
            "Compare two sets of visit notes and identify text differences.",
        ),
        (
            "Medical Timeline",
            "Add and review visit records in chronological order.",
        ),
        (
            "Medication Review",
            "Compare two medication lists and find entries unique to each.",
        ),
        (
            "Health Trends",
            "Visualize sample numerical measurements over time.",
        ),
        (
            "AI Question Classifier",
            "Classify a question into one of the model's trained categories.",
        ),
    ]

    for start in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, (title, description) in zip(
            cols, cards[start:start + 2]
        ):
            with col:
                with st.container(border=True):
                    st.subheader(title)
                    st.write(description)

    st.info(
        "Select **Load Demo for All Checks** in the sidebar to populate "
        "the prototype with fictional sample data."
    )


# --------------------------------------------------
# 2. VISIT COMPARISON
# --------------------------------------------------

elif page == "Visit Comparison":
    st.header("Visit Comparison")
    st.write(
        "Compare two sets of visit notes and identify text differences."
    )

    st.caption(
        "Enter each item on a new line, or separate entries with commas "
        "or semicolons."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visit A")
        visit_date_a = st.date_input(
            "Visit A date",
            key="visit_date_a",
        )
        notes_a = st.text_area(
            "Visit A details",
            key="visit_a_notes",
            height=180,
            placeholder=(
                "Routine check-up\n"
                "Blood pressure recorded\n"
                "Follow-up planned"
            ),
        )

    with col2:
        st.subheader("Visit B")
        visit_date_b = st.date_input(
            "Visit B date",
            key="visit_date_b",
        )
        notes_b = st.text_area(
            "Visit B details",
            key="visit_b_notes",
            height=180,
            placeholder=(
                "Routine check-up\n"
                "Blood pressure recorded\n"
                "Follow-up appointment scheduled"
            ),
        )

    if st.button("Compare Visit Notes", use_container_width=True):
        entries_a = split_entries(notes_a)
        entries_b = split_entries(notes_b)

        set_a = set(entries_a)
        set_b = set(entries_b)

        st.session_state.comparison_report = {
            "matching": sorted(set_a & set_b),
            "only_a": sorted(set_a - set_b),
            "only_b": sorted(set_b - set_a),
            "date_a": str(visit_date_a),
            "date_b": str(visit_date_b),
        }

    report = st.session_state.comparison_report

    if report is not None:
        st.divider()
        st.subheader("Comparison Results")

        st.caption(
            f"Visit A: {report['date_a']}  |  "
            f"Visit B: {report['date_b']}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Matching Entries", len(report["matching"]))

        with col2:
            st.metric("Only in Visit A", len(report["only_a"]))

        with col3:
            st.metric("Only in Visit B", len(report["only_b"]))

        result_cols = st.columns(3)

        groups = [
            ("Matching Entries", report["matching"]),
            ("Only in Visit A", report["only_a"]),
            ("Only in Visit B", report["only_b"]),
        ]

        for col, (title, entries) in zip(result_cols, groups):
            with col:
                st.markdown(f"**{title}**")
                if entries:
                    for entry in entries:
                        st.write(f"• {entry}")
                else:
                    st.write("No entries.")

        export_rows = []

        for entry in report["matching"]:
            export_rows.append(
                {"Category": "Matching", "Entry": entry}
            )

        for entry in report["only_a"]:
            export_rows.append(
                {"Category": "Only in Visit A", "Entry": entry}
            )

        for entry in report["only_b"]:
            export_rows.append(
                {"Category": "Only in Visit B", "Entry": entry}
            )

        st.download_button(
            "Download Comparison CSV",
            data=make_csv_download(pd.DataFrame(
                export_rows, columns=["Category", "Entry"]
            )),
            file_name="caretrail_visit_comparison.csv",
            mime="text/csv",
        )

        st.warning(
            "This is exact text comparison, not clinical interpretation. "
            "Different wording may be treated as different entries."
        )


# --------------------------------------------------
# 3. MEDICAL TIMELINE
# --------------------------------------------------

elif page == "Medical Timeline":
    st.header("Medical Timeline")
    st.write("Add and review visit records.")

    with st.form("add_visit_form", clear_on_submit=True):
        st.subheader("Add a visit record")

        visit_date = st.date_input("Visit date")
        visit_type = st.text_input(
            "Visit type",
            placeholder="e.g. Routine check-up",
        )
        visit_notes = st.text_area(
            "Visit notes",
            placeholder="Enter a brief visit summary",
        )
        provider = st.text_input(
            "Provider or facility (optional)",
            placeholder="e.g. Demo Provider",
        )

        submitted = st.form_submit_button(
            "Add Visit Record",
            use_container_width=True,
        )

        if submitted:
            if not visit_type.strip():
                st.error("Please enter a visit type.")
            elif not visit_notes.strip():
                st.error("Please enter visit notes.")
            else:
                st.session_state.visits.append(
                    {
                        "Date": str(visit_date),
                        "Visit Type": visit_type.strip(),
                        "Notes": visit_notes.strip(),
                        "Provider": provider.strip() or "Not specified",
                    }
                )
                st.success("Visit record added.")

    st.divider()
    st.subheader("Visit history")

    if st.session_state.visits:
        visits_df = pd.DataFrame(st.session_state.visits)

        if "Date" in visits_df.columns:
            visits_df = visits_df.sort_values(
                "Date", ascending=False
            )

        st.dataframe(
            visits_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Visit History CSV",
            data=make_csv_download(visits_df),
            file_name="caretrail_visit_history.csv",
            mime="text/csv",
        )

        st.caption(
            "These records are temporary and stored in session memory."
        )
    else:
        st.info(
            "No visit records yet. Load demo data or add a record above."
        )


# --------------------------------------------------
# 4. MEDICATION REVIEW
# --------------------------------------------------

elif page == "Medication Review":
    st.header("Medication Review")
    st.write(
        "Compare two medication lists to identify entries that match "
        "or appear only in one list."
    )

    st.warning(
        "This tool compares text only. It does not check dosages, "
        "interactions, safety, or whether a medication should be stopped."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Medication List A")
        meds_a = st.text_area(
            "Enter List A",
            key="meds_a",
            height=180,
            placeholder="Medicine A, 1 tablet daily",
        )

    with col2:
        st.subheader("Medication List B")
        meds_b = st.text_area(
            "Enter List B",
            key="meds_b",
            height=180,
            placeholder="Medicine A, 1 tablet daily",
        )

    if st.button("Compare Medication Lists", use_container_width=True):
        list_a = split_entries(meds_a)
        list_b = split_entries(meds_b)

        normalized_a = {item.casefold(): item for item in list_a}
        normalized_b = {item.casefold(): item for item in list_b}

        keys_a = set(normalized_a)
        keys_b = set(normalized_b)

        st.session_state.medication_report = {
            "matching": [
                normalized_a[key] for key in sorted(keys_a & keys_b)
            ],
            "only_a": [
                normalized_a[key] for key in sorted(keys_a - keys_b)
            ],
            "only_b": [
                normalized_b[key] for key in sorted(keys_b - keys_a)
            ],
        }

    report = st.session_state.medication_report

    if report is not None:
        st.divider()
        st.subheader("Medication Comparison Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Matching", len(report["matching"]))

        with col2:
            st.metric("Only in List A", len(report["only_a"]))

        with col3:
            st.metric("Only in List B", len(report["only_b"]))

        result_cols = st.columns(3)

        groups = [
            ("Matching", report["matching"]),
            ("Only in List A", report["only_a"]),
            ("Only in List B", report["only_b"]),
        ]

        for col, (title, entries) in zip(result_cols, groups):
            with col:
                st.markdown(f"**{title}**")
                if entries:
                    for entry in entries:
                        st.write(f"• {entry}")
                else:
                    st.write("No entries.")

        rows = []

        for category in ("matching", "only_a", "only_b"):
            for entry in report[category]:
                rows.append(
                    {"Category": category, "Medication Entry": entry}
                )

        st.download_button(
            "Download Medication Comparison CSV",
            data=make_csv_download(pd.DataFrame(
                rows, columns=["Category", "Medication Entry"]
            )),
            file_name="caretrail_medication_comparison.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# 5. HEALTH TRENDS
# --------------------------------------------------

elif page == "Health Trends":
    st.header("Health Trends")
    st.write(
        "Review a simple visualization of numerical measurements over time."
    )

    st.caption(
        "The built-in examples are fictional demonstration data, "
        "not real patient measurements."
    )

    with st.form("add_measurement_form", clear_on_submit=True):
        st.subheader("Add a measurement")

        measurement_date = st.date_input("Measurement date")
        measurement_name = st.text_input(
            "Measurement name",
            placeholder="e.g. Sample metric A",
        )
        measurement_value = st.number_input(
            "Numeric value",
            value=0.0,
            format="%.2f",
        )

        add_measurement = st.form_submit_button(
            "Add Measurement",
            use_container_width=True,
        )

        if add_measurement:
            if not measurement_name.strip():
                st.error("Please enter a measurement name.")
            else:
                st.session_state.measurements.append(
                    {
                        "Date": str(measurement_date),
                        "Measurement": measurement_name.strip(),
                        "Value": float(measurement_value),
                    }
                )
                st.success("Measurement added.")

    st.divider()
    st.subheader("Measurement history")

    if st.session_state.measurements:
        trends_df = pd.DataFrame(st.session_state.measurements)
        trends_df["Date"] = pd.to_datetime(
            trends_df["Date"], errors="coerce"
        )
        trends_df["Value"] = pd.to_numeric(
            trends_df["Value"], errors="coerce"
        )
        trends_df = trends_df.dropna(subset=["Date", "Value"])

        measurement_names = sorted(
            trends_df["Measurement"].dropna().unique().tolist()
        )

        selected_measurement = st.selectbox(
            "Choose a measurement",
            measurement_names,
        )

        selected_df = trends_df[
            trends_df["Measurement"] == selected_measurement
        ].sort_values("Date")

        if not selected_df.empty:
            st.line_chart(
                selected_df.set_index("Date")["Value"],
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Latest Value",
                    f"{selected_df['Value'].iloc[-1]:.2f}",
                )

            with col2:
                st.metric(
                    "Minimum",
                    f"{selected_df['Value'].min():.2f}",
                )

            with col3:
                st.metric(
                    "Maximum",
                    f"{selected_df['Value'].max():.2f}",
                )

        st.dataframe(
            trends_df.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Measurements CSV",
            data=make_csv_download(trends_df),
            file_name="caretrail_health_trends.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No measurement data available. Load demo data or add a "
            "measurement above."
        )


# --------------------------------------------------
# 6. AI QUESTION CLASSIFIER
# --------------------------------------------------

elif page == "AI Question Classifier":
    st.header("AI Question Classifier")

    st.write(
        "Enter a healthcare-related question to classify it using "
        "your trained machine-learning model."
    )

    st.caption(
        "The classifier identifies text categories. It does not diagnose "
        "conditions, recommend treatment, or replace a healthcare professional."
    )

    question = st.text_area(
        "Enter your question",
        key="classifier_question",
        height=120,
        placeholder=(
            "e.g. How can I understand the symptoms I am experiencing?"
        ),
    )

    if st.button("Classify Question", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                model, model_path = load_intent_model()

                prediction = model.predict([question.strip()])[0]

                probability = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        [question.strip()]
                    )[0]
                    probability = float(max(probabilities))

                st.session_state.classifier_result = {
                    "prediction": str(prediction),
                    "probability": probability,
                    "model_path": model_path,
                    "question": question.strip(),
                    "error": None,
                }

            except Exception as exc:
                st.session_state.classifier_result = {
                    "prediction": None,
                    "probability": None,
                    "model_path": None,
                    "question": question.strip(),
                    "error": str(exc),
                }

    result = st.session_state.classifier_result

    if result is not None:
        st.divider()
        st.subheader("Classification Result")

        if result["error"]:
            st.error(
                "The trained model could not be loaded or the question "
                "could not be classified."
            )

            st.markdown("**Technical details**")
            st.code(result["error"])

            st.info(
                "Check that the model file is committed to GitHub, "
                "that its filename and folder match the app, and that "
                "the required packages are installed."
            )

        else:
            st.success(
                f"Predicted category: {result['prediction']}"
            )

            if result["probability"] is not None:
                st.metric(
                    "Model confidence score",
                    f"{result['probability'] * 100:.1f}%",
                )

                st.caption(
                    "This score reflects the model's output, not the "
                    "medical accuracy or urgency of the question."
                )

            st.caption(
                f"Model loaded from: {result['model_path']}"
            )

            st.markdown("**Trained categories**")
            st.write(
                "- `symptom_help`\n"
                "- `medication_query`\n"
                "- `records_query`\n"
                "- `health_trends`\n"
                "- `general_help`"
            )

            st.info(
                "The prediction is an educational text classification "
                "result, not medical advice."
            )


# --------------------------------------------------
# 7. ABOUT CARETRAIL
# --------------------------------------------------

elif page == "About CareTrail":
    st.header("About CareTrail")

    st.write(
        """
        CareTrail is an academic healthcare information prototype
        designed to demonstrate basic health-record organization,
        text comparison, numerical trend visualization, and
        machine-learning-based question classification.
        """
    )

    st.subheader("Features")

    st.markdown(
        """
        - **Dashboard:** Summary of available demo records.
        - **Visit Comparison:** Text-based comparison of two visit notes.
        - **Medical Timeline:** Temporary visit record management.
        - **Medication Review:** Text comparison of two medication lists.
        - **Health Trends:** Visualization of numerical measurements.
        - **AI Question Classifier:** Prediction using a trained text model.
        """
    )

    st.subheader("Technology")

    st.markdown(
        """
        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - Joblib
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - Data entered in this prototype is not stored in a permanent database.
        - The classifier is limited to the categories used during training.
        - Demo records are fictional and are not real patient data.
        - The application does not provide medical diagnosis or treatment.
        """
    )

    st.warning(
        "CareTrail is an academic healthcare information prototype. "
        "It is not a substitute for professional medical advice."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p style="text-align:center; color:#64748b; font-size:0.85rem;">
    CareTrail · Academic healthcare information prototype<br>
    For educational demonstration only. Not a substitute for
    professional medical advice.
    </p>
    """,
    unsafe_allow_html=True,
)
