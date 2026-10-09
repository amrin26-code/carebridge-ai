
from pathlib import Path
from datetime import date, timedelta
import re

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareTrail | Health Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
        color: #172b4d;
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,251,0.95);
    }

    [data-testid="stSidebar"] {
        background: #10243a;
    }

    [data-testid="stSidebar"] * {
        color: #f4f8fc;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #b6c9dc;
    }

    h1, h2, h3, h4, p, label {
        color: #172b4d;
    }

    h1 {
        letter-spacing: -1px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf1;
        padding: 19px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(20,40,70,0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #61718a;
    }

    div[data-testid="stMetricValue"] {
        color: #10243a;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: #087f8c;
        border-color: #087f8c;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #066875;
        border-color: #066875;
        color: white;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-radius: 14px;
        border-color: #e5eaf1;
    }

    .hero {
        background: linear-gradient(120deg, #10243a 0%, #14576c 58%, #087f8c 100%);
        padding: 27px 30px;
        border-radius: 18px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white !important;
        margin-bottom: 5px;
        font-size: 2.1rem;
    }

    .hero p {
        color: #e1f4f6 !important;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #8fe1dc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-intro {
        color: #64748b;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .footer {
        color: #718096;
        text-align: center;
        font-size: 0.82rem;
        padding: 14px 0;
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

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# 4. HELPERS
# =========================================================

def split_entries(text):
    """Split text into individual entries."""
    return [
        item.strip()
        for item in re.split(r"[\n,;]+", text or "")
        if item.strip()
    ]


def export_csv(data):
    """Return a CSV download payload."""
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


def add_visit(record):
    st.session_state.visits.append(record)


def add_measurement(record):
    st.session_state.measurements.append(record)


def load_demo_data():
    """Load clearly labelled fictional demonstration records."""

    st.session_state.visits = [
        {
            "ID": 1,
            "Date": "2026-10-01",
            "Visit Type": "Routine check-up",
            "Notes": "Routine visit; blood pressure recorded.",
            "Provider": "Demo Provider",
        },
        {
            "ID": 2,
            "Date": "2026-10-04",
            "Visit Type": "Laboratory review",
            "Notes": "Laboratory report reviewed.",
            "Provider": "Demo Provider",
        },
        {
            "ID": 3,
            "Date": "2026-10-07",
            "Visit Type": "Follow-up",
            "Notes": "Follow-up appointment planned.",
            "Provider": "Demo Provider",
        },
        {
            "ID": 4,
            "Date": "2026-10-09",
            "Visit Type": "Review appointment",
            "Notes": "Sample record for dashboard testing.",
            "Provider": "Demo Provider",
        },
    ]

    st.session_state.measurements = [
        {"Date": "2026-10-01", "Measurement": "Sample metric A", "Value": 72.0},
        {"Date": "2026-10-02", "Measurement": "Sample metric A", "Value": 73.5},
        {"Date": "2026-10-03", "Measurement": "Sample metric A", "Value": 71.8},
        {"Date": "2026-10-04", "Measurement": "Sample metric A", "Value": 74.2},
        {"Date": "2026-10-05", "Measurement": "Sample metric A", "Value": 73.0},
        {"Date": "2026-10-06", "Measurement": "Sample metric A", "Value": 75.1},
        {"Date": "2026-10-07", "Measurement": "Sample metric A", "Value": 74.0},
        {"Date": "2026-10-08", "Measurement": "Sample metric A", "Value": 76.2},
        {"Date": "2026-10-09", "Measurement": "Sample metric A", "Value": 74.5},
        {"Date": "2026-10-01", "Measurement": "Sample metric B", "Value": 6.8},
        {"Date": "2026-10-03", "Measurement": "Sample metric B", "Value": 7.0},
        {"Date": "2026-10-05", "Measurement": "Sample metric B", "Value": 6.7},
        {"Date": "2026-10-07", "Measurement": "Sample metric B", "Value": 7.2},
        {"Date": "2026-10-09", "Measurement": "Sample metric B", "Value": 7.1},
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


def reset_demo_and_records():
    """Clear temporary data and analysis results."""
    for key, default_value in DEFAULTS.items():
        st.session_state[key] = default_value


# =========================================================
# 5. MODEL LOADING
# =========================================================

@st.cache_resource
def load_intent_model():
    """
    Look for the trained model in likely locations relative
    to app.py. Raise a useful error if it cannot be found.
    """
    base = Path(__file__).resolve().parent

    paths = [
        base / "intelliphr_intent_model.joblib",
        base / "models" / "intelliphr_intent_model.joblib",
        base / "model" / "intelliphr_intent_model.joblib",
        base / "artifacts" / "intelliphr_intent_model.joblib",
    ]

    for path in paths:
        if path.is_file():
            return joblib.load(path), str(path)

    checked = "\n".join(str(path) for path in paths)

    raise FileNotFoundError(
        "The trained model file was not found. Locations checked:\n"
        + checked
    )


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🩺 CareTrail")
    st.caption("HEALTH COMPANION")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Medical Timeline",
            "Visit Comparison",
            "Medication Review",
            "Health Trends",
            "AI Question Classifier",
            "About CareTrail",
        ],
        key="page_navigation",
    )

    st.divider()
    st.markdown("### Quick actions")

    if st.button("Load Demo Data", use_container_width=True):
        load_demo_data()
        st.rerun()

    if st.button("Clear Temporary Data", use_container_width=True):
        reset_demo_and_records()
        st.rerun()

    st.divider()

    st.caption(
        "Demo records are fictional. Data is held in temporary session memory."
    )


# =========================================================
# 7. PAGE HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PERSONAL HEALTH INFORMATION</div>
        <h1>CareTrail Health Companion</h1>
        <p>
        One workspace for visit records, comparisons, numerical trends,
        and AI-powered text classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 8. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Your dashboard")
    st.markdown(
        '<div class="section-intro">A quick overview of the current session.</div>',
        unsafe_allow_html=True,
    )

    visits_df = pd.DataFrame(st.session_state.visits)
    measurements_df = pd.DataFrame(st.session_state.measurements)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Visit records", len(visits_df))

    with col2:
        st.metric("Measurements", len(measurements_df))

    with col3:
        metrics_count = (
            measurements_df["Measurement"].nunique()
            if not measurements_df.empty
            and "Measurement" in measurements_df.columns
            else 0
        )
        st.metric("Metric types", metrics_count)

    with col4:
        st.metric(
            "Demo status",
            "Loaded" if st.session_state.demo_loaded else "Not loaded",
        )

    st.markdown("### Explore your workspace")

    cards = [
        ("🗂️", "Medical Timeline", "Create and manage visit records."),
        ("↔️", "Visit Comparison", "Compare visit notes side by side."),
        ("💊", "Medication Review", "Identify text differences in medication lists."),
        ("📈", "Health Trends", "Explore measurements with interactive charts."),
        ("🤖", "AI Question Classifier", "Classify questions with your trained model."),
    ]

    for start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (emoji, title, description) in zip(
            cols, cards[start:start + 3]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {emoji} {title}")
                    st.write(description)

    st.markdown("### Recent visits")

    if not visits_df.empty:
        recent = visits_df.copy()
        if "Date" in recent.columns:
            recent = recent.sort_values("Date", ascending=False)

        st.dataframe(
            recent.head(5),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No visit records yet. Select Load Demo Data or add a visit.")

    st.markdown("### Getting started")

    if st.button("Load a complete demo workspace", type="primary"):
        load_demo_data()
        st.rerun()


# =========================================================
# 9. MEDICAL TIMELINE
# =========================================================

elif page == "Medical Timeline":

    st.subheader("Medical Timeline")
    st.markdown(
        '<div class="section-intro">Record and review appointments and visit notes.</div>',
        unsafe_allow_html=True,
    )

    with st.form("visit_form", clear_on_submit=True):
        st.markdown("### Add a visit record")

        col1, col2 = st.columns(2)

        with col1:
            visit_date = st.date_input("Visit date", value=date.today())
            visit_type = st.selectbox(
                "Visit type",
                [
                    "Routine check-up",
                    "Follow-up",
                    "Laboratory review",
                    "Specialist consultation",
                    "Other",
                ],
            )

        with col2:
            provider = st.text_input(
                "Provider or facility",
                placeholder="Optional",
            )

        notes = st.text_area(
            "Visit notes",
            height=120,
            placeholder="Write a brief summary of the visit.",
        )

        save_visit = st.form_submit_button(
            "Save visit record",
            type="primary",
            use_container_width=True,
        )

        if save_visit:
            if not notes.strip():
                st.error("Please enter visit notes before saving.")
            else:
                next_id = max(
                    [int(v.get("ID", 0)) for v in st.session_state.visits]
                    + [0]
                ) + 1

                add_visit(
                    {
                        "ID": next_id,
                        "Date": visit_date.isoformat(),
                        "Visit Type": visit_type,
                        "Notes": notes.strip(),
                        "Provider": provider.strip() or "Not specified",
                    }
                )
                st.success("Visit record saved for this session.")

    st.divider()
    st.markdown("### Visit history")

    if st.session_state.visits:
        visits_df = pd.DataFrame(st.session_state.visits)
        visits_df["Date"] = pd.to_datetime(
            visits_df["Date"], errors="coerce"
        )
        visits_df = visits_df.sort_values("Date", ascending=False)

        filter_text = st.text_input(
            "Search visit records",
            placeholder="Search notes, visit type, or provider",
        )

        if filter_text.strip():
            mask = visits_df.astype(str).apply(
                lambda col: col.str.contains(
                    filter_text, case=False, na=False
                )
            ).any(axis=1)
            visits_df = visits_df[mask]

        st.dataframe(
            visits_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_df = visits_df.copy()
        csv_df["Date"] = csv_df["Date"].dt.strftime("%Y-%m-%d")

        st.download_button(
            "Download visit history",
            data=export_csv(csv_df.to_dict("records")),
            file_name="caretrail_visit_history.csv",
            mime="text/csv",
        )

        with st.expander("Delete a visit record"):
            ids = [
                v.get("ID", i)
                for i, v in enumerate(st.session_state.visits)
            ]

            labels = {
                v.get("ID", i):
                f"{v.get('Date')} — {v.get('Visit Type')}"
                for i, v in enumerate(st.session_state.visits)
            }

            selected_id = st.selectbox(
                "Choose a record",
                ids,
                format_func=lambda item: labels[item],
            )

            if st.button("Delete selected visit"):
                st.session_state.visits = [
                    v for i, v in enumerate(st.session_state.visits)
                    if v.get("ID", i) != selected_id
                ]
                st.rerun()

    else:
        st.info("No visit records yet. Add a visit above or load demo data.")


# =========================================================
# 10. VISIT COMPARISON
# =========================================================

elif page == "Visit Comparison":

    st.subheader("Visit Comparison")
    st.markdown(
        '<div class="section-intro">Compare two sets of notes and identify exact text differences.</div>',
        unsafe_allow_html=True,
    )

    visit_records = st.session_state.visits

    if len(visit_records) >= 2:
        use_saved_visits = st.toggle(
            "Use saved visit records",
            value=False,
        )
    else:
        use_saved_visits = False

    if use_saved_visits:
        labels = {
            i: f"{v.get('Date')} — {v.get('Visit Type')}"
            for i, v in enumerate(visit_records)
        }

        col1, col2 = st.columns(2)

        with col1:
            a_index = st.selectbox(
                "First visit",
                range(len(visit_records)),
                format_func=lambda i: labels[i],
                key="saved_visit_a",
            )

        with col2:
            b_index = st.selectbox(
                "Second visit",
                range(len(visit_records)),
                index=1,
                format_func=lambda i: labels[i],
                key="saved_visit_b",
            )

        notes_a = visit_records[a_index]["Notes"]
        notes_b = visit_records[b_index]["Notes"]

        st.text_area("First visit notes", value=notes_a, disabled=True)
        st.text_area("Second visit notes", value=notes_b, disabled=True)

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Visit A")
            notes_a = st.text_area(
                "Enter Visit A notes",
                key="visit_a_notes",
                height=180,
                placeholder="Enter each item on a separate line.",
            )

        with col2:
            st.markdown("#### Visit B")
            notes_b = st.text_area(
                "Enter Visit B notes",
                key="visit_b_notes",
                height=180,
                placeholder="Enter each item on a separate line.",
            )

    if st.button("Compare notes", type="primary", use_container_width=True):
        entries_a = split_entries(notes_a)
        entries_b = split_entries(notes_b)

        if not entries_a or not entries_b:
            st.warning("Enter notes for both visits before comparing.")
            st.session_state.comparison_report = None
        else:
            set_a = {x.casefold(): x for x in entries_a}
            set_b = {x.casefold(): x for x in entries_b}

            common_keys = set(set_a) & set(set_b)
            only_a_keys = set(set_a) - set(set_b)
            only_b_keys = set(set_b) - set(set_a)

            st.session_state.comparison_report = {
                "matching": [set_a[k] for k in sorted(common_keys)],
                "only_a": [set_a[k] for k in sorted(only_a_keys)],
                "only_b": [set_b[k] for k in sorted(only_b_keys)],
            }

    report = st.session_state.comparison_report

    if report is not None:
        st.divider()
        st.markdown("### Comparison results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Matching entries", len(report["matching"]))
        c2.metric("Only in A", len(report["only_a"]))
        c3.metric("Only in B", len(report["only_b"]))

        result_cols = st.columns(3)

        for col, title, key in zip(
            result_cols,
            ["Matching", "Only in Visit A", "Only in Visit B"],
            ["matching", "only_a", "only_b"],
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if report[key]:
                        for entry in report[key]:
                            st.write(f"• {entry}")
                    else:
                        st.caption("No entries")

        rows = []
        for key, category in [
            ("matching", "Matching"),
            ("only_a", "Only in Visit A"),
            ("only_b", "Only in Visit B"),
        ]:
            for entry in report[key]:
                rows.append({"Category": category, "Entry": entry})

        st.download_button(
            "Download comparison report",
            data=export_csv(rows),
            file_name="caretrail_visit_comparison.csv",
            mime="text/csv",
        )

        st.caption(
            "Text comparison only. Similar meanings expressed in different "
            "words may still be treated as different entries."
        )


# =========================================================
# 11. MEDICATION REVIEW
# =========================================================

elif page == "Medication Review":

    st.subheader("Medication Review")
    st.markdown(
        '<div class="section-intro">Compare two lists to identify entries added or removed.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "This is a text comparison tool. It does not verify drug safety, "
        "dosages, interactions, or treatment suitability."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current or earlier list")
        meds_a = st.text_area(
            "Medication list A",
            key="meds_a",
            height=180,
            placeholder="Enter one medicine per line.",
        )

    with col2:
        st.markdown("#### New or later list")
        meds_b = st.text_area(
            "Medication list B",
            key="meds_b",
            height=180,
            placeholder="Enter one medicine per line.",
        )

    if st.button(
        "Compare medication lists",
        type="primary",
        use_container_width=True,
    ):
        entries_a = split_entries(meds_a)
        entries_b = split_entries(meds_b)

        dict_a = {x.casefold(): x for x in entries_a}
        dict_b = {x.casefold(): x for x in entries_b}

        st.session_state.medication_report = {
            "matching": [dict_a[k] for k in sorted(set(dict_a) & set(dict_b))],
            "only_a": [dict_a[k] for k in sorted(set(dict_a) - set(dict_b))],
            "only_b": [dict_b[k] for k in sorted(set(dict_b) - set(dict_a))],
        }

    report = st.session_state.medication_report

    if report is not None:
        st.divider()
        st.markdown("### Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Matching", len(report["matching"]))
        c2.metric("Only in list A", len(report["only_a"]))
        c3.metric("Only in list B", len(report["only_b"]))

        result_cols = st.columns(3)

        for col, title, key in zip(
            result_cols,
            ["Matching", "Only in list A", "Only in list B"],
            ["matching", "only_a", "only_b"],
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if report[key]:
                        for item in report[key]:
                            st.write(f"• {item}")
                    else:
                        st.caption("No entries")

        rows = []
        for key, label in [
            ("matching", "Matching"),
            ("only_a", "Only in list A"),
            ("only_b", "Only in list B"),
        ]:
            for item in report[key]:
                rows.append({"Category": label, "Entry": item})

        st.download_button(
            "Download medication comparison",
            data=export_csv(rows),
            file_name="caretrail_medication_comparison.csv",
            mime="text/csv",
        )


# =========================================================
# 12. HEALTH TRENDS
# =========================================================

elif page == "Health Trends":

    st.subheader("Health Trends")
    st.markdown(
        '<div class="section-intro">Explore numerical measurements using interactive charts and summary statistics.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Demo values are fictional and use generic sample metrics. "
        "They are not medical reference ranges or real patient measurements."
    )

    # Form to add a new measurement
    with st.form("measurement_form", clear_on_submit=True):
        st.markdown("### Add measurement")

        col1, col2, col3 = st.columns(3)

        with col1:
            measure_date = st.date_input(
                "Measurement date",
                value=date.today(),
            )

        with col2:
            measure_name = st.text_input(
                "Measurement name",
                placeholder="e.g. Weight, blood pressure, sample metric",
            )

        with col3:
            measure_value = st.number_input(
                "Numeric value",
                value=0.0,
                step=0.1,
                format="%.2f",
            )

        save_measurement = st.form_submit_button(
            "Add measurement",
            type="primary",
            use_container_width=True,
        )

        if save_measurement:
            if not measure_name.strip():
                st.error("Enter a measurement name.")
            else:
                add_measurement(
                    {
                        "Date": measure_date.isoformat(),
                        "Measurement": measure_name.strip(),
                        "Value": float(measure_value),
                    }
                )
                st.success("Measurement added.")
                st.rerun()

    st.divider()

    if not st.session_state.measurements:
        st.warning(
            "There are no measurements to plot. Click Load Demo Data in "
            "the sidebar or add at least one measurement above."
        )

    else:
        df = pd.DataFrame(st.session_state.measurements)

        required_columns = {"Date", "Measurement", "Value"}

        if not required_columns.issubset(df.columns):
            st.error(
                "Measurement data has an unexpected format. Clear the "
                "temporary data and load the demo dataset again."
            )
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            df["Measurement"] = df["Measurement"].astype(str)

            df = df.dropna(
                subset=["Date", "Value", "Measurement"]
            ).sort_values("Date")

            if df.empty:
                st.warning(
                    "No valid numerical measurements are available to plot."
                )
            else:
                st.markdown("### Trend explorer")

                metric_names = sorted(
                    df["Measurement"].unique().tolist()
                )

                selected_metric = st.selectbox(
                    "Select a measurement to visualize",
                    metric_names,
                )

                metric_df = df[
                    df["Measurement"] == selected_metric
                ].copy()

                min_date = metric_df["Date"].min().date()
                max_date = metric_df["Date"].max().date()

                if min_date == max_date:
                    start_date, end_date = min_date, max_date
                else:
                    selected_range = st.date_input(
                        "Filter date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )

                    if isinstance(selected_range, (tuple, list)):
                        if len(selected_range) == 2:
                            start_date, end_date = selected_range
                        elif len(selected_range) == 1:
                            start_date = end_date = selected_range[0]
                        else:
                            start_date, end_date = min_date, max_date
                    else:
                        start_date = end_date = selected_range

                metric_df = metric_df[
                    (metric_df["Date"].dt.date >= start_date)
                    & (metric_df["Date"].dt.date <= end_date)
                ].sort_values("Date")

                if metric_df.empty:
                    st.warning(
                        "No data points fall within the selected date range."
                    )
                else:
                    latest = metric_df.iloc[-1]["Value"]
                    minimum = metric_df["Value"].min()
                    maximum = metric_df["Value"].max()
                    average = metric_df["Value"].mean()

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("Latest", f"{latest:,.2f}")
                    c2.metric("Average", f"{average:,.2f}")
                    c3.metric("Minimum", f"{minimum:,.2f}")
                    c4.metric("Maximum", f"{maximum:,.2f}")

                    st.markdown("### Measurement over time")

                    chart_df = metric_df[
                        ["Date", "Value"]
                    ].copy()

                    chart_df["Date"] = chart_df["Date"].dt.strftime(
                        "%Y-%m-%d"
                    )

                    fig = px.line(
                        chart_df,
                        x="Date",
                        y="Value",
                        markers=True,
                        title=f"{selected_metric} — measurement trend",
                        labels={
                            "Date": "Measurement date",
                            "Value": "Recorded value",
                        },
                        template="plotly_white",
                    )

                    fig.update_traces(
                        line=dict(width=3, color="#087f8c"),
                        marker=dict(size=8, color="#087f8c"),
                        hovertemplate=(
                            "<b>Date:</b> %{x}<br>"
                            "<b>Value:</b> %{y:.2f}"
                            "<extra></extra>"
                        ),
                    )

                    fig.update_layout(
                        height=430,
                        margin=dict(l=20, r=20, t=65, b=20),
                        title_font_size=18,
                        hovermode="x unified",
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="#e8edf4",
                            title="Date",
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="#e8edf4",
                            title="Value",
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.markdown("### Data points")

                    display_df = metric_df.copy()
                    display_df["Date"] = display_df["Date"].dt.strftime(
                        "%Y-%m-%d"
                    )

                    st.dataframe(
                        display_df[["Date", "Measurement", "Value"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download filtered measurement data",
                        data=export_csv(
                            display_df[
                                ["Date", "Measurement", "Value"]
                            ].to_dict("records")
                        ),
                        file_name="caretrail_health_trends.csv",
                        mime="text/csv",
                    )

    # Show all recorded data, including other metrics
    if st.session_state.measurements:
        with st.expander("View all measurements"):
            all_df = pd.DataFrame(st.session_state.measurements)
            st.dataframe(
                all_df,
                use_container_width=True,
                hide_index=True,
            )

            if not all_df.empty:
                st.download_button(
                    "Download all measurements",
                    data=export_csv(all_df.to_dict("records")),
                    file_name="caretrail_all_measurements.csv",
                    mime="text/csv",
                    key="download_all_measurements",
                )

        with st.expander("Delete a measurement"):
            all_measurements = st.session_state.measurements

            selected_row = st.selectbox(
                "Select a measurement",
                range(len(all_measurements)),
                format_func=lambda i: (
                    f"{all_measurements[i]['Date']} — "
                    f"{all_measurements[i]['Measurement']} — "
                    f"{all_measurements[i]['Value']}"
                ),
            )

            if st.button("Delete selected measurement"):
                st.session_state.measurements.pop(selected_row)
                st.rerun()


# =========================================================
# 13. AI QUESTION CLASSIFIER
# =========================================================

elif page == "AI Question Classifier":

    st.subheader("AI Question Classifier")
    st.markdown(
        '<div class="section-intro">Classify the text of a question using your trained machine-learning model.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "This model categorizes text. It does not diagnose diseases, "
        "assess emergencies, or recommend treatment."
    )

    question = st.text_area(
        "Enter a question",
        key="classifier_question",
        height=130,
        placeholder="Type a question to classify...",
    )

    if st.button(
        "Classify question",
        type="primary",
        use_container_width=True,
    ):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                model, model_path = load_intent_model()
                prediction = model.predict([question.strip()])[0]

                probabilities = None

                if hasattr(model, "predict_proba"):
                    values = model.predict_proba([question.strip()])[0]
                    probabilities = sorted(
                        zip(model.classes_, values),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                st.session_state.classifier_result = {
                    "error": None,
                    "question": question.strip(),
                    "prediction": str(prediction),
                    "model_path": model_path,
                    "probabilities": (
                        [(str(label), float(score))
                         for label, score in probabilities]
                        if probabilities is not None
                        else None
                    ),
                }

            except Exception as exc:
                st.session_state.classifier_result = {
                    "error": str(exc),
                    "question": question.strip(),
                    "prediction": None,
                    "model_path": None,
                    "probabilities": None,
                }

    result = st.session_state.classifier_result

    if result is not None:
        st.divider()
        st.markdown("### Model output")

        if result["error"]:
            st.error("The classifier could not complete the prediction.")
            st.code(result["error"])

            st.markdown("**Check the following:**")
            st.markdown(
                """
                1. The actual `.joblib` file is committed to GitHub.
                2. Its filename and folder match the path in this app.
                3. `requirements.txt` contains compatible versions of
                   `scikit-learn` and `joblib`.
                4. The file is not a Git LFS pointer instead of the model.
                """
            )

        else:
            st.success(
                f"Predicted category: **{result['prediction']}**"
            )

            st.caption(f"Loaded model: `{result['model_path']}`")

            if result["probabilities"] is not None:
                st.markdown("#### Scores by category")

                score_df = pd.DataFrame(
                    result["probabilities"],
                    columns=["Category", "Model score"],
                )

                score_df["Model score"] *= 100

                score_fig = px.bar(
                    score_df.sort_values("Model score"),
                    x="Model score",
                    y="Category",
                    orientation="h",
                    text=score_df.sort_values(
                        "Model score"
                    )["Model score"].map(lambda x: f"{x:.1f}%"),
                    labels={
                        "Model score": "Model score (%)",
                        "Category": "Category",
                    },
                    template="plotly_white",
                )

                score_fig.update_traces(
                    marker_color="#087f8c",
                    textposition="outside",
                    cliponaxis=False,
                )

                score_fig.update_layout(
                    height=340,
                    margin=dict(l=20, r=65, t=20, b=20),
                    xaxis=dict(range=[0, 110]),
                )

                st.plotly_chart(
                    score_fig,
                    use_container_width=True,
                )

                st.caption(
                    "Scores are model outputs, not clinical confidence "
                    "or measures of medical risk."
                )

            st.markdown("**Model categories**")
            st.code(
                "symptom_help\n"
                "medication_query\n"
                "records_query\n"
                "health_trends\n"
                "general_help"
            )

    st.markdown("### Try an example")

    examples = {
        "Symptoms": "How can I understand the symptoms I am experiencing?",
        "Medication": "I have a question about my medication list.",
        "Records": "How can I view my previous visit records?",
        "Health trends": "How can I review changes in my measurements?",
        "General help": "Can you explain how this application works?",
    }

    cols = st.columns(5)

    for col, (label, text) in zip(cols, examples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.classifier_question = text
                st.rerun()


# =========================================================
# 14. ABOUT
# =========================================================

elif page == "About CareTrail":

    st.subheader("About CareTrail")

    st.write(
        """
        CareTrail is an academic healthcare information prototype that
        demonstrates simple record organization, text comparisons,
        numerical visualization, and trained-model text classification.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Features")
            st.markdown(
                """
                - Medical visit timeline
                - Visit-note comparison
                - Medication-list comparison
                - Interactive measurement charts
                - Machine-learning text classification
                - CSV exports
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Technology")
            st.markdown(
                """
                - Python
                - Streamlit
                - Pandas
                - Plotly
                - Scikit-learn
                - Joblib
                """
            )

    st.markdown("### Data and privacy limitations")

    st.markdown(
        """
        - Data is held in temporary session memory.
        - This prototype does not use a permanent database.
        - Sample data is fictional.
        - The classifier is limited to its training categories.
        - This application is not a clinical decision-support system.
        """
    )

    st.warning(
        "Do not enter identifiable patient information into this academic demo."
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>CareTrail</b> · Academic healthcare information prototype<br>
        For educational demonstration only. Not a substitute for professional
        medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

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
