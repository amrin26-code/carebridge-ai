
from pathlib import Path
from datetime import date, datetime, timedelta
import json
import re
import io
import html

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CareTrail | Personal Health",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "intelliphr_intent_model.joblib"

DEFAULTS = {
    "visits": [],
    "notes": [],
    "vitals": [],
    "symptoms": [],
    "medications": [],
    "documents": [],
    "classifier_result": None,
    "classifier_question": "",
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default.copy() if isinstance(default, list) else default


# ============================================================
# 2. CONTRAST-SAFE STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    .stApp {
        background: #f3f6fa !important;
        color: #172b4d !important;
    }

    [data-testid="stHeader"] {
        background: #f3f6fa !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #f3f6fa !important;
    }

    [data-testid="stMain"] {
        background: #f3f6fa !important;
    }

    /* Main content: consistently dark text on light backgrounds */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] span,
    [data-testid="stMain"] small {
        color: #172b4d;
    }

    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #52647b !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #10243a !important;
    }

    [data-testid="stSidebar"] * {
        color: #f4f8fc !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #c5d4e3 !important;
    }

    /* Inputs: white field + dark text */
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="textarea"] > div {
        background: #ffffff !important;
        color: #172b4d !important;
        border-color: #cbd5e1 !important;
    }

    [data-testid="stMain"] input::placeholder,
    [data-testid="stMain"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    [data-testid="stMain"] [data-baseweb="select"] *,
    [data-testid="stMain"] [data-baseweb="popover"] * {
        color: #172b4d !important;
    }

    /* Radio buttons, checkboxes, sliders */
    [data-testid="stMain"] [data-testid="stRadio"] label,
    [data-testid="stMain"] [data-testid="stCheckbox"] label {
        color: #172b4d !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e0e7ef !important;
        border-radius: 13px !important;
        padding: 18px !important;
        box-shadow: 0 2px 8px rgba(16,36,58,.035);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #52647b !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #10243a !important;
    }

    /* Cards and forms */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-color: #e0e7ef !important;
        border-radius: 13px !important;
    }

    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e0e7ef !important;
        border-radius: 13px !important;
        padding: 18px !important;
    }

    /* Buttons */
    [data-testid="stMain"] .stButton button,
    [data-testid="stMain"] .stDownloadButton button,
    [data-testid="stMain"] .stFormSubmitButton button {
        border-radius: 9px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMain"] .stButton button[kind="primary"],
    [data-testid="stMain"] .stFormSubmitButton button[kind="primary"] {
        background: #087f8c !important;
        color: #ffffff !important;
        border-color: #087f8c !important;
    }

    [data-testid="stMain"] .stButton button[kind="secondary"],
    [data-testid="stMain"] .stDownloadButton button {
        background: #ffffff !important;
        color: #172b4d !important;
        border-color: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] .stButton button {
        background: #173c52 !important;
        color: #ffffff !important;
        border-color: #35556c !important;
    }

    /* Tables */
    [data-testid="stMain"] [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Alerts: readable text */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] li {
        color: #172b4d !important;
    }

    .ct-hero {
        background: linear-gradient(115deg, #10243a, #14576c);
        padding: 28px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .ct-hero h1 {
        color: #ffffff !important;
        margin: 0 0 7px 0;
        font-size: 2rem;
    }

    .ct-hero p {
        color: #e2f2f6 !important;
        margin: 0;
    }

    .ct-kicker {
        color: #a7eee3 !important;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .ct-muted {
        color: #52647b !important;
        font-size: .92rem;
    }

    .ct-footer {
        color: #52647b !important;
        text-align: center;
        font-size: .82rem;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def csv_bytes(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def parse_entries(text):
    return [
        item.strip()
        for item in re.split(r"[\n,;]+", text or "")
        if item.strip()
    ]


def next_id(records):
    return max(
        [int(item.get("ID", 0)) for item in records] + [0]
    ) + 1


def add_record(key, record):
    st.session_state[key].append(record)


def safe_date(value):
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def make_demo_data():
    st.session_state.visits = [
        {
            "ID": 1,
            "Date": "2026-10-01",
            "Type": "Routine check-up",
            "Notes": "Sample visit. Blood pressure recorded.",
            "Provider": "Demo Provider",
        },
        {
            "ID": 2,
            "Date": "2026-10-04",
            "Type": "Laboratory review",
            "Notes": "Sample laboratory review appointment.",
            "Provider": "Demo Provider",
        },
        {
            "ID": 3,
            "Date": "2026-10-08",
            "Type": "Follow-up",
            "Notes": "Follow-up visit for demonstration.",
            "Provider": "Demo Provider",
        },
    ]

    st.session_state.notes = [
        {
            "ID": 1,
            "Date": "2026-10-01",
            "Title": "First demo note",
            "Category": "General",
            "Text": "Example note for testing the CareTrail timeline.",
        },
        {
            "ID": 2,
            "Date": "2026-10-08",
            "Title": "Follow-up note",
            "Category": "Follow-up",
            "Text": "Example follow-up note. This is fictional data.",
        },
    ]

    st.session_state.vitals = [
        {"Date": "2026-10-01", "Metric": "Sample metric A", "Value": 72.0, "Unit": "units"},
        {"Date": "2026-10-02", "Metric": "Sample metric A", "Value": 73.5, "Unit": "units"},
        {"Date": "2026-10-03", "Metric": "Sample metric A", "Value": 71.8, "Unit": "units"},
        {"Date": "2026-10-04", "Metric": "Sample metric A", "Value": 74.2, "Unit": "units"},
        {"Date": "2026-10-05", "Metric": "Sample metric A", "Value": 73.0, "Unit": "units"},
        {"Date": "2026-10-06", "Metric": "Sample metric A", "Value": 75.1, "Unit": "units"},
        {"Date": "2026-10-07", "Metric": "Sample metric A", "Value": 74.0, "Unit": "units"},
        {"Date": "2026-10-08", "Metric": "Sample metric A", "Value": 76.2, "Unit": "units"},
        {"Date": "2026-10-09", "Metric": "Sample metric A", "Value": 74.5, "Unit": "units"},
        {"Date": "2026-10-01", "Metric": "Sample metric B", "Value": 6.8, "Unit": "units"},
        {"Date": "2026-10-03", "Metric": "Sample metric B", "Value": 7.0, "Unit": "units"},
        {"Date": "2026-10-05", "Metric": "Sample metric B", "Value": 6.7, "Unit": "units"},
        {"Date": "2026-10-07", "Metric": "Sample metric B", "Value": 7.2, "Unit": "units"},
        {"Date": "2026-10-09", "Metric": "Sample metric B", "Value": 7.1, "Unit": "units"},
    ]

    st.session_state.symptoms = [
        {
            "Date": "2026-10-06",
            "Symptom": "Headache",
            "Severity": 3,
            "Body area": "Head",
            "Notes": "Fictional demonstration entry",
        },
        {
            "Date": "2026-10-08",
            "Symptom": "Fatigue",
            "Severity": 5,
            "Body area": "General",
            "Notes": "Fictional demonstration entry",
        },
    ]

    st.session_state.medications = [
        {
            "ID": 1,
            "Name": "Example medicine A",
            "Dose": "As prescribed",
            "Schedule": "Daily",
            "Start date": "2026-10-01",
            "End date": "",
            "Notes": "Fictional demo entry; not a treatment recommendation.",
        },
    ]

    st.session_state.documents = []
    st.session_state.classifier_result = None
    st.session_state.classifier_question = ""
    st.session_state.demo_loaded = True


# ============================================================
# 4. OPTIONAL MODEL LOADING
# ============================================================

@st.cache_resource
def load_classifier():
    candidates = [
        BASE_DIR / MODEL_NAME,
        BASE_DIR / "models" / MODEL_NAME,
        BASE_DIR / "model" / MODEL_NAME,
        BASE_DIR / "artifacts" / MODEL_NAME,
    ]

    for path in candidates:
        if path.is_file():
            return joblib.load(path), str(path)

    raise FileNotFoundError(
        "Model file not found. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


# ============================================================
# 5. OPTIONAL DOCUMENT PARSING
# ============================================================

def extract_document_text(uploaded_file):
    """
    Extract text from text files and PDFs when pypdf is installed.
    For images or scanned PDFs, OCR requires additional packages
    and an available Tesseract installation.
    """
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if filename.endswith((".txt", ".md", ".csv")):
        return raw.decode("utf-8", errors="replace"), "Text extraction"

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if text.strip():
                return text, "PDF text extraction"

            return (
                "",
                "No selectable text found. This may be a scanned PDF; OCR is not enabled.",
            )

        except ImportError:
            return "", "Install pypdf to extract text from PDFs."

        except Exception as exc:
            return "", f"PDF extraction failed: {exc}"

    if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        try:
            from PIL import Image
            import pytesseract

            image_obj = Image.open(io.BytesIO(raw))
            text = pytesseract.image_to_string(image_obj)
            return text, "OCR extraction"

        except ImportError:
            return "", "Image OCR requires Pillow and pytesseract."

        except Exception as exc:
            return (
                "",
                "OCR could not run. Tesseract may not be installed: "
                + str(exc),
            )

    return "", "Unsupported format. Upload a PDF, TXT, CSV, or image."


def local_summary(text, max_sentences=5):
    """Simple extractive summary; does not use an external AI model."""
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return "No readable text was extracted."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected = [s for s in sentences if s.strip()][:max_sentences]

    return " ".join(selected) if selected else cleaned[:1000]


# ============================================================
# 6. BACKUP AND RESTORE
# ============================================================

def create_backup():
    data = {
        key: st.session_state.get(key, [])
        for key in [
            "visits",
            "notes",
            "vitals",
            "symptoms",
            "medications",
            "documents",
        ]
    }
    data["exported_at"] = datetime.now().isoformat(timespec="seconds")
    data["format_version"] = 1

    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def restore_backup(uploaded):
    data = json.loads(uploaded.getvalue().decode("utf-8"))

    expected = [
        "visits",
        "notes",
        "vitals",
        "symptoms",
        "medications",
        "documents",
    ]

    for key in expected:
        value = data.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"The backup field '{key}' must be a list.")

    for key in expected:
        st.session_state[key] = data.get(key, [])

    st.session_state.classifier_result = None


# ============================================================
# 7. SIDEBAR AND NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## CareTrail")
    st.caption("PERSONAL HEALTH COMPANION")
    st.divider()

    page = st.radio(
        "WORKSPACE",
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
        key="navigation",
    )

    st.divider()

    if st.button("Load Demo Data", use_container_width=True):
        make_demo_data()
        st.rerun()

    if st.button("Clear All Temporary Data", use_container_width=True):
        for key, default in DEFAULTS.items():
            st.session_state[key] = (
                default.copy() if isinstance(default, list) else default
            )
        st.session_state["demo_loaded"] = False
        st.rerun()

    st.divider()
    st.caption("Academic prototype. Not a clinical system.")


# ============================================================
# 8. COMMON HEADER
# ============================================================

st.markdown(
    """
    <div class="ct-hero">
        <div class="ct-kicker">HEALTH INFORMATION WORKSPACE</div>
        <h1>CareTrail</h1>
        <p>Health notes, trends, documents, and care coordination.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Overview")
    st.markdown(
        '<p class="ct-muted">A summary of your current session data.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    summary = [
        ("Visits", len(st.session_state.visits)),
        ("Health notes", len(st.session_state.notes)),
        ("Measurements", len(st.session_state.vitals)),
        ("Symptom logs", len(st.session_state.symptoms)),
        ("Medications", len(st.session_state.medications)),
    ]

    for col, (label, value) in zip(cols, summary):
        with col:
            st.metric(label, value)

    st.markdown("### Your workspace")

    cards = [
        ("Health Notes & Timeline", "Record visits and health notes."),
        ("Vitals & Analytics", "Explore numerical measurements and charts."),
        ("Symptom Tracker", "Record symptoms and severity over time."),
        ("Medications & Reminders", "Track medication entries and reminder schedules."),
        ("Document Vault", "Upload and extract supported document text."),
        ("Data Backup & Restore", "Export or restore your session records."),
    ]

    for start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (title, description) in zip(
            cols, cards[start:start + 3]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.write(description)

    st.markdown("### Recent care activity")

    events = []

    for item in st.session_state.visits:
        events.append({
            "Date": item.get("Date", ""),
            "Type": "Visit",
            "Description": item.get("Type", ""),
        })

    for item in st.session_state.notes:
        events.append({
            "Date": item.get("Date", ""),
            "Type": "Health note",
            "Description": item.get("Title", ""),
        })

    for item in st.session_state.symptoms:
        events.append({
            "Date": item.get("Date", ""),
            "Type": "Symptom",
            "Description": item.get("Symptom", ""),
        })

    if events:
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values("Date", ascending=False)
        st.dataframe(events_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No activity yet. Load demo data or add your first record.")


# ============================================================
# 10. HEALTH NOTES AND TIMELINE
# ============================================================

elif page == "Health Notes & Timeline":

    st.subheader("Health Notes & Timeline")

    tab_visit, tab_note, tab_history = st.tabs(
        ["Add a visit", "Write a health note", "View timeline"]
    )

    with tab_visit:
        with st.form("visit_form", clear_on_submit=True):
            st.markdown("#### New visit")

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
            provider = st.text_input("Doctor or facility (optional)")
            visit_text = st.text_area("Visit notes", height=130)

            save_visit = st.form_submit_button(
                "Save visit", type="primary", use_container_width=True
            )

            if save_visit:
                if not visit_text.strip():
                    st.error("Enter visit notes before saving.")
                else:
                    add_record(
                        "visits",
                        {
                            "ID": next_id(st.session_state.visits),
                            "Date": visit_date.isoformat(),
                            "Type": visit_type,
                            "Notes": visit_text.strip(),
                            "Provider": provider.strip() or "Not specified",
                        },
                    )
                    st.success("Visit saved.")

    with tab_note:
        with st.form("note_form", clear_on_submit=True):
            st.markdown("#### New health note")

            note_date = st.date_input("Note date", value=date.today())
            note_title = st.text_input("Note title")
            note_category = st.selectbox(
                "Category",
                [
                    "General",
                    "Symptoms",
                    "Appointment",
                    "Test result",
                    "Follow-up",
                    "Other",
                ],
            )
            note_text = st.text_area("Write your note", height=150)

            save_note = st.form_submit_button(
                "Save health note", type="primary", use_container_width=True
            )

            if save_note:
                if not note_title.strip() or not note_text.strip():
                    st.error("Enter both a title and note.")
                else:
                    add_record(
                        "notes",
                        {
                            "ID": next_id(st.session_state.notes),
                            "Date": note_date.isoformat(),
                            "Title": note_title.strip(),
                            "Category": note_category,
                            "Text": note_text.strip(),
                        },
                    )
                    st.success("Health note saved.")

    with tab_history:
        events = []

        for item in st.session_state.visits:
            events.append({
                "Date": item.get("Date"),
                "Type": "Visit",
                "Title": item.get("Type"),
                "Details": item.get("Notes"),
            })

        for item in st.session_state.notes:
            events.append({
                "Date": item.get("Date"),
                "Type": "Health note",
                "Title": item.get("Title"),
                "Details": item.get("Text"),
            })

        for item in st.session_state.symptoms:
            events.append({
                "Date": item.get("Date"),
                "Type": "Symptom",
                "Title": item.get("Symptom"),
                "Details": (
                    f"Severity: {item.get('Severity')}/10; "
                    f"Area: {item.get('Body area')}; "
                    f"{item.get('Notes', '')}"
                ),
            })

        if events:
            timeline_df = pd.DataFrame(events)
            timeline_df["Date"] = pd.to_datetime(
                timeline_df["Date"], errors="coerce"
            )
            timeline_df = timeline_df.sort_values("Date", ascending=False)

            st.dataframe(
                timeline_df,
                use_container_width=True,
                hide_index=True,
            )

            export_df = timeline_df.copy()
            export_df["Date"] = export_df["Date"].astype(str)

            st.download_button(
                "Download timeline CSV",
                data=csv_bytes(export_df.to_dict("records")),
                file_name="caretrail_timeline.csv",
                mime="text/csv",
            )
        else:
            st.info("No timeline entries yet.")


# ============================================================
# 11. VITALS AND ANALYTICS
# ============================================================

elif page == "Vitals & Analytics":

    st.subheader("Vitals & Analytics")
    st.markdown(
        '<p class="ct-muted">Record measurements, filter the data, and inspect changes over time.</p>',
        unsafe_allow_html=True,
    )

    with st.form("vitals_form", clear_on_submit=True):
        st.markdown("#### Add a measurement")

        c1, c2 = st.columns(2)

        with c1:
            vital_date = st.date_input("Measurement date", value=date.today())
            metric = st.selectbox(
                "Measurement",
                [
                    "Blood pressure — systolic",
                    "Blood pressure — diastolic",
                    "Heart rate",
                    "Blood glucose",
                    "Weight",
                    "Temperature",
                    "Oxygen saturation",
                    "Custom measurement",
                ],
            )

        with c2:
            if metric == "Custom measurement":
                metric_name = st.text_input("Custom measurement name")
            else:
                metric_name = metric

            value = st.number_input(
                "Recorded value",
                value=0.0,
                step=0.1,
                format="%.2f",
            )

            unit = st.text_input(
                "Unit",
                value={
                    "Blood pressure — systolic": "mmHg",
                    "Blood pressure — diastolic": "mmHg",
                    "Heart rate": "bpm",
                    "Blood glucose": "mg/dL",
                    "Weight": "kg",
                    "Temperature": "°C",
                    "Oxygen saturation": "%",
                    "Custom measurement": "",
                }.get(metric, ""),
            )

        save_vital = st.form_submit_button(
            "Add measurement", type="primary", use_container_width=True
        )

        if save_vital:
            if not metric_name.strip():
                st.error("Enter a measurement name.")
            else:
                add_record(
                    "vitals",
                    {
                        "Date": vital_date.isoformat(),
                        "Metric": metric_name.strip(),
                        "Value": float(value),
                        "Unit": unit.strip(),
                    },
                )
                st.success("Measurement added.")
                st.rerun()

    st.divider()

    if not st.session_state.vitals:
        st.info("Add measurements or load demo data to generate a chart.")
    else:
        df = pd.DataFrame(st.session_state.vitals)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Date", "Value"])

        if df.empty:
            st.warning("No valid numerical measurements are available.")
        else:
            available_metrics = sorted(df["Metric"].unique().tolist())

            selected_metric = st.selectbox(
                "Choose a measurement to graph",
                available_metrics,
            )

            filtered = df[df["Metric"] == selected_metric].copy()
            filtered = filtered.sort_values("Date")

            min_date = filtered["Date"].min().date()
            max_date = filtered["Date"].max().date()

            if min_date == max_date:
                start_date = end_date = min_date
            else:
                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                    start_date, end_date = date_range
                elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
                    start_date = end_date = date_range[0]
                elif isinstance(date_range, date):
                    start_date = end_date = date_range
                else:
                    start_date, end_date = min_date, max_date

            filtered = filtered[
                (filtered["Date"].dt.date >= start_date)
                & (filtered["Date"].dt.date <= end_date)
            ]

            if filtered.empty:
                st.warning("No values are available in that date range.")
            else:
                unit_values = [
                    str(x) for x in filtered["Unit"].dropna().unique()
                    if str(x).strip()
                ]
                unit_label = unit_values[0] if len(unit_values) == 1 else ""

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest", f"{filtered.iloc[-1]['Value']:.2f} {unit_label}")
                c2.metric("Average", f"{filtered['Value'].mean():.2f} {unit_label}")
                c3.metric("Minimum", f"{filtered['Value'].min():.2f} {unit_label}")
                c4.metric("Maximum", f"{filtered['Value'].max():.2f} {unit_label}")

                fig = px.line(
                    filtered,
                    x="Date",
                    y="Value",
                    markers=True,
                    title=f"{selected_metric} over time",
                    labels={
                        "Date": "Date",
                        "Value": f"Value ({unit_label})" if unit_label else "Value",
                    },
                    template="plotly_white",
                )

                fig.update_traces(
                    line=dict(color="#087f8c", width=3),
                    marker=dict(color="#087f8c", size=8),
                    hovertemplate="%{x|%d %b %Y}<br>Value: %{y:.2f}<extra></extra>",
                )

                fig.update_layout(
                    height=420,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#172b4d"),
                    margin=dict(l=20, r=20, t=65, b=20),
                    xaxis=dict(showgrid=True, gridcolor="#e5eaf1"),
                    yaxis=dict(showgrid=True, gridcolor="#e5eaf1"),
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Data table")

                show_df = filtered.copy()
                show_df["Date"] = show_df["Date"].dt.strftime("%Y-%m-%d")

                st.dataframe(
                    show_df[["Date", "Metric", "Value", "Unit"]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download filtered data",
                    data=csv_bytes(
                        show_df[["Date", "Metric", "Value", "Unit"]].to_dict("records")
                    ),
                    file_name="caretrail_vitals.csv",
                    mime="text/csv",
                )

                with st.expander("Delete a measurement"):
                    if len(st.session_state.vitals) > 0:
                        index = st.selectbox(
                            "Choose measurement",
                            range(len(st.session_state.vitals)),
                            format_func=lambda i: (
                                f"{st.session_state.vitals[i]['Date']} | "
                                f"{st.session_state.vitals[i]['Metric']} | "
                                f"{st.session_state.vitals[i]['Value']}"
                            ),
                        )

                        if st.button("Delete selected measurement"):
                            st.session_state.vitals.pop(index)
                            st.rerun()


# ============================================================
# 12. SYMPTOM TRACKER
# ============================================================

elif page == "Symptom Tracker":

    st.subheader("Symptom Tracker")
    st.markdown(
        '<p class="ct-muted">Record the date, severity, and body area associated with a symptom.</p>',
        unsafe_allow_html=True,
    )

    with st.form("symptom_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            symptom_date = st.date_input("Date", value=date.today())
            symptom_name = st.text_input(
                "Symptom",
                placeholder="Enter symptom",
            )
            body_area = st.selectbox(
                "Body area",
                [
                    "General",
                    "Head",
                    "Chest",
                    "Abdomen",
                    "Back",
                    "Arms",
                    "Legs",
                    "Other",
                ],
            )

        with c2:
            severity = st.slider(
                "Severity (1 = mild, 10 = severe)",
                min_value=1,
                max_value=10,
                value=3,
            )

            symptom_notes = st.text_area(
                "Additional notes",
                height=110,
            )

        save_symptom = st.form_submit_button(
            "Save symptom log", type="primary", use_container_width=True
        )

        if save_symptom:
            if not symptom_name.strip():
                st.error("Enter a symptom.")
            else:
                add_record(
                    "symptoms",
                    {
                        "Date": symptom_date.isoformat(),
                        "Symptom": symptom_name.strip(),
                        "Severity": int(severity),
                        "Body area": body_area,
                        "Notes": symptom_notes.strip(),
                    },
                )
                st.success("Symptom entry saved.")
                st.rerun()

    st.divider()
    st.markdown("### Symptom history")

    if st.session_state.symptoms:
        symptoms_df = pd.DataFrame(st.session_state.symptoms)
        symptoms_df["Date"] = pd.to_datetime(
            symptoms_df["Date"], errors="coerce"
        )

        st.dataframe(
            symptoms_df.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Severity": st.column_config.ProgressColumn(
                    "Severity",
                    min_value=0,
                    max_value=10,
                    format="%d/10",
                ),
            },
        )

        st.markdown("### Severity over time")

        selected_symptom = st.selectbox(
            "Choose symptom",
            sorted(symptoms_df["Symptom"].unique().tolist()),
        )

        plot_df = symptoms_df[
            symptoms_df["Symptom"] == selected_symptom
        ].sort_values("Date")

        fig = px.line(
            plot_df,
            x="Date",
            y="Severity",
            markers=True,
            template="plotly_white",
            labels={
                "Date": "Date",
                "Severity": "Severity (1–10)",
            },
        )

        fig.update_traces(
            line=dict(color="#087f8c", width=3),
            marker=dict(size=8),
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#172b4d"),
            height=350,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download symptom history",
            data=csv_bytes(st.session_state.symptoms),
            file_name="caretrail_symptoms.csv",
            mime="text/csv",
        )

    else:
        st.info("No symptom entries yet. Add your first log above.")

    st.warning(
        "This tracker does not diagnose symptoms. Seek urgent medical help "
        "if you experience a potentially serious or rapidly worsening symptom."
    )


# ============================================================
# 13. MEDICATIONS AND REMINDERS
# ============================================================

elif page == "Medications & Reminders":

    st.subheader("Medications & Reminders")
    st.markdown(
        '<p class="ct-muted">Maintain a medication list and create calendar reminder files.</p>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Enter medication instructions exactly as provided by your healthcare "
        "professional. This app does not prescribe medicines or verify interactions."
    )

    with st.form("medication_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            med_name = st.text_input("Medication name")
            med_dose = st.text_input(
                "Dose / instructions",
                placeholder="As prescribed",
            )
            med_schedule = st.text_input(
                "Schedule",
                placeholder="e.g. Every day",
            )

        with c2:
            med_start = st.date_input("Start date", value=date.today())
            has_end = st.checkbox("Has an end date?")
            med_end = st.date_input(
                "End date",
                value=date.today() + timedelta(days=7),
                disabled=not has_end,
            )

            reminder_time = st.time_input(
                "Reminder time",
                value=datetime.strptime("09:00", "%H:%M").time(),
            )

        med_notes = st.text_area("Notes (optional)")

        save_med = st.form_submit_button(
            "Save medication entry", type="primary", use_container_width=True
        )

        if save_med:
            if not med_name.strip():
                st.error("Enter a medication name.")
            elif has_end and med_end < med_start:
                st.error("The end date cannot be earlier than the start date.")
            else:
                add_record(
                    "medications",
                    {
                        "ID": next_id(st.session_state.medications),
                        "Name": med_name.strip(),
                        "Dose": med_dose.strip(),
                        "Schedule": med_schedule.strip(),
                        "Start date": med_start.isoformat(),
                        "End date": med_end.isoformat() if has_end else "",
                        "Reminder time": reminder_time.strftime("%H:%M"),
                        "Notes": med_notes.strip(),
                    },
                )
                st.success("Medication entry saved.")
                st.rerun()

    st.divider()
    st.markdown("### Medication list")

    if st.session_state.medications:
        med_df = pd.DataFrame(st.session_state.medications)

        st.dataframe(
            med_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download medication list",
            data=csv_bytes(st.session_state.medications),
            file_name="caretrail_medications.csv",
            mime="text/csv",
        )

        st.markdown("### Calendar reminder export")

        medication_options = list(range(len(st.session_state.medications)))

        selected_med_index = st.selectbox(
            "Choose medication for calendar reminder",
            medication_options,
            format_func=lambda i: (
                f"{st.session_state.medications[i]['Name']} — "
                f"{st.session_state.medications[i].get('Reminder time', '09:00')}"
            ),
        )

        reminder_days = st.number_input(
            "Number of days to include",
            min_value=1,
            max_value=90,
            value=7,
        )

        if st.button("Create calendar file"):
            med = st.session_state.medications[selected_med_index]
            today = date.today()
            end_date_text = med.get("End date", "")

            med_end_date = safe_date(end_date_text) if end_date_text else None
            effective_end = today + timedelta(days=int(reminder_days) - 1)

            if med_end_date:
                effective_end = min(effective_end, med_end_date)

            try:
                hour, minute = map(
                    int, med.get("Reminder time", "09:00").split(":")
                )
            except Exception:
                hour, minute = 9, 0

            events = []

            current_day = today
            while current_day <= effective_end:
                start_dt = datetime.combine(
                    current_day,
                    datetime.min.time().replace(hour=hour, minute=minute),
                )
                end_dt = start_dt + timedelta(minutes=5)

                def utc_stamp(dt):
                    return dt.strftime("%Y%m%dT%H%M%S")

                uid = (
                    f"caretrail-{med.get('ID', selected_med_index)}-"
                    f"{current_day.strftime('%Y%m%d')}@caretrail.local"
                )

                summary = re.sub(r"[\r\n,;]", " ", med["Name"])
                description = re.sub(
                    r"[\r\n,;]",
                    " ",
                    med.get("Dose", "Follow clinician instructions"),
                )

                events.append(
                    "\r\n".join([
                        "BEGIN:VEVENT",
                        f"UID:{uid}",
                        f"DTSTAMP:{utc_stamp(datetime.utcnow())}",
                        f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}",
                        f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}",
                        f"SUMMARY:Medication reminder - {summary}",
                        f"DESCRIPTION:{description}",
                        "END:VEVENT",
                    ])
                )

                current_day += timedelta(days=1)

            calendar_text = (
                "BEGIN:VCALENDAR\r\n"
                "VERSION:2.0\r\n"
                "PRODID:-//CareTrail//Medication Reminders//EN\r\n"
                + "\r\n".join(events)
                + "\r\nEND:VCALENDAR\r\n"
            )

            st.download_button(
                "Download .ics calendar file",
                data=calendar_text.encode("utf-8"),
                file_name="caretrail_reminders.ics",
                mime="text/calendar",
            )

        with st.expander("Delete a medication entry"):
            med_ids = [
                m.get("ID", i)
                for i, m in enumerate(st.session_state.medications)
            ]

            labels = {
                m.get("ID", i): m["Name"]
                for i, m in enumerate(st.session_state.medications)
            }

            chosen_id = st.selectbox(
                "Select medication",
                med_ids,
                format_func=lambda i: labels[i],
            )

            if st.button("Delete selected medication"):
                st.session_state.medications = [
                    m for i, m in enumerate(st.session_state.medications)
                    if m.get("ID", i) != chosen_id
                ]
                st.rerun()

    else:
        st.info("No medication entries yet.")


# ============================================================
# 14. DOCUMENT VAULT
# ============================================================

elif page == "Document Vault":

    st.subheader("Document Vault")
    st.markdown(
        '<p class="ct-muted">Upload supported documents and extract readable text where possible.</p>',
        unsafe_allow_html=True,
    )

    st.warning(
        "This prototype stores uploaded document content in temporary session "
        "memory. Do not upload identifiable patient documents to a public demo."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "csv", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            if any(
                doc.get("Filename") == file.name
                for doc in st.session_state.documents
            ):
                continue

            text, method = extract_document_text(file)

            st.session_state.documents.append(
                {
                    "Filename": file.name,
                    "Uploaded": datetime.now().isoformat(timespec="seconds"),
                    "Method": method,
                    "Text": text,
                    "Bytes": len(file.getvalue()),
                }
            )

        st.success("Document processing completed for this upload.")
        st.rerun()

    if st.session_state.documents:
        st.markdown("### Stored documents")

        doc_df = pd.DataFrame([
            {
                "Filename": doc["Filename"],
                "Uploaded": doc["Uploaded"],
                "Processing": doc["Method"],
                "Size (bytes)": doc["Bytes"],
            }
            for doc in st.session_state.documents
        ])

        st.dataframe(doc_df, use_container_width=True, hide_index=True)

        selected_doc = st.selectbox(
            "Select a document",
            range(len(st.session_state.documents)),
            format_func=lambda i: st.session_state.documents[i]["Filename"],
        )

        doc = st.session_state.documents[selected_doc]

        st.markdown("### Extracted text")

        if doc["Text"].strip():
            st.text_area(
                "Document text",
                value=doc["Text"],
                height=250,
                key=f"document_text_{selected_doc}",
            )

            st.markdown("### Plain-language preview")

            st.caption(
                "This is a simple extractive preview, not a medical interpretation."
            )

            st.write(local_summary(doc["Text"]))

            st.download_button(
                "Download extracted text",
                data=doc["Text"].encode("utf-8"),
                file_name=Path(doc["Filename"]).stem + "_extracted.txt",
                mime="text/plain",
            )

        else:
            st.info(doc["Method"])

        if st.button("Remove selected document"):
            st.session_state.documents.pop(selected_doc)
            st.rerun()

    else:
        st.info("No documents uploaded yet.")

    with st.expander("Document extraction details"):
        st.markdown(
            """
            - Text files: processed locally.
            - PDFs: selectable text can be extracted when `pypdf` is installed.
            - Scanned PDFs and images: OCR needs `pytesseract`, Pillow, and a
              working Tesseract installation.
            - Encrypted, corrupted, or unsupported documents may fail to extract.
            """
        )


# ============================================================
# 15. VISIT COMPARISON
# ============================================================

elif page == "Visit Comparison":

    st.subheader("Visit Comparison")

    st.markdown(
        '<p class="ct-muted">Compare two sets of notes and review what changed.</p>',
        unsafe_allow_html=True,
    )

    use_saved = (
        len(st.session_state.visits) >= 2
        and st.checkbox("Compare saved visit records")
    )

    if use_saved:
        visits = st.session_state.visits
        labels = {
            i: f"{v['Date']} — {v['Type']}"
            for i, v in enumerate(visits)
        }

        c1, c2 = st.columns(2)

        with c1:
            a_idx = st.selectbox(
                "First visit",
                range(len(visits)),
                format_func=lambda i: labels[i],
            )

        with c2:
            b_idx = st.selectbox(
                "Second visit",
                range(len(visits)),
                index=1,
                format_func=lambda i: labels[i],
            )

        notes_a = visits[a_idx]["Notes"]
        notes_b = visits[b_idx]["Notes"]

        st.text_area("First visit notes", notes_a, disabled=True)
        st.text_area("Second visit notes", notes_b, disabled=True)

    else:
        c1, c2 = st.columns(2)

        with c1:
            notes_a = st.text_area(
                "Visit A",
                height=170,
                placeholder="One entry per line",
                key="compare_a",
            )

        with c2:
            notes_b = st.text_area(
                "Visit B",
                height=170,
                placeholder="One entry per line",
                key="compare_b",
            )

    if st.button("Compare visits", type="primary", use_container_width=True):
        a = {x.casefold(): x for x in parse_entries(notes_a)}
        b = {x.casefold(): x for x in parse_entries(notes_b)}

        st.session_state.visit_comparison = {
            "matching": [a[k] for k in sorted(set(a) & set(b))],
            "only_a": [a[k] for k in sorted(set(a) - set(b))],
            "only_b": [b[k] for k in sorted(set(b) - set(a))],
        }

    report = st.session_state.get("visit_comparison")

    if report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Matching", len(report["matching"]))
        c2.metric("Only in A", len(report["only_a"]))
        c3.metric("Only in B", len(report["only_b"]))

        cols = st.columns(3)

        for col, title, key in zip(
            cols,
            ["Matching entries", "Only in A", "Only in B"],
            ["matching", "only_a", "only_b"],
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if report[key]:
                        for item in report[key]:
                            st.write("- " + item)
                    else:
                        st.caption("No entries")

        rows = []

        for key, label in [
            ("matching", "Matching"),
            ("only_a", "Only in A"),
            ("only_b", "Only in B"),
        ]:
            for item in report[key]:
                rows.append({"Category": label, "Entry": item})

        st.download_button(
            "Download comparison",
            data=csv_bytes(rows),
            file_name="caretrail_visit_comparison.csv",
            mime="text/csv",
        )

        st.caption(
            "This compares text entries; it does not interpret clinical meaning."
        )


# ============================================================
# 16. AI QUESTION CLASSIFIER
# ============================================================

elif page == "AI Question Classifier":

    st.subheader("AI Question Classifier")

    st.markdown(
        '<p class="ct-muted">Run your trained machine-learning model on a healthcare-related question.</p>',
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Question",
        key="classifier_question",
        height=120,
        placeholder="Enter a question to classify.",
    )

    if st.button("Classify question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                model, model_path = load_classifier()
                prediction = str(model.predict([question.strip()])[0])

                scores = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([question.strip()])[0]
                    scores = sorted(
                        zip(model.classes_, probabilities),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                st.session_state.classifier_result = {
                    "error": None,
                    "prediction": prediction,
                    "model_path": model_path,
                    "scores": (
                        [(str(k), float(v)) for k, v in scores]
                        if scores is not None
                        else None
                    ),
                }

            except Exception as exc:
                st.session_state.classifier_result = {
                    "error": str(exc),
                    "prediction": None,
                    "model_path": None,
                    "scores": None,
                }

    result = st.session_state.classifier_result

    if result:
        st.divider()

        if result["error"]:
            st.error("The trained model could not be loaded.")
            st.code(result["error"])

        else:
            st.success("Predicted category: " + result["prediction"])
            st.caption("Model file: " + result["model_path"])

            if result["scores"]:
                score_df = pd.DataFrame(
                    result["scores"],
                    columns=["Category", "Model score"],
                )
                score_df["Model score"] *= 100

                fig = px.bar(
                    score_df.sort_values("Model score"),
                    x="Model score",
                    y="Category",
                    orientation="h",
                    template="plotly_white",
                    labels={"Model score": "Model score (%)"},
                )

                fig.update_traces(marker_color="#087f8c")
                fig.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#172b4d"),
                )

                st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "This is text classification, not diagnosis or treatment advice."
    )


# ============================================================
# 17. DATA BACKUP AND RESTORE
# ============================================================

elif page == "Data Backup & Restore":

    st.subheader("Data Backup & Restore")

    st.markdown(
        "Export your current session data as JSON or restore a previously exported backup."
    )

    st.markdown("### Export")

    st.download_button(
        "Download complete JSON backup",
        data=create_backup(),
        file_name="caretrail_backup.json",
        mime="application/json",
        type="primary",
        use_container_width=True,
    )

    st.markdown("### Restore")

    backup_file = st.file_uploader(
        "Choose a CareTrail JSON backup",
        type=["json"],
        key="restore_backup_file",
    )

    if backup_file is not None:
        st.caption(
            "Restoring replaces the current records for the supported data sections."
        )

        confirm_restore = st.checkbox(
            "I understand that restoring replaces current records."
        )

        if st.button("Restore backup", disabled=not confirm_restore):
            try:
                restore_backup(backup_file)
                st.success("Backup restored.")
                st.rerun()
            except Exception as exc:
                st.error("Backup could not be restored.")
                st.code(str(exc))

    st.divider()
    st.markdown("### Individual exports")

    export_options = {
        "Visits": "visits",
        "Health notes": "notes",
        "Measurements": "vitals",
        "Symptoms": "symptoms",
        "Medications": "medications",
    }

    for label, key in export_options.items():
        records = st.session_state[key]

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])

            with c1:
                st.markdown(f"**{label}**")
                st.caption(f"{len(records)} records")

            with c2:
                st.download_button(
                    "Download CSV",
                    data=csv_bytes(records),
                    file_name=f"caretrail_{key}.csv",
                    mime="text/csv",
                    key=f"export_{key}",
                    use_container_width=True,
                )


# ============================================================
# 18. ABOUT AND PRIVACY
# ============================================================

elif page == "About & Privacy":

    st.subheader("About CareTrail")

    st.write(
        """
        CareTrail is an academic healthcare information prototype for
        organizing health notes, visits, symptoms, measurements, documents,
        and medication schedules.
        """
    )

    st.markdown("### Implemented in this prototype")

    st.markdown(
        """
        - Visit and health-note records
        - Interactive care timeline
        - Numerical trend charts using Plotly
        - Symptom severity logging and charts
        - Medication list and calendar reminder file generation
        - Text extraction for supported documents
        - Simple extractive text summaries
        - Trained question classifier
        - CSV and JSON backup exports
        """
    )

    st.markdown("### Optional integrations not automatically connected")

    st.markdown(
        """
        - OCR for scanned documents requires additional system dependencies.
        - LLM-generated medical summaries require a configured model or API.
        - Drug-interaction checks require a real, maintained drug data source.
        - Wearable imports need validation for the specific export format.
        - User authentication, encryption at rest, and a secure database
          require additional implementation.
        """
    )

    st.markdown("### Privacy limitations")

    st.warning(
        "This Streamlit prototype uses session memory and is not a secure "
        "medical-record platform. Session data may be lost on reset. Do not "
        "upload identifiable patient documents or use it to store sensitive "
        "health information in a public deployment."
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="ct-footer">
        CareTrail · Academic healthcare information prototype<br>
        For educational demonstration only. Not a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
