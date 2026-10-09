
import streamlit as st
import joblib
import re
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="CareTrail",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD NLP MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("intelliphr_intent_model.joblib")

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #164e63, #287c8e);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APP HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🩺 CareTrail</h1>
    <p>Your Health Record Review Assistant</p>
    <p>
        Compare medical visit records, identify differences,
        and organise questions for your healthcare professional.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic prototype using synthetic data. CareTrail does not "
    "diagnose medical conditions, verify clinical accuracy, or "
    "recommend treatment changes. Confirm health information "
    "with a qualified healthcare professional."
)

# ---------------- HELPER FUNCTIONS ----------------
def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def parse_medications(text):
    medications = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        name, details = line.split(":", 1)
        name = name.strip()
        details = details.strip()

        if name and details:
            medications[name.casefold()] = {
                "name": name,
                "details": details
            }

    return medications


def build_findings(
    old_meds,
    new_meds,
    old_followup,
    new_followup
):
    findings = []

    # Compare medication entries
    for key in sorted(set(old_meds) | set(new_meds)):
        old = old_meds.get(key)
        new = new_meds.get(key)

        if old is None:
            category = "NEW ENTRY"
            details = f"Current record: {new['details']}"
            action = (
                "Confirm this entry with your "
                "healthcare professional."
            )

        elif new is None:
            category = "MISSING ENTRY"
            details = f"Previous record: {old['details']}"
            action = (
                "Confirm whether this item should "
                "still be listed."
            )

        elif normalize(old["details"]) != normalize(new["details"]):
            category = "DETAILS DIFFER"
            details = (
                f"Previous: {old['details']} | "
                f"Current: {new['details']}"
            )
            action = (
                "Ask your healthcare professional "
                "to clarify the difference."
            )

        else:
            category = "TEXT MATCH"
            details = (
                "Recorded details match after basic "
                "text normalization."
            )
            action = (
                "Verify that the information is "
                "complete and accurate."
            )

        findings.append({
            "item": (new or old)["name"],
            "category": category,
            "details": details,
            "action": action
        })

    # Compare follow-up instructions
    if not new_followup:
        category = "MISSING INFORMATION"
        details = (
            "No current follow-up instruction was entered."
        )
        action = (
            "Ask whether a follow-up appointment "
            "or instruction is needed."
        )

    elif not old_followup:
        category = "NEW ENTRY"
        details = f"Current record: {new_followup}"
        action = "Confirm the current follow-up plan."

    elif normalize(old_followup) != normalize(new_followup):
        category = "TEXT DIFFERS"
        details = (
            f"Previous: {old_followup} | "
            f"Current: {new_followup}"
        )
        action = (
            "Confirm whether the follow-up plan has changed."
        )

    else:
        category = "TEXT MATCH"
        details = (
            "Follow-up text matches after basic "
            "normalization."
        )
        action = (
            "Verify that the instructions are complete "
            "and accurate."
        )

    findings.append({
        "item": "Follow-up instructions",
        "category": category,
        "details": details,
        "action": action
    })

    return findings


# ---------------- MAIN TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📋 Visit Comparison",
    "🤖 AI Question Classifier",
    "ℹ️ About CareTrail"
])

# ==================================================
# TAB 1: VISIT COMPARISON
# ==================================================
with tab1:
    st.subheader("Compare Medical Visit Records")

    st.write(
        "Enter one medication per line using this format: "
        "`Medicine name: details`"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Previous Visit")

        old_text = st.text_area(
            "Previous medication records",
            placeholder=(
                "Medicine A: 500 mg once daily\n"
                "Medicine B: 10 mg at night"
            ),
            height=160,
            key="old_meds"
        )

        old_followup = st.text_area(
            "Previous follow-up instructions",
            placeholder="Example: Review after two weeks",
            key="old_followup"
        )

    with col2:
        st.markdown("### Current Visit")

        new_text = st.text_area(
            "Current medication records",
            placeholder=(
                "Medicine A: 500 mg twice daily\n"
                "Medicine C: 5 mg once daily"
            ),
            height=160,
            key="new_meds"
        )

        new_followup = st.text_area(
            "Current follow-up instructions",
            placeholder="Example: Review after one month",
            key="new_followup"
        )

    compare = st.button(
        "Compare Records",
        type="primary",
        use_container_width=True
    )

    if compare:
        old_meds = parse_medications(old_text)
        new_meds = parse_medications(new_text)

        findings = build_findings(
            old_meds,
            new_meds,
            old_followup.strip(),
            new_followup.strip()
        )

        st.session_state["findings"] = findings

    if "findings" in st.session_state:
        findings = st.session_state["findings"]

        st.divider()
        st.subheader("Review Results")

        flagged_categories = {
            "DETAILS DIFFER",
            "NEW ENTRY",
            "MISSING ENTRY",
            "MISSING INFORMATION",
            "TEXT DIFFERS"
        }

        flagged_count = sum(
            item["category"] in flagged_categories
            for item in findings
        )

        matching_count = sum(
            item["category"] == "TEXT MATCH"
            for item in findings
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Items reviewed", len(findings))
        c2.metric("Items to review", flagged_count)
        c3.metric("Text matches", matching_count)

        for item in findings:
            with st.container(border=True):
                st.markdown(f"**{item['item']}**")
                st.caption(f"Status: {item['category']}")
                st.write(item["details"])
                st.info(item["action"])

        # Generate downloadable report
        report_lines = [
            "CARETRAIL - VISIT COMPARISON REPORT",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            "",
            "Academic prototype using synthetic data.",
            "This report is not a medical diagnosis.",
            ""
        ]

        for item in findings:
            report_lines.extend([
                f"Item: {item['item']}",
                f"Status: {item['category']}",
                f"Details: {item['details']}",
                f"Suggested question: {item['action']}",
                "-" * 45
            ])

        report_text = "\n".join(report_lines)

        st.download_button(
            "⬇️ Download CareTrail Report",
            data=report_text,
            file_name="CareTrail_Visit_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ==================================================
# TAB 2: NLP QUESTION CLASSIFIER
# ==================================================
with tab2:
    st.subheader("AI Question Classifier")

    st.write(
        "Enter a question to classify it into one of "
        "the categories learned by the NLP model."
    )

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: Can you explain the medication list?"
        )
    )

    if st.button("Classify Question", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")

        elif model is None:
            st.error("The NLP model could not be loaded.")
            st.caption(str(model_error))

        else:
            prediction = model.predict([question])[0]

            labels = {
                "symptom_help": "Symptom-related question",
                "medication_query": "Medication-related question",
                "records_query": "Health record question",
                "health_trends": "Health trends question",
                "general_help": "General help"
            }

            st.success(
                "Predicted category: "
                + labels.get(prediction, prediction)
            )

            st.caption(
                "This result classifies the question's intent. "
                "It does not assess symptoms or provide medical advice."
            )

# ==================================================
# TAB 3: ABOUT
# ==================================================
with tab3:
    st.subheader("About CareTrail")

    st.write(
        "CareTrail is an academic prototype designed to help "
        "users review health information recorded across visits."
    )

    st.markdown("### Main Features")

    st.markdown("""
    - Compare medication entries between two visits.
    - Highlight new, missing, or differing information.
    - Review changes in follow-up instructions.
    - Classify user questions using natural language processing.
    - Download a visit comparison report.
    """)

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Joblib
    """)

    st.markdown("### Limitations")

    st.write(
        "The prototype uses synthetic examples and basic text "
        "comparison. Matching text does not establish medical "
        "correctness. CareTrail does not connect to hospital systems, "
        "diagnose conditions, or recommend treatment changes."
    )

st.divider()

st.caption(
    "CareTrail | Academic Healthcare AI Project | "
    "For demonstration and educational use only"
)
