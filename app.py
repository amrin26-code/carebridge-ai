
import streamlit as st
import joblib
import re
from datetime import datetime

st.set_page_config(
    page_title="CareBridge AI",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("intelliphr_intent_model.joblib")

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #123c69, #227c9d);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.small-note {
    color: #536477;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🩺 CareBridge AI</h1>
    <p>Personal Health Record Review Assistant</p>
    <p>Compare visit records, identify information differences,
    and classify health-related questions using NLP.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic prototype using synthetic data. "
    "This tool does not diagnose conditions or recommend treatment changes. "
    "Always confirm medical records with a qualified healthcare professional."
)

tab1, tab2, tab3 = st.tabs([
    "📋 Visit Comparison",
    "🤖 AI Question Classifier",
    "ℹ️ About"
])

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

def build_findings(old_meds, new_meds, old_followup, new_followup):
    findings = []

    for key in sorted(set(old_meds) | set(new_meds)):
        old = old_meds.get(key)
        new = new_meds.get(key)

        if old is None:
            category = "NEW ENTRY"
            details = f"Current record: {new['details']}"
            action = "Confirm this entry with your healthcare professional."
        elif new is None:
            category = "MISSING ENTRY"
            details = f"Previous record: {old['details']}"
            action = "Confirm whether this item should still be listed."
        elif normalize(old["details"]) != normalize(new["details"]):
            category = "DETAILS DIFFER"
            details = (
                f"Previous: {old['details']} | "
                f"Current: {new['details']}"
            )
            action = "Ask your healthcare professional to clarify the difference."
        else:
            category = "TEXT MATCH"
            details = "Recorded details match after basic text normalization."
            action = "Verify that the information is complete and accurate."

        findings.append({
            "item": (new or old)["name"],
            "category": category,
            "details": details,
            "action": action
        })

    if not new_followup:
        category = "MISSING INFORMATION"
        details = "No current follow-up instruction was entered."
        action = "Ask whether a follow-up appointment or instruction is needed."
    elif not old_followup:
        category = "NEW ENTRY"
        details = f"Current record: {new_followup}"
        action = "Confirm the current follow-up plan."
    elif normalize(old_followup) != normalize(new_followup):
        category = "TEXT DIFFERS"
        details = f"Previous: {old_followup} | Current: {new_followup}"
        action = "Confirm whether the follow-up plan has changed."
    else:
        category = "TEXT MATCH"
        details = "Follow-up text matches after basic normalization."
        action = "Verify that the instructions are complete and accurate."

    findings.append({
        "item": "Follow-up instructions",
        "category": category,
        "details": details,
        "action": action
    })

    return findings

with tab1:
    st.subheader("Compare Two Visit Records")
    st.write(
        "Enter one medication per line in the format "
        "`Medicine name: details`."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Previous Visit")
        old_text = st.text_area(
            "Previous medication records",
            placeholder="Example:\nMedicine A: 500 mg once daily\nMedicine B: 10 mg at night",
            height=180,
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
            placeholder="Example:\nMedicine A: 500 mg twice daily\nMedicine C: 5 mg once daily",
            height=180,
            key="new_meds"
        )
        new_followup = st.text_area(
            "Current follow-up instructions",
            placeholder="Example: Review after one month",
            key="new_followup"
        )

    if st.button("Compare Records", type="primary", use_container_width=True):
        old_meds = parse_medications(old_text)
        new_meds = parse_medications(new_text)

        findings = build_findings(
            old_meds, new_meds,
            old_followup.strip(), new_followup.strip()
        )

        st.session_state["findings"] = findings
        st.session_state["report_inputs"] = {
            "old_text": old_text,
            "new_text": new_text,
            "old_followup": old_followup,
            "new_followup": new_followup
        }

    if "findings" in st.session_state:
        findings = st.session_state["findings"]
        st.markdown("---")
        st.subheader("Review Results")

        counts = {}
        for item in findings:
            counts[item["category"]] = counts.get(item["category"], 0) + 1

        metric_cols = st.columns(3)
        metric_cols[0].metric("Items reviewed", len(findings))
        metric_cols[1].metric(
            "Differences / new / missing",
            sum(v for k, v in counts.items()
                if k in ["DETAILS DIFFER", "NEW ENTRY",
                         "MISSING ENTRY", "MISSING INFORMATION",
                         "TEXT DIFFERS"])
        )
        metric_cols[2].metric(
            "Text matches",
            counts.get("TEXT MATCH", 0)
        )

        for item in findings:
            with st.container(border=True):
                st.markdown(f"**{item['item']}**")
                st.caption(item["category"])
                st.write(item["details"])
                st.info(item["action"])

        report = [
            "CAREBRIDGE AI - VISIT COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "ACADEMIC DEMONSTRATION ONLY",
            "Not a diagnosis or a treatment recommendation.",
            ""
        ]

        for item in findings:
            report.extend([
                f"Item: {item['item']}",
                f"Category: {item['category']}",
                f"Details: {item['details']}",
                f"Suggested question: {item['action']}",
                "-" * 45
            ])

        st.download_button(
            "⬇️ Download Review Report",
            data="\n".join(report),
            file_name="CareBridge_Visit_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

with tab2:
    st.subheader("AI Question Classifier")
    st.write(
        "Enter a question to see which intent category the NLP model predicts."
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: Can you explain the medication list?"
    )

    if st.button("Classify Question", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        elif model is None:
            st.error("The model could not be loaded.")
            st.caption(model_error)
        else:
            prediction = model.predict([question])[0]
            labels = {
                "symptom_help": "Symptom-related question",
                "medication_query": "Medication-related question",
                "records_query": "Health record question",
                "health_trends": "Health trends question",
                "general_help": "General help"
            }
            st.success(f"Predicted category: {labels.get(prediction, prediction)}")
            st.caption(
                "This is an intent classification result, not a medical assessment."
            )

with tab3:
    st.subheader("About CareBridge AI")
    st.write("""
    CareBridge AI is an academic prototype designed to support review
    of personal health record information across visits.

    **Main features**
    - Compare medication entries between two visits.
    - Highlight new, missing, or differing information.
    - Review changes in follow-up instructions.
    - Classify user questions using a TF-IDF and Logistic Regression model.
    - Download a text summary of the comparison.

    **Technology:** Python, Streamlit, scikit-learn, TF-IDF,
    Logistic Regression, and joblib.

    **Limitations:** The prototype uses synthetic examples and basic text
    matching. It does not verify medical correctness, access clinical systems,
    diagnose disease, or recommend medication changes.
    """)
