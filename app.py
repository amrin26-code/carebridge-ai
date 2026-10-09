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
