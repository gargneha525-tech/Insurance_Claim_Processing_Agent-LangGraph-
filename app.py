
"""
Insurance Claim Processing Dashboard
Streamlit + LangGraph

Run:
    python -m streamlit run app.py
"""

import streamlit as st

from workflow import (
    process_claim,
    REQUIRED_DOCUMENTS,
    DOCUMENT_LABELS,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Claim Processing Agent",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Insurance Claim Processing Agent")

st.caption(
    "LangGraph workflow for document verification, policy eligibility, "
    "fraud-risk screening, claim assessment, and human review."
)

st.info(
    "Coursework demonstration only. This application does not connect "
    "to a real insurer or verify actual policy records. Risk indicators "
    "are not proof of fraud. Real claim decisions require authorized review."
)


# ============================================================
# 2. DOCUMENT OPTIONS
# ============================================================

DOCUMENT_OPTIONS = REQUIRED_DOCUMENTS.copy()


# ============================================================
# 3. CLAIM INPUT FORM
# ============================================================

st.subheader("Enter Insurance Claim Details")

st.write(
    "Complete the required fields. No sample claim information is "
    "pre-filled. Select only the documents actually provided."
)

with st.form("claim_processing_form", clear_on_submit=False):

    st.markdown("### A. Claim and Policy Information")

    col1, col2 = st.columns(2)

    with col1:
        claim_id = st.text_input(
            "Claim ID *",
            value="",
            placeholder="Enter the claim ID",
        )

        policy_id = st.text_input(
            "Policy ID *",
            value="",
            placeholder="Enter the policy ID",
        )

        claim_amount_text = st.text_input(
            "Claim Amount *",
            value="",
            placeholder="Enter amount, e.g. 2400.00",
            help="Enter a positive number without currency symbols.",
        )

        previous_claims_text = st.text_input(
            "Number of Previous Claims *",
            value="",
            placeholder="Enter a whole number, e.g. 0",
        )

    with col2:
        coverage_limit_text = st.text_input(
            "Coverage Limit *",
            value="",
            placeholder="Enter the policy coverage limit",
            help="Enter a positive number without currency symbols.",
        )

        policy_status = st.selectbox(
            "Policy Status *",
            options=[
                "active",
                "expired",
                "inactive",
                "cancelled",
                "unknown",
            ],
            index=None,
            placeholder="Select the policy status",
            format_func=lambda value: value.replace(
                "_", " "
            ).title(),
        )

    st.markdown("### B. Documents")

    selected_documents = st.multiselect(
        "Select all documents provided with this claim",
        options=DOCUMENT_OPTIONS,
        default=[],
        format_func=lambda value: DOCUMENT_LABELS[value],
        help=(
            "Leave a document unselected if it was not provided. "
            "Missing required documents will trigger review."
        ),
    )

    st.markdown("### C. Incident Information")

    incident_description = st.text_area(
        "Incident Description *",
        value="",
        placeholder=(
            "Describe what happened, including the type of incident "
            "and relevant details."
        ),
        height=120,
    )

    st.markdown("### D. Additional Risk Information")

    fraud_notes = st.text_area(
        "Additional Risk Notes (Optional)",
        value="",
        placeholder=(
            "Enter any additional information that may require review, "
            "or leave blank if there are no notes."
        ),
        height=90,
    )

    high_risk = st.checkbox(
        "Flag this claim for additional high-risk review",
        value=False,
        help=(
            "Select this if the claim requires additional scrutiny. "
            "This flag does not prove fraud."
        ),
    )

    st.markdown("### E. Human Review")

    human_choice = st.selectbox(
        "Human Review Decision (Optional)",
        options=["approve", "reject"],
        index=None,
        placeholder="No human decision — leave unselected",
        format_func=lambda value: {
            "approve": "Approve claim",
            "reject": "Reject claim",
        }[value],
        help=(
            "Use only if you are authorized to make the review decision. "
            "If no decision is selected, claims requiring review remain pending."
        ),
    )

    submitted = st.form_submit_button(
        "Process Insurance Claim",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 4. VALIDATE AND PROCESS THE SUBMISSION
# ============================================================

if submitted:

    validation_errors = []

    if not claim_id.strip():
        validation_errors.append("Claim ID is required.")

    if not policy_id.strip():
        validation_errors.append("Policy ID is required.")

    if policy_status is None:
        validation_errors.append("Please select a policy status.")

    if not incident_description.strip():
        validation_errors.append("Incident description is required.")

    # Parse numeric fields without silently substituting example values.
    claim_amount = None
    coverage_limit = None
    previous_claims = None

    try:
        if not claim_amount_text.strip():
            raise ValueError("Claim Amount is required.")

        claim_amount = float(claim_amount_text.strip())

        if claim_amount <= 0:
            raise ValueError("Claim Amount must be greater than zero.")

    except ValueError as exc:
        validation_errors.append(str(exc))

    try:
        if not coverage_limit_text.strip():
            raise ValueError("Coverage Limit is required.")

        coverage_limit = float(coverage_limit_text.strip())

        if coverage_limit <= 0:
            raise ValueError("Coverage Limit must be greater than zero.")

    except ValueError as exc:
        validation_errors.append(str(exc))

    try:
        if not previous_claims_text.strip():
            raise ValueError("Number of Previous Claims is required.")

        # Reject values such as 2.5 or 2.0 when a whole number is expected.
        raw_previous_claims = previous_claims_text.strip()
        previous_claims = int(raw_previous_claims)

        if previous_claims < 0:
            raise ValueError(
                "Number of Previous Claims cannot be negative."
            )

    except ValueError as exc:
        validation_errors.append(
            "Number of Previous Claims must be a non-negative whole number."
            if "invalid literal" in str(exc).lower()
            else str(exc)
        )

    if validation_errors:
        st.error("Please correct the following input errors:")

        for error in validation_errors:
            st.markdown(f"- {error}")

        st.session_state.pop("claim_processing_result", None)
        st.session_state.pop("claim_processing_input", None)

    else:
        claim_data = {
            "claim_id": claim_id.strip(),
            "policy_id": policy_id.strip(),
            "claim_amount": claim_amount,
            "coverage_limit": coverage_limit,
            "policy_status": policy_status,
            "documents": list(selected_documents),
            "required_documents": list(REQUIRED_DOCUMENTS),
            "incident_description": incident_description.strip(),
            "previous_claims": previous_claims,
            "fraud_notes": fraud_notes.strip(),
            "high_risk": bool(high_risk),
            "human_approval_decision": human_choice or "pending",
        }

        with st.spinner("Running the claim-processing agents..."):
            try:
                result = process_claim(claim_data)

                st.session_state["claim_processing_result"] = result
                st.session_state["claim_processing_input"] = claim_data

            except Exception as exc:
                st.session_state.pop("claim_processing_result", None)
                st.session_state.pop("claim_processing_input", None)

                st.error("The claim workflow could not be completed.")
                st.exception(exc)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

result = st.session_state.get("claim_processing_result")

if result:

    st.divider()
    st.header("Claim Processing Results")

    decision = result.get(
        "decision",
        result.get("final_decision", "MANUAL REVIEW"),
    )

    final_status = result.get(
        "final_status",
        result.get("claim_status", "Pending Review"),
    )

    amount = float(result.get("claim_amount", 0.0) or 0.0)
    limit = float(result.get("coverage_limit", 0.0) or 0.0)
    approved_amount = float(
        result.get("approved_amount", 0.0) or 0.0
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Claim Decision", decision)
    metric2.metric("Claim Amount", f"{amount:,.2f}")
    metric3.metric("Approved Amount", f"{approved_amount:,.2f}")

    info1, info2 = st.columns(2)

    with info1:
        st.write(f"**Claim ID:** {result.get('claim_id', '')}")
        st.write(f"**Policy ID:** {result.get('policy_id', '')}")

    with info2:
        st.write(f"**Coverage Limit:** {limit:,.2f}")
        st.write(f"**Current Status:** {final_status}")

    if result.get("review_required", False):
        st.warning(
            "Human review is required. This claim is not finally approved."
        )
    elif "REJECTED" in decision:
        st.error(
            "The configured workflow rules resulted in rejection."
        )
    else:
        st.success("The workflow completed without pending human review.")

    # --------------------------------------------------------
    # 5A. DOCUMENT VERIFICATION
    # --------------------------------------------------------

    st.subheader("1. Document Verification Agent")

    doc = (
        result.get("doc_verification")
        or result.get("document_verification")
        or {}
    )

    st.write("**Status:**", doc.get("status", "Unknown"))
    st.write(
        "**Assessment:**",
        doc.get("reason", doc.get("assessment", "No assessment returned.")),
    )

    missing_docs = (
        doc.get("missing_documents")
        or doc.get("missing_docs")
        or []
    )

    if missing_docs:
        st.write("**Missing documents:**")

        for document in missing_docs:
            label = DOCUMENT_LABELS.get(
                document,
                str(document).replace("_", " ").title(),
            )
            st.markdown(f"- {label}")
    else:
        st.success("All required documents are present.")

    with st.expander("View document verification details"):
        st.json(doc)

    # --------------------------------------------------------
    # 5B. POLICY ELIGIBILITY
    # --------------------------------------------------------

    st.subheader("2. Policy Eligibility Agent")

    eligibility = (
        result.get("policy_eligibility")
        or result.get("eligibility")
        or result.get("policy_check")
        or {}
    )

    st.write("**Status:**", eligibility.get("status", "Unknown"))
    st.write(
        "**Assessment:**",
        eligibility.get(
            "reason",
            eligibility.get("assessment", "No assessment returned."),
        ),
    )

    with st.expander("View policy eligibility details"):
        st.json(eligibility)

    # --------------------------------------------------------
    # 5C. FRAUD-RISK ASSESSMENT
    # --------------------------------------------------------

    st.subheader("3. Fraud-Risk Assessment Agent")

    fraud = (
        result.get("fraud_check")
        or result.get("fraud_assessment")
        or {}
    )

    if fraud:
        st.write("**Risk Level:**", fraud.get("risk_level", "Unknown"))
        st.write(
            "**Assessment:**",
            fraud.get("reason", fraud.get("assessment", "No assessment returned.")),
        )

        indicators = (
            fraud.get("indicators")
            or fraud.get("risk_indicators")
            or []
        )

        if indicators:
            st.write("**Indicators requiring attention:**")

            for indicator in indicators:
                st.markdown(f"- {indicator}")
        else:
            st.write("No configured risk indicators were triggered.")

        st.caption(
            "Risk indicators are screening signals, not proof of fraud."
        )

        with st.expander("View fraud assessment details"):
            st.json(fraud)
    else:
        st.warning("No fraud assessment was returned.")

    # --------------------------------------------------------
    # 5D. CLAIM ASSESSMENT
    # --------------------------------------------------------

    st.subheader("4. Claim Assessment Agent")

    assessment = result.get("claim_assessment") or {}

    st.write("**Status:**", assessment.get("status", "Unknown"))
    st.write(
        "**Assessment:**",
        assessment.get(
            "reason",
            assessment.get("assessment", "No assessment returned."),
        ),
    )

    with st.expander("View claim assessment details"):
        st.json(assessment)

    # --------------------------------------------------------
    # 5E. HUMAN APPROVAL
    # --------------------------------------------------------

    st.subheader("5. Human Approval Agent")

    human = result.get("human_approval") or {}

    if human:
        st.write("**Status:**", human.get("status", "Unknown"))
        st.write(
            "**Assessment:**",
            human.get("reason", human.get("assessment", "No assessment returned.")),
        )

        if human.get("pending", False):
            st.info(
                "No final human decision has been recorded. To record one, "
                "select Approve or Reject in the form and submit the claim again."
            )

        with st.expander("View human approval details"):
            st.json(human)

    elif result.get("review_required", False):
        st.info(
            "Human review is required. Select an authorized decision in "
            "the form above and submit the claim again."
        )
    else:
        st.write("Human approval was not required by the configured rules.")

    # --------------------------------------------------------
    # 5F. FINAL SUMMARY AND REPORT
    # --------------------------------------------------------

    st.subheader("6. Final Claim Summary")

    st.write(
        result.get(
            "assessment_summary",
            "No final summary was returned.",
        )
    )

    final_report = result.get("final_report", "")

    if final_report:
        with st.expander("View complete claim report", expanded=True):
            st.text(final_report)

        report_filename = (
            str(result.get("claim_id", "claim"))
            .replace("/", "_")
            .replace("\\", "_")
            + "_report.txt"
        )

        st.download_button(
            label="Download Claim Report",
            data=final_report,
            file_name=report_filename,
            mime="text/plain",
        )

    with st.expander("View complete workflow result"):
        st.json(result)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Workflow Information")

    st.write(
        "Enter a claim using the form on the main page. The workflow "
        "runs document, policy, fraud-risk, and claim assessments before "
        "creating a final decision."
    )

    st.markdown("**Possible workflow outcomes**")

    st.markdown(
        "- **Auto Approved:** Configured checks passed.\n"
        "- **Manual Review:** Additional information or review is needed.\n"
        "- **Rejected:** The configured rules identify an ineligible policy."
    )

    st.divider()

    st.caption(
        "Testing tip: try different policy statuses, omit a document, "
        "enter several previous claims, or flag a claim for additional review."
    )

    if st.button("Clear Displayed Results", use_container_width=True):
        st.session_state.pop("claim_processing_result", None)
        st.session_state.pop("claim_processing_input", None)
        st.rerun()


# ============================================================
# 7. FOOTER
# ============================================================

st.divider()

st.caption(
    "Insurance Claim Processing Agent | Streamlit | LangGraph | "
    "Document Verification | Policy Eligibility | Risk Screening | Human Review"
)
