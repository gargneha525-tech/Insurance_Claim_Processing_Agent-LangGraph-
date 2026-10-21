
"""
Insurance Claim Processing Workflow
LangGraph-based multi-agent claim assessment.

Agents:
1. Document Verification
2. Policy Eligibility
3. Fraud-Risk Assessment
4. Claim Assessment
5. Claim Summary
6. Human Approval
"""

from typing import Any, Dict, List, TypedDict

from langgraph.graph import START, END, StateGraph


# ============================================================
# 1. CONSTANTS AND WORKFLOW STATE
# ============================================================

REQUIRED_DOCUMENTS = [
    "claim_form",
    "identity_proof",
    "supporting_evidence",
]

DOCUMENT_LABELS = {
    "claim_form": "Claim Form",
    "identity_proof": "Identity Proof",
    "supporting_evidence": "Supporting Evidence",
}

ACTIVE_STATUSES = {"active", "valid", "in_force", "inforce"}

INACTIVE_STATUSES = {
    "expired",
    "inactive",
    "cancelled",
    "canceled",
    "terminated",
    "lapsed",
}


class ClaimState(TypedDict, total=False):
    claim_id: str
    policy_id: str
    claim_amount: float
    coverage_limit: float
    policy_status: str
    documents: List[str]
    required_documents: List[str]
    incident_description: str
    previous_claims: int
    fraud_notes: str
    high_risk: bool
    human_approval_decision: str

    doc_verification: Dict[str, Any]
    document_verification: Dict[str, Any]
    policy_eligibility: Dict[str, Any]
    policy_check: Dict[str, Any]
    eligibility: Dict[str, Any]
    fraud_check: Dict[str, Any]
    fraud_assessment: Dict[str, Any]
    claim_assessment: Dict[str, Any]
    human_approval: Dict[str, Any]

    decision: str
    final_decision: str
    final_status: str
    claim_status: str
    approved_amount: float
    review_required: bool
    assessment_summary: str
    final_report: str
    errors: List[str]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_policy_status(value: Any) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_document_name(value: Any) -> str:
    name = str(value).strip().lower()
    name = name.replace("-", "_").replace(" ", "_")

    aliases = {
        "claimform": "claim_form",
        "claim_application": "claim_form",
        "claim_application_form": "claim_form",
        "identity": "identity_proof",
        "id_proof": "identity_proof",
        "identity_document": "identity_proof",
        "government_id": "identity_proof",
        "supporting_documents": "supporting_evidence",
        "supporting_document": "supporting_evidence",
        "medical_report": "supporting_evidence",
        "medical_records": "supporting_evidence",
        "evidence": "supporting_evidence",
    }

    return aliases.get(name, name)


def normalize_documents(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, dict):
        values = [key for key, present in value.items() if present]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []

    documents = []

    for item in values:
        if isinstance(item, dict):
            item = (
                item.get("name")
                or item.get("document")
                or item.get("type")
                or ""
            )

        name = normalize_document_name(item)

        if name and name not in documents:
            documents.append(name)

    return documents


def get_documents_from_state(state: ClaimState) -> List[str]:
    for key in (
        "documents",
        "provided_documents",
        "uploaded_documents",
    ):
        if key in state:
            return normalize_documents(state[key])

    return []


# ============================================================
# 3. DOCUMENT VERIFICATION AGENT
# ============================================================

def document_verification_agent(
    state: ClaimState,
) -> Dict[str, Any]:

    provided = get_documents_from_state(state)

    required = normalize_documents(
        state.get("required_documents", REQUIRED_DOCUMENTS)
    )

    missing = [
        document for document in required
        if document not in provided
    ]

    is_valid = not missing

    if is_valid:
        status = "Verified"
        reason = "All required documents are present."
    else:
        status = "Incomplete"
        missing_labels = [
            DOCUMENT_LABELS.get(
                item, item.replace("_", " ").title()
            )
            for item in missing
        ]
        reason = (
            "Required documents are missing: "
            + ", ".join(missing_labels)
            + "."
        )

    result = {
        "agent": "Document Verification Agent",
        "is_valid": is_valid,
        "passed": is_valid,
        "valid": is_valid,
        "status": status,
        "reason": reason,
        "assessment": reason,
        "required_documents": required,
        "provided_documents": provided,
        "missing_documents": missing,
        "missing_docs": missing,
        "errors": [reason] if missing else [],
    }

    return {
        "doc_verification": result,
        "document_verification": result,
    }


# ============================================================
# 4. POLICY ELIGIBILITY AGENT
# ============================================================

def policy_eligibility_agent(
    state: ClaimState,
) -> Dict[str, Any]:

    raw_status = state.get("policy_status", "")
    normalized_status = normalize_policy_status(raw_status)

    if normalized_status in ACTIVE_STATUSES:
        eligible = True
        status = "Eligible"
        reason = "The policy status is active."
        errors = []

    elif normalized_status in INACTIVE_STATUSES:
        eligible = False
        status = "Ineligible"
        reason = (
            f"The policy is not eligible because its status "
            f"is {raw_status}."
        )
        errors = [reason]

    else:
        eligible = False
        status = "Needs Verification"
        reason = (
            "The policy status is unknown or missing. "
            "Confirm policy status before making a final decision."
        )
        errors = [reason]

    result = {
        "agent": "Policy Eligibility Agent",
        "is_valid": eligible,
        "is_eligible": eligible,
        "passed": eligible,
        "valid": eligible,
        "status": status,
        "policy_status": str(raw_status),
        "policy_id": state.get("policy_id", ""),
        "reason": reason,
        "assessment": reason,
        "errors": errors,
    }

    return {
        "policy_eligibility": result,
        "policy_check": result,
        "eligibility": result,
    }


# ============================================================
# 5. FRAUD-RISK ASSESSMENT AGENT
# ============================================================

def fraud_risk_agent(
    state: ClaimState,
) -> Dict[str, Any]:

    amount = float(state["claim_amount"])
    limit = float(state["coverage_limit"])
    previous_claims = int(state["previous_claims"])

    notes = str(state.get("fraud_notes") or "").strip()
    incident = str(
        state.get("incident_description") or ""
    ).strip()
    user_flagged = bool(state.get("high_risk", False))

    indicators = []

    if amount >= 25000:
        indicators.append(
            "The claim amount is high and may need additional review."
        )

    if limit > 0 and amount > limit:
        indicators.append(
            "The claim amount exceeds the stated coverage limit."
        )

    if limit > 0 and amount >= 0.8 * limit:
        indicators.append(
            "The claim amount is at least 80% of the coverage limit."
        )

    if previous_claims >= 3:
        indicators.append(
            "Three or more previous claims were reported."
        )

    if user_flagged:
        indicators.append(
            "The user flagged the claim for high-risk review."
        )

    if notes:
        indicators.append(
            "Additional review notes were provided."
        )

    if not incident:
        indicators.append(
            "The incident description was not provided."
        )

    high_risk = (
        amount >= 25000
        or (limit > 0 and amount > limit)
        or previous_claims >= 3
        or user_flagged
        or bool(notes)
    )

    if high_risk:
        risk_level = "High"
        reason = (
            "Potential risk indicators were identified. "
            "Human review is recommended; these indicators "
            "do not establish that fraud has occurred."
        )
    elif indicators:
        risk_level = "Medium"
        reason = (
            "Some information may require additional verification."
        )
    else:
        risk_level = "Low"
        reason = (
            "No high-risk indicators were detected by the configured "
            "rules. This does not guarantee that the claim is fraud-free."
        )

    result = {
        "agent": "Fraud-Risk Assessment Agent",
        "is_valid": not high_risk,
        "passed": not high_risk,
        "valid": not high_risk,
        "risk_level": risk_level,
        "high_risk": high_risk,
        "fraud_suspected": high_risk,
        "reason": reason,
        "assessment": reason,
        "indicators": indicators,
        "risk_indicators": indicators,
        "errors": indicators,
    }

    return {
        "fraud_check": result,
        "fraud_assessment": result,
    }


# ============================================================
# 6. CLAIM ASSESSMENT AGENT
# ============================================================

def claim_assessment_agent(
    state: ClaimState,
) -> Dict[str, Any]:

    amount = float(state["claim_amount"])
    limit = float(state["coverage_limit"])

    if amount <= 0:
        valid = False
        status = "Invalid Claim Amount"
        reason = "The claim amount must be greater than zero."

    elif limit <= 0:
        valid = False
        status = "Invalid Coverage Limit"
        reason = "The coverage limit must be greater than zero."

    elif amount > limit:
        valid = False
        status = "Exceeds Coverage Limit"
        reason = (
            f"The claim amount ({amount:,.2f}) exceeds "
            f"the coverage limit ({limit:,.2f})."
        )

    else:
        valid = True
        status = "Within Coverage Limit"
        reason = (
            f"The claim amount ({amount:,.2f}) is within "
            f"the coverage limit ({limit:,.2f})."
        )

    eligible_amount = (
        min(amount, limit)
        if amount > 0 and limit > 0
        else 0.0
    )

    result = {
        "agent": "Claim Assessment Agent",
        "is_valid": valid,
        "passed": valid,
        "valid": valid,
        "status": status,
        "reason": reason,
        "assessment": reason,
        "claim_amount": amount,
        "coverage_limit": limit,
        "eligible_amount": eligible_amount,
        "approved_amount": 0.0,
        "errors": [] if valid else [reason],
    }

    return {"claim_assessment": result}


# ============================================================
# 7. CLAIM SUMMARY AGENT
# ============================================================

def summarize_claim(
    state: ClaimState,
) -> Dict[str, Any]:

    doc = state.get("doc_verification", {})
    policy = state.get("policy_eligibility", {})
    fraud = state.get("fraud_check", {})
    assessment = state.get("claim_assessment", {})

    docs_valid = bool(doc.get("is_valid", False))
    policy_valid = bool(policy.get("is_eligible", False))
    claim_valid = bool(assessment.get("is_valid", False))
    fraud_high = bool(fraud.get("high_risk", False))

    policy_status = normalize_policy_status(
        state.get("policy_status", "")
    )

    amount = float(state["claim_amount"])
    limit = float(state["coverage_limit"])

    errors = []
    for section in (doc, policy, fraud, assessment):
        errors.extend(section.get("errors", []) or [])

    errors = list(dict.fromkeys(str(item) for item in errors if item))

    # An explicitly inactive policy is rejected by these demo rules.
    if policy_status in INACTIVE_STATUSES:
        decision = "REJECTED"
        final_status = "Rejected"
        review_required = False
        approved_amount = 0.0

    # Unknown policy status requires verification.
    elif not policy_valid:
        decision = "MANUAL REVIEW"
        final_status = "Pending Review"
        review_required = True
        approved_amount = 0.0

    # Missing documents or invalid amounts require review.
    elif not docs_valid or not claim_valid:
        decision = "MANUAL REVIEW"
        final_status = "Pending Review"
        review_required = True
        approved_amount = 0.0

    # High-risk indicators require review.
    elif fraud_high:
        decision = "MANUAL REVIEW"
        final_status = "Pending Review"
        review_required = True
        approved_amount = 0.0

    else:
        decision = "AUTO APPROVED"
        final_status = "Approved"
        review_required = False
        approved_amount = min(amount, limit)

    summary = (
        f"Decision: {decision}. "
        f"Document verification: {doc.get('reason', 'Not checked')} "
        f"Policy eligibility: {policy.get('reason', 'Not checked')} "
        f"Fraud assessment: {fraud.get('reason', 'Not checked')} "
        f"Claim assessment: {assessment.get('reason', 'Not checked')}"
    )

    report = f"""
INSURANCE CLAIM PROCESSING REPORT

Claim ID: {state["claim_id"]}
Policy ID: {state["policy_id"]}

Decision: {decision}
Final Status: {final_status}

Claim Amount: {amount:.2f}
Coverage Limit: {limit:.2f}
Approved Amount: {approved_amount:.2f}

1. DOCUMENT VERIFICATION
Status: {doc.get('status', 'Unknown')}
Assessment: {doc.get('reason', 'Not checked')}
Missing Documents: {doc.get('missing_documents', [])}

2. POLICY ELIGIBILITY
Status: {policy.get('status', 'Unknown')}
Assessment: {policy.get('reason', 'Not checked')}

3. FRAUD-RISK ASSESSMENT
Risk Level: {fraud.get('risk_level', 'Unknown')}
Assessment: {fraud.get('reason', 'Not checked')}
Indicators: {fraud.get('indicators', [])}

4. CLAIM ASSESSMENT
Status: {assessment.get('status', 'Unknown')}
Assessment: {assessment.get('reason', 'Not checked')}

Human Review Required: {"Yes" if review_required else "No"}

Additional Issues:
{chr(10).join(errors) if errors else "No additional issues reported."}

Note: This is a demonstration workflow, not a real insurance
coverage determination or a finding of fraud.
""".strip()

    return {
        "decision": decision,
        "final_decision": decision,
        "final_status": final_status,
        "claim_status": final_status,
        "doc_verification": doc,
        "document_verification": doc,
        "policy_eligibility": policy,
        "policy_check": policy,
        "eligibility": policy,
        "fraud_check": fraud,
        "fraud_assessment": fraud,
        "claim_assessment": assessment,
        "claim_amount": amount,
        "coverage_limit": limit,
        "approved_amount": approved_amount,
        "review_required": review_required,
        "assessment_summary": summary,
        "final_report": report,
        "errors": errors,
    }


# ============================================================
# 8. HUMAN APPROVAL AGENT
# ============================================================

def human_approval_agent(
    state: ClaimState,
) -> Dict[str, Any]:

    choice = str(
        state.get("human_approval_decision") or "pending"
    ).strip().lower()

    amount = float(state["claim_amount"])
    limit = float(state["coverage_limit"])

    if choice in {"approve", "approved", "accept"}:
        decision = "APPROVED BY HUMAN"
        status = "Approved by Human"
        approved_amount = min(amount, limit)
        pending = False
        reason = "The claim was approved by the human reviewer."

    elif choice in {"reject", "rejected", "decline"}:
        decision = "REJECTED BY HUMAN"
        status = "Rejected by Human"
        approved_amount = 0.0
        pending = False
        reason = "The claim was rejected by the human reviewer."

    else:
        decision = "MANUAL REVIEW"
        status = "Pending Human Approval"
        approved_amount = 0.0
        pending = True
        reason = (
            "Human review is required. No final human decision "
            "has been recorded."
        )

    approval_result = {
        "agent": "Human Approval Agent",
        "decision": choice,
        "status": status,
        "approved": choice in {"approve", "approved", "accept"},
        "pending": pending,
        "reason": reason,
        "assessment": reason,
    }

    report = (
        state.get("final_report", "")
        + "\n\n5. HUMAN APPROVAL\n"
        + f"Status: {status}\n"
        + f"Assessment: {reason}"
    ).strip()

    return {
        "human_approval": approval_result,
        "decision": decision,
        "final_decision": decision,
        "final_status": status,
        "claim_status": status,
        "approved_amount": approved_amount,
        "review_required": pending,
        "assessment_summary": (
            state.get("assessment_summary", "")
            + " "
            + reason
        ).strip(),
        "final_report": report,
    }


# ============================================================
# 9. CONDITIONAL ROUTING
# ============================================================

def route_after_summary(state: ClaimState) -> str:
    if state.get("review_required", False):
        return "human_approval_agent"

    return END


# ============================================================
# 10. BUILD LANGGRAPH
# ============================================================

def build_claim_workflow():
    graph = StateGraph(ClaimState)

    graph.add_node(
        "document_verification_agent",
        document_verification_agent,
    )
    graph.add_node(
        "policy_eligibility_agent",
        policy_eligibility_agent,
    )
    graph.add_node("fraud_risk_agent", fraud_risk_agent)
    graph.add_node("claim_assessment_agent", claim_assessment_agent)
    graph.add_node("summarize_claim", summarize_claim)
    graph.add_node("human_approval_agent", human_approval_agent)

    # Run initial assessments in parallel.
    graph.add_edge(START, "document_verification_agent")
    graph.add_edge(START, "policy_eligibility_agent")
    graph.add_edge(START, "fraud_risk_agent")
    graph.add_edge(START, "claim_assessment_agent")

    # Wait for all four assessments.
    graph.add_edge(
        [
            "document_verification_agent",
            "policy_eligibility_agent",
            "fraud_risk_agent",
            "claim_assessment_agent",
        ],
        "summarize_claim",
    )

    graph.add_conditional_edges(
        "summarize_claim",
        route_after_summary,
        {
            "human_approval_agent": "human_approval_agent",
            END: END,
        },
    )

    graph.add_edge("human_approval_agent", END)

    return graph.compile()


claim_app = build_claim_workflow()
workflow = claim_app
app = claim_app


# ============================================================
# 11. PUBLIC FUNCTION
# ============================================================

def process_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate submitted claim data and execute the LangGraph workflow.
    Required values are not silently populated with sample data.
    """

    if not isinstance(claim_data, dict):
        raise TypeError("Claim data must be provided as a dictionary.")

    data = dict(claim_data)

    required_fields = [
        "claim_id",
        "policy_id",
        "claim_amount",
        "coverage_limit",
        "policy_status",
        "incident_description",
        "previous_claims",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
        or data[field] is None
        or str(data[field]).strip() == ""
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing_fields)
        )

    try:
        data["claim_amount"] = float(data["claim_amount"])
        data["coverage_limit"] = float(data["coverage_limit"])
        data["previous_claims"] = int(data["previous_claims"])
    except (TypeError, ValueError):
        raise ValueError(
            "Claim amount, coverage limit, and previous claims "
            "must be valid numbers."
        )

    if data["claim_amount"] <= 0:
        raise ValueError("Claim amount must be greater than zero.")

    if data["coverage_limit"] <= 0:
        raise ValueError("Coverage limit must be greater than zero.")

    if data["previous_claims"] < 0:
        raise ValueError("Previous claims cannot be negative.")

    data["claim_id"] = str(data["claim_id"]).strip()
    data["policy_id"] = str(data["policy_id"]).strip()
    data["policy_status"] = str(data["policy_status"]).strip()
    data["incident_description"] = str(
        data["incident_description"]
    ).strip()

    if not data["incident_description"]:
        raise ValueError("Incident description is required.")

    if not data["policy_status"]:
        raise ValueError("Policy status must be selected.")

    data["documents"] = normalize_documents(data.get("documents", []))
    data["required_documents"] = REQUIRED_DOCUMENTS.copy()
    data["fraud_notes"] = str(data.get("fraud_notes") or "").strip()
    data["high_risk"] = bool(data.get("high_risk", False))
    data["human_approval_decision"] = str(
        data.get("human_approval_decision") or "pending"
    ).strip().lower()

    result = claim_app.invoke(data)

    # Ensure all dashboard result keys are available.
    result.setdefault("doc_verification", {})
    result.setdefault(
        "document_verification", result["doc_verification"]
    )
    result.setdefault("policy_eligibility", {})
    result.setdefault("policy_check", result["policy_eligibility"])
    result.setdefault("eligibility", result["policy_eligibility"])
    result.setdefault("fraud_check", {})
    result.setdefault("fraud_assessment", result["fraud_check"])
    result.setdefault("claim_assessment", {})
    result.setdefault("human_approval", {})
    result.setdefault("decision", "MANUAL REVIEW")
    result.setdefault("final_decision", result["decision"])
    result.setdefault("final_status", "Pending Review")
    result.setdefault("claim_status", result["final_status"])
    result.setdefault("approved_amount", 0.0)
    result.setdefault("review_required", True)
    result.setdefault("assessment_summary", "")
    result.setdefault("final_report", "")
    result.setdefault("errors", [])

    return result


if __name__ == "__main__":
    print(
        "Insurance Claim Processing workflow loaded successfully. "
        "Run app.py with Streamlit to enter a claim."
    )
