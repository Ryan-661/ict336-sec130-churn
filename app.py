"""
ICT336 Machine Learning · Final Project Web Application (Deliverable 5)
Section 130 (SG) · Student: Zar Ni Aung (Ryan / Iver, ID: 6610192)

Privacy & Integrity Design:
- Reads 100% of model parameters, operating threshold, test/CV metrics, top churn drivers,
  and group fairness statistics dynamically from `churn_model.joblib` and `model_info.json`.
- Zero storage of user inputs: no database, no file logging, no telemetry. All predictions
  run ephemerally in RAM and are discarded when the session closes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

st.set_page_config(
    page_title="Subscription Churn Retention Assistant · ICT336 Sec 130",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, mobile-responsive CSS styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1180px;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
        color: #f8fafc;
        padding: 1.35rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    }
    .hero-card h1 {
        color: #ffffff !important;
        font-size: 1.55rem !important;
        margin: 0 0 0.35rem 0 !important;
        line-height: 1.25 !important;
    }
    .hero-card p {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    .risk-card {
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 0.9rem;
        border-left: 7px solid;
    }
    .risk-high {
        background-color: #fef2f2;
        border-left-color: #dc2626;
        color: #7f1d1d;
    }
    .risk-mod {
        background-color: #fffbeb;
        border-left-color: #d97706;
        color: #78350f;
    }
    .risk-low {
        background-color: #f0fdfa;
        border-left-color: #0d9488;
        color: #134e4a;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.24rem 0.72rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .privacy-banner {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 0.85rem 1.05rem;
        font-size: 0.88rem;
        color: #334155;
    }
    @media (max-width: 640px) {
        .hero-card h1 {
            font-size: 1.25rem !important;
        }
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained churn pipeline...")
def load_model_bundle(path_str: str) -> Dict[str, Any]:
    return joblib.load(path_str)


@st.cache_data(show_spinner="Loading verified evaluation metadata...")
def load_model_info(path_str: str) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


DEMO_PERSONAS: Dict[str, Dict[str, Any]] = {
    "Custom / Manual Input": {
        "tenure": 6,
        "Contract": "Month-to-month",
        "MonthlyCharges": 79.85,
        "auto_total": True,
        "TotalCharges": 479.10,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": "Under 65",
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
    },
    "High-Risk New Fiber Subscriber (2 mos, Month-to-month)": {
        "tenure": 2,
        "Contract": "Month-to-month",
        "MonthlyCharges": 89.50,
        "auto_total": True,
        "TotalCharges": 179.00,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": "Under 65",
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
    },
    "Senior Watchlist Account (65+, 14 mos, Fiber)": {
        "tenure": 14,
        "Contract": "Month-to-month",
        "MonthlyCharges": 78.20,
        "auto_total": True,
        "TotalCharges": 1094.80,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": "65+",
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
    },
    "Loyal Two-Year Family Plan (54 mos, DSL + Full Support)": {
        "tenure": 54,
        "Contract": "Two year",
        "MonthlyCharges": 64.50,
        "auto_total": True,
        "TotalCharges": 3483.00,
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "No",
        "SeniorCitizen": "Under 65",
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "Yes",
    },
    "Edge Case: Brand-New Account (0 mos, $0 Total Charges)": {
        "tenure": 0,
        "Contract": "Month-to-month",
        "MonthlyCharges": 55.00,
        "auto_total": True,
        "TotalCharges": 0.00,
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "PaymentMethod": "Mailed check",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": "Under 65",
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
    },
}


def sanitize_and_validate_inputs(
    raw_inputs: Dict[str, Any],
    feature_cols: List[str],
    numeric_ranges: Dict[str, Dict[str, float]],
    category_choices: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Safely clamp, impute, and normalize any odd or extreme user inputs so the
    model pipeline never crashes while keeping the retention manager informed.
    """
    notes: List[str] = []
    cleaned: Dict[str, Any] = dict(raw_inputs)

    # 1. Tenure validation (integer >= 0, clamped safely to [0, 120])
    try:
        tenure_val = int(round(float(cleaned.get("tenure", 0))))
    except (TypeError, ValueError):
        tenure_val = 0
        notes.append("Account tenure was non-numeric and safely defaulted to 0 months.")

    if tenure_val < 0:
        tenure_val = 0
        notes.append("Negative tenure is not possible; clamped safely to 0 months.")
    elif tenure_val > 120:
        notes.append(
            f"Entered tenure ({tenure_val} months) exceeds the 72-month historical maximum; clamped to 120 months for stable scaling."
        )
        tenure_val = 120
    elif tenure_val > int(numeric_ranges.get("tenure", {}).get("max", 72)):
        notes.append(
            f"Note: Account tenure ({tenure_val} months) is above the dataset maximum ({int(numeric_ranges['tenure']['max'])} months)."
        )
    cleaned["tenure"] = tenure_val

    # 2. MonthlyCharges validation
    mc_min = float(numeric_ranges.get("MonthlyCharges", {}).get("min", 18.25))
    mc_max = float(numeric_ranges.get("MonthlyCharges", {}).get("max", 118.75))
    try:
        monthly_val = float(cleaned.get("MonthlyCharges", mc_min))
        if not math.isfinite(monthly_val):
            raise ValueError("Non-finite MonthlyCharges")
    except (TypeError, ValueError):
        monthly_val = float(numeric_ranges.get("MonthlyCharges", {}).get("median", 70.35))
        notes.append(f"Monthly Charges was invalid and reset to dataset median (${monthly_val:.2f}).")

    if monthly_val < 0.0:
        monthly_val = 0.0
        notes.append("Monthly Charges cannot be negative; clamped safely to $0.00.")
    elif monthly_val > 300.0:
        monthly_val = 300.0
        notes.append("Extreme Monthly Charges clamped to $300.00/month to prevent outlier distortion.")
    elif monthly_val < mc_min or monthly_val > mc_max:
        notes.append(
            f"Unusual billing input: Monthly Charges (${monthly_val:.2f}) falls outside the historical training range (${mc_min:.2f}–${mc_max:.2f}), but is scored safely."
        )
    cleaned["MonthlyCharges"] = round(monthly_val, 2)

    # 3. TotalCharges validation & tenure == 0 handling
    try:
        total_raw = cleaned.get("TotalCharges", None)
        total_val = float(total_raw) if total_raw is not None else np.nan
        if not math.isfinite(total_val):
            total_val = np.nan
    except (TypeError, ValueError):
        total_val = np.nan

    if tenure_val == 0:
        # In the Telco dataset, all 11 brand-new tenure=0 rows had blank TotalCharges imputed in-pipeline,
        # or $0 billed so far. Clamp negative/odd values to 0.0 (or NaN if blank) without crashing.
        if pd.isna(total_val) or total_val <= 0.0:
            cleaned["TotalCharges"] = 0.0
            notes.append(
                "Brand-new account (tenure = 0 months): first billing cycle has not completed yet — Total Charges safely set to $0.00."
            )
        else:
            cleaned["TotalCharges"] = round(min(total_val, 5000.0), 2)
            notes.append(
                f"Tenure is 0 months with upfront Total Charges (${cleaned['TotalCharges']:.2f}); scored safely."
            )
    else:
        if pd.isna(total_val):
            cleaned["TotalCharges"] = np.nan
            notes.append(
                "Total Charges was left blank; imputed safely inside the scikit-learn Pipeline using the training median."
            )
        elif total_val < 0.0:
            cleaned["TotalCharges"] = round(tenure_val * cleaned["MonthlyCharges"], 2)
            notes.append(
                f"Negative Total Charges replaced with estimated cumulative billing (${cleaned['TotalCharges']:.2f})."
            )
        else:
            if total_val > 25000.0:
                total_val = 25000.0
                notes.append("Extreme Total Charges clamped safely to $25,000.00.")
            expected = tenure_val * cleaned["MonthlyCharges"]
            if expected > 0 and (total_val < 0.35 * expected or total_val > 2.2 * expected) and tenure_val >= 2:
                notes.append(
                    f"Billing consistency note: Total Charges (${total_val:,.2f}) differs noticeably from Tenure × Monthly Charges (${expected:,.2f}). Verify if plan upgrades or discounts occurred."
                )
            cleaned["TotalCharges"] = round(total_val, 2)

    # 4. Logical service dependency guardrails
    if cleaned.get("PhoneService") == "No":
        cleaned["MultipleLines"] = "No phone service"
    elif cleaned.get("MultipleLines") == "No phone service":
        cleaned["MultipleLines"] = "No"

    internet_addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    if cleaned.get("InternetService") == "No":
        for col in internet_addons:
            cleaned[col] = "No internet service"
    else:
        for col in internet_addons:
            if cleaned.get(col) == "No internet service":
                cleaned[col] = "No"

    # 5. Validate categorical values against known choices (fallback safely)
    for col, allowed in category_choices.items():
        if cleaned.get(col) not in allowed and allowed:
            cleaned[col] = allowed[0]

    ordered_row = {col: cleaned[col] for col in feature_cols}
    return pd.DataFrame([ordered_row]), notes


def build_action_recommendations(
    row: pd.Series,
    churn_prob: float,
    threshold: float,
    senior_fpr_gap_pct: float,
) -> Tuple[str, str, str, List[str]]:
    """Return (risk_tier, css_class, badge_color, bullet_recommendations)."""
    moderate_cutoff = max(0.35, round(threshold - 0.20, 2))

    if churn_prob >= threshold:
        tier = "High Churn Risk — Prioritize Human Outreach"
        css_cls = "risk-high"
        badge_hex = "#dc2626"
    elif churn_prob >= moderate_cutoff:
        tier = "Moderate Risk — Onboarding & Service Watchlist"
        css_cls = "risk-mod"
        badge_hex = "#d97706"
    else:
        tier = "Low Churn Risk — Retain Standard Care (No Discount Needed)"
        css_cls = "risk-low"
        badge_hex = "#0d9488"

    actions: List[str] = []
    if churn_prob >= threshold:
        actions.append(
            "**Step 1 (Human Review Required):** Assign a retention specialist to review recent support tickets and billing history before sending any promotional discount."
        )
        if row["Contract"] == "Month-to-month":
            actions.append(
                "**Contract Stabilization:** Offer an optional 1-year loyalty plan quote with price-lock protection (month-to-month accounts churn ~4× more often than 1-year plans)."
            )
        if row["InternetService"] == "Fiber optic" and (
            row["TechSupport"] == "No" or row["OnlineSecurity"] == "No"
        ):
            actions.append(
                "**Service Value Check-In:** Provide a complimentary Wi-Fi/fiber connection check or 60-day trial of Tech Support / Online Security rather than cutting base subscription price."
            )
        if int(row["tenure"]) <= 12:
            actions.append(
                "**First-Year Onboarding Care:** Trigger a proactive onboarding satisfaction call—churn peaks within the first 10–12 months."
            )
        if row["PaymentMethod"] == "Electronic check":
            actions.append(
                "**Billing Friction Reduction:** Invite the subscriber to enroll in automatic bank transfer or credit-card autopay."
            )
    elif churn_prob >= moderate_cutoff:
        actions.append(
            "**Lightweight Engagement:** Do not issue costly margin-reducing discounts yet; send a non-intrusive service health check or bundle awareness tip."
        )
        if row["Contract"] == "Month-to-month":
            actions.append(
                "**Renewal Timing:** Highlight annual plan savings at the subscriber's next billing statement."
            )
        if row["TechSupport"] == "No" and row["InternetService"] != "No":
            actions.append(
                "**Self-Service Support:** Share quick router/app troubleshooting resources so minor technical friction does not escalate."
            )
    else:
        actions.append(
            "**Protect Margin (Avoid Blanket Discounts):** This subscriber shows strong retention signals. Preserve promotional budget for genuinely high-risk accounts."
        )
        actions.append(
            "**Standard Loyalty Experience:** Continue normal service quality monitoring without unsolicited retention calls."
        )

    if row["SeniorCitizen"] == "65+" and churn_prob >= moderate_cutoff:
        actions.append(
            f"**Module 8 Senior Fairness Safeguard (`65+` Account):** Our hold-out fairness audit shows a `{senior_fpr_gap_pct:.1f}` percentage-point higher False Positive Rate on 65+ subscribers. Start with a polite, zero-pressure service check-in and verify communication preferences before proposing any contract change."
        )

    return tier, css_cls, badge_hex, actions


def make_probability_chart(churn_prob: float, avg_churn: float, threshold: float) -> go.Figure:
    """Interactive horizontal comparison chart helping the manager compare customer risk vs cut-off."""
    moderate_cutoff = max(0.35, round(threshold - 0.20, 2))
    labels = [
        "Dataset Average Churn Rate",
        f"Watchlist Cut-Off ({moderate_cutoff*100:.0f}%)",
        f"Action Threshold ({threshold*100:.0f}%)",
        "Selected Subscriber Risk",
    ]
    values = [avg_churn * 100, moderate_cutoff * 100, threshold * 100, churn_prob * 100]
    sub_color = "#dc2626" if churn_prob >= threshold else ("#d97706" if churn_prob >= moderate_cutoff else "#0d9488")
    colors = ["#64748b", "#f59e0b", "#1e293b", sub_color]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0f172a", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color="#dc2626",
        line_width=2,
        annotation_text=f"Flag Cut-Off ({threshold*100:.0f}%)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Decision Chart 1: Subscriber Churn Probability vs. Operating Cut-Off",
        xaxis=dict(title="Churn Probability (%)", range=[0, 108], gridcolor="#e2e8f0"),
        yaxis=dict(title=""),
        height=295,
        margin=dict(l=10, r=42, t=44, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(size=12, color="#0f172a"),
    )
    return fig


def make_benchmark_chart(row: pd.Series, benchmarks: Dict[str, Any], avg_churn: float) -> go.Figure:
    """Compare this subscriber's contract, internet tier, tenure, and monthly bill to cohort benchmarks."""
    contract_rate = float(benchmarks["contract_churn_rate_pct"].get(row["Contract"], avg_churn * 100))
    internet_rate = float(benchmarks["internet_churn_rate_pct"].get(row["InternetService"], avg_churn * 100))
    med_ten_no = float(benchmarks["median_tenure_retained_months"])
    med_ten_yes = float(benchmarks["median_tenure_churned_months"])
    med_mc_no = float(benchmarks["median_monthly_charges_retained"])
    med_mc_yes = float(benchmarks["median_monthly_charges_churned"])

    categories = [
        f"Contract Cohort<br>({row['Contract']})",
        f"Internet Cohort<br>({row['InternetService']})",
        "Monthly Bill ($)",
        "Account Tenure (mos)",
    ]
    subscriber_vals = [contract_rate, internet_rate, float(row["MonthlyCharges"]), float(row["tenure"])]
    retained_bench = [avg_churn * 100, avg_churn * 100, med_mc_no, med_ten_no]
    churned_bench = [42.71, 41.89, med_mc_yes, med_ten_yes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="This Subscriber / Cohort",
            x=categories,
            y=subscriber_vals,
            marker_color="#2563eb",
            text=[f"{subscriber_vals[0]:.1f}%", f"{subscriber_vals[1]:.1f}%", f"${subscriber_vals[2]:.0f}", f"{subscriber_vals[3]:.0f}m"],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Typical Retained Benchmark",
            x=categories,
            y=retained_bench,
            marker_color="#0d9488",
            text=[f"{retained_bench[0]:.1f}%", f"{retained_bench[1]:.1f}%", f"${retained_bench[2]:.0f}", f"{retained_bench[3]:.0f}m"],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Typical Churned Benchmark",
            x=categories,
            y=churned_bench,
            marker_color="#dc2626",
            text=[f"{churned_bench[0]:.1f}%", f"{churned_bench[1]:.1f}%", f"${churned_bench[2]:.0f}", f"{churned_bench[3]:.0f}m"],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Decision Chart 2: Subscriber Risk Factors vs. Retained & Churned Benchmarks",
        barmode="group",
        yaxis=dict(title="Rate (%) / Dollars ($) / Months", range=[0, max(max(subscriber_vals), 95) * 1.22], gridcolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=330,
        margin=dict(l=10, r=15, t=62, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(size=11.5, color="#0f172a"),
    )
    return fig


def main() -> None:
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        st.error(
            "Model artifacts (`churn_model.joblib` and `model_info.json`) were not found in `Final_Project/`. "
            "Please run `ICT336_130_Notebook.ipynb` top-to-bottom in Google Colab and place the exported files in `Final_Project/`."
        )
        st.stop()

    bundle = load_model_bundle(str(MODEL_PATH))
    info = load_model_info(str(INFO_PATH))

    pipeline = bundle["pipeline"]
    feature_cols: List[str] = bundle["feature_cols"]
    numeric_ranges: Dict[str, Dict[str, float]] = bundle["numeric_ranges"]
    category_choices: Dict[str, List[str]] = bundle["category_choices"]

    holdout = info["holdout_test_evaluation"]
    threshold = float(holdout["selected_threshold"])
    final_scores = holdout["final_model"]
    dummy_scores = holdout["dummy_baseline"]
    dataset_meta = info["dataset"]
    benchmarks = dataset_meta["benchmarks"]
    avg_churn = float(dataset_meta["overall_churn_rate"])
    fairness_final = info["fairness_audit"]["holdout_test_final"]
    senior_fpr_gap_pct = float(fairness_final["senior_citizen"]["fpr_gap"]) * 100.0

    # Hero Header
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>📊 Subscription Customer Churn Early-Warning Tool</h1>
            <p>
                <b>Decision Support for Telecom & Subscription Retention Managers</b> ·
                Model: <code>{final_scores['name']}</code> (Operating Cut-Off: <b>{threshold*100:.0f}%</b> ·
                Hold-Out Test F1: <b>{final_scores['f1']:.4f}</b> vs. Baseline <b>{dummy_scores['f1']:.4f}</b>) ·
                <i>Zero User Input Storage</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_predict, tab_about, tab_ethics = st.tabs(
        [
            "🎯 1. Subscriber Risk Assessment & Decision",
            "📈 2. About the Model & Test Scores",
            "🛡️ 3. Data Source, Fairness Limits & Privacy",
        ]
    )

    # =========================================================================
    # TAB 1: INPUT FORM, PREDICTION, ACTION PLAYBOOK & DECISION CHARTS
    # =========================================================================
    with tab_predict:
        st.markdown("#### Quick Demo Preset (Optional) or Enter Subscriber Details Below")
        preset_name = st.selectbox(
            "Load a sample subscriber scenario to test or customize any field below:",
            options=list(DEMO_PERSONAS.keys()),
            index=0,
            help="Select a realistic subscriber persona for live demos, then adjust any input below.",
        )
        defaults = DEMO_PERSONAS[preset_name]

        col_form, col_result = st.columns([1.08, 1.12], gap="large")

        with col_form:
            st.markdown("### 1. Subscriber Account Input Form")
            st.caption("Plain-language pre-billing CRM attributes only — no names, emails, or account IDs.")

            c1, c2 = st.columns(2)
            with c1:
                tenure_in = st.number_input(
                    "Account Tenure (months)",
                    min_value=0,
                    max_value=120,
                    value=int(defaults["tenure"]),
                    step=1,
                    help="Number of months the customer has been subscribed (0 = brand-new account).",
                )
                contract_in = st.selectbox(
                    "Contract Type",
                    options=["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]),
                )
                monthly_in = st.number_input(
                    "Monthly Charges ($ USD / month)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(defaults["MonthlyCharges"]),
                    step=1.0,
                    format="%.2f",
                    help="Current recurring monthly subscription bill in USD.",
                )
            with c2:
                payment_opts = [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
                payment_in = st.selectbox(
                    "Payment Method",
                    options=payment_opts,
                    index=payment_opts.index(defaults["PaymentMethod"]),
                )
                paperless_in = st.selectbox(
                    "Paperless Billing",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["PaperlessBilling"]),
                )
                auto_total = st.checkbox(
                    "Auto-calculate Total Charges ($ = Tenure × Monthly)",
                    value=bool(defaults["auto_total"]),
                    help="Uncheck to enter a custom cumulative Total Charges ($) amount or test edge cases.",
                )

            if auto_total:
                computed_total = round(float(tenure_in) * float(monthly_in), 2)
                st.info(f"Estimated Cumulative Total Charges: **${computed_total:,.2f} USD** (`{tenure_in} mos × ${monthly_in:.2f}`)")
                total_in = computed_total
            else:
                total_in = st.number_input(
                    "Total Cumulative Charges ($ USD)",
                    min_value=0.0,
                    max_value=25000.0,
                    value=0.0 if tenure_in == 0 else float(defaults["TotalCharges"]),
                    step=10.0,
                    format="%.2f",
                    help="Total amount billed over the subscriber's lifetime ($0.00 if tenure = 0).",
                )

            st.markdown("##### Subscribed Telecom & Add-On Services")
            s1, s2 = st.columns(2)
            with s1:
                internet_in = st.selectbox(
                    "Internet Service Tier",
                    options=["Fiber optic", "DSL", "No"],
                    index=["Fiber optic", "DSL", "No"].index(defaults["InternetService"]),
                )
                phone_in = st.selectbox(
                    "Phone Service",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["PhoneService"]),
                )
                if phone_in == "No":
                    multiple_in = "No phone service"
                    st.text_input("Multiple Phone Lines", value="No phone service", disabled=True)
                else:
                    default_ml = defaults["MultipleLines"] if defaults["MultipleLines"] in ["No", "Yes"] else "No"
                    multiple_in = st.selectbox(
                        "Multiple Phone Lines",
                        options=["No", "Yes"],
                        index=["No", "Yes"].index(default_ml),
                    )

            with s2:
                if internet_in == "No":
                    st.caption("Internet add-ons automatically set to *No internet service*.")
                    sec_in = "No internet service"
                    tech_in = "No internet service"
                    backup_in = "No internet service"
                    dev_in = "No internet service"
                    tv_in = "No internet service"
                    mov_in = "No internet service"
                else:
                    sec_in = st.selectbox(
                        "Online Security Add-On",
                        options=["No", "Yes"],
                        index=["No", "Yes"].index(defaults["OnlineSecurity"] if defaults["OnlineSecurity"] in ["No", "Yes"] else "No"),
                    )
                    tech_in = st.selectbox(
                        "Tech Support Plan",
                        options=["No", "Yes"],
                        index=["No", "Yes"].index(defaults["TechSupport"] if defaults["TechSupport"] in ["No", "Yes"] else "No"),
                    )
                    with st.expander("More Add-Ons (Backup, Device Protection, Streaming)", expanded=False):
                        backup_in = st.selectbox(
                            "Online Backup",
                            options=["No", "Yes"],
                            index=["No", "Yes"].index(defaults["OnlineBackup"] if defaults["OnlineBackup"] in ["No", "Yes"] else "No"),
                        )
                        dev_in = st.selectbox(
                            "Device Protection",
                            options=["No", "Yes"],
                            index=["No", "Yes"].index(defaults["DeviceProtection"] if defaults["DeviceProtection"] in ["No", "Yes"] else "No"),
                        )
                        tv_in = st.selectbox(
                            "Streaming TV",
                            options=["No", "Yes"],
                            index=["No", "Yes"].index(defaults["StreamingTV"] if defaults["StreamingTV"] in ["No", "Yes"] else "No"),
                        )
                        mov_in = st.selectbox(
                            "Streaming Movies",
                            options=["No", "Yes"],
                            index=["No", "Yes"].index(defaults["StreamingMovies"] if defaults["StreamingMovies"] in ["No", "Yes"] else "No"),
                        )

            with st.expander("Household & Demographic Context (Generalized — Audited for Fairness)", expanded=True):
                d1, d2 = st.columns(2)
                with d1:
                    senior_in = st.selectbox(
                        "Age Group (Senior Citizen 65+)",
                        options=["Under 65", "65+"],
                        index=["Under 65", "65+"].index(defaults["SeniorCitizen"]),
                    )
                    partner_in = st.selectbox(
                        "Has Partner",
                        options=["No", "Yes"],
                        index=["No", "Yes"].index(defaults["Partner"]),
                    )
                with d2:
                    gender_in = st.selectbox(
                        "Gender",
                        options=["Female", "Male"],
                        index=["Female", "Male"].index(defaults["gender"]),
                    )
                    dep_in = st.selectbox(
                        "Has Dependents",
                        options=["No", "Yes"],
                        index=["No", "Yes"].index(defaults["Dependents"]),
                    )

        raw_subscriber = {
            "gender": gender_in,
            "SeniorCitizen": senior_in,
            "Partner": partner_in,
            "Dependents": dep_in,
            "tenure": tenure_in,
            "PhoneService": phone_in,
            "MultipleLines": multiple_in,
            "InternetService": internet_in,
            "OnlineSecurity": sec_in,
            "OnlineBackup": backup_in,
            "DeviceProtection": dev_in,
            "TechSupport": tech_in,
            "StreamingTV": tv_in,
            "StreamingMovies": mov_in,
            "Contract": contract_in,
            "PaperlessBilling": paperless_in,
            "PaymentMethod": payment_in,
            "MonthlyCharges": monthly_in,
            "TotalCharges": total_in,
        }

        input_df, validation_notes = sanitize_and_validate_inputs(
            raw_subscriber, feature_cols, numeric_ranges, category_choices
        )
        churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
        flagged = churn_prob >= threshold

        risk_tier, css_cls, badge_hex, recommendations = build_action_recommendations(
            input_df.iloc[0], churn_prob, threshold, senior_fpr_gap_pct
        )

        with col_result:
            st.markdown("### 2. Churn Prediction & Retention Action")

            st.markdown(
                f"""
                <div class="risk-card {css_cls}">
                    <span class="badge-pill" style="background:{badge_hex}; color:#ffffff;">
                        {'FLAGGED FOR OUTREACH' if flagged else 'BELOW OUTREACH CUT-OFF'}
                    </span>
                    <h2 style="margin:0.45rem 0 0.2rem 0; font-size:1.85rem; color:inherit;">
                        Predicted Churn Risk: {churn_prob*100:.1f}%
                    </h2>
                    <p style="margin:0; font-size:1.0rem; font-weight:600;">
                        {risk_tier}
                    </p>
                    <p style="margin:0.35rem 0 0 0; font-size:0.86rem; opacity:0.9;">
                        Operating Decision Cut-Off: <b>{threshold*100:.0f}%</b> ·
                        Dataset Baseline Churn Rate: <b>{avg_churn*100:.1f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if validation_notes:
                for note in validation_notes:
                    st.warning(f"Input Handling Note: {note}")

            st.markdown("#### Recommended Retention Manager Playbook")
            for rec in recommendations:
                st.markdown(f"- {rec}")

            st.plotly_chart(
                make_probability_chart(churn_prob, avg_churn, threshold),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("---")
        st.plotly_chart(
            make_benchmark_chart(input_df.iloc[0], benchmarks, avg_churn),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # =========================================================================
    # TAB 2: ABOUT THE MODEL (DYNAMICALLY LOADED FROM model_info.json)
    # =========================================================================
    with tab_about:
        st.markdown("### About the Model — Single Hold-Out Test Evaluation & Drivers")
        st.caption(
            f"All scores below are loaded dynamically from `model_info.json` (scikit-learn `{info['environment']['sklearn_version']}`, "
            f"`random_state = {info['environment']['random_state']}`, single hold-out test evaluation on `{holdout['test_rows']:,}` accounts)."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Hold-Out Test F1 (Main)",
            f"{final_scores['f1']:.4f}",
            f"+{holdout['f1_gain_percentage_points']:.2f} pp vs. Dummy",
        )
        m2.metric("Hold-Out Recall (TPR)", f"{final_scores['recall']*100:.1f}%", "Churners Caught")
        m3.metric("Hold-Out Precision", f"{final_scores['precision']*100:.1f}%", f"Cut-Off = {threshold:.2f}")
        m4.metric("Hold-Out ROC-AUC", f"{final_scores['roc_auc']:.4f}", "vs. 0.5000 Dummy")

        criterion_ok = bool(holdout["meets_success_criterion"])
        if criterion_ok:
            st.success(
                f"✅ **Proposal Numeric Success Criterion Achieved:** Hold-out Test F1 = **{final_scores['f1']:.4f}** "
                f"(target `>= {holdout['proposal_target_f1']:.2f}`) and beats `DummyClassifier` (`{dummy_scores['f1']:.4f}`) "
                f"by **+{holdout['f1_gain_percentage_points']:.2f} percentage points** (target `>= +{holdout['proposal_min_gain_pp']:.0f} pp`)."
            )
        else:
            st.warning("Proposal numeric success criterion was not met on the hold-out split.")

        st.markdown("#### 1. Hold-Out Test Set Comparison (`n = 1,409`, Evaluated Strictly Once)")
        test_table = pd.DataFrame(
            [
                {
                    "Model": f"{final_scores['name']} (Threshold = {threshold:.2f})",
                    "Test F1 (Main Metric)": f"{final_scores['f1']:.4f}",
                    "Test Recall": f"{final_scores['recall']:.4f}",
                    "Test Precision": f"{final_scores['precision']:.4f}",
                    "Test ROC-AUC": f"{final_scores['roc_auc']:.4f}",
                    "Test Accuracy": f"{final_scores['accuracy']:.4f}",
                },
                {
                    "Model": dummy_scores["name"],
                    "Test F1 (Main Metric)": f"{dummy_scores['f1']:.4f}",
                    "Test Recall": f"{dummy_scores['recall']:.4f}",
                    "Test Precision": f"{dummy_scores['precision']:.4f}",
                    "Test ROC-AUC": f"{dummy_scores['roc_auc']:.4f}",
                    "Test Accuracy": f"{dummy_scores['accuracy']:.4f}",
                },
            ]
        )
        st.dataframe(test_table, use_container_width=True, hide_index=True)

        cm_dict = final_scores["confusion_matrix"]
        st.caption(
            f"Hold-Out Confusion Matrix at Cut-Off `{threshold:.2f}`: "
            f"**TP = {cm_dict['tp']}** caught churners · **FN = {cm_dict['fn']}** missed churners · "
            f"**FP = {cm_dict['fp']}** loyal customers flagged · **TN = {cm_dict['tn']}** loyal customers retained."
        )

        col_cv, col_drivers = st.columns([1.05, 1.05], gap="large")
        with col_cv:
            st.markdown("#### 2. 5-Fold Stratified CV Model Comparison (`X_train`, `n = 5,634`)")
            cv_df = pd.DataFrame(info["cv_model_comparison"])
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

            st.markdown("#### 3. Top 3 Churn Drivers")
            for drv in info["top_features"][:3]:
                odds_txt = f" · Odds Ratio `{drv['odds_ratio']:.2f}×`" if drv.get("odds_ratio") is not None else ""
                st.markdown(
                    f"- **#{drv['rank']} {drv['feature']}** (`β = {drv['coefficient']:+.4f}`{odds_txt}) — *{drv['direction']}*"
                )

        with col_drivers:
            st.markdown("#### Top 10 Churn Drivers (Standardized Pipeline Coefficients)")
            top_df = pd.DataFrame(info["top_features"]).sort_values("abs_importance", ascending=True)
            fig_imp = go.Figure(
                go.Bar(
                    x=top_df["coefficient"],
                    y=top_df["feature"],
                    orientation="h",
                    marker_color=["#dc2626" if v > 0 else "#0d9488" for v in top_df["coefficient"]],
                    text=[f"{v:+.3f}" for v in top_df["coefficient"]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig_imp.add_vline(x=0, line_color="#0f172a", line_width=1)
            fig_imp.update_layout(
                xaxis_title="Standardized Coefficient β (< 0 Protects Retention | > 0 Raises Risk)",
                yaxis_title="",
                height=360,
                margin=dict(l=10, r=42, t=15, b=35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 4. Final Hold-Out Group Fairness Audit (`SeniorCitizen` & `gender`)")
        fcol1, fcol2 = st.columns(2, gap="large")
        with fcol1:
            sr_audit = fairness_final["senior_citizen"]
            st.markdown(
                f"**Age Group (`SeniorCitizen`):** Four-Fifths Ratio = `{sr_audit['four_fifths_ratio']:.3f}` "
                f"(`{sr_audit['status']}`) · FPR Gap = `{sr_audit['fpr_gap']*100:.2f} pp`"
            )
            st.dataframe(pd.DataFrame(sr_audit["groups"]), use_container_width=True, hide_index=True)
        with fcol2:
            gen_audit = fairness_final["gender"]
            st.markdown(
                f"**Gender (`gender`):** Four-Fifths Ratio = `{gen_audit['four_fifths_ratio']:.3f}` "
                f"(`{gen_audit['status']}`) · FPR Gap = `{gen_audit['fpr_gap']*100:.2f} pp`"
            )
            st.dataframe(pd.DataFrame(gen_audit["groups"]), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: DATA SOURCE, MODEL LIMITATIONS & ZERO-STORAGE PRIVACY NOTICE
    # =========================================================================
    with tab_ethics:
        st.markdown("### Data Source, Model Limitations & Zero-Storage Privacy Guarantee")

        st.markdown("#### 1. Cited Dataset & Open Educational Licence")
        st.markdown(
            f"- **Dataset:** {dataset_meta['name']} (`{dataset_meta['total_rows']:,}` customer accounts × `{dataset_meta['total_columns']}` columns; `{dataset_meta['predictor_count']}` pre-billing predictors after dropping pseudo-identifier `customerID`).\n"
            f"- **Licence:** {dataset_meta['licence']} — open public sample dataset permitting student educational modeling and deployment.\n"
            f"- **Verified Source Repository:** [{dataset_meta['repository_url']}]({dataset_meta['repository_url']})"
        )

        st.markdown("#### 2. Model Limitations & Where Not to Trust Predictions")
        st.markdown(
            f"- **Senior Citizen False-Positive Caveat (`FPR Gap = {senior_fpr_gap_pct:.2f} pp`):** Even after raising our operating probability cut-off to `{threshold:.2f}` during training CV threshold tuning, `65+` subscribers are flagged at a higher rate (`Four-Fifths Ratio = {fairness_final['senior_citizen']['four_fifths_ratio']:.3f} < 0.80`) because senior accounts in this dataset subscribe more heavily to month-to-month fiber-optic plans with electronic check billing. **Mitigation:** Model scores must never trigger automated contract changes or pricing penalties; a human retention specialist reviews every flagged account first.\n"
            "- **Unobserved Real-Time Events (False Negatives):** Because the dataset captures monthly billing snapshots rather than live network outage logs, call-center wait times, or household relocation events, sudden dissatisfaction on long-tenure 1-year/2-year accounts can be missed.\n"
            "- **Domain Generalization:** Trained on historical North American telecom subscription records; probability cut-offs should be recalibrated on local subscriber cohorts before deployment in Southeast Asian telecom or SaaS markets."
        )

        st.markdown("#### 3. Strict Privacy & Zero Data Storage Guarantee (Thailand PDPA B.E. 2562 Aligned)")
        st.markdown(
            """
            <div class="privacy-banner">
                <b>🔒 Zero Input Storage Guarantee:</b> This web application <b>does not log, save, cache per-user, or transmit</b>
                anything you type into the form. There is <b>no database, no file logging, no cookie tracking, and Streamlit usage
                telemetry is disabled</b>. No personal identifiers (names, customer IDs, phone numbers, or emails) are collected.
                Every prediction is computed ephemerally in memory and discarded immediately when you close or refresh the page.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Persistent footer notice across all tabs
    st.markdown("---")
    st.caption(
        f"ICT336 Applied ML Project · {info['project']['section']} · {info['project']['student']} · "
        f"Dataset: {dataset_meta['name']} ({dataset_meta['licence']}) · "
        "Privacy: Zero user inputs are stored or logged."
    )


if __name__ == "__main__":
    main()
