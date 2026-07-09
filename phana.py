# phishing_pandas_analytics.py
# 
# # Core Identity

# Analytics & Business Logic Engine: It is the central quantitative brain of the platform. It takes the model-ready features from Config.py and turns them into human-understandable insights, structured data marts, and actionable training plans.

# Primary Responsibilities & Logic Paths

# The "Single Source of Truth" for Risk: It houses the absolute definition of phishing risk (calculate_risk_score), ensuring that every UI component, geographic summary, and individual employee profile computes mathematical risk using the exact same formula.

# Analytical Data Mart Generation: Acts as the data aggregator (via the GenericAnalyzer class). It dynamically slices the 50,000-row DataFrame by any dimension (City, Grade, Department, Month) to compute total events, click rates, tied risk groups, and top clicked themes.

# Business Guidance & Training: Acts as the recommendation engine. It maps semantic themes to specific courses (e.g., "Urgency Recognition Training") and attaches concrete action plans using its internal IMPROVEMENT_ACTION_LIBRARY.

# Employee Profiling: Builds and retrieves deep, historical profiles for individual users (get_employee_profile), identifying their specific vulnerabilities (e.g., "High risk on Finance-themed emails") and forecasting their high-risk status.

# Data & Security Management

# Strict Role-Based Access Control (RBAC): Acts as the absolute privacy gateway. The sanitize_record_for_role function actively strips or masks personal identifiers (Names, Emails, exact BRIDs) from the JSON payloads before they are sent to the UI or the LLM, unless the requester has explicit "admin" privileges.

# Dynamic Filtering & Standardization: Handles the messy reality of data querying by resolving flexible column names (e.g., mapping "job_level" to "corporate_grade") and standardizing date filters so the LLM doesn't have to write perfect Pandas syntax.

import logging
import traceback
from collections import Counter
import numpy as np
import pandas as pd
from Config import BRID_COLUMN, EMPLOYEE_ID_COLUMNS, PERSONAL_COLUMNS, apply_feature_engineering

LOGGER = logging.getLogger("phishing_pandas_analytics")

DEFAULT_TOP_N = 20
RISK_THEME_COLUMNS = {
    "security_credential_similarity": "security_credential",
    "financial_similarity": "financial",
    "urgency_pressure_similarity": "urgency_pressure",
    "authority_trust_similarity": "authority_trust",
    "link_attachment_similarity": "link_attachment"
}

TRAINING_MAP = {
    "security_credential": "Credential Theft Awareness Training",
    "financial": "Financial Fraud Awareness Training",
    "urgency_pressure": "Urgency Recognition Training",
    "authority_trust": "Authority Impersonation Awareness Training",
    "link_attachment": "Attachment Handling Training"
}

THEME_ACTION_MAP = {
    "security_credential": [
        "Train the employee to verify login URLs before entering credentials.",
        "Use short simulations around password reset, account verification, and fake login pages.",
        "Reinforce checking sender domain, URL destination, and unexpected authentication prompts."
    ],
    "financial": [
        "Train the employee to validate payroll, invoice, refund, and payment messages through trusted internal channels.",
        "Use finance-themed simulations with realistic but safe examples.",
        "Reinforce double-checking urgency and payment-related requests before clicking."
    ],
    "urgency_pressure": [
        "Train the employee to pause before reacting to urgent, deadline-based, or threatening language.",
        "Use micro-learning focused on emotional pressure and rushed decision-making.",
        "Encourage reporting suspicious urgent emails instead of engaging quickly."
    ],
    "authority_trust": [
        "Train the employee to verify requests appearing to come from leaders, HR, managers, or compliance teams.",
        "Use authority-impersonation simulations with executive and HR-style subjects.",
        "Reinforce checking unusual tone, unexpected requests, and sender authenticity."
    ],
    "link_attachment": [
        "Train the employee to inspect attachments and shared-document links before opening.",
        "Use simulations around shared files, document review, and attachment-based phishing.",
        "Reinforce safe handling of links, files, and unexpected document access requests."
    ]
}

# Improvement action library moved here as the single source of truth for business guidance
IMPROVEMENT_ACTION_LIBRARY = {
    "security_credential": [
        "Focus on verifying login pages before entering credentials.",
        "Check sender, URL, and domain carefully before responding to account verification emails.",
        "Avoid using links inside emails for password reset or account unlock requests."
    ],
    "financial": [
        "Apply extra caution to invoice, payment, refund, payroll, and wire-transfer emails.",
        "Validate financial requests through an approved internal channel before taking action.",
        "Do not approve payment or banking requests directly from email links."
    ],
    "urgency_pressure": [
        "Pause before acting on urgent, threatening, or deadline-driven emails.",
        "Look for pressure tactics such as suspension, expiry, final notice, or immediate action.",
        "Report urgent suspicious emails instead of clicking quickly."
    ],
    "authority_trust": [
        "Do not trust an email only because it appears to come from HR, IT, security, payroll, or leadership.",
        "Verify unexpected internal requests using a trusted channel.",
        "Be cautious when authority-based emails ask for credentials, approvals, or file access."
    ],
    "link_attachment": [
        "Avoid opening unexpected attachments or shared documents.",
        "Hover over links and inspect destination domains before clicking.",
        "Use official portals instead of email links for sensitive actions."
    ],
    "low_reporting": [
        "Practise reporting suspicious emails instead of ignoring them.",
        "Use the phishing report button whenever an email looks unusual.",
        "Report first when unsure; do not click to investigate."
    ],
    "high_no_action": [
        "Reduce passive behaviour by encouraging users to report suspicious emails.",
        "Use simple reminders on when to report instead of taking no action.",
        "Track reporting adoption in the next simulation."
    ],
    "general": [
        "Run short scenario-based phishing awareness sessions.",
        "Use examples based on the most-clicked subjects and templates.",
        "Track progress using click rate, report rate, and no-action rate."
    ]
}

EMAIL_DRIVER_COLUMNS = {k: v for k, v in RISK_THEME_COLUMNS.items()}

def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def _safe_get(record, key, default=None):
    try:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)
    except Exception:
        return default

def get_top_email_risk_drivers(record, top_n=3):
    scores = []
    for column, driver in EMAIL_DRIVER_COLUMNS.items():
        value = _safe_float(_safe_get(record, column, 0.0))
        scores.append({"driver": driver, "score": round(value, 4)})
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores[:top_n]

def build_improvement_guidance(record):
    predicted_label = _safe_get(record, "predicted_label", None)
    clicked_probability = _safe_float(_safe_get(record, "prob_clicked_link", 0.0))
    reported_probability = _safe_float(_safe_get(record, "prob_reported", 0.0))
    no_action_probability = _safe_float(_safe_get(record, "prob_no_action", 0.0))

    template_subject = _safe_get(record, "templatesubject", "")
    template_name = _safe_get(record, "templatename", "")
    department = _safe_get(record, "usertags-Department", "")
    businessarea1 = _safe_get(record, "businessarea1", "")

    top_drivers = get_top_email_risk_drivers(record, top_n=3)
    actions = []
    focus_areas = []

    for item in top_drivers:
        driver = item["driver"]
        if item["score"] > 0:
            focus_areas.append(driver)
            actions.extend(IMPROVEMENT_ACTION_LIBRARY.get(driver, [])[:2])

    if reported_probability < 0.2:
        focus_areas.append("low_reporting")
        actions.extend(IMPROVEMENT_ACTION_LIBRARY["low_reporting"][:2])

    if no_action_probability >= 0.45:
        focus_areas.append("high_no_action")
        actions.extend(IMPROVEMENT_ACTION_LIBRARY["high_no_action"][:2])

    if clicked_probability >= 0.3:
        actions.extend(IMPROVEMENT_ACTION_LIBRARY["general"][:2])

    unique_actions = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    if not unique_actions:
        unique_actions = IMPROVEMENT_ACTION_LIBRARY["general"]

    return {
        "predicted_label": predicted_label,
        "click_probability": round(clicked_probability, 4),
        "reported_probability": round(reported_probability, 4),
        "report_probability": round(reported_probability, 4),
        "no_action_probability": round(no_action_probability, 4),
        "department": department or "unknown",
        "businessarea1": businessarea1 or "unknown",
        "template_name": template_name or "unknown",
        "template_subject": template_subject or "unknown",
        "top_risk_drivers": top_drivers,
        "focus_areas": list(dict.fromkeys(focus_areas)),
        "recommended_actions": unique_actions[:6]
    }

def build_population_guidance(records):
    try:
        all_focus_areas = []
        for record in records or []:
            guidance = build_improvement_guidance(record)
            all_focus_areas.extend(guidance.get("focus_areas", []))
        primary_population_vulnerabilities = [item[0] for item in Counter(all_focus_areas).most_common(3)]
        return {
            "primary_population_vulnerabilities": primary_population_vulnerabilities,
            "recommended_campaign_focus": primary_population_vulnerabilities,
        }
    except Exception:
        return {}

def _calculate_brid_risk_profile(brid_data: pd.DataFrame):
    """Compute a standardised risk profile and recommendations for a BRID.
    Accepts either a DataFrame of events for the BRID or a summary-like record (dict).
    Returns a consistent dict with rates, probabilities, risk_score and guidance.
    """
    try:
        if brid_data is None:
            return {}
        # If it's a DataFrame of events
        if isinstance(brid_data, pd.DataFrame):
            df = normalize_dataframe_features(brid_data)
            total_events = int(len(df))
            if total_events == 0:
                return {}
            clicked = int((df["target"] == 1).sum()) if "target" in df.columns else 0
            reported = int((df["target"] == 2).sum()) if "target" in df.columns else 0
            no_action = int((df["target"] == 0).sum()) if "target" in df.columns else 0

            click_rate = safe_percentage(clicked, total_events)
            report_rate = safe_percentage(reported, total_events)
            no_action_rate = safe_percentage(no_action, total_events)

            # Probabilities (0..1)
            prob_clicked = (clicked / total_events) if total_events else 0.0
            prob_reported = (reported / total_events) if total_events else 0.0
            prob_no_action = (no_action / total_events) if total_events else 0.0

            # risk score consistent with group metric (centralized)
            risk_score = calculate_risk_score(click_rate, report_rate)

            # Top drivers from mean similarity columns if present
            driver_record = {}
            for col, driver in EMAIL_DRIVER_COLUMNS.items():
                if col in df.columns:
                    driver_record[driver] = float(pd.to_numeric(df[col], errors="coerce").mean() or 0.0)

            record_for_guidance = {
                "predicted_label": None,
                "prob_clicked_link": prob_clicked,
                "prob_reported": prob_reported,
                "prob_no_action": prob_no_action,
                "templatesubject": df["templatesubject"].dropna().astype(str).iloc[0] if "templatesubject" in df.columns and not df["templatesubject"].dropna().empty else "",
                "templatename": df["templatename"].dropna().astype(str).iloc[0] if "templatename" in df.columns and not df["templatename"].dropna().empty else "",
            }
            # merge similarity drivers into record
            for k, v in driver_record.items():
                record_for_guidance[f"{k}"] = v

            guidance = build_improvement_guidance(record_for_guidance)

            return {
                "total_events": total_events,
                "click_rate_percent": click_rate,
                "report_rate_percent": report_rate,
                "no_action_rate_percent": no_action_rate,
                "risk_score": risk_score,
                "prob_clicked_link": round(prob_clicked, 4),
                "prob_reported": round(prob_reported, 4),
                "prob_no_action": round(prob_no_action, 4),
                "top_risk_drivers": guidance.get("top_risk_drivers", []),
                "focus_areas": guidance.get("focus_areas", []),
                "recommended_actions": guidance.get("recommended_actions", [])
            }

        # If it's a dict-like summary/profile
        if isinstance(brid_data, dict):
            # Use available fields to construct guidance
            prob_clicked = _safe_float(_safe_get(brid_data, "prob_clicked_link", _safe_get(brid_data, "click_rate_percent", 0.0)))
            prob_reported = _safe_float(_safe_get(brid_data, "prob_reported", _safe_get(brid_data, "report_rate_percent", 0.0)))
            prob_no_action = _safe_float(_safe_get(brid_data, "prob_no_action", _safe_get(brid_data, "no_action_rate_percent", 0.0)))
            record_for_guidance = dict(brid_data or {})
            record_for_guidance["prob_clicked_link"] = prob_clicked
            record_for_guidance["prob_reported"] = prob_reported
            record_for_guidance["prob_no_action"] = prob_no_action
            guidance = build_improvement_guidance(record_for_guidance)
            risk_score = _safe_float(_safe_get(brid_data, "risk_score", 0.0))
            return {
                "total_events": int(_safe_get(brid_data, "total_events", 0)),
                "click_rate_percent": _safe_get(brid_data, "click_rate_percent", 0.0),
                "report_rate_percent": _safe_get(brid_data, "report_rate_percent", 0.0),
                "no_action_rate_percent": _safe_get(brid_data, "no_action_rate_percent", 0.0),
                "risk_score": risk_score,
                "prob_clicked_link": round(prob_clicked, 4),
                "prob_reported": round(prob_reported, 4),
                "prob_no_action": round(prob_no_action, 4),
                "top_risk_drivers": guidance.get("top_risk_drivers", []),
                "focus_areas": guidance.get("focus_areas", []),
                "recommended_actions": guidance.get("recommended_actions", [])
            }

        return {}
    except Exception:
        log_error(traceback.format_exc())
        return {}

MONTH_MAP = {
    "jan": "01", "january": "01", "1": "01", "01": "01",
    "feb": "02", "february": "02", "2": "02", "02": "02",
    "mar": "03", "march": "03", "3": "03", "03": "03",
    "apr": "04", "april": "04", "4": "04", "04": "04",
    "may": "05", "5": "05", "05": "05",
    "jun": "06", "june": "06", "6": "06", "06": "06",
    "jul": "07", "july": "07", "7": "07", "07": "07",
    "aug": "08", "august": "08", "8": "08", "08": "08",
    "sep": "09", "sept": "09", "september": "09", "9": "09", "09": "09",
    "oct": "10", "october": "10", "10": "10",
    "nov": "11", "november": "11", "11": "11",
    "dec": "12", "december": "12", "12": "12"
}

THEME_LABEL_MAP = {
    "security_credential": "Credential Theft",
    "financial": "Financial Fraud",
    "urgency_pressure": "Urgency Action",
    "authority_trust": "Authority Impersonation",
    "link_attachment": "Attachment Risk"
}

def log_info(message):
    LOGGER.info(message)

def log_error(message):
    LOGGER.error(message)

def safe_percentage(numerator, denominator):
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0:
            return 0.0
        return round((numerator / denominator) * 100, 2)
    except Exception:
        return 0.0

def calculate_risk_score(click_rate_percent, report_rate_percent):
    """
    SINGLE SOURCE OF TRUTH FOR RISK MATH.
    Calculates a standardized 0-100 risk score based on behavioral rates.
    Weighting: 70% penalty for clicking, 30% penalty for failing to report.
    """
    try:
        click = float(click_rate_percent or 0.0)
        report = float(report_rate_percent or 0.0)
        score = (click * 0.7) + ((100.0 - report) * 0.3)
        return round(max(0.0, min(100.0, score)), 2)
    except Exception:
        return 0.0

def build_sample_size_warning(total_events, minimum=5):
    try:
        total_events = int(total_events or 0)
        if total_events < minimum:
            return f"Low sample size: based on only {total_events} simulation event(s). Interpret cautiously."
        return ""
    except Exception:
        return ""

def clean_json_value(value):
    try:
        if value is pd.NA:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return None
            return float(value)
        if isinstance(value, pd.Timestamp):
            return str(value)
        if isinstance(value, dict):
            return {str(k): clean_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean_json_value(x) for x in value]
        return value
    except Exception:
        return value

def is_admin(user_role):
    return "admin" in str(user_role or "user").strip().lower()

def is_employee_id_column(column):
    text = str(column or "").strip().lower()
    return text in [x.lower() for x in EMPLOYEE_ID_COLUMNS]

def is_personal_column(column):
    text = str(column or "").strip().lower()
    return text in [x.lower() for x in PERSONAL_COLUMNS]

def sanitize_record_for_role(record, user_role="user", allow_employee_id=False):
    try:
        record = dict(record or {})
        if is_admin(user_role):
            return clean_json_value(record)
        sanitized = {}
        hidden = False
        for key, value in record.items():
            if is_personal_column(key):
                hidden = True
                continue
            if is_employee_id_column(key) and not allow_employee_id:
                hidden = True
                continue
            sanitized[key] = value
        if hidden:
            sanitized["personal_information_hidden"] = True
        return clean_json_value(sanitized)
    except Exception:
        return {}

def sanitize_filters_for_role(filters, user_role="user", allow_employee_id=False):
    try:
        filters = dict(filters or {})
        if is_admin(user_role):
            return clean_json_value(filters)
        sanitized = {}
        hidden = False
        for key, value in filters.items():
            if is_personal_column(key):
                hidden = True
                continue
            if is_employee_id_column(key) and not allow_employee_id:
                hidden = True
                continue
            sanitized[key] = value
        if hidden:
            sanitized["personal_information_hidden"] = True
        return clean_json_value(sanitized)
    except Exception:
        return {}

def sanitize_records_for_role(records, user_role="user", allow_employee_id=False):
    try:
        if is_admin(user_role):
            return clean_json_value(records)
        return [sanitize_record_for_role(row, user_role=user_role, allow_employee_id=allow_employee_id) for row in records]
    except Exception:
        return []

def normalize_month_value(try_value):
    try:
        value = str(try_value or "").strip().lower().replace(".", "").replace("-", "").replace("/", "")
        return MONTH_MAP.get(value, value.zfill(2) if value.isdigit() and len(value) <= 2 else value)
    except Exception:
        return str(try_value or "")

def normalize_dataframe_features(df):
    try:
        if df is None:
            return pd.DataFrame()
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame()

def resolve_column(columns, candidates):
    try:
        lower_map = {str(col).lower(): col for col in columns}
        for candidate in candidates:
            key = str(candidate).lower()
            if key in lower_map:
                return lower_map[key]
        return None
    except Exception:
        return None

def add_time_features(df):
    try:
        df = df.copy()
        sent_col = resolve_column(df.columns, ["senttime", "sent_time", "timestamp"])
        if sent_col is not None:
            sent_dt = pd.to_datetime(df[sent_col], errors="coerce")
            df["sent_year"] = sent_dt.dt.year.astype("Int64")
            df["sent_month_num"] = sent_dt.dt.month.astype("Int64")
            df["sent_month_name"] = sent_dt.dt.month_name().fillna("").str.slice(0, 3).str.lower()
            df["sent_month"] = sent_dt.dt.strftime("%Y-%m")
        return df
    except Exception:
        log_error(traceback.format_exc())
        return df

def apply_feature_engineering_df(df):
    try:
        return apply_feature_engineering(df)
    except Exception:
        log_error(traceback.format_exc())
        return df

def get_top_value_counts(df, column, top_n=10):
    try:
        if df is None or df.empty or column not in df.columns:
            return pd.DataFrame(columns=[column, "count"])
        res = df[column].fillna("Unknown").astype(str).str.strip().value_counts().head(top_n).reset_index()
        res.columns = [column, "count"]
        return res
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame(columns=[column, "count"])

def get_top_value_counts_by_candidates(df, candidates, top_n=10):
    try:
        col = resolve_column(df.columns, candidates)
        if not col:
            return pd.DataFrame()
        return get_top_value_counts(df, col, top_n=top_n)
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame()

def _derive_theme_from_similarity(row):
    try:
        available = [col for col in RISK_THEME_COLUMNS if col in row.index]
        if not available:
            return "unknown"
        best_theme = None
        best_score = 0
        for col in available:
            theme = RISK_THEME_COLUMNS[col]
            try:
                score = float(row[col] or 0)
                if score > best_score:
                    best_score = score
                    best_theme = theme
            except Exception:
                continue
        return best_theme if best_theme else "unknown"
    except Exception:
        return "unknown"

def _derive_theme_fallback(row):
    try:
        merged = " ".join([str(row.get(col, "")) for col in ["templatesubject", "subject", "campaignname", "campaign"]])
        checks = [
            ("financial", ["invoice", "payment", "refund", "payroll", "tax", "salary", "bank", "bonus", "expense", "reimbursement", "account statement"]),
            ("security_credential", ["password", "credential", "login", "authentication", "reset", "verify", "account", "secure login", "mfa", "otp"]),
            ("link_attachment", ["attachment", "document", "file", "shared", "sharepoint", "onedrive", "download", "review document"]),
            ("authority_trust", ["hr", "manager", "leadership", "executive", "compliance", "ceo", "director", "admin", "it support"]),
            ("urgency_pressure", ["urgent", "immediate", "deadline", "expire", "action required", "last chance", "final reminder", "asap"])
        ]
        for theme, keywords in checks:
            if any(x in merged.lower() for x in keywords):
                return theme
        return "unknown"
    except Exception:
        return "unknown"

def _derive_theme(row):
    try:
        text_theme = _derive_theme_from_similarity(row)
        if text_theme != "unknown":
            return text_theme
        return _derive_theme_fallback(row)
    except Exception:
        return "unknown"

def detect_top_click_themes(clicked_df):
    try:
        if clicked_df is None or clicked_df.empty:
            return []
        themes = []
        for _, row in clicked_df.iterrows():
            theme = _derive_theme(row)
            if theme != "unknown":
                themes.append(theme)
        counter = Counter(themes)
        ranked = counter.most_common(5)
        return [r[0] for r in ranked if r[0] in RISK_THEME_COLUMNS.values()]
    except Exception:
        log_error(traceback.format_exc())
        return []

def build_training_recommendations(clicked_df):
    try:
        themes = detect_top_click_themes(clicked_df)
        mapping = {
            "Credential Theft": "Credential Theft Awareness Training",
            "Financial Fraud": "Financial Fraud Awareness Training",
            "Urgency Action": "Urgency Recognition Training",
            "Authority Impersonation": "Authority Impersonation Awareness Training",
            "Attachment Risk": "Attachment Handling Training"
        }
        recommendations = []
        for theme in themes:
            value = mapping.get(THEME_LABEL_MAP.get(theme, theme), "General Awareness Training")
            recs = THEME_ACTION_MAP.get(theme, [])
            if value not in [r["training_name"] for r in recommendations]:
                recommendations.append({"theme": theme, "training_name": value, "actions": recs})
        return recommendations
    except Exception:
        log_error(traceback.format_exc())
        return []

def resolve_dimension_column(column_by_groups, columns):
    mapping = {
        "city": ["city"],
        "department": ["usertags-Department", "department"],
        "coo_area": ["COO Area", "coo_area", "area"],
        "grade": ["grade", "designation", "corporate_grade"],
        "corporate_grade": ["corporate_grade", "grade", "designation"],
        "designation": ["designation", "corporate_grade", "grade"],
        "job_level": ["corporate_grade", "designation", "grade"],
        "templatename": ["template", "templatename"],
        "subject": ["templatesubject", "subject"],
        "campaignname": ["campaignname", "campaign"],
        "sent_month": ["sent_month", "month"],
        "sent_month_num": ["sent_month_num", "month_num"],
        "sent_year": ["sent_year", "year"],
        "period": ["sent_month", "month_year", "period"],
        "businessarea1": ["businessarea1", "businessarea2", "businessarea3", "businessarea4", "businessarea5"],
        "businessarea2": ["businessarea2", "businessarea1", "businessarea3", "businessarea4", "businessarea5"],
        "businessarea3": ["businessarea3", "businessarea1", "businessarea2", "businessarea4", "businessarea5"],
        "businessarea4": ["businessarea4", "businessarea1", "businessarea2", "businessarea3", "businessarea5"],
        "businessarea5": ["businessarea5", "businessarea1", "businessarea2", "businessarea3", "businessarea4"],
        "user": ["usertags-BRID", "proofpoint_brid", "brid", "BRID", "employee", "employee_id", "user", "user_id"],
        "employee_id": ["usertags-BRID", "proofpoint_brid", "brid", "BRID", "employee", "employee_id", "user", "user_id"],
        "brid": ["usertags-BRID", "proofpoint_brid", "brid", "BRID", "employee", "employee_id", "user", "user_id"]
    }
    candidates = mapping.get(str(column_by_groups), [column_by_groups])
    return resolve_column(columns, candidates)

def apply_filters(df, filters):
    try:
        if not filters:
            return df
        if df is None or df.empty:
            return pd.DataFrame()
        res_df = df.copy()
        for column, value in filters.items():
            key = resolve_column(res_df.columns, [column])
            if key in ["month", "campaign_month"]:
                month_value = normalize_month_value(value)
                if "sent_month_num" in res_df.columns:
                    filtered_df = res_df[res_df["sent_month_num"].fillna("").astype(str).str.zfill(2) == month_value]
                    if not filtered_df.empty:
                        res_df = filtered_df
                        continue
                if "sent_month" in res_df.columns:
                    filtered_df = res_df[res_df["sent_month"].fillna("").astype(str).str.slice(start=5).str.zfill(2) == month_value]
                    if not filtered_df.empty:
                        res_df = filtered_df
                        continue
                if "sent_month_name" in res_df.columns:
                    filtered_df = res_df[res_df["sent_month_name"] == month_value]
                    if not filtered_df.empty:
                        res_df = filtered_df
                        continue
            elif key in ["year", "campaign_year"]:
                year_value = str(value or "").strip()
                if "sent_year" in res_df.columns:
                    filtered_df = res_df[res_df["sent_year"].fillna("").astype(str) == year_value]
                    if not filtered_df.empty:
                        res_df = filtered_df
                        continue
            resolve_dim = resolve_dimension_column(column, res_df.columns)
            if resolve_dim is not None:
                if isinstance(value, list):
                    res_df = res_df[res_df[resolve_dim].fillna("").astype(str).str.lower().isin([str(v).strip().lower() for v in value])]
                else:
                    val_str = str(value or "").strip().lower()
                    if val_str:
                        if val_str.startswith("regex:"):
                            regex_val = val_str.split("regex:", 1)[1].strip()
                            res_df = res_df[res_df[resolve_dim].fillna("").astype(str).str.contains(regex_val, case=False, na=False)]
                        else:
                            res_df = res_df[res_df[resolve_dim].fillna("").astype(str).str.lower() == val_str]
        return res_df
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame()

def get_highest_risk_group(summary_df):
    try:
        if summary_df is None or summary_df.empty:
            return {}
        return clean_json_value(summary_df.sort_values("risk_score", ascending=False).head(1).to_dict("records")[0])
    except Exception:
        return {}

def get_highest_reporting_group(summary_df):
    try:
        if summary_df is None or summary_df.empty:
            return {}
        return clean_json_value(summary_df.sort_values("report_rate_percent", ascending=False).head(1).to_dict("records")[0])
    except Exception:
        return {}

def get_top_tied_groups(summary_df, metric, highest=True):
    try:
        if summary_df is None or summary_df.empty or metric not in summary_df.columns:
            return []
        target_value = summary_df[metric].max() if highest else summary_df[metric].min()
        metric_values = pd.to_numeric(summary_df[metric], errors="coerce")
        tied = summary_df[np.isclose(metric_values, float(target_value), equal_nan=False)]
        tied = tied.sort_values("total_events", ascending=False)
        return clean_json_value(tied.to_dict("records"))
    except Exception:
        return []

def build_recommended_actions(summary_df):
    try:
        actions = []
        if summary_df.empty:
            return actions
        highest_risk_group = summary_df.sort_values("risk_score", ascending=False).head(1).to_dict("records")[0]
        if float(highest_risk_group.get("click_rate_percent", 0) or 0) >= 20:
            actions.append({
                "priority": "High",
                "action": "Run targeted phishing awareness sessions for the highest-risk group.",
                "target": highest_risk_group.get("group_value", "Medium"),
                "action_details": "Improve reporting adoption with reminders and reporting-button guidance."
            })
        if float(highest_risk_group.get("no_action_rate_percent", 0) or 0) >= 70:
            actions.append({
                "priority": "Medium",
                "action": "Use nudges because many users are taking no action instead of reporting suspicious emails."
            })
        actions.append({"priority": "Standard", "action": "Track click rate, report rate, and no-action rate in the next simulation cycle."})
        return actions
    except Exception:
        log_error(traceback.format_exc())
        return []

def generate_insights(summary_df):
    try:
        insights = []
        if summary_df.empty:
            return insights
        top_click = summary_df.sort_values("click_rate_percent", ascending=False).head(1)
        top_report = summary_df.sort_values("report_rate_percent", ascending=False).head(1)
        top_risk = summary_df.sort_values("risk_score", ascending=False).head(1)
        
        if not top_click.empty:
            insights.append({"type": "highest_click_rate", "value": clean_json_value(top_click.iloc[0].to_dict())})
        if not top_report.empty:
            insights.append({"type": "highest_report_rate", "value": clean_json_value(top_report.iloc[0].to_dict())})
        if not top_risk.empty:
            insights.append({"type": "highest_risk_score", "value": clean_json_value(top_risk.iloc[0].to_dict())})
        return insights
    except Exception:
        log_error(traceback.format_exc())
        return []

def build_metric_summary(df, dimensions):
    try:
        if df is None or df.empty or not dimensions:
            return pd.DataFrame()
        res_df = df.copy()
        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        col_by = resolve_dimension_column(dimensions[0], res_df.columns)
        if not col_by:
            return pd.DataFrame()
        
        result = res_df.groupby(col_by, dropna=False).agg(
            total_events=("target", "count"),
            clicked_count=("target", lambda x: int((x == 1).sum())),
            reported_count=("target", lambda x: int((x == 2).sum())),
            no_action_count=("target", lambda x: int((x == 0).sum()))
        ).reset_index()
        
        result["result"] = result["total_events"].apply(build_sample_size_warning)
        result["click_rate_percent"] = result.apply(lambda r: safe_percentage(r["clicked_count"], r["total_events"]), axis=1)
        result["report_rate_percent"] = result.apply(lambda r: safe_percentage(r["reported_count"], r["total_events"]), axis=1)
        result["no_action_rate_percent"] = result.apply(lambda r: safe_percentage(r["no_action_count"], r["total_events"]), axis=1)
        
        result["risk_score"] = result.apply(
            lambda r: calculate_risk_score(r["click_rate_percent"], r["report_rate_percent"]),
            axis=1
        )
        result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
        return result
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame()

def build_risk_analysis(summary_df):
    try:
        if summary_df.empty:
            return []
        risks = []
        for _, row in summary_df.head(10).iterrows():
            score = float(row.get("risk_score", 0) or 0)
            if score >= 50:
                risk_level = "High"
            elif score >= 20:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            risks.append({"risk_level": risk_level, "risk_score": round(score, 2)})
        return risks
    except Exception:
        log_error(traceback.format_exc())
        return []

def _top_n(items, n=5):
    items = [x for x in items if x and str(x).strip().lower() not in ["nan", "none", "unknown"]]
    return [item[0] for item in Counter(items).most_common(n)] + [""] * n

def build_employee_profile_cache(raw_df):
    try:
        fe_df = normalize_dataframe_features(raw_df)
        if fe_df.empty or "target" not in fe_df.columns:
            return pd.DataFrame()
        
        brid_col = resolve_column(fe_df.columns, [BRID_COLUMN, "usertags-BRID", "proofpoint_brid", "brid"])
        if not brid_col:
            return pd.DataFrame()
            
        dept_col = resolve_column(fe_df.columns, ["usertags-Department", "department"])
        city_col = resolve_column(fe_df.columns, ["city"])
        subject_col = resolve_column(fe_df.columns, ["templatesubject", "subject"])
        template_col = resolve_column(fe_df.columns, ["templatename", "template"])
        campaign_col = resolve_column(fe_df.columns, ["campaignname", "campaign"])
        
        fe_df["derived_theme"] = fe_df.apply(_derive_theme, axis=1)
        fe_df["brid_col"] = fe_df[brid_col].fillna("").astype(str).str.strip().str.lower()
        
        profiles = []
        for brid, grp in fe_df.groupby("brid_col"):
            total_events = int(len(grp))
            target_sum = int((grp["target"] == 1).sum())
            reported_count = int((grp["target"] == 2).sum())
            no_action_count = int((grp["target"] == 0).sum())
            
            click_rate = safe_percentage(target_sum, total_events)
            report_rate = safe_percentage(reported_count, total_events)
            no_action_rate = safe_percentage(no_action_count, total_events)
            
            clicked = grp[grp["target"] == 1].copy()
            subjects = _top_n(clicked[subject_col].tolist(), 3) if subject_col and subject_col in clicked.columns else ["", ""]
            templates = _top_n(clicked[template_col].tolist(), 3) if template_col and template_col in clicked.columns else ["", ""]
            campaigns = _top_n(clicked[campaign_col].tolist(), 3) if campaign_col and campaign_col in clicked.columns else ["", ""]
            themes = _top_n(clicked["derived_theme"].tolist(), 3)

            risk_score = calculate_risk_score(click_rate, report_rate)
            
            profiles.append({
                "brid": str(brid),
                "department": str(grp[dept_col].dropna().iloc[0]).strip() if dept_col and not grp[dept_col].dropna().empty else "",
                "city": str(grp[city_col].dropna().iloc[0]).strip() if city_col and not grp[city_col].dropna().empty else "",
                "total_events": total_events,
                "clicked_count": target_sum,
                "reported_count": reported_count,
                "no_action_count": no_action_count,
                "click_rate_percent": click_rate,
                "report_rate_percent": report_rate,
                "no_action_rate_percent": no_action_rate,
                "top_clicked_theme_1": themes[0] if len(themes) > 0 else "",
                "top_clicked_theme_2": themes[1] if len(themes) > 1 else "",
                "top_clicked_theme_3": themes[2] if len(themes) > 2 else "",
                "risk_score": risk_score
            })
        return pd.DataFrame(profiles)
    except Exception:
        log_error(traceback.format_exc())
        return pd.DataFrame()

def get_employee_profile(profile_cache_df, brid, user_role="user"):
    try:
        if profile_cache_df is None or profile_cache_df.empty:
            return {}
        target = str(brid or "").strip().lower()
        result = profile_cache_df[profile_cache_df["brid"].fillna("").astype(str).str.strip().str.lower() == target]
        if result.empty:
            return {}
        profile = clean_json_value(result.iloc[0].to_dict())
        # Add canonical risk profile and guidance
        try:
            risk = _calculate_brid_risk_profile(profile)
            profile["risk_profile"] = risk
        except Exception:
            profile["risk_profile"] = {}
        # Sanitize according to role; specific BRID requests may allow employee id visibility
        return sanitize_record_for_role(profile, user_role=user_role, allow_employee_id=True)
    except Exception:
        log_error(traceback.format_exc())
        return {}

def get_top_risky_employees(profile_cache_df, limit=10, user_role="user"):
    try:
        if profile_cache_df is None or profile_cache_df.empty:
            return []
        result = profile_cache_df.sort_values("risk_score", ascending=False).head(limit).to_dict("records")
        return sanitize_records_for_role(result, user_role=user_role, allow_employee_id=True)
    except Exception:
        log_error(traceback.format_exc())
        return []

def get_employee_improvement_profile(profile_cache_df, brid, user_role="user"):
    try:
        if profile_cache_df is None or profile_cache_df.empty:
            return {}
        # Use admin retrieval to get the full canonical profile (we'll sanitize below)
        profile = get_employee_profile(profile_cache_df, brid, user_role="admin")
        if not profile:
            return {}

        # Extract canonical risk profile and themes
        risk_profile = profile.get("risk_profile", {})

        themes = []
        for key in ["top_clicked_theme_1", "top_clicked_theme_2", "top_clicked_theme_3"]:
            val = profile.get(key)
            if val and val != "":
                themes.append(val)

        recommendations = []
        for theme in themes:
            training_name = TRAINING_MAP.get(theme, "General Awareness Training")
            actions = THEME_ACTION_MAP.get(theme, [])
            recommendations.append({"theme": theme, "training_name": training_name, "actions": actions})

        output = {
            "brid": profile.get("brid"),
            "risk_score": profile.get("risk_score"),
            "click_rate_percent": profile.get("click_rate_percent"),
            "report_rate_percent": profile.get("report_rate_percent"),
            "training_recommendations": recommendations,
            "recommended_actions": risk_profile.get("recommended_actions", []),
            "focus_areas": risk_profile.get("focus_areas", [])
        }

        if not is_admin(user_role):
            output["privacy_note"] = "Raw personal profile details are hidden for non-admin users. BRID-specific behavioural insight is shown because a specific BRID was requested."
        else:
            output["department"] = profile.get("department")
            output["city"] = profile.get("city")

        return clean_json_value(output)
    except Exception:
        log_error(traceback.format_exc())
        return {}

def predict_high_risk_population(df, probability_column="prob_clicked_link", threshold=0.6, top_n=100, user_role="user"):
    try:
        df = normalize_dataframe_features(df)
        if df.empty:
            return {"status": "success", "high_risk_users": []}
        if probability_column not in df.columns:
            return {"status": "error", "message": "probability_column not found."}
            
        high_risk = df[pd.to_numeric(df[probability_column], errors="coerce").fillna(0) >= threshold]
        high_risk = high_risk.sort_values(probability_column, ascending=False)
        training_focus = build_training_recommendations(high_risk)
        
        users = []
        if user_role == "admin":
            columns = [col for col in ["brid", "corporate_grade", "designation", "usertags-Department", "department", probability_column] if col in high_risk.columns]
            users = clean_json_value(high_risk[columns].head(top_n).to_dict("records"))
        else:
            columns = [col for col in ["corporate_grade", "designation"] if col in high_risk.columns]
            users = clean_json_value(high_risk[columns].head(top_n).to_dict("records"))
        # Enforce sanitization gateway
        users = sanitize_records_for_role(users, user_role=user_role, allow_employee_id=(user_role=="admin"))
            
        return {
            "status": "success",
            "high_risk_count": int(len(high_risk)),
            "threshold": float(threshold),
            "high_risk_users": users,
            "training_focus": training_focus,
            "campaign_recommendations": [{"priority": "High", "action": "Run targeted phishing awareness sessions for the identified high-risk population."}]
        }
    except Exception as e:
        log_error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

class BaseAnalyzer:
    def run(self, df, dimensions, top_n, analysis_type=None):
        raise NotImplementedError

class GenericAnalyzer(BaseAnalyzer):
    def run(self, df, dimensions, top_n, analysis_type=None):
        summary_df = build_metric_summary(df, dimensions)
        summary_df = sort_summary_for_analysis(summary_df, analysis_type, dimensions)
        clicked_df = df[df["target"] == 1].copy() if "target" in df.columns else pd.DataFrame()
        return {
            "summary": summary_df.head(top_n).to_dict("records"),
            "highest_risk_group": get_highest_risk_group(summary_df),
            "highest_reporting_group": get_highest_reporting_group(summary_df),
            "lowest_risk_groups": get_top_tied_groups(summary_df, "risk_score", highest=False),
            "top_clicked_subjects": get_top_value_counts_by_candidates(clicked_df, ["templatesubject", "subject"], top_n),
            "top_clicked_templates": get_top_value_counts_by_candidates(clicked_df, ["templatename", "template"], top_n),
            "top_clicked_campaigns": get_top_value_counts_by_candidates(clicked_df, ["campaignname", "campaign"], top_n),
            "top_clicked_themes": detect_top_click_themes(clicked_df),
            "insights": generate_insights(summary_df),
            "risk_analysis": build_risk_analysis(summary_df),
            "training_recommendations": build_training_recommendations(clicked_df),
            "recommended_actions": build_recommended_actions(summary_df)
        }

ANALYSIS_REGISTRY = {
    "overall_analysis": GenericAnalyzer(),
    "city_performance": GenericAnalyzer(),
    "area_performance": GenericAnalyzer(),
    "coo_area_performance": GenericAnalyzer(),
    "grade_performance": GenericAnalyzer(),
    "designation_performance": GenericAnalyzer(),
    "campaign_performance": GenericAnalyzer(),
    "template_performance": GenericAnalyzer(),
    "subject_performance": GenericAnalyzer(),
    "monthly_trend": GenericAnalyzer(),
    "month_year_trend": GenericAnalyzer(),
    "training_analysis": GenericAnalyzer(),
    "recommendation_analysis": GenericAnalyzer(),
    "risk_analysis": GenericAnalyzer(),
    "business_role_analysis": GenericAnalyzer(),
    "employee_performance": GenericAnalyzer(),
    "brid_performance": GenericAnalyzer(),
    "user_performance": GenericAnalyzer()
}

def sanitize_result_for_role(result, user_role="user"):
    try:
        if is_admin(user_role):
            return result
        personal_dimension = any(is_personal_column(dim) or is_employee_id_column(dim) for dim in result.get("dimensions", []))
        if not personal_dimension:
            return result
            
        result = dict(result or {})
        result["privacy_note"] = "Personal identifiers are hidden for non-admin users in bulk/grouped views."
        if "summary" in result and isinstance(result["summary"], list):
            result["summary"] = sanitize_records_for_role(result["summary"], user_role=user_role, allow_employee_id=False)
            
        for key in ["highest_risk_group", "highest_reporting_group", "lowest_risk_group", "highest_waste_group"]:
            if key in result and isinstance(result[key], dict):
                result[key] = sanitize_record_for_role(result[key], user_role=user_role, allow_employee_id=False)
                
        if "insights" in result and isinstance(result["insights"], list):
            sanitized_insights = []
            for item in result["insights"]:
                item = dict(item or {})
                if isinstance(item.get("value"), dict):
                    item["value"] = sanitize_record_for_role(item["value"], user_role=user_role, allow_employee_id=False)
                sanitized_insights.append(item)
            result["insights"] = sanitized_insights
        return result
    except Exception:
        return result

def filters_have_specific_employee_id(filters):
    try:
        for key, value in filters.items():
            if is_employee_id_column(key) and str(value or "").strip():
                return True
        return False
    except Exception:
        return False

def sort_summary_for_analysis(summary_df, analysis_type, dimensions):
    try:
        if summary_df is None or summary_df.empty:
            return summary_df
        if analysis_type == "monthly_trend" and "sent_month" in summary_df.columns:
            return summary_df.sort_values("sent_month", ascending=True)
        if analysis_type == "yearly_trend" and "sent_year" in summary_df.columns:
            return summary_df.sort_values("sent_year", ascending=True)
        return summary_df.sort_values("risk_score", ascending=False)
    except Exception:
        return summary_df

def run_analysis(df, analysis_type, group_by=None, filters=None, user_role="user", top_n=DEFAULT_TOP_N, metadata=None):
    try:
        log_info(f"ANALYSIS_START | {analysis_type}")
        df = normalize_dataframe_features(df)
        if df.empty:
            return {"status": "success", "message": "No rows available."}
            
        df = apply_filters(df, filters)
        if df.empty:
            return {"status": "success", "message": "No rows available after filters.", "summary": []}
            
        dimensions = group_by if isinstance(group_by, list) else ([group_by] if group_by else [])
        analyzer = ANALYSIS_REGISTRY.get(analysis_type, GenericAnalyzer())
        result = analyzer.run(df=df, dimensions=dimensions, top_n=top_n, analysis_type=analysis_type)
        
        allow_id = filters_have_specific_employee_id(filters)
        output = {
            "status": "success",
            "filters_applied": sanitize_filters_for_role(filters, user_role=user_role, allow_employee_id=allow_id),
            "rows_analyzed": int(len(df)),
            "dimensions": dimensions,
            "metadata": metadata or {},
            "metric_explanation": {
                "total_events": "Total phishing simulation events analysed after filters.",
                "click_rate_percent": "Percentage of events where the user clicked the phishing link.",
                "report_rate_percent": "Percentage of events reported as phishing.",
                "no_action_count": "Number of events with no user action.",
                "risk_score": "Higher means more high click behaviour and lower reporting behaviour."
            },
            "result": sanitize_result_for_role(result, user_role=user_role)
        }
        log_info(f"ANALYSIS_END | rows={len(df)}")
        return clean_json_value(output)
    except Exception as e:
        log_error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    sample_df = pd.DataFrame([
        {
            "usertags-BRID": "75b4ae75-b", "proofpoint_brid": "75b4ae75-b", "city": "nottingham",
            "COO_Area": "Group Control", "corporate_grade": "AVP", "usertags-Department": "tax",
            "businessarea1": "Finance", "businessarea2": "tax operations", "businessarea3": "corporate tax",
            "campaignname": "PhishingSimulation_January2026", "templatename": "Password Reset",
            "templatesubject": "Urgent Password Reset Required", "eventtype": "Clicked Link",
            "senttimestamp": "2026-01-10 09:00:00", "LocalHireRehireDate": "2020-01-01", "is_hugs": "No"
        },
        {
            "usertags-BRID": "75b4ae75-b", "proofpoint_brid": "75b4ae75-b", "city": "nottingham",
            "COO_Area": "Group Control", "corporate_grade": "AVP", "usertags-Department": "tax",
            "businessarea1": "Finance", "businessarea2": "tax operations", "businessarea3": "corporate tax",
            "campaignname": "PhishingSimulation_February2026", "templatename": "Secure Login Required",
            "templatesubject": "Secure Login Verification Required", "eventtype": "Clicked Link",
            "senttimestamp": "2026-02-10 09:00:00", "LocalHireRehireDate": "2020-01-01", "is_hugs": "No"
        },
        {
            "usertags-BRID": "75b4ae75-b", "proofpoint_brid": "75b4ae75-b", "city": "nottingham",
            "COO_Area": "Group Control", "corporate_grade": "AVP", "usertags-Department": "tax",
            "businessarea1": "Finance", "businessarea2": "tax operations", "businessarea3": "corporate tax",
            "campaignname": "PhishingSimulation_March2026", "templatename": "Employee Star Awards Nomination",
            "templatesubject": "Employee Recognition Awards", "eventtype": "Clicked Link",
            "senttimestamp": "2026-03-10 09:00:00", "LocalHireRehireDate": "2020-01-01", "is_hugs": "No"
        },
        {
            "usertags-BRID": "75b4ae75-b", "proofpoint_brid": "75b4ae75-b", "city": "nottingham",
            "COO_Area": "Group Control", "corporate_grade": "AVP", "usertags-Department": "tax",
            "businessarea1": "Finance", "businessarea2": "tax operations", "businessarea3": "corporate tax",
            "campaignname": "PhishingSimulation_April2026", "templatename": "Security Reminder",
            "templatesubject": "Security Awareness Reminder", "eventtype": "Reported",
            "senttimestamp": "2026-04-10 09:00:00", "LocalHireRehireDate": "2020-01-01", "is_hugs": "No"
        },
        {
            "usertags-BRID": "c9cc69cb-3", "proofpoint_brid": "c9cc69cb-3", "city": "pune",
            "COO_Area": "Chief Information Security Office", "corporate_grade": "BA4", "usertags-Department": "technology",
            "businessarea1": "security", "businessarea2": "cyber", "businessarea3": "engineering",
            "campaignname": "PhishingSimulation_January2026", "templatename": "Payroll Update",
            "templatesubject": "Payroll Adjustment Notice", "eventtype": "Clicked Link",
            "senttimestamp": "2026-01-12 09:00:00", "LocalHireRehireDate": "2022-01-01", "is_hugs": "No"
        },
        {
            "usertags-BRID": "a1111111-x", "proofpoint_brid": "a1111111-x", "city": "london",
            "COO_Area": "Operations", "corporate_grade": "VP", "usertags-Department": "operations",
            "businessarea1": "payments", "businessarea2": "support",
            "campaignname": "PhishingSimulation_March2026", "templatename": "Invoice Payment",
            "templatesubject": "Urgent Invoice Payment Required", "eventtype": "Clicked Link",
            "senttimestamp": "2026-03-12 09:00:00", "LocalHireRehireDate": "2019-01-01", "is_hugs": "Yes"
        },
        {
            "usertags-BRID": "b2222222-y", "proofpoint_brid": "b2222222-y", "city": "mumbai",
            "COO_Area": "Shared Technology", "corporate_grade": "BA3", "usertags-Department": "engineering",
            "businessarea1": "technology", "businessarea2": "platform", "businessarea3": "dev",
            "campaignname": "PhishingSimulation_May2026", "templatename": "Security Reminder",
            "templatesubject": "Security Awareness Training", "eventtype": "No Action",
            "senttimestamp": "2026-05-12 09:00:00", "LocalHireRehireDate": "2021-01-01", "is_hugs": "No"
        }
    ])

    def print_json(title, data):
        print("\n" + "=" * 120)
        print(title)
        print("-" * 120)
        print(json.dumps(data, indent=2, default=str))

    test_cases = [
        {"title": "OVERALL ANALYSIS", "analysis_type": "overall_analysis", "group_by": ["department"]},
        {"title": "CITY PERFORMANCE", "analysis_type": "city_performance", "group_by": ["city"]},
        {"title": "DEPARTMENT PERFORMANCE", "analysis_type": "department_performance", "group_by": ["department"]},
        {"title": "BRID PERFORMANCE", "analysis_type": "brid_performance", "group_by": ["brid"]},
        {"title": "EMPLOYEE PERFORMANCE", "analysis_type": "employee_performance", "group_by": ["employee"]},
        {"title": "USER PERFORMANCE", "analysis_type": "user_performance", "group_by": ["user"]},
        {"title": "BRID FILTER ANALYSIS", "analysis_type": "brid_performance", "group_by": ["brid"], "filters": {"brid": "75b4ae75-b"}},
        {"title": "BRID AND DEPARTMENT GROUPING", "analysis_type": "brid_performance", "group_by": ["department", "brid"], "filters": {"brid": ["75b4ae75-b"]}},
        {"title": "CAMPAIGN PERFORMANCE", "analysis_type": "campaign_performance", "group_by": ["campaignname"]},
        {"title": "TEMPLATE PERFORMANCE", "analysis_type": "template_performance", "group_by": ["templatename"]},
        {"title": "SUBJECT PERFORMANCE", "analysis_type": "subject_performance", "group_by": ["templatesubject"]},
        {"title": "MONTHLY TREND", "analysis_type": "monthly_trend", "group_by": ["sent_month"]},
        {"title": "BUSINESS AREA PERFORMANCE", "analysis_type": "business_role_analysis", "group_by": ["businessarea1"]},
        {"title": "YEAR 2026 ANALYSIS", "analysis_type": "risk_analysis", "group_by": ["year"], "filters": {"year": "2026"}},
        {"title": "JANUARY 2026 ANALYSIS", "analysis_type": "monthly_trend", "group_by": ["month"], "filters": {"month": "January"}},
        {"title": "JANUARY 2026 ANALYSIS", "analysis_type": "month_year_trend", "group_by": ["month_year"], "filters": {"month_year": "January 2026"}},
        {"title": "DESIGNATION PERFORMANCE", "analysis_type": "designation_performance", "group_by": ["designation"]},
        {"title": "DESIGNATION VP FILTER", "analysis_type": "designation_performance", "group_by": ["designation"], "filters": {"designation": "VP"}},
        {"title": "CITY + YEAR FILTER", "analysis_type": "city_performance", "group_by": ["city"], "filters": {"city": "pune", "year": "2026"}},
        {"title": "DEPARTMENT + BRID + MONTH YEAR", "analysis_type": "brid_performance", "group_by": ["department", "brid"], "filters": {"month_year": "January 2026"}}
    ]

    for case in test_cases:
        res_analysis = run_analysis(df=sample_df, analysis_type=case["analysis_type"], group_by=case.get("group_by"), filters=case.get("filters"), user_role="admin", top_n=20)
        print_json(case["title"], res_analysis)

    profile_cache = build_employee_profile_cache(sample_df)
    print_json("EMPLOYEE PROFILE CACHE", profile_cache.to_dict("records"))
    
    profile = get_employee_profile(profile_cache, "75b4ae75-b")
    improvement = get_employee_improvement_profile(profile_cache, "75b4ae75-b", user_role="admin")
    top_risky = get_top_risky_employees(profile_cache, limit=5, user_role="admin")
    
    print_json("EMPLOYEE IMPROVEMENT PROFILE 75b4ae75-b", improvement)
    print_json("TOP RISKY EMPLOYEES", top_risky)

    risk_df = pd.DataFrame({
        "usertags-BRID": ["A", "B", "C"],
        "proofpoint_brid": ["A", "B", "C"],
        "city": ["Pune", "London", "Mumbai"],
        "COO_Area": ["Security", "Risk", "Technology"],
        "corporate_grade": ["BA4", "VP", "AVP"],
        "usertags-Department": ["Cyber", "Risk", "Engineering"],
        "prob_clicked_link": [0.91, 0.25, 0.80],
        "templatesubject": ["Password Reset", "Payroll Update", "Urgent Login Verification"],
        "templatename": ["Password Reset", "Payroll", "Login Verification"]
    })

    print_json("PREDICT HIGH RISK POPULATION ADMIN MODE", predict_high_risk_population(risk_df, probability_column="prob_clicked_link", threshold=0.6, user_role="admin"))
    print_json("TOP RISKY EMPLOYEES USER MODE", get_top_risky_employees(profile_cache, limit=5, user_role="user"))
    print_json("EMPLOYEE IMPROVEMENT USER MODE", get_employee_improvement_profile(profile_cache, "75b4ae75-b", user_role="user"))
    print_json("PREDICT HIGH RISK POPULATION USER MODE", predict_high_risk_population(risk_df, probability_column="prob_clicked_link", threshold=0.6, user_role="user"))
    print_json("BRID PERFORMANCE USER MODE PRIVACY TEST", run_analysis(df=sample_df, analysis_type="brid_performance", group_by=["brid"], user_role="user", top_n=20))
    print_json("BRID PERFORMANCE ADMIN MODE FULL DATA TEST", run_analysis(df=sample_df, analysis_type="brid_performance", group_by=["brid"], user_role="admin", top_n=20))

    print("\n" + "=" * 120)
    print("ALL ANALYTICS TESTS COMPLETED")
    print("=" * 120)