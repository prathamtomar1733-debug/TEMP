# phishing_mcp_server.py
# MCP Action Engine & Tool Host: It acts as the "Hands" of the operation. While the llm_server.py decides what to do, this server is the registered host that actually executes the tools, manages state, and interacts with the underlying data infrastructure.
# Primary Responsibilities & Logic Paths
# Tool Registry & Lifecycle: Serves as the authoritative registry for all MCP-exposed tools (e.g., predict_risk, simulation_users). It registers these functions with the FastMCP protocol, handling the parsing of JSON arguments provided by the LLM.
# Pipeline Orchestration: Acts as the glue between the Data Manager (Source of Truth) and the Analytics Engine. It retrieves the required data slices, pipes them into the phishing_pandas_analytics module for processing, and formats the output into structured, actionable JSON payloads.
# Model Inference & Prediction: Handles the loading and hosting of the pre-trained ML model. It manages the real-time inference lifecycle, converting model probabilities into human-readable risk labels and populating prediction metadata for the UI.
# Testing & Diagnostics: Provides a built-in suite of integration tests. The server can be invoked in a test mode to run end-to-end execution of all tools, ensuring that changes in the analytics or configuration layers haven't broken the downstream MCP functionality.
# Data & Security Management
# Privacy Gatekeeping: Implements strict Role-Based Access Control (RBAC). It acts as the final firewall, ensuring that sensitive PII or raw BRID data is sanitized and masked (e.g., sanitize_records) before being sent to the LLM or UI if the user role is not authorized.
# Resilient Error Handling: Encapsulates all tool execution in robust try-except blocks. If an analytical operation fails, the server catches the exception, logs the event for debugging, and returns a clean, non-crashing error message to the client, preventing service disruption.


import os
import sys
import json
import time
import pickle
import logging
from datetime import datetime
from functools import wraps
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from mcp.server.fastmcp import FastMCP
from Config import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    TARGET_LABELS,
    DROP_AFTER_FEATURE_ENGINEERING,
    _align_to_feature_columns,
    BRID_COLUMN,
    PII_COLUMNS,
)
from phana import (
    ANALYSIS_REGISTRY,
    build_improvement_guidance,
    build_population_guidance,
    get_employee_improvement_profile,
    get_employee_profile,
    get_top_risky_employees,
    predict_high_risk_population as analytics_high_risk_population,
    run_analysis,
)

from dman import get_primary_cache, refresh_primary_cache, get_profile_dataframe as dm_get_profile_dataframe, get_cache_stats, clear_primary_cache

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger("phishing_mcp_server")

mcp = FastMCP("PhishingAnalytics")

MODEL = None
FEATURE_COLUMNS = None
ENGINE = None
PREDICTION_CACHE = None

GLOBAL_ANALYTICS_ROWS = int(os.getenv("GLOBAL_ANALYTICS_ROWS", "50000"))
CACHE_AUTO_REFRESH_MINUTES = int(os.getenv("CACHE_AUTO_REFRESH_MINUTES", "60"))
ENABLE_CACHE = str(os.getenv("ENABLE_CACHE", "true")).strip().lower() in {"1", "true", "yes", "y"}
DEFAULT_POPULATION_LIMIT = int(os.getenv("DEFAULT_POPULATION_LIMIT", "1000"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.60"))

TOOL_REGISTRY = {
    "run_analytics": {
        "description": "Historical aggregate phishing analytics with counts, percentages, rates, trends, and grouped summaries.",
        "strict_guidance": "Use this tool only for historical aggregate metrics and never for user-level predictions.",
        "allowed_args": ["analysis_type", "group_by", "filters", "top_n", "user_role"],
        "ui_instruction": "Render the tool output as a human-readable summary and show the execution trace as JSON in the debug panel.",
    },
    "employee_lookup": {
        "description": "Employee historical lookup for profiles, top risky employees, and filtered employee searches.",
        "strict_guidance": "Use this tool for history and employee lookup only. Do not use it for prediction or recommendations.",
        "allowed_args": ["mode", "brid", "city", "department", "limit", "user_role"],
        "ui_instruction": "Render the employee profile or matching records in the UI and show the trace JSON below.",
    },
    "predict_risk": {
        "description": "Prediction tool for BRID-based or payload-based phishing risk scoring.",
        "strict_guidance": "Use this tool only when the user asks about predicted behaviour, risk likelihood, or high-risk population.",
        "allowed_args": ["mode", "brid", "payload", "limit", "threshold", "user_role"],
        "ui_instruction": "Render the prediction result and probability breakdown in the UI, with the JSON trace beneath it.",
    },
    "recommend_actions": {
        "description": "Actionable improvement recommendations for an employee, group, or overall organisation.",
        "strict_guidance": "Use this tool only for improvement guidance, training suggestions, or action plans.",
        "allowed_args": ["mode", "brid", "group_by", "filters", "top_n", "user_role"],
        "ui_instruction": "Render the recommendations and focus areas in the UI while keeping the execution trace available in JSON.",
    },
    "simulation_users": {
        "description": "List users in a simulation campaign for clicked, reported, or no-action outcomes.",
        "strict_guidance": "Use this tool for campaign user lists only. Avoid using it for aggregate analytics.",
        "allowed_args": ["campaign_month", "campaign_year", "campaign_name", "event_type", "limit", "user_role"],
        "ui_instruction": "Display the campaign user list and keep the JSON trace in the debug panel.",
    },
    "cache_control": {
        "description": "Refresh or clear the in-memory analytics cache.",
        "strict_guidance": "Use this tool only for cache refresh or cache clear intent.",
        "allowed_args": ["action"],
        "ui_instruction": "Show the cache refresh/clear outcome and the trace JSON below it.",
    },
    "cache_status": {
        "description": "Report cache state and statistics.",
        "strict_guidance": "Use this tool only for cache status or cache statistics requests.",
        "allowed_args": ["include_statistics"],
        "ui_instruction": "Show the cache state and statistics in the UI, plus the full JSON trace.",
    },
    "system_info": {
        "description": "Provide health status, schema information, features, or environment diagnostics.",
        "strict_guidance": "Use this tool only for system health, schema, feature-column, or environment diagnostics.",
        "allowed_args": ["mode"],
        "ui_instruction": "Show the diagnostics and schema details in the UI while keeping the execution trace available.",
    },
}

SAFE_HIGH_RISK_COLUMNS = [
    "city",
    "COO_Area",
    "corporate_grade",
    "usertags-Department",
    "department",
    "businessarea1",
    "businessarea2",
    "businessarea3",
    "campaignname",
    "templatename",
    "templatesubject",
    "eventtype",
    "senttimestamp",
    "predicted_label",
    "prob_no_action",
    "prob_clicked_link",
    "prob_reported",
]

ADMIN_EXTRA_COLUMNS = [
    "userfirstname",
    "userlastname",
    "useremailaddress",
    "sso_id",
    "usertags-Azure UPN",
    "usertags-BRID",
    "proofpoint_brid",
]

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        LOGGER.info(f"ENTER | {func.__name__}")
        try:
            result = func(*args, **kwargs)
            LOGGER.info(f"EXIT | {func.__name__} | {round(time.time() - start, 2)}s")
            return result
        except Exception as e:
            LOGGER.exception(f"ERROR | {func.__name__} | {str(e)}")
            raise
    return wrapper

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_admin(user_role=None):
    return str(user_role or "user").strip().lower() == "admin"

def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default

def clean_json_value(value):
    try:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return None if pd.isna(value) or np.isinf(value) else float(value)
        if isinstance(value, float):
            return None if pd.isna(value) or np.isinf(value) else value
        if isinstance(value, (pd.Timestamp, datetime)):
            return str(value)
        if isinstance(value, dict):
            return {str(k): clean_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean_json_value(x) for x in value]
        return value
    except Exception:
        return str(value)

def get_target_label(value):
    try:
        return TARGET_LABELS[int(value)]
    except Exception:
        return str(value)

def label_probabilities(probabilities, classes=None):
    if probabilities is None:
        return []
    classes = classes or [0, 1, 2]
    rows = []
    for row in probabilities:
        item = {}
        for class_value, prob in zip(classes, row):
            item[get_target_label(class_value)] = round(float(prob), 4)
        rows.append(item)
    return rows

def prediction_llm_context():
    return {
        "result_type": "classification_prediction",
        "important_instruction": "Use probabilities labeled. Do not assume probability order.",
        "recommended_summary": [
            "state predicted label",
            "state No Action probability",
            "state Clicked Link probability",
            "state Reported probability",
        ],
    }

def analytics_llm_context():
    return {
        "result_type": "grouped_phishing_analytics",
        "risk_score_definition": "risk_score compares click behaviour against reporting behaviour.",
        "recommended_summary": [
            "state highest risk group",
            "state click/report/no-action rates",
            "state recommendations",
            "do not invent risk labels",
        ],
    }

def improvement_llm_context():
    return {
        "result_type": "improvement_recommendations",
        "important_instruction": "Use behavioural security language, not employee performance assessment.",
        "recommended_summary": [
            "state behaviour pattern",
            "state focus areas",
            "state training",
            "state concrete actions",
        ],
    }

def employee_llm_context(user_role="user"):
    return {
        "result_type": "employee_historical_behaviour_profile",
        "privacy_mode": "admin" if is_admin(user_role) else "user",
        "important_instruction": "This is historical phishing behaviour analysis, not performance assessment.",
    }

def high_risk_llm_context(user_role="user", threshold=HIGH_RISK_THRESHOLD):
    return {
        "result_type": "high_risk_population",
        "threshold_used": threshold,
        "privacy_mode": "admin" if is_admin(user_role) else "user",
        "pii_policy": "PII only in admin mode.",
    }

def simulation_llm_context(user_role="user", search_type=None):
    guidance = None
    if search_type == "month_only":
        guidance = "Month only was supplied; answer using selected campaign and suggest adding year for precision."
    elif search_type == "year_only":
        guidance = "Year only was supplied; answer using selected campaign and suggest adding month for precision."
    elif search_type == "month_year":
        guidance = "Month and year supplied; no ambiguity note needed."
    return {
        "result_type": "simulation_users",
        "privacy_mode": "admin" if is_admin(user_role) else "user",
        "campaign_search_type": search_type,
        "campaign_guidance": guidance,
        "recommended_summary": [
            "state campaign",
            "state event type",
            "state user count",
            "do not print all BRIDs unless explicitly needed and admin mode is enabled",
        ],
    }

def cache_llm_context():
    return {
        "result_type": "cache_status",
        "recommended_summary": [
            "state cache loaded status",
            "state rows",
            "state employee profile count",
            "state refresh time",
        ],
    }

def system_llm_context(result_type="system_info"):
    return {"result_type": result_type}

@log_execution
def validate_environment():
    checks = {
        "DB_SERVER": bool(os.getenv("DB_SERVER")),
        "DB_DATABASE": bool(os.getenv("DB_DATABASE")),
        "DB_TABLE": bool(os.getenv("DB_TABLE")),
        "ODBC_DRIVER": bool(os.getenv("ODBC_DRIVER")),
        "MODEL_PATH": bool(MODEL_PATH),
        "FEATURE_COLUMNS_PATH": bool(FEATURE_COLUMNS_PATH),
        "GLOBAL_ANALYTICS_ROWS": GLOBAL_ANALYTICS_ROWS > 0,
        "SEMANTIC_MODEL_PATH": bool(os.getenv("SEMANTIC_MODEL_PATH")),
    }
    missing = [k for k, v in checks.items() if not v]
    return {"valid": len(missing) == 0, "checks": checks, "missing": missing}

@log_execution
def validate_prediction_payload(payload):
    payload = dict(payload or {})
    recommended_fields = [
        "city", "COO_Area", "corporate_grade", "usertags-Department",
        "businessarea1", "businessarea2", "businessarea3", "businessarea4", "businessarea5",
        "campaignname", "templatename", "templatesubject", "eventtype", "senttimestamp",
        "LocalHireRehireDate", "is_hugs"
    ]
    minimum_fields = ["templatesubject"]
    field_labels = {
        "city": "City", "COO_Area": "COO area", "corporate_grade": "Corporate grade/designation",
        "usertags-Department": "Department", "businessarea1": "Business area 1", "businessarea2": "Business area 2",
        "businessarea3": "Business area 3", "businessarea4": "Business area 4", "businessarea5": "Business area 5",
        "campaignname": "Campaign name", "templatename": "Template name", "templatesubject": "Template subject",
        "eventtype": "Event type", "senttimestamp": "Sent timestamp", "LocalHireRehireDate": "Local hire/rehire date",
        "is_hugs": "HUGS / monitored-user flag"
    }
    
    missing_minimum_fields = []
    missing_recommended_fields = []
    empty_fields = []
    present_fields = []
    
    for field in recommended_fields:
        value = payload.get(field)
        is_empty = value is None or str(value).strip() == "" or str(value).strip().lower() in ["none", "nan", "null"]
        if is_empty:
            missing_recommended_fields.append(field)
            empty_fields.append(field)
        else:
            present_fields.append(field)
            
    for field in minimum_fields:
        value = payload.get(field)
        is_empty = value is None or str(value).strip() == "" or str(value).strip().lower() in ["none", "nan", "null"]
        if is_empty:
            missing_minimum_fields.append(field)

    present_count = len(present_fields)
    total_count = len(recommended_fields)
    completeness_percent = round((present_count / max(total_count, 1)) * 100, 2)
    
    if missing_minimum_fields:
        input_quality = "invalid"
    elif completeness_percent >= 80:
        input_quality = "high"
    elif completeness_percent >= 40:
        input_quality = "medium"
    else:
        input_quality = "low"
        
    can_predict = len(missing_minimum_fields) == 0
    if input_quality == "high":
        user_message = "Prediction input is strong. Most recommended fields are present."
    elif can_predict:
        user_message = "Prediction input is complete enough to proceed, but adding missing fields can improve accuracy."
    else:
        user_message = "One or more required fields are missing."
        
    return {
        "can_predict": can_predict,
        "input_quality": input_quality,
        "completeness_percent": completeness_percent,
        "present_fields": present_fields,
        "empty_fields": empty_fields,
        "missing_recommended_fields": missing_recommended_fields,
        "missing_minimum_fields": missing_minimum_fields,
        "missing_recommended_fields_readable": [field_labels.get(x, x) for x in missing_recommended_fields],
        "missing_minimum_fields_readable": [field_labels.get(x, x) for x in missing_minimum_fields],
        "recommended_fields": recommended_fields,
        "recommended_fields_readable": [field_labels.get(x, x) for x in recommended_fields],
        "user_message": user_message,
        "llm_instruction": "If input_quality is medium or low, briefly mention that prediction was generated with incomplete input and list the most important missing fields. Do not overstate confidence.",
    }

@log_execution
def load_prediction_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    return MODEL

@log_execution
def load_feature_columns():
    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is not None:
        return FEATURE_COLUMNS
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(f"FEATURE_COLUMNS_PATH not found: {FEATURE_COLUMNS_PATH}")
    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
    return FEATURE_COLUMNS

def get_db_table():
    table_name = os.getenv("DB_TABLE") or os.getenv("table")
    if not table_name:
        raise ValueError("DB_TABLE missing in .env")
    return table_name

@log_execution
def get_sql_engine():
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    server = os.getenv("DB_SERVER") or os.getenv("database")
    driver = os.getenv("ODBC_DRIVER") or os.getenv("driver") or "ODBC Driver 17 for SQL Server"
    if not server or not os.getenv("DB_DATABASE"):
        raise ValueError("DB_SERVER/DB_DATABASE missing in .env")
    connection_string = (
        f"mssql+pyodbc://@{server}/{os.getenv('DB_DATABASE')}?"
        f"driver={driver.replace(' ', '+')}&trusted_connection=yes&TrustServerCertificate=yes"
    )
    ENGINE = create_engine(connection_string, pool_pre_ping=True)
    return ENGINE

@log_execution
def execute_query(query, params=None):
    try:
        df = pd.read_sql(text(query), get_sql_engine(), params=params or {})
        LOGGER.info(f"QUERY_SUCCESS | rows={len(df)}")
        return df
    except Exception as e:
        LOGGER.exception(f"QUERY_FAILED | {str(e)}")
        return pd.DataFrame()

def build_profile_lookup(profile_df):
    if profile_df is None or profile_df.empty or "brid" not in profile_df.columns:
        return {}
    output = {}
    for row in profile_df.to_dict("records"):
        brid = str(row.get("brid", "")).strip().lower()
        if brid:
            output[brid] = row
    return output

def validate_cache_frames(raw_df, profile_df):
    validation = {
        "raw_rows": 0,
        "raw_valid": False,
        "profile_profiles": 0,
        "profile_valid": False,
        "missing_raw_columns": [],
        "warnings": [],
    }
    required_raw = ["eventtype", "senttimestamp"]
    raw_cols = set(raw_df.columns) if raw_df is not None else set()
    validation["raw_valid"] = int(len(raw_df)) > 0 if raw_df is not None else False
    validation["raw_rows"] = int(len(raw_df)) if raw_df is not None else 0
    validation["missing_raw_columns"] = [c for c in required_raw if c not in raw_cols]
    
    brid_present = any(c in raw_cols for c in [BRID_COLUMN, "usertags-BRID", "proofpoint_brid", "brid"])
    if not brid_present:
        validation["warnings"].append("No BRID column found in raw dataset.")
    validation["raw_valid"] = validation["raw_valid"] > 0 and len(validation["missing_raw_columns"]) == 0
    validation["profile_valid"] = profile_df is not None and not profile_df.empty and "brid" in profile_df.columns
    return validation

@log_execution
def clear_cache_internal():
    # Delegate clearing to data_manager and return fresh metadata from the source of truth.
    try:
        meta = clear_primary_cache() or {}
        global PREDICTION_CACHE
        PREDICTION_CACHE = None
        return clean_json_value({
            "loaded": False,
            "last_refresh_time": None,
            "total_rows": 0,
            "employee_profiles": 0,
            "profile_lookup_count": 0,
            "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
            "cache_enabled": ENABLE_CACHE,
            "source": meta.get("source", "database"),
            "validation": {},
        })
    except Exception as e:
        LOGGER.exception(str(e))
        return clean_json_value({"loaded": False, "error": str(e)})

@log_execution
def refresh_cache_internal():
    global PREDICTION_CACHE
    try:
        new_raw = refresh_primary_cache()
        new_profile = get_profile_dataframe()

        if new_raw is None or (hasattr(new_raw, 'empty') and new_raw.empty):
            return clean_json_value({
                "loaded": False,
                "error": "Refresh failed: no rows loaded. Existing cache preserved.",
                "last_refresh_time": None,
                "total_rows": 0,
                "employee_profiles": 0,
                "profile_lookup_count": 0,
                "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
                "cache_enabled": ENABLE_CACHE,
                "source": "database",
                "validation": {},
            })

        validation = validate_cache_frames(new_raw, new_profile)
        if not validation.get("raw_valid"):
            return clean_json_value({
                "loaded": False,
                "error": "Refresh failed validation. Existing cache preserved.",
                "last_refresh_time": None,
                "total_rows": 0,
                "employee_profiles": 0,
                "profile_lookup_count": 0,
                "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
                "cache_enabled": ENABLE_CACHE,
                "source": "database",
                "validation": validation,
            })

        PREDICTION_CACHE = None
        stats = get_cache_stats() or {}
        return clean_json_value({
            "loaded": True,
            "last_refresh_time": stats.get("last_refresh_time") or now_string(),
            "total_rows": int(len(new_raw)),
            "employee_profiles": int(len(new_profile)) if new_profile is not None else 0,
            "profile_lookup_count": int(len(build_profile_lookup(new_profile))) if new_profile is not None else 0,
            "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
            "cache_enabled": ENABLE_CACHE,
            "source": stats.get("source", "database"),
            "validation": validation,
        })
    except Exception as e:
        LOGGER.exception(str(e))
        return clean_json_value({"loaded": False, "error": str(e)})

@log_execution
def ensure_cache():
    if not ENABLE_CACHE:
        return False
    stats = get_cache_stats() or {}
    if not stats.get("loaded"):
        refresh_cache_internal()
        return True
    _refresh_time = stats.get("last_refresh_time")
    if not _refresh_time:
        refresh_cache_internal()
        return True
    try:
        age = (datetime.now() - datetime.strptime(_refresh_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
        if age >= CACHE_AUTO_REFRESH_MINUTES:
            refresh_cache_internal()
            return True
    except Exception:
        refresh_cache_internal()
        return True
    return True

@log_execution
def get_analytics_dataframe():
    # Delegate retrieval to data_manager single-source cache
    df = get_primary_cache()
    if df is None or (hasattr(df, 'empty') and df.empty):
        raise ValueError("Analytics dataset unavailable")
    return df.copy()

@log_execution
def get_profile_dataframe():
    # Delegate to data_manager profile getter
    df = dm_get_profile_dataframe()
    if df is None:
        return pd.DataFrame()
    return df.copy()

def cache_metadata_for_analysis():
    stats = get_cache_stats() or {}
    profile_df = get_profile_dataframe()
    return {
        "cache_loaded": stats.get("loaded"),
        "cache_rows": stats.get("total_rows"),
        "employee_profiles": stats.get("employee_profiles"),
        "profile_lookup_count": int(len(build_profile_lookup(profile_df))) if profile_df is not None else 0,
        "last_refresh_time": stats.get("last_refresh_time"),
        "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
    }

@log_execution
def prepare_prediction_features_and_source(df):
    feature_columns = load_feature_columns()

    # Trust the input: df is already engineered by data_manager.py
    source_df = df.copy()

    # Strip down to only what the model needs
    prediction_df = source_df.drop(columns=[c for c in DROP_AFTER_FEATURE_ENGINEERING if c in source_df.columns], errors="ignore")
    prediction_df = _align_to_feature_columns(prediction_df, feature_columns)

    LOGGER.info(f"PREDICTION_INPUT_SHAPE | {prediction_df.shape}")
    return prediction_df, source_df

def sanitize_records(records, user_role="user"):
    try:
        if is_admin(user_role):
            return clean_json_value(records)
        safe_records = []
        for row in records:
            item = dict(row or {})
            for col in PII_COLUMNS:
                item.pop(col, None)
            item["personal_information_hidden"] = True
            safe_records.append(item)
        return clean_json_value(safe_records)
    except Exception:
        return []

@log_execution
def predict_dataframe(df, user_role="admin"):
    try:
        if df is None or df.empty:
            return {"status": "error", "message": "No data provided", "llm_context": prediction_llm_context()}
        model = load_prediction_model()
        X, source_df = prepare_prediction_features_and_source(df)
        if X.empty:
            return {"status": "error", "message": "No valid rows after feature engineering", "llm_context": prediction_llm_context()}
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]
        class_index = {c: i for i, c in enumerate(classes)}
        
        output_df = source_df.reset_index(drop=True).head(len(predictions)).copy()
        output_df["predicted_class"] = predictions
        output_df["predicted_label"] = output_df["predicted_class"].map(get_target_label)
        output_df["prob_no_action"] = probabilities[:, class_index.get(0, 0)]
        output_df["prob_clicked_link"] = probabilities[:, class_index.get(1, min(1, probabilities.shape[1] - 1))]
        output_df["prob_reported"] = probabilities[:, class_index.get(2, min(2, probabilities.shape[1] - 1))]
        
        full_records = clean_json_value(output_df.head(20).to_dict("records"))
        safe_records = sanitize_records(full_records, user_role)
        
        return clean_json_value({
            "status": "success",
            "rows": int(len(X)),
            "predictions": predictions.tolist(),
            "prediction_labels": [get_target_label(x) for x in predictions],
            "probabilities_labeled": label_probabilities(probabilities, classes),
            "prediction_records": safe_records,
            "improvement_guidance": [build_improvement_guidance(r) for r in full_records],
            "llm_context": prediction_llm_context(),
        })
    except Exception as e:
        LOGGER.exception(str(e))
        return {"status": "error", "message": str(e), "llm_context": prediction_llm_context()}

def prediction_summary(predictions):
    try:
        return clean_json_value(pd.Series(predictions).map(get_target_label).value_counts().to_dict())
    except Exception:
        return {}

def latest_rows_by_brid(df, brid):
    if df is None or df.empty or not brid or str(brid).strip() == "":
        return pd.DataFrame()
    cols = [c for c in [BRID_COLUMN, "usertags-BRID", "proofpoint_brid", "brid"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask = mask | (df[col].fillna("").astype(str).str.strip().str.lower() == str(brid).strip().lower())
    if df[mask].empty:
        return df[mask].copy()
    res = df[mask].copy()
    res["_sort_dt"] = pd.to_datetime(res["senttimestamp"], errors="coerce")
    result = res.sort_values("_sort_dt", ascending=False).drop(columns=["_sort_dt"], errors="ignore")
    return result

def build_campaign_search_filters(campaign_month=None, campaign_year=None, campaign_name=None):
    name = str(campaign_name or "").strip()
    month = str(campaign_month or "").strip()
    year = str(campaign_year or "").strip()
    if name:
        return {"search_type": "campaign_name", "campaign_pattern": f"%{name}%"}
    if month and year:
        return {"search_type": "month_year", "month_pattern": f"%{month}%", "year_pattern": f"%{year}%"}
    if month:
        return {"search_type": "month_only", "month_pattern": f"%{month}%"}
    if year:
        return {"search_type": "year_only", "year_pattern": f"%{year}%"}
    return {"search_type": "unknown"}

def get_cache_statistics_payload():
    df = get_analytics_dataframe()
    profile_df = get_profile_dataframe()
    stats = get_cache_stats() or {}
    departments = 0
    if "usertags-Department" in df.columns:
        departments = int(df["usertags-Department"].nunique())
    elif "department" in df.columns:
        departments = int(df["department"].nunique())
        
    return {
        "rows": int(len(df)),
        "employee_profiles": int(len(profile_df)),
        "cities": int(df["city"].nunique()) if "city" in df.columns else 0,
        "departments": departments,
        "campaigns": int(df["campaignname"].nunique()) if "campaignname" in df.columns else 0,
        "templates": int(df["templatename"].nunique()) if "templatename" in df.columns else 0,
        "last_refresh_time": stats.get("last_refresh_time"),
        "global_rows_limit": GLOBAL_ANALYTICS_ROWS,
    }

def run_prediction_by_payload(payload, user_role="user"):
    payload = dict(payload or {})
    input_validation = validate_prediction_payload(payload)
    if not input_validation.get("can_predict"):
        return clean_json_value({
            "status": "error",
            "message": "Prediction input is incomplete.",
            "input_validation": input_validation,
            "user_guidance": "Ask the user to provide the missing minimum fields before prediction.",
        })
    result = predict_dataframe(pd.DataFrame([payload]), user_role=user_role)
    if result.get("status") != "success":
        return result
        
    return clean_json_value({
        "status": "success",
        "prediction": result.get("prediction_labels", [None])[0],
        "probabilities_labeled": result.get("probabilities_labeled", [{}])[0],
        "prediction_records": result.get("prediction_records", [{}])[0],
        "improvement_guidance": result.get("improvement_guidance", [{}])[0],
        "input_validation": input_validation,
        "user_guidance": {
            "summary": "Prediction complete.",
            "input_validation": input_validation,
            "completeness_percent": input_validation.get("completeness_percent"),
            "missing_fields_to_improve_prediction": input_validation.get("missing_recommended_fields_readable"),
        },
        "llm_context": {
            **prediction_llm_context(),
            "input_validation": input_validation,
            "important_instruction": "Summarize the prediction briefly. If input quality is medium or low, mention that providing missing fields can improve prediction accuracy. Do not overstate confidence."
        }
    })

def run_prediction_by_brid(brid, user_role="user"):
    df = latest_rows_by_brid(get_analytics_dataframe(), brid).head(1)
    if df.empty:
        query = f"SELECT TOP 1 * FROM {get_db_table()} WHERE {BRID_COLUMN} = :brid OR proofpoint_brid = :brid ORDER BY senttimestamp DESC"
        df = execute_query(query, {"brid": brid})
        if df.empty:
            return {
                "status": "error",
                "message": "BRID not found",
                "lookup_context": {"lookup_type": "BRID", "brid": brid if is_admin(user_role) else "hidden_user_mode"},
                "llm_context": {"result_type": "brid_prediction_not_found"},
                "user_message": f"BRID {brid if is_admin(user_role) else 'hidden_user_mode'} not found in historical records."
            }
    result = predict_dataframe(df, user_role=user_role)
    if result.get("status") != "success":
        return result
    return clean_json_value({
        "status": "success",
        "rows": int(len(df)),
        "prediction_records": result.get("prediction_records", []),
        "probabilities_labeled": result.get("probabilities_labeled", []),
        "prediction_summary": prediction_summary(result.get("predictions", [])),
        "llm_context": {"result_type": "recent_population_prediction_distribution"},
    })

def run_high_risk_population_prediction(limit=DEFAULT_POPULATION_LIMIT, user_role="user"):
    global PREDICTION_CACHE
    threshold = HIGH_RISK_THRESHOLD
    df = get_analytics_dataframe().copy()
    limit = max(1, min(safe_int(limit, DEFAULT_POPULATION_LIMIT), GLOBAL_ANALYTICS_ROWS))
    df = df.head(limit)
    
    cache_key = f"rows: {len(df)}, threshold: {threshold}, user_role: {user_role}"
    
    model = load_prediction_model()
    X, source_df = prepare_prediction_features_and_source(df)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]
    class_index = {c: i for i, c in enumerate(classes)}
    
    output_df = source_df.reset_index(drop=True).head(len(predictions)).copy()
    output_df["predicted_class"] = predictions
    output_df["predicted_label"] = output_df["predicted_class"].map(get_target_label)
    output_df["prob_no_action"] = probabilities[:, class_index.get(0, 0)]
    output_df["prob_clicked_link"] = probabilities[:, class_index.get(1, min(1, probabilities.shape[1] - 1))]
    output_df["prob_reported"] = probabilities[:, class_index.get(2, min(2, probabilities.shape[1] - 1))]
    
    high_risk_df = output_df[output_df["prob_clicked_link"] >= threshold].copy()
    top_full_df = high_risk_df.sort_values("prob_clicked_link", ascending=False).head(20).copy()
    
    columns = SAFE_HIGH_RISK_COLUMNS + (ADMIN_EXTRA_COLUMNS if is_admin(user_role) else [])
    columns = [c for c in columns if c in top_full_df.columns]
    
    top_records = clean_json_value(top_full_df[columns].to_dict("records"))
    guidance_records = clean_json_value(top_full_df.to_dict("records"))
    guidance = [build_improvement_guidance(r) for r in guidance_records]
    
    for i, record in enumerate(top_records):
        record["improvement_guidance"] = guidance[i]
        
    analytics_summary = analytics_high_risk_population(
        output_df, probability_column="prob_clicked_link", threshold=threshold, top_n=20, user_role=user_role
    )
    
    response = {
        "status": "success",
        "user_role": "admin" if is_admin(user_role) else "user",
        "pii_exposed": is_admin(user_role),
        "population_size": int(len(output_df)),
        "high_risk_count": int(len(high_risk_df)),
        "high_risk_percentage": round((len(high_risk_df) / max(len(output_df), 1)) * 100, 2),
        "avg_click_probability": round(float(output_df["prob_clicked_link"].mean()), 4),
        "threshold_used": threshold,
        "top_high_risk_users": top_records,
        "population_improvement_guidance": build_population_guidance(guidance_records),
        "analytics_high_risk_summary": analytics_summary,
        "prediction_cache_key": cache_key,
        "llm_context": high_risk_llm_context(user_role, threshold),
    }
    PREDICTION_CACHE = {"created_at": now_string(), "cache_key": cache_key, "response": response}
    return clean_json_value(response)

def first_non_empty(*values, default=None):
    for value in values:
        if value not in [None, "", [], {}]:
            return value
    return default

def compact_percent(value):
    try:
        if value is None or pd.isna(value):
            return None
        return f"{round(float(value), 2)}%"
    except Exception:
        return None

def get_label_from_row(row):
    if not row:
        return None
    if not isinstance(row, dict):
        return "Selected group"
    for key in ["group", "city", "department", "campaign", "campaignname", "template", "templatename", "subject", "templatesubject", "month", "month_year", "brid"]:
        if row.get(key) not in [None, ""]:
            return str(row.get(key))
    return "Selected group"

def build_ui_summary(tool_name, result):
    status = result.get("status", "unknown")
    if status == "error":
        return first_non_empty(result.get("message"), result.get("validation_error"), "The request could not be completed.")
        
    if tool_name == "run_analytics":
        rows = result.get("rows_analyzed")
        summary = result.get("summary") or []
        highest = result.get("highest_risk_group") or {}
        if highest:
            label = get_label_from_row(highest)
            click_rate = compact_percent(highest.get("click_rate_percent") or highest.get("click_rate"))
            report_rate = compact_percent(highest.get("report_rate_percent") or highest.get("report_rate"))
            return f"Analysis completed on {rows} rows. Highest-risk group is {label} with click rate {click_rate} and report rate {report_rate}."
        if summary:
            top = summary[0]
            label = get_label_from_row(top)
            click_rate = compact_percent(top.get("click_rate_percent") or top.get("click_rate"))
            report_rate = compact_percent(top.get("report_rate_percent") or top.get("report_rate"))
            return f"Analysis completed on {rows} rows. Top result is {label} with click rate {click_rate} and report rate {report_rate}."
        return f"Analysis completed on {rows} rows."
        
    if tool_name == "employee_lookup":
        mode = result.get("mode")
        count = result.get("count")
        return f"Employee lookup completed. Found {count} result(s) in mode '{mode}'."
        
    if tool_name == "predict_risk":
        if "high_risk_count" in result:
            return f"Found {result.get('high_risk_count')} high-risk users out of {result.get('population_size', 'the population')} ."

        prediction = result.get("prediction")
        probabilities = result.get("probabilities_labeled") or {}
        if prediction and isinstance(probabilities, dict):
            clicked = probabilities.get("Clicked Link", 0)
            reported = probabilities.get("Reported", 0)
            no_action = probabilities.get("No Action", 0)
            return f"Prediction completed: {prediction}. Clicked: {compact_percent(clicked)}, Reported: {compact_percent(reported)}, No Action: {compact_percent(no_action)}."
        return "Prediction completed."
        
    if tool_name in ["recommend_actions", "recommendations", "get_recommendations"]:
        actions = result.get("recommended_actions") or result.get("recommendations_actions") or []
        if actions:
            first_action = actions[0].get("action") if isinstance(actions[0], dict) else actions[0]
            return f"Recommendations generated. Suggested action: {first_action}."
        return "Recommendations generated."
        
    if tool_name == "simulation_users":
        campaign = result.get("campaignname") or "selected campaign"
        event_type = result.get("event_type") or "selected event"
        count = result.get("user_count", 0)
        return f"Found {count} user(s) for event '{event_type}' in campaign '{campaign}'."
        
    if tool_name == "cache_control":
        action = result.get("action")
        metadata = result.get("cache_metadata") or {}
        rows = metadata.get("total_rows")
        if action == "clear":
            return "Cache cleared successfully."
        return f"Cache {action} completed successfully. Cached rows: {rows}."
        
    if tool_name == "cache_status":
        metadata = result.get("cache_metadata") or {}
        loaded = metadata.get("loaded")
        rows = metadata.get("total_rows")
        profiles = metadata.get("employee_profiles")
        refresh_time = metadata.get("last_refresh_time")
        if loaded:
            return f"Cache is loaded with {rows} rows and {profiles} employee profiles. Last refresh: {refresh_time}."
        return "Cache is not currently loaded."
        
    if tool_name == "system_info":
        mode = result.get("mode", "system")
        status_value = result.get("status", "unknown")
        if mode in ["health", "health_check", "status"]:
            return f"System health check completed. Current status: {status_value}."
        if mode in ["schema", "table_schema", "columns"]:
            return f"Schema lookup completed. Found {result.get('column_count', 0)} column(s)."
        if mode in ["features", "model_features"]:
            return f"Feature check completed. Found {result.get('feature_column_count', 0)} model feature(s)."
        return "System information retrieved."
        
    return "Request completed successfully."

def build_ui_highlights(tool_name, result):
    if not isinstance(result, dict):
        return []
    highlights = []
    for key in ["rows_analyzed", "population_size", "high_risk_count", "high_risk_percentage", "user_count", "count", "total_rows", "employee_profiles", "last_refresh_time", "threshold_used", "threshold"]:
        if result.get(key) not in [None, "", [], {}]:
            highlights.append({"label": key, "value": result.get(key)})
            
    if tool_name == "predict_risk":
        probabilities = result.get("probabilities_labeled")
        if isinstance(probabilities, dict):
            try:
                for k, v in probabilities.items():
                    highlights.append({"label": f"{k} probability", "value": f"{round(float(v) * 100, 2)}%"})
            except Exception:
                pass
        elif isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], dict):
            try:
                for k, v in probabilities[0].items():
                    highlights.append({"label": f"{k} probability", "value": f"{round(float(v) * 100, 2)}%"})
            except Exception:
                pass
    return clean_json_value(highlights[:8])

def log_tool_event(tool_name: str, stage: str, message: str, payload: dict = None) -> None:
    try:
        # Emit standardized terminal log line for orchestrator ingestion
        terminal_msg = f"TERMINAL_AWS_LOG | Location: phishing_mcp_server.py | Tool: {tool_name} | Stage: {stage} | Message: {message}"
        LOGGER.info(terminal_msg)
        if payload is not None:
            payload_text = json.dumps(clean_json_value(payload), ensure_ascii=False, default=str)
            LOGGER.debug(f"TOOL_EVENT_PAYLOAD | {tool_name} | {payload_text[:4000]}")
    except Exception:
        pass

def attach_ui_response(tool_name, result):
    if not isinstance(result, dict):
        return result
    meta = TOOL_REGISTRY.get(tool_name, {})
    result["ui_summary"] = build_ui_summary(tool_name, result)
    result["ui_highlights"] = build_ui_highlights(tool_name, result)
    result["ui_response_type"] = "deterministic_mcp_summary"
    result["tool_catalog_entry"] = {
        "tool_name": tool_name,
        "description": meta.get("description", ""),
        "strict_guidance": meta.get("strict_guidance", ""),
        "allowed_args": meta.get("allowed_args", []),
    }
    result["ui_instruction"] = meta.get("ui_instruction", "Render the tool output in the UI and keep the JSON trace available in the debug panel.")
    result["trace_instruction"] = "Show the full JSON execution trace in the UI trace panel and preserve the tool output in human-readable form."
    result["execution_trace"] = [{"step": "mcp_tool_completed", "tool_name": tool_name, "status": result.get("status", "unknown")}]
    result["llm_prompt_hint"] = "When summarizing this tool response, keep the human-readable UI output and the JSON execution trace aligned with the backend result."
    return clean_json_value(result)

@mcp.tool()
@log_execution
def run_analytics(analysis_type: str, group_by: list = None, filters: dict = None, user_role: str = "user", top_n: int = 20):
    """
    Historical aggregate phishing analytics: counts, percentages, rates, trends, grouped summaries.
    Use for city, department, campaign, template, subject, month, year, designation, grade, COO area, business area, or overall metrics.
    """
    log_tool_event("run_analytics", "start", "Starting historical analytics tool", {"analysis_type": analysis_type, "group_by": group_by, "filters": filters, "top_n": top_n, "user_role": user_role})
    try:
        analysis_warning = None
        if analysis_type not in ANALYSIS_REGISTRY:
            analysis_warning = f"Unknown analysis_type '{analysis_type}'. Generic analyzer was used."
        result = run_analysis(
            df=get_analytics_dataframe(),
            analysis_type=analysis_type,
            group_by=group_by or [],
            filters=filters or {},
            user_role=user_role,
            top_n=max(1, safe_int(top_n, 20)),
            metadata=cache_metadata_for_analysis(),
        )
        result["llm_context"] = analytics_llm_context()
        if analysis_warning:
            result["analysis_warning"] = analysis_warning
        log_tool_event("run_analytics", "success", "Historical analytics completed", {"rows": result.get("rows_analyzed"), "summary_len": len(result.get("summary", []) or [])})
        return attach_ui_response("run_analytics", result)
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("run_analytics", "error", str(e), {"analysis_type": analysis_type})
        return attach_ui_response("run_analytics", {"status": "error", "message": str(e), "llm_context": analytics_llm_context()})

@mcp.tool()
@log_execution
def employee_lookup(mode: str = "profile", brid: str = None, city: str = None, department: str = None, limit: int = 10, user_role: str = "user"):
    """
    Employee historical lookup. Modes:
    - profile: profile for a specific BRID
    - top_risky: top historically risky employees
    - find: find/filter employees by city or department
    """
    log_tool_event("employee_lookup", "start", "Starting employee lookup tool", {"mode": mode, "brid": brid, "city": city, "department": department, "limit": limit, "user_role": user_role})
    try:
        mode = str(mode or "profile").strip().lower()
        profile_df = get_profile_dataframe()
        
        if mode in ["profile", "employee_profile", "brid_profile", "lookup"]:
            if not brid:
                return {
                    "status": "error",
                    "message": "brid is required for employee profile lookup.",
                    "missing_fields": ["brid"],
                    "llm_context": employee_llm_context(user_role),
                }
            profile = get_employee_profile(profile_df, brid, user_role=user_role)
            if not profile:
                return {
                    "status": "error",
                    "message": "BRID not found",
                    "brid": brid if is_admin(user_role) else "hidden_user_mode",
                    "llm_context": employee_llm_context(user_role),
                }
            return clean_json_value({"status": "success", "profile": profile, "llm_context": employee_llm_context(user_role)})
            
        if mode in ["top_risky", "risky", "high_risk_historical", "historical_risky_users"]:
            records = get_top_risky_employees(profile_df, limit=max(1, min(safe_int(limit, 10), 100)), user_role=user_role)
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "count": len(records),
                "employees": sanitize_records(records, user_role),
                "llm_context": employee_llm_context(user_role),
            })
            
        if mode in ["find", "search", "filter"]:
            if profile_df.empty:
                return {"status": "success", "mode": mode, "count": 0, "employees": [], "llm_context": employee_llm_context(user_role)}
            result = profile_df.copy()
            if city and "city" in result.columns:
                result = result[result["city"].fillna("").astype(str).str.lower().str.contains(str(city).lower(), regex=False, na=False)]
            if department and "department" in result.columns:
                result = result[result["department"].fillna("").astype(str).str.lower().str.contains(str(department).lower(), regex=False, na=False)]
            
            records = result.sort_values(["risk_score", "clicked_count", "total_events"], ascending=[False, False, False]).head(max(1, min(safe_int(limit, 10), 100))).to_dict("records")
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "count": len(records),
                "employees": sanitize_records(records, user_role),
                "filters": {"city": city, "department": department},
                "llm_context": employee_llm_context(user_role),
            })
            
        return {
            "status": "error",
            "message": f"Unsupported employee_lookup mode: {mode}.",
            "supported_modes": ["profile", "top_risky", "find"],
            "llm_context": employee_llm_context(user_role),
        }
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("employee_lookup", "error", str(e), {"mode": mode, "brid": brid})
        return attach_ui_response("employee_lookup", {"status": "error", "message": str(e), "llm_context": employee_llm_context(user_role)})

@mcp.tool()
@log_execution
def predict_risk(mode: str = "by_brid", brid: str = None, payload: dict = None, limit: int = None, user_role: str = "user"):
    """
    All ML prediction flows. Modes:
    - by_brid: predict using latest available row for BRID
    - from_payload: predict from natural-language extracted/manual fields
    - recent_population: prediction distribution for recent cached population
    - high_risk_population: high risk probability threshold
    """
    log_tool_event("predict_risk", "start", "Starting prediction tool", {"mode": mode, "brid": brid, "payload_keys": list((payload or {}).keys()) if isinstance(payload, dict) else None, "limit": limit, "user_role": user_role})
    try:
        mode = str(mode or "by_brid").strip().lower()
        if mode in ["by_brid", "brid", "user", "predict_user"]:
            return run_prediction_by_brid(brid, user_role=user_role)
        if mode in ["from_payload", "payload", "input", "json", "predict_payload"]:
            return run_prediction_by_payload(payload, user_role=user_role)
        if mode in ["recent_population", "recent", "cache"]:
            return run_high_risk_population_prediction(limit=GLOBAL_ANALYTICS_ROWS, user_role=user_role)
        if mode in ["high_risk_population", "high_risk", "high_risk_users", "predicted_high_risk"]:
            eff_limit = limit if limit is not None else DEFAULT_POPULATION_LIMIT
            return run_high_risk_population_prediction(limit=eff_limit, user_role=user_role)
            
        return {
            "status": "error",
            "message": f"Unsupported predict_risk mode: {mode}",
            "supported_modes": ["by_brid", "from_payload", "recent_population", "high_risk_population"],
            "llm_context": prediction_llm_context(),
        }
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("predict_risk", "error", str(e), {"mode": mode, "brid": brid})
        return attach_ui_response("predict_risk", {"status": "error", "message": str(e), "llm_context": prediction_llm_context()})

@mcp.tool()
@log_execution
def recommend_actions(mode: str = "employee_improvement", brid: str = None, group_by: list = None, filters: dict = None, top_n: int = 5, user_role: str = "user"):
    """
    Improvement guidance, training recommendations, and risk-reduction actions. Modes:
    - employee_improvement: individual employee recommendations
    - group_recommendations: department/city/group recommendations
    - overall_recommendations: top level global recommendations
    """
    log_tool_event("recommend_actions", "start", "Starting recommendation tool", {"mode": mode, "brid": brid, "group_by": group_by, "filters": filters, "top_n": top_n, "user_role": user_role})
    try:
        mode = str(mode or "employee_improvement").strip().lower()
        if mode in ["employee_improvement", "employee", "user", "brid"]:
            if not brid:
                return {"status": "error", "message": "brid is required for employee_improvement recommendations.", "missing_fields": ["brid"], "llm_context": improvement_llm_context()}
            result = employee_lookup(mode="profile", brid=brid, user_role=user_role)
            if result.get("status") != "success":
                return result
            profile = result.get("profile") or {}
            recommendations = profile.get("improvement_guidance", []) or []
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "employee_profile": profile,
                "recommendations": recommendations,
                "recommended_actions": recommendations,
                "recommendations_actions": recommendations,
                "training_recommendations": [],
                "llm_context": improvement_llm_context(),
            })

        if mode in ["group_recommendations", "group", "department", "city"]:
            analysis_type = "overall" if not group_by else ("department" if "department" in group_by or "usertags-Department" in group_by else "city")
            result = run_analytics(analysis_type=analysis_type, group_by=group_by, filters=filters, top_n=top_n, user_role=user_role)
            if result.get("status") != "success":
                return result
            actions = result.get("recommended_actions", []) or result.get("recommendations_actions", []) or []
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "analysis_type": analysis_type,
                "group_by": group_by,
                "filters": filters,
                "summary": result.get("summary", []),
                "highest_risk_group": result.get("highest_risk_group", {}),
                "highest_reporting_group": result.get("highest_reporting_group", {}),
                "highest_no_action_rules": result.get("highest_no_action_rules", {}),
                "top_highest_risk_groups": result.get("top_highest_risk_groups", []),
                "recommended_actions": actions,
                "recommendations_actions": actions,
                "training_recommendations": result.get("training_recommendations", []),
                "risk_analysis": result.get("risk_analysis", {}),
                "llm_context": improvement_llm_context(),
            })

        if mode in ["overall_recommendations", "overall", "general"]:
            result = run_analytics(analysis_type="overall", group_by=[], filters=filters, top_n=5, user_role=user_role)
            if result.get("status") != "success":
                return result
            actions = result.get("recommended_actions", []) or result.get("recommendations_actions", []) or []
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "summary": result.get("summary", []),
                "highest_risk_group": result.get("highest_risk_group", {}),
                "recommended_actions": actions,
                "recommendations_actions": actions,
                "training_recommendations": result.get("training_recommendations", []),
                "risk_analysis": result.get("risk_analysis", {}),
                "llm_context": improvement_llm_context(),
            })

        return {
            "status": "error",
            "message": f"Unsupported recommend_actions mode: {mode}",
            "supported_modes": ["employee_improvement", "group_recommendations", "overall_recommendations"],
            "llm_context": improvement_llm_context(),
        }
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("recommend_actions", "error", str(e), {"mode": mode, "brid": brid})
        return attach_ui_response("recommend_actions", {"status": "error", "message": str(e), "llm_context": improvement_llm_context()})

def get_recommendations(*args, **kwargs):
    return recommend_actions(*args, **kwargs)

@mcp.tool()
@log_execution
def simulation_users(campaign_month: str = None, campaign_year: str = None, campaign_name: str = None, event_type: str = "clicked link", user_role: str = "user", limit: int = 5000):
    """
    Users who clicked, reported, or took no action in a simulation/campaign.
    Use for simulation/campaign user lists by month, year, campaign name, and event type.
    """
    log_tool_event("simulation_users", "start", "Starting simulation user list tool", {"campaign_month": campaign_month, "campaign_year": campaign_year, "campaign_name": campaign_name, "event_type": event_type, "limit": limit, "user_role": user_role})
    try:
        limit = max(1, min(safe_int(limit, 5000), 10000))
        search = build_campaign_search_filters(campaign_month=campaign_month, campaign_year=campaign_year, campaign_name=campaign_name)
        search_type = search.get("search_type")
        
        if search_type == "unknown":
            return {"status": "error", "message": "Please specify campaign name, campaign month, campaign year, or both month and year.", "llm_context": simulation_llm_context(user_role, search_type)}
            
        df = get_analytics_dataframe()
        if df.empty:
            return {"status": "error", "message": "Analytics dataset unavailable.", "llm_context": simulation_llm_context(user_role, search_type)}

        campaign_col = df["campaignname"].fillna("").astype(str).str.lower()
        if search_type == "campaign_name":
            pattern = str(campaign_name or "").strip().lower()
            mask = campaign_col.str.contains(pattern, regex=False)
        elif search_type == "month_year":
            month_pattern = str(search.get("month_pattern", "")).lower()
            year_pattern = str(search.get("year_pattern", "")).lower()
            mask = campaign_col.str.contains(month_pattern, regex=False) & campaign_col.str.contains(year_pattern, regex=False)
        elif search_type == "month_only":
            month_pattern = str(search.get("month_pattern", "")).lower()
            mask = campaign_col.str.contains(month_pattern, regex=False)
        elif search_type == "year_only":
            year_pattern = str(search.get("year_pattern", "")).lower()
            mask = campaign_col.str.contains(year_pattern, regex=False)
        else:
            mask = pd.Series([False] * len(df), index=df.index)

        campaign_df = df[mask].copy()
        if campaign_df.empty:
            return {
                "status": "error",
                "message": "No campaign found matching the supplied filters.",
                "campaign_name": campaign_name, "campaign_month": campaign_month, "campaign_year": campaign_year,
                "search_type": search_type, "llm_context": simulation_llm_context(user_role, search_type),
            }

        selected_campaign = campaign_df.sort_values("senttimestamp", ascending=False).iloc[0]["campaignname"]
        event_mask = df["eventtype"].fillna("").astype(str).str.lower() == str(event_type or "").strip().lower()
        users_df = df[(df["campaignname"] == selected_campaign) & event_mask].sort_values("senttimestamp", ascending=False).head(limit)
        
        if users_df.empty:
            return clean_json_value({
                "status": "success", "campaignname": selected_campaign, "campaign_year": campaign_year, "campaign_month": campaign_month,
                "event_type": event_type, "user_count": 0, "brid_ids": [], "total_unique_brids": 0, "users": [],
                "pii_exposed": is_admin(user_role), "llm_context": simulation_llm_context(user_role, search_type),
            })
            
        brid_cols = [c for c in ["proofpoint_brid", "brid", "usertags-BRID", BRID_COLUMN] if c in users_df.columns]
        total_unique_brids = []
        if brid_cols:
            total_unique_brids = users_df[brid_cols[0]].dropna().unique().tolist()
            
        brid_ids = total_unique_brids if is_admin(user_role) else ["hidden_user_mode"]
        
        output_cols = ["campaignname", "eventtype", "senttimestamp", "city", "usertags-Department", "department", "businessarea1"] + brid_cols
        if is_admin(user_role):
            output_cols += ["userfirstname", "userlastname"]
            
        output_cols = [c for c in output_cols if c in users_df.columns]
        cleaned = users_df[output_cols].copy()
        
        for col in cleaned.columns:
            if "datetime" in str(cleaned[col].dtype):
                cleaned[col] = cleaned[col].astype(str)
        cleaned = cleaned.where(pd.notna(cleaned), None)
        
        return clean_json_value({
            "status": "success",
            "campaignname": selected_campaign,
            "campaign_month": campaign_month,
            "campaign_year": campaign_year,
            "search_type": search_type,
            "event_type": event_type,
            "user_count": int(len(users_df)),
            "total_unique_brids": len(total_unique_brids),
            "brid_ids": brid_ids,
            "users": cleaned.to_dict("records"),
            "pii_exposed": is_admin(user_role),
            "llm_context": simulation_llm_context(user_role, search_type),
        })
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("simulation_users", "error", str(e), {"campaign_name": campaign_name, "event_type": event_type})
        return {"status": "error", "message": str(e), "llm_context": simulation_llm_context(user_role)}

@mcp.tool()
@log_execution
def cache_control(action: str = "refresh"):
    """
    Cache management control. Actions: refresh, clear
    """
    log_tool_event("cache_control", "start", "Starting cache control tool", {"action": action})
    try:
        if action in ["clear", "reset", "delete"]:
            metadata = clear_cache_internal()
            response = clean_json_value({
                "status": "success" if not metadata.get("loaded") else "error",
                "action": "clear",
                "cache_metadata": metadata,
                "llm_context": cache_llm_context(),
            })
            log_tool_event("cache_control", "success", "Cache cleared", metadata)
            return attach_ui_response("cache_control", response)
        metadata = refresh_cache_internal()
        response = clean_json_value({
            "status": "success" if metadata.get("loaded") else "error",
            "action": "refresh",
            "cache_metadata": metadata,
            "llm_context": cache_llm_context(),
        })
        log_tool_event("cache_control", "success", "Cache refreshed", metadata)
        return attach_ui_response("cache_control", response)
    except Exception as e:
        LOGGER.exception(str(e))
        log_tool_event("cache_control", "error", str(e), {"action": action})
        return attach_ui_response("cache_control", {"status": "error", "message": str(e), "llm_context": cache_llm_context()})

@mcp.tool()
@log_execution
def cache_status(include_statistics: bool = True):
    """
    Read cache metrics, metadata and statistics.
    """
    log_tool_event("cache_status", "start", "Starting cache status tool", {"include_statistics": include_statistics})
    try:
        ensure_cache()
    except Exception as e:
        LOGGER.exception(f"CACHE_STATUS_ENSURE_FAILED | {str(e)}")
        log_tool_event("cache_status", "error", str(e), {"include_statistics": include_statistics})
        
    stats = get_cache_stats() or {}
    if include_statistics and stats.get("loaded"):
        payload = get_cache_statistics_payload()
        response = clean_json_value({"status": "success", "cache_metadata": {**stats, **payload}, "llm_context": cache_llm_context()})
        log_tool_event("cache_status", "success", "Cache status returned", response.get("cache_metadata", {}))
        return attach_ui_response("cache_status", response)
    response = clean_json_value({"status": "success", "cache_metadata": stats, "llm_context": cache_llm_context()})
    log_tool_event("cache_status", "success", "Cache status returned", response.get("cache_metadata", {}))
    return attach_ui_response("cache_status", response)

@mcp.tool()
@log_execution
def system_info(mode: str = "health"):
    """
    System diagnostics and health checking. Modes: health, schema, features, environment
    """
    try:
        mode = str(mode or "health").strip().lower()
        if mode in ["health", "health_check", "status"]:
            env = validate_environment()
            db_connected = False
            model_available = bool(MODEL_PATH and os.path.exists(MODEL_PATH))
            feature_columns_available = bool(FEATURE_COLUMNS_PATH and os.path.exists(FEATURE_COLUMNS_PATH))
            feature_count = 0
            
            try:
                if feature_columns_available:
                    feature_count = len(load_feature_columns())
            except Exception as e:
                LOGGER.exception(f"FEATURE_COLUMNS_HEALTH_FAILED | {str(e)}")
                
            try:
                with get_sql_engine().connect() as conn:
                    conn.exec_driver_sql("SELECT 1")
                    db_connected = True
            except Exception as e:
                LOGGER.exception(f"DB_HEALTH_FAILED | {str(e)}")
                
            status = "healthy" if env.get("valid") and db_connected and feature_count > 0 and model_available else "degraded"
            cache_stats = get_cache_stats() or {}
            return clean_json_value({
                "status": status,
                "mode": mode,
                "environment": env,
                "model_loaded": MODEL is not None,
                "model_available": model_available,
                "db_connected": db_connected,
                "cache_metadata": cache_stats,
                "feature_columns_available": feature_columns_available,
                "feature_column_count": feature_count,
                "system_llm_context": system_llm_context("system_info"),
            })
            
        if mode in ["schema", "table_schema", "columns"]:
            df = execute_query(f"SELECT TOP 1 * FROM {get_db_table()}")
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "columns": list(df.columns),
                "column_count": int(len(df.columns)),
                "llm_context": system_llm_context("table_schema"),
            })
            
        if mode in ["features", "feature_columns", "model_features"]:
            columns = load_feature_columns()
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "feature_columns": columns,
                "feature_column_count": len(columns),
                "llm_context": system_llm_context("model_features"),
            })
            
        if mode in ["environment", "env", "config"]:
            cache_stats = get_cache_stats() or {}
            return clean_json_value({
                "status": "success",
                "mode": mode,
                "environment": validate_environment(),
                "cache_metadata": cache_stats,
                "enable_cache": ENABLE_CACHE,
                "cache_auto_refresh_minutes": CACHE_AUTO_REFRESH_MINUTES,
                "global_analytics_rows": GLOBAL_ANALYTICS_ROWS,
                "default_population_limit": DEFAULT_POPULATION_LIMIT,
                "high_risk_threshold": HIGH_RISK_THRESHOLD,
                "brid_column": BRID_COLUMN,
                "llm_context": system_llm_context("environment"),
            })
            
        return {"status": "error", "message": f"Unsupported system_info mode: {mode}", "supported_modes": ["health", "schema", "features", "environment"], "llm_context": system_llm_context("system_info")}
    except Exception as e:
        LOGGER.exception(str(e))
        return {"status": "error", "message": str(e), "llm_context": system_llm_context("system_info")}

def print_json(title, data, max_chars=1500):
    print(f"\n--- {title} ---")
    try:
        value = json.dumps(data, indent=2, default=str)
        print(value[:max_chars])
        if len(value) > max_chars:
            print(f"\n... output truncated ...")
    except Exception as e:
        print(f"FAILED_TO_PRINT_JSON | {str(e)}")

def get_test_dataframe_and_brid():
    df = get_analytics_dataframe()
    test_brid = "c9cc69cb-3"
    if df is not None and not df.empty and "usertags-BRID" in df.columns:
        valid_brids = df["usertags-BRID"].dropna().astype(str)
        valid_brids = valid_brids[valid_brids.str.strip() != ""]
        if not valid_brids.empty:
            test_brid = valid_brids.iloc[0]
    return df, test_brid

def run_server_tests_core():
    print("\n=== Phishing MCP SERVER CORE TEST MODE ===")
    print_json("SYSTEM HEALTH", system_info("health"), max_chars=1500)
    print_json("SYSTEM SCHEMA", system_info("schema"), max_chars=1500)
    print_json("CACHE REFRESH", cache_control("refresh"), max_chars=1500)
    print_json("CACHE STATUS", cache_status(True), max_chars=1500)
    
    df, test_brid = get_test_dataframe_and_brid()
    print(f"\nAnalytics Shape: {df.shape}")
    if df.empty:
        print("\nNO DATA FOUND. STOPPING CORE TESTS.")
        return
        
    analytics_test_cases = [
        {"title": "OVERALL ANALYSIS", "analysis_type": "overall_analysis", "group_by": [], "filters": {}},
        {"title": "CITY PERFORMANCE", "analysis_type": "city_performance", "group_by": ["city"], "filters": {}},
        {"title": "DEPARTMENT PERFORMANCE", "analysis_type": "department_performance", "group_by": ["department"], "filters": {}},
        {"title": "BRID PERFORMANCE ADMIN", "analysis_type": "brid_performance", "group_by": ["brid"], "filters": {}, "user_role": "admin"},
        {"title": "BRID PERFORMANCE USER PRIVACY", "analysis_type": "brid_performance", "group_by": ["brid"], "filters": {}, "user_role": "user"},
        {"title": "CAMPAIGN PERFORMANCE", "analysis_type": "campaign_performance", "group_by": ["campaign"], "filters": {}},
        {"title": "TEMPLATE PERFORMANCE", "analysis_type": "template_performance", "group_by": ["template"], "filters": {}},
        {"title": "SUBJECT PERFORMANCE", "analysis_type": "subject_performance", "group_by": ["subject"], "filters": {}},
        {"title": "MONTHLY TREND", "analysis_type": "monthly_trend", "group_by": ["month"], "filters": {}},
        {"title": "YEARLY TREND", "analysis_type": "yearly_trend", "group_by": ["year"], "filters": {}},
        {"title": "JANUARY 2026 FILTER", "analysis_type": "month_year_trend", "group_by": ["month_year"], "filters": {"month": "January"}},
    ]
    
    for case in analytics_test_cases:
        result = run_analytics(analysis_type=case["analysis_type"], group_by=case["group_by"], filters=case["filters"], user_role=case.get("user_role", "admin"), top_n=10)
        print_json(case["title"], result, max_chars=1200)
        
    print("\n=== EMPLOYEE LOOKUP TESTS ===")
    print_json(f"EMPLOYEE LOOKUP TEST (using Test BRID: {test_brid})", employee_lookup(mode="profile", brid=test_brid, user_role="admin"), max_chars=1200)
    print_json("EMPLOYEE PROFILE USER", employee_lookup(mode="profile", brid=test_brid, user_role="user"), max_chars=1200)
    print_json("TOP RISKY EMPLOYEES ADMIN", employee_lookup(mode="top_risky", limit=10, user_role="admin"), max_chars=1200)
    print_json("TOP RISKY EMPLOYEES USER", employee_lookup(mode="top_risky", limit=10, user_role="user"), max_chars=1200)
    print_json("FIND EMPLOYEES PUNE ADMIN", employee_lookup(mode="find", city="pune", limit=10, user_role="admin"), max_chars=1200)
    
    print("\n=== RECOMMENDATION TESTS ===")
    print_json("EMPLOYEE IMPROVEMENT ADMIN", get_recommendations(mode="employee_improvement", brid=test_brid, user_role="admin"), max_chars=1500)
    print_json("EMPLOYEE IMPROVEMENT USER", get_recommendations(mode="employee_improvement", brid=test_brid, user_role="user"), max_chars=1500)
    print_json("GROUP RECOMMENDATIONS", get_recommendations(mode="group_recommendations", group_by=["department"], user_role="user"), max_chars=1500)
    print("\n--- CORE TEST MODE COMPLETE ---")

def run_server_tests_prediction():
    print("\n=== Phishing MCP SERVER PREDICTION TEST MODE ===")
    print_json("CACHE STATUS", cache_status(True), max_chars=1000)
    
    df, test_brid = get_test_dataframe_and_brid()
    print(f"\nAnalytics Shape: {df.shape}")
    if df.empty:
        print("\nNO DATA FOUND. STOPPING PREDICTION TESTS.")
        return
        
    print_json(f"USING TEST BRID: {test_brid}", {"test_brid": test_brid}, max_chars=500)
    print_json("PREDICT BY BRID USER", predict_risk(mode="by_brid", brid=test_brid, user_role="user"), max_chars=2500)
    print_json("PREDICT BY BRID ADMIN", predict_risk(mode="by_brid", brid=test_brid, user_role="admin"), max_chars=2500)
    
    sample_payload = {
        "city": "pune",
        "corporate_grade": "BA4",
        "usertags-Department": "technology",
        "businessarea1": "security",
        "templatename": "Payroll Update",
        "templatesubject": "Payroll Adjustment Notice",
        "eventtype": "No Action",
        "senttimestamp": "2026-01-12 09:00:00",
    }
    
    print_json("PREDICT FROM PAYLOAD", predict_risk(mode="from_payload", user_role="user", payload=sample_payload), max_chars=2500)
    print_json("PREDICT RECENT POPULATION", predict_risk(mode="recent_population", user_role="user"), max_chars=1800)
    print_json("PREDICT HIGH RISK POPULATION USER", predict_risk(mode="high_risk_population", limit=500, user_role="user"), max_chars=1800)
    print_json("PREDICT HIGH RISK POPULATION ADMIN", predict_risk(mode="high_risk_population", limit=500, user_role="admin"), max_chars=1800)
    print_json("SIMULATION USERS JAN2026 ADMIN", simulation_users(campaign_month="January", event_type="clicked link", user_role="admin", limit=100), max_chars=1500)
    print_json("SIMULATION USERS JAN2026 USER", simulation_users(campaign_month="January", event_type="clicked link", user_role="user", limit=100), max_chars=1500)
    print_json("SIMULATION USERS JANUARY CAMPAIGN_NAME", simulation_users(campaign_name="January", event_type="clicked link", user_role="user", limit=100), max_chars=1500)
    print("\n-- PREDICTION TEST MODE COMPLETE --")

if __name__ == "__main__":
    if "--test-all" in sys.argv or "--test" in sys.argv:
        run_server_tests_core()
        run_server_tests_prediction()
    elif "--test-core" in sys.argv:
        run_server_tests_core()
    elif "--test-prediction" in sys.argv:
        run_server_tests_prediction()
    else:
        if ENABLE_CACHE:
            try:
                refresh_cache_internal()
            except Exception as e:
                LOGGER.exception(f"INITIAL_CACHE_BUILD_FAILED | {str(e)}")
        mcp.run()