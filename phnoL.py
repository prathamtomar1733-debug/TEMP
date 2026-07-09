# phishing_streamlit_ui_noLLM.py
# Deterministic Rendering: Bypasses LLM natural language summarization entirely to display the backend's raw tool outputs exactly as they are returned.

# Dynamic Data Translation: Automatically parses varying backend data structures (like dictionaries, lists, and primitives) directly into native UI components (tables, metrics, JSON blocks) using hardcoded logic.

# Streamlined Interaction: Focuses purely on executing analytical tasks and returning immediate, structured factual data without narrative interpretation.

import logging
import os
import re
import json
import time
import math
import html
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("phishing_streamlit_ui_noLLM")

EXECUTION_MODE = os.getenv("MCP_EXECUTION_MODE", "direct").strip().lower()
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python")
MCP_SERVER_SCRIPT = os.getenv("MCP_SERVER_SCRIPT", "phishing_mcp_server.py")
DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "user")

APP_TITLE = "Phishing Simulation Analytics Copilot - No LLM"
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "20"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "10"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.60"))

st.set_page_config(page_title=APP_TITLE, page_icon="🎣", layout="wide", initial_sidebar_state="expanded")

MONTHS = {
    "jan": "January", "january": "January", "01": "January",
    "feb": "February", "february": "February", "02": "February",
    "mar": "March", "march": "March", "03": "March",
    "apr": "April", "april": "April", "04": "April",
    "may": "May", "jun": "June", "june": "June", "06": "June",
    "jul": "July", "july": "July", "07": "July",
    "aug": "August", "august": "August", "08": "August",
    "sep": "September", "september": "September", "09": "September",
    "oct": "October", "october": "October", "10": "October",
    "nov": "November", "november": "November", "11": "November",
    "dec": "December", "december": "December", "12": "December"
}

TOOL_LABELS = {
    "run_analytics": "Historical analytics",
    "employee_lookup": "Employee lookup",
    "predict_risk": "ML prediction",
    "recommend_actions": "Recommendations",
    "simulation_users": "Simulation users",
    "cache_control": "Cache control",
    "cache_status": "Cache status",
    "system_info": "System info"
}

TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "run_analytics": {
        "description": "Historical aggregate phishing analytics counts percentages rates trends grouped summaries city department campaign template subject month year designation grade overall metrics.",
        "examples": [
            "percentage clicked January", "city with most clickers",
            "department wise click rate", "campaign performance", "monthly trend"
        ],
        "keywords": [
            "percentage", "percent", "count", "counts", "rate", "trend", "analytics",
            "overall", "city", "department", "campaign", "template", "subject",
            "month", "year", "designation", "grade", "overall metrics"
        ]
    },
    "employee_lookup": {
        "description": "Employee historical lookup profile find employees top historically risky employees not prediction.",
        "examples": [
            "show profile for BRID", "find employees in Pune",
            "top risky employees", "high risk user historical"
        ],
        "keywords": [
            "profile", "employee", "employees", "brid", "find", "lookup",
            "top risky", "historically risky", "high risk user", "risky employees", "history"
        ]
    },
    "predict_risk": {
        "description": "Machine learning prediction probability likelihood forecast by BRID manual fields population predicted high risk users.",
        "examples": [
            "predict no action probability for BRID",
            "predict for city Pune department cyber subject password reset",
            "predicted high risk users"
        ],
        "keywords": [
            "predict", "prediction", "probability", "likely", "chance", "forecast",
            "ml", "model", "no action probability", "clicked probability",
            "predicted high risk", "predicted", "high risk population"
        ]
    },
    "recommend_actions": {
        "description": "Improvement guidance training suggestions risk reduction actions recommendations for employee group or overall.",
        "examples": [
            "how can this BRID improve", "recommend training for risky department",
            "what should we do to reduce phishing clicks"
        ],
        "keywords": [
            "improve", "improvement", "recommend", "recommendation", "training",
            "actions", "reduce", "guidance", "what should", "action plan", "coach"
        ]
    },
    "simulation_users": {
        "description": "Users who clicked reported or took no action in a simulation campaign user list by month year campaign event type.",
        "examples": [
            "who clicked in March 2026 simulation", "show trapped users",
            "users reported in campaign"
        ],
        "keywords": [
            "users who", "who clicked", "who reported", "trapped", "simulation users",
            "campaign users", "list users", "clicked users", "reported users"
        ]
    },
    "cache_control": {
        "description": "Cache changing actions refresh clear reload update rebuild reset cache.",
        "examples": [
            "refresh cache", "clear cache", "new data added update cache"
        ],
        "keywords": [
            "refresh cache", "clear cache", "reload cache", "update cache",
            "rebuild cache", "reset cache", "new data added"
        ]
    },
    "cache_status": {
        "description": "Read cache status statistics loaded refresh time rows campaigns templates departments cities.",
        "examples": [
            "cache status", "cache statistics", "cache stats", "cache loaded",
            "refresh time", "cache rows", "rows loaded"
        ],
        "keywords": [
            "cache status", "cache statistics", "when was cache refreshed", "cache status"
        ]
    },
    "system_info": {
        "description": "System diagnostics health check database schema columns model features environment configuration.",
        "examples": [
            "health check", "show table schema", "feature columns", "environment config"
        ],
        "keywords": [
            "health check", "health", "system info", "schema", "columns", "feature",
            "features", "environment", "config", "configuration", "diagnostic", "system", "model path"
        ]
    }
}

EXAMPLE_QUESTIONS = [
    "percentage of people clicked on January phishing campaign",
    "city with most clickers",
    "department wise click rate in 2026",
    "top risky employees",
    "show profile for BRID 75b4ae75-b",
    "predict no action probability for BRID 75b4ae75-b",
    "predict for city Pune department cyber subject password reset",
    "predicted high risk users",
    "how can BRID 75b4ae75-b improve",
    "who clicked in March 2026 simulation",
    "cache status",
    "refresh cache",
    "health check",
    "show table schema"
]

st.markdown("""
<style>
:root {
    --bg-main: #08111f;
    --bg-sidebar: #07101d;
    --bg-card: #0f1b2d;
    --bg-card-2: #111f34;
    --border: #253449;
    --border-soft: #1e2b3d;
    --text: #a5edf7;
    --muted: #9caec4;
    --accent: #2dd4bf;
    --blue: #60a5fa;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}
.stApp {
    background: linear-gradient(135deg, #08111f 0%, #0b1220 40%, #0b1728 100%);
    color: var(--text);
}
.block-container {
    max-width: 1360px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text);
}
.hero-card {
    background: rgba(15,27,45,.9);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 18px 45px rgba(0,0,0,.18);
}
.main-title {
    font-size: 2.05rem;
    font-weight: 780;
    letter-spacing: -.04em;
    color: #f8fafc;
    margin-bottom: .12rem;
}
.sub-title {
    font-size: .95rem;
    color: var(--muted);
}
.metric-card {
    background: rgba(17,31,52,.9);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: .9rem 1rem;
    min-height: 88px;
}
.metric-label {
    font-size: .78rem;
    color: var(--muted);
    margin-bottom: .28rem;
}
.metric-value {
    font-size: 1.05rem;
    color: #f8fafc;
    font-weight: 720;
}
.metric-caption {
    font-size: .75rem;
    color: var(--muted);
    margin-top: .18rem;
}
.user-line {
    background: rgba(96,165,250,.09);
    border: 1px solid rgba(96,165,250,.24);
    border-left: 4px solid var(--blue);
    border-radius: 16px;
    padding: .9rem 1rem;
    margin: .7rem 0;
}
.assistant-line {
    background: rgba(45,212,191,.07);
    border: 1px solid rgba(45,212,191,.22);
    border-left: 4px solid var(--accent);
    border-radius: 16px;
    padding: .9rem 1rem;
    margin: .7rem 0;
}
.msg-meta {
    font-size: .78rem;
    color: var(--muted);
    margin-bottom: .35rem;
}
.badge {
    display: inline-block;
    padding: .18rem .52rem;
    border-radius: 999px;
    border: 1px solid var(--border-soft);
    background: #101d31;
    color: var(--muted);
    font-size: .73rem;
    margin-right: .35rem;
    margin-top: .25rem;
}
.badge-ok {
    border-color: rgba(34,197,94,.35);
    color: #86efac;
    background: rgba(34,197,94,.08);
}
.badge-error {
    border-color: rgba(239,68,68,.4);
    color: #fca5a5;
    background: rgba(239,68,68,.08);
}
.badge-tool {
    border-color: rgba(45,212,191,.35);
    color: #99f6e4;
    background: rgba(45,212,191,.08);
}
.badge-route {
    border-color: rgba(96,165,250,.35);
    color: #bfdbfe;
    background: rgba(96,165,250,.08);
}
.trace-title {
    font-size: .95rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: .7rem;
    margin-bottom: .25rem;
}
.step-card {
    background: #0b1626;
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: .65rem .8rem;
    margin-bottom: .48rem;
}
.step-main {
    font-size: .86rem;
    font-weight: 700;
    color: #99f6e4;
}
.step-detail {
    font-size: .76rem;
    color: var(--muted);
    margin-top: .12rem;
}
.clear-box {
    background: #0b1626;
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: .9rem .95rem;
    margin: .5rem 0;
}
.clear-title {
    font-size: .88rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: .25rem;
}
.clear-small {
    font-size: .78rem;
    color: var(--muted);
}
.stButton>button {
    background: #102035;
    color: #e5edf7;
    border: 1px solid #2b3b52;
    border-radius: 12px;
    font-weight: 650;
}
.stButton>button:hover {
    background: #172a44;
    border-color: #2dd4bf;
    color: #fff;
}
.stTextInput input, .stTextArea textarea {
    background: #0f1b2d !important;
    color: #e5edf7 !important;
    border: 1px solid #2b3b52 !important;
    border-radius: 13px !important;
}
.stSelectbox div[data-baseweb="select"]>div {
    background: #0f1b2d !important;
    border-color: #2b3b52 !important;
    color: #e5edf7 !important;
}
[data-testid="stChatInput"] textarea {
    background: #0f1b2d !important;
    border: 1px solid #2b3b52 !important;
}
hr {
    border-color: var(--border);
}
</style>
""", unsafe_allow_html=True)

def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")

def clean_json_value(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): clean_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean_json_value(x) for x in value]
        return str(value)
    except Exception:
        return str(value)

def compact_preview(value: Any, limit: int = 1800) -> str:
    try:
        text = json.dumps(clean_json_value(value), default=str, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    return text[:limit] + "\n...truncated..." if len(text) > limit else text

def add_step(steps: List[Dict[str, Any]], title: str, details: str = "", data: Any = None) -> None:
    item = {
        "time": now_time(),
        "title": title,
        "details": details,
        "data": clean_json_value(data) if data is not None else None
    }
    steps.append(item)
    print(f"[{item['time']}] {title} | {details}", flush=True)
    if data is not None:
        print(compact_preview(data), flush=True)

def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s-]", "", str(text or "")).strip().lower()

def tokenize(text: str) -> List[str]:
    return [x for x in re.findall(r"[a-zA-Z0-9_-]+", normalize_text(text)) if len(x) > 0]

def cosine_similarity_tokens(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    ca: Dict[str, int] = {}
    cb: Dict[str, int] = {}
    for x in ta: ca[x] = ca.get(x, 0) + 1
    for x in tb: cb[x] = cb.get(x, 0) + 1
    common = set(ca) & set(cb)
    dot = sum(ca[x] * cb[x] for x in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / max(na * nb, 1e-9)

def keyword_boost(question: str, tool_name: str) -> float:
    low = normalize_text(question)
    score = 0.0
    for kw in TOOL_CATALOG[tool_name].get("keywords", []):
        if kw in low:
            score += 0.20
            if " " + kw + " " in " " + low + " ":
                score += 0.10
    return score

def hard_rule_router(question: str) -> Dict[str, Any]:
    low = normalize_text(question)
    if any(x in low for x in ["refresh cache", "reload cache", "update cache", "rebuild cache", "new data added"]):
        return {"selected_tool": "cache_control", "score": 1.0, "reason": "hard_rule_cache_refresh"}
    if any(x in low for x in ["cache status", "cache stats", "cache statistics", "cache loaded", "when was cache refreshed", "cache rows"]):
        return {"selected_tool": "cache_status", "score": 1.0, "reason": "hard_rule_cache_status"}
    if any(x in low for x in ["health check", "system health", "diagnostic"]):
        return {"selected_tool": "system_info", "score": 1.0, "reason": "hard_rule_system_health"}
    if any(x in low for x in ["table schema", "columns available", "available columns", "database columns"]):
        return {"selected_tool": "system_info", "score": 1.0, "reason": "hard_rule_system_schema"}
    if any(x in low for x in ["feature columns", "model features", "features available"]):
        return {"selected_tool": "system_info", "score": 1.0, "reason": "hard_rule_system_features"}
    if any(x in low for x in ["environment", "config", "configuration"]):
        return {"selected_tool": "system_info", "score": 1.0, "reason": "hard_rule_system_environment"}
    return {}

def top1_similarity_tool(question: str) -> Dict[str, Any]:
    hard = hard_rule_router(question)
    if hard:
        return {**hard, "source": "hard_rule_router", "ranked_candidates": [hard], "selection": hard}
    ranked = []
    for tool, meta in TOOL_CATALOG.items():
        corpus = " ".join([tool, meta.get("description", ""), " ".join(meta.get("examples", [])), " ".join(meta.get("keywords", []))])
        cosine = cosine_similarity_tokens(question, corpus)
        boost = keyword_boost(question, tool)
        total = cosine + boost
        ranked.append(({"tool": tool, "label": TOOL_LABELS.get(tool, tool), "cosine_score": round(float(cosine), 4), "keyword_boost": round(float(boost), 4), "final_score": round(float(total), 4)}, total))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    if ranked:
        top = ranked[0][0]
        return {"selected_tool": top["tool"], "score": top["final_score"], "source": "top1_cosine_similarity", "ranked_candidates": [x[0] for x in ranked], "selection": top}
    return {"selected_tool": "run_analytics", "score": 0.0, "source": "top1_cosine_similarity", "ranked_candidates": [], "selection": {}}

def looks_like_brid(token: str) -> bool:
    token = str(token or "").strip().strip(",.;")
    if not token or len(token) < 3:
        return False
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", token, re.IGNORECASE):
        return True
    if "-" in token or "_" in token:
        return True
    return bool(re.search(r"\d", token))


def extract_brid(text: str) -> str:
    patterns = [
        r"\bbrid\s*[:=-]?\s*([a-zA-Z0-9_-]{3,})",
        r"\buser\s*[:=-]?\s*([a-zA-Z0-9_-]{3,})"
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip(",.;")
            if looks_like_brid(candidate):
                return candidate
    return ""

def extract_year(text: str) -> str:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text or "")
    return match.group(1) if match else ""

def extract_month(text: str) -> str:
    low = normalize_text(text)
    for key, value in MONTHS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", low):
            return value
    return ""

def extract_limit(text: str, default: int = DEFAULT_LIMIT) -> int:
    low = normalize_text(text)
    match = re.search(r"\btop\s*(\d+)\b", low) or re.search(r"\blimit\s*[:=-]?\s*(\d+)\b", low)
    if match:
        return max(1, min(int(match.group(1)), 10000))
    return default

def extract_threshold(text: str) -> float:
    low = normalize_text(text)
    match = re.search(r"\bthreshold\s*[:=-]?\s*(0\.\d+|1\.0|\d+)\b", low)
    if match:
        return max(0.0, min(float(match.group(1)), 1.0))
    pct = re.search(r"(\d+)\s*%", low)
    if pct:
        return max(0.0, min(float(pct.group(1)) / 100.0, 1.0))
    return HIGH_RISK_THRESHOLD

def extract_event_type(text: str) -> str:
    low = normalize_text(text)
    if "reported" in low or "report" in low:
        return "reported"
    if "no action" in low or "no-action" in low or "no_action" in low:
        return "no action"
    return "clicked"

def extract_simple_payload(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    rules = {
        "city": [
            r"\bcity\s*(?:as|s)?[:=-]?\s*([a-zA-Z ]+?)(?=\s+department|\s+dept|\s+subject|\s+template|\s+campaign|\s+event|$)"
        ],
        "usertags-Department": [
            r"\bdepartment\s*(?:as|s)?[:=-]?\s*([a-zA-Z &]+?)(?=\s+city|\s+subject|\s+template|\s+campaign|\s+event|$)",
            r"\bdept\s*(?:as|s)?[:=-]?\s*([a-zA-Z &]+?)(?=\s+city|\s+subject|\s+template|\s+campaign|\s+event|$)"
        ],
        "corporate_grade": [
            r"\bgrade\s*(?:as|s)?[:=-]?\s*([a-zA-Z0-9-]+)",
            r"\bdesignation\s*(?:as|s)?[:=-]?\s*([a-zA-Z0-9-]+)"
        ],
        "templatesubject": [
            r"\bsubject\s*(?:as|s)?[:=-]?\s*(.+?)(?=\s+template|\s+campaign|\s+event|$)",
            r"\btemplate\s*(?:as|s)?[:=-]?\s*(.+?)(?=\s+subject|\s+campaign|\s+event|$)"
        ],
        "campaignname": [
            r"\bcampaign\s*(?:as|s)?[:=-]?\s*(.+?)(?=\s+subject|\s+template|\s+event|$)"
        ]
    }
    for field, patterns in rules.items():
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip(",.;")
                if value:
                    payload[field] = value
                    break
    return payload

def infer_analysis_from_question(question: str) -> Tuple[str, List[str]]:
    filters = []
    month = extract_month(question)
    year = extract_year(question)
    if month and year:
        filters.append("month_year")
    elif month:
        filters.append("month")
    elif year:
        filters.append("year")
    low = normalize_text(question)
    if any(x in low for x in ["city", "cities"]):
        return "city_performance", ["city"] + filters
    if any(x in low for x in ["department", "dept"]):
        return "department_performance", ["department"] + filters
    if any(x in low for x in ["designation", "job level"]):
        return "designation_performance", ["designation"] + filters
    if "grade" in low:
        return "grade_performance", ["grade"] + filters
    if "coo" in low or "area" in low:
        return "coo_area_performance", ["coo_area"] + filters
    if "campaign" in low or "simulation" in low:
        return "campaign_performance", ["campaign"] + filters
    if "template" in low:
        return "template_performance", ["template"] + filters
    if "subject" in low:
        return "subject_performance", ["subject"] + filters
    if month and year:
        return "month_year_trend", ["month_year"]
    if month:
        return "monthly_trend", ["month"]
    if year:
        return "yearly_trend", ["year"]
    if "brid" in low or "employee" in low or "user" in low:
        return "brid_performance", ["brid"]
    return "overall_analysis", filters

def build_tool_args(tool_name: str, question: str, user_role: str) -> Dict[str, Any]:
    low = normalize_text(question)
    brid = extract_brid(question)
    month = extract_month(question)
    year = extract_year(question)
    limit = extract_limit(question)
    
    if tool_name == "run_analytics":
        analysis_type, group_by = infer_analysis_from_question(question)
        return {"analysis_type": analysis_type, "group_by": group_by, "filters": extract_simple_payload(question), "user_role": user_role, "top_n": DEFAULT_TOP_N}
    if tool_name == "employee_lookup":
        if any(x in low for x in ["top risky", "risky employees", "high risk user", "historically risky"]):
            return {"mode": "top_risky", "limit": limit, "user_role": user_role}
        if any(x in low for x in ["find employees", "employees in", "employee in"]):
            payload = extract_simple_payload(question)
            return {"mode": "find", "city": payload.get("city"), "department": payload.get("usertags-Department"), "limit": limit, "user_role": user_role}
        return {"mode": "profile", "brid": brid, "limit": limit, "user_role": user_role}
    if tool_name == "predict_risk":
        actual_limit = limit if limit != DEFAULT_LIMIT else None
        if any(x in low for x in ["predicted high risk", "high risk population", "high risk users"]) and not brid:
            return {"mode": "high_risk_population", "limit": actual_limit, "threshold": extract_threshold(question), "user_role": user_role}
        if "population" in low and not brid:
            return {"mode": "recent_population", "limit": actual_limit, "user_role": user_role}
        if brid:
            return {"mode": "by_brid", "brid": brid, "user_role": user_role}
        return {"mode": "from_payload", "payload": extract_simple_payload(question), "user_role": user_role, "argument_quality": "no_llm_rule_extracted"}
    if tool_name == "recommend_actions":
        if brid:
            return {"mode": "employee_improvement", "brid": brid, "user_role": user_role}
        analysis_type, group_by = infer_analysis_from_question(question)
        if group_by:
            return {"mode": "group_recommendations", "analysis_type": analysis_type, "group_by": group_by, "filters": extract_simple_payload(question), "top_n": 5, "user_role": user_role}
        return {"mode": "overall_recommendations", "filters": extract_simple_payload(question), "top_n": 5, "user_role": user_role}
    if tool_name == "simulation_users":
        return {"campaign_month": month or None, "campaign_year": year or None, "campaign_name": None, "event_type": extract_event_type(question), "user_role": user_role, "limit": limit if limit != DEFAULT_LIMIT else 5000}
    if tool_name == "cache_control":
        action = "clear" if any(x in low for x in ["clear", "reset", "delete"]) else "refresh"
        return {"action": action}
    if tool_name == "cache_status":
        return {}
    if tool_name == "system_info":
        if "schema" in low or "column" in low:
            return {"mode": "schema"}
        if "feature" in low or "environment" in low or "config" in low:
            return {"mode": "environment"}
        return {"mode": "health"}
    return {}

def validate_selection(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    if tool_name == "employee_lookup" and args.get("mode") == "profile" and not args.get("brid"):
        return False, "BRID is required for employee profile lookup. Example: 'show profile for BRID 75b4ae75-b'."
    if tool_name == "predict_risk" and args.get("mode") == "by_brid" and not args.get("brid"):
        return False, "BRID is required for BRID prediction. Example: 'predict no action probability for BRID 75b4ae75-b'."
    if tool_name == "recommend_actions" and args.get("mode") == "employee_improvement" and not args.get("brid"):
        return False, "BRID is required for employee improvement recommendations. Example: 'how can BRID 75b4ae75-b improve'."
    if tool_name == "simulation_users" and not args.get("campaign_month") and not args.get("campaign_year") and not args.get("campaign_name"):
        return False, "Campaign month, year, or campaign name is required for simulation users. Example: 'who clicked in March 2026 simulation'."
    return True, ""

def select_tool_no_llm(question: str, user_role: str, steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    selection = top1_similarity_tool(question)
    tool_name = selection.get("selected_tool")
    args = build_tool_args(tool_name, question, user_role)
    valid, error = validate_selection(tool_name, args)
    
    result = {
        "selected_tool": tool_name,
        "selected_label": TOOL_LABELS.get(tool_name, tool_name),
        "mode": args.get("mode"),
        "args": args,
        "validated": valid
    }
    if not valid:
        result["validation_error"] = error
    result["score"] = selection.get("score")
    result["source"] = selection.get("source")
    result["ranked_candidates"] = selection.get("ranked_candidates")
    result["routing_type"] = "top_cosine_similarity_no_llm"
    add_step(steps, "Tool selected using top-1 cosine similarity", f"{tool_name} | score={selection.get('score')} | source={selection.get('source')}", result)
    return clean_json_value(result)

def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}

def execute_tool_direct(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    steps = steps if steps is not None else []
    try:
        add_step(steps, "Direct execution started", f"Importing {MCP_SERVER_SCRIPT}")
        import phser as server
        server_map = {
            "run_analytics": server.run_analytics,
            "employee_lookup": server.employee_lookup,
            "predict_risk": server.predict_risk,
            "recommend_actions": server.recommend_actions,
            "simulation_users": server.simulation_users,
            "cache_control": server.cache_control,
            "cache_status": server.cache_status,
            "system_info": server.system_info
        }
        if tool_name not in server_map:
            result = {"status": "error", "message": f"Unsupported tool: {tool_name}"}
            add_step(steps, "Unsupported tool", tool_name, result)
            return result
        started = time.time()
        add_step(steps, "Tool execution started", tool_name, tool_args)
        result = server_map[tool_name](**tool_args) or {}
        result = clean_json_value(result)
        add_step(steps, "Tool execution completed", f"{round((time.time() - started) * 1000, 2)} ms", {"status": result.get("status") if isinstance(result, dict) else "unknown"})
        return result
    except Exception as e:
        result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "Tool execution failed", str(e), result)
        return result

async def execute_tool_mcp_stdio_async(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import asyncio
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        add_step(steps, "MCP stdio starting", f"{MCP_SERVER_COMMAND} {MCP_SERVER_SCRIPT}")
        server_params = StdioServerParameters(command=MCP_SERVER_COMMAND, args=[MCP_SERVER_SCRIPT])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                add_step(steps, "MCP session initialized", tool_name)
                started = time.time()
                result = await session.call_tool(tool_name, arguments=tool_args or {})
                add_step(steps, "MCP tool returned", f"{round((time.time() - started) * 1000, 2)} ms")
                if hasattr(result, "content") and result.content:
                    text = getattr(result.content[0], "text", None)
                    if text:
                        parsed = safe_json_loads(text)
                        final = parsed if parsed else {"status": "success", "raw_text": text}
                        add_step(steps, "MCP response parsed", "", final)
                        return clean_json_value(final)
                final = result.model_dump() if hasattr(result, "model_dump") else result
                add_step(steps, "MCP response returned as object", "", final)
                return clean_json_value(final)
    except Exception as e:
        result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "MCP execution failed", str(e), result)
        return result

def execute_tool_mcp_stdio(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import asyncio
    try:
        return asyncio.run(execute_tool_mcp_stdio_async(tool_name, tool_args, steps))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(execute_tool_mcp_stdio_async(tool_name, tool_args, steps))
        finally:
            loop.close()

def execute_selected_tool(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if EXECUTION_MODE == "mcp_stdio":
        return execute_tool_mcp_stdio(tool_name, tool_args, steps)
    return execute_tool_direct(tool_name, tool_args, steps)

def percent(value: Any) -> str:
    try:
        return f"{round(float(value) * 100, 2)}%"
    except Exception:
        return "N/A"

def summarize_deterministic(tool_name: str, tool_args: Dict[str, Any], tool_output: Dict[str, Any]) -> str:
    if isinstance(tool_output, dict) and "ui_summary" in tool_output:
        return tool_output["ui_summary"]

    status = tool_output.get("status", "unknown") if isinstance(tool_output, dict) else "unknown"
    if status == "error":
        return f"Request Failed. The selected tool returned an error: {tool_output.get('message', 'Unknown error')}"

    tool_label = TOOL_LABELS.get(tool_name, tool_name)
    return f"Executed `{tool_label}` successfully. View the structured output below:"

def run_full_pipeline(question: str, user_role: str) -> Dict[str, Any]:
    started = time.time()
    steps: List[Dict[str, Any]] = []
    add_step(steps, "Request received", f"role={user_role}, question='{question}'")
    try:
        selection = select_tool_no_llm(question, user_role, steps)
        if selection.get("validation_error"):
            add_step(steps, "Validation failed", selection.get("validation_error"))
            return {
                "status": "validation_error",
                "final_answer": selection.get("validation_error"),
                "selected_tool": selection.get("selected_tool"),
                "tool_args": selection.get("args", {}),
                "tool_output": {},
                "selection": selection,
                "steps": steps,
                "latency_ms": round((time.time() - started) * 1000, 2)
            }
        tool_name = selection.get("selected_tool")
        tool_args = selection.get("args") or {}
        tool_output = execute_selected_tool(tool_name, tool_args, steps)
        final_answer = summarize_deterministic(tool_name, tool_args, tool_output)
        status = "success" if isinstance(tool_output, dict) and tool_output.get("status") != "error" else "error"
        latency_ms = round((time.time() - started) * 1000, 2)
        add_step(steps, "Pipeline completed", status=status, latency_ms=latency_ms)
        return {
            "status": status,
            "final_answer": final_answer,
            "selected_tool": tool_name,
            "tool_args": tool_args,
            "tool_output": tool_output,
            "selection": selection,
            "steps": steps,
            "latency_ms": latency_ms
        }
    except Exception as e:
        failure = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "Pipeline failed", str(e), failure)
        return {
            "status": "error",
            "final_answer": "Something failed while processing the question: " + str(e),
            "selected_tool": None,
            "tool_args": {},
            "tool_output": failure,
            "selection": {},
            "steps": steps,
            "latency_ms": round((time.time() - started) * 1000, 2)
        }

def init_state() -> None:
    defaults = {"messages": [], "user_role": DEFAULT_USER_ROLE, "show_debug": True, "pending_question": ""}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def queue_question(question: str) -> None:
    st.session_state.pending_question = str(question or "").strip()

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Control Panel")
        st.session_state.user_role = st.selectbox("Access mode", ["user", "admin"], index=0 if st.session_state.user_role != "admin" else 1)
        if st.session_state.user_role == "admin":
            st.warning("Admin mode may expose identifiers only if backend policy allows it.")
        else:
            st.info("User mode hides personal identifiers by default.")
        st.session_state.show_debug = st.toggle("Show trace below answers", value=st.session_state.show_debug)
        st.markdown("---")
        st.caption("(No LLM mode)")
        st.caption("Tool selection: top-1 cosine similarity")
        st.caption("Summary: Deterministic only")
        st.caption(f"Execution mode: {EXECUTION_MODE}")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", width="stretch"):
                st.session_state.messages = []
                st.session_state.pending_question = ""
                st.rerun()
        with c2:
            if st.button("Cache status", width="stretch"):
                queue_question("Cache status")
                st.rerun()
        if st.button("Refresh cache", width="stretch"):
            queue_question("Refresh cache")
            st.rerun()
        st.markdown("---")
        st.markdown("### Quick prompts")
        for idx, item in enumerate(EXAMPLE_QUESTIONS):
            if st.button(item, key=f"ex_{idx}", width="stretch"):
                queue_question(item)
                st.rerun()

def render_header() -> None:
    st.markdown(f"""<div class="hero-card"><div class="main-title">{APP_TITLE}</div><div class="sub-title">Local top-1 cosine routing · Deterministic arguments · Deterministic summaries · No LLM calls</div></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Routing</div><div class="metric-value">Top-1 Cosine</div><div class="metric-caption">Local lexical similarity</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Execution</div><div class="metric-value">{EXECUTION_MODE}</div><div class="metric-caption">MCP/Direct tool layer</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Summary</div><div class="metric-value">Deterministic</div><div class="metric-caption">No LLM generation</div></div>""", unsafe_allow_html=True)
    with c4:
        role = st.session_state.user_role
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Access Mode</div><div class="metric-value">{role}</div><div class="metric-caption">Role-aware backend output</div></div>""", unsafe_allow_html=True)
    with st.expander("Current architecture", expanded=False):
        st.code("phishing_streamlit_ui_noLLM.py -> local top1_similarity_tool -> cosine similarity over tool catalog -> selected tool + deterministic args -> phishing_mcp_server.py -> executes selected MCP/direct tool -> deterministic summary -> no LLM selection -> no LLM generation", language="text")

def render_clear_tool_output(tool_output: Any) -> None:
    if isinstance(tool_output, list):
        if tool_output and isinstance(tool_output[0], dict):
            st.dataframe(tool_output, use_container_width=True)
        else:
            st.write(tool_output)
        return

    if not isinstance(tool_output, dict):
        st.code(str(tool_output), language="text")
        return

    ui_highlights = tool_output.get("ui_highlights", [])
    if ui_highlights:
        st.markdown("#### 📊 Highlights")
        cols = st.columns(min(len(ui_highlights), 4))
        for i, highlight in enumerate(ui_highlights):
            cols[i % 4].metric(label=highlight.get("label", ""), value=highlight.get("value", ""))
        st.write("")

    status = tool_output.get("status")
    message = tool_output.get("message")
    c1, c2 = st.columns(2)
    c1.metric("Status", str(status or "-").capitalize())

    count_val = tool_output.get("count", tool_output.get("user_count", tool_output.get("total_rows", tool_output.get("rows_analyzed", "-"))))
    c2.metric("Records / Count", str(count_val))

    if message:
        st.info(str(message))

    skip_keys = {
        "status", "message", "count", "user_count", "total_rows", "rows_analyzed",
        "tool_catalog_entry", "ui_instruction", "trace_instruction", "execution_trace",
        "ui_summary", "ui_highlights"
    }

    for key, value in tool_output.items():
        if key in skip_keys or value is None or value == "":
            continue

        display_title = key.replace("_", " ").title()
        st.markdown(f"#### {display_title}")

        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                st.dataframe(value[:100], use_container_width=True)
                if len(value) > 100:
                    st.caption(f"Showing first 100 of {len(value)} items")
            else:
                for item in value[:20]:
                    st.markdown(f"- {item}")
                if len(value) > 20:
                    st.caption(f"...and {len(value) - 20} more items.")
        elif isinstance(value, dict):
            st.json(value)
        else:
            st.write(value)

def render_steps(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        st.info("No trace steps captured.")
        return
    for step in steps:
        st.markdown(f"""<div class="step-card"><div class="step-main {html.escape(str(step.get("status", ""))) or ""}">{html.escape(str(step.get("time", "")))} | {html.escape(str(step.get("title", "")))}</div><div class="step-detail">{html.escape(str(step.get("details", "")))}</div></div>""", unsafe_allow_html=True)
        if step.get("data") is not None:
            st.code(compact_preview(step.get("data"), limit=1800), language="json")

def render_trace(backend: Dict[str, Any], idx) -> None:
    if not backend or not st.session_state.show_debug:
        return
    selected_tool = backend.get("selected_tool")
    latency_ms = backend.get("latency_ms")
    status = backend.get("status") or "completed"
    badge_class = "badge-ok" if status != "error" else "badge-error"
    tool_output = backend.get("tool_output") or {}
    tool_meta = (tool_output or {}).get("tool_catalog_entry") or {}
    ui_instruction = (tool_output or {}).get("ui_instruction") or (tool_output or {}).get("trace_instruction")
    st.markdown(f"""<div class="answer-toolbar"><span class="badge {badge_class}">Tool: {html.escape(str(selected_tool or "-"))}</span><span class="badge {badge_class}">Routing: Top-1 cosine</span><span class="badge {badge_class}">Score: {html.escape(str(backend.get("score") or "-"))}</span><span class="badge">Latency: {html.escape(str(latency_ms or "-"))} ms</span><span class="badge {badge_class}">Status: {html.escape(str(status))}</span></div>""", unsafe_allow_html=True)
    if tool_meta:
        st.caption(f"Tool contract: {tool_meta.get('description', '')}")
    if ui_instruction:
        st.info(ui_instruction)
    with st.expander("Trace", expanded=False):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Tool Selection", "Tool Arguments", "Tool Contract", "Execution Steps", "Raw JSON", "Download"])
        with tab1:
            st.markdown('<div class="trace-title">Tool Selection - Top 1 Cosine Similarity</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Tool", str(backend.get("selected_tool") or "-"))
            c2.metric("Score", str(backend.get("score") or "-"))
            selection = backend.get("selection") or {}
            c3.metric("Mode", str(selection.get("mode") or "-"))
            ranked = backend.get("ranked_candidates")
            if ranked:
                st.markdown("#### Ranked candidates")
                st.dataframe(ranked, width="stretch")
            st.markdown("#### Selection JSON")
            st.json(selection)
        with tab2:
            st.markdown('<div class="trace-title">Tool Arguments</div>', unsafe_allow_html=True)
            st.json(backend.get("tool_args") or {})
        with tab3:
            st.markdown('<div class="trace-title">Tool Contract</div>', unsafe_allow_html=True)
            st.json(tool_meta or {})
            if isinstance(tool_output, dict):
                st.json({
                    "ui_highlights": tool_output.get("ui_highlights") or [],
                    "ui_summary": tool_output.get("ui_summary"),
                    "execution_trace": tool_output.get("execution_trace") or [],
                })
        with tab4:
            st.markdown('<div class="trace-title">Execution Steps</div>', unsafe_allow_html=True)
            render_steps(backend.get("steps") or [])
        with tab5:
            st.markdown('<div class="trace-title">Raw JSON Trace</div>', unsafe_allow_html=True)
            st.json(backend)
        with tab6:
            trace_key = f"trace_download_{backend.get('request_id', id(backend))}"
            st.download_button("Download trace JSON", data=json.dumps(backend, indent=2, default=str), file_name=f"trace_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", key=trace_key)

def render_chat() -> None:
    if not st.session_state.messages:
        st.info("Ask a question below or choose a quick prompt from the sidebar.")
        return
    st.markdown("### Conversation")
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role")
        if role == "user":
            st.markdown(f"""<div class="user-line"><div class="msg-meta">You ({msg.get('time', '')})</div><div class="msg-content">{msg.get('content', '')}</div></div>""", unsafe_allow_html=True)
        else:
            backend = msg.get("backend", {}) or {}
            tool = backend.get("selected_tool")
            latency = backend.get("latency_ms")
            meta = f"Copilot ({msg.get('time', '')})"
            if tool:
                meta += f" · {tool} ({latency} ms)"
            st.markdown(f"""<div class="assistant-line"><div class="msg-meta">{meta}</div><div class="msg-content">{msg.get('content', '')}</div></div>""", unsafe_allow_html=True)
            tool_output = backend.get("tool_output")
            if tool_output not in (None, {}):
                with st.container():
                    st.write("")
                    render_clear_tool_output(tool_output)
            render_trace(backend, idx)

def handle_question(question: str) -> None:
    question = str(question or "").strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question, "time": now_time()})
    try:
        with st.status("Running phishing analytics workflow...", expanded=False) as status:
            result = run_full_pipeline(question, st.session_state.user_role)
            if result.get("status") == "success":
                status.update(label="Workflow completed", state="complete", expanded=False)
            else:
                status.update(label="Workflow completed with issue", state="error", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": result.get("final_answer") or "Could not generate a final answer.", "time": now_time(), "backend": result})
    except Exception as e:
        error_backend = {"status": "error", "traceback": traceback.format_exc(), "logs": [{"time": now_time(), "step": "UI EXCEPTION", "message": str(e), "data": traceback.format_exc()}]}
        print(f"UI EXCEPTION: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        st.session_state.messages.append({"role": "assistant", "content": f"Something failed while processing the question: {str(e)}", "time": now_time(), "backend": error_backend})

def main() -> None:
    init_state()
    render_sidebar()
    render_header()
    render_chat()
    if st.session_state.pending_question:
        pending = st.session_state.pending_question
        st.session_state.pending_question = ""
        handle_question(pending)
        st.rerun()
    if question := st.chat_input("Ask about phishing analytics, predictions, recommendations, cache, or system status..."):
        handle_question(question)
        st.rerun()

if __name__ == "__main__":
    main()


