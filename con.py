# Config.py
# 
# # Here is a concise breakdown of what `Config.py` is responsible for and actively doing in your architecture:

# **Core Identity**

# * **Server-Side Feature Pipeline:** It acts as the dedicated data preparation and transformation module, sitting between raw data extraction and the core analytical/ML engines.

# **Primary Responsibilities & Logic Paths**

# * **End-to-End Feature Engineering:** Exposes a single public gateway (`apply_feature_engineering`) that orchestrates the entire transformation of a raw simulation DataFrame into a strictly typed, model-ready format.
# * **Semantic AI Classification:** Loads and caches local Hugging Face transformer models to dynamically categorize free-text fields (like email subjects and department names) using vector embeddings and cosine similarity matrix calculations.
# * **Data Sanitization & Normalization:** Safely cleans dirty inputs (handling nulls, standardizing text formatting, and mapping ambiguous terms to predefined "other" categories) to prevent downstream pipeline crashes.
# * **Categorical & Temporal Encoding:** Applies fixed business logic to map string variables (e.g., Corporate Grades, City Zones, COO Areas) into integer encodings and extracts actionable temporal features (e.g., business hours, tenure).

# **System & Configuration Management**

# * **Strict Encapsulation:** Keeps all heavy AI embedding logic, mapping dictionaries, and sanitization utilities strictly private to prevent downstream routing layers from accidentally importing them.
# * **Environment Configuration:** Serves as the central loader for `.env` variables related to AI model paths, feature column definitions, and data limits.
# * **Hardware Fallback:** Dynamically checks for available NVIDIA CUDA cores and gracefully degrades to CPU execution if GPU hardware is unavailable.

# Config.py # Used by phishing_mcp_server # Contains preprocessing, feature engineering, semantic classification and model save/load config.
import logging
import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import (AutoTokenizer, AutoModel)

load_dotenv()
LOGGER = logging.getLogger("Config")
MODEL_DIR = "models"
MODEL_PATH = os.getenv('MODEL_PATH')
SEMANTIC_CATEGORY_EMBEDDINGS = {}
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH")
BRID_COLUMN = os.getenv("BRID_COLUMN", "usertags-BRID")
BRID_COLUMN_CANDIDATES = [BRID_COLUMN, "usertags-BRID", "proofpoint_brid", "brid"]

EMPLOYEE_ID_COLUMNS = [
    "brid",
    "usertags-BRID",
    "proofpoint_brid",
    "employee",
    "employee_id",
    "user",
    "user_id",
]
PERSONAL_COLUMNS = ["name", "email", "mail", "upn"]

MAX_ANALYTICS_ROWS = int(os.getenv("MAX_ANALYTICS_ROWS", 10000))
MAX_PREDICTION_ROWS = int(os.getenv("MAX_PREDICTION_ROWS", 10000))

DROP_COLUMNS = [
    'reportingdate', 'userfirstname', 'userlastname', 'useremailaddress', 'useractiveflag', 
    'userdeleteddate', 'eventtimestamp', 'campaignname', 'campaignstartdate', 'campaignenddate', 
    'autoenrollment', 'campaigntype', 'campaignstatus', 'assessmentisarchived', 'sso_id', 
    'usertags-Azure UPN', 'usertags-Date Added', 'usertags-Business Unit', 'usertags-On-Premises Domain Name', 
    'usertags-On-Premises Extension Attribute 5', 'usertags-On-Premises Extension Attribute 6', 
    'cio', 'region', 'CISO', 'legal_entity', 'bu1', 'loaddatetime', 'employee_type', 
    'usertags-Location', 'COO', 'proofpoint_brid'
]

# PII columns used across the application (single source of truth)
PII_COLUMNS = [
    "userfirstname",
    "userlastname",
    "useremailaddress",
    "sso_id",
    "proofpoint_brid",
    "brid",
    "usertags-BRID",
    "usertags-Azure UPN",
    "usertags-On-Premises Extension Attribute 5",
    "usertags-On-Premises Extension Attribute 6",
]

DROP_AFTER_FEATURE_ENGINEERING = [
    'eventtype', 'is_hugs', 'COO_Area', 'coo_area_grouped', 'corporate_grade', 
    'usertags-Department', 'businessarea1', 'businessarea2', 'businessarea3', 
    'businessarea4', 'businessarea5', 'templatename', 'templatesubject', 'senttimestamp', 
    'localHireRehireDate', 'country', 'city', 'city_zone', '# sent_month', '# sent_day'
]

TARGET_MAP = {
    0: ['no action', 'noaction', 'email view', 'view', 'viewed', 'opened', 'tm sent'],
    1: ['email click', 'clicked link', 'clicked', 'click'],
    2: ['reported', 'report', 'reported phishing', 'reported phish']
}

target_function = {item: label for label, items in TARGET_MAP.items() for item in items}
TARGET_LABELS = {0: 'No Action', 1: 'Clicked Link', 2: 'Reported'}

### Utils
def _is_series(x):
    return isinstance(x, pd.Series)

def _safe_text_scalar(x):
    if x is None or pd.isna(x): 
        return ''
    return re.sub(r'\s+', ' ', str(x).strip().lower())

def _safe_text_series(s):
    if not _is_series(s): 
        s = pd.Series([s])
    return (s.fillna("").astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True))

def _clean_unknown_scalar(x):
    x = _safe_text_scalar(x)
    if x in ['', 'unknown', 'unk', 'null', 'none', 'nan', 'na', 'n/a', 'not available']: 
        return 'other'
    return x

def _clean_unknown_series(s):
    s = _safe_text_series(s)
    return s.replace({'': 'other', 'unknown': 'other', 'unk': 'other', 'null': 'other', 'none': 'other', 'nan': 'other', 'na': 'other', 'n/a': 'other', 'not available': 'other'})

def _combine_text_columns(*cols):
    max_len = 1
    for col in cols:
        if _is_series(col):
            max_len = len(col)
            break
    output = pd.Series([''] * max_len)
    for col in cols:
        if _is_series(col):
            current = _safe_text_series(col).reset_index(drop=True)
        else:
            current = pd.Series([_safe_text_scalar(col)] * max_len)
        output = output + ' ' + current
    return _safe_text_series(output)

def _map_target(eventtype):
    if _is_series(eventtype):
        return _safe_text_series(eventtype).map(target_function).fillna(-1).astype(int)
    return int(target_function.get(_safe_text_scalar(eventtype), -1))

TOKENIZER = None
SEMANTIC_MODEL = None
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") 
# print(f"Semantic Device: {DEVICE}")

EMAIL_CATEGORIES = {
    "security_credential": (
        "email about password, login, credentials, authentication, account, credit card, debit card, PayPal, wire transfer, "
        "account verification, account security, suspicious login, "
        "compromised account or security alert"
    ),
    "financial": (
        "email about invoice, payment, billing, payroll, refund, "
        "bank account suspension, account lockout, warning, final notice, "
        "transaction, purchase or subscription"
    ),
    "urgency_pressure": (
        "email demanding urgent action, immediate response, deadline, "
        "expiry, restricted access or failure to comply"
    ),
    "authority_trust": (
        "email appearing to come from HR, IT support, compliance, "
        "security team, payroll, management, administrator, "
        "leadership or internal department"
    ),
    "link_attachment": (
        "email asking user to click a link, open a document, "
        "download a file, review an attachment, access a shared file, "
        "approve a request or view online content"
    )
}

BUSINESS_CATEGORIES = {
    "technology_security": (
        "technology, engineering, software development, cyber security, "
        "information security, infrastructure, cloud, platform engineering, "
        "network engineering, database administration, application support, "
        "technology operations, identity and access management"
    ),
    "finance_risk_control": (
        "finance, accounting, treasury, payments, payroll, audit, "
        "procurement, taxation, risk management, compliance, legal, "
        "governance, controls and regulatory functions"
    ),
    "customer_business_operations": (
        "customer service, customer support, client operations, "
        "retail banking, wealth management, corporate banking, "
        "business operations, fraud operations, onboarding, servicing "
        "and front office functions"
    ),
    "leadership_strategy": (
        "leadership, executive management, business strategy, "
        "chief officer functions, directors, vice presidents, "
        "managing directors, transformation and organisational management"
    )
}

def _get_semantic_model():
    global TOKENIZER
    global SEMANTIC_MODEL
    if TOKENIZER is None:
        LOGGER.info("Loading semantic tokenizer from %s", SEMANTIC_MODEL_PATH)
        TOKENIZER = AutoTokenizer.from_pretrained(SEMANTIC_MODEL_PATH)
    if SEMANTIC_MODEL is None:
        LOGGER.info("Loading semantic model from %s on device %s", SEMANTIC_MODEL_PATH, DEVICE)
        SEMANTIC_MODEL = (AutoModel.from_pretrained(SEMANTIC_MODEL_PATH).to(DEVICE))
    SEMANTIC_MODEL.eval()
    LOGGER.info("Semantic model ready")
    return TOKENIZER, SEMANTIC_MODEL

def _get_embeddings(texts):
    LOGGER.info("Computing semantic embeddings for %s items", len(texts) if hasattr(texts, '__len__') else 1)
    tokenizer, model = _get_semantic_model()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad(): 
        output = model(**encoded)
    token_embeddings = output.last_hidden_state
    mask = (encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float())
    embeddings = (torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9))
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def _get_embeddings_batched(texts, batch_size=256):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = _get_embeddings(batch)
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)

def _get_category_embeddings(categories):
    global SEMANTIC_CATEGORY_EMBEDDINGS
    category_key = tuple(categories.keys())
    if category_key not in SEMANTIC_CATEGORY_EMBEDDINGS:
        SEMANTIC_CATEGORY_EMBEDDINGS[category_key] = (_get_embeddings(list(categories.values())))
    return SEMANTIC_CATEGORY_EMBEDDINGS[category_key]

def _classify_semantic_text(text, categories):
    text = _safe_text_scalar(text)
    category_names = list(categories.keys())
    if not text: 
        return {f"{cat}_similarity": 0.0 for cat in category_names}
    category_embeddings = _get_category_embeddings(categories)
    text_embedding = _get_embeddings([text])[0]
    similarities = (category_embeddings @ text_embedding)
    return {f"{cat}_similarity": round(float(similarities[i]), 2) for i, cat in enumerate(category_names)}

def _classify_semantic_series(text_series, categories):
    text_series = _safe_text_series(text_series)
    unique_values = text_series.unique()
    # print(f"Rows={len(text_series)};") 
    # print(f"Unique={len(unique_values)};")
    if len(unique_values) == 0: 
        return {}
    text_embeddings = _get_embeddings_batched(unique_values.tolist(), batch_size=256)
    category_names = list(categories.keys())
    category_embeddings = _get_category_embeddings(categories)
    similarity_matrix = (text_embeddings @ category_embeddings.T)
    cache = {}
    for idx, value in enumerate(unique_values):
        result = {}
        for cat_idx, cat in enumerate(category_names):
            # result[f"{cat}_similarity"] = round(float(similarity_matrix[idx, cat_idx]),2)
            result[f"{cat}_similarity"] = float(similarity_matrix[idx, cat_idx])
        cache[value] = result
    mapped = text_series.map(cache)
    features = {}
    for cat in category_names:
        col = f"{cat}_similarity"
        features[col] = mapped.apply(lambda x: x[col]).astype(float)
    return features

### Email Classification
def _classify_email(template_name, template_subject):
    if _is_series(template_name):
        if not _is_series(template_subject):
            template_subject = pd.Series([''] * len(template_name), index=template_name.index)
        combined = _combine_text_columns(template_name, template_subject)
        return _classify_semantic_series(combined, categories=EMAIL_CATEGORIES)
    combined = f"{_safe_text_scalar(template_name)} {_safe_text_scalar(template_subject)}"
    return _classify_semantic_text(combined, categories=EMAIL_CATEGORIES)

### Business FE
def _extract_business_features(usertags_Department, businessarea1, businessarea2, businessarea3, businessarea4, businessarea5):
    if _is_series(usertags_Department):
        combined = _combine_text_columns(usertags_Department, businessarea1, businessarea2, businessarea3, businessarea4, businessarea5)
        return _classify_semantic_series(combined, categories=BUSINESS_CATEGORIES)
    combined = (f"{_safe_text_scalar(usertags_Department)} " f"{_safe_text_scalar(businessarea1)} " f"{_safe_text_scalar(businessarea2)} " f"{_safe_text_scalar(businessarea3)} " f"{_safe_text_scalar(businessarea4)} " f"{_safe_text_scalar(businessarea5)}")
    return _classify_semantic_text(combined, categories=BUSINESS_CATEGORIES)

def _calculate_tenure(hire_date, sent_time=None):
    if sent_time is None: 
        sent_time = pd.Timestamp.now()
    hire_dt = pd.to_datetime(hire_date, errors='coerce')
    sent_ts = pd.to_datetime(sent_time, errors='coerce')
    if _is_series(hire_dt):
        if not _is_series(sent_ts):
            sent_ts = pd.Series([sent_ts] * len(hire_dt), index=hire_dt.index)
        hire_dt = hire_dt.mask(hire_dt.dt.strftime('%Y-%m-%d') == '1900-01-01')
        tenure_years = (sent_ts - hire_dt).dt.days / 365.25
        tenure_years = tenure_years.replace([np.inf, -np.inf], np.nan)
        tenure_years = tenure_years.clip(lower=0)
        return tenure_years.fillna(0)
    if pd.isna(hire_dt) or pd.isna(sent_ts): 
        return 0.0
    if hire_dt.strftime("%Y-%m-%d") == '1900-01-01': 
        return 0.0
    tenure_years = (sent_ts - hire_dt).days / 365.25
    return max(float(tenure_years), 0.0)

### Time
def _extract_time_features(senttimestamp):
    sent_ts = pd.to_datetime(senttimestamp, errors='coerce')
    if _is_series(sent_ts):
        return {
            'sent_hour': sent_ts.dt.hour.fillna(0).astype(int),
            'sent_day': sent_ts.dt.day.fillna(0).astype(int),
            'sent_month': sent_ts.dt.month.fillna(0).astype(int),
            'sent_weekday': sent_ts.dt.weekday.fillna(0).astype(int),
            'is_weekend': sent_ts.dt.weekday.isin([5, 6]).astype(int),
            'is_business_hour': sent_ts.dt.hour.between(9, 18).astype(int),
        }
    if pd.isna(sent_ts):
        return {
            'sent_hour': 0,
            'sent_day': 0,
            'sent_month': 0,
            'sent_weekday': 0,
            'is_weekend': 0,
            'is_business_hour': 0,
        }
    return {
        'sent_hour': int(sent_ts.hour),
        'sent_day': int(sent_ts.day),
        'sent_month': int(sent_ts.month),
        'sent_weekday': int(sent_ts.weekday()),
        'is_weekend': int(sent_ts.weekday() in [5, 6]),
        'is_business_hour': int(9 <= sent_ts.hour <= 18),
    }

### is Hugs (monitored people)
hugs_mapping = {'yes': 1, 'no': 0, 'other': 0}

def _hugs_map(is_hugs):
    if _is_series(is_hugs):
        return _clean_unknown_series(is_hugs).map(hugs_mapping).fillna(0).astype(int)
    return int(hugs_mapping.get(_clean_unknown_scalar(is_hugs), 0))

### Designation
grade_mapping = {'unknown': -1, 'other': -1, 'ba1': 0, 'ba2': 1, 'ba3': 2, 'ba4': 3, 'avp': 4, 'vp': 5, 'd': 6, 'md': 7}

def _grade_map(corporate_grade):
    if _is_series(corporate_grade):
        return _clean_unknown_series(corporate_grade).map(grade_mapping).fillna(-1).astype(int)
    return int(grade_mapping.get(_clean_unknown_scalar(corporate_grade), -1))

### COO Area
COO_CATEGORIES = {
    'group control': 'Risk_Control',
    'chief information security office': 'Technology_Security',
    'group shared technology': 'Technology_Security',
    'barclays uk': 'Banking_Business',
    'uk corporate bank': 'Banking_Business',
    'investment bank': 'Banking_Business',
    'barclays us consumer bank': 'Banking_Business',
    'private bank and wealth management': 'Banking_Business',
    'barclays europe': 'Banking_Business',
    'corp and payments': 'Banking_Business',
    'bx coo': 'Operations_Transformation',
    'unknown': 'Other',
    'other': 'Other'
}
COO_AREA_VALUES = ['Banking_Business', 'Technology_Security', 'Risk_Control', 'Operations_Transformation', 'Other']

def _coo_area_group(coo_area):
    if _is_series(coo_area):
        normalized = _clean_unknown_series(coo_area)
        return normalized.map(lambda x: COO_CATEGORIES.get(x, 'Other'))
    value = _clean_unknown_scalar(coo_area)
    return COO_CATEGORIES.get(value, 'Other')

### City Zones
city_to_zone = {
    '# india': 'India_West', 'pune': 'India_West', 'mumbai': 'India_West', 'nagpur': 'India_West', 'gandhinagar': 'India_West',
    'noida': 'India_North', 'new delhi': 'India_North', 'delhi': 'India_North', 'gurgaon': 'India_North', 'gurugram': 'India_North',
    'chennai': 'India_South', 'bangalore': 'India_South', 'bengaluru': 'India_South', 'hyderabad': 'India_South', 'kochi': 'India_South', 'nelamangala': 'India_South',
    'kolkata': 'India_East',
    'indore': 'India_Central',
    'india - default legacy site': 'India_Other',
    '# UK': 'UK_London', 'london': 'UK_London',
    'glasgow': 'UK_Scotland', 'edinburgh': 'UK_Scotland', 'aberdeen': 'UK_Scotland', 'kilmarnock': 'UK_Scotland',
    'knutsford': 'UK_North_West', 'chester': 'UK_North_West', 'manchester': 'UK_North_West', 'liverpool': 'UK_North_West', 'preston': 'UK_North_West', 'warrington': 'UK_North_West', 'northwich': 'UK_North_West', 'crewe': 'UK_North_West', 'wythenshawe': 'UK_North_West',
    'northampton': 'UK_Midlands', 'birmingham': 'UK_Midlands', 'nottingham': 'UK_Midlands', 'derby': 'UK_Midlands', 'wolverhampton': 'UK_Midlands', 'coventry': 'UK_Midlands', 'leicester': 'UK_Midlands', 'milton keynes': 'UK_Midlands', 'telford': 'UK_Midlands', 'solihull': 'UK_Midlands',
    'uk - default legacy site': 'UK_Other',
    '# US': 'US_North_East', 'new york': 'US_North_East', 'jersey city': 'US_North_East', 'whippany': 'US_North_East', 'wilmington': 'US_North_East', 'mount laurel': 'US_North_East', 'edgewood': 'US_North_East', 'boston': 'US_North_East',
    'chicago': 'US_Midwest', 'cleveland': 'US_Midwest',
    'henderson': 'US_West', 'irvine': 'US_West', 'san francisco': 'US_West', 'menlo park': 'US_West', 'los angeles': 'US_West', 'san jose': 'US_West', 'seattle': 'US_West',
    'houston': 'US_South', 'miami': 'US_South', 'raleigh': 'US_South', 'washington': 'US_South',
    '# Philippines': 'Philippines', 'quezon city': 'Philippines', 'taguig city': 'Philippines', 'makati city': 'Philippines', 'makati city metro manila': 'Philippines', 'pasig city metro manila': 'Philippines', 'iloilo': 'Philippines',
    '# Europe': 'Europe_West', 'zurich': 'Europe_West', 'geneva': 'Europe_West', 'hamburg': 'Europe_West', 'frankfurt am main': 'Europe_West', 'madrid': 'Europe_West', 'dublin': 'Europe_West', 'paris': 'Europe_West', 'milan': 'Europe_West', 'monaco': 'Europe_West', 'lisbon': 'Europe_West', 'luxembourg': 'Europe_West', 'amsterdam': 'Europe_West', 'brussels': 'Europe_West',
    'stockholm': 'Europe_North',
    'prague': 'Europe_Central', 'krakow': 'Europe_Central',
    'vilnius': 'Europe_East',
    'istanbul': 'Europe_Middle_East',
    '# APAC': 'APAC_Singapore', 'singapore': 'APAC_Singapore',
    'hong kong': 'APAC_Hong_Kong',
    'tokyo': 'APAC_Japan',
    'shanghai': 'APAC_China', 'beijing': 'APAC_China',
    'taipei': 'APAC_Taiwan',
    'sydney': 'APAC_Australia',
    'tel aviv': 'Middle_East',
    '# Other': 'Canada', 'toronto': 'Canada', 'calgary': 'Canada',
    'guatemala city': 'LATAM', 'mexico city': 'LATAM', 'monterrey': 'LATAM', 'sao paulo': 'LATAM', 'cali-valle del cauca': 'LATAM', 'hato rey': 'LATAM',
    'johannesburg': 'Africa',
    'unknown': 'Other'
}
CITY_ZONE_VALUES = ['India', 'UK', 'US', 'Europe', 'APAC', 'Canada', 'LATAM', 'Africa', 'Philippines', 'Middle_East', 'Remote', 'Other']
COO_AREA_LABEL_MAP = {'Banking_Business': 0, 'Technology_Security': 1, 'Risk_Control': 2, 'Operations_Transformation': 3, 'Other': 4}
CITY_ZONE_LABEL_MAP = {'India': 0, 'UK': 1, 'US': 2, 'Europe': 3, 'APAC': 4, 'Canada': 5, 'LATAM': 6, 'Africa': 7, 'Philippines': 8, 'Middle_East': 9, 'Remote': 10, 'Other': 11}
US_STATE_CODES = {'al', 'az', 'ca', 'co', 'dc', 'de', 'fl', 'ga', 'il', 'ky', 'ma', 'md', 'me', 'mi', 'mn', 'nc', 'nj', 'nm', 'nv', 'ny', 'oh', 'or', 'pa', 'sc', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wy'}

def _reduce_city_zone(zone):
    zone = str(zone)
    if zone.startswith('India'): return 'India'
    if zone.startswith('UK'): return 'UK'
    if zone.startswith('US'): return 'US'
    if zone.startswith('Europe'): return 'Europe'
    if zone.startswith('APAC'): return 'APAC'
    if zone.startswith('Remote'): return 'Remote'
    if zone.startswith('Philippines'): return 'Philippines'
    if zone.startswith('Middle_'): return 'Middle_East'
    if zone in ['Canada', 'LATAM', 'Africa']: return zone
    return 'Other'

def _remote_zone(city_text):
    c = _safe_text_scalar(city_text)
    if 'working from home' in c: return 'Remote_US'
    if 'working from home' not in c and not c.startswith('uk-working from home'): return None
    if 'india' in c: return 'Remote_India'
    if 'uk' in c: return 'Remote_UK'
    if 'ireland' in c: return 'Remote_Ireland'
    if any(country in c for country in ['italy', 'spain', 'netherlands', 'germany', 'france', 'switzerland', 'portugal', 'czechia', 'monaco']): return 'Remote_Europe'
    if any(country in c for country in ['japan', 'singapore']): return 'Remote_APAC'
    match = re.search(r'\s*([a-z]{2})$', c)
    if match and match.group(1) in US_STATE_CODES: return 'Remote_US'
    return 'Remote_Other'

def _city_zone(city):
    if _is_series(city):
        normalized = _safe_text_series(city)
        remote = normalized.apply(_remote_zone)
        mapped = normalized.map(city_to_zone).fillna('Other')
        final_zone = remote.where(remote.notna(), mapped)
        return final_zone.apply(_reduce_city_zone)
    remote_value = _remote_zone(city)
    if remote_value is not None: 
        return _reduce_city_zone(remote_value)
    mapped = city_to_zone.get(_safe_text_scalar(city), 'Other')
    return _reduce_city_zone(mapped)

### OHE
def _one_hot_with_prefix(series, prefix, expected_columns=None):
    if not _is_series(series):
        series = pd.Series([series])
    dummies = pd.get_dummies(series.fillna('Other').astype(str), prefix=prefix, dtype=int)
    if expected_columns is not None:
        for col in expected_columns:
            if col not in dummies.columns: 
                dummies[col] = 0
        dummies = dummies[expected_columns]
    return dummies

def _align_to_feature_columns(df, feature_columns):
    df = df.copy()
    for col in feature_columns:
        if col not in df.columns: 
            df[col] = 0
    extra_cols = [col for col in df.columns if col not in feature_columns]
    if extra_cols: 
        df = df.drop(columns=extra_cols)
    return df[feature_columns]

### History
def _calculate_historical_behavior(df, user_col='usertags-BRID', date_col='senttimestamp', target_col='target', span=3):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([user_col, date_col])
    df['did_click'] = (df[target_col] == 1).astype(int)
    df['did_report'] = (df[target_col] == 2).astype(int)
    df['did_no_action'] = (df[target_col] == 0).astype(int)
    df['past_click_rate_ema'] = df.groupby(user_col)['did_click'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['past_report_rate_ema'] = df.groupby(user_col)['did_report'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['past_no_action_rate_ema'] = df.groupby(user_col)['did_no_action'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['past_total_events'] = df.groupby(user_col).cumcount()
    history_cols = ['past_click_rate_ema', 'past_report_rate_ema', 'past_no_action_rate_ema']
    df[history_cols] = df[history_cols].fillna(0)
    return df.drop(columns=['did_click', 'did_report', 'did_no_action'])

# FE
def apply_feature_engineering(df, calculate_history=False):
    df = df.copy()
    
    # Target
    if 'eventtype' in df.columns: 
        df['target'] = _map_target(df['eventtype'])
    if 'target' not in df.columns: 
        df['target'] = -1
    df['target'] = df['target'].fillna(-1).astype(int)
    
    # Tenure
    if 'localHireRehireDate' in df.columns:
        sent_time = df['senttimestamp'] if 'senttimestamp' in df.columns else None
        df['TimeInCompany_years'] = _calculate_tenure(df['localHireRehireDate'], sent_time)
        
    # Time features
    if 'senttimestamp' in df.columns:
        time_features = _extract_time_features(df['senttimestamp'])
        for col, values in time_features.items(): 
            df[col] = values
            
    # Email semantic features
    template_name = (df['templatename'] if 'templatename' in df.columns else pd.Series([''] * len(df), index=df.index))
    template_subject = (df['templatesubject'] if 'templatesubject' in df.columns else pd.Series([''] * len(df), index=df.index))
    email_features = _classify_email(template_name, template_subject)
    for col, values in email_features.items(): 
        df[col] = values
        
    # Business fallback columns
    business_cols = ['usertags-Department', 'businessarea1', 'businessarea2', 'businessarea3', 'businessarea4', 'businessarea5']
    for col in business_cols:
        if col not in df.columns: 
            df[col] = ''
            
    # Business semantic features
    business_features = _extract_business_features(df['usertags-Department'], df['businessarea1'], df['businessarea2'], df['businessarea3'], df['businessarea4'], df['businessarea5'])
    for col, values in business_features.items(): 
        df[col] = values
        
    # HUGS
    if 'is_hugs' in df.columns: 
        df['is_hugs_mapped'] = _hugs_map(df['is_hugs'])
    else: 
        df['is_hugs_mapped'] = 0
        
    # Corporate grade
    if 'corporate_grade' in df.columns: 
        df['corporate_grade_encoded'] = _grade_map(df['corporate_grade'])
    else: 
        df['corporate_grade_encoded'] = -1
        
    # COO Area
    if 'COO_Area' in df.columns: 
        df['coo_area_grouped'] = _coo_area_group(df['COO_Area'])
    else: 
        df['coo_area_grouped'] = 'Other'
    df['coo_area_encoded'] = (df["coo_area_grouped"].fillna("Other").map(COO_AREA_LABEL_MAP).fillna(COO_AREA_LABEL_MAP["Other"])).astype(int)
    
    # City zone
    if 'city' in df.columns: 
        df['city_zone'] = _city_zone(df['city'])
    else: 
        df['city_zone'] = 'Other'
    df['city_zone_encoded'] = (df["city_zone"].fillna("Other").map(CITY_ZONE_LABEL_MAP).fillna(CITY_ZONE_LABEL_MAP["Other"])).astype(int)
    
    # Historical behaviour
    if calculate_history: 
        df = _calculate_historical_behavior(df)
        
    return df

def _save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f: 
        pickle.dump(model, f)

def _load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _debug_similarity(text, categories):
    text_embedding = _get_embeddings([text])[0]
    category_embeddings = _get_category_embeddings(categories)
    similarities = (category_embeddings @ text_embedding)
    for cat, score in zip(categories.keys(), similarities):
        print(f"{cat}: {score:.4f}")

# Recommendation engine moved to phishing_pandas_analytics.py

if __name__ == "__main__":
    print(_classify_semantic_text("cyber security engineering team", BUSINESS_CATEGORIES))
    _debug_similarity("cyber security engineering team", BUSINESS_CATEGORIES)
    _debug_similarity("reset your password immediately", EMAIL_CATEGORIES)
    print(_classify_semantic_text("reset your password immediately", EMAIL_CATEGORIES))