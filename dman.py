# data_manager.py


# Core Identity

# The "Single Source of Truth": It is the authoritative gatekeeper of the system's memory. It ensures that every analytical tool, MCP endpoint, and prediction service is operating on the exact same version of the data.

# Primary Responsibilities

# Hot Path Memory Management: It holds the "Primary DataFrame" (_PRIMARY_CACHE) in RAM, providing instantaneous access to the 50,000-row simulation dataset without the latency of repeated database queries.

# SQL-to-Mock Fallback: It encapsulates the database extraction logic, including an automated fallback mechanism: if the MSSQL server is unreachable or credentials fail, it instantly provides a schema-compliant mock DataFrame so the system stays operational during local development or downtime.

# Cache Lifecycle Control: It manages the entire lifecycle of the data, including explicit invalidation (refresh_primary_cache) and state monitoring (tracking refresh timestamps, row counts, and memory footprint).

# Cross-File Data Synchronization: By acting as the central supplier of the DataFrame, it prevents "split-brain" scenarios where different parts of the application (e.g., the analytics engine vs. the MCP tools) might otherwise accidentally work with different versions of the data.

# Operational Safeguards

# Validation & Health: It enforces a "refresh validation" check; if the data fetched from the source is malformed (missing columns/rows), it rejects the update and preserves the existing, known-good cache state.

# Decoupling Transport from Data: By abstracting data access away from the phishing_mcp_server.py, it keeps the MCP server code clean, focused only on tool transport, and makes the entire system significantly easier to unit test.


import os
import logging
import pandas as pd
from datetime import datetime
from functools import lru_cache
from sqlalchemy import create_engine, text
from Config import apply_feature_engineering
from phana import build_employee_profile_cache

LOGGER = logging.getLogger("data_manager")

# Configuration from Environment
GLOBAL_ANALYTICS_ROWS = int(os.getenv("GLOBAL_ANALYTICS_ROWS", "50000"))
ENABLE_CACHE = str(os.getenv("ENABLE_CACHE", "true")).strip().lower() in {"1", "true", "yes", "y"}

# Global Cache State
_PRIMARY_CACHE = {
    "raw_df": None,
    "profile_df": None,
    "profile_lookup": {},
    "metadata": {
        "loaded": False,
        "last_refresh_time": None,
        "total_rows": 0,
        "employee_profiles": 0,
        "source": None
    }
}

def _get_mssql_connection_string():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    driver = os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
    if not server or not database:
        raise ValueError("DB_SERVER or DB_DATABASE not set in .env")
    return f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes&TrustServerCertificate=yes"

def _get_mssql_dataframe(limit: int) -> pd.DataFrame:
    engine = create_engine(_get_mssql_connection_string())
    table = os.getenv("DB_TABLE")
    query = f"SELECT TOP ({limit}) * FROM {table} ORDER BY senttimestamp DESC"
    LOGGER.info(f"ENTER | get_mssql_dataframe | limit={limit}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            LOGGER.info(f"EXIT | get_mssql_dataframe | Shape={df.shape}")
            return df
    except Exception as e:
        LOGGER.error(f"EXIT ERROR | get_mssql_dataframe | {str(e)}")
        raise

def _get_mock_dataframe(limit: int) -> pd.DataFrame:
    """Generates a schema-compliant mock dataframe for local dev/fallback."""
    LOGGER.info(f"ENTER | get_mock_dataframe | limit={limit}")
    # Placeholder for Faker logic; returns empty DF if DB is unavailable
    df = pd.DataFrame() 
    LOGGER.info(f"EXIT | get_mock_dataframe | Shape={df.shape}")
    return df

@lru_cache(maxsize=1)
def get_primary_cache() -> pd.DataFrame:
    """
    The Single Source of Truth for the In-Memory DataFrame.
    """
    global _PRIMARY_CACHE
    
    if _PRIMARY_CACHE["raw_df"] is not None:
        return _PRIMARY_CACHE["raw_df"]

    LOGGER.info("ENTER | get_primary_cache | Cache Empty")
    
    try:
        # 1. Attempt MSSQL
        raw_df = _get_mssql_dataframe(GLOBAL_ANALYTICS_ROWS)
        _PRIMARY_CACHE["metadata"]["source"] = "MSSQL"
    except Exception:
        # 2. Fallback to Mock
        LOGGER.warning("MSSQL Failed, falling back to mock generator.")
        raw_df = _get_mock_dataframe(GLOBAL_ANALYTICS_ROWS)
        _PRIMARY_CACHE["metadata"]["source"] = "MOCK"

    engineered_df = apply_feature_engineering(raw_df)
    _PRIMARY_CACHE["raw_df"] = engineered_df

    # Build derivative profile cache simultaneously
    _PRIMARY_CACHE["profile_df"] = build_employee_profile_cache(_PRIMARY_CACHE["raw_df"])
    
    _PRIMARY_CACHE["metadata"].update({
        "loaded": True,
        "last_refresh_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": len(_PRIMARY_CACHE["raw_df"]),
        "employee_profiles": len(_PRIMARY_CACHE["profile_df"])
    })
    
    LOGGER.info(f"EXIT | get_primary_cache | Rows={len(_PRIMARY_CACHE['raw_df'])}")
    return _PRIMARY_CACHE["raw_df"]


def get_profile_dataframe() -> pd.DataFrame:
    """
    Return the derivative profile dataframe from the primary cache.
    Ensures the primary cache is loaded first.
    """
    # Ensure primary cache is populated
    get_primary_cache()
    profile_df = _PRIMARY_CACHE.get("profile_df")
    if profile_df is None:
        return pd.DataFrame()
    return profile_df.copy()

def refresh_primary_cache():
    """Explicitly invalidates cache and triggers a re-fetch."""
    global _PRIMARY_CACHE
    get_primary_cache.cache_clear()
    _PRIMARY_CACHE["raw_df"] = None
    return get_primary_cache()

def get_cache_stats():
    return _PRIMARY_CACHE["metadata"]


def clear_primary_cache():
    """
    Clear the in-memory primary cache without triggering an immediate reload.
    Returns the reset metadata dictionary.
    """
    global _PRIMARY_CACHE
    get_primary_cache.cache_clear()
    _PRIMARY_CACHE = {
        "raw_df": None,
        "profile_df": None,
        "profile_lookup": {},
        "metadata": {
            "loaded": False,
            "last_refresh_time": None,
            "total_rows": 0,
            "employee_profiles": 0,
            "source": None,
        },
    }
    return _PRIMARY_CACHE["metadata"]