# llm_server.py
# 
# # Freezed
# 
# Core Identity

# Semantic Router & Argument Generator: It acts purely as the brain of the operation. It decides which tools to use and how to use them, but it never executes the tools itself.

# Primary Responsibilities & Logic Paths

# Hard-Rule Bypass: Instantly intercepts system-level queries (e.g., "cache status," "health check") and routes them directly, completely skipping the AI/semantic engine to save time.

# Logic 1 (Batch Routing for noLLM.py): Uses semantic search to find all tools that pass a specific similarity threshold. It then prompts the LLM to generate JSON arguments for all matched tools simultaneously so the UI can execute them in parallel.

# Logic 2 (Precision Routing for UI.py): Uses semantic search to find the single best Top-1 tool match and generates a precise JSON argument payload for it.

# Logic 3 (AI Summarization): Provides an optional /summarize endpoint that takes raw JSON tool outputs (after the UI executes them) and translates them into human-readable, behavioral security explanations.

# Data & Security Management

# Schema Enforcement: Forces the LLM to strictly adhere to the TOOL_CATALOG schemas to prevent hallucinations when generating arguments.

# Role-Based Masking (RBAC): Automatically masks sensitive identifiers (e.g., replacing real IDs with hidden_user_mode) depending on whether the requester is an admin or a standard user.

# Dual Model Orchestration: Manages the handoff between the Sentence Transformer (for fast semantic vector matching) and the local LLM/Ollama (for argument generation and text summarization).

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_NAME = os.getenv("APP_NAME", "phishing_llm_router")
MODEL_PATH = os.getenv("MODEL_PATH", "")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", "")
USE_LLM = str(os.getenv("USE_LLM", "true")).strip().lower() in {"1", "true", "yes", "y"}
LOAD_LLM_ON_STARTUP = str(os.getenv("LOAD_LLM_ON_STARTUP", "false")).strip().lower() in {"1", "true", "yes", "y"}
LOCAL_LLM_LOCAL_FILES_ONLY = str(os.getenv("LOCAL_LLM_LOCAL_FILES_ONLY", "false")).strip().lower() in {"1", "true", "yes", "y"}
MAX_NEW_TOKENS_ROUTER = int(os.getenv("MAX_NEW_TOKENS_ROUTER", "200"))
MAX_NEW_TOKENS_SUMMARY = int(os.getenv("MAX_NEW_TOKENS_SUMMARY", "200"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "20"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "10"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.60"))
MIN_PREDICTION_PAYLOAD_FIELDS = int(os.getenv("MIN_PREDICTION_PAYLOAD_FIELDS", "1"))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

LLM_TOKENIZER = None
LLM_MODEL = None
SEMANTIC_MODEL = None
TOOL_EMBEDDINGS = None

app = FastAPI()

TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "run_analytics": {
        "description": "Historical aggregate phishing analytics: counts, percentages, rates, trends, grouped summaries.",
        "modes": [],
        "keywords": ["percentage", "percent", "count", "rate", "trend", "city", "department", "campaign", "summary", "group", "aggregate"],
        "schema": {"analysis_type": "str", "group_by": "list[str]", "filters": "dict", "user_role": "str", "top_n": "int"},
        "mode_guidance": [
            "Use for historical aggregate metrics, counts, percentages, rates, trends, and grouped summaries.",
            "Use when the question asks about city, department, campaign, template, subject, month, year, designation, grade, or overall metrics.",
            "Do not use for ML prediction or future probability questions.",
            "Do not use for employee profile lookup unless the question asks aggregate BRID-level historical analytics."
        ]
    },
    "employee_lookup": {
        "description": "Employee historical lookup: profile, find employees, top historically risky employees. Not ML prediction.",
        "modes": ["profile", "top_risky", "find"],
        "keywords": ["profile", "employee", "employees", "brid", "find", "lookup", "history", "top risky", "historically risky", "historical high risk", "high risk user", "risky employees", "employee record", "employee lookup"],
        "schema": {"mode": "profile|top_risky|find", "brid": "str|null", "city": "str|null", "department": "str|null", "limit": "int", "user_role": "str"},
        "mode_guidance": [
            "Use profile when the user asks for a specific BRID profile, user history, or employee historical record.",
            "Use top_risky when the user asks for top risky employees, historically risky employees, or high-risk user history.",
            "Use find when the user asks to find employees by city, department, or lookup filters.",
            "Do not use for ML prediction, probabilities, forecast, likely behaviour, or predicted high-risk users."
        ]
    },
    "predict_risk": {
        "description": "All ML prediction flows: by BRID, from manual fields, recent population, predicted high-risk population.",
        "modes": ["by_brid", "from_payload", "recent_population", "high_risk_population"],
        "keywords": ["predict", "prediction", "probability", "likely", "chance", "forecast", "ml", "model", "no action probability", "clicked probability", "click probability", "reported probability", "predicted high risk", "high risk population", "recent population", "risk prediction", "predict risk"],
        "schema": {"mode": "by_brid|from_payload|recent_population|high_risk_population", "brid": "str|null", "payload": "dict|null", "limit": "int", "threshold": "float|null", "user_role": "str"},
        "mode_guidance": [
            "Use by_brid only when the user provides a real BRID value.",
            "Use from_payload when the user provides prediction fields in natural language instead of BRID.",
            "Use high_risk_population when the user asks for predicted high-risk users or high-risk population.",
            "Use recent_population when the user asks to predict recent population or recent users.",
            "Possible payload keys: city, usertags-Department, corporate_grade, templatesubject, templatename, campaignname, eventtype."
        ]
    },
    "recommend_actions": {
        "description": "Improvement guidance, training suggestions, risk-reduction actions.",
        "modes": ["employee_improvement", "group_recommendations", "overall_recommendations"],
        "keywords": ["improve", "improvement", "recommend", "recommendation", "training", "actions", "action", "reduce clicks", "reduce phishing clicks", "guidance", "what should we do", "action plan", "focus areas", "awareness", "training plan", "next action", "mitigation"],
        "schema": {"mode": "employee_improvement|group_recommendations|overall_recommendations", "brid": "str|null", "group_by": "list[str]", "filters": "dict", "top_n": "int", "user_role": "str"},
        "mode_guidance": [
            "Use employee_improvement when the user asks how a specific BRID or employee can improve.",
            "Use group_recommendations for a department, city, campaign, group, designation, grade, template, or subject.",
            "Use overall_recommendations for general phishing risk reduction or organisation-level guidance."
        ]
    },
    "simulation_users": {
        "description": "Users who clicked, reported, or took no action in a simulation or campaign.",
        "modes": [],
        "keywords": ["users who", "who clicked", "who reported", "trapped", "simulation users", "campaign users", "no action users", "list users", "clicked users", "reported users", "took no action", "who took no action", "campaign list"],
        "schema": {"campaign_month": "str|null", "campaign_year": "str|null", "campaign_name": "str|null", "event_type": "str", "user_role": "str", "limit": "int"},
        "mode_guidance": [
            "Use when the user asks who clicked, who reported, trapped users, no-action users, or campaign user lists.",
            "Do not use for aggregate counts or rates. Use run_analytics for aggregate metrics."
        ]
    },
    "cache_control": {
        "description": "Cache-changing actions: refresh or clear.",
        "modes": ["refresh", "clear"],
        "keywords": ["refresh cache", "clear cache", "reload cache", "update cache", "rebuild cache", "reset cache", "delete cache", "new data added"],
        "schema": {"action": "refresh|clear"},
        "mode_guidance": [
            "Use refresh when the user asks to refresh, reload, update, or rebuild cache.",
            "Use clear when the user asks to clear, reset, or delete cache."
        ]
    },
    "cache_status": {
        "description": "Read cache status or cache statistics.",
        "modes": [],
        "keywords": ["cache status", "cache statistics", "cache stats", "cache loaded", "refresh time", "cache rows", "cache count", "cache info", "is cache loaded"],
        "schema": {"include_statistics": "bool"},
        "mode_guidance": [
            "Use when the user asks to read cache status, cache statistics, refresh time, cache rows, or whether cache is loaded."
        ]
    },
    "system_info": {
        "description": "System diagnostics: health check, database schema, model features, environment/config.",
        "modes": ["health", "schema", "features", "environment"],
        "keywords": ["health", "health check", "schema", "table schema", "columns", "available columns", "feature", "features", "feature columns", "environment", "config", "configuration", "diagnostic", "system", "model path", "model features", "database schema"],
        "schema": {"mode": "health|schema|features|environment"},
        "mode_guidance": [
            "Use health for health check or diagnostics.",
            "Use schema for table schema, database schema, columns, or available columns.",
            "Use features for model features or feature columns.",
            "Use environment for environment, config, model path, or runtime settings."
        ]
    }
}

class SelectToolRequest(BaseModel):
    question: str
    user_role: str = "user"
    top_k: int = 1
    previous_selected_tools: Optional[List[str]] = None

class SelectToolBatchRequest(BaseModel):
    question: str
    user_role: str = "user"
    top_k: int = 3
    similarity_threshold: float = 0.35
    previous_selected_tools: Optional[List[str]] = None

class SummarizeRequest(BaseModel):
    question: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Dict[str, Any]
    user_role: str = "user"

class LLMRequest(BaseModel):
    prompt: str
    system: Optional[str] = None

def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if (np is not None and np.isnan(value)) else value
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(x) for x in value]
    return str(value)

def terminal_json(title: str, data: Any, max_chars: int = 12000) -> None:
    try:
        text = json.dumps(clean_json_value(data), indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(data)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...truncated..."
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("-" * 90, flush=True)
    print(text, flush=True)
    print("-" * 90 + "\n", flush=True)


def log_router_event(event: str, message: str, data: Any = None) -> None:
    payload = clean_json_value(data) if data is not None else {}
    LOGGER.info("%s | %s | %s", event, message, json.dumps(payload, ensure_ascii=False, default=str))


def parse_llm_json(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def load_local_llm():
    global LLM_TOKENIZER, LLM_MODEL
    if LLM_TOKENIZER is not None and LLM_MODEL is not None:
        return LLM_TOKENIZER, LLM_MODEL
    if not USE_LLM:
        return None, None
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        LOGGER.warning("LOCAL_LLM_DEPENDENCIES_UNAVAILABLE")
        return None, None
    try:
        LOGGER.info(f"LOCAL_LLM_LOAD_START | path={MODEL_PATH} | local_files_only={LOCAL_LLM_LOCAL_FILES_ONLY}")
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=LOCAL_LLM_LOCAL_FILES_ONLY)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=LOCAL_LLM_LOCAL_FILES_ONLY,
            low_cpu_mem_usage=True,
        )
        LLM_MODEL.eval()
        if LLM_TOKENIZER.pad_token_id is None:
            LLM_TOKENIZER.pad_token_id = LLM_TOKENIZER.eos_token_id
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
        print(f"MODEL DEVICE MAP: {getattr(LLM_MODEL, 'hf_device_map', str(next(LLM_MODEL.parameters()).device))}", flush=True)
        LOGGER.info("LOCAL_LLM_LOAD_SUCCESS")
        return LLM_TOKENIZER, LLM_MODEL
    except Exception as exc:
        LOGGER.warning(f"LOCAL_LLM_LOAD_FAILED | {str(exc)}")
        return None, None

def call_local_llm(prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_new_tokens: int = 200) -> str:
    if not USE_LLM:
        LOGGER.info("LLM_DISABLED")
        return ""
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        LOGGER.info("LOCAL_LLM_DEPENDENCIES_UNAVAILABLE")
        return ""
    try:
        tok, mdl = load_local_llm()
        if tok is None or mdl is None:
            LOGGER.warning("LOCAL_LLM_NOT_AVAILABLE")
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        encoded = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = encoded["input_ids"].to(mdl.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(mdl.device)
        with torch.inference_mode():
            outputs = mdl.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        LOGGER.info(f"LOCAL_LLM_GENERATION_SUCCESS | chars={len(text)}")
        return text
    except Exception as e:
        LOGGER.exception(f"LOCAL_LLM_GENERATION_FAILED | {str(e)}")
        return ""

call_ollama = call_local_llm

def get_semantic_model():
    global SEMANTIC_MODEL
    if SEMANTIC_MODEL is not None:
        return SEMANTIC_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        model_path = SEMANTIC_MODEL_PATH.strip() if SEMANTIC_MODEL_PATH else "all-MiniLM-L6-v2"
        LOGGER.info(f"SEMANTIC_MODEL_LOAD_START | path={model_path}")
        SEMANTIC_MODEL = SentenceTransformer(model_path)
        LOGGER.info("SEMANTIC_MODEL_LOAD_SUCCESS")
        return SEMANTIC_MODEL
    except Exception as e:
        LOGGER.warning(f"SEMANTIC_MODEL_UNAVAILABLE | {str(e)}")
        SEMANTIC_MODEL = False
        terminal_json("SEMANTIC MODEL FAILED", {"error": str(e)})
        return None

def tool_text(name: str, meta: Dict[str, Any]) -> str:
    return " ".join([name, meta.get("description", ""), " ".join(meta.get("keywords", [])), " ".join(meta.get("mode_guidance", []))])

def keyword_top_1_tool(question: str) -> List[Dict[str, Any]]:
    low = str(question or "").lower()
    ranked = []
    for name, meta in TOOL_CATALOG.items():
        score = 0.0
        for kw in meta.get("keywords", []):
            if str(kw).lower() in low:
                score += 4.0 if " " in str(kw) else 1.0
        ranked.append((name, score))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    name, score = ranked[0]
    result = [{"tool_name": name, "score": round(float(score), 4), "description": TOOL_CATALOG[name].get("description")}]
    terminal_json("KEYWORD FALLBACK TOP 1 TOOL", {"input_query": question, "scoring_mode": "keyword_fallback", "top_1": {"tool_name": name, "score": round(float(score), 4), "description": TOOL_CATALOG[name].get("description")}})
    return result

def semantic_top_1_tool(question: str) -> List[Dict[str, Any]]:
    if np is None:
        return keyword_top_1_tool(question)
    model = get_semantic_model()
    if model is None:
        return keyword_top_1_tool(question)
    try:
        global TOOL_EMBEDDINGS
        if TOOL_EMBEDDINGS is None:
            names = list(TOOL_CATALOG.keys())
            TOOL_EMBEDDINGS = model.encode([tool_text(name, TOOL_CATALOG[name]) for name in names], normalize_embeddings=True)
        q_emb = model.encode([question], normalize_embeddings=True)[0]
        scores = np.dot(TOOL_EMBEDDINGS, q_emb)
        best_index = int(np.argmax(scores))
        best_name = list(TOOL_CATALOG.keys())[best_index]
        best_score = round(float(scores[best_index]), 4)
        result = [{"tool_name": best_name, "score": best_score, "description": TOOL_CATALOG[best_name].get("description")}]
        terminal_json("SEMANTIC TOP 1 TOOL", {"input_query": question, "scoring_mode": "sentence_transformer", "top_1": {"tool_name": best_name, "score": best_score, "description": TOOL_CATALOG[best_name].get("description")}})
        return result
    except Exception as e:
        LOGGER.warning(f"SEMANTIC_SCORING_FAILED | {str(e)}")
        return keyword_top_1_tool(question)


def hard_rule_tool_selection(question: str) -> Optional[Dict[str, Any]]:
    low = str(question or "").lower()
    if any(token in low for token in ["cache status", "cache stats", "cache loaded", "refresh time", "cache rows", "is cache loaded"]):
        return {"tool_name": "cache_status", "score": 1.0, "source": "hard_rule"}
    if any(token in low for token in ["refresh cache", "clear cache", "reload cache", "reset cache", "delete cache", "rebuild cache"]):
        return {"tool_name": "cache_control", "score": 1.0, "source": "hard_rule"}
    if any(token in low for token in ["health", "health check", "schema", "table schema", "columns", "feature", "features", "environment", "config", "diagnostic"]):
        return {"tool_name": "system_info", "score": 1.0, "source": "hard_rule"}
    return None


def rank_tool_candidates(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    low = str(question or "").lower()
    ranked: List[Dict[str, Any]] = []
    for name, meta in TOOL_CATALOG.items():
        keyword_score = 0.0
        for kw in meta.get("keywords", []):
            token = str(kw).lower()
            if token in low:
                keyword_score += 4.0 if " " in token else 1.0
        ranked.append({
            "tool_name": name,
            "keyword_score": round(float(keyword_score), 4),
            "semantic_score": 0.0,
            "final_score": round(float(keyword_score), 4),
            "description": meta.get("description", "")
        })
    if np is None:
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[: max(1, int(top_k))] if top_k is not None else ranked
    model = get_semantic_model()
    if model is not None:
        try:
            global TOOL_EMBEDDINGS
            names = [item["tool_name"] for item in ranked]
            if TOOL_EMBEDDINGS is None:
                tool_texts = [tool_text(name, TOOL_CATALOG[name]) for name in names]
                TOOL_EMBEDDINGS = model.encode(tool_texts, normalize_embeddings=True)
            q_emb = model.encode([question], normalize_embeddings=True)[0]
            scores = np.dot(TOOL_EMBEDDINGS, q_emb)
            for idx, item in enumerate(ranked):
                item["semantic_score"] = round(float(scores[idx]), 4)
                item["final_score"] = round(float(item["keyword_score"] + item["semantic_score"] * 0.35), 4)
        except Exception as exc:
            LOGGER.warning(f"TOOL_RANKING_FALLBACK | {str(exc)}")
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[: max(1, int(top_k))] if top_k is not None else ranked


def rank_all_tool_candidates(question: str) -> List[Dict[str, Any]]:
    return rank_tool_candidates(question, top_k=None)


def orchestrate_tool_selection(question: str, user_role: str, previous_selected_tools: Optional[List[str]] = None, top_k: int = 1) -> Dict[str, Any]:
    hard_rule = hard_rule_tool_selection(question)
    if hard_rule is not None:
        return {
            "selected_tool": hard_rule["tool_name"],
            "mode": None,
            "args": {},
            "selection_mode": "hard_rule",
            "source": hard_rule["source"],
            "candidates": [hard_rule],
            "requires_clarification": False,
            "clarification_message": None,
        }
    effective_top_k = max(1, int(top_k or 1))
    if previous_selected_tools:
        previous_set = {str(tool).strip() for tool in previous_selected_tools if str(tool).strip()}
        ranked = [item for item in rank_tool_candidates(question, top_k=max(effective_top_k, len(previous_set))) if item["tool_name"] in previous_set]
        if ranked:
            selected = ranked[0]
            return {
                "selected_tool": selected["tool_name"],
                "mode": None,
                "args": {},
                "selection_mode": "conversational_fallback",
                "source": "previous_selected_tools",
                "candidates": ranked,
                "requires_clarification": False,
                "clarification_message": None,
            }
    ranked = rank_tool_candidates(question, top_k=effective_top_k)
    selected = ranked[0]
    requires_clarification = False
    clarification_message = None
    if len(ranked) > 1 and abs(ranked[0]["final_score"] - ranked[1]["final_score"]) < 0.1:
        requires_clarification = True
        clarification_message = "The request could match multiple tools. Please refine it so I can choose the right one."
    return {
        "selected_tool": selected["tool_name"],
        "mode": None,
        "args": {},
        "selection_mode": "semantic_top_k" if effective_top_k > 1 else ("semantic_top_1" if not requires_clarification else "semantic_top_1_ask_user"),
        "source": "semantic_ranking",
        "candidates": ranked,
        "requires_clarification": requires_clarification,
        "clarification_message": clarification_message,
    }


def orchestrate_tool_batch_selection(question: str, user_role: str, previous_selected_tools: Optional[List[str]] = None, top_k: int = 3, similarity_threshold: float = 0.35) -> Dict[str, Any]:
    hard_rule = hard_rule_tool_selection(question)
    if hard_rule is not None:
        return {
            "selected_tools": [hard_rule],
            "selection_mode": "hard_rule",
            "source": hard_rule["source"],
            "requires_clarification": False,
            "clarification_message": None,
            "threshold": similarity_threshold,
        }
    effective_top_k = max(1, int(top_k or 3))
    ranked = rank_tool_candidates(question, top_k=max(effective_top_k, len(TOOL_CATALOG)))
    if previous_selected_tools:
        previous_set = {str(tool).strip() for tool in previous_selected_tools if str(tool).strip()}
        filtered = [item for item in ranked if item["tool_name"] in previous_set]
        if filtered:
            ranked = filtered
    candidates = [item for item in ranked if item.get("final_score", 0.0) >= float(similarity_threshold)]
    if not candidates:
        candidates = ranked[:max(1, effective_top_k)]
    return {
        "selected_tools": candidates,
        "selection_mode": "semantic_batch",
        "source": "semantic_ranking",
        "requires_clarification": False,
        "clarification_message": None,
        "threshold": similarity_threshold,
    }


def build_batch_tool_argument_prompt(question: str, user_role: str, selected_tools: List[Dict[str, Any]]) -> str:
    payload = [{
        "tool_name": item["tool_name"],
        "description": item.get("description", ""),
        "allowed_modes": TOOL_CATALOG.get(item["tool_name"], {}).get("modes", []),
        "arguments_schema": TOOL_CATALOG.get(item["tool_name"], {}).get("schema", {}),
        "mode_guidance": TOOL_CATALOG.get(item["tool_name"], {}).get("mode_guidance", [])
    } for item in selected_tools]
    return f"""User role: {user_role}
User question: {question}
Selected MCP tools:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Task: Generate JSON arguments for each selected tool in the exact order provided.
Strict rules:
- Return only valid JSON.
- Do not use markdown.
- Do not explain.
- Do not choose a different tool.
- Do not include extra fields beyond tool_name, mode, args.
- Extract tool arguments from the user question.
- Use null for missing values.
- Do not invent values.
- Do not copy placeholder/example values.
- If the user gives BRID, copy the exact BRID from the question.
- If the user gives month/year/city/department/campaign/template/subject, place them in the correct schema field.
- If the selected tool uses filters, put extracted filter values inside filters.
- For run_analytics, decide analysis_type, group_by, filters, and top_n.
- For predict_risk, decide mode. Use by_brid only when BRID exists. Use from_payload when prediction fields are given without BRID.
- For recommend_actions, decide the best mode based on the question.
- For simulation_users, decide event_type and campaign fields if present.
- For cache/system tools, return the matching action or mode.
- If required values are missing, still return best possible JSON and use null.

Return JSON exactly in this shape:
{{
  "tools": [
    {{"tool_name": "<tool_name>", "mode": <mode_or_null>, "args": {{...}}}}
  ]
}}
""".strip()


def parse_llm_tool_payloads(raw: str) -> List[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tools"), list):
            return [item for item in parsed["tools"] if isinstance(item, dict)]
        if isinstance(parsed.get("tool_payloads"), list):
            return [item for item in parsed["tool_payloads"] if isinstance(item, dict)]
        if "tool_name" in parsed:
            return [parsed]
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def llm_generate_arguments_for_tool_batch(question: str, user_role: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    prompt = build_batch_tool_argument_prompt(question, user_role, candidates)
    terminal_json("LLM BATCH ARGUMENT GENERATION INPUT", {"question": question, "user_role": user_role, "selected_tools": [c["tool_name"] for c in candidates]})
    raw = call_ollama(prompt, system="Return only valid JSON. No markdown. No explanation.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_ROUTER)
    terminal_json("LLM BATCH ARGUMENT GENERATION RAW RESPONSE", raw)
    parsed_payloads = parse_llm_tool_payloads(raw)
    terminal_json("LLM BATCH ARGUMENT GENERATION PARSED RESPONSE", parsed_payloads)
    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        tool_name = candidate["tool_name"]
        default_entry = {"tool_name": tool_name, "mode": None, "args": {}, "source": "semantic_batch_default"}
        matched = next((item for item in parsed_payloads if item.get("tool_name") == tool_name), None)
        if matched is None:
            results.append(default_entry)
            continue
        mode = matched.get("mode") if matched.get("mode") is not None else None
        args = matched.get("args") if isinstance(matched.get("args"), dict) else {}
        normalized = normalize_args_by_schema(tool_name, mode, args, user_role)
        normalized["tool_name"] = tool_name
        normalized["source"] = "semantic_batch_llm_args" if raw else "semantic_batch_default"
        results.append(normalized)
    if not results:
        return [
            {"tool_name": c["tool_name"], "mode": None, "args": {}, "source": "semantic_batch_default"}
            for c in candidates
        ]
    return results


def compact_tool_catalog() -> List[Dict[str, Any]]:
    return [{"tool_name": name, "description": meta.get("description", ""), "modes": meta.get("modes", []), "schema": meta.get("schema", {}), "mode_guidance": meta.get("mode_guidance", [])} for name, meta in TOOL_CATALOG.items()]

def build_single_tool_argument_prompt(question: str, user_role: str, selected_tool: Dict[str, Any]) -> str:
    payload = {
        "tool_name": selected_tool["tool_name"],
        "description": selected_tool.get("description", ""),
        "allowed_modes": selected_tool.get("modes", []),
        "arguments_schema": selected_tool.get("schema", {}),
        "mode_guidance": selected_tool.get("mode_guidance", [])
    }
    return f"""User role: {user_role}
User question: {question}
Selected MCP tool:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Task: Generate JSON arguments for the already-selected tool.
Strict rules:
- Return only valid JSON.
- Do not use markdown.
- Do not explain.
- Do not choose a different tool.
- Do not include selected_tool unless asked in the exact output shape.
- Extract tool arguments from the user question.
- Use null for missing values.
- Do not invent values.
- Do not copy placeholder/example values.
- If the user gives BRID, copy the exact BRID from the question.
- If the user gives month/year/city/department/campaign/template/subject, place them in the correct schema field.
- If the selected tool uses filters, put extracted filter values inside filters.
- For run_analytics, decide analysis_type, group_by, filters, and top_n.
- For predict_risk, decide mode. Use by_brid only when BRID exists. Use from_payload when prediction fields are given without BRID.
- For recommend_actions, decide the best mode based on the question.
- For simulation_users, decide event_type and campaign fields if present.
- For cache/system tools, return the matching action or mode.
- If required values are missing, still return best possible JSON and use null.

Return JSON exactly in this shape: {{ "mode": null, "args": {{}} }}
""".strip()

def llm_generate_arguments_for_tool(question: str, user_role: str, selected_tool: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_single_tool_argument_prompt(question, user_role, selected_tool)
    terminal_json("LLM ARGUMENT GENERATION INPUT", {"question": question, "user_role": user_role, "selected_tool": selected_tool.get("tool_name")})
    raw = call_ollama(prompt, system="Return only valid JSON. No markdown. No explanation.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_ROUTER)
    terminal_json("LLM ARGUMENT GENERATION RAW RESPONSE", raw)
    parsed = parse_llm_json(raw)
    terminal_json("LLM ARGUMENT GENERATION PARSED RESPONSE", parsed)
    return clean_json_value({
        "selected_tool": selected_tool["tool_name"],
        "mode": parsed.get("mode") if isinstance(parsed, dict) else None,
        "args": parsed.get("args") if isinstance(parsed.get("args"), dict) else {},
        "source": "semantic_top_1_plus_llm_args",
        "llm_raw": raw[:800] if raw else ""
    })

def normalize_args_by_schema(tool: str, mode: Any, args: Dict[str, Any], user_role: str) -> Dict[str, Any]:
    args = dict(args or {})
    schema = TOOL_CATALOG.get(tool, {}).get("schema", {})
    if "user_role" in schema:
        args["user_role"] = user_role
    if tool == "run_analytics":
        args["analysis_type"] = args.get("analysis_type") or "overall_analysis"
        args["group_by"] = args.get("group_by") if isinstance(args.get("group_by"), list) else ([] if args.get("group_by") is None else [args.get("group_by")])
        args["filters"] = args.get("filters") if isinstance(args.get("filters"), dict) else {}
        args["top_n"] = int(args.get("top_n") or DEFAULT_TOP_N)
    elif tool == "employee_lookup":
        args["mode"] = str(mode or args.get("mode") or "").lower() or None
        args["limit"] = int(args.get("limit") or DEFAULT_LIMIT)
    elif tool == "predict_risk":
        args["mode"] = str(mode or args.get("mode") or "").lower() or None
        args["payload"] = args.get("payload") if isinstance(args.get("payload"), dict) else {}
        args["limit"] = int(args.get("limit")) if args.get("limit") else None
        args["threshold"] = float(args.get("threshold")) if args.get("threshold") else None
    elif tool == "recommend_actions":
        args["mode"] = str(mode or args.get("mode") or "").lower() or None
        args["group_by"] = args.get("group_by") if isinstance(args.get("group_by"), list) else ([] if args.get("group_by") is None else [args.get("group_by")])
        args["filters"] = args.get("filters") if isinstance(args.get("filters"), dict) else {}
        args["top_n"] = int(args.get("top_n") or 5)
    elif tool == "simulation_users":
        args["limit"] = int(args.get("limit") or 5000)
    elif tool == "cache_control":
        args["action"] = str(args.get("action") or "").lower() or "refresh"
    elif tool == "cache_status":
        args["include_statistics"] = bool(args.get("include_statistics", True))
    elif tool == "system_info":
        args["mode"] = str(mode or args.get("mode") or "").lower() or "health"
    return clean_json_value(args)

def validate_tool_selection(selection: Dict[str, Any], user_role: str) -> Dict[str, Any]:
    tool = selection.get("selected_tool")
    if tool not in TOOL_CATALOG:
        return clean_json_value({"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": f"Unknown tool selected: {tool}", "source": selection.get("source")})
    normalized = normalize_args_by_schema(tool, selection.get("mode"), selection.get("args", {}), user_role)
    mode = normalized["mode"]
    args = normalized["args"]
    missing_fields = []
    if tool == "employee_lookup":
        if mode not in ["profile", "top_risky", "find"]:
            missing_fields.append("mode")
        if mode == "profile" and not args.get("brid"):
            missing_fields.append("brid")
    elif tool == "predict_risk":
        if mode not in ["by_brid", "from_payload", "recent_population", "high_risk_population"]:
            missing_fields.append("mode")
        if mode == "by_brid" and not args.get("brid"):
            missing_fields.append("brid")
        if mode == "from_payload":
            payload = args.get("payload")
            useful_payload = {k: v for k, v in payload.items() if v not in [None, "", [], {}]} if isinstance(payload, dict) else {}
            if len(useful_payload) < MIN_PREDICTION_PAYLOAD_FIELDS:
                missing_fields.append("additional_prediction_fields")
    elif tool == "recommend_actions":
        if mode not in ["employee_improvement", "group_recommendations", "overall_recommendations"]:
            missing_fields.append("mode")
        if mode == "employee_improvement" and not args.get("brid"):
            missing_fields.append("brid")
    elif tool == "simulation_users":
        if not args.get("campaign_month") and not args.get("campaign_year") and not args.get("campaign_name"):
            missing_fields.append("campaign_month/campaign_year/campaign_name")
        if not args.get("event_type"):
            missing_fields.append("event_type")
    elif tool == "cache_control":
        if args.get("action") not in ["refresh", "clear"]:
            missing_fields.append("action")
    elif tool == "system_info":
        if args.get("mode") not in ["health", "schema", "features", "environment"]:
            missing_fields.append("mode")
    selection["mode"] = mode
    selection["args"] = clean_json_value(args)
    selection["validated"] = len(missing_fields) == 0
    if missing_fields:
        selection["validation_error"] = "Missing required field(s): " + ", ".join(missing_fields) + ". The correct tool path may be selected, but more input is required."
    return clean_json_value(selection)

def compact_tool_output(tool_output: Dict[str, Any], max_chars: int = 12000) -> str:
    text = json.dumps(clean_json_value(tool_output), indent=2, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...truncated..."
    return text

def fallback_summary(question: str, tool_name: str, tool_args: Dict[str, Any], tool_output: Dict[str, Any]) -> str:
    if not isinstance(tool_output, dict):
        return "Completed."

    if "ui_summary" in tool_output:
        return tool_output["ui_summary"]

    status = tool_output.get("status", "unknown")
    if status == "error":
        return f"I could not complete this using {tool_name}. Reason: {tool_output.get('message', 'Unknown error')}"

    return f"{tool_name} completed with status: {status}."

def summarize_with_llm(question: str, tool_name: str, tool_args: Dict[str, Any], tool_output: Dict[str, Any], user_role: str = "user") -> Dict[str, Any]:
    mcp_context = json.dumps(clean_json_value(tool_output.get("llm_context", {})), ensure_ascii=False)
    mcp_hint = tool_output.get("llm_prompt_hint", "")
    prompt = f"""You are the explanation layer only. Do not choose tools. Do not invent data. Summarize only the given MCP tool output. Use behavioural security language. Do not judge employee performance. For normal users, do not expose hidden personal identifiers.
User role: {user_role}
Question: {question}
Tool: {tool_name}
Tool args: {json.dumps(clean_json_value(tool_args), ensure_ascii=False, default=str)}
Tool output: {compact_tool_output(tool_output)}

Context Instructions from Backend: {mcp_context}
UI Hint: {mcp_hint}

Write a concise answer with: 1. direct answer 2. important numbers/groups if present 3. recommended next action if present""".strip()
    terminal_json("SUMMARY LLM INPUT TEXT / PROMPT", prompt, max_chars=20000)
    raw = call_ollama(prompt, system="You summarize tool outputs only. Never invent values.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_SUMMARY)
    terminal_json("SUMMARY LLM RAW RESPONSE", raw)
    if raw:
        return {"answer": raw, "used_llm": True, "summary_mode": "local_transformers_llm"}
    return {"answer": fallback_summary(question, tool_name, tool_args, tool_output), "used_llm": False, "summary_mode": "deterministic_fallback"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "architecture": "semantic_top_1_tool -> llm_argument_json -> schema_validation -> llm_summary",
        "tools": list(TOOL_CATALOG.keys()),
        "use_llm": USE_LLM,
        "local_llm_model_path": MODEL_PATH,
        "local_llm_loaded": LLM_MODEL is not None,
        "semantic_model_loaded": SEMANTIC_MODEL is not None and SEMANTIC_MODEL is not False,
        "cuda_available": torch.cuda.is_available() if torch is not None else False
    }

@app.get("/tools")
def get_tools():
    return {"status": "success", "tools": compact_tool_catalog()}

@app.post("/select_tool")
def select_tool(req: SelectToolRequest):
    start = time.time()
    try:
        question = req.question.strip()
        user_role = req.user_role or "user"
        effective_top_k = max(1, int(req.top_k or 1))
        terminal_json("SELECT TOOL API INPUT", {"question": question, "user_role": user_role, "requested_top_k": req.top_k, "effective_top_k": effective_top_k, "previous_selected_tools": req.previous_selected_tools})
        log_router_event("select_tool_start", "Selecting tool for question", {"question": question, "user_role": user_role, "previous_selected_tools": req.previous_selected_tools})
        selection = orchestrate_tool_selection(question, user_role, req.previous_selected_tools, top_k=effective_top_k)
        if not selection.get("selected_tool"):
            response = {"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": "No candidate tool found.", "source": "no_semantic_candidate", "latency_ms": round((time.time() - start) * 1000, 2), "candidates": []}
            terminal_json("SELECT TOOL FINAL RESPONSE", response)
            log_router_event("select_tool_error", "No candidate tool found", {"question": question})
            return clean_json_value(response)
        selected_tool_meta = TOOL_CATALOG.get(selection["selected_tool"], {})
        selected_tool_info = {
            "tool_name": selection["selected_tool"],
            "description": selected_tool_meta.get("description", ""),
            "modes": selected_tool_meta.get("modes", []),
            "schema": selected_tool_meta.get("schema", {}),
            "mode_guidance": selected_tool_meta.get("mode_guidance", []),
        }
        terminal_json("SEMANTIC TOOL SELECTED", {"tool_name": selected_tool_info.get("tool_name"), "selection_mode": selection.get("selection_mode"), "top_k": req.top_k, "candidates": selection.get("candidates")})
        selected = llm_generate_arguments_for_tool(question, user_role, selected_tool_info)
        selected["selection_mode"] = selection.get("selection_mode")
        selected["source"] = selection.get("source")
        selected["requires_clarification"] = selection.get("requires_clarification", False)
        selected["clarification_message"] = selection.get("clarification_message")
        selected["candidates"] = selection.get("candidates", [])
        terminal_json("SELECTED TOOL BEFORE VALIDATION", selected)
        validated = validate_tool_selection(selected, user_role)
        validated["latency_ms"] = round((time.time() - start) * 1000, 2)
        validated["selection_mode"] = selection.get("selection_mode")
        validated["requires_clarification"] = selection.get("requires_clarification", False)
        validated["clarification_message"] = selection.get("clarification_message")
        validated["candidates"] = selection.get("candidates", [])
        terminal_json("SELECT TOOL FINAL VALIDATED RESPONSE", validated)
        log_router_event("select_tool_complete", "Tool selection completed", {"selected_tool": validated.get("selected_tool"), "validated": validated.get("validated"), "selection_mode": validated.get("selection_mode")})
        return clean_json_value(validated)
    except Exception as exc:
        LOGGER.exception("SELECT_TOOL_ROUTE_FAILED | %s", str(exc))
        response = {"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": str(exc), "source": "router_exception", "latency_ms": round((time.time() - start) * 1000, 2), "candidates": []}
        terminal_json("SELECT TOOL FINAL ERROR RESPONSE", response)
        return clean_json_value(response)

@app.post("/select_tool_batch")
def select_tool_batch(req: SelectToolBatchRequest):
    start = time.time()
    try:
        question = req.question.strip()
        user_role = req.user_role or "user"
        effective_top_k = max(1, int(req.top_k or 3))
        threshold = float(req.similarity_threshold or 0.35)
        terminal_json("SELECT TOOL BATCH API INPUT", {"question": question, "user_role": user_role, "top_k": req.top_k, "similarity_threshold": threshold, "previous_selected_tools": req.previous_selected_tools})
        log_router_event("select_tool_batch_start", "Selecting tool batch for question", {"question": question, "user_role": user_role, "previous_selected_tools": req.previous_selected_tools, "threshold": threshold})
        selection = orchestrate_tool_batch_selection(question, user_role, req.previous_selected_tools, top_k=effective_top_k, similarity_threshold=threshold)
        tools = selection.get("selected_tools", [])
        if not tools:
            response = {"selected_tools": [], "validated": False, "validation_error": "No candidate tools found.", "source": "no_semantic_candidate", "latency_ms": round((time.time() - start) * 1000, 2), "candidates": []}
            terminal_json("SELECT TOOL BATCH FINAL RESPONSE", response)
            log_router_event("select_tool_batch_error", "No candidate tools found", {"question": question})
            return clean_json_value(response)
        selected_tool_args = llm_generate_arguments_for_tool_batch(question, user_role, tools)
        response = {
            "selected_tools": selected_tool_args,
            "candidates": tools,
            "selection_mode": selection.get("selection_mode"),
            "source": selection.get("source"),
            "threshold": selection.get("threshold"),
            "latency_ms": round((time.time() - start) * 1000, 2)
        }
        terminal_json("SELECT TOOL BATCH FINAL RESPONSE", response)
        log_router_event("select_tool_batch_complete", "Tool batch selection completed", {"selected_tool_count": len(selected_tool_args), "threshold": threshold})
        return clean_json_value(response)
    except Exception as exc:
        LOGGER.exception("SELECT_TOOL_BATCH_ROUTE_FAILED | %s", str(exc))
        response = {"selected_tools": [], "validated": False, "validation_error": str(exc), "source": "router_exception", "latency_ms": round((time.time() - start) * 1000, 2), "candidates": []}
        terminal_json("SELECT TOOL BATCH FINAL ERROR RESPONSE", response)
        return clean_json_value(response)

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    start = time.time()
    try:
        terminal_json("SUMMARY API INPUT", {"question": req.question, "tool_name": req.tool_name, "tool_args": req.tool_args, "user_role": req.user_role, "tool_output_preview": compact_tool_output(req.tool_output, max_chars=5000)})
        log_router_event("summary_start", "Generating summary for tool output", {"tool_name": req.tool_name, "user_role": req.user_role})
        result = summarize_with_llm(question=req.question, tool_name=req.tool_name, tool_args=req.tool_args, tool_output=req.tool_output, user_role=req.user_role)
        response = {
            "status": "success",
            "final_answer": result["answer"],
            "used_llm": result["used_llm"],
            "summary_mode": result["summary_mode"],
            "latency_ms": round((time.time() - start) * 1000, 2)
        }
        terminal_json("SUMMARY API FINAL RESPONSE", response)
        log_router_event("summary_complete", "Summary generated", {"tool_name": req.tool_name, "summary_mode": result["summary_mode"]})
        return clean_json_value(response)
    except Exception as exc:
        LOGGER.exception("SUMMARY_ROUTE_FAILED | %s", str(exc))
        response = {"status": "error", "final_answer": str(exc), "used_llm": False, "summary_mode": "router_exception", "latency_ms": round((time.time() - start) * 1000, 2)}
        terminal_json("SUMMARY API FINAL ERROR RESPONSE", response)
        return clean_json_value(response)

@app.post("/llm")
def llm(req: LLMRequest):
    try:
        response = call_ollama(req.prompt, system=req.system, temperature=0.0)
        return {"status": "success" if response else "error", "response": response}
    except Exception as exc:
        LOGGER.exception("LLM_ROUTE_FAILED | %s", str(exc))
        return {"status": "error", "response": str(exc)}

ROUTER_TEST_CASES = [
    "percentage of people clicked on January phishing mail",
    "city with most clickers",
    "department wise click rate in 2026",
    "show profile for BRID 75b4ae75-b",
    "top risky employees",
    "find employees in Pune",
    "predict no action probability for BRID 75b4ae75-b",
    "predict for city Pune department Cyber subject password reset campaign January 2026",
    "predict risk for a user from Pune",
    "predict this user risk",
    "predicted high risk users with threshold 60 percent",
    "predict risk for recent population limit 100",
    "how can BRID 75b4ae75-b improve",
    "how can this employee improve",
    "recommend training for risky department",
    "what should we do to reduce phishing clicks",
    "who clicked in March 2026 simulation",
    "who clicked in the simulation",
    "cache status",
    "refresh cache",
    "health check",
    "show table schema",
    "show me risky stuff"
]

def run_router_tests() -> None:
    print("\n" + "=" * 100, flush=True)
    print("LLM ROUTER TEST SUITE START", flush=True)
    print("=" * 100 + "\n", flush=True)
    results = []
    for idx, question in enumerate(ROUTER_TEST_CASES, start=1):
        print("\n" + "-" * 100, flush=True)
        print(f"TEST {idx}/{len(ROUTER_TEST_CASES)}", flush=True)
        print("-" * 100, flush=True)
        print(f"QUESTION: {question}", flush=True)
        try:
            result = select_tool(SelectToolRequest(question=question, user_role="admin", top_k=1))
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        results.append({"question": question, "result": result})
        print("RESULT:", json.dumps(clean_json_value(result), indent=2, ensure_ascii=False, default=str), flush=True)
    print("\n" + "=" * 100, flush=True)
    print("LLM ROUTER TEST SUITE SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(json.dumps(clean_json_value(results), indent=2, ensure_ascii=False, default=str), flush=True)
    print("=" * 100 + "\n", flush=True)

if __name__ == "__main__":
    try:
        import uvicorn
    except Exception:
        print("uvicorn not installed; skipping server launch", flush=True)
        sys.exit(0)
    if "--test-router" in sys.argv:
        run_router_tests()
        sys.exit(0)
    host = os.getenv("LLM_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_SERVER_PORT", "8001"))
    if LOAD_LLM_ON_STARTUP and USE_LLM:
        try:
            load_local_llm()
        except Exception as e:
            LOGGER.exception(f"INITIAL_LOCAL_LLM_LOAD_FAILED | {str(e)}")
            LOGGER.warning("Server will start, but LLM routes may fail until model loads correctly.")
    LOGGER.info(f"LLM_SERVER_START | http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
