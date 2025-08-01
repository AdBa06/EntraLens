# main.py

import os
import json
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# ── Load env vars ────────────────────────────────────────────────────────────
load_dotenv()

# ── Imports ──────────────────────────────────────────────────────────────────
from clustering_utils import cluster_intents
from classification_utils import classify_workloads, client
from error_analysis_utils import analyze_errors

# Chat‐completion deployment name for Mode 4
CHAT_COMPLETION_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory caches ─────────────────────────────────────────────────────────
_intent_cache: List[Dict] = []
_intent_mtime: float = 0.0

_workload_cache: List[Dict] = []

# ── Pydantic models ─────────────────────────────────────────────────────────
class Questions(BaseModel):
    customer_questions: List[str]

class ErrorSummarizeRequest(BaseModel):
    workload: str
    errors: List[str]

# ── Pre-load Modes 1 & 2 ─────────────────────────────────────────────────────
@app.on_event("startup")
def preload_intents_and_workloads():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "synthetic_data.xlsx")
    print("[startup] Loading synthetic_data.xlsx for Modes 1 & 2 at", path)
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            qs = df["customer_questions"].dropna().astype(str).tolist()
            global _intent_mtime, _intent_cache, _workload_cache
            _intent_mtime = os.path.getmtime(path)
            print(f"[startup]  • Clustering {len(qs)} questions…")
            _intent_cache = cluster_intents(qs)
            print(f"[startup]  ✓ Mode 1: {_intent_cache.__len__()} clusters cached")
            print(f"[startup]  • Workload-grouping on same data…")
            _workload_cache = classify_workloads(qs)
            print(f"[startup]  ✓ Mode 2: {_workload_cache.__len__()} workload groups cached")
        except Exception as e:
            print(f"[startup]  ✗ Failed preload: {e}")
    else:
        print("[startup]  ✗ synthetic_data.xlsx not found, skipping Modes 1 & 2")

# ── Mode 1: Intent Analysis ─────────────────────────────────────────────────
@app.get("/api/intent-analysis/info")
def intent_analysis_info():
    return {"file_mtime": _intent_mtime, "cached": bool(_intent_cache)}

@app.post("/api/intent-analysis")
def intent_analysis(q: Questions):
    return cluster_intents(q.customer_questions)

@app.get("/api/intent-analysis/default")
def intent_analysis_default():
    if not _intent_cache:
        raise HTTPException(500, "Intent clusters not preloaded")
    return _intent_cache

# ── Mode 2: Workload Grouping ────────────────────────────────────────────────
@app.post("/api/workload-grouping")
def workload_grouping(q: Questions):
    return classify_workloads(q.customer_questions)

@app.get("/api/workload-grouping/default")
def workload_grouping_default():
    if not _workload_cache:
        raise HTTPException(500, "Workload groups not preloaded")
    return _workload_cache

# ── Mode 3: Error Grouping (precomputed JSON) ────────────────────────────────
@app.get("/api/error-grouping/default")
def error_grouping_default():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "error_results.json")
    if not os.path.exists(path):
        raise HTTPException(500, f"{path} not found; run pipeline first.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(500, f"Failed to load error_results.json: {e}")

@app.post("/api/error-grouping")
def error_grouping(q: Questions):
    # ad-hoc (raw Excel) grouping if needed
    base = os.path.dirname(__file__)
    err_path = os.path.join(base, "data", "errors.xlsx")
    return {"analysis": analyze_errors(q.customer_questions, err_path)}

# ── Mode 4: Error Summarization ──────────────────────────────────────────────
@app.post("/api/error-summarize")
def error_summarize(req: ErrorSummarizeRequest):
    """
    Generate a concise summary for the given workload’s errors on demand.
    """
    # Build the prompt
    prompt = (
        f"You are an expert at identifying themes in error logs.\n"
        f"Workload: {req.workload}\n"
        "Errors:\n" +
        "\n".join(f"- {e}" for e in req.errors) +
        "\n\nPlease provide a concise paragraph summarizing the common issues."
    )
    messages = [
        {"role": "system", "content": "Summarize error patterns for Azure workloads."},
        {"role":   "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_COMPLETION_MODEL,
            messages=messages
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Error generating summary: {e}"
    return {"summary": summary}
