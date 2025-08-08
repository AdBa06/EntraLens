# main.py

import os
import json
import traceback
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

load_dotenv()

from clustering_utils import cluster_intents
from classification_utils import classify_workloads, client
from error_analysis_utils import analyze_errors

CHAT_COMPLETION_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_intent_cache: List[Dict] = []
_intent_mtime: float = 0.0
_source_map: Dict[str, str] = {}
_workload_cache: List[Dict] = []


class Questions(BaseModel):
    customer_questions: List[str]


class ErrorSummarizeRequest(BaseModel):
    workload: str
    errors: List[str]


@app.on_event("startup")
def preload_intents_and_workloads():
    global _intent_cache, _intent_mtime, _source_map, _workload_cache

    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "synthetic_data.xlsx")
    print("[startup] Loading synthetic_data.xlsx for Modes 1 & 2 at", path)
    if not os.path.exists(path):
        print("[startup]  ✗ synthetic_data.xlsx not found, skipping Modes 1 & 2")
        return

    try:
        df = pd.read_excel(path)
        df2 = df.dropna(subset=["customer_questions", "copilot_source"])
        qs   = df2["customer_questions"].astype(str).tolist()
        srcs = df2["copilot_source"].astype(str).tolist()
        _source_map = dict(zip(qs, srcs))

        _intent_mtime = os.path.getmtime(path)
        print(f"[startup]  • Clustering {len(qs)} questions…")
        raw_clusters = cluster_intents(qs)

        enhanced_clusters = []
        for c in raw_clusters:
            new_qs = []
            for q in c["questions"]:
                q["source"] = _source_map.get(q["original"], "unknown")
                new_qs.append(q)
            c["questions"] = new_qs
            enhanced_clusters.append(c)
        _intent_cache = enhanced_clusters
        print(f"[startup]  ✓ Mode 1: {len(_intent_cache)} clusters cached")

        print(f"[startup]  • Workload-grouping on same data…")
        raw_wl = classify_workloads(qs)

        enhanced_wl = []
        for g in raw_wl:
            new_qs = []
            for q in g["questions"]:
                q["source"] = _source_map.get(q["original"], "unknown")
                new_qs.append(q)
            g["questions"] = new_qs
            enhanced_wl.append(g)
        _workload_cache = enhanced_wl
        print(f"[startup]  ✓ Mode 2: {len(_workload_cache)} workload groups cached")

    except Exception:
        print(f"[startup]  ✗ Failed preload:")
        traceback.print_exc()


@app.get("/api/intent-analysis/info")
def intent_analysis_info():
    return {"file_mtime": _intent_mtime, "cached": bool(_intent_cache)}


@app.post("/api/intent-analysis")
def intent_analysis(q: Questions):
    try:
        clusters = cluster_intents(q.customer_questions)
        return clusters
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, detail="Clustering failed")


@app.get("/api/intent-analysis/default")
def intent_analysis_default():
    if not _intent_cache:
        raise HTTPException(500, detail="Intent cache not populated on startup")
    return _intent_cache


@app.post("/api/workload-grouping")
def workload_grouping(q: Questions):
    return classify_workloads(q.customer_questions)


@app.get("/api/workload-grouping/default")
def workload_grouping_default():
    if not _workload_cache:
        raise HTTPException(500, detail="Workload groups not preloaded")
    return _workload_cache


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
    base = os.path.dirname(__file__)
    err_path = os.path.join(base, "data", "errors.xlsx")
    return {"analysis": analyze_errors(q.customer_questions, err_path)}


@app.post("/api/error-summarize")
def error_summarize(req: ErrorSummarizeRequest):
    """
    Ask the LLM to assign each raw error message a short label, returning
    a JSON array of labels of the same length as the input list.
    Counting is done in code.
    """
    prompt = (
        "You are an expert at categorizing error messages.\n"
        f"Workload: {req.workload}\n\n"
        "Here are the raw errors (one per line):\n"
        + "\n".join(req.errors)
        + "\n\n"
        "Please return ONLY a JSON array of short labels (max 5 words each), "
        "with one label for each error in the same order as input. "
        "Do NOT include counts or any extra text.\n"
        'Example: ["Timeout", "Null reference", "Timeout", "Placeholder missing"]'
    )

    messages = [
        {"role": "system", "content": "You are a JSON-only assistant."},
        {"role": "user",   "content": prompt}
    ]

    try:
        resp = client.chat.completions.create(
            model=CHAT_COMPLETION_MODEL,
            messages=messages
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.lstrip("```json").rstrip("```").strip()
        labels = json.loads(raw)
        if not isinstance(labels, list):
            raise ValueError("Expected a JSON array")
    except Exception:
        labels = []

    return {"labels": labels}
