# classification_utils.py

import os
import json
import numpy as np
from openai import AzureOpenAI
from typing import List, Dict

from translation_utils import translate_if_needed
from fewshot_classifier import classify_few_shot, FEW_SHOT_LABELS

# Azure client setup
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
UNDEFINED_WORKLOAD = "Undefined"


def get_embeddings(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536))
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=float)
    except Exception:
        return np.zeros((len(texts), 1536))


def classify_workloads(questions: List[str]) -> List[Dict]:
    # Wrap each question with original and translated text
    items = [{"original": q, "translated": translate_if_needed(q)} for q in questions]

    fewshot_overrides: Dict[str, str] = {}
    filtered_items = []

    # Few-shot overrides & undefined detection
    for itm in items:
        orig = itm["original"]
        trans = itm["translated"]
        try:
            label = classify_few_shot(trans)
            if label in FEW_SHOT_LABELS:
                fewshot_overrides[orig] = label
            elif len(orig.strip().split()) <= 2:
                fewshot_overrides[orig] = UNDEFINED_WORKLOAD
            else:
                filtered_items.append(itm)
        except Exception:
            filtered_items.append(itm)

    # Load workload → sample mappings
    path = os.path.join(os.path.dirname(__file__), "data", "workload.json")
    try:
        raw = open(path, "rb").read()
        encoding = "utf-16" if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else "utf-8-sig"
        entries = json.loads(raw.decode(encoding))
    except Exception:
        return []

    samples: Dict[str, List[str]] = {}
    for e in entries:
        wl = e["workload"]
        txt = e["customerQuestions"]
        samples.setdefault(wl, []).append(txt)

    workloads = list(samples.keys())
    sample_texts = [t for texts in samples.values() for t in texts]
    sample_labels = [wl for wl, texts in samples.items() for _ in texts]
    sample_embs = get_embeddings(sample_texts)

    # Compute centroids
    centroids = {
        wl: sample_embs[[i for i, lab in enumerate(sample_labels) if lab == wl]].mean(axis=0)
        for wl in workloads
    }

    # Embed filtered questions
    q_embs = get_embeddings([itm["translated"] for itm in filtered_items])

    def cosine(a, b): return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Assign by highest cosine similarity
    grouped_items: Dict[str, List[Dict]] = {wl: [] for wl in workloads + [UNDEFINED_WORKLOAD]}
    for itm, emb in zip(filtered_items, q_embs):
        sims = {wl: cosine(emb, centroids[wl]) for wl in workloads}
        best = max(sims.items(), key=lambda x: x[1])[0]
        grouped_items[best].append(itm)

    result: List[Dict] = []
    total = len(questions)

    # Inject few-shot & undefined buckets
    for label in list(FEW_SHOT_LABELS) + [UNDEFINED_WORKLOAD]:
        group = [itm for itm in items if fewshot_overrides.get(itm["original"]) == label]
        if group:
            pct = (len(group) / total) * 100
            result.append({
                "workload": f"{label} ({len(group)} questions, {pct:.1f}%)",
                "questions": [
                    {"original": itm["original"], "display": itm["translated"]}
                    for itm in group
                ]
            })

    # Add embedding-based groups
    for wl in workloads:
        group = grouped_items.get(wl, [])
        if group:
            pct = (len(group) / total) * 100
            result.append({
                "workload": f"{wl} ({len(group)} questions, {pct:.1f}%)",
                "questions": [
                    {"original": itm["original"], "display": itm["translated"]}
                    for itm in group
                ]
            })

    # Sort by group size
    result.sort(key=lambda x: len(x["questions"]), reverse=True)
    return result
