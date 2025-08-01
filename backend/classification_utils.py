import os
import json
import numpy as np
from openai import AzureOpenAI
from typing import List, Dict

from fewshot_classifier import classify_few_shot, FEW_SHOT_LABELS

# ── Azure client ────────────────────────────────────────────────────
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
    except Exception as e:
        print(f"[Embedding ERROR] {e}")
        return np.zeros((len(texts), 1536))


def classify_workloads(questions: List[str]) -> List[Dict]:
    fewshot_overrides = {}
    filtered_questions = []

    for q in questions:
        try:
            label = classify_few_shot(q)
            if label in FEW_SHOT_LABELS:
                fewshot_overrides[q] = label
            elif len(q.strip().split()) <= 2:
                fewshot_overrides[q] = UNDEFINED_WORKLOAD
            else:
                filtered_questions.append(q)
        except Exception as e:
            print(f"[FewShot classify error] {e}")
            filtered_questions.append(q)

    # ── Load workload mappings ───────────────────────────────────────
    try:
        path = os.path.join(os.path.dirname(__file__), "data", "workload.json")
        raw = open(path, "rb").read()
        encoding = "utf-16" if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else "utf-8-sig"
        entries = json.loads(raw.decode(encoding))
    except Exception as e:
        print(f"[Workload.json ERROR] {e}")
        return []

    samples: Dict[str, List[str]] = {}
    for e in entries:
        wl = e["workload"]
        txt = e["customerQuestions"]
        samples.setdefault(wl, []).append(txt)

    workloads = list(samples.keys())
    all_workloads = workloads + [UNDEFINED_WORKLOAD]

    sample_texts = [t for texts in samples.values() for t in texts]
    sample_labels = [wl for wl, texts in samples.items() for _ in texts]
    sample_embs = get_embeddings(sample_texts)

    centroids = {
        wl: sample_embs[[i for i, lab in enumerate(sample_labels) if lab == wl]].mean(axis=0)
        for wl in workloads
    }

    q_embs = get_embeddings(filtered_questions)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    grouped: Dict[str, List[str]] = {wl: [] for wl in all_workloads}
    for q, emb in zip(filtered_questions, q_embs):
        sims = {wl: cosine(emb, centroids[wl]) for wl in workloads}
        best_wl, _ = max(sims.items(), key=lambda item: item[1])
        grouped[best_wl].append(q)

    result: List[Dict] = []
    total = len(questions)

    # Inject few-shot labels
    for label in FEW_SHOT_LABELS.union({UNDEFINED_WORKLOAD}):
        qs = [q for q, lbl in fewshot_overrides.items() if lbl == label]
        if qs:
            pct = (len(qs) / total) * 100
            result.append({
                "workload": f"{label} ({len(qs)} questions, {pct:.1f}%)",
                "questions": qs
            })

    for wl in workloads:
        qs = grouped[wl]
        if qs:
            pct = (len(qs) / total) * 100
            result.append({
                "workload": f"{wl} ({len(qs)} questions, {pct:.1f}%)",
                "questions": qs
            })

    result.sort(key=lambda x: len(x["questions"]), reverse=True)
    return result
