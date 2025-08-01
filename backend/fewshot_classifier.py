from dotenv import load_dotenv
load_dotenv()
import json
import os
import numpy as np
from openai import AzureOpenAI
from typing import List

# ── Few-shot labels ─────────────────────────────────────────────────
FEW_SHOT_LABELS = {"Help with errors"}

# ── OpenAI client setup ─────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# ── Load few-shot examples from JSON ────────────────────────────────
def load_few_shot_examples(path="data/intent_fewshot_examples.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[FewShot] Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"[FewShot] JSON parse error in few-shot examples: {e}")

FEW_SHOT_EXAMPLES = load_few_shot_examples()

# ── Get embeddings with guardrails ──────────────────────────────────
def get_embeddings(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536))  # adjust dim if needed
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        if not resp or not hasattr(resp, "data"):
            raise ValueError("Empty or malformed response from embeddings API.")
        return np.array([d.embedding for d in resp.data])
    except Exception as e:
        print(f"[FewShot:embedding ERROR] {e}")
        raise

# ── Compute class centroids ─────────────────────────────────────────
def compute_centroids() -> dict:
    grouped = {label: [] for label in FEW_SHOT_LABELS}
    for entry in FEW_SHOT_EXAMPLES:
        cat = entry.get("category")
        if cat in grouped:
            grouped[cat].append(entry["question"])
        else:
            print(f"[FewShot:warn] Skipping unknown label: {cat}")

    centroids = {}
    for label, texts in grouped.items():
        if not texts:
            raise ValueError(f"[FewShot] No examples found for category: {label}")
        embs = get_embeddings(texts)
        centroids[label] = np.mean(embs, axis=0)
    return centroids

FEWSHOT_CENTROIDS = compute_centroids()

# ── Cosine similarity ───────────────────────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Main few-shot classifier ────────────────────────────────────────
def classify_few_shot(question: str, threshold: float = 0.85) -> str:
    if not question.strip() or len(question.strip().split()) <= 2:
        return "Undefined"
    try:
        emb = get_embeddings([question])[0]
        similarities = {
            label: cosine(emb, FEWSHOT_CENTROIDS[label]) for label in FEW_SHOT_LABELS
        }
        best_label, best_score = max(similarities.items(), key=lambda x: x[1])
        if best_score >= threshold:
            return best_label
        return "Undefined"
    except Exception as e:
        print(f"[FewShot:classify ERROR] {e}")
        return "Undefined"



# if __name__ == "__main__":
#     print(classify_few_shot("how do I clean up guest users?"))
#     print(classify_few_shot("AADSTS90033: A transient error has occurred."))
#     print(classify_few_shot("what is conditional access?"))  # should fallback
