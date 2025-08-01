# import os
# import numpy as np
# import math
# from openai import AzureOpenAI
# from hdbscan import HDBSCAN
# from sklearn.cluster import KMeans
# from typing import List, Dict

# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )

# EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
# CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# def get_embeddings(texts: List[str]) -> np.ndarray:
#     resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
#     return np.array([d.embedding for d in resp.data])


# def enforce_max_cluster_size(
#     labels: np.ndarray, embeddings: np.ndarray, max_size: int
# ) -> np.ndarray:
#     """
#     Recursively split any cluster whose size > max_size
#     into sub-clusters until every cluster ≤ max_size.
#     """
#     labels = labels.copy()
#     next_label = labels.max() + 1

#     while True:
#         # find any oversized cluster
#         oversized = None
#         for cluster in set(labels):
#             if cluster == -1:
#                 continue
#             members = np.where(labels == cluster)[0]
#             if len(members) > max_size:
#                 oversized = (cluster, members)
#                 break

#         # if none found, we’re done
#         if not oversized:
#             break

#         # otherwise split it
#         _, members = oversized
#         k = math.ceil(len(members) / max_size)
#         sub_emb = embeddings[members]
#         sub_labels = KMeans(n_clusters=k).fit_predict(sub_emb)

#         # reassign those members to new, unique labels
#         for idx_in_list, sub_l in zip(members, sub_labels):
#             labels[idx_in_list] = next_label + sub_l
#         next_label += k

#     return labels


# def cluster_intents(questions: List[str]) -> List[Dict]:
#     # 1) Embed all questions
#     embeddings = get_embeddings(questions)
#     N = len(questions)

#     # 2) Density-based clustering (HDBSCAN)
#     base_labels = HDBSCAN(min_cluster_size=2).fit_predict(embeddings)

#     # 3) Enforce strict max cluster size = 10% of total
#     max_size = max(1, int(0.1 * N))
#     labels = enforce_max_cluster_size(base_labels, embeddings, max_size)

#     # 4) Summarize each cluster with GPT
#     clusters = []
#     for lbl in sorted(set(labels)):
#         if lbl == -1:
#             continue
#         idxs = [i for i, l in enumerate(labels) if l == lbl]
#         cluster_qs = [questions[i] for i in idxs]

#         prompt = (
#             "You are an AI analyst. Given these customer questions:\n"
#             + "\n".join(cluster_qs[:10])
#             + "\nGenerate a concise title and summary of their common intent."
#         )
#         resp = client.chat.completions.create(
#             model=CHAT_MODEL,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         summary = resp.choices[0].message.content

#         clusters.append({
#             "cluster_id": int(lbl),
#             "questions": cluster_qs,
#             "summary": summary
#         })

#     return clusters

import os
import re
import json
import numpy as np
import math
import warnings
from openai import AzureOpenAI
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from typing import List, Dict

from fewshot_classifier import classify_few_shot, FEW_SHOT_LABELS

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
CHAT_MODEL     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def get_embeddings(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def enforce_max_cluster_size(labels: np.ndarray, embeddings: np.ndarray, max_size: int) -> np.ndarray:
    labels = labels.copy()
    next_label = int(labels.max()) + 1

    def split_cluster(cluster_id, member_indices):
        nonlocal next_label
        sub_emb = embeddings[member_indices]
        unique_count = np.unique(sub_emb, axis=0).shape[0]

        if unique_count < 2:
            labels[member_indices] = -1
            return

        k = math.ceil(len(member_indices) / max_size)
        k = max(2, min(k, unique_count))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            sub_labels = KMeans(n_clusters=k).fit_predict(sub_emb)

        for i, sub_lbl in zip(member_indices, sub_labels):
            labels[i] = next_label + sub_lbl
        next_label += k

    while True:
        oversized = None
        for cluster in set(labels):
            if cluster == -1:
                continue
            members = np.where(labels == cluster)[0]
            if len(members) > max_size:
                oversized = (cluster, members)
                break
        if not oversized:
            break
        _, members = oversized
        split_cluster(_, members)

    return labels


def cluster_intents(questions: List[str]) -> List[Dict]:
    fewshot_overrides = {}
    undefined_questions = []
    to_cluster = []

    for q in questions:
        if not q.strip() or len(q.strip().split()) <= 2:
            undefined_questions.append(q)
        else:
            label = classify_few_shot(q)
            if label in FEW_SHOT_LABELS:
                fewshot_overrides[q] = label
            else:
                to_cluster.append(q)

    result: List[Dict] = []

    # --- Few-shot override clusters ---
    for label in FEW_SHOT_LABELS:
        qs = [q for q, lbl in fewshot_overrides.items() if lbl == label]
        if qs:
            percent = (len(qs) / len(questions)) * 100
            result.append({
                "cluster_id": f"fewshot::{label.lower().replace(' ', '_')}",
                "title": f"{label} ({len(qs)} questions, {percent:.1f}%)",
                "summary": f"These questions fall under: {label}.",
                "questions": qs
            })

    # --- Undefined (≤2 words) bucket ---
    if undefined_questions:
        percent = (len(undefined_questions) / len(questions)) * 100
        result.append({
            "cluster_id": -1,
            "title": f"Undefined ({len(undefined_questions)} questions, {percent:.1f}%)",
            "summary": "These questions were too short (≤2 words) to classify.",
            "questions": undefined_questions
        })

    # --- HDBSCAN clustering with max 10% rule ---
    if to_cluster:
        total_questions = len(questions)
        embeddings = get_embeddings(to_cluster)
        base_labels = HDBSCAN(min_cluster_size=2).fit_predict(embeddings)
        max_size = max(1, int(0.1 * total_questions))
        labels = enforce_max_cluster_size(base_labels, embeddings, max_size)

        for lbl in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            cluster_qs = [str(to_cluster[i]) for i in idxs]

            # Safety check: discard if it violates max cluster size (except fewshot)
            if len(cluster_qs) > max_size:
                continue

            prompt = (
                "You are an AI analyst. Given these customer questions:\n"
                + "\n".join(cluster_qs[:10])
                + "\n\nReturn ONLY a JSON object with exactly two keys:\n"
                '  "title": "short descriptive title",\n'
                '  "summary": "concise paragraph of their common intent."\n'
                "Do NOT output any markdown fences or extra text—must be valid JSON."
            )
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a JSON-only assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'```$', '', raw)

            try:
                obj = json.loads(raw)
                title = str(obj["title"]).strip()
                summary = str(obj["summary"]).strip()
            except Exception:
                title = f"Cluster {int(lbl)}"
                summary = raw.replace("\n", " ").strip()

            percent = (len(cluster_qs) / total_questions) * 100
            result.append({
                "cluster_id": int(lbl),
                "title": f"{title} ({len(cluster_qs)} questions, {percent:.1f}%)",
                "summary": summary,
                "questions": cluster_qs
            })

    result.sort(key=lambda c: len(c["questions"]), reverse=True)
    return result
