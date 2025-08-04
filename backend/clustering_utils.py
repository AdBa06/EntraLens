# # clustering_utils.py

# import os
# import re
# import json
# import numpy as np
# import math
# import warnings
# from openai import AzureOpenAI
# from hdbscan import HDBSCAN
# from sklearn.cluster import KMeans
# from sklearn.exceptions import ConvergenceWarning
# from typing import List, Dict

# from translation_utils import translate_if_needed
# from fewshot_classifier import classify_few_shot, FEW_SHOT_LABELS

# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )

# EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
# CHAT_MODEL     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# def get_embeddings(texts: List[str]) -> np.ndarray:
#     resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
#     return np.array([d.embedding for d in resp.data])


# def enforce_max_cluster_size(
#     labels: np.ndarray, embeddings: np.ndarray, max_size: int
# ) -> np.ndarray:
#     labels = labels.copy()
#     next_label = int(labels.max()) + 1

#     def split_cluster(cluster_id, member_indices):
#         nonlocal next_label
#         sub_emb = embeddings[member_indices]
#         if np.unique(sub_emb, axis=0).shape[0] < 2:
#             labels[member_indices] = -1
#             return

#         k = max(2, min(
#             math.ceil(len(member_indices) / max_size),
#             np.unique(sub_emb, axis=0).shape[0]
#         ))
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=ConvergenceWarning)
#             sub_labels = KMeans(n_clusters=k).fit_predict(sub_emb)

#         for idx, sub_lbl in zip(member_indices, sub_labels):
#             labels[idx] = next_label + sub_lbl
#         next_label += k

#     while True:
#         oversized = None
#         for lbl in set(labels):
#             if lbl == -1:
#                 continue
#             members = np.where(labels == lbl)[0]
#             if len(members) > max_size:
#                 oversized = (lbl, members)
#                 break
#         if not oversized:
#             break
#         _, members = oversized
#         split_cluster(_, members)

#     return labels


# def cluster_intents(questions: List[str]) -> List[Dict]:
#     # 1) Wrap each question with its original text and translated display text
#     items = [
#         {"original": q, "display": translate_if_needed(q)}
#         for q in questions
#     ]

#     fewshot_overrides: Dict[str, str] = {}
#     undefined_items = []
#     to_cluster_items = []

#     # 2) Classify few-shot overrides and mark undefined based on display text
#     for itm in items:
#         disp = itm["display"]
#         if not disp.strip() or len(disp.strip().split()) <= 2:
#             undefined_items.append(itm)
#         else:
#             label = classify_few_shot(disp)
#             if label in FEW_SHOT_LABELS:
#                 fewshot_overrides[itm["original"]] = label
#             else:
#                 to_cluster_items.append(itm)

#     result: List[Dict] = []
#     total = len(items)

#     # 3) Few-shot override clusters
#     for label in FEW_SHOT_LABELS:
#         cluster_items = [
#             itm for itm in items
#             if fewshot_overrides.get(itm["original"]) == label
#         ]
#         if not cluster_items:
#             continue
#         pct = len(cluster_items) / total * 100
#         result.append({
#             "cluster_id": f"fewshot::{label.lower().replace(' ', '_')}",
#             "title":      f"{label} ({len(cluster_items)} questions, {pct:.1f}%)",
#             "summary":    f"These questions fall under: {label}.",
#             "questions": [
#                 {"original": itm["original"], "display": itm["display"]}
#                 for itm in cluster_items
#             ]
#         })

#     # 4) Undefined (≤2 words) bucket
#     if undefined_items:
#         pct = len(undefined_items) / total * 100
#         result.append({
#             "cluster_id": -1,
#             "title":      f"Undefined ({len(undefined_items)} questions, {pct:.1f}%)",
#             "summary":    "These questions were too short (≤2 words) to classify.",
#             "questions": [
#                 {"original": itm["original"], "display": itm["display"]}
#                 for itm in undefined_items
#             ]
#         })

#     # 5) HDBSCAN clustering for the rest
#     if to_cluster_items:
#         embeds = get_embeddings([itm["display"] for itm in to_cluster_items])
#         base_labels = HDBSCAN(min_cluster_size=2).fit_predict(embeds)
#         max_size = max(1, int(0.1 * total))
#         labels   = enforce_max_cluster_size(base_labels, embeds, max_size)

#         for lbl in sorted(set(labels)):
#             idxs = [i for i, l in enumerate(labels) if l == lbl]
#             cluster_items = [to_cluster_items[i] for i in idxs]
#             if len(cluster_items) > max_size:
#                 continue

#             prompt = (
#                 "You are an AI analyst. Given these customer questions:\n"
#                 + "\n".join(itm["display"] for itm in cluster_items[:10])
#                 + "\n\nReturn ONLY a JSON object with two keys:\n"
#                   '  "title": "short descriptive title",\n'
#                   '  "summary": "concise paragraph of their common intent."\n'
#                 "Do NOT output any markdown or extra text—just valid JSON."
#             )
#             resp = client.chat.completions.create(
#                 model=CHAT_MODEL,
#                 messages=[
#                     {"role": "system", "content": "You are a JSON-only assistant."},
#                     {"role": "user",   "content": prompt}
#                 ]
#             )
#             raw = resp.choices[0].message.content.strip()
#             raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
#             raw = re.sub(r'```$', '', raw)

#             try:
#                 obj = json.loads(raw)
#                 title   = obj["title"].strip()
#                 summary = obj["summary"].strip()
#             except Exception:
#                 title   = f"Cluster {int(lbl)}"
#                 summary = raw.replace("\n", " ").strip()

#             pct = len(cluster_items) / total * 100
#             result.append({
#                 "cluster_id": int(lbl),
#                 "title":      f"{title} ({len(cluster_items)} questions, {pct:.1f}%)",
#                 "summary":    summary,
#                 "questions": [
#                     {"original": itm["original"], "display": itm["display"]}
#                     for itm in cluster_items
#                 ]
#             })

#     # 6) Sort by size descending
#     result.sort(key=lambda c: len(c["questions"]), reverse=True)
#     return result

# clustering_utils.py
# clustering_utils.py

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

from translation_utils import translate_if_needed
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


def enforce_max_cluster_size(
    labels: np.ndarray,
    embeddings: np.ndarray,
    max_size: int
) -> np.ndarray:
    labels = labels.copy()
    next_label = int(labels.max()) + 1

    def split_cluster(cluster_id, member_indices):
        nonlocal next_label
        sub_emb = embeddings[member_indices]
        if np.unique(sub_emb, axis=0).shape[0] < 2:
            labels[member_indices] = -1
            return

        k = max(2, min(
            math.ceil(len(member_indices) / max_size),
            np.unique(sub_emb, axis=0).shape[0]
        ))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            sub_labels = KMeans(n_clusters=k).fit_predict(sub_emb)

        for idx, sub_lbl in zip(member_indices, sub_labels):
            labels[idx] = next_label + sub_lbl
        next_label += k

    while True:
        oversized = None
        for lbl in set(labels):
            if lbl == -1:
                continue
            members = np.where(labels == lbl)[0]
            if len(members) > max_size:
                oversized = (lbl, members)
                break
        if not oversized:
            break
        _, members = oversized
        split_cluster(_, members)

    return labels


def cluster_intents(questions: List[str]) -> List[Dict]:
    # 1) Wrap each question with its original and translated text
    items = [
        {"original": q, "display": translate_if_needed(q)}
        for q in questions
    ]

    fewshot_overrides: Dict[str, str] = {}
    undefined_items = []
    to_cluster_items = []

    # 2) Few-shot overrides & undefined based solely on display text length
    for itm in items:
        disp = itm["display"].strip()
        # 2 words or fewer → undefined
        if not disp or len(disp.split()) <= 2:
            undefined_items.append(itm)
        else:
            label = classify_few_shot(disp)
            if label in FEW_SHOT_LABELS:
                fewshot_overrides[itm["original"]] = label
            else:
                to_cluster_items.append(itm)

    result: List[Dict] = []
    total = len(items)

    # 3) Few-shot override clusters
    for label in FEW_SHOT_LABELS:
        cluster_items = [
            itm for itm in items
            if fewshot_overrides.get(itm["original"]) == label
        ]
        if not cluster_items:
            continue
        pct = len(cluster_items) / total * 100
        result.append({
            "cluster_id": f"fewshot::{label.lower().replace(' ', '_')}",
            "title":      f"{label} ({len(cluster_items)} questions, {pct:.1f}%)",
            "summary":    f"These questions fall under: {label}.",
            "questions":  [
                {"original": itm["original"], "display": itm["display"]}
                for itm in cluster_items
            ]
        })

    # 4) Undefined bucket
    if undefined_items:
        pct = len(undefined_items) / total * 100
        result.append({
            "cluster_id": -1,
            "title":      f"Undefined ({len(undefined_items)} questions, {pct:.1f}%)",
            "summary":    "These inputs were too short (≤2 words) to classify.",
            "questions":  [
                {"original": itm["original"], "display": itm["display"]}
                for itm in undefined_items
            ]
        })

    # 5) HDBSCAN clustering for the rest
    if to_cluster_items:
        embeds      = get_embeddings([itm["display"] for itm in to_cluster_items])
        base_labels = HDBSCAN(min_cluster_size=2).fit_predict(embeds)
        max_size    = max(1, int(0.1 * total))
        labels      = enforce_max_cluster_size(base_labels, embeds, max_size)

        for lbl in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            cluster_items = [to_cluster_items[i] for i in idxs]
            if len(cluster_items) > max_size:
                continue

            # Summarize cluster via chat model
            prompt = (
                "You are an AI analyst. Given these customer questions:\n"
                + "\n".join(itm["display"] for itm in cluster_items[:10])
                + "\n\nReturn ONLY a JSON object with two keys:\n"
                  '  "title": "short descriptive title",\n'
                  '  "summary": "concise paragraph of their common intent."\n'
                "Do NOT output any markdown or extra text—just valid JSON."
            )
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a JSON-only assistant."},
                    {"role": "user",   "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'```$', '', raw)

            try:
                obj     = json.loads(raw)
                title   = obj["title"].strip()
                summary = obj["summary"].strip()
            except Exception:
                title   = f"Cluster {int(lbl)}"
                summary = raw.replace("\n", " ").strip()

            pct = len(cluster_items) / total * 100
            result.append({
                "cluster_id": int(lbl),
                "title":      f"{title} ({len(cluster_items)} questions, {pct:.1f}%)",
                "summary":    summary,
                "questions":  [
                    {"original": itm["original"], "display": itm["display"]}
                    for itm in cluster_items
                ]
            })

    # 6) Sort by descending size
    result.sort(key=lambda c: len(c["questions"]), reverse=True)

    # --- DEBUGGING: catch any questions that got dropped entirely ---
    clustered = {
        itm["original"]
        for cluster in result
        for itm in cluster["questions"]
    }
    missing = [q for q in questions if q not in clustered]
    if missing:
        print(f"[cluster_intents] ⚠️ {len(missing)} questions dropped: {missing}")
        # optionally include them in an "unclustered" bucket:
        result.append({
            "cluster_id": "unclustered",
            "title":      f"Unclustered ({len(missing)} questions, {(len(missing)/total)*100:.1f}%)",
            "summary":    "These questions didn’t get picked up by any cluster.",
            "questions":  [{"original": q, "display": translate_if_needed(q)} for q in missing]
        })

    return result
