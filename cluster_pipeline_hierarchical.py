#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_pipeline_hierarchical.py (v2)
-------------------------------------

Purpose
=======
1) Evaluate an expanded grid of K (KMeans) values to pick THREE best variants:
   - best in 35–44
   - best in 45–54
   - best in 55–64
   (30 total evaluations: K ∈ {35..64}).

2) For each chosen K, build ONE hierarchical level above flat clusters by
   agglomerative merging of *nearby* cluster centroids in cosine space
   with an auto-threshold τ (median nearest-neighbor distance), clamped
   to [0.15, 0.35]. If clusters are far apart, no grouping occurs.

3) Name the groups (and clusters fallback) using TF‑IDF top-terms (+ optional LLM).

4) Save three independent files: one per chosen K with full structure and metrics.

Notes
=====
- Reuses precomputed embeddings (embeddings.npy) + map (embeddings_map.json).
- Uses cards.jsonl to retrieve names/texts for TF‑IDF and assignments.
- No network calls by default. Optional LLM labeling (DeepSeek) can be enabled
  via CLI flag and environment variables.

Author: AI Analyst — clustering module
Date  : 2025-09-02
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable

import numpy as np
from tqdm import tqdm

# sklearn imports
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Utility helpers
# -----------------------------

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between A (nxd) and B (mxd).
    A and B expected L2-normalized; otherwise will normalize internally.
    """
    A_norm = l2_normalize(A) if np.any(np.abs(np.linalg.norm(A, axis=1) - 1.0) > 1e-3) else A
    B_norm = l2_normalize(B) if np.any(np.abs(np.linalg.norm(B, axis=1) - 1.0) > 1e-3) else B
    return A_norm @ B_norm.T


def cosine_distance_matrix(C: np.ndarray) -> np.ndarray:
    """Return pairwise cosine distance matrix for centroids C."""
    S = cosine_similarity(C, C)
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def bincount_stats(labels: np.ndarray, K: int | None = None) -> Dict[str, float]:
    if K is None:
        K = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=K)
    return {
        "min": float(counts.min() if len(counts) else 0.0),
        "max": float(counts.max() if len(counts) else 0.0),
        "mean": float(counts.mean() if len(counts) else 0.0),
        "std": float(counts.std() if len(counts) else 0.0),
        "counts": counts.tolist(),
    }


# -----------------------------
# I/O helpers
# -----------------------------

def load_embeddings(emb_path: str) -> np.ndarray:
    """Load embeddings from .npy robustly.
    - First try standard np.load (allow_pickle=False).
    - If it fails with the common "allow_pickle=False" error, retry with allow_pickle=True
      and convert possible object arrays (list-of-arrays) into a 2D float32 matrix.
    - Validate consistent dimensionality: final shape must be (N, D) with D>=2.
    """
    try:
        X = np.load(emb_path)  # allow_pickle=False (safe path)
    except ValueError as e:
        msg = str(e)
        if "allow_pickle=False" in msg:
            # Fallback: load pickled object arrays (trusted local file assumed)
            X = np.load(emb_path, allow_pickle=True)
        else:
            raise

    # If this is an object array (e.g., list of vectors), stack to 2D
    if X.dtype == object:
        rows = []
        for i, row in enumerate(X):
            arr = np.asarray(row, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.ravel()
            rows.append(arr)
        lengths = {len(r) for r in rows}
        if len(lengths) != 1:
            # inconsistent lengths — cannot form a proper matrix
            ex = sorted(list(lengths))[:5]
            raise ValueError(
                f"Inconsistent embedding lengths in '{emb_path}': {ex} (showing up to 5). "
                "Regenerate embeddings as a uniform 2D array."
            )
        X = np.vstack(rows)
    else:
        # Enforce float32 and 2D shape
        if X.ndim == 1:
            raise ValueError(
                f"Embeddings in '{emb_path}' have shape {X.shape} — expected (N, D). "
                "Regenerate embeddings as a 2D array."
            )
        X = X.astype(np.float32, copy=False)

    if X.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array. Got shape {X.shape}")
    if X.shape[1] < 2:
        raise ValueError(f"Embedding dimensionality too small: {X.shape[1]}")
    return X


def load_map(map_path: str) -> Dict[str, int]:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both {tech_id: idx} and list-of-pairs formats
    if isinstance(data, dict):
        # keys might be str/int; ensure str keys
        return {str(k): int(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Expect list of {"tech_id":..., "index":...}
        out = {}
        for row in data:
            if isinstance(row, dict) and "tech_id" in row and "index" in row:
                out[str(row["tech_id"])] = int(row["index"])
        if not out:
            raise ValueError("Unsupported embeddings_map.json format (list without expected keys)")
        return out
    else:
        raise ValueError("Unsupported embeddings_map.json format")


def try_json_load(line: str) -> Any | None:
    try:
        return json.loads(line)
    except Exception:
        return None


def load_cards_jsonl(cards_path: str) -> Dict[str, Dict[str, Any]]:
    """Return dict tech_id -> card (raw JSON). Skips invalid lines gracefully."""
    out = {}
    if not os.path.exists(cards_path):
        return out
    with open(cards_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = try_json_load(line)
            if obj is None:
                # skip malformed
                continue
            tid = str(obj.get("tech_id", "")).strip()
            if not tid:
                continue
            out[tid] = obj
    return out


def build_text_corpus(emb_map: Dict[str, int], cards: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Return (texts_by_index, names_by_index) aligned with embeddings indices.
    Text prefers `embedding_snippet`, fallback to concatenation of definition/method/features/applications/functional_role.
    """
    N = max(emb_map.values()) + 1 if emb_map else 0
    texts = [""] * N
    names = [""] * N
    for tid, idx in emb_map.items():
        card = cards.get(tid, {})
        text = (
            card.get("embedding_snippet")
            or " ".join(
                filter(
                    None,
                    [
                        card.get("definition"),
                        card.get("method"),
                        " ".join(card.get("technical_features", []) if isinstance(card.get("technical_features"), list) else []),
                        " ".join(card.get("applications", []) if isinstance(card.get("applications"), list) else []),
                        card.get("functional_role"),
                    ],
                )
            )
        )
        texts[idx] = (text or "").replace("\n", " ").strip()
        names[idx] = str(card.get("name") or tid)
    return texts, names


# -----------------------------
# Evaluation across K grid
# -----------------------------

@dataclass
class KEval:
    k: int
    labels: np.ndarray
    centers: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    size_stats: Dict[str, Any]


def evaluate_k_grid(
    X_norm: np.ndarray,
    ks: Iterable[int],
    random_state: int = 42,
    n_init: int | str = "auto",
) -> List[KEval]:
    results: List[KEval] = []
    for k in tqdm(list(ks), desc="Evaluating K grid"):
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X_norm)
        centers = l2_normalize(model.cluster_centers_)
        # Metrics with safety
        try:
            sil = float(silhouette_score(X_norm, labels, metric="cosine"))
        except Exception:
            sil = float("-inf")
        try:
            db = float(davies_bouldin_score(X_norm, labels))
        except Exception:
            db = float("inf")
        try:
            ch = float(calinski_harabasz_score(X_norm, labels))
        except Exception:
            ch = float("-inf")
        stats = bincount_stats(labels, K=k)
        results.append(KEval(k, labels, centers, sil, db, ch, stats))
    return results


def pick_best_by_bands(results: List[KEval], bands: List[Tuple[int, int]]) -> List[KEval]:
    out: List[KEval] = []
    for lo, hi in bands:
        cand = [r for r in results if lo <= r.k <= hi]
        if not cand:
            continue
        # sort: best silhouette (desc), then DB (asc), then CH (desc), then size std (asc)
        cand.sort(key=lambda r: (
            -(r.silhouette if math.isfinite(r.silhouette) else -1e9),
            (r.davies_bouldin if math.isfinite(r.davies_bouldin) else 1e9),
            -(r.calinski_harabasz if math.isfinite(r.calinski_harabasz) else -1e9),
            (r.size_stats.get("std", 1e9)),
        ))
        out.append(cand[0])
    return out


# -----------------------------
# Hierarchical super-clusters
# -----------------------------

def median_nearest_neighbor_distance(D: np.ndarray) -> float:
    # D is square matrix of distances; diagonal 0.
    n = D.shape[0]
    if n <= 1:
        return 0.0
    arr = []
    for i in range(n):
        # exclude self (0), take min over j!=i
        row = np.delete(D[i], i)
        if len(row) == 0:
            continue
        arr.append(float(np.min(row)))
    if not arr:
        return 0.0
    return float(np.median(arr))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_superclusters(centers: np.ndarray, tau: float | None = None) -> np.ndarray:
    """Return group labels for clusters via agglomerative clustering with distance threshold.
    If tau is None, it's auto-chosen as clamp(median_NN_distance, 0.15, 0.35).
    """
    if centers.shape[0] <= 1:
        return np.zeros((centers.shape[0],), dtype=int)

    D = cosine_distance_matrix(centers)
    if tau is None:
        tau = clamp(median_nearest_neighbor_distance(D), 0.15, 0.35)

    # Try metric='precomputed' (newer sklearn); fallback to affinity='precomputed'
    try:
        agg = AgglomerativeClustering(
            n_clusters=None,
            linkage="average",
            metric="precomputed",
            distance_threshold=tau,
        )
    except TypeError:
        agg = AgglomerativeClustering(
            n_clusters=None,
            linkage="average",
            affinity="precomputed",  # deprecated in new sklearn, but used in old
            distance_threshold=tau,
        )
    labels = agg.fit(D).labels_
    # Reindex groups by size (descending) for stable IDs
    unique, counts = np.unique(labels, return_counts=True)
    order = [g for g, _ in sorted(zip(unique, counts), key=lambda x: -x[1])]
    remap = {g: i for i, g in enumerate(order)}
    return np.array([remap[g] for g in labels], dtype=int)


# -----------------------------
# TF‑IDF and labeling
# -----------------------------

def fit_tfidf(texts: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    Xtf = vec.fit_transform(texts)
    return vec, Xtf


def top_terms_for_cluster(Xtf, labels: np.ndarray, cluster_id: int, vec: TfidfVectorizer, topk: int = 12) -> List[str]:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    mean_vec = Xtf[idx].mean(axis=0)  # sparse mean
    mean_arr = np.asarray(mean_vec).ravel()
    top_idx = np.argsort(-mean_arr)[:topk]
    terms = vec.get_feature_names_out()
    return [terms[i] for i in top_idx]


def anchors_for_cluster(X_norm: np.ndarray, labels: np.ndarray, cluster_id: int, names: List[str], tech_ids: List[str], topn: int = 6) -> List[Tuple[str, str, float]]:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    cluster_vec = l2_normalize(np.mean(X_norm[idx], axis=0, keepdims=True))
    sims = (X_norm[idx] @ cluster_vec.T).ravel()
    order = np.argsort(-sims)[: min(topn, len(sims))]
    out = []
    for r in order:
        i = idx[r]
        out.append((tech_ids[i], names[i], float(sims[r])))
    return out


def label_from_terms(terms: List[str], prefix: str | None = None, max_terms: int = 3) -> str:
    core = ", ".join([t.title() for t in terms[:max_terms]]) if terms else "General"
    return f"{prefix}: {core}" if prefix else core


def short_desc_from_terms_and_anchors(terms: List[str], anchors: List[Tuple[str, str, float]], is_group: bool = False) -> str:
    key_terms = ", ".join(terms[:4]) if terms else "mixed topics"
    anchor_names = ", ".join([a[1] for a in anchors[:3]]) if anchors else "sample technologies"
    head = "Group of related clusters" if is_group else "Cluster focusing on"
    return f"{head} {key_terms}; examples: {anchor_names}."


# Optional: LLM labeling (DeepSeek)
# ---------------------------------
import requests


def maybe_label_with_llm(prompt: str, system: str | None = None, 
                         provider: str = "deepseek", timeout: int = 40) -> str | None:
    """Try to call LLM provider if API key & URL are configured. Return string label or None on failure."""
    provider = provider.lower() if provider else ""
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        if not api_key:
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content.strip().split("\n")[0][:120]
        except Exception:
            return None
    return None


# -----------------------------
# Build full variant for a chosen K
# -----------------------------

def build_variant(
    X_norm: np.ndarray,
    tech_ids: List[str],
    names: List[str],
    texts: List[str],
    eval_res: KEval,
    tfidf_vec: TfidfVectorizer,
    Xtf,
    use_llm_labels: bool = False,
    llm_provider: str = "deepseek",
) -> Dict[str, Any]:
    k = eval_res.k
    labels = eval_res.labels
    centers = eval_res.centers

    # Super-clusters (groups)
    group_labels_arr = build_superclusters(centers)

    # Per-cluster info
    clusters_out = []
    for c in range(k):
        c_terms = top_terms_for_cluster(Xtf, labels, c, tfidf_vec, topk=12)
        anchors = anchors_for_cluster(X_norm, labels, c, names, tech_ids, topn=6)
        fallback_label = label_from_terms(c_terms)
        c_label = fallback_label
        if use_llm_labels:
            prompt = (
                "Propose a short, canonical technology cluster label (max 8 words)\n"
                f"Top terms: {', '.join(c_terms[:10])}\n"
                f"Examples: {', '.join([n for _, n, _ in anchors])}\n"
                "Return only the label."
            )
            res = maybe_label_with_llm(prompt, system="You label technology clusters succinctly.", provider=llm_provider)
            if res:
                c_label = res
        c_desc = short_desc_from_terms_and_anchors(c_terms, anchors, is_group=False)
        members_idx = np.where(labels == c)[0]
        tech_items = [
            {"tech_id": tech_ids[i], "name": names[i]} for i in members_idx
        ]
        clusters_out.append({
            "cluster_id": int(c),
            "label": c_label,
            "cluster_short_description": c_desc,
            "top_terms": c_terms[:8],
            "group_id": int(group_labels_arr[c]),
            "tech_items": tech_items,
        })

    # Group-level info
    groups_out = []
    n_groups = int(group_labels_arr.max()) + 1 if len(group_labels_arr) else 0
    for g in range(n_groups):
        member_clusters = [c for c in range(k) if int(group_labels_arr[c]) == g]
        # Aggregate terms: mean of cluster means approximated via member docs
        member_docs = np.concatenate([np.where(labels == c)[0] for c in member_clusters]) if member_clusters else np.array([], dtype=int)
        if member_docs.size > 0:
            mean_vec = Xtf[member_docs].mean(axis=0)
            mean_arr = np.asarray(mean_vec).ravel()
            top_idx = np.argsort(-mean_arr)[:12]
            terms = [tfidf_vec.get_feature_names_out()[i] for i in top_idx]
        else:
            terms = []
        # Anchors for group: take top by similarity to group centroid
        if member_clusters:
            c_vecs = centers[member_clusters]
            g_vec = l2_normalize(np.mean(c_vecs, axis=0, keepdims=True))
            sims = (X_norm @ g_vec.T).ravel()
            order = np.argsort(-sims)[:8]
            anchors = [(tech_ids[i], names[i], float(sims[i])) for i in order]
        else:
            anchors = []

        fallback_label = label_from_terms(terms, prefix="Group")
        g_label = fallback_label
        if use_llm_labels:
            prompt = (
                "Propose a short, canonical GROUP label (max 8 words) that unifies its clusters.\n"
                f"Top terms: {', '.join(terms[:10])}\n"
                f"Examples: {', '.join([n for _, n, _ in anchors])}\n"
                "Return only the label."
            )
            res = maybe_label_with_llm(prompt, system="You label technology groups succinctly.", provider=llm_provider)
            if res:
                g_label = res
        g_desc = short_desc_from_terms_and_anchors(terms, anchors, is_group=True)

        groups_out.append({
            "group_id": int(g),
            "group_label": g_label,
            "group_short_description": g_desc,
            "member_cluster_ids": [int(c) for c in member_clusters],
        })

    # Compose output
    out = {
        "k": int(k),
        "metrics": {
            "silhouette_cosine": eval_res.silhouette,
            "davies_bouldin": eval_res.davies_bouldin,
            "calinski_harabasz": eval_res.calinski_harabasz,
            "cluster_size_stats": {
                k: (float(v) if not isinstance(v, list) else v)
                for k, v in eval_res.size_stats.items()
            },
        },
        "clusters": clusters_out,
        "groups": groups_out,
    }
    return out


# -----------------------------
# Save helpers
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_assignments_csv(struct: Dict[str, Any], out_csv: str):
    import csv
    # Build lookups
    cluster_labels = {c["cluster_id"]: c.get("label", "") for c in struct.get("clusters", [])}
    group_ids = {c["cluster_id"]: c.get("group_id", -1) for c in struct.get("clusters", [])}
    group_labels = {g["group_id"]: g.get("group_label", "") for g in struct.get("groups", [])}

    rows = []
    for c in struct.get("clusters", []):
        cid = c["cluster_id"]
        for item in c.get("tech_items", []):
            tid = item.get("tech_id", "")
            name = item.get("name", "")
            gid = group_ids.get(cid, -1)
            rows.append([
                tid, name, cid, gid, cluster_labels.get(cid, ""), group_labels.get(gid, "")
            ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tech_id", "name", "cluster_id", "group_id", "cluster_label", "group_label"])
        w.writerows(rows)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="KMeans grid + 1-level hierarchy for tech clusters")
    parser.add_argument("--embeddings", default="embeddings.npy", help="Path to embeddings .npy")
    parser.add_argument("--map", default="embeddings_map.json", help="Path to embeddings map JSON")
    parser.add_argument("--cards", default="cards.jsonl", help="Path to cards JSONL (for names/texts)")
    parser.add_argument("--outdir", default="out_clusters", help="Output directory")
    parser.add_argument("--kmin", type=int, default=35)
    parser.add_argument("--kmax", type=int, default=64)
    parser.add_argument("--bands", default="35-44,45-54,55-64", help="Comma-separated ranges like '35-44,45-54,55-64'")
    parser.add_argument("--use-llm-labels", action="store_true", help="If set, try LLM labeling when API is configured")
    parser.add_argument("--llm-provider", default="deepseek", help="LLM provider id (currently 'deepseek')")
    parser.add_argument("--save-assignments-csv", action="store_true", help="Save CSV with assignments for each K variant")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Load data
    print("[Load] embeddings…")
    X = load_embeddings(args.embeddings)
    print("[Load] map…")
    emb_map = load_map(args.map)

    if X.shape[0] != (max(emb_map.values()) + 1):
        print(
            f"[Warn] Embeddings count {X.shape[0]} != max(map)+1 {(max(emb_map.values()) + 1)}. Proceeding but check consistency.")

    tech_ids_by_index = [""] * X.shape[0]
    for tid, idx in emb_map.items():
        if 0 <= idx < X.shape[0]:
            tech_ids_by_index[idx] = tid

    print("[Load] cards.jsonl…")
    cards = load_cards_jsonl(args.cards)
    texts, names = build_text_corpus(emb_map, cards)

    # Normalize embeddings for cosine metrics
    X_norm = l2_normalize(X)

    # TF-IDF (on all documents)
    print("[TF-IDF] fitting…")
    tfidf_vec, Xtf = fit_tfidf(texts, max_features=5000)

    # Build K grid
    ks = list(range(args.kmin, args.kmax + 1))
    if len(ks) != 30:
        print(f"[Info] Grid has {len(ks)} values (expected 30). Using given range.")

    # Evaluate
    results = evaluate_k_grid(X_norm, ks)

    # Pick best in bands
    def parse_band(b: str) -> Tuple[int, int]:
        a, b = b.split("-")
        return int(a), int(b)

    bands = [parse_band(x) for x in args.bands.split(",") if x.strip()]
    best_three = pick_best_by_bands(results, bands)

    # Build & save each variant
    for res in best_three:
        print(f"[Variant] Building hierarchical structure for k={res.k}…")
        struct = build_variant(
            X_norm,
            tech_ids_by_index,
            names,
            texts,
            res,
            tfidf_vec,
            Xtf,
            use_llm_labels=args.use_llm_labels,
            llm_provider=args.llm_provider,
        )
        out_json = os.path.join(args.outdir, f"hierarchical_clusters_k{res.k}.json")
        save_json(struct, out_json)
        print(f"[Saved] {out_json}")

        if args.save_assignments_csv:
            out_csv = os.path.join(args.outdir, f"assignments_k{res.k}.csv")
            save_assignments_csv(struct, out_csv)
            print(f"[Saved] {out_csv}")

    print("[Done]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
