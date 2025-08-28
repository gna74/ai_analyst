import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Step 1: Read technology cards from JSONL and generate embeddings via DashScope
cards_file = "cards.jsonl"
if not os.path.isfile(cards_file):
    raise FileNotFoundError(f"Input file {cards_file} not found.")

# Load cards data from cards.jsonl (JSON lines format)
cards = []
with open(cards_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cards.append(json.loads(line))

# Prepare text for each card: use 'embedding_snippet' if available, otherwise concatenate definition, method, technical_features, applications as fallback
embedding_texts = []
embedding_ids = []
for card in cards:
    text = ""
    if card.get("embedding_snippet"):
        text = str(card["embedding_snippet"]).strip()
    if not text:
        parts = []
        for field in ["definition", "method", "technical_features", "applications"]:
            val = card.get(field)
            if val:
                val_str = str(val).strip()
                if val_str:
                    parts.append(val_str)
        text = " ".join(parts).strip()
    embedding_texts.append(text)
    tech_id = card.get("tech_id")
    if tech_id is None:
        raise KeyError("Each card must have a tech_id.")
    embedding_ids.append(tech_id)

# Debug: print number of cards loaded and a sample text
print("Total cards loaded:", len(cards))
print("Total texts prepared:", len(embedding_texts))
print("Example text snippet:", embedding_texts[0] if embedding_texts else "NONE")

# Generate embeddings using DashScope text-embedding-v3 model (batch up to 48 texts per request)
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_key:
    raise RuntimeError("DASHSCOPE_API_KEY is not set. Please set your DashScope API key as an environment variable.")
dashscope_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {dashscope_key}"
}

# DashScope limits batch size to 10 texts per request
BATCH_SIZE = 10  # ограничение DashScope

embeddings_list = []
for i in tqdm(range(0, len(embedding_texts), BATCH_SIZE), desc="Generating embeddings"):
    batch_texts = embedding_texts[i : i+BATCH_SIZE]

    payload = {
        "model": "text-embedding-v3",
        "input": batch_texts
    }
    response = requests.post(dashscope_url, headers=headers, json=payload)

    # Отладочный вывод на случай ошибки
    if response.status_code != 200:
        print("DashScope error:", response.status_code, response.text)

    response.raise_for_status()

    result = response.json()
    data_items = result.get("data", [])
    # Ensure order corresponds to input order using the index field:contentReference[oaicite:0]{index=0}
    data_items.sort(key=lambda x: x.get("index", 0))
    for item in data_items:
        embeddings_list.append(item["embedding"])

# Convert list of embeddings to numpy array
embeddings = np.array(embeddings_list, dtype=np.float32)

# Save embeddings and mapping (tech_id -> index)
np.save("embeddings.npy", embeddings)
embedding_map = {tech_id: idx for idx, tech_id in enumerate(embedding_ids)}
with open("embeddings_map.json", "w", encoding="utf-8") as f:
    json.dump(embedding_map, f, ensure_ascii=False)

# Step 2: Clustering with KMeans and selecting optimal k via silhouette score:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}
# Normalize embeddings to unit length for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1e-9  # avoid division by zero
embeddings_normed = embeddings / norms

best_k = None
best_score = -1.0
best_labels = None
for k in [40, 44, 48, 52, 56, 60]:
    # Silhouette score is defined only if 2 <= n_clusters <= n_samples - 1:contentReference[oaicite:3]{index=3}
    if k < 2 or k > len(embeddings_normed):
        continue
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings_normed)
    score = silhouette_score(embeddings_normed, labels, metric='cosine')
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

# If no valid k (e.g., dataset too small), default to a single cluster
if best_labels is None:
    best_k = 1
    best_labels = np.zeros(len(embeddings_normed), dtype=int)

# Assign cluster_id to each card for later reference
for card, label in zip(cards, best_labels):
    card["cluster_id"] = int(label)

# Step 3: Automatic cluster labeling using DeepSeek (LLM-based)
# Group cards by cluster
clusters = {}
for idx, label in enumerate(best_labels):
    clusters.setdefault(label, []).append(idx)

# Compute top 10 terms for each cluster using TF-IDF:contentReference[oaicite:4]{index=4}
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(embedding_texts)
feature_names = vectorizer.get_feature_names_out()

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Please set your DeepSeek API key as an environment variable.")
deepseek_url = "https://api.deepseek.com/v1/chat/completions"
ds_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {deepseek_key}"
}

cluster_labels_info = []  # to accumulate results for cluster_labels.csv
for cluster_id, indices in tqdm(clusters.items(), desc="Labeling clusters"):
    # Determine top terms by average TF-IDF weight in this cluster
    top_terms = []
    if indices:
        mean_tfidf = tfidf_matrix[indices].mean(axis=0).A1  # average TF-IDF vector for cluster
        if np.count_nonzero(mean_tfidf) > 0:
            top_idx = np.argsort(mean_tfidf)[-10:][::-1]
            top_terms = [feature_names[j] for j in top_idx if mean_tfidf[j] > 0]
    top_terms = top_terms[:10]
    top_terms_str = ", ".join(top_terms)

    # Find 3–5 anchor technologies closest to cluster centroid
    centroid = embeddings_normed[indices].mean(axis=0) if indices else np.zeros(embeddings_normed.shape[1])
    cen_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    cos_sims = embeddings_normed[indices].dot(cen_norm) if indices else np.array([])
    anchor_count = 5 if len(indices) >= 5 else len(indices)
    top_idx_sorted = np.argsort(-cos_sims)[:anchor_count] if anchor_count > 0 else []
    anchor_indices = [indices[i] for i in top_idx_sorted]  # indices of top-similarity cards in this cluster

    # Prepare anchor technology info for the prompt (name and a brief description)
    anchor_lines = []
    for idx in anchor_indices:
        tech_name = cards[idx].get("name", "")
        # Use embedding_snippet if available, otherwise a short definition
        desc = cards[idx].get("embedding_snippet") or cards[idx].get("definition") or ""
        desc = str(desc).strip()
        # Truncate the description for brevity (approx first sentence or ~200 chars)
        if len(desc) > 200:
            period_idx = desc.find('. ')
            if 0 < period_idx < 200:
                desc = desc[:period_idx+1]
            else:
                desc = desc[:200].rstrip() + "..."
        anchor_lines.append(f"- {tech_name}: {desc}")
    anchor_text = "\n".join(anchor_lines)

    # Formulate DeepSeek prompts for labeling
    system_prompt = (
        "You are a helpful assistant with expertise in technology topics. "
        "Analyze the given cluster terms and technology snippets, and output a JSON with a descriptive cluster label and a short description."
    )
    user_prompt = (
        f"Top terms: {top_terms_str}\n"
        f"Technologies in this cluster:\n{anchor_text}\n\n"
        "Provide a suitable cluster name (<= 60 characters) and a brief description of the cluster (<= 300 tokens). "
        "Respond **only** with a JSON object having keys \"label\" and \"cluster_short_description\"."
    )

    # Call DeepSeek API (model: deepseek-chat) to get cluster label and description
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    response = requests.post(deepseek_url, headers=ds_headers, json=payload)
    response.raise_for_status()
    result = response.json()
    
    # Extract and parse the assistant's JSON answer:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
    assistant_content = result["choices"][0]["message"].get("content", "")
    cluster_label = ""
    cluster_desc = ""
    if assistant_content:
        try:
            output = json.loads(assistant_content)
            cluster_label = str(output.get("label", "")).strip()
            cluster_desc = str(output.get("cluster_short_description", "")).strip()
        except json.JSONDecodeError:
            # If JSON parsing fails (should not happen with 'json_object'), use raw content as fallback
            cluster_desc = assistant_content.strip()
    # Ensure character limits (label <= 60 chars, description ~<=300 tokens ≈ 1500 chars)
    if len(cluster_label) > 60:
        cluster_label = cluster_label[:59] + "…"
    if len(cluster_desc) > 1500:
        cluster_desc = cluster_desc[:1499].rstrip() + "…"

    cluster_labels_info.append({
        "cluster_id": cluster_id,
        "label": cluster_label,
        "cluster_short_description": cluster_desc,
        "top_terms": top_terms_str
    })

# Save cluster labels info to CSV (cluster_id; label; cluster_short_description; top_terms)
with open("cluster_labels.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["cluster_id", "label", "cluster_short_description", "top_terms"])
    for info in cluster_labels_info:
        # Replace any newlines in fields to keep CSV one-line per record
        label = info["label"].replace("\n", " ").strip()
        desc = info["cluster_short_description"].replace("\n", " ").strip()
        terms = info["top_terms"].replace("\n", " ").strip()
        writer.writerow([info["cluster_id"], label, desc, terms])

# Step 4: Construct multi-level cluster structure JSON
cluster_structure = []
for cluster_id in sorted(clusters.keys()):
    # Retrieve the label and description for this cluster
    label_info = next((item for item in cluster_labels_info if item["cluster_id"] == cluster_id), {})
    cluster_label = label_info.get("label", "")
    cluster_desc = label_info.get("cluster_short_description", "")
    tech_names = [cards[idx]["name"] for idx in clusters[cluster_id]]
    cluster_structure.append({
        "cluster_label": cluster_label,
        "cluster_short_description": cluster_desc,
        "technologies": tech_names
    })
with open("cluster_structure.json", "w", encoding="utf-8") as f:
    json.dump(cluster_structure, f, ensure_ascii=False, indent=2)

print(f"Clustering complete. Selected k = {best_k} with silhouette score {best_score:.4f}.")
print("Outputs saved to embeddings.npy, embeddings_map.json, cluster_labels.csv, and cluster_structure.json.")
