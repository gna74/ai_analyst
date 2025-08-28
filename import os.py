import os
import csv
import json
import time
import hashlib
import datetime
import asyncio
from urllib.parse import urlparse, unquote
import ssl
import certifi

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from jsonschema import validate, ValidationError
import numpy as np
from dotenv import load_dotenv

import aiohttp
from aiohttp import ClientSession

# =========================
# Init & Config
# =========================

load_dotenv()  # подхват переменных из .env

def env_required(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Отсутствует переменная окружения {name}. Добавьте её в .env или Secret Manager.")
    return val

# Ключи и эндпоинты
DEEPSEEK_API_KEY  = env_required("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL  = "https://api.deepseek.com/chat/completions"  # стабильный endpoint
DASHSCOPE_API_KEY = env_required("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"
DASHSCOPE_MODEL   = "text-embedding-v3"

# Параметры производительности
CONCURRENCY   = 12    # параллельных генераций карточек
MAX_TOKENS    = 900   # немного ужали, чтобы ускорить
BATCH_SIZE_EMB = 48   # размер батча для эмбеддингов (32–64 обычно ок)

# Тайм-ауты для aiohttp
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=120,
    connect=20,
    sock_connect=20,
    sock_read=90
)

# HTTP session (requests) с ретраями — применяем для DashScope эмбеддингов
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
)
session = requests.Session()
session.verify = certifi.where()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# =========================
# Schema & Utilities
# =========================

# JSON Schema (сохранено из вашей версии)
tech_card_schema = {
    "type": "object",
    "properties": {
        "tech_id": {"type": ["string", "number"]},
        "name": {"type": "string"},
        "definition": {"type": "string"},
        "method": {"type": "string"},
        "technical_features": {
            "type": "array",
            "items": {"type": "string"}
        },
        "applications": {
            "type": "array",
            "items": {"type": "string"}
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_url": {"type": "string"},
                    "source_title": {"type": "string"}
                },
                "required": ["source_url", "source_title"]
            }
        },
        "last_updated": {"type": "string", "format": "date-time"}
    },
    "required": [
        "tech_id", "name", "definition", "method",
        "technical_features", "applications", "evidence", "last_updated"
    ]
}

def _infer_title_from_url(url: str) -> str | None:
    try:
        p = urlparse(url.strip())
        if p.scheme not in ("http", "https") or not p.netloc:
            return None
        path_parts = [seg for seg in p.path.split("/") if seg]
        if path_parts:
            tail = unquote(path_parts[-1]).replace("-", " ").replace("_", " ").strip()
            if "." in tail:
                base, ext = tail.rsplit(".", 1)
                if ext.lower() in ("html", "htm", "php", "aspx", "jsp"):
                    tail = base
            candidate = tail or p.netloc
        else:
            candidate = p.netloc
        title = candidate[:120].strip()
        return (title[0].upper() + title[1:]) if title else None
    except Exception:
        return None

def _normalize_evidence(card: dict) -> None:
    raw = card.get("evidence", [])
    fixed = []
    if not isinstance(raw, list):
        raw = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("source_url", "")).strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        title = str(item.get("source_title", "")).strip()
        if not title:
            guess = _infer_title_from_url(url)
            title = guess or "Untitled"
        fixed.append({"source_url": url, "source_title": title[:120]})
    card["evidence"] = fixed

def build_prompt(tech_name: str) -> str:
    return (
        "Return a JSON object with fields: "
        "- tech_id (as provided), "
        "- name (as provided), "
        "- definition (2–4 sentences; English; clear genus–differentia), "
        "- method (4–6 sentences on operation principles and stages; English), "
        "- technical_features (exactly 5–7 concise bullets ≤15 words; English), "
        "- applications (exactly 3–4 bullets with industry use cases; English), "
        "- evidence (3–5 items; each item MUST have non-empty 'source_url' (http/https) "
        "  and non-empty 'source_title' (page/article title)), "
        "- last_updated (current date in ISO 8601 UTC), "
        "- embedding_snippet (single paragraph, 250–300 tokens; English; structure: "
        "  1) genus–differentia opening; "
        "  2) 4–6 discriminators with numeric ranges and SI or computing metrics; "
        "  3) 2–3 main applications; "
        "  4) short 'Not to be confused with …'). "
        "Rules: "
        "- Language: English, neutral, no marketing; "
        "- Use numbers with units (e.g., 10–50 TOPS, 5–20 nm, 600–1100 °C, ms, %, kt/yr); "
        "- For 'evidence': only public web sources; valid http/https URLs; meaningful non-empty titles; "
        "- If data is missing, write 'data limited' rather than inventing; "
        "- Output strictly valid JSON with only the listed fields. "
        f"Technology: {tech_name}."
    )

import re

def _extract_balanced_json(text: str) -> str | None:
    """
    Извлекает ПЕРВЫЙ балансный JSON-объект из текста (по фигурным скобкам).
    Возвращает строку "{...}" или None.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2018": "'", "\u2019": "'", "\u2032": "'", "\u2033": '"'
}

def _clean_json_string(s: str) -> str:
    """
    Бережная чистка: убираем ```json / ``` fences, BOM, неразрывные пробелы,
    «умные» кавычки; убираем висячие запятые перед } и ].
    НЕ меняем структуру ключей/значений.
    """
    # code fences
    s = re.sub(r"^\s*```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    # BOM/nbsp
    s = s.replace("\ufeff", "").replace("\u00a0", " ")
    # smart quotes → ascii
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    # trailing commas:  , }
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s

def safe_json_loads_from_content(content: str) -> dict:
    """
    Пытается распарсить JSON из ответа LLM: балансная вырезка + чистка + json.loads.
    При неудаче пишет сырьё в failed_json.log и поднимает JSONDecodeError дальше.
    """
    raw_block = _extract_balanced_json(content) or content
    cleaned = _clean_json_string(raw_block)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # логируем для последующего анализа
        os.makedirs("errors", exist_ok=True)
        with open("errors/failed_json.log", "a", encoding="utf-8") as f:
            f.write("\n--- RAW CONTENT ---\n")
            f.write(content)
            f.write("\n--- EXTRACTED ---\n")
            f.write(raw_block)
            f.write("\n--- CLEANED ---\n")
            f.write(cleaned)
            f.write(f"\n--- ERROR: {e}\n")
        raise

# =========================
# Async DeepSeek helpers
# =========================

async def _fetch_evidence_only_async(session: ClientSession, tech_name: str, headers):
    fix_prompt = (
        "Return JSON with field 'evidence' ONLY. "
        "Constraints: array of 3–5 items; each with non-empty 'source_url' (http/https) "
        "and non-empty 'source_title'. Do not invent URLs. "
        f"Technology: {tech_name}."
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You extract high-quality sources for technology cards."},
            {"role": "user", "content": fix_prompt}
        ],
        "max_tokens": 350,
        "temperature": 0
    }
    async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=90) as r:
        r.raise_for_status()
        jd = await r.json()
        content = jd["choices"][0]["message"]["content"].strip()
        data = safe_json_loads_from_content(content)
        ev = data.get("evidence", [])
        return ev if isinstance(ev, list) else []
        return []

async def _fetch_card(session: ClientSession, tech_id, tech_name, headers):
    request_payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an expert assistant that returns technology cards in JSON format."},
            {"role": "user", "content": build_prompt(tech_name)}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0
    }
    for attempt in range(5):
        try:
            async with session.post(DEEPSEEK_API_URL, headers=headers, json=request_payload, timeout=90) as resp:
                resp.raise_for_status()
                jd = await resp.json()
                content = jd["choices"][0]["message"]["content"].strip()
                result_json = safe_json_loads_from_content(content)

                # обязательные поля
                result_json["tech_id"] = tech_id
                result_json["name"] = tech_name
                result_json["last_updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                # evidence: нормализация + (если мало) дозапрос
                _normalize_evidence(result_json)
                if len(result_json.get("evidence", [])) < 3:
                    try:
                        extra_ev = await _fetch_evidence_only_async(session, tech_name, headers)
                        if extra_ev:
                            result_json["evidence"] = (result_json.get("evidence", []) + extra_ev)[:5]
                            _normalize_evidence(result_json)
                    except Exception:
                        pass

                # валидация схемой
                validate(instance=result_json, schema=tech_card_schema)

                # потоковая запись по мере готовности
                with open('cards.jsonl', 'a', encoding='utf-8') as outfile:
                    outfile.write(json.dumps(result_json, ensure_ascii=False) + '\n')

                return result_json
        except Exception as e:
            print(f"[RETRY] tech_id={tech_id} attempt failed: {e}")
            await asyncio.sleep(min(60, 2 ** attempt))
    return None

async def build_cards_async(rows):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    # подчистим файл перед записью
    open('cards.jsonl', 'w', encoding='utf-8').close()

    cards_by_index = [None] * len(rows)
    sem = asyncio.Semaphore(CONCURRENCY)

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=SSL_CONTEXT)

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, connector=connector) as session:
        pbar = tqdm(total=len(rows), desc="Building tech cards", unit="card")
        async def worker(idx, row):
            tech_id = row.get('tech_id'); tech_name = row.get('name')
            if not tech_id or not tech_name:
                pbar.update(1)
                return
            try:
                async with sem:
                    card = await _fetch_card(session, tech_id, tech_name, headers)
                    cards_by_index[idx] = card
            except Exception as e:
                print(f"[ERROR] worker idx={idx}, tech_id={tech_id}: {e}")
            finally:
                pbar.update(1)

        tasks = [asyncio.create_task(worker(i, r)) for i, r in enumerate(rows)]
        await asyncio.gather(*tasks)  # не подавляем исключения
        pbar.close()

    # сохраняем порядок входа, фильтруя None
    return [c for c in cards_by_index if c]

# =========================
# Main: build cards
# =========================

def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        return list(reader)

def main():
    print("[INIT] Starting…")
    print("[INIT] DEEPSEEK_API_KEY loaded:", bool(os.getenv("DEEPSEEK_API_KEY")))
    print("[INIT] DASHSCOPE_API_KEY loaded:", bool(os.getenv("DASHSCOPE_API_KEY")))

    # Загрузка CSV
    try:
        rows = load_rows('tech_list.csv')
        print(f"[CSV] Loaded rows: {len(rows)}")
    except FileNotFoundError:
        print("[Error] CSV file 'tech_list.csv' not found. Please ensure the file exists.")
        rows = []

    # Генерация карточек асинхронно
    print("[STAGE] Building cards (async)…")
    tech_cards = asyncio.run(build_cards_async(rows))
    print(f"[STAGE] Built cards: {len(tech_cards)} (see cards.jsonl)")

    # Если ничего не построилось — смоук-тест синхронным запросом (диагностика ключа/эндпоинта)
    if not tech_cards and rows:
        try:
            test_name = rows[0].get('name', 'TEST_TECH')
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert assistant that returns technology cards in JSON format."},
                    {"role": "user", "content": build_prompt(test_name)}
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0
            }
            r = session.post(DEEPSEEK_API_URL,
                             headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                                      "Content-Type": "application/json"},
                             json=payload, timeout=90)
            r.raise_for_status()
            print("[SMOKE] Sync request succeeded — проверьте асинхронный блок/лимиты API.")
        except Exception as e:
            print(f"[SMOKE] Sync request failed: {e}")

    # Подготовка текстов для эмбеддингов
    print("[STAGE] Preparing embedding texts…")
    texts_to_embed = []
    tech_id_list = []
    for card in tech_cards:
        embed_text = card.get('embedding_snippet', '')
        if not embed_text:
            definition = card.get('definition', '')
            method = card.get('method', '')
            features = " ".join(card.get('technical_features', []))
            applications = " ".join(card.get('applications', []))
            fallback = f"{definition} {method} {features} {applications}".split()
            embed_text = " ".join(fallback[:220])  # ~180–220 words fallback
        texts_to_embed.append(embed_text)
        tech_id_list.append(card.get('tech_id'))

    # Кэш эмбеддингов
    print(f"[STAGE] Embedding {len(texts_to_embed)} texts in batches…")
    EMB_CACHE_PATH = "emb_cache.jsonl"
    _emb_cache: dict[str, list[float]] = {}

    def _h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def cache_load():
        if os.path.exists(EMB_CACHE_PATH):
            with open(EMB_CACHE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        _emb_cache[rec["h"]] = rec["v"]
                    except:
                        pass

    def cache_append(h, v):
        with open(EMB_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"h": h, "v": v}) + "\n")

    cache_load()

    texts_to_calc, calc_indices = [], []
    final_vectors: list[list[float] | None] = [None] * len(texts_to_embed)
    for i, t in enumerate(texts_to_embed):
        hh = _h(t)
        if hh in _emb_cache:
            final_vectors[i] = _emb_cache[hh]
        else:
            texts_to_calc.append(t); calc_indices.append(i)

    # Батч-эмбеддинги через DashScope (OpenAI-совместимый формат)
    def batched(iterable, n):
        batch = []
        for x in iterable:
            batch.append(x)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    headers_emb = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}

    cursor = 0
    for batch in tqdm(list(batched(texts_to_calc, BATCH_SIZE_EMB)), desc="Embedding batches", unit="batch"):
        for attempt in range(5):
            try:
                resp = session.post(
                    DASHSCOPE_API_URL,
                    headers=headers_emb,
                    json={"model": DASHSCOPE_MODEL, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data.get("data"), list):
                    # ожидаем список {index, embedding, ...}; если index нет — используем порядок
                    emb_list = data["data"]
                    # если у элементов есть index — отсортируем; иначе берём порядок
                    if all(isinstance(item.get("index", 0), int) for item in emb_list):
                        emb_list = sorted(emb_list, key=lambda d: d.get("index", 0))
                    for k, item in enumerate(emb_list):
                        vec = item["embedding"]
                        global_i = calc_indices[cursor + k]
                        final_vectors[global_i] = vec
                        cache_append(_h(texts_to_calc[cursor + k]), vec)
                else:
                    # fallback на одиночный ответ
                    vec = data["data"][0]["embedding"]
                    global_i = calc_indices[cursor]
                    final_vectors[global_i] = vec
                    cache_append(_h(texts_to_calc[cursor]), vec)
                break
            except requests.RequestException as e:
                print(f"[RETRY-EMB] batch at {cursor} failed: {e}")
                time.sleep(min(60, 2 ** attempt))
        cursor += len(batch)
        time.sleep(0.5)  # мягкий троттлинг

    if any(v is None for v in final_vectors):
        missing = sum(1 for v in final_vectors if v is None)
        print(f"[WARN] Missing embeddings for {missing} items.")

    # Сохранение результатов
    print("[STAGE] Saving files…")
    embedding_array = np.array(final_vectors, dtype=object)
    try:
        np.save('embeddings.npy', embedding_array)
    except Exception as e:
        print(f"[Error] Could not save embeddings.npy: {e}")

    tech_id_to_index = {str(tid): idx for idx, tid in enumerate(tech_id_list)}
    try:
        with open('embeddings_map.json', 'w', encoding='utf-8') as f:
            json.dump(tech_id_to_index, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Error] Could not save embeddings_map.json: {e}")

    print("[DONE] All stages finished.")

if __name__ == "__main__":
    main()
