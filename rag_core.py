import json
import os
import re
import time
from dataclasses import dataclass

import chromadb
import cohere
from chromadb.config import Settings
from rank_bm25 import BM25Okapi


EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-v3.5"
GENERATE_MODEL = "command-r7b-12-2024"


@dataclass
class IndexBundle:
    collection: object
    co: object
    ids: list[str]
    documents: list[str]
    sources: list[str]
    embeddings: list[list[float]]


def load_articles(data_dir):
    articles = {}
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            name = filename.replace(".txt", "")
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as file:
                articles[name] = file.read()
    return articles


def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def chunk_text(text, size=500, overlap=50):
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text_with_spans(text, size=500, overlap=50):
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        raw = text[start : start + size]
        chunk = raw.strip()
        if chunk:
            offset = len(raw) - len(raw.lstrip())
            chunks.append(
                {
                    "start": start + offset,
                    "end": start + offset + len(chunk),
                    "text": chunk,
                }
            )
    return chunks


def make_chunks(articles, size=500, overlap=50, selected_sources=None):
    selected = set(selected_sources or articles.keys())
    all_chunks = []
    sources = []
    for source, text in articles.items():
        if source not in selected:
            continue
        chunks = chunk_text(text, size=size, overlap=overlap)
        all_chunks.extend(chunks)
        sources.extend([source] * len(chunks))
    return all_chunks, sources


def make_ids(sources):
    counters = {}
    ids = []
    for source in sources:
        index = counters.get(source, 0)
        ids.append(f"{source}_{index}")
        counters[source] = index + 1
    return ids


def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is missing. Add it to .env first.")
    return cohere.Client(api_key)


def embed_texts(co, texts, input_type):
    if not texts:
        return []
    response = call_with_retries(
        lambda: co.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type=input_type,
        )
    )
    embeddings = response.embeddings
    if hasattr(embeddings, "float"):
        return embeddings.float
    return embeddings


def call_with_retries(fn, attempts=3, base_sleep=2):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                marker in message
                for marker in [
                    "too many requests",
                    "rate limit",
                    "server disconnected",
                    "remote protocol",
                    "timeout",
                    "temporarily unavailable",
                ]
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_sleep * (attempt + 1))
    raise last_error


def safe_collection_name(name):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return cleaned[:63].strip("_") or "rag_collection"


def reset_collection(path, collection_name):
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


def build_index(
    all_chunks,
    sources,
    co=None,
    path="/app/chroma_demo",
    collection_name="wiki_demo",
):
    co = co or get_cohere_client()
    ids = make_ids(sources)
    embeddings = embed_texts(co, all_chunks, input_type="search_document")
    metadatas = [{"source": source} for source in sources]
    collection = reset_collection(path=path, collection_name=collection_name)
    if all_chunks:
        collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    return IndexBundle(
        collection=collection,
        co=co,
        ids=ids,
        documents=all_chunks,
        sources=sources,
        embeddings=embeddings,
    )


def search(query, co, collection, top_k=5):
    top_k = min(top_k, collection.count())
    if top_k <= 0 or not query.strip():
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "query_embedding": None,
            "latency_ms": {"embed": 0.0, "search": 0.0, "total": 0.0},
            "ranks": {},
        }

    started = time.perf_counter()
    query_embedding = embed_texts(co, [query], input_type="search_query")[0]
    embedded_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    search_ms = (time.perf_counter() - started) * 1000

    ids = results["ids"][0]
    return {
        "ids": ids,
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results.get("distances", [[]])[0],
        "query_embedding": query_embedding,
        "latency_ms": {
            "embed": embedded_ms,
            "search": search_ms,
            "total": embedded_ms + search_ms,
        },
        "ranks": {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)},
    }


def bm25_search(query, ids, documents, sources, top_k=5):
    if not query.strip():
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": [],
            "latency_ms": {"total": 0.0},
            "ranks": {},
        }

    started = time.perf_counter()
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[
        :top_k
    ]
    elapsed = (time.perf_counter() - started) * 1000
    selected_ids = [ids[index] for index in order]
    return {
        "ids": selected_ids,
        "documents": [documents[index] for index in order],
        "metadatas": [{"source": sources[index]} for index in order],
        "scores": [float(scores[index]) for index in order],
        "latency_ms": {"total": elapsed},
        "ranks": {chunk_id: rank for rank, chunk_id in enumerate(selected_ids, start=1)},
    }


def search_with_rerank(query, co, collection, top_k=3, candidate_k=10):
    started_total = time.perf_counter()
    initial = search(query, co, collection, top_k=candidate_k)
    documents = initial["documents"]
    if not documents:
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "baseline_ids": [],
            "rank_movement": [],
            "latency_ms": initial["latency_ms"],
        }

    started = time.perf_counter()
    reranked = call_with_retries(
        lambda: co.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )
    )
    rerank_ms = (time.perf_counter() - started) * 1000

    ids = []
    docs = []
    metas = []
    movement = []
    for new_rank, result in enumerate(reranked.results, start=1):
        original_index = result.index
        chunk_id = initial["ids"][original_index]
        ids.append(chunk_id)
        docs.append(documents[original_index])
        metas.append(initial["metadatas"][original_index])
        movement.append(
            {
                "id": chunk_id,
                "baseline_rank": original_index + 1,
                "rerank_rank": new_rank,
            }
        )

    latency = dict(initial["latency_ms"])
    latency["rerank"] = rerank_ms
    latency["total"] = (time.perf_counter() - started_total) * 1000
    return {
        "ids": ids,
        "documents": docs,
        "metadatas": metas,
        "baseline_ids": initial["ids"],
        "rank_movement": movement,
        "latency_ms": latency,
        "ranks": {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)},
    }


def hybrid_search(query, co, collection, ids, documents, sources, top_k=5):
    started = time.perf_counter()
    candidate_k = min(max(top_k * 3, 10), len(ids))
    vector = search(query, co, collection, top_k=candidate_k)
    bm25 = bm25_search(query, ids, documents, sources, top_k=candidate_k)

    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector["ids"], start=1)}
    bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25["ids"], start=1)}
    scores = {}
    for chunk_id in set(vector_rank) | set(bm25_rank):
        scores[chunk_id] = 0.0
        if chunk_id in vector_rank:
            scores[chunk_id] += 1 / (60 + vector_rank[chunk_id])
        if chunk_id in bm25_rank:
            scores[chunk_id] += 1 / (60 + bm25_rank[chunk_id])

    order = sorted(scores, key=scores.get, reverse=True)[:top_k]
    lookup = {chunk_id: index for index, chunk_id in enumerate(ids)}
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "ids": order,
        "documents": [documents[lookup[chunk_id]] for chunk_id in order],
        "metadatas": [{"source": sources[lookup[chunk_id]]} for chunk_id in order],
        "scores": [scores[chunk_id] for chunk_id in order],
        "latency_ms": {"total": elapsed},
        "ranks": {chunk_id: rank for rank, chunk_id in enumerate(order, start=1)},
    }


def retrieve(query, bundle, mode="Vector", top_k=5):
    if mode == "Rerank":
        return search_with_rerank(query, bundle.co, bundle.collection, top_k=top_k)
    if mode == "BM25":
        return bm25_search(query, bundle.ids, bundle.documents, bundle.sources, top_k=top_k)
    if mode == "Hybrid RRF":
        return hybrid_search(
            query,
            bundle.co,
            bundle.collection,
            bundle.ids,
            bundle.documents,
            bundle.sources,
            top_k=top_k,
        )
    return search(query, bundle.co, bundle.collection, top_k=top_k)


def evaluate(questions, bundle, mode="Vector", ks=(1, 3, 5)):
    details = []
    reciprocal_ranks = []
    max_k = max(ks)

    if mode == "Vector":
        query_embeddings = embed_texts(
            bundle.co,
            [item["q"] for item in questions],
            input_type="search_query",
        )
        raw_results = bundle.collection.query(
            query_embeddings=query_embeddings,
            n_results=min(max_k, bundle.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        result_ids = raw_results["ids"]
    else:
        result_ids = [
            retrieve(item["q"], bundle, mode=mode, top_k=max_k)["ids"]
            for item in questions
        ]

    for item, ids in zip(questions, result_ids):
        correct_id = item["correct_id"]
        rank = ids.index(correct_id) + 1 if correct_id in ids else None
        reciprocal_ranks.append(1 / rank if rank else 0)
        row = {
            "q": item["q"],
            "correct_id": correct_id,
            "rank": rank,
            "retrieved_ids": ids,
        }
        for k in ks:
            row[f"recall@{k}"] = correct_id in ids[:k]
        details.append(row)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = sum(row[f"recall@{k}"] for row in details) / len(details)
    metrics["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return {"metrics": metrics, "details": details}


def build_prompt(query, documents):
    context = "\n\n".join(documents)
    return (
        "Answer using ONLY the context below. "
        "If unsure, say 'I don't know'.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(query, co, documents, temperature=0.1, max_tokens=200):
    prompt = build_prompt(query, documents)
    started = time.perf_counter()
    response = call_with_retries(
        lambda: co.chat(
            model=GENERATE_MODEL,
            message=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "answer": response.text.strip(),
        "prompt": prompt,
        "latency_ms": elapsed,
    }


def rag_answer(query, bundle, mode="Rerank", top_k=3, temperature=0.1):
    retrieved = retrieve(query, bundle, mode=mode, top_k=top_k)
    generated = generate_answer(
        query,
        bundle.co,
        retrieved["documents"],
        temperature=temperature,
    )
    return {
        "answer": generated["answer"],
        "prompt": generated["prompt"],
        "source_ids": retrieved["ids"],
        "documents": retrieved["documents"],
        "retrieval_latency_ms": retrieved.get("latency_ms", {}),
        "generation_latency_ms": generated["latency_ms"],
    }


def ablation_experiment(
    articles,
    questions,
    co,
    path="/app/chroma_demo",
    collection_prefix="ablation",
    sizes=(300, 500, 1000),
    overlap=50,
    mode="Vector",
):
    rows = []
    for size in sizes:
        chunks, sources = make_chunks(articles, size=size, overlap=overlap)
        bundle = build_index(
            chunks,
            sources,
            co=co,
            path=path,
            collection_name=safe_collection_name(f"{collection_prefix}_{size}_{overlap}"),
        )
        evaluation = evaluate(questions, bundle, mode=mode, ks=(1, 3, 5))
        rows.append(
            {
                "chunk_size": size,
                "overlap": overlap,
                "chunks": len(chunks),
                **evaluation["metrics"],
            }
        )
    return rows
