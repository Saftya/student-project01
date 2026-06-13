import json
import os

import chromadb
import cohere
from dotenv import load_dotenv

# BONUS imports:
# from rank_bm25 import BM25Okapi
# from sklearn.preprocessing import minmax_scale


ASSIGNMENT_DATA_DIR = "data_assignment"
ASSIGNMENT_GROUND_TRUTH = "ground_truth_assignment.json"
CHROMA_PATH = "/app/chroma_assignment"
COLLECTION_NAME = "space_assignment"


def load_articles(data_dir=ASSIGNMENT_DATA_DIR):
    articles = {}
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            name = filename.replace(".txt", "")
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as file:
                articles[name] = file.read()
    return articles


def chunk_text(text, size=500, overlap=50):
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def make_chunks(articles, size=500, overlap=50):
    all_chunks = []
    sources = []
    for source, text in articles.items():
        chunks = chunk_text(text, size=size, overlap=overlap)
        all_chunks.extend(chunks)
        sources.extend([source] * len(chunks))
    return all_chunks, sources


def load_questions(path=ASSIGNMENT_GROUND_TRUTH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_index(all_chunks, sources):
    api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key)

    resp = co.embed(
        texts=all_chunks,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    vectors = resp.embeddings

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    metadatas = []
    for i, source in enumerate(sources):
            ids.append(f"{source}_{i}")
            metadatas.append({"source": source})

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=vectors,
        metadatas=metadatas,
    )

    return collection, co



def search(query, co, collection, top_k=5):
    # resp = co.embed(
    #     texts=[query],
    #     model="embed-english-v3.0",
    #     input_type="search_query",
    # )
    # vectors = resp.embeddings[0]
        
    #     collection.query(
    #         query_embeddings=[вектор], n_results=top_k)(
    #     ids=ids,
    #     documents=all_chunks,
    #     embeddings=vectors,
    #     metadatas=metadatas,
    # )

    return collection, co
    # HINT: query нужно эмбеддить с input_type="search_query".
    # HINT: collection.query возвращает вложенные списки, поэтому достань первый список ids и documents.
    # HINT: верни dict с ключами "ids" и "documents".
    pass


def evaluate(questions, co, collection):
    # HINT: для каждого вопроса вызови search(..., top_k=3).
    # HINT: если correct_id есть в первых 3 ids, это hit.
    # HINT: верни hits / len(questions).
    pass


def search_with_rerank(query, co, collection, top_k=3):
    # HINT: сначала сделай обычный search top_k=10.
    # HINT: затем передай найденные documents в co.rerank(..., top_n=top_k).
    # HINT: верни ids и documents в новом порядке после rerank.
    # HINT: формат результата такой же, как у search: dict с "ids" и "documents".
    pass


def rag_answer(query, co, collection):
    # HINT: достань top-3 чанка через rerank или обычный search.
    # HINT: склей chunks через "\n\n" и попроси модель отвечать только по context.
    # HINT: верни dict с ключами "answer" и "source_ids", чтобы показать grounding.
    pass


def ablation_experiment(articles, questions, co):
    # HINT: проверь chunk_size 300, 500, 1000 при overlap=50.
    # HINT: для каждого размера пересоздай чанки, индекс и посчитай evaluate.
    # HINT: напечатай size, количество чанков и accuracy.
    pass


def hybrid_search(query, co, collection, all_chunks, bm25):
    # BONUS HINT: BM25 хорошо ищет точные слова, vector search хорошо ищет смысл.
    # BONUS HINT: объедини ранги через RRF: 1/(60 + bm25_rank) + 1/(60 + vector_rank).
    # BONUS HINT: верни top-3 ids.
    pass


if __name__ == "__main__":
    load_dotenv()

    articles = load_articles()
    all_chunks, sources = make_chunks(articles, size=500, overlap=50)
    questions = load_questions()

    print(f"Loaded articles: {len(articles)}")
    print(f"Created chunks: {len(all_chunks)}")
    print(f"Loaded questions: {len(questions)}")

    # Раскомментируй по порядку:

    collection, co = build_index(all_chunks, sources)
    print(collection.count())
    # results = search("Why was Jezero crater important for Perseverance?", co, collection)
    # print(results["ids"])

    # accuracy = evaluate(questions, co, collection)
    # print(f"Recall@3: {accuracy:.2f}")

    # reranked = search_with_rerank("Why was Jezero crater important for Perseverance?", co, collection)
    # print(reranked["ids"])

    # answer = rag_answer("How did Ingenuity help Mars exploration?", co, collection)
    # print(answer)

    # ablation_experiment(articles, questions, co)
