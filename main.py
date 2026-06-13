import os

from dotenv import load_dotenv

from rag_core import (
    ablation_experiment,
    build_index,
    evaluate,
    get_cohere_client,
    load_articles,
    load_questions,
    make_chunks,
    rag_answer,
    retrieve,
)


DEMO_DATA_DIR = "data_demo"
DEMO_GROUND_TRUTH = "ground_truth_demo.json"
CHROMA_PATH = "/app/chroma_demo"
COLLECTION_NAME = "wiki_demo"


if __name__ == "__main__":
    load_dotenv()

    articles = load_articles(DEMO_DATA_DIR)
    all_chunks, sources = make_chunks(articles, size=500, overlap=50)
    questions = load_questions(DEMO_GROUND_TRUTH)

    print(f"Loaded articles: {len(articles)}")
    print(f"Created chunks: {len(all_chunks)}")
    print(f"Loaded questions: {len(questions)}")

    co = get_cohere_client()
    bundle = build_index(
        all_chunks,
        sources,
        co=co,
        path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    query = "How does cosine similarity compare text embeddings?"
    search_results = retrieve(query, bundle, mode="Vector", top_k=3)
    print("\nSemantic search ids:")
    print(search_results["ids"])

    evaluation = evaluate(questions, bundle, mode="Vector", ks=(1, 3, 5))
    print("\nBaseline metrics:")
    for metric, value in evaluation["metrics"].items():
        print(f"{metric}: {value:.2f}")

    reranked = retrieve(query, bundle, mode="Rerank", top_k=3)
    print("\nReranked ids:")
    print(reranked["ids"])

    answer = rag_answer(
        "Why should retrieval quality be evaluated before adding generation?",
        bundle,
        mode="Rerank",
        top_k=3,
        temperature=0.1,
    )
    print("\nRAG answer:")
    print(answer["answer"])
    print("Sources:", answer["source_ids"])

    if os.getenv("RUN_ABLATION") == "1":
        print("\nAblation experiment:")
        rows = ablation_experiment(
            articles,
            questions,
            co,
            path=CHROMA_PATH,
            collection_prefix="wiki_demo_ablation",
            mode="Vector",
        )
        for row in rows:
            print(
                f"Size {row['chunk_size']}: chunks={row['chunks']}, "
                f"recall@3={row['recall@3']:.2f}, mrr={row['mrr']:.2f}"
            )
    else:
        print("\nAblation skipped. Run with RUN_ABLATION=1 to enable it.")
