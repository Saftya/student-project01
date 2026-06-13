import hashlib
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sklearn.decomposition import PCA

from rag_core import (
    ablation_experiment,
    build_index,
    build_prompt,
    chunk_text_with_spans,
    evaluate,
    generate_answer,
    get_cohere_client,
    load_articles,
    load_questions,
    make_chunks,
    retrieve,
    safe_collection_name,
)


DEMO_DATA_DIR = "data_demo"
DEMO_GROUND_TRUTH = "ground_truth_demo.json"
CHROMA_PATH = "/app/chroma_demo_ui"
RETRIEVAL_MODES = ["Vector", "Rerank", "BM25", "Hybrid RRF"]
FAILURE_QUERIES = [
    "Which model is best for every retrieval problem?",
    "Did BERT use a lunar rover during the Apollo program?",
    "What is the exact cost of building ChromaDB?",
    "Can a vector database guarantee that a generated answer is true?",
]


st.set_page_config(
    page_title="Interactive RAG Lab",
    page_icon=":mag:",
    layout="wide",
)


@st.cache_data
def cached_articles():
    return load_articles(DEMO_DATA_DIR)


@st.cache_data
def cached_questions():
    return load_questions(DEMO_GROUND_TRUTH)


@st.cache_resource(show_spinner=False)
def cached_client():
    load_dotenv()
    return get_cohere_client()


def settings_key(chunk_size, overlap, selected_sources):
    raw = f"{chunk_size}:{overlap}:{','.join(selected_sources)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


@st.cache_resource(show_spinner="Embedding chunks and building Chroma index...")
def cached_bundle(chunk_size, overlap, selected_sources):
    articles = cached_articles()
    co = cached_client()
    chunks, sources = make_chunks(
        articles,
        size=chunk_size,
        overlap=overlap,
        selected_sources=selected_sources,
    )
    collection_name = safe_collection_name(
        f"wiki_demo_ui_{chunk_size}_{overlap}_{settings_key(chunk_size, overlap, selected_sources)}"
    )
    return build_index(
        chunks,
        sources,
        co=co,
        path=CHROMA_PATH,
        collection_name=collection_name,
    )


def preview(text, full=False):
    if full or len(text) <= 420:
        return text
    return text[:420].strip() + "..."


def result_rows(results, show_full=False):
    rows = []
    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    distances = results.get("distances", [])
    scores = results.get("scores", [])
    for index, chunk_id in enumerate(ids):
        source = metas[index].get("source", "") if index < len(metas) else ""
        row = {
            "rank": index + 1,
            "id": chunk_id,
            "source": source,
            "preview": preview(docs[index], show_full),
        }
        if index < len(distances):
            row["distance"] = round(float(distances[index]), 4)
        if index < len(scores):
            row["score"] = round(float(scores[index]), 4)
        rows.append(row)
    return rows


def metric_card(label, value):
    st.metric(label, f"{value:.2f}")


def hit_badge(correct_id, ids, top_k):
    if not correct_id:
        st.info("Choose a ground-truth question to see hit/miss.")
        return
    if correct_id in ids[:top_k]:
        st.success(f"Correct chunk found: {correct_id}")
    else:
        st.error(f"Missed correct chunk: {correct_id}")


def latency_table(results):
    latency = results.get("latency_ms", {})
    if latency:
        st.dataframe(
            pd.DataFrame(
                [{"step": key, "ms": round(value, 1)} for key, value in latency.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )


def chunk_boundary_rows(article_name, text, chunk_size, overlap):
    spans = chunk_text_with_spans(text, size=chunk_size, overlap=overlap)
    rows = []
    for index, span in enumerate(spans):
        rows.append(
            {
                "id": f"{article_name}_{index}",
                "start": span["start"],
                "end": span["end"],
                "chars": span["end"] - span["start"],
                "preview": preview(span["text"]),
            }
        )
    return rows


def show_rank_movement(results):
    movement = results.get("rank_movement", [])
    if not movement:
        st.caption("Rank movement appears when Rerank mode is used.")
        return
    st.dataframe(pd.DataFrame(movement), use_container_width=True, hide_index=True)


def source_mix(results):
    sources = [
        metadata.get("source", "")
        for metadata in results.get("metadatas", [])
        if metadata.get("source")
    ]
    return len(set(sources)), sources


articles = cached_articles()
questions = cached_questions()

with st.sidebar:
    st.title("RAG Lab")
    st.caption("Teacher demo for retrieval, evaluation, and RAG behavior.")

    all_sources = list(articles.keys())
    selected_sources = st.multiselect(
        "Sources",
        options=all_sources,
        default=all_sources,
    )
    if not selected_sources:
        st.warning("Select at least one source.")
        st.stop()

    chunk_size = st.slider("Chunk size", 200, 1200, 500, step=50)
    overlap = st.slider("Overlap", 0, min(200, chunk_size - 50), 50, step=25)
    top_k = st.slider("Top k", 1, 10, 3)
    retrieval_mode = st.radio("Retrieval mode", RETRIEVAL_MODES, horizontal=False)
    generate = st.toggle("Generate RAG answer", value=False)
    temperature = st.slider("Generation temperature", 0.0, 1.0, 0.1, step=0.1)
    show_full_chunks = st.toggle("Show full chunks", value=False)

bundle = cached_bundle(chunk_size, overlap, tuple(selected_sources))

st.title("Interactive RAG Lab")
st.caption(
    "Move the controls, watch retrieval change, then let students rebuild the same ideas in CLI."
)

tab_pipeline, tab_chunks, tab_arena, tab_eval, tab_map, tab_failure, tab_playground = st.tabs(
    [
        "Pipeline",
        "Chunk Lab",
        "Search Arena",
        "Evaluation",
        "Embedding Map",
        "Failure Lab",
        "RAG Playground",
    ]
)

with tab_pipeline:
    st.subheader("Query to retrieval to answer")
    question_options = ["Custom question"] + [item["q"] for item in questions]
    selected_question = st.selectbox("Question", question_options, key="pipeline_question")
    if selected_question == "Custom question":
        query = st.text_input(
            "Query",
            value="How does cosine similarity compare text embeddings?",
            key="pipeline_custom_query",
        )
        correct_id = None
    else:
        query = selected_question
        correct_id = next(item["correct_id"] for item in questions if item["q"] == query)

    query = query.strip() or "How does cosine similarity compare text embeddings?"

    started = time.perf_counter()
    results = retrieve(query, bundle, mode=retrieval_mode, top_k=top_k)
    total_ms = (time.perf_counter() - started) * 1000

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Chunks indexed", len(bundle.documents))
    col_b.metric("Top k", top_k)
    col_c.metric("Sources in top-k", source_mix(results)[0])
    col_d.metric("Total retrieval ms", f"{total_ms:.1f}")

    hit_badge(correct_id, results["ids"], top_k)
    st.dataframe(
        pd.DataFrame(result_rows(results, show_full_chunks)),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Latency details"):
        latency_table(results)
    with st.expander("Rank movement"):
        show_rank_movement(results)

    if generate:
        answer = generate_answer(query, bundle.co, results["documents"], temperature=temperature)
        st.markdown("#### RAG answer")
        st.write(answer["answer"])
        st.caption(f"Generation latency: {answer['latency_ms']:.1f} ms")

with tab_chunks:
    st.subheader("Chunk boundary microscope")
    article_name = st.selectbox("Article", all_sources, key="chunk_article")
    rows = chunk_boundary_rows(article_name, articles[article_name], chunk_size, overlap)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Document chars", len(articles[article_name]))
    col_b.metric("Chunks", len(rows))
    col_c.metric("Overlap", overlap)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected_chunk = st.selectbox("Inspect chunk", [row["id"] for row in rows])
    selected_row = next(row for row in rows if row["id"] == selected_chunk)
    text = articles[article_name][selected_row["start"] : selected_row["end"]]
    st.text_area("Chunk text", value=text, height=240)

with tab_arena:
    st.subheader("Search Arena")
    arena_query = st.text_input(
        "Compare retrieval modes",
        value="Why does reranking help vector search?",
        key="arena_query",
    ).strip() or "Why does reranking help vector search?"
    cols = st.columns(4)
    for column, mode in zip(cols, RETRIEVAL_MODES):
        with column:
            st.markdown(f"#### {mode}")
            mode_results = retrieve(arena_query, bundle, mode=mode, top_k=top_k)
            st.dataframe(
                pd.DataFrame(result_rows(mode_results, show_full_chunks)),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Sources in top-k: {source_mix(mode_results)[0]} | "
                f"Latency: {mode_results.get('latency_ms', {}).get('total', 0):.1f} ms"
            )

with tab_eval:
    st.subheader("Evaluate before generation")
    eval_mode = st.selectbox("Evaluation mode", RETRIEVAL_MODES, index=0)
    evaluation = evaluate(questions, bundle, mode=eval_mode, ks=(1, 3, 5))
    metrics = evaluation["metrics"]
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        metric_card("Recall@1", metrics["recall@1"])
    with col_b:
        metric_card("Recall@3", metrics["recall@3"])
    with col_c:
        metric_card("Recall@5", metrics["recall@5"])
    with col_d:
        metric_card("MRR", metrics["mrr"])

    details = pd.DataFrame(evaluation["details"])
    details["hit@3"] = details["recall@3"].map({True: "hit", False: "miss"})
    st.dataframe(details, use_container_width=True, hide_index=True)

    st.markdown("#### Ablation chart")
    if st.button("Run ablation for current mode"):
        rows = ablation_experiment(
            articles,
            questions,
            bundle.co,
            path=CHROMA_PATH,
            collection_prefix="ui_ablation",
            sizes=(300, 500, 1000),
            overlap=50,
            mode=eval_mode,
        )
        ablation_df = pd.DataFrame(rows)
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                ablation_df,
                x="chunk_size",
                y=["recall@1", "recall@3", "recall@5", "mrr"],
                barmode="group",
                title="Retrieval quality by chunk size",
            ),
            use_container_width=True,
        )

with tab_map:
    st.subheader("Embedding Map")
    map_query = st.text_input(
        "Highlight query",
        value="What is cosine similarity?",
        key="map_query",
    ).strip() or "What is cosine similarity?"
    map_results = retrieve(map_query, bundle, mode="Vector", top_k=top_k)
    vectors = list(bundle.embeddings) + [map_results["query_embedding"]]
    labels = bundle.ids + ["QUERY"]
    sources = bundle.sources + ["query"]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "id": labels,
            "source": sources,
            "top_k": [chunk_id in map_results["ids"] for chunk_id in labels],
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        symbol="top_k",
        hover_data=["id"],
        title="2D PCA projection of chunk embeddings",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is a projection, not the real high-dimensional space. Use it as intuition, not proof."
    )

with tab_failure:
    st.subheader("Failure Lab")
    failure_query = st.selectbox("Try a tricky query", FAILURE_QUERIES)
    failure_results = retrieve(failure_query, bundle, mode=retrieval_mode, top_k=top_k)
    st.dataframe(
        pd.DataFrame(result_rows(failure_results, show_full_chunks)),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Discussion prompt: did the retriever find evidence, or only text that sounds related?"
    )
    if generate:
        answer = generate_answer(
            failure_query,
            bundle.co,
            failure_results["documents"],
            temperature=temperature,
        )
        st.markdown("#### Generated answer")
        st.write(answer["answer"])
        with st.expander("Prompt sent to the model"):
            st.code(answer["prompt"])

with tab_playground:
    st.subheader("RAG Playground")
    playground_query = st.text_area(
        "Question",
        value="Why can a RAG system hallucinate when retrieval is poor?",
        height=90,
    ).strip() or "Why can a RAG system hallucinate when retrieval is poor?"
    playground_results = retrieve(
        playground_query,
        bundle,
        mode=retrieval_mode,
        top_k=top_k,
    )
    selected_ids = st.multiselect(
        "Context chunks to include",
        options=playground_results["ids"],
        default=playground_results["ids"][: min(3, len(playground_results["ids"]))],
    )
    selected_docs = [
        doc
        for chunk_id, doc in zip(playground_results["ids"], playground_results["documents"])
        if chunk_id in selected_ids
    ]
    prompt = build_prompt(playground_query, selected_docs)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Selected context")
        for chunk_id, doc in zip(playground_results["ids"], playground_results["documents"]):
            if chunk_id in selected_ids:
                st.markdown(f"**{chunk_id}**")
                st.write(preview(doc, show_full_chunks))
    with col_right:
        st.markdown("#### Prompt")
        st.code(prompt)

    if st.button("Generate answer from selected context"):
        generated = generate_answer(
            playground_query,
            bundle.co,
            selected_docs,
            temperature=temperature,
        )
        st.markdown("#### Answer")
        st.write(generated["answer"])
        st.caption(f"Source ids: {', '.join(selected_ids)}")
