import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(data_dir.glob("doc_*.txt")):
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    return docs


def fixed_chunks(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def recursive_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def build_chunks(strategy: str) -> list[Chunk]:
    if strategy not in {"fixed", "recursive"}:
        raise ValueError("strategy must be 'fixed' or 'recursive'")

    chunks: list[Chunk] = []
    chunker = fixed_chunks if strategy == "fixed" else recursive_chunks

    for source, text in load_documents():
        for idx, chunk_text in enumerate(chunker(text)):
            chunks.append(Chunk(id=f"{source}__{idx}", source=source, text=chunk_text))

    return chunks


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _encode_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_tfidf(corpus: list[str], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        ngram_range=(1, 2),
        max_features=50000,
    )
    doc_matrix = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
    query_matrix = vectorizer.transform(queries).toarray().astype(np.float32)
    return _normalize_rows(doc_matrix), _normalize_rows(query_matrix)


def retrieve_many(
    questions: list[str],
    strategy: str,
    k: int = 5,
    backend: str = "sentence-transformers",
) -> tuple[list[dict], list[Chunk], str]:
    chunks = build_chunks(strategy)
    corpus = [chunk.text for chunk in chunks]

    used_backend = backend
    if backend == "sentence-transformers":
        try:
            doc_embeddings = _encode_sentence_transformers(corpus)
            query_embeddings = _encode_sentence_transformers(questions)
        except Exception as exc:
            print(f"sentence-transformers failed, fallback to tfidf: {exc}")
            used_backend = "tfidf"
            doc_embeddings, query_embeddings = _encode_tfidf(corpus, questions)
    elif backend == "tfidf":
        doc_embeddings, query_embeddings = _encode_tfidf(corpus, questions)
    else:
        raise ValueError("backend must be 'sentence-transformers' or 'tfidf'")

    scores = query_embeddings @ doc_embeddings.T
    results = []
    for row in scores:
        top_idx = np.argsort(-row)[:k]
        results.append(
            {
                "ids": [chunks[i].id for i in top_idx],
                "sources": [chunks[i].source for i in top_idx],
                "scores": [float(row[i]) for i in top_idx],
                "texts": [chunks[i].text for i in top_idx],
            }
        )

    return results, chunks, used_backend


def print_stats() -> None:
    docs = load_documents()
    total_chars = sum(len(text) for _, text in docs)
    print(f"documents: {len(docs)}")
    print(f"characters: {total_chars}")
    for strategy in ["fixed", "recursive"]:
        chunks = build_chunks(strategy)
        print(f"{strategy}: {len(chunks)} chunks")


def ask(query: str, strategy: str, k: int, backend: str) -> None:
    results, _, used_backend = retrieve_many([query], strategy=strategy, k=k, backend=backend)
    print(f"backend: {used_backend}")
    print(f"strategy: {strategy}")
    for rank, (chunk_id, score, text) in enumerate(
        zip(results[0]["ids"], results[0]["scores"], results[0]["texts"]), start=1
    ):
        preview = re.sub(r"\s+", " ", text)[:300]
        print(f"{rank}. {chunk_id} score={score:.4f}\n   {preview}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats")

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    ask_parser.add_argument("--k", type=int, default=5)
    ask_parser.add_argument("--backend", choices=["sentence-transformers", "tfidf"], default="sentence-transformers")

    export_parser = sub.add_parser("export-chunks")
    export_parser.add_argument("--strategy", choices=["fixed", "recursive"], required=True)

    args = parser.parse_args()
    if args.cmd == "stats":
        print_stats()
    elif args.cmd == "ask":
        ask(args.query, strategy=args.strategy, k=args.k, backend=args.backend)
    elif args.cmd == "export-chunks":
        INDEX_DIR.mkdir(exist_ok=True)
        chunks = build_chunks(args.strategy)
        out = INDEX_DIR / f"chunks_{args.strategy}.json"
        out.write_text(
            json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved {len(chunks)} chunks to {out}")


if __name__ == "__main__":
    main()
