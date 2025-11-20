# src/rag_pipeline.py
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from .models import lm_template, generate


@dataclass
class DocChunk:
    idx: int
    content: str
    embedding: np.ndarray = None
    score: float = None


@dataclass
class LookupQuery:
    question: str


def create_embedding_model(
    emb_model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = SentenceTransformer(emb_model_name).to(device).eval()
    return model


def query_template(query: LookupQuery) -> str:
    return f"{query.question}"


def prompt_template(context: str, query: LookupQuery) -> str:
    return f"""
References:
{context}
Question:
{query.question}
Do not use markdown syntax to answer and put the answer after "Answer:".
"""


def preprocess_pages2chunks(pages_list: List[dict]) -> List[DocChunk]:
    chunked_list = []
    for page in pages_list:
        text = page["text"].strip()
        if not text:
            continue
        chunked_list.append({"text": text})

    all_chunks: List[DocChunk] = []
    for idx, chunked in enumerate(chunked_list):
        all_chunks.append(DocChunk(idx=idx, content=chunked["text"]))
    return all_chunks


def load_doc_from_path(
    documents_path: str,
    embedding_model,
) -> tuple[list[DocChunk], faiss.Index]:
    pages_list = []
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages_list.append(json.loads(line))

    chunk_list = preprocess_pages2chunks(pages_list)
    contents = [doc.content for doc in chunk_list]

    embeddings = embedding_model.encode(contents)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    for doc, emb in zip(chunk_list, embeddings):
        doc.embedding = emb

    return chunk_list, index


def retrieve_relevant_docs(
    search_query: str,
    embedding_model,
    index: faiss.Index,
    chunk_list: List[DocChunk],
    top_k: int = 3,
) -> List[DocChunk]:
    query_embedding = embedding_model.encode([search_query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)

    candidates: List[DocChunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunk_list):
            d = chunk_list[idx]
            d.score = float(score)
            candidates.append(d)

    seen = set()
    unique = []
    for d in candidates:
        if d.idx not in seen:
            seen.add(d.idx)
            unique.append(d)
    return unique


def rag_ask(
    user_query: str,
    model,
    tokenizer,
    embedding_model,
    index: faiss.Index,
    chunk_list: List[DocChunk],
    top_k: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> dict:
    query = LookupQuery(question=user_query)
    search_query = query_template(query)
    relevant_docs = retrieve_relevant_docs(
        search_query=search_query,
        embedding_model=embedding_model,
        index=index,
        chunk_list=chunk_list,
        top_k=top_k,
    )

    context = ""
    for i, doc in enumerate(relevant_docs):
        context += f"Reference {i+1}: {doc.content}\n"

    text = prompt_template(context=context, query=query)
    prompt = lm_template(text)

    response = generate(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        "response": response.strip(),
        "prompt": prompt,
        "relevant_docs": relevant_docs,
    }


def predict_with_rag_ml(
    question: str,
    model,
    tokenizer,
    embedding_model,
    index: faiss.Index,
    chunk_list: List[DocChunk],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 3,
) -> str:
    out = rag_ask(
        user_query=question,
        model=model,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        index=index,
        chunk_list=chunk_list,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return out["response"]
