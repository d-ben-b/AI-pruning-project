# main.py
import argparse
import os

from src.models import create_model, predict_no_rag
from src.rag_pipeline import (
    create_embedding_model,
    load_doc_from_path,
    predict_with_rag_ml,
)
from src.datasets import load_qa_dataset

import evaluate
from bert_score import score as bertscore_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Lecture RAG QA demo"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=os.path.join("data", "ml_lectures_corpus.jsonl"),
        help="Path to RAG corpus jsonl.",
    )
    parser.add_argument(
        "--qa_path",
        type=str,
        default=os.path.join("data", "ml_qa_dataset.jsonl"),
        help="Path to ML QA dataset jsonl.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "eval"],
        default="single",
        help="single: answer one question; eval: run on QA dataset.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask in single mode.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k documents for RAG.",
    )
    parser.add_argument(
        "--ref_answer",
        type=str,
        default=None,
        help="Reference answer for scoring in single mode.",
    )
    # ★ 預期的 emoji，用來檢查有沒有選對（emoji RAG 用）
    parser.add_argument(
        "--expected_emoji",
        type=str,
        default=None,
        help="Expected emoji for emoji-RAG evaluation (single mode only).",
    )
    parser.add_argument(
        "--lm_model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="HF model name for the chat LLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading LLM ...")
    model, tokenizer = create_model(lm_model_name=args.lm_model_name)

    print("Loading embedding model ...")
    embedding_model = create_embedding_model()

    print(f"Loading corpus from {args.corpus_path} ...")
    chunk_list_ml, index_ml = load_doc_from_path(
        documents_path=args.corpus_path,
        embedding_model=embedding_model,
    )
    print(f"Loaded {len(chunk_list_ml)} chunks from lectures.\n")

    if args.mode == "single":
        question = args.question or input("Enter your question: ").strip()
        print("\n[Question]")
        print(question)

        print("\n[No RAG]")
        ans_no_rag = predict_no_rag(question, model, tokenizer)
        print(ans_no_rag)

        print("\n[With RAG]")
        ans_rag = predict_with_rag_ml(
            question,
            model,
            tokenizer,
            embedding_model,
            index_ml,
            chunk_list_ml,
            top_k=args.top_k,
        )
        print(ans_rag)

        # 如果有提供參考答案，就算 BLEU / ROUGE-L / BERTScore
        if args.ref_answer is not None:
            ref = args.ref_answer

            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")

            # BLEU
            bleu_no = bleu.compute(
                predictions=[ans_no_rag],
                references=[[ref]],
            )["bleu"]
            bleu_rag = bleu.compute(
                predictions=[ans_rag],
                references=[[ref]],
            )["bleu"]

            # ROUGE-L
            rouge_no = rouge.compute(
                predictions=[ans_no_rag],
                references=[ref],
            )["rougeL"]
            rouge_rag = rouge.compute(
                predictions=[ans_rag],
                references=[ref],
            )["rougeL"]

            # BERTScore：現在是英文 QA → 用 lang="en"
            P_no, R_no, F_no = bertscore_score(
                [ans_no_rag], [ref], lang="en"
            )
            P_rag, R_rag, F_rag = bertscore_score(
                [ans_rag], [ref], lang="en"
            )

            print("\n=== Scores vs. reference ===")
            print("Reference:")
            print(ref)
            print("\n[No RAG]")
            print(f"BLEU     : {bleu_no:.4f}")
            print(f"ROUGE-L  : {rouge_no:.4f}")
            print(f"BERTScore: {F_no[0].item():.4f}")

            print("\n[With RAG]")
            print(f"BLEU     : {bleu_rag:.4f}")
            print(f"ROUGE-L  : {rouge_rag:.4f}")
            print(f"BERTScore: {F_rag[0].item():.4f}")

        # emoji 任務專用：檢查有沒有包含預期 emoji
        if args.expected_emoji is not None:
            emo = args.expected_emoji
            has_no = emo in ans_no_rag
            has_rag = emo in ans_rag

            print("\n=== Emoji accuracy ===")
            print(f"Expected emoji: {emo}")
            print(f"No RAG contains it?   {has_no}")
            print(f"With RAG contains it? {has_rag}")

    else:  # eval mode
        print(f"Loading QA dataset from {args.qa_path} ...")
        qa_data = load_qa_dataset(args.qa_path)
        print(f"Total questions: {len(qa_data)}\n")

        for item in qa_data:
            q = item["question"]
            print("=" * 80)
            print("Q:", q)

            ans_no_rag = predict_no_rag(q, model, tokenizer)
            ans_rag = predict_with_rag_ml(
                q,
                model,
                tokenizer,
                embedding_model,
                index_ml,
                chunk_list_ml,
                top_k=args.top_k,
            )

            print("\n[No RAG]")
            print(ans_no_rag)
            print("\n[With RAG]")
            print(ans_rag)
            print()


if __name__ == "__main__":
    main()
