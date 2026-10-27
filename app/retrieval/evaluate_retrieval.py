
from app.retrieval.embeddings import load_embedding_model
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import PassageReranker

THRESHOLD = 0.65
TOP_K = 5
CANDIDATE_K = 15

EVALUATION_CASES = [
    {
        "question": "How should supply voltage and electrical connections be selected?",
        "expected_pages": {12, 13},
        "should_abstain": False,
    },
    {
        "question": "What is the exact torque specification for this motor?",
        "expected_pages": {17, 18},
        "should_abstain": False,
    },
    {
        "question": "What is the capital of Japan?",
        "expected_pages": set(),
        "should_abstain": True,
    },
    {
        "question": "Where can I find the motor's voltage and connection configuration?",
        "expected_pages": {13},
        "should_abstain": False,
    },
]


def score_results(results, expected_pages, should_abstain):
    if should_abstain:
        return {
            "precision": None,
            "recall": None,
            "mrr": None,
            "abstention": len(results) == 0,
        }

    pages = [item["page_number"] for item in results]
    relevant_chunks = sum(p in expected_pages for p in pages)

    precision = relevant_chunks / len(results) if results else 0.0
    recall = (
        len(set(pages) & expected_pages) / len(expected_pages)
        if expected_pages else 0.0
    )

    mrr = 0.0
    for rank, page in enumerate(pages, start=1):
        if page in expected_pages:
            mrr = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "abstention": None,
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    print("Loading embedding model...")
    model = load_embedding_model()

    print("Loading vector store...")
    store = VectorStore(model)

    if store.collection.count() == 0:
        print("Vector store is empty. Index the documents first.")
        return

    print("Loading cross-encoder reranker...")
    reranker = PassageReranker()

    print(f"Indexed chunks: {store.collection.count()}")
    print(f"Threshold={THRESHOLD}, candidates={CANDIDATE_K}, top_k={TOP_K}")

    summaries = {
        "Baseline": {"precision": [], "recall": [], "mrr": [], "abstention": []},
        "Reranked": {"precision": [], "recall": [], "mrr": [], "abstention": []},
    }

    for case in EVALUATION_CASES:
        question = case["question"]
        print(f"\n{'=' * 70}\n{question}\n{'=' * 70}")

        # Retrieve up to 15 candidates once.
        candidates = store.search(
            question,
            top_k=CANDIDATE_K,
            max_distance=THRESHOLD,
        )

        baseline = candidates[:TOP_K]

        # Rerank the same candidate set for a fair comparison.
        reranked = (
            reranker.rerank(question, candidates, top_k=TOP_K)
            if candidates else []
        )

        for label, results in [
            ("Baseline", baseline),
            ("Reranked", reranked),
        ]:
            metrics = score_results(
                results,
                case["expected_pages"],
                case["should_abstain"],
            )

            print(f"\n{label}:")
            if not results:
                print("  No passages retrieved.")
            else:
                for rank, item in enumerate(results, start=1):
                    extra = (
                        f", rerank_score={item['rerank_score']:.4f}"
                        if "rerank_score" in item else ""
                    )
                    print(
                        f"  #{rank} page {item['page_number']}, "
                        f"distance={item['distance']:.4f}{extra}"
                    )

            if case["should_abstain"]:
                passed = metrics["abstention"]
                summaries[label]["abstention"].append(float(passed))
                print(f"Abstention: {'PASS' if passed else 'FAIL'}")
            else:
                for metric in ("precision", "recall", "mrr"):
                    summaries[label][metric].append(metrics[metric])
                    print(f"{metric}: {metrics[metric]:.3f}")

    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(f"{'Pipeline':<12}{'Precision@5':>14}{'Page Recall@5':>16}{'MRR@5':>10}{'Abstention':>14}")

    for label, values in summaries.items():
        print(
            f"{label:<12}"
            f"{average(values['precision']):>14.3f}"
            f"{average(values['recall']):>16.3f}"
            f"{average(values['mrr']):>10.3f}"
            f"{average(values['abstention']):>13.0%}"
        )

    print(
        "\nMetrics use expected pages as relevance labels; "
        "manually inspect passages before claiming true precision."
    )


if __name__ == "__main__":
    main()
