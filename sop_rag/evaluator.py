"""
Evaluation metrics for SOPRAG benchmark.
Implements:
- Retrieval Quality: MRR, Acc@1, Acc@3, Acc@5, Latency
- Generation Quality: SOP Quality Score (has_step_by_step, is_precise, is_concise, is_clear)
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class RetrievalEvaluator:
    """Evaluates ranking metrics: MRR, Acc@1, Acc@3, Acc@5."""

    @staticmethod
    def evaluate(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        results is a list of dicts:
        {
            "query_id": str,
            "target_sop_id": str,
            "ranked_sop_ids": List[str],
            "latency": float
        }
        """
        if not results:
            return {"mrr": 0.0, "acc@1": 0.0, "acc@3": 0.0, "acc@5": 0.0, "latency": 0.0}

        reciprocal_ranks = []
        hits_at_1 = []
        hits_at_3 = []
        hits_at_5 = []
        latencies = []

        for item in results:
            target = item["target_sop_id"]
            ranked = item["ranked_sop_ids"]
            latencies.append(item.get("latency", 0.0))

            rank = None
            for idx, candidate in enumerate(ranked):
                if candidate == target:
                    rank = idx + 1
                    break

            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
                hits_at_1.append(1.0 if rank == 1 else 0.0)
                hits_at_3.append(1.0 if rank <= 3 else 0.0)
                hits_at_5.append(1.0 if rank <= 5 else 0.0)
            else:
                reciprocal_ranks.append(0.0)
                hits_at_1.append(0.0)
                hits_at_3.append(0.0)
                hits_at_5.append(0.0)

        return {
            "mrr": float(np.mean(reciprocal_ranks)),
            "acc@1": float(np.mean(hits_at_1)),
            "acc@3": float(np.mean(hits_at_3)),
            "acc@5": float(np.mean(hits_at_5)),
            "latency": float(np.mean(latencies))
        }


class GenerationEvaluator:
    """Evaluates generation quality using SOP Quality Score."""

    @staticmethod
    def evaluate(quality_scores: List[Dict[str, Any]]) -> Dict[str, float]:
        if not quality_scores:
            return {"has_step_by_step": 0.0, "is_precise": 0.0, "is_concise": 0.0, "is_clear": 0.0, "sop_score": 0.0}

        step_by_step = [q.get("has_step_by_step", 0) for q in quality_scores]
        precise = [q.get("is_precise", 0) for q in quality_scores]
        concise = [q.get("is_concise", 0) for q in quality_scores]
        clear = [q.get("is_clear", 0) for q in quality_scores]
        scores = [q.get("score", 0.0) for q in quality_scores]

        return {
            "has_step_by_step": float(np.mean(step_by_step)),
            "is_precise": float(np.mean(precise)),
            "is_concise": float(np.mean(concise)),
            "is_clear": float(np.mean(clear)),
            "sop_score": float(np.mean(scores))
        }
