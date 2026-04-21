from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever:
    def __init__(self, config, vector_store, bm25_store):
        self.config = config
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.cross_encoder = CrossEncoder(config.cross_encoder_model)

    @staticmethod
    def _rrf(results_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.
        Each dict in results_lists must have a unique "content" field.
        Returns list of dicts sorted by RRF score.
        """
        scores = {}
        for results in results_lists:
            for rank, item in enumerate(results, start=1):
                key = item["content"]
                scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
        # Sort by RRF score descending
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Reconstruct items preserving original metadata (first occurrence)
        merged_items = []
        seen = set()
        for key, _ in sorted_keys:
            if key in seen:
                continue
            seen.add(key)
            # Find the first occurrence from original lists
            for lst in results_lists:
                for item in lst:
                    if item["content"] == key:
                        merged_items.append(item.copy())
                        break
                else:
                    continue
                break
        return merged_items

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval with reranking."""
        # 1. Get candidates from vector and BM25
        vector_results = self.vector_store.search(query, top_k=self.config.hybrid_top_k_vector)
        bm25_results = self.bm25_store.search(query, top_k=self.config.hybrid_top_k_bm25)

        # 2. RRF fusion
        fused = self._rrf([vector_results, bm25_results], k=60)

        # 3. Cross-Encoder reranking
        if fused:
            pairs = [(query, item["content"]) for item in fused]
            scores = self.cross_encoder.predict(pairs)
            # 可选归一化
            # scores = 1 / (1 + np.exp(-scores))   # sigmoid 到 (0,1)
            for i, item in enumerate(fused):
                item["rerank_score"] = float(scores[i])
            fused.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 4. Threshold filtering and limit
        final = [item for item in fused if item.get("rerank_score", 0) >= self.config.hybrid_score_threshold]
        return final[:self.config.hybrid_rerank_top_k]