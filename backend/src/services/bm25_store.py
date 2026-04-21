import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.corpus = []          # list of chunk dicts

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Build BM25 index from chunks."""
        self.corpus = chunks
        tokenized = [c["content"].split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.bm25, self.corpus), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.bm25, self.corpus = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            chunk = self.corpus[idx]
            results.append({
                "content": chunk["content"],
                "doc": chunk["doc"],
                "page": chunk["page"],
                "score": scores[idx]
            })
        return results