from config import Configuration
from services.vector_store import VectorStore
from services.bm25_store import BM25Store
from services.hybrid_retriever import HybridRetriever
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

config = Configuration.from_env()
vs = VectorStore(config)
bm = BM25Store()
bm.load(config.bm25_index_path)
retriever = HybridRetriever(config, vs, bm)

query = "What is path loss?"
results = retriever.retrieve(query)
for r in results:
    print(f"Score: {r['rerank_score']:.4f} | {r['doc']} p.{r['page']}\n{r['content'][:200]}\n")