import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, config):
        self.config = config
        # 根据模式选择连接方式
        if config.qdrant_mode == "docker":
            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=config.qdrant_timeout
            )
        else:
            self.client = QdrantClient(path=config.qdrant_path)
        
        self.model = SentenceTransformer(config.embedding_model)
        self.collection_name = config.qdrant_collection

        # 确保集合存在（如果不存在则创建）
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks with payload (content, doc, page)."""
        points = []
        texts = [c["content"] for c in chunks]
        embeddings = self.model.encode(texts)

        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload=chunk
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return list of dicts: {content, doc, page, score}."""
        vector = self.model.encode(query).tolist()
        
        # 新版 qdrant-client 使用 query_points 方法
        # 如果仍然想用旧版 search，可以捕获 AttributeError 降级
        try:
            # 尝试新 API
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            results = []
            for point in response.points:
                results.append({
                    "content": point.payload["content"],
                    "doc": point.payload["doc"],
                    "page": point.payload["page"],
                    "score": point.score
                })
            return results
        except AttributeError:
            # 降级到旧版 search 方法（如果存在）
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k
            )
            return [{
                "content": r.payload["content"],
                "doc": r.payload["doc"],
                "page": r.payload["page"],
                "score": r.score
            } for r in results]