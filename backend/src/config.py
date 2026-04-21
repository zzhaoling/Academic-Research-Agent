import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./src/notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )
    # ------------- 新增 ---------------
    enable_rag: bool = Field(
        default=True,
        title="Enable RAG",
        description="Whether to use local RAG knowledge base",
    )
    rag_knowledge_base_path: str = Field(
        default="./src/rag/rag_knowledge",
        title="RAG Knowledge Base Path",
        description="Directory to store RAG vector index and documents",
    )
    enable_memory: bool = Field(
        default=True,
        title="Enable Memory",
        description="Whether to store research memories",
    )
    # ========== 混合检索新增 ==========
    enable_hybrid_retrieval: bool = Field(
        default=True,
        title="Enable Hybrid Retrieval",
        description="Use Qdrant+BM25+CrossEncoder hybrid retrieval instead of basic RAGTool"
    )
    pdf_dir: str = Field(
        default="./src/papers",
        title="PDF Directory",
        description="Directory containing PDF files for indexing"
    )
    chunk_size: int = Field(default=300, description="Number of words per chunk")
    chunk_overlap: int = Field(default=50, description="Overlap words between chunks")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        title="Embedding Model",
        description="SentenceTransformer model for vector embeddings"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        title="Cross-Encoder Model",
        description="Model for reranking"
    )
    qdrant_mode: str = Field(default="embedded", description="embedded or docker")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="API key if required")
    qdrant_timeout: int = Field(default=30, description="Request timeout in seconds")

    qdrant_path: str = Field(
        default="./src/store/qdrant",
        title="Qdrant Storage Path",
        description="Directory to persist Qdrant vector index"
    )
    qdrant_collection: str = Field(
        default="papers",
        title="Qdrant Collection Name",
        description="Collection name for vectors"
    )
    bm25_index_path: str = Field(
        default="./src/store/bm25.pkl",
        title="BM25 Index Path",
        description="Path to pickled BM25 index"
    )
    chunks_metadata_path: str = Field(
        default="./src/store/chunks.json",
        title="Chunks Metadata Path",
        description="Path to JSON file storing chunk metadata"
    )
    hybrid_top_k_vector: int = Field(default=5, description="Top-k for vector search")
    hybrid_top_k_bm25: int = Field(default=5, description="Top-k for BM25 search")
    hybrid_rerank_top_k: int = Field(default=3, description="Final top-k after reranking")
    hybrid_score_threshold: float = Field(default=0.3, description="Minimum rerank score to include")
    
    max_retries_per_task: int = Field(default=2, description="每个任务最大重试次数")
    quality_threshold: float = Field(default=0.6, description="任务总结质量阈值（0~1）")
    enable_global_replan: bool = Field(default=True, description="是否启用全局重新规划")

    llm_timeout: int = Field(
        default=60,
        title="LLM Timeout",
        description="Timeout in seconds for LLM API calls",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
            "enable_rag": os.getenv("ENABLE_RAG"),
            "rag_knowledge_base_path": os.getenv("RAG_KNOWLEDGE_BASE_PATH"),
            "enable_memory": os.getenv("ENABLE_MEMORY"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm

