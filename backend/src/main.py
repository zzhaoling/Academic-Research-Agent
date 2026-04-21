"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv

# 强制加载当前目录下的 .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
import json
import sys
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from agent import DeepResearchAgent
# 添加
from fastapi import UploadFile, File
from pathlib import Path
import shutil

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)

PAPERS_DIR = "./src/papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


def create_app() -> FastAPI:
    app = FastAPI(title="Academic Research Agent")

    # -------------- 添加全局实例 ----------------
    
    # 加载配置
    config = Configuration.from_env()

    # 全局 RAG 工具（局部变量）
    rag_tool = None
    if config.enable_rag:
        from hello_agents.tools import RAGTool
        rag_tool = RAGTool(
            # knowledge_base_path=config.rag_knowledge_base_path,
            knowledge_base_path = config.pdf_dir,
            rag_namespace="deep_research_global"
        )

    # 全局记忆工具
    memory_tool = None
    if config.enable_memory:
        from hello_agents.tools import MemoryTool
        memory_tool = MemoryTool(user_id="deep_research_user", memory_types=["working", "episodic", "semantic"])

    # 导入增量更新函数（放在函数内部，避免循环导入）
    def _run_update_db():
        try:
            from rag.update_db import update_database
            update_database()
        except ImportError:
            logger.warning("update_db module not found, skip database update")
        except Exception as e:
            logger.error(f"Database update failed: {e}")
    
    #  -----------CORS 中间件----------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # ---------- 启动日志 ----------
    @app.on_event("startup")
    def log_startup_configuration() -> None:
        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    # ---------- 上传文档端点 ----------
    @app.post("/upload_document")
    async def upload_document(file: UploadFile = File(...)):
        if rag_tool is None:
            raise HTTPException(status_code=501, detail="RAG not enabled")
        
        # 检查文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")
        # 保存原始 PDF 到 papers 目录
        file_path = os.path.join(PAPERS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 触发增量更新（同步执行，小文件可接受）
        try:
            _run_update_db()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件已保存，但向量库更新失败: {str(e)}")
        
        return {"status": "success", "filename": file.filename, "saved_path": file_path}
    
    # ---------- 检索端点 ----------
    @app.post("/search_papers")
    def search_papers(query: str, limit: int = 5):
        if rag_tool is None:
            raise HTTPException(status_code=501, detail="RAG not enabled")
        results = rag_tool.run({"action": "search", "query": query, "limit": limit})
        return {"query": query, "results": results}

    # ---------- 记忆回顾端点 ----------
    @app.post("/recall")
    def recall(query: str, limit: int = 5):
        if memory_tool is None:
            raise HTTPException(status_code=501, detail="Memory not enabled")
        result = memory_tool.run({"action": "search", "query": query, "limit": limit})
        return {"memories": result}
    
    # ---------- 研究端点（同步） ----------
    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except Exception as exc:
            import traceback
            traceback.print_exc()   # 打印到控制台
            raise HTTPException(status_code=500, detail=str(exc)) from exc
            
        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )
    
    # ---------- 研究端点（流式） ----------
    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
