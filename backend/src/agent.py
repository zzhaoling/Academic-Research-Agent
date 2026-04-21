"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import logging
import re
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Iterator

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool
from hello_agents.tools import RAGTool, MemoryTool
from openai import OpenAI

from config import Configuration
from prompts import (
    report_writer_instructions,
    todo_planner_system_prompt,
)
from models import SummaryState, SummaryStateOutput, TodoItem
from services.planner import PlanningService
from services.reporter import ReportingService
from services.search import dispatch_search, prepare_research_context
from services.tool_events import ToolCallTracker
from services.vector_store import VectorStore
from services.bm25_store import BM25Store
from services.hybrid_retriever import HybridRetriever
from models import TaskState
from services.scorer import TaskScorer
from services.skills import HybridRetrievalSkill, AdaptiveRetrievalSkill, WebSearchSkill, SummaryScoringSkill

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents."""

    def __init__(self, config: Configuration | None = None) -> None:
        self.config = config or Configuration.from_env()
        # 原来的 HelloAgentsLLM（用于 Planner / Reporter）
        self.llm = self._init_old_llm()
        # OpenAI 兼容客户端（用于 function calling）
        self.fc_client = self._init_fc_client()

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = None
        if self.note_tool:
            registry = ToolRegistry()
            registry.register_tool(self.note_tool)
            self.tools_registry = registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False

        # RAG 工具（仅外部使用）
        self.rag_tool: RAGTool | None = None
        if self.config.enable_rag:
            self.rag_tool = RAGTool(
                knowledge_base_path=self.config.rag_knowledge_base_path,
                rag_namespace=f"deep_research_{self.config.notes_workspace or 'default'}"
            )
            logger.info("RAG tool initialized with knowledge base: %s", self.config.rag_knowledge_base_path)

        # 记忆工具（可选）
        self.memory_tool: MemoryTool | None = None
        if self.config.enable_memory:
            self.memory_tool = MemoryTool(
                user_id="deep_research_user",
                memory_types=["working", "episodic", "semantic"]
            )
            logger.info("Memory tool initialized for user: deep_research_user")

        # 混合检索初始化
        self.hybrid_retriever = None
        if config.enable_hybrid_retrieval:
            try:
                # 加载 BM25 索引
                bm25_store = BM25Store()
                bm25_store.load(config.bm25_index_path)
                vector_store = VectorStore(config)
                self.hybrid_retriever = HybridRetriever(config, vector_store, bm25_store)
                logger.info("Hybrid retriever initialized.")
            except Exception as e:
                logger.warning(f"Failed to load hybrid retrieval indexes: {e}. Falling back to basic RAG.")
                self.hybrid_retriever = None

        # 初始化 Skills
        self.skills = {}
        if self.hybrid_retriever:
            base_skill = HybridRetrievalSkill(self.hybrid_retriever)
            self.skills["local_retrieval"] = base_skill
            # self.skills["multi_query_retrieval"] = MultiQueryRetrievalSkill(base_skill)  # 可选
        
        # 网络搜索 Skill
        self.skills["web_search"] = WebSearchSkill(config)
        
        # 自适应检索 Skill
        self.skills["adaptive_retrieval"] = AdaptiveRetrievalSkill(
            local_skill=self.skills["local_retrieval"],
            web_skill=self.skills["web_search"],
            config=config
        )
        self.skills["scoring"] = SummaryScoringSkill()

        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.reporting = ReportingService(self.report_agent, self.config)
        self._last_search_notices: list[str] = []
        print("Registered tools:", self.tools_registry.list_tools() if self.tools_registry else [])

    # ------------------------------------------------------------------
    # 初始化方法
    # ------------------------------------------------------------------
    def _init_old_llm(self) -> HelloAgentsLLM:
        llm_kwargs: dict[str, Any] = {"temperature": 0.0}
        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id
        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider
        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            llm_kwargs["api_key"] = self.config.llm_api_key or "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        llm_kwargs["timeout"] = self.config.llm_timeout
        return HelloAgentsLLM(**llm_kwargs)

    def _init_fc_client(self) -> OpenAI:
        api_key = self.config.llm_api_key
        base_url = self.config.llm_base_url
        if api_key is None:
            provider = (self.config.llm_provider or "").strip()
            if provider in ("ollama", "lmstudio"):
                api_key = "ollama"
            else:
                raise ValueError("LLM_API_KEY must be set for function calling")
        if base_url is None:
            base_url = "https://api.openai.com/v1"
        return OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.llm_timeout)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        return ToolAwareSimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=self.tools_registry is not None,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        self._tool_event_sink_enabled = sink is not None
        self._tool_tracker.set_event_sink(sink)

    # ------------------------------------------------------------------
    # Function calling 工具定义
    # ------------------------------------------------------------------
    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag",
                    "description": "Retrieve knowledge from local database",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }
        ]

    def _call_tool(self, name: str, args: dict, loop_count: int = 0) -> str:
        if name == "search":
            result, _, answer, _ = dispatch_search(args["query"], self.config, loop_count)
            return answer or str(result)
        elif name == "rag":
            return self._retrieve_rag_context(args["query"])
        return "Unknown tool"

    def run_agent_for_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """使用 function calling 生成总结，自动截断过长的上下文"""
        system_prompt = """你是一名研究执行专家。你的任务是基于给定的任务目标和已有上下文，生成要点总结。
你可以使用 search 工具搜索互联网获取补充信息，或使用 rag 工具从本地知识库检索。
最终输出必须是 Markdown 格式，以"任务总结"为标题，包含3-5条关键发现，每条发现要有详细解释。"""

        user_prompt_template = f"""任务主题：{state.research_topic}
任务名称：{task.title}
任务目标：{task.intent}
检索查询：{task.query}
已有上下文（来自网络搜索和本地知识库）：
{{context}}

请根据需要调用工具获取更多信息（如果需要），然后生成最终总结。"""

        # --- 防止上下文过长（粗略估算：1 token ≈ 4 字符）---
        MAX_PROMPT_CHARS = 8000   # 安全值，约 5000 tokens
        if len(context) > MAX_PROMPT_CHARS:
            keep_start = int(MAX_PROMPT_CHARS * 0.7)
            keep_end = MAX_PROMPT_CHARS - keep_start
            context = context[:keep_start] + "\n...[中间部分已截断]...\n" + context[-keep_end:]

        user_prompt = user_prompt_template.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        tools = self._build_tools()

        for _ in range(3):  # 最多3轮工具调用
            resp = self.fc_client.chat.completions.create(
                model=self.config.llm_model_id or "gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "暂无可用信息"
            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self._call_tool(name, args, state.research_loop_count)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        # 超过轮数强制要求生成
        messages.append({"role": "user", "content": "请根据已有信息直接生成最终总结。"})
        final = self.fc_client.chat.completions.create(
            model=self.config.llm_model_id or "gpt-4o-mini",
            messages=messages,
            temperature=0.0,
        )
        return final.choices[0].message.content or "生成失败"

    # ------------------------------------------------------------------
    # 混合 检索
    # ------------------------------------------------------------------
    def _retrieve_rag_context(self, query: str, max_chunks: int = 3) -> str:
        """Enhanced retrieval using hybrid retriever if available."""
        if self.hybrid_retriever:
            try:
                results = self.hybrid_retriever.retrieve(query)
                if not results:
                    return ""
                # 分级使用：高相关度（>0.7）和中等相关度（0.4-0.7）
                high = [r for r in results if r.get("rerank_score", 0) >= 0.7]
                medium = [r for r in results if 0.4 <= r.get("rerank_score", 0) < 0.7]
                
                formatted = "\n\n## 本地知识库相关内容（混合检索）\n\n"
                if high:
                    formatted += "### 高相关度片段\n"
                    for i, r in enumerate(high, 1):
                        formatted += f"**片段 {i}** (相关度: {r['rerank_score']:.2f}) - 来自 {r['doc']} 第{r['page']}页\n{r['content'][:800]}\n\n---\n\n"
                if medium:
                    formatted += "### 中等相关度片段\n"
                    for i, r in enumerate(medium, 1):
                        formatted += f"**片段 {i}** (相关度: {r['rerank_score']:.2f}) - 来自 {r['doc']} 第{r['page']}页\n{r['content'][:600]}\n\n---\n\n"
                return formatted
            except Exception as e:
                logger.warning(f"Hybrid retrieval failed: {e}, falling back to basic RAG")
                # 降级到原有 RAGTool
                return self._legacy_rag_retrieve(query, max_chunks)
        else:
            return self._legacy_rag_retrieve(query, max_chunks)

    def _legacy_rag_retrieve(self, query: str, max_chunks: int = 3) -> str:
        if not self.rag_tool:
            return ""
        try:
            result = self.rag_tool.run({
                "action": "search",
                "query": query,
                "limit": max_chunks
            })
            chunks = []
            if isinstance(result, dict):
                chunks = result.get("results", [])
            elif isinstance(result, list):
                chunks = result
            elif isinstance(result, str):
                return f"\n\n## 本地知识库检索结果\n\n{result}\n\n"
            if not chunks:
                return ""
            formatted = "\n\n## 本地知识库相关内容\n\n"
            for i, chunk in enumerate(chunks, 1):
                if isinstance(chunk, dict):
                    content = chunk.get("content", chunk.get("text", ""))[:1000]  # 限制1000字符
                    score = chunk.get("score", "N/A")
                    formatted += f"**片段 {i}** (相关度: {score})\n{content}\n\n---\n\n"
                else:
                    formatted += f"**片段 {i}**:\n{str(chunk)}\n\n---\n\n"
            return formatted
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return ""

    # agent.py 中修改 _execute_task_with_fsm 方法签名和实现
    def _execute_task_with_fsm(self, state: SummaryState, task: TodoItem, 
                            emit_stream: bool = False, step: int = None,
                            on_event: Optional[Callable[[dict], None]] = None):
        """状态机版本的任务执行，支持重试与策略调整，可选发送流式事件"""
        max_attempts = getattr(self.config, 'max_retries_per_task', 2)
        attempt = 0
        current_state = TaskState.SEARCHING
        query = task.query

        # 辅助发送事件
        def send_event(event_type, **kwargs):
            if on_event:
                payload = {"type": event_type, "task_id": task.id, **kwargs}
                if step is not None:
                    payload["step"] = step
                on_event(payload)

        # 发送任务开始状态
        send_event("task_status", status="in_progress", title=task.title, 
                intent=task.intent, note_id=task.note_id, note_path=task.note_path)

        while attempt < max_attempts:
            if current_state == TaskState.SEARCHING:
                # try:
                #     search_result, notices, answer_text, backend = dispatch_search(
                #         query, self.config, state.research_loop_count
                #     )
                # except Exception as e:
                #     logger.warning(f"搜索失败: {e}")
                #     current_state = TaskState.RETRY
                #     continue

                # if not search_result or not search_result.get("results"):
                #     logger.info(f"无搜索结果，尝试重试（查询: {query}）")
                #     current_state = TaskState.RETRY
                #     continue

                # sources_summary, context = prepare_research_context(search_result, answer_text, self.config)
                # rag_context = self._retrieve_rag_context(query)
                # if rag_context:
                #     context = context + "\n\n" + rag_context
                # task.sources_summary = sources_summary
                context = self.skills["adaptive_retrieval"].execute(
                    query=query,
                    top_k=3,
                    score_threshold=self.config.hybrid_score_threshold,
                    loop_count=state.research_loop_count,
                    return_str=True
                )
                task._context = context

                # # 发送 sources 事件
                # send_event("sources", latest_sources=sources_summary, raw_context=context, backend=backend,
                #         note_id=task.note_id, note_path=task.note_path)

                current_state = TaskState.GENERATING

            elif current_state == TaskState.GENERATING:
                summary_text = self.run_agent_for_task(state, task, task._context)
                task.summary = summary_text.strip() if summary_text else "暂无可用信息"
                # 发送总结 chunk（这里假设整个总结作为一个 chunk）
                send_event("task_summary_chunk", content=task.summary, note_id=task.note_id)
                current_state = TaskState.SCORING

            elif current_state == TaskState.SCORING:
                from services.scorer import TaskScorer
                score = TaskScorer.score_summary(task.summary)
                task.score = score
                if score >= self.config.quality_threshold:
                    task.status = "completed"
                    send_event("task_status", status="completed", summary=task.summary,
                            sources_summary=task.sources_summary, note_id=task.note_id, note_path=task.note_path)
                    return
                else:
                    logger.info(f"任务 {task.id} 质量得分 {score:.2f} < {self.config.quality_threshold}，触发重试")
                    current_state = TaskState.RETRY

            elif current_state == TaskState.RETRY:
                attempt += 1
                if attempt == 1:
                    query = f"{task.query} 原理 机制"
                    self.skills["adaptive_retrieval"].strategy = "parallel"   # 强制并行
                elif attempt == 2:
                    query = f"{task.query} 最新研究 综述"
                    self.skills["adaptive_retrieval"].strategy = "web_only"   # 只用网络搜索
                else:
                    task.status = "failed"
                    send_event("task_status", status="failed", detail="超过最大重试次数")
                    return
                state.research_loop_count += 1
                current_state = TaskState.SEARCHING

        # 超出重试次数
        task.status = "failed"
        send_event("task_status", status="failed", detail="超过最大重试次数")
        # 记忆工具记录
        if self.memory_tool and summary_text:
            self.memory_tool.run({
                "action": "add",
                "content": f"任务 {task.title} 总结摘要:\n{summary_text[:500]}",
                "memory_type": "episodic",
                "importance": 0.7,
                "task_id": task.id
            })
    # ------------------------------------------------------------------
    # 核心执行逻辑（同步模式）
    # ------------------------------------------------------------------
    def run(self, topic: str) -> SummaryStateOutput:
        print("=== run() method started ===")
        state = SummaryState(research_topic=topic)
        state.todo_items = self.planner.plan_todo_list(state)
        print(f"=== planner returned {len(state.todo_items)} tasks ===")
        self._drain_tool_events(state)

        if not state.todo_items:
            logger.info("No TODO items generated; falling back to single task")
            state.todo_items = [self.planner.create_fallback_task(state)]

        print("=== about to enter task loop ===")

        for idx, task in enumerate(state.todo_items):
            print(f"=== 执行任务 {task.id}: {task.title} ===")
            # 直接内联 _execute_task 逻辑（已确保执行）
            # self._execute_task_inline(state, task, emit_stream=False)
            self._execute_task_with_fsm(state, task, emit_stream=False)
            print(f"=== 任务 {task.id} 最终状态: {task.status} ===")
        print("=== task loop finished ===")

        # ----- 全局评估与回退 -----
        completed_tasks = [t for t in state.todo_items if t.status == "completed"]
        success_rate = len(completed_tasks) / len(state.todo_items) if state.todo_items else 0
        
        if success_rate < 0.5 and self.config.enable_global_replan:  # 需要新增配置项
            logger.info(f"整体成功率 {success_rate:.2%} 低于 50%，触发重新规划")
            # 重新规划（基于当前已有的部分结果）
            new_tasks = self.planner.replan_tasks(state)  # 需要实现该方法
            if new_tasks:
                state.todo_items = new_tasks
                # 重新执行新任务
                for task in state.todo_items:
                    self._execute_task_with_fsm(state, task, emit_stream=False)
        
        # 生成报告
        report = self.reporting.generate_report(state)
        self._drain_tool_events(state)
        state.structured_report = report
        state.running_summary = report
        self._persist_final_report(state, report)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    # def _execute_task_inline(self, state: SummaryState, task: TodoItem, emit_stream: bool = False, step: int = None):
    #     """内联版任务执行（同步，无流式）"""
    #     task.status = "in_progress"

    #     # 搜索
    #     try:
    #         search_result, notices, answer_text, backend = dispatch_search(
    #             task.query, self.config, state.research_loop_count
    #         )
    #     except Exception as e:
    #         print(f"搜索失败: {e}")
    #         task.status = "failed"
    #         return

    #     if not search_result or not search_result.get("results"):
    #         print("无搜索结果，跳过任务")
    #         task.status = "skipped"
    #         return

    #     # 构建上下文
    #     sources_summary, context = prepare_research_context(search_result, answer_text, self.config)
    #     rag_context = self._retrieve_rag_context(task.query)
    #     if rag_context:
    #         context = context + "\n\n" + rag_context
    #     task.sources_summary = sources_summary

    #     # 更新状态
    #     with self._state_lock:
    #         state.web_research_results.append(context)
    #         state.sources_gathered.append(sources_summary)
    #         state.research_loop_count += 1

    #     # 生成总结（使用 function calling，自动截断 context）
    #     summary_text = self.run_agent_for_task(state, task, context)
    #     task.summary = summary_text.strip() if summary_text else "暂无可用信息"
    #     task.status = "completed"

    #     # 记忆工具记录
    #     if self.memory_tool and summary_text:
    #         self.memory_tool.run({
    #             "action": "add",
    #             "content": f"任务 {task.title} 总结摘要:\n{summary_text[:500]}",
    #             "memory_type": "episodic",
    #             "importance": 0.7,
    #             "task_id": task.id
    #         })

        
    # ------------------------------------------------------------------
    # 流式模式（保留原有框架，但内部未实现流式总结，可暂时忽略）
    # ------------------------------------------------------------------
    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """流式研究（简化版：任务执行同步，总结作为单个chunk输出）"""
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}

        state.todo_items = self.planner.plan_todo_list(state)
        for event in self._drain_tool_events(state, step=0):
            yield event
        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

        # 为每个任务生成流式token
        channel_map: dict[int, dict[str, Any]] = {}
        for index, task in enumerate(state.todo_items, start=1):
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": index, "token": token}

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        # 事件队列（用于工具调用事件）
        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(event: dict[str, Any], *, task: TodoItem | None = None, step_override: int | None = None):
            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id
            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            enqueue(event)

        self._set_tool_event_sink(tool_event_sink)

        threads: list[Thread] = []

        def worker(task: TodoItem, step: int):
            try:
                # 定义 on_event 回调，将事件放入队列
                def on_event(event: dict):
                    # 确保包含必要字段
                    if "task_id" not in event:
                        event["task_id"] = task.id
                    if "step" not in event and step is not None:
                        event["step"] = step
                    # 添加 stream_token
                    channel = channel_map.get(task.id)
                    if channel:
                        event["stream_token"] = channel["token"]
                    event_queue.put(event)

                # 调用状态机执行
                self._execute_task_with_fsm(
                    state, task, emit_stream=True, step=step, on_event=on_event
                )
            except Exception as exc:
                logger.exception("Task execution failed")
                # 发送失败事件
                on_event({
                    "type": "task_status",
                    "status": "failed",
                    "detail": str(exc),
                })
            finally:
                event_queue.put({"type": "__task_done__", "task_id": task.id})
            # try:
            #     # 发送任务开始状态
            #     enqueue({
            #         "type": "task_status",
            #         "task_id": task.id,
            #         "status": "in_progress",
            #         "title": task.title,
            #         "intent": task.intent,
            #         "note_id": task.note_id,
            #         "note_path": task.note_path,
            #     }, task=task)

            #     # 执行内联任务（同步，但会产生 sources 事件）
            #     # 注意：内联任务内部会调用 dispatch_search 和 run_agent_for_task
            #     # 我们需要在内部 yield 事件，但 worker 不是生成器，因此需要改造
            #     # 方案：将内联任务拆分为几个阶段，手动 enqueue 事件
            #     task.status = "in_progress"

            #     # 搜索
            #     try:
            #         search_result, notices, answer_text, backend = dispatch_search(
            #             task.query, self.config, state.research_loop_count
            #         )
            #     except Exception as e:
            #         logger.exception(f"搜索失败: {e}")
            #         enqueue({
            #             "type": "task_status",
            #             "task_id": task.id,
            #             "status": "failed",
            #             "detail": str(e),
            #         }, task=task)
            #         return

            #     if not search_result or not search_result.get("results"):
            #         task.status = "skipped"
            #         enqueue({
            #             "type": "task_status",
            #             "task_id": task.id,
            #             "status": "skipped",
            #         }, task=task)
            #         return

            #     # 构建上下文
            #     sources_summary, context = prepare_research_context(search_result, answer_text, self.config)
            #     rag_context = self._retrieve_rag_context(task.query)
            #     if rag_context:
            #         context = context + "\n\n" + rag_context
            #     task.sources_summary = sources_summary

            #     # 发送 sources 事件
            #     enqueue({
            #         "type": "sources",
            #         "task_id": task.id,
            #         "latest_sources": sources_summary,
            #         "raw_context": context,
            #         "backend": backend,
            #         "note_id": task.note_id,
            #         "note_path": task.note_path,
            #     }, task=task)

            #     # 生成总结（同步，但将完整结果作为一个 chunk 发送）
            #     summary_text = self.run_agent_for_task(state, task, context)
            #     task.summary = summary_text.strip() if summary_text else "暂无可用信息"
            #     task.status = "completed"

            #     # 发送总结 chunk（模拟流式）
            #     if task.summary:
            #         enqueue({
            #             "type": "task_summary_chunk",
            #             "task_id": task.id,
            #             "content": task.summary,
            #             "note_id": task.note_id,
            #         }, task=task)

            #     # 发送任务完成状态
            #     enqueue({
            #         "type": "task_status",
            #         "task_id": task.id,
            #         "status": "completed",
            #         "summary": task.summary,
            #         "sources_summary": task.sources_summary,
            #         "note_id": task.note_id,
            #         "note_path": task.note_path,
            #     }, task=task)

            # except Exception as exc:
            #     logger.exception("Task execution failed")
            #     enqueue({
            #         "type": "task_status",
            #         "task_id": task.id,
            #         "status": "failed",
            #         "detail": str(exc),
            #     }, task=task)
            # finally:
            #     enqueue({"type": "__task_done__", "task_id": task.id})

        # 启动线程
        for task in state.todo_items:
            step = channel_map.get(task.id, {}).get("step", 0)
            thread = Thread(target=worker, args=(task, step), daemon=True)
            threads.append(thread)
            thread.start()

        active_workers = len(state.todo_items)
        finished_workers = 0

        try:
            while finished_workers < active_workers:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            # 清空剩余事件
            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            self._set_tool_event_sink(None)
            for thread in threads:
                thread.join()

        # 生成最终报告
        report = self.reporting.generate_report(state)
        final_step = len(state.todo_items) + 1
        for event in self._drain_tool_events(state, step=final_step):
            yield event
        state.structured_report = report
        state.running_summary = report

        note_event = self._persist_final_report(state, report)
        if note_event:
            yield note_event

        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }
        yield {"type": "done"}
    
    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _drain_tool_events(self, state: SummaryState, *, step: int | None = None) -> list[dict[str, Any]]:
        events = self._tool_tracker.drain(state, step=step)
        if self._tool_event_sink_enabled:
            return []
        return events

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        if not self.note_tool or not report or not report.strip():
            return None
        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()
        note_id = self._find_existing_report_note_id(state)
        response = ""
        if note_id:
            response = self.note_tool.run({
                "action": "update",
                "note_id": note_id,
                "title": note_title,
                "note_type": "conclusion",
                "tags": tags,
                "content": content,
            })
            if response.startswith("❌"):
                note_id = None
        if not note_id:
            response = self.note_tool.run({
                "action": "create",
                "title": note_title,
                "note_type": "conclusion",
                "tags": tags,
                "content": content,
            })
            note_id = self._extract_note_id_from_text(response)
        if not note_id:
            return None
        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None
        payload = {"type": "report_note", "note_id": note_id, "title": note_title, "content": content}
        if note_path:
            payload["note_path"] = str(note_path)
        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id
        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue
            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue
            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue
            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue
            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker._extract_note_id(event.get("result", ""))
            if note_id:
                return note_id
        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None
        match = re.search(r"ID:\s*([^\n]+)", response)
        return match.group(1).strip() if match else None


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)