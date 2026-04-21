"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""

        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""

        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        # 2. 处理 dict 类型的返回
        if isinstance(json_payload, dict):
            # 优先使用标准 tasks 字段
            if "tasks" in json_payload and isinstance(json_payload["tasks"], list):
                for item in json_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)
                return tasks

            # 兼容模型偶尔返回的 task_plan_overview 格式
            if "task_plan_overview" in json_payload and isinstance(json_payload["task_plan_overview"], list):
                for item in json_payload["task_plan_overview"]:
                    if not isinstance(item, dict):
                        continue
                    # 将 focus 作为 intent，title 作为 title 和 query
                    tasks.append({
                        "title": item.get("title", ""),
                        "intent": item.get("focus", ""),
                        "query": item.get("title", "")
                    })
                return tasks

            # 其他可能的格式（如直接包含任务数组）
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
                return tasks

        # 3. 处理 list 类型的返回（纯数组）
        if isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)
            return tasks

        # 4. 最后尝试从工具调用中提取（原有 fallback）
        tool_payload = self._extract_tool_payload(text)
        if tool_payload and isinstance(tool_payload.get("tasks"), list):
            for item in tool_payload["tasks"]:
                if isinstance(item, dict):
                    tasks.append(item)
            return tasks

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """Try to locate and parse a JSON object or array from the text."""

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """Parse the first TOOL_CALL expression in the output."""

        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None

    def replan_tasks(self, state: SummaryState) -> List[TodoItem]:
        """基于已有结果重新规划任务（用于全局回退）"""
        # 构建失败任务信息
        failed_info = []
        for task in state.todo_items:
            if task.status != "completed":
                failed_info.append(f"任务 {task.id}（{task.title}）失败，已有总结：{task.summary or '无'}")
        
        if not failed_info:
            return []
        
        extra_prompt = f"""
    先前规划的部分任务未能达到质量标准，需要重新规划。
    失败任务详情：
    {chr(10).join(failed_info)}

    请补充或替换这些任务，确保覆盖研究主题的核心问题。
    输出格式与初次规划相同（JSON数组）。
    """
        # 调用 agent 生成新任务
        response = self._agent.run(extra_prompt)
        self._agent.clear_history()
        tasks_payload = self._extract_tasks(response)
        # 转换为 TodoItem，注意 id 延续
        base_id = max([t.id for t in state.todo_items], default=0)
        new_items = []
        for idx, item in enumerate(tasks_payload, start=base_id+1):
            new_items.append(TodoItem(
                id=idx,
                title=item.get("title", f"补充任务{idx}"),
                intent=item.get("intent", ""),
                query=item.get("query", state.research_topic),
            ))
        return new_items