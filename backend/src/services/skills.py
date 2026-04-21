from typing import List, Dict, Any, Optional
from services.hybrid_retriever import HybridRetriever
from services.scorer import TaskScorer
from services.search import dispatch_search, prepare_research_context
from config import Configuration
import logging

logger = logging.getLogger(__name__)

class HybridRetrievalSkill:
    """混合检索技能（返回结构化数据）"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
    
    def execute(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3,
        return_str: bool = False
    ) -> List[Dict[str, Any]] | str:
        """
        执行混合检索
        :param return_str: True 返回格式化文本（供 LLM 使用），False 返回结构化列表
        """
        results = self.retriever.retrieve(query)  # 假设 retrieve 已返回带 score/doc/page 的列表
        
        # 过滤阈值
        results = [r for r in results if r.get("rerank_score", 0) >= score_threshold]
        results = results[:top_k]
        
        if not results:
            return "未找到相关内容。" if return_str else []
        
        if not return_str:
            # 转换为 RetrievalResult 格式
            return [
                {
                    "content": r["content"],
                    "score": r.get("rerank_score", r.get("score", 0.0)),
                    "doc": r.get("doc", "未知"),
                    "page": r.get("page", 0)
                }
                for r in results
            ]
        else:
            # 格式化文本
            formatted = []
            for r in results:
                formatted.append(
                    f"[{r.get('doc', '未知')} - 第{r.get('page', '?')}页]\n"
                    f"相关度: {r.get('rerank_score', 0.0):.2f}\n"
                    f"{r['content'][:300]}"
                )
            return "\n\n".join(formatted)


class SummaryScoringSkill:
    """总结质量评分技能"""
    
    @staticmethod
    def execute(summary: str) -> float:
        return TaskScorer.score_summary(summary)


class WebSearchSkill:
    def __init__(self, config: Configuration):
        self.config = config

    def execute(
        self,
        query: str,
        loop_count: int = 0,
        return_str: bool = True
    ) -> str | Dict[str, Any]:
        """
        执行网络搜索，返回上下文文本或原始结果
        """
        try:
            search_result, notices, answer_text, backend = dispatch_search(
                query, self.config, loop_count
            )
            if not search_result or not search_result.get("results"):
                return "未找到相关网络信息。" if return_str else {"error": "no_results"}
            
            sources_summary, context = prepare_research_context(
                search_result, answer_text, self.config
            )
            if return_str:
                return context
            else:
                return {
                    "context": context,
                    "sources_summary": sources_summary,
                    "backend": backend
                }
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return "网络搜索失败。" if return_str else {"error": str(e)}
        
class MultiQueryRetrievalSkill:
    """多查询检索技能（组合多个查询结果）"""
    
    def __init__(self, base_skill: HybridRetrievalSkill):
        self.base = base_skill
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        return_str: bool = False
    ) -> List[Dict[str, Any]] | str:
        queries = [
            query,
            f"{query} 原理 机制",
            f"{query} 方法 对比",
            f"{query} 最新研究"
        ]
        all_results = []
        for q in queries:
            results = self.base.execute(q, top_k=top_k, score_threshold=score_threshold, return_str=False)
            if isinstance(results, list):
                all_results.extend(results)
        
        # 去重（基于 content）
        unique = {}
        for r in all_results:
            if r["content"] not in unique:
                unique[r["content"]] = r
        
        merged = list(unique.values())
        # 按分数排序
        merged.sort(key=lambda x: x["score"], reverse=True)
        merged = merged[:top_k]
        
        if not return_str:
            return merged
        else:
            formatted = []
            for r in merged:
                formatted.append(
                    f"[{r['doc']} - 第{r['page']}页]\n"
                    f"相关度: {r['score']:.2f}\n"
                    f"{r['content'][:300]}"
                )
            return "\n\n".join(formatted)

class AdaptiveRetrievalSkill:
    """
    自适应检索：优先本地知识库，如果结果不足或分数过低，自动调用网络搜索补充。
    支持策略切换。
    """
    def __init__(self, local_skill: HybridRetrievalSkill, web_skill: WebSearchSkill, config):
        self.local = local_skill
        self.web = web_skill
        self.config = config
        self.strategy = "local_then_web"  # 可选: "parallel", "web_only", "local_only"

    def execute(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3,
        loop_count: int = 0,
        return_str: bool = True
    ) -> str:
        """
        根据策略返回合并后的上下文（文本形式，便于状态机直接使用）
        """
        contexts = []
        if self.strategy == "local_then_web":
            # 先尝试本地
            local_results = self.local.execute(query, top_k=top_k, score_threshold=score_threshold, return_str=False)
            if local_results and isinstance(local_results, list) and len(local_results) >= top_k//2:
                # 本地结果足够好
                context = self.local.execute(query, top_k=top_k, score_threshold=score_threshold, return_str=True)
                contexts.append(f"【本地知识库检索结果】\n{context}")
            else:
                # 本地不足，补充网络
                web_context = self.web.execute(query, loop_count=loop_count, return_str=True)
                contexts.append(f"【网络搜索补充】\n{web_context}")
                # 也加入本地已有的部分（如果有）
                if local_results:
                    local_context = self.local.execute(query, top_k=top_k, score_threshold=0.0, return_str=True)
                    contexts.append(f"【本地知识库（低分补充）】\n{local_context}")
        
        elif self.strategy == "parallel":
            # 并行获取两者（实际顺序调用）
            local_context = self.local.execute(query, top_k=top_k, score_threshold=score_threshold, return_str=True)
            web_context = self.web.execute(query, loop_count=loop_count, return_str=True)
            contexts.append(f"【本地知识库】\n{local_context}")
            contexts.append(f"【网络搜索】\n{web_context}")
        
        elif self.strategy == "web_only":
            contexts.append(self.web.execute(query, loop_count=loop_count, return_str=True))
        
        else:  # local_only
            contexts.append(self.local.execute(query, top_k=top_k, score_threshold=score_threshold, return_str=True))
        
        return "\n\n".join(contexts)