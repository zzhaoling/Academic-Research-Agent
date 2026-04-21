from typing import List, Dict, Any, Optional
from config import Configuration
import logging
# from services.search import dispatch_search, prepare_research_context
# logger = logging.getLogger(__name__)
# class WebSearchSkill:
#     def __init__(self, config: Configuration):
#         self.config = config

#     def execute(
#         self,
#         query: str,
#         loop_count: int = 0,
#         return_str: bool = True
#     ) -> str | Dict[str, Any]:
#         """
#         执行网络搜索，返回上下文文本或原始结果
#         """
#         try:
#             search_result, notices, answer_text, backend = dispatch_search(
#                 query, self.config, loop_count
#             )
#             if not search_result or not search_result.get("results"):
#                 return "未找到相关网络信息。" if return_str else {"error": "no_results"}
            
#             sources_summary, context = prepare_research_context(
#                 search_result, answer_text, self.config
#             )
#             if return_str:
#                 return context
#             else:
#                 return {
#                     "context": context,
#                     "sources_summary": sources_summary,
#                     "backend": backend
#                 }
#         except Exception as e:
#             logger.warning(f"Web search failed: {e}")
#             return "网络搜索失败。" if return_str else {"error": str(e)}