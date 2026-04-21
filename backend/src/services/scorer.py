# services/scorer.py
import re
from typing import Optional

class TaskScorer:
    """对任务总结进行质量评分（返回 0~1 之间）"""
    
    @staticmethod
    def score_summary(summary: Optional[str]) -> float:
        if not summary or "暂无可用信息" in summary:
            return 0.0
        
        # 长度分数（最多500字符）
        length_score = min(1.0, len(summary) / 500)
        
        # 是否包含数字（指标/数据）
        has_numbers = 0.2 if re.search(r'\d+', summary) else 0.0
        
        # 是否有引用标记 [1] 或 (作者,年份)
        has_citations = 0.2 if re.search(r'\[(\d+)\]|\([^)]*\d{4}\)', summary) else 0.0
        
        # 是否有结构化标题（### 或 1. 2. 等）
        structure_bonus = 0.2 if re.search(r'###|\d+\.\s', summary) else 0.0
        
        return min(1.0, length_score + has_numbers + has_citations + structure_bonus)