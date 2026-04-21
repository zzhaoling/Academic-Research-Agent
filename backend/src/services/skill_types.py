from typing import TypedDict, List, Dict, Any

class RetrievalResult(TypedDict):
    content: str          # 文本内容
    score: float          # 相关度分数（归一化或原始分数）
    doc: str              # 文档名或来源标识（如 URL 或 PDF 文件名）
    page: int             # 页码（若无则为 0）
    # 可选扩展
    title: str | None     # 标题
    url: str | None       # 网页链接