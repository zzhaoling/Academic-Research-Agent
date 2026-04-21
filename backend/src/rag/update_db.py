import os
import hashlib
import json
from pathlib import Path
from hello_agents.tools import RAGTool
from dotenv import load_dotenv

# 加载环境变量（确保能读到 EMBED_* 配置）
load_dotenv()

# 初始化 RAGTool（会自动读取嵌入模型配置）
rag = RAGTool(
    knowledge_base_path="./src/rag/rag_knowledge",   # 向量库持久化目录
    rag_namespace="paper_library"            # 命名空间，隔离不同项目
)

# 原始 PDF 存放目录
PAPERS_DIR = Path("./src/papers")
STATE_FILE = Path("./src/rag/processed_files.json")   # 记录已处理文件的哈希值

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def update_database():
    # 确保目录存在
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载已处理记录
    processed = {}
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            processed = json.load(f)
    
    new_count = 0
    for pdf_path in PAPERS_DIR.glob("*.pdf"):
        current_hash = get_file_hash(pdf_path)
        if processed.get(str(pdf_path)) != current_hash:
            print(f"Processing {pdf_path.name} ...")
            try:
                # 调用 RAGTool 添加文档（自动分块、嵌入、存储）
                result = rag.run({"action": "add_document", "file_path": str(pdf_path)})
                print(f"  Result: {result}")
                processed[str(pdf_path)] = current_hash
                new_count += 1
            except Exception as e:
                print(f"  Error: {e}")
    
    # 保存更新后的状态
    with open(STATE_FILE, "w") as f:
        json.dump(processed, f, indent=2)
    
    print(f"Done. Processed {new_count} new/changed files.")
    return new_count

if __name__ == "__main__":
    update_database()