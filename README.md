# HelloAgents 深度研究助手

基于 **HelloAgents** 框架的智能学术研究助手，支持：
- 自动拆解研究主题为可执行任务
- 混合检索（本地知识库 PDF + 网络搜索 + RRF 融合 + Cross-Encoder 重排）
- 任务状态机（自动重试、质量评分、策略调整）
- 笔记持久化（每个任务独立 Markdown 笔记）
- 最终结构化报告生成

## 功能特性

- 🔍 **多源检索**：本地 PDF 向量 + BM25 + 网络搜索（DuckDuckGo/Tavily/Perplexity）
- 🧠 **智能规划**：使用 LLM 将主题拆分为 3~5 个互补的研究任务
- 📝 **任务笔记**：每个任务自动创建/更新笔记，便于追溯和协作
- 🔁 **自动重试**：任务质量未达阈值时自动调整检索策略并重试
- 📊 **最终报告**：汇总所有任务结果生成结构化 Markdown 报告

## 技术栈

- **后端**：Python 3.10+ / FastAPI / Qdrant / rank-bm25 / sentence-transformers / cross-encoder
- **前端**：Vue 3 / TypeScript / Vite
- **LLM 支持**：OpenAI 兼容 API（Ollama, LM Studio, 自定义）

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/yourname/helloagents-deepresearch.git
cd helloagents-deepresearch

# 创建虚拟环境（后端）
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 安装后端依赖
pip install -r requirements.txt   # 请根据实际依赖生成
```
### 2. 配置环境变量
复制 .env.example 为 .env 并修改关键配置：
```bash
# LLM 配置（以 Ollama 为例）
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_ID=llama3.2

# 搜索后端
SEARCH_API=tavily

# 笔记存储路径
NOTES_WORKSPACE=./src/notes

# 混合检索（可选）
ENABLE_HYBRID_RETRIEVAL=true
PDF_DIR=./src/papers
QDRANT_MODE=embedded
```
### 3. 构建本地知识库索引
将 PDF 文件放入 src/papers/，然后运行：

```bash
python src/build_index.py
```
该脚本会：
- 解析 PDF 并按页分块
- 建立 Qdrant 向量索引
- 保存 BM25 索引 (store/bm25.pkl)

### 4. 启动服务
```bash
# 启动后端 API
uvicorn main:app --reload --port 8000

# 启动前端（另开终端）
cd frontend
npm install
npm run dev
```
