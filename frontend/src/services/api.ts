const baseURL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ResearchRequest {
  topic: string;
  search_api?: string;
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  const response = await fetch(`${baseURL}/research/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `研究请求失败，状态码：${response.status}`
    );
  }

  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应，无法获取研究进度");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent.startsWith("data:")) {
        const dataPayload = rawEvent.slice(5).trim();
        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            onEvent(event);

            if (event.type === "error" || event.type === "done") {
              return;
            }
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      // 处理可能的尾巴事件
      if (buffer.trim()) {
        const rawEvent = buffer.trim();
        if (rawEvent.startsWith("data:")) {
          const dataPayload = rawEvent.slice(5).trim();
          if (dataPayload) {
            try {
              const event = JSON.parse(dataPayload) as ResearchStreamEvent;
              onEvent(event);
            } catch (error) {
              console.error("解析流式事件失败：", error, dataPayload);
            }
          }
        }
      }
      break;
    }
  }
}
// ==================== 新增 API 接口 ====================

/**
 * 上传文档（PDF 等）到知识库
 * @param file 要上传的文件
 * @returns 服务器返回的上传结果
 */
export async function uploadDocument(file: File): Promise<{
  status: string;
  filename: string;
  saved_path: string;
}> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${baseURL}/upload_document`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `上传失败，状态码：${response.status}`);
  }
  return response.json();
}

/**
 * 检索论文（RAG 知识库）
 * @param query 检索查询
 * @param limit 返回结果数量（默认5）
 * @returns 检索结果
 */
export async function searchPapers(
  query: string,
  limit: number = 5
): Promise<{
  query: string;
  results: Array<{
    content?: string;
    text?: string;
    score?: number;
    metadata?: Record<string, unknown>;
  }>;
}> {
  const url = new URL(`${baseURL}/search_papers`);
  url.searchParams.append("query", query);
  url.searchParams.append("limit", String(limit));
  const response = await fetch(url.toString(), {
    method: "POST",
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `检索失败，状态码：${response.status}`);
  }
  return response.json();
}

/**
 * 记忆回顾（从记忆系统检索历史研究记忆）
 * @param query 回顾查询
 * @param limit 返回记忆数量（默认5）
 * @returns 记忆列表
 */
export async function recallMemories(
  query: string,
  limit: number = 5
): Promise<{
  memories: Array<{
    content?: string;
    memory_type?: string;
    importance?: number;
    timestamp?: string;
    [key: string]: unknown;
  }>;
}> {
  const url = new URL(`${baseURL}/recall`);
  url.searchParams.append("query", query);
  url.searchParams.append("limit", String(limit));
  const response = await fetch(url.toString(), {
    method: "POST",
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `记忆回顾失败，状态码：${response.status}`);
  }
  return response.json();
}