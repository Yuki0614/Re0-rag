import type { QueryResult, SettingsForm, SourceItem, StreamEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail ?? detail;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchSources(): Promise<SourceItem[]> {
  const data = await request<{ items: SourceItem[] }>("/api/sources");
  return data.items;
}

export async function fetchSettings(): Promise<SettingsForm> {
  return request<SettingsForm>("/api/settings");
}

export async function saveSettings(settings: SettingsForm): Promise<SettingsForm> {
  return request<SettingsForm>("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
}

export async function askQuestionStream(
  question: string,
  threadId: string | undefined,
  onEvent: (event: StreamEvent) => void
): Promise<QueryResult> {
  const response = await fetch(`${API_BASE}/api/query-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId, trace: true })
  });

  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail ?? detail;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResult: QueryResult | undefined;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const event = JSON.parse(line.slice(6)) as StreamEvent;
      onEvent(event);
      if (event.type === "done") {
        finalResult = event.result;
      }
      if (event.type === "error") {
        throw new Error(event.error);
      }
    }
  }

  if (!finalResult) {
    throw new Error("问答流结束，但没有收到最终答案。");
  }
  return finalResult;
}

export async function uploadPdf(file: File): Promise<SourceItem[]> {
  const form = new FormData();
  form.append("file", file);
  const data = await request<{ sources: SourceItem[] }>("/api/upload", {
    method: "POST",
    body: form
  });
  return data.sources;
}

export async function deleteSource(source: string): Promise<void> {
  await request(`/api/sources/${encodeURIComponent(source)}`, {
    method: "DELETE"
  });
}
