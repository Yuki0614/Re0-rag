import { useEffect, useMemo, useState } from "react";
import { askQuestionStream, deleteSource, fetchSources, uploadPdf } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { UploadPanel } from "./components/UploadPanel";
import type { AppPage, ChatMessage, SourceItem, StreamEvent } from "./types";

function nowTime() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

const initialMessages: ChatMessage[] = [
  {
    id: "welcome",
    role: "assistant",
    content: "你好，我是 Re0 RAG。选择一篇本地论文，或者直接问我文献里的方法、实验和结论。",
    time: nowTime()
  }
];

export default function App() {
  const [activePage, setActivePage] = useState<AppPage>("chat");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [threadId, setThreadId] = useState<string | undefined>();
  const [loadingSources, setLoadingSources] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deletingSource, setDeletingSource] = useState<string | null>(null);
  const [sourceError, setSourceError] = useState("");

  const latestResult = useMemo(
    () => [...messages].reverse().find((message) => message.result)?.result,
    [messages]
  );

  async function refreshSources() {
    setLoadingSources(true);
    setSourceError("");
    try {
      setSources(await fetchSources());
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingSources(false);
    }
  }

  useEffect(() => {
    refreshSources();
  }, []);

  async function handleAsk(question: string) {
    const pendingId = crypto.randomUUID();
    setBusy(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: question, time: nowTime() },
      {
        id: pendingId,
        role: "assistant",
        content: "正在启动 RAG 工作流",
        time: nowTime(),
        pending: true,
        progress: []
      }
    ]);

    function applyStreamEvent(event: StreamEvent) {
      if (event.type !== "progress") return;
      setMessages((current) =>
        current.map((message) =>
          message.id === pendingId
            ? {
                ...message,
                content: event.status,
                progress: [...(message.progress ?? []), { status: event.status, detail: event.detail }]
              }
            : message
        )
      );
    }

    try {
      const result = await askQuestionStream(question, threadId, applyStreamEvent);
      setThreadId(result.thread_id);
      setMessages((current) =>
        current.map((message) =>
          message.id === pendingId
            ? {
                ...message,
                content: result.answer || "没有生成回答。",
                result,
                pending: false
              }
            : message
        )
      );
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === pendingId
            ? {
                ...message,
                content: "这次问答没有完成。",
                error: error instanceof Error ? error.message : String(error),
                pending: false
              }
            : message
        )
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setSourceError("");
    try {
      setSources(await uploadPdf(file));
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : String(error));
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(source: string) {
    if (deletingSource) return;
    setDeletingSource(source);
    setSourceError("");
    try {
      await deleteSource(source);
      await refreshSources();
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      setDeletingSource(null);
    }
  }

  return (
    <main className="app-shell">
      <Sidebar activePage={activePage} sourcesCount={sources.length} onPageChange={setActivePage} />
      {activePage === "chat" ? (
        <ChatPanel
          busy={busy}
          messages={messages}
          sources={sources}
          latestResult={latestResult}
          onAsk={handleAsk}
        />
      ) : (
        <SettingsPanel />
      )}
      <UploadPanel
        sources={sources}
        loading={loadingSources}
        uploading={uploading}
        deletingSource={deletingSource}
        error={sourceError}
        onRefresh={refreshSources}
        onUpload={handleUpload}
        onDelete={handleDelete}
      />
    </main>
  );
}
