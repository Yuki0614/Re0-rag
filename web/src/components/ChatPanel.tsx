import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Send,
  ThumbsDown,
  ThumbsUp,
  UserRound
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import type { ChatMessage, QueryResult, SourceItem } from "../types";
import { WorkflowBar } from "./WorkflowBar";

type Props = {
  busy: boolean;
  messages: ChatMessage[];
  sources: SourceItem[];
  latestResult?: QueryResult;
  onAsk: (question: string) => Promise<void>;
};

type Vote = "up" | "down" | null;

export function ChatPanel({ busy, messages, sources, latestResult, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const [expandedFlows, setExpandedFlows] = useState<Set<string>>(new Set());
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = listRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || busy) return;
    setQuestion("");
    await onAsk(value);
  }

  function toggleFlow(id: string) {
    setExpandedFlows((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <section className="chat-pane">
      <header className="pane-header">
        <div>
          <h1>智能问答</h1>
          <p>{sources.length} 篇文献已入库</p>
        </div>
        <button className="ghost-button" type="button" onClick={() => navigator.clipboard?.writeText(window.location.href)}>
          <Copy size={16} />
          复制链接
        </button>
      </header>

      <WorkflowBar latestResult={latestResult} />

      <div className="message-list" ref={listRef}>
        {messages.map((message) => (
          <article className={`message-row ${message.role}`} key={message.id}>
            <div className="message-icon">
              {message.role === "user" ? (
                <UserRound size={20} />
              ) : (
                <img className="assistant-avatar" src="/assistant-avatar.jpg" alt="Re0 RAG" />
              )}
            </div>
            <div className="message-card">
              <div className="message-meta">
                <span>{message.role === "user" ? "你" : "Re0 RAG"}</span>
                <time>{message.time}</time>
              </div>
              <p className={message.pending ? "shimmer-text" : ""}>{message.content}</p>
              {message.progress?.length ? (
                <FlowDetails
                  message={message}
                  expanded={message.pending || expandedFlows.has(message.id)}
                  onToggle={() => toggleFlow(message.id)}
                />
              ) : null}
              {message.error ? <div className="error-line">{message.error}</div> : null}
              {message.result ? <ResultDetails result={message.result} /> : null}
            </div>
          </article>
        ))}
      </div>

      <form className="ask-box" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              handleSubmit(event);
            }
          }}
          placeholder="输入你的论文问题..."
        />
        <button className="send-button" disabled={busy || !question.trim()} type="submit">
          <Send size={18} />
          发送
        </button>
      </form>
    </section>
  );
}

function FlowDetails({
  message,
  expanded,
  onToggle
}: {
  message: ChatMessage;
  expanded: boolean;
  onToggle: () => void;
}) {
  const steps = message.progress ?? [];
  const lastStep = steps.at(-1);
  return (
    <div className={`flow-panel ${expanded ? "expanded" : ""}`}>
      <button className="flow-toggle" type="button" onClick={onToggle}>
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>{message.pending ? "正在处理" : "RAG 流程"}</span>
        <b>{lastStep?.status ?? "准备中"}</b>
      </button>
      <div className="flow-content">
        <ol className="flow-list">
          {steps.map((step, index) => (
            <li key={`${step.status}-${index}`}>
              <span>{step.status}</span>
              <small>{step.detail}</small>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function ResultDetails({ result }: { result: QueryResult }) {
  const [vote, setVote] = useState<Vote>(null);
  const [copied, setCopied] = useState(false);

  function toggleVote(nextVote: Exclude<Vote, null>) {
    setVote((current) => (current === nextVote ? null : nextVote));
  }

  async function copyAnswer() {
    await navigator.clipboard?.writeText(result.answer);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="result-block">
      <div className="message-actions">
        <button
          className={vote === "up" ? "active" : ""}
          title={vote === "up" ? "已标记有帮助" : "有帮助"}
          onClick={() => toggleVote("up")}
          type="button"
        >
          <ThumbsUp size={15} />
        </button>
        <button
          className={vote === "down" ? "active danger" : ""}
          title={vote === "down" ? "已标记没帮助" : "没帮助"}
          onClick={() => toggleVote("down")}
          type="button"
        >
          <ThumbsDown size={15} />
        </button>
        <button className={copied ? "active" : ""} title={copied ? "已复制" : "复制回答"} onClick={copyAnswer} type="button">
          {copied ? <Check size={15} /> : <Copy size={15} />}
        </button>
        {vote ? <span className="action-feedback">{vote === "up" ? "已标记有帮助" : "已标记没帮助"}</span> : null}
        {copied ? <span className="action-feedback">已复制</span> : null}
      </div>

      {result.evidence.length ? (
        <div className="evidence-grid">
          {result.evidence.map((item, index) => (
            <div className="evidence-card" key={`${item.source}-${index}`}>
              <div className="evidence-top">
                <span>{item.source || item.title || "文献片段"}</span>
                {typeof item.score === "number" ? <b>相关度 {item.score.toFixed(2)}</b> : null}
              </div>
              <p>{item.content}</p>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
