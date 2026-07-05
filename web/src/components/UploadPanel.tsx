import { FileText, RefreshCw, Search, Trash2, UploadCloud, X } from "lucide-react";
import { ChangeEvent, useMemo, useRef, useState } from "react";
import type { SourceItem } from "../types";

type Props = {
  sources: SourceItem[];
  loading: boolean;
  uploading: boolean;
  deletingSource: string | null;
  error: string;
  onRefresh: () => void;
  onUpload: (file: File) => Promise<void>;
  onDelete: (source: string) => Promise<void>;
};

export function UploadPanel({
  sources,
  loading,
  uploading,
  deletingSource,
  error,
  onRefresh,
  onUpload,
  onDelete
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [query, setQuery] = useState("");
  const [confirmingSource, setConfirmingSource] = useState<SourceItem | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const isDeleting = Boolean(deletingSource);
  const filtered = useMemo(
    () =>
      sources.filter((source) => {
        const text = `${source.title} ${source.source} ${source.journal}`.toLowerCase();
        return text.includes(query.trim().toLowerCase());
      }),
    [sources, query]
  );

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
    event.target.value = "";
  }

  async function confirmDelete() {
    if (!confirmingSource || isDeleting) return;
    const source = confirmingSource.source;
    setDeleteError("");
    try {
      await onDelete(source);
      setConfirmingSource(null);
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <aside className="right-pane">
      <div className="right-title">
        <h2>知识库上传</h2>
        <button className="icon-button" onClick={onRefresh} disabled={loading || isDeleting} title="刷新">
          <RefreshCw size={17} />
        </button>
      </div>

      <section className="upload-box">
        <h2>上传文献</h2>
        <p>支持 PDF</p>
        <button className="drop-zone" onClick={() => inputRef.current?.click()} disabled={uploading || isDeleting}>
          <UploadCloud size={42} />
          <strong>{uploading ? "正在入库..." : "上传文件"}</strong>
          <span>单个文件不超过 200MB</span>
        </button>
        <input ref={inputRef} hidden type="file" accept="application/pdf,.pdf" onChange={handleFile} />
      </section>

      <section className="source-panel">
        <div className="source-head">
          <h2>知识库文件列表</h2>
        </div>
        <label className="search-box">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文件名" disabled={isDeleting} />
        </label>
        {isDeleting ? <div className="status-line">正在删除 {deletingSource}，请稍候...</div> : null}
        {error ? <div className="error-line">{error}</div> : null}
        <div className="source-list">
          {filtered.length === 0 ? (
            <div className="empty-line">{loading ? "正在读取..." : "暂无文献"}</div>
          ) : (
            filtered.map((source) => {
              const deletingThis = deletingSource === source.source;
              return (
                <article className={`source-item ${deletingThis ? "deleting" : ""}`} key={source.source}>
                  <FileText size={18} />
                  <div>
                    <strong>{source.title}</strong>
                    <span>{deletingThis ? "正在删除..." : `${source.source} · ${source.chunks} chunks`}</span>
                  </div>
                  <button
                    className="icon-button danger"
                    onClick={() => {
                      setDeleteError("");
                      setConfirmingSource(source);
                    }}
                    title="删除"
                    disabled={isDeleting}
                  >
                    <Trash2 size={16} />
                  </button>
                </article>
              );
            })
          )}
        </div>
      </section>

      {confirmingSource ? (
        <div className="modal-backdrop" role="presentation">
          <div className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <button
              className="modal-close"
              type="button"
              onClick={() => {
                setDeleteError("");
                setConfirmingSource(null);
              }}
              disabled={isDeleting}
              title="关闭"
            >
              <X size={18} />
            </button>
            <h3 id="delete-title">确认删除文献？</h3>
            <p>
              将从本地 Qdrant 索引和本地文件中删除
              <strong>{confirmingSource.title || confirmingSource.source}</strong>。
            </p>
            {deleteError ? <div className="modal-error">{deleteError}</div> : null}
            <div className="modal-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  setDeleteError("");
                  setConfirmingSource(null);
                }}
                disabled={isDeleting}
              >
                取消
              </button>
              <button className="danger-button" type="button" onClick={confirmDelete} disabled={isDeleting}>
                {isDeleting ? "正在删除..." : "确认删除"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
