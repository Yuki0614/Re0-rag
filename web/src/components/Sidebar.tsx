import { Database, MessageSquareText, Settings } from "lucide-react";
import type { AppPage } from "../types";

type Props = {
  activePage: AppPage;
  sourcesCount: number;
  onPageChange: (page: AppPage) => void;
};

export function Sidebar({ activePage, sourcesCount, onPageChange }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <img className="brand-avatar" src="/assistant-avatar.jpg" alt="Re0 RAG" />
        </div>
        <div>
          <strong>Re0 RAG</strong>
          <span>本地论文助手</span>
        </div>
      </div>

      <nav className="nav-stack" aria-label="主导航">
        <button
          className={`nav-item ${activePage === "chat" ? "active" : ""}`}
          onClick={() => onPageChange("chat")}
          type="button"
        >
          <MessageSquareText size={20} />
          智能问答
        </button>
        <button
          className={`nav-item ${activePage === "settings" ? "active" : ""}`}
          onClick={() => onPageChange("settings")}
          type="button"
        >
          <Settings size={20} />
          系统设置
        </button>
      </nav>

      <div className="sidebar-spacer" />

      <div className="profile-strip">
        <div className="avatar">R</div>
        <div>
          <strong>researcher</strong>
          <span>
            <Database size={13} />
            {sourcesCount} 篇文献
          </span>
        </div>
      </div>
    </aside>
  );
}
