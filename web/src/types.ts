export type SourceItem = {
  source: string;
  chunks: number;
  title: string;
  authors: string[];
  journal: string;
  abstract: string;
  status: "ready" | "indexing" | "failed";
};

export type EvidenceItem = {
  content: string;
  source: string;
  title: string;
  page?: string | number;
  score?: number;
  doc_type: string;
};

export type QueryResult = {
  answer: string;
  thread_id: string;
  sources: string[];
  evidence: EvidenceItem[];
  documents: EvidenceItem[];
  route_history: Array<{
    action: string;
    query: string;
    reason?: string;
  }>;
  judge_result: {
    passed?: boolean;
    reason?: string;
    suggested_action?: string;
  };
  memory_summary_result?: {
    triggered?: boolean;
    removed_count?: number;
    summary_length?: number;
  };
};

export type FlowStep = {
  status: string;
  detail: string;
};

export type StreamEvent =
  | { type: "progress"; status: string; detail: string }
  | { type: "done"; result: QueryResult }
  | { type: "error"; error: string };

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  time: string;
  result?: QueryResult;
  progress?: FlowStep[];
  pending?: boolean;
  error?: string;
};

export type AppPage = "chat" | "settings";

export type SettingsForm = {
  base_url: string;
  api_key: string;
  model: string;
};
