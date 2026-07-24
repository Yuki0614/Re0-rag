import { Archive, ArrowDownUp, BrainCircuit, CheckCircle2, GitBranch, Network, Search, ShieldCheck, Sparkles } from "lucide-react";
import type { QueryResult } from "../types";

const fallbackSteps = [
  { label: "LangGraph 工作流", icon: GitBranch, state: "ready" },
  { label: "本地 Embedding", icon: BrainCircuit, state: "ready" },
  { label: "Qdrant 检索", icon: Search, state: "ready" },
  { label: "BCE Rerank 精排", icon: ArrowDownUp, state: "ready" },
  { label: "答案生成", icon: Sparkles, state: "ready" },
  { label: "Reflection", icon: ShieldCheck, state: "ready" }
];

type Props = {
  latestResult?: QueryResult;
};

export function WorkflowBar({ latestResult }: Props) {
  const route = latestResult?.route_history?.at(-1);
  let baseSteps = latestResult?.memory_summary_result?.triggered
    ? [
        fallbackSteps[0],
        { label: "记忆摘要", icon: Archive, state: "passed" },
        ...fallbackSteps.slice(1)
      ]
    : fallbackSteps;

  if (route?.use_graph) {
    const answerIndex = baseSteps.findIndex((step) => step.label === "答案生成");
    baseSteps = [
      ...baseSteps.slice(0, answerIndex),
      { label: "文献图谱增强", icon: Network, state: "passed" },
      ...baseSteps.slice(answerIndex)
    ];
  }

  const steps = baseSteps.map((step) => {
    if (step.label === "Qdrant 检索" && route?.action) {
      if (route.action === "no_retrieval") {
        return { ...step, label: "跳过本地检索", state: "skipped" };
      }
      return { ...step, label: route.action === "keyword_search" ? "BM25 初筛" : "Qdrant 初筛" };
    }
    if (step.label === "BCE Rerank 精排") {
      if (route?.action === "no_retrieval") {
        return { ...step, label: "跳过 BCE 精排", state: "skipped" };
      }
      if (latestResult?.retrieval?.rerank_enabled === false) {
        return { ...step, label: "BCE 精排已关闭", state: "skipped" };
      }
      const { candidate_count, result_count } = latestResult?.retrieval ?? {};
      if (candidate_count !== undefined && result_count !== undefined) {
        return { ...step, label: `BCE 精排 ${candidate_count} → ${result_count}` };
      }
    }
    if (step.label === "Reflection" && latestResult?.judge_result) {
      return { ...step, state: latestResult.judge_result.passed ? "passed" : "reviewed" };
    }
    return step;
  });

  return (
    <div className="workflow-bar">
      {steps.map(({ label, icon: Icon, state }) => (
        <div className="workflow-step" key={label}>
          <Icon size={17} />
          <span>{label}</span>
          {state === "passed" ? <CheckCircle2 className="step-ok" size={14} /> : <i />}
        </div>
      ))}
    </div>
  );
}
