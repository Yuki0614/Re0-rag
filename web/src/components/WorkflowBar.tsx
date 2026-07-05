import { BrainCircuit, CheckCircle2, GitBranch, Search, ShieldCheck, Sparkles } from "lucide-react";
import type { QueryResult } from "../types";

const fallbackSteps = [
  { label: "LangGraph 工作流", icon: GitBranch, state: "ready" },
  { label: "Qdrant 检索", icon: Search, state: "ready" },
  { label: "本地 Embedding", icon: BrainCircuit, state: "ready" },
  { label: "答案生成", icon: Sparkles, state: "ready" },
  { label: "Reflection", icon: ShieldCheck, state: "ready" }
];

type Props = {
  latestResult?: QueryResult;
};

export function WorkflowBar({ latestResult }: Props) {
  const route = latestResult?.route_history?.at(-1);
  const steps = fallbackSteps.map((step) => {
    if (step.label === "Qdrant 检索" && route?.action) {
      return { ...step, label: route.action === "keyword_search" ? "BM25 检索" : "Qdrant 检索" };
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
