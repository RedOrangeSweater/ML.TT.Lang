import React, { useCallback, useState } from "react";
import { VerticalGraphView } from "./views/VerticalGraphView";
import { ResourceView } from "./views/ResourceView";
import { HWTopologyView } from "./views/HWTopologyView";

const API_BASE = "";

interface SchedulerState {
  op_graph: { nodes: Array<{ id: string; op_type: string; resource: string | null; predecessors: string[]; successors: string[] }> };
  topology: { grid_cols: number; grid_rows: number };
  plan: { placement: Record<string, [number, number]>; grid_cols: number; grid_rows: number };
}

export default function App() {
  const [state, setState] = useState<SchedulerState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [actionSpaceN, setActionSpaceN] = useState<number>(0);
  const [nextNode, setNextNode] = useState<number>(0);
  const [reward, setReward] = useState<number | null>(null);
  const [planB, setPlanB] = useState<Record<string, [number, number]> | null>(null);

  const loadFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setError(null);
    setLoading(true);
    f.text()
      .then((text) => JSON.parse(text))
      .then(async (data) => {
        const res = await fetch(`${API_BASE}/load`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(await res.text());
        const out = await res.json();
        setState(out.full_state);
        setActionSpaceN(out.action_space_n ?? 0);
        setNextNode(out.observation?.next_node ?? 0);
        setProgress(out.full_state?.plan?.placement ? Object.keys(out.full_state.plan.placement).length : 0);
        setReward(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const loadPlanB = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    f.text()
      .then((text) => JSON.parse(text))
      .then((data: { plan?: { placement?: Record<string, [number, number]> }; placement?: Record<string, [number, number]> }) => {
        const pl = data.plan?.placement ?? data.placement ?? {};
        setPlanB(pl as Record<string, [number, number]>);
      })
      .catch(() => setPlanB(null));
  }, []);

  const step = useCallback(async (action: number) => {
    if (state == null) return;
    setError(null);
    const res = await fetch(`${API_BASE}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    const out = await res.json();
    setState(out.full_state);
    setNextNode(out.observation?.next_node ?? 0);
    const pl = out.full_state?.plan?.placement ?? {};
    setProgress(Object.keys(pl).length);
    if (out.terminated) setReward(out.reward);
  }, [state]);

  const nNodes = state?.op_graph?.nodes?.length ?? 0;
  const runAll = useCallback(async () => {
    if (state == null || actionSpaceN === 0 || nNodes === 0) return;
    for (let i = 0; i < nNodes; i++) {
      await step(0);
    }
  }, [state, actionSpaceN, nNodes, step]);

  const nodes = state?.op_graph?.nodes ?? [];
  const placement = state?.plan?.placement as Record<string, [number, number]> | undefined;
  const nTotal = nodes.length;
  const progressPct = nTotal > 0 ? Math.round((progress / nTotal) * 100) : 0;
  const canStep = actionSpaceN > 0 && progress < nTotal;

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: "0 auto" }}>
      <h1>Scheduler Viz</h1>
      <div style={{ marginBottom: 16 }}>
        <label>
          Load JSON: <input type="file" accept=".json" onChange={loadFile} disabled={loading} />
        </label>
        {state != null && (
          <label style={{ marginLeft: 16 }}>
            Compare Plan B: <input type="file" accept=".json" onChange={loadPlanB} />
          </label>
        )}
        {loading && <span style={{ marginLeft: 8 }}>Loading...</span>}
        {error != null && <div style={{ color: "crimson", marginTop: 8 }}>{error}</div>}
      </div>

      {state != null && (
        <>
          <div style={{ marginBottom: 16, display: "flex", gap: 16, alignItems: "center" }}>
            <span>Progress: {progress} / {nTotal} ({progressPct}%)</span>
            {reward != null && <span>Reward: {reward}</span>}
            {canStep && (
              <>
                <button type="button" onClick={() => step(0)}>Step (core 0)</button>
                <button type="button" onClick={runAll}>Run</button>
              </>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            <VerticalGraphView nodes={nodes} placement={placement} />
            <ResourceView nodes={nodes} />
            <HWTopologyView
              gridCols={state.topology.grid_cols}
              gridRows={state.topology.grid_rows}
              placement={placement}
              nodeIds={nodes.map((n) => n.id)}
            />
          </div>
          {planB != null && (
            <div style={{ marginTop: 24 }}>
              <h3>Compare: Plan A (env) vs Plan B</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <h4>Plan A (current)</h4>
                  <HWTopologyView
                    gridCols={state.topology.grid_cols}
                    gridRows={state.topology.grid_rows}
                    placement={placement}
                    nodeIds={nodes.map((n) => n.id)}
                  />
                </div>
                <div>
                  <h4>Plan B</h4>
                  <HWTopologyView
                    gridCols={state.topology.grid_cols}
                    gridRows={state.topology.grid_rows}
                    placement={planB}
                    nodeIds={nodes.map((n) => n.id)}
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {state == null && !loading && (
        <p>Load a scheduler export JSON (from tt-lang export_scheduler_input_to_json) and start backend: uvicorn backend:app --reload --port 8000</p>
      )}
    </div>
  );
}
