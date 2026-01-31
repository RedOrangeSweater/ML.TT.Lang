import React from "react";

interface Node {
  id: string;
  op_type: string;
  resource: string | null;
  predecessors: string[];
  successors: string[];
}

interface Props {
  nodes: Node[];
  placement?: Record<string, [number, number]>;
}

export function VerticalGraphView({ nodes, placement }: Props) {
  return (
    <div className="view-panel">
      <h3>Vertical op graph (algo nodes)</h3>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {nodes.map((n) => (
          <li key={n.id} style={{ marginBottom: "0.5rem" }}>
            <strong>{n.id}</strong> [{n.op_type}]
            {n.resource != null ? ` (${n.resource})` : ""}
            {placement?.[n.id] != null && (
              <span style={{ marginLeft: "0.5rem", color: "#666" }}>
                → core {placement[n.id]![0]},{placement[n.id]![1]}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
