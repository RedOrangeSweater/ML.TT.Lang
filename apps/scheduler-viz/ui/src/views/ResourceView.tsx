import React from "react";

interface Node {
  id: string;
  op_type: string;
  resource: string | null;
}

interface Props {
  nodes: Node[];
}

const RESOURCE_COLORS: Record<string, string> = {
  NOC: "#4a9",
  SFPU: "#c84",
  FPU: "#84c",
};

export function ResourceView({ nodes }: Props) {
  const byResource = nodes.reduce<Record<string, Node[]>>((acc, n) => {
    const r = n.resource ?? "dm";
    if (!acc[r]) acc[r] = [];
    acc[r].push(n);
    return acc;
  }, {});

  return (
    <div className="view-panel">
      <h3>Resource view (FPU / SFPU / NOC)</h3>
      {Object.entries(byResource).map(([resource, list]) => (
        <div key={resource} style={{ marginBottom: "1rem" }}>
          <span
            style={{
              backgroundColor: RESOURCE_COLORS[resource] ?? "#999",
              padding: "2px 8px",
              borderRadius: 4,
            }}
          >
            {resource}
          </span>
          <ul style={{ listStyle: "none", paddingLeft: "1rem", margin: "0.25rem 0 0" }}>
            {list.map((n) => (
              <li key={n.id}>{n.id} [{n.op_type}]</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
