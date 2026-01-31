import React from "react";

interface Props {
  gridCols: number;
  gridRows: number;
  placement?: Record<string, [number, number]>;
  nodeIds?: string[];
}

export function HWTopologyView({ gridCols, gridRows, placement, nodeIds }: Props) {
  const cells: { col: number; row: number; labels: string[] }[] = [];
  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      const labels =
        placement && nodeIds
          ? nodeIds.filter((nid) => {
              const p = placement[nid];
              return p && p[0] === c && p[1] === r;
            })
          : [];
      cells.push({ col: c, row: r, labels });
    }
  }

  return (
    <div className="view-panel">
      <h3>HW topology (floor / cities with distances)</h3>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${gridCols}, 60px)`,
          gap: 4,
          width: "fit-content",
        }}
      >
        {cells.map(({ col, row, labels }) => (
          <div
            key={`${col},${row}`}
            style={{
              border: "1px solid #ccc",
              padding: 4,
              minHeight: 40,
              fontSize: 11,
            }}
          >
            ({col},{row})
            {labels.length > 0 && (
              <div style={{ marginTop: 4, color: "#06c" }}>
                {labels.join(", ")}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
