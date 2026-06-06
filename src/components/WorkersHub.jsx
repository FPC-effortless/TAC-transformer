import { useMemo, useState } from "react";

export default function WorkersHub({ workers, orders, onToggleOrderTask }) {
  const [selectedWorkerName, setSelectedWorkerName] = useState(workers[0]?.name || "");

  // Compute active commissions and task metrics per worker
  const workersMetrics = useMemo(() => {
    return workers.map(worker => {
      const activeGarments = orders.filter(o => 
        o.currentStage !== "Completed & Ready" &&
        Object.values(o.workerAssignments).some(name => name.toLowerCase() === worker.name.toLowerCase())
      );

      // Collect specific assignments in orders
      const assignments = [];
      orders.forEach(o => {
        if (o.currentStage === "Completed & Ready") return;
        Object.entries(o.workerAssignments).forEach(([role, name]) => {
          if (name.toLowerCase() === worker.name.toLowerCase()) {
            assignments.push({
              orderId: o.id,
              garmentName: o.garmentName,
              clientName: o.clientName,
              roleInOrder: role,
              stage: o.currentStage
            });
          }
        });
      });

      // Get all tasks assigned to this worker's current orders
      const pendingTasks = [];
      const completedTasks = [];
      
      activeGarments.forEach(order => {
        order.tasks.forEach(t => {
          // If the task corresponds to the worker's stage
          // For simplicity, we show tasks for the active orders they are involved in
          if (t.completed) {
            completedTasks.push({ ...t, orderId: order.id, orderName: order.garmentName });
          } else {
            pendingTasks.push({ ...t, orderId: order.id, orderName: order.garmentName });
          }
        });
      });

      return {
        ...worker,
        activeCount: activeGarments.length,
        assignments,
        pendingTasks,
        completedTasks
      };
    });
  }, [workers, orders]);

  const activeWorkerMetric = useMemo(() => {
    return workersMetrics.find(wm => wm.name.toLowerCase() === selectedWorkerName.toLowerCase()) || workersMetrics[0];
  }, [workersMetrics, selectedWorkerName]);

  // Color mappings for worker visual bar
  const workerColors = {
    "marie": "#C5A880", // gold
    "david": "#2B4C7E", // blue
    "elena": "#2E5A44", // green
    "fatima": "#8C3B4E" // burgundy
  };

  return (
    <div className="anim-slide-up" style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      
      {/* Workers Workspace Header */}
      <div className="workspace-header">
        <div className="header-title-block">
          <h2>Atelier Staff & Workroom Floor</h2>
          <p>Assign tasks, review workload allocations, and check worker-specific progress tables.</p>
        </div>
      </div>

      {/* Visual Workload Balance Visualizer */}
      <div className="couture-card">
        <h3 className="font-serif" style={{ fontSize: "20px", marginBottom: "8px" }}>Interactive Workroom Balance</h3>
        <p style={{ color: "var(--text-secondary)", fontSize: "13px" }}>
          Relative distribution of active tailoring duties currently allocated on the production floor.
        </p>

        {/* Dynamic Multi-segment Bar */}
        <div className="worker-distribution-bar">
          {workersMetrics.map(wm => {
            const totalLoad = workersMetrics.reduce((acc, curr) => acc + curr.activeCount, 0);
            const percentage = totalLoad > 0 ? (wm.activeCount / totalLoad) * 100 : 0;
            if (percentage === 0) return null;

            return (
              <div 
                key={wm.id}
                className="worker-dist-chunk"
                style={{ 
                  width: `${percentage}%`, 
                  backgroundColor: workerColors[wm.name.toLowerCase()] || "var(--accent)" 
                }}
                onClick={() => setSelectedWorkerName(wm.name)}
              >
                <div className="worker-dist-tooltip">
                  {wm.name}: {wm.activeCount} Garments ({Math.round(percentage)}%)
                </div>
              </div>
            );
          })}
        </div>

        {/* Legend / Directory list */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px", marginTop: "24px" }}>
          {workersMetrics.map(wm => {
            const color = workerColors[wm.name.toLowerCase()] || "var(--accent)";
            const isSelected = selectedWorkerName.toLowerCase() === wm.name.toLowerCase();
            
            return (
              <div 
                key={wm.id}
                className="worker-card"
                onClick={() => setSelectedWorkerName(wm.name)}
                style={{ 
                  cursor: "pointer",
                  borderColor: isSelected ? color : "var(--border-color)",
                  borderWidth: isSelected ? "2px" : "1px",
                  boxShadow: isSelected ? `0 0 12px ${color}20` : "none",
                  backgroundColor: isSelected ? "var(--accent-light)" : "var(--bg-card)"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <span className="user-avatar" style={{ border: `1px solid ${color}`, backgroundColor: `${color}15` }}>
                    {wm.avatar}
                  </span>
                  <div>
                    <h4 style={{ fontSize: "14px", fontWeight: "700" }}>{wm.name}</h4>
                    <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>{wm.role}</span>
                  </div>
                </div>
                <span className="badge" style={{ backgroundColor: `${color}15`, color: color }}>
                  {wm.activeCount} Active
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Worker Dedicated Bench / Task Center */}
      <div className="dashboard-workspace-grid" style={{ gridTemplateColumns: "1.2fr 2fr" }}>
        
        {/* Left Bench Column: Active Assignments List */}
        <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div>
            <h3 style={{ fontSize: "18px", fontWeight: "600", fontFamily: "var(--font-serif)" }}>
              {activeWorkerMetric.name}'s Assigned Board
            </h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "12px", marginTop: "4px" }}>
              Active pieces where {activeWorkerMetric.name} is tasked with production.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {activeWorkerMetric.assignments.length === 0 ? (
              <div style={{ textAlign: "center", padding: "32px 0", color: "var(--text-muted)", fontSize: "13px" }}>
                💤 No active garment assignments. Bench is currently clear!
              </div>
            ) : (
              activeWorkerMetric.assignments.map(ass => (
                <div 
                  key={ass.orderId}
                  className="couture-card"
                  style={{ 
                    padding: "16px", 
                    backgroundColor: "var(--bg-hover)", 
                    borderLeft: `4px solid ${workerColors[activeWorkerMetric.name.toLowerCase()]}`
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", fontWeight: "600", marginBottom: "4px" }}>
                    <span>{ass.orderId}</span>
                    <span style={{ textTransform: "uppercase", color: "var(--accent-hover)" }}>Role: {ass.roleInOrder}</span>
                  </div>
                  <h4 style={{ fontSize: "13px", fontWeight: "700" }}>{ass.garmentName}</h4>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px" }}>
                    <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>Client: {ass.clientName}</span>
                    <span className="badge badge-warning" style={{ fontSize: "9px" }}>{ass.stage}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Bench Column: Live Interactive Checklist */}
        <div className="couture-card">
          <div className="card-title-block">
            <div>
              <h3 className="font-serif" style={{ fontSize: "24px" }}>
                {activeWorkerMetric.name}'s Bench Workroom
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
                Select checkmarks directly to update master garment schedules.
              </p>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {/* Uncompleted Checklist */}
            <div>
              <h4 style={{ fontSize: "13px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px", marginBottom: "12px" }}>
                Unfinished Steps ({activeWorkerMetric.pendingTasks.length})
              </h4>
              
              {activeWorkerMetric.pendingTasks.length === 0 ? (
                <div style={{ color: "var(--success)", fontSize: "13px", padding: "16px", backgroundColor: "var(--success-light)", borderRadius: "var(--radius)", textAlign: "center" }}>
                  ✨ All steps checked off! Brilliant craftsmanship.
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column" }}>
                  {activeWorkerMetric.pendingTasks.map(t => (
                    <div 
                      key={t.id} 
                      className="checklist-item"
                      onClick={() => onToggleOrderTask(t.orderId, t.id)}
                    >
                      <div className="checklist-checkbox"></div>
                      <div className="checklist-text">
                        <strong>{t.text}</strong>
                        <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                          Garment: {t.orderName} ({t.orderId})
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Completed Checklist */}
            {activeWorkerMetric.completedTasks.length > 0 && (
              <div>
                <h4 style={{ fontSize: "13px", textTransform: "uppercase", letterSpacing: "1px", color: "var(--text-secondary)", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px", marginBottom: "12px" }}>
                  Recently Handled Tasks ({activeWorkerMetric.completedTasks.length})
                </h4>
                <div style={{ display: "flex", flexDirection: "column", opacity: 0.75 }}>
                  {activeWorkerMetric.completedTasks.map(t => (
                    <div 
                      key={t.id} 
                      className="checklist-item completed"
                      onClick={() => onToggleOrderTask(t.orderId, t.id)}
                    >
                      <div className="checklist-checkbox checked"></div>
                      <div className="checklist-text completed">
                        <strong>{t.text}</strong>
                        <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                          Garment: {t.orderName} ({t.orderId})
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
