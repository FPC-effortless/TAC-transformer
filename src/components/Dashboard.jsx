import { useMemo, useState } from "react";

export default function Dashboard({ orders, workers, onViewOrder, onNewOrder }) {
  const [filterWorker, setFilterWorker] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  // Get current greeting based on time
  const greeting = useMemo(() => {
    const hours = new Date().getHours();
    if (hours < 12) return "Good Morning";
    if (hours < 18) return "Good Afternoon";
    return "Good Evening";
  }, []);

  // Compute Statistics
  const stats = useMemo(() => {
    const totalOrders = orders.length;
    const activeOrders = orders.filter(o => o.currentStage !== "Completed & Ready").length;
    const inFitting = orders.filter(o => o.currentStage === "Fitting Session").length;
    const readyForQC = orders.filter(o => o.currentStage === "Finishing & QC").length;
    
    // Financial calculations
    let totalRevenue = 0;
    let totalDeposits = 0;
    orders.forEach(o => {
      totalRevenue += o.price || 0;
      totalDeposits += o.deposit || 0;
    });
    
    const pendingBalance = totalRevenue - totalDeposits;

    return {
      totalOrders,
      activeOrders,
      inFitting,
      readyForQC,
      totalRevenue,
      totalDeposits,
      pendingBalance
    };
  }, [orders]);

  // Compute remaining days and sort urgent orders first
  const deadlineWatchlist = useMemo(() => {
    const active = orders.filter(o => o.currentStage !== "Completed & Ready");
    
    return active
      .map(order => {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const due = new Date(order.deadline);
        due.setHours(0, 0, 0, 0);
        
        const diffTime = due.getTime() - today.getTime();
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        let status = "on-track";
        if (diffDays <= 3) status = "urgent";
        else if (diffDays <= 7) status = "approaching";
        
        return {
          ...order,
          daysRemaining: diffDays,
          timelineStatus: status
        };
      })
      .sort((a, b) => a.daysRemaining - b.daysRemaining);
  }, [orders]);

  // Filtered Orders list
  const filteredOrders = useMemo(() => {
    return orders.filter(order => {
      const matchesWorker = filterWorker === "all" || 
        Object.values(order.workerAssignments).some(name => name.toLowerCase() === filterWorker.toLowerCase());
      
      const matchesSearch = searchQuery === "" || 
        order.clientName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        order.garmentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        order.fabric.toLowerCase().includes(searchQuery.toLowerCase()) ||
        order.id.toLowerCase().includes(searchQuery.toLowerCase());
        
      return matchesWorker && matchesSearch;
    });
  }, [orders, filterWorker, searchQuery]);

  return (
    <div className="anim-slide-up" style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      
      {/* Editorial Welcome Header */}
      <div className="couture-card" style={{ background: "linear-gradient(135deg, var(--bg-card), var(--accent-light))", padding: "40px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <span style={{ fontSize: "11px", letterSpacing: "3px", textTransform: "uppercase", color: "var(--accent)", fontWeight: "600" }}>
              Bibiere Atelier Dashboard
            </span>
            <h2 className="font-serif" style={{ fontSize: "42px", fontWeight: "300", marginTop: "8px" }}>
              {greeting}, Sister & Team
            </h2>
            <p style={{ color: "var(--text-secondary)", marginTop: "8px", maxWidth: "600px", fontSize: "14px" }}>
              Welcome back to your couture suite. You have <strong style={{ color: "var(--text-primary)" }}>{stats.activeOrders} active commissions</strong> currently in development on the workroom floor.
            </p>
          </div>
          <button className="btn-luxury btn-gold" onClick={onNewOrder}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Commission
          </button>
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="stats-grid">
        <div className="couture-card stat-card">
          <span className="stat-label">Active Commissions</span>
          <span className="stat-value">{stats.activeOrders}</span>
          <div className="stat-footer">
            <span className="text-info">●</span> {stats.inFitting} in fittings right now
          </div>
        </div>
        
        <div className="couture-card stat-card">
          <span className="stat-label">Fitting Sessions</span>
          <span className="stat-value">{stats.inFitting}</span>
          <div className="stat-footer">
            <span className="text-warning">●</span> Requires client appointments
          </div>
        </div>

        <div className="couture-card stat-card">
          <span className="stat-label">Finishing & QC</span>
          <span className="stat-value">{stats.readyForQC}</span>
          <div className="stat-footer">
            <span className="text-success">●</span> Nearing delivery readiness
          </div>
        </div>

        <div className="couture-card stat-card">
          <span className="stat-label">Pending Book Balance</span>
          <span className="stat-value">${stats.pendingBalance.toLocaleString()}</span>
          <div className="stat-footer">
            Total Revenue Booked: ${stats.totalRevenue.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Main Dashboard Interactive Split View */}
      <div className="dashboard-workspace-grid">
        
        {/* Left Side: Order Pipeline & Search */}
        <div className="couture-card" style={{ padding: "32px 24px" }}>
          <div className="card-title-block" style={{ flexDirection: "column", alignItems: "flex-start", gap: "16px", marginBottom: "32px" }}>
            <div>
              <h3>Production Pipeline</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
                Monitor garment construction status across the workrooms.
              </p>
            </div>
            
            {/* Filter controls */}
            <div style={{ display: "flex", width: "100%", gap: "16px", flexWrap: "wrap", marginTop: "8px" }}>
              <div style={{ flexGrow: 1, position: "relative" }}>
                <input 
                  type="text" 
                  className="designer-input" 
                  placeholder="Search client, garment name, fabric or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{ paddingLeft: "40px" }}
                />
                <span style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)", opacity: 0.4 }}>
                  🔍
                </span>
              </div>
              
              <select 
                className="designer-input" 
                value={filterWorker}
                onChange={(e) => setFilterWorker(e.target.value)}
                style={{ width: "200px" }}
              >
                <option value="all">All Workers Workload</option>
                {workers.map(w => (
                  <option key={w.id} value={w.name}>{w.name} ({w.role})</option>
                ))}
              </select>
            </div>
          </div>

          {/* Orders Pipeline List */}
          <div className="luxury-table-container">
            {filteredOrders.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 0", color: "var(--text-muted)" }}>
                <div style={{ fontSize: "36px", marginBottom: "16px" }}>🪡</div>
                <p>No active apparel commissions match your filters.</p>
              </div>
            ) : (
              <table className="luxury-table">
                <thead>
                  <tr>
                    <th>Commission ID</th>
                    <th>Client / Garment</th>
                    <th>Deadline</th>
                    <th>Current Stage</th>
                    <th>Worker Team</th>
                    <th style={{ textAlign: "right" }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredOrders.map(order => {
                    const progressPercentage = (() => {
                      const stages = [
                        "Order Placed",
                        "Measurements Verified",
                        "Fabric & Details Sourced",
                        "Pattern & Cutting",
                        "Basting (First Stitch)",
                        "Fitting Session",
                        "Final Stitching",
                        "Finishing & QC",
                        "Completed & Ready"
                      ];
                      const idx = stages.indexOf(order.currentStage);
                      return Math.round(((idx + 1) / stages.length) * 100);
                    })();

                    return (
                      <tr key={order.id} style={{ cursor: "pointer" }} onClick={() => onViewOrder(order.id)}>
                        <td style={{ fontWeight: "700", color: "var(--accent-hover)" }}>
                          {order.id}
                        </td>
                        <td>
                          <div className="garment-spec">
                            <span className="garment-title">{order.garmentName}</span>
                            <span className="garment-subtitle">Client: {order.clientName} • {order.fabric}</span>
                          </div>
                        </td>
                        <td>
                          <span style={{ fontSize: "13px", fontWeight: "500" }}>
                            {new Date(order.deadline).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                            <span className={`badge ${order.currentStage === "Completed & Ready" ? "badge-success" : order.currentStage === "Fitting Session" ? "badge-info" : "badge-warning"}`} style={{ width: "fit-content" }}>
                              {order.currentStage}
                            </span>
                            <div style={{ width: "100px", height: "3px", backgroundColor: "var(--border-color)", borderRadius: "2px", overflow: "hidden" }}>
                              <div style={{ width: `${progressPercentage}%`, height: "100%", backgroundColor: "var(--accent)" }}></div>
                            </div>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: "flex", gap: "4px" }}>
                            {Object.entries(order.workerAssignments).slice(0, 3).map(([role, name]) => {
                              const wInfo = workers.find(w => w.name.toLowerCase() === name.toLowerCase());
                              return (
                                <span key={role} className="user-avatar" title={`${role.toUpperCase()}: ${name}`} style={{ width: "24px", height: "24px", fontSize: "11px" }}>
                                  {wInfo?.avatar || "🪡"}
                                </span>
                              );
                            })}
                          </div>
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <button className="btn-icon" onClick={(e) => {
                            e.stopPropagation();
                            onViewOrder(order.id);
                          }}>
                            →
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Right Side: Deadline Watchlist */}
        <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div>
            <h3 className="font-serif" style={{ fontSize: "22px", fontWeight: "400" }}>Atelier Urgency Board</h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
              Garments prioritized by closest delivery deadline.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "16px", overflowY: "auto", maxHeight: "550px", paddingRight: "4px" }}>
            {deadlineWatchlist.length === 0 ? (
              <div style={{ textAlign: "center", padding: "24px", color: "var(--text-muted)" }}>
                🎉 No active delivery deadlines remaining!
              </div>
            ) : (
              deadlineWatchlist.map(order => {
                let badgeClass = "badge-success";
                let badgeText = `${order.daysRemaining} days left`;
                
                if (order.daysRemaining < 0) {
                  badgeClass = "badge-error";
                  badgeText = `OVERDUE BY ${Math.abs(order.daysRemaining)} DAYS`;
                } else if (order.daysRemaining === 0) {
                  badgeClass = "badge-error";
                  badgeText = "DUE TODAY";
                } else if (order.daysRemaining <= 3) {
                  badgeClass = "badge-error";
                  badgeText = `${order.daysRemaining} days left (Urgent)`;
                } else if (order.daysRemaining <= 7) {
                  badgeClass = "badge-warning";
                  badgeText = `${order.daysRemaining} days left (Approaching)`;
                }

                return (
                  <div 
                    key={order.id} 
                    className="couture-card" 
                    onClick={() => onViewOrder(order.id)}
                    style={{ 
                      padding: "16px 20px", 
                      cursor: "pointer",
                      borderLeft: order.daysRemaining <= 3 ? "4px solid var(--error)" : order.daysRemaining <= 7 ? "4px solid var(--warning)" : "1px solid var(--border-color)",
                      backgroundColor: "var(--bg-hover)"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                      <span style={{ fontWeight: "700", fontSize: "12px", color: "var(--text-muted)" }}>
                        {order.id}
                      </span>
                      <span className={`badge ${badgeClass}`}>
                        {badgeText}
                      </span>
                    </div>

                    <h4 style={{ fontSize: "14px", fontWeight: "600" }}>{order.garmentName}</h4>
                    <p style={{ color: "var(--text-secondary)", fontSize: "12px", marginTop: "2px" }}>
                      Client: {order.clientName}
                    </p>
                    
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "12px", paddingTop: "12px", borderTop: "1px solid var(--border-color)" }}>
                      <span style={{ fontSize: "11px", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: "600" }}>
                        Stage: {order.currentStage}
                      </span>
                      <span style={{ fontSize: "11px", color: "var(--accent-hover)", fontWeight: "700" }}>
                        ${order.price}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
