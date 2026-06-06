import { useMemo, useState } from "react";

export default function ClientsList({ clients, orders, onUpdateClient, onAddClient, onStartOrderWithClient }) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedClientId, setSelectedClientId] = useState(clients[0]?.id || "");
  const [isEditing, setIsEditing] = useState(false);
  const [isAdding, setIsAdding] = useState(false);

  // Form states for adding/editing clients
  const [clientForm, setClientForm] = useState({
    name: "",
    email: "",
    phone: "",
    notes: "",
    measurements: {
      bust: 0, waist: 0, hips: 0, shoulder: 0, armLength: 0, bicep: 0, wrist: 0, neck: 0, backWidth: 0, frontLength: 0, backLength: 0,
      lowWaist: 0, lowerHips: 0, inseam: 0, outseam: 0, rise: 0, thigh: 0, knee: 0, calf: 0, ankle: 0, totalLength: 0,
      fitPreference: "Comfort Fit", generalNotes: ""
    }
  });

  // Highlighted measurement in visual silhouette map
  const [hoveredMeasurement, setHoveredMeasurement] = useState(null);

  // Search filter
  const filteredClients = useMemo(() => {
    return clients.filter(c => 
      c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.phone.includes(searchQuery)
    );
  }, [clients, searchQuery]);

  const activeClient = useMemo(() => {
    return clients.find(c => c.id === selectedClientId) || clients[0];
  }, [clients, selectedClientId]);

  const clientHistory = useMemo(() => {
    if (!activeClient) return [];
    return orders.filter(o => o.clientId === activeClient.id);
  }, [orders, activeClient]);

  // Edit / Add Actions
  const handleStartEdit = () => {
    if (!activeClient) return;
    setClientForm({ ...activeClient });
    setIsEditing(true);
    setIsAdding(false);
  };

  const handleStartAdd = () => {
    setClientForm({
      name: "",
      email: "",
      phone: "",
      notes: "",
      measurements: {
        bust: 86, waist: 66, hips: 94, shoulder: 38, armLength: 60, bicep: 28, wrist: 15, neck: 34, backWidth: 36, frontLength: 43, backLength: 41,
        lowWaist: 70, lowerHips: 96, inseam: 78, outseam: 104, rise: 27, thigh: 54, knee: 38, calf: 34, ankle: 22, totalLength: 105,
        fitPreference: "Regular Fit", generalNotes: ""
      }
    });
    setIsAdding(true);
    setIsEditing(false);
  };

  const handleSave = (e) => {
    e.preventDefault();
    if (isEditing) {
      onUpdateClient(clientForm);
      setIsEditing(false);
    } else if (isAdding) {
      const newId = "c-" + Date.now();
      const newClient = { ...clientForm, id: newId };
      onAddClient(newClient);
      setSelectedClientId(newId);
      setIsAdding(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setClientForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleMeasurementChange = (e) => {
    const { name, value } = e.target;
    const parsedVal = name === "fitPreference" || name === "generalNotes" ? value : parseFloat(value) || 0;
    setClientForm(prev => ({
      ...prev,
      measurements: {
        ...prev.measurements,
        [name]: parsedVal
      }
    }));
  };

  return (
    <div className="anim-slide-up" style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      
      {/* CRM Header */}
      <div className="workspace-header">
        <div className="header-title-block">
          <h2>Client Sizing Registry</h2>
          <p>Manage customer profiles, order histories, and precise tailored body dimensions.</p>
        </div>
        <div className="header-actions">
          <button className="btn-luxury btn-gold" onClick={handleStartAdd}>
            + Add Client Profile
          </button>
        </div>
      </div>

      <div className="dashboard-workspace-grid" style={{ gridTemplateColumns: "1fr 2.2fr" }}>
        
        {/* Left Side: Searchable Client List Directory */}
        <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "20px", maxHeight: "800px", padding: "24px" }}>
          <div>
            <h3 style={{ fontSize: "18px", fontWeight: "600", fontFamily: "var(--font-serif)" }}>Client Roster</h3>
            <div style={{ position: "relative", marginTop: "12px" }}>
              <input 
                type="text" 
                className="designer-input" 
                placeholder="Search clients..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ paddingLeft: "36px" }}
              />
              <span style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", opacity: 0.4 }}>🔍</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto", flexGrow: 1, paddingRight: "4px" }}>
            {filteredClients.map(client => {
              const isActive = activeClient && activeClient.id === client.id && !isAdding;
              const ordCount = orders.filter(o => o.clientId === client.id).length;
              return (
                <div 
                  key={client.id}
                  onClick={() => {
                    setSelectedClientId(client.id);
                    setIsEditing(false);
                    setIsAdding(false);
                  }}
                  className={`nav-item ${isActive ? "active" : ""}`}
                  style={{ 
                    padding: "16px", 
                    borderRadius: "var(--radius)", 
                    flexDirection: "column", 
                    alignItems: "flex-start", 
                    gap: "4px",
                    cursor: "pointer"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
                    <span style={{ fontWeight: "700", fontSize: "14px", color: "var(--text-primary)" }}>{client.name}</span>
                    <span className="badge badge-info" style={{ fontSize: "9px" }}>{ordCount} orders</span>
                  </div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{client.phone}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Side: Visual Profile & Form Workspace */}
        <div className="couture-card">
          
          {isAdding || isEditing ? (
            /* ADD / EDIT FORM WORKSPACE */
            <form onSubmit={handleSave} className="designer-form">
              <div className="card-title-block">
                <h3>{isAdding ? "Draft New Client Profile" : `Modify Profile: ${clientForm.name}`}</h3>
                <div style={{ display: "flex", gap: "12px" }}>
                  <button type="button" className="btn-luxury btn-outline btn-sm" onClick={() => { setIsAdding(false); setIsEditing(false); }}>
                    Cancel
                  </button>
                  <button type="submit" className="btn-luxury btn-gold btn-sm">
                    Save Profile
                  </button>
                </div>
              </div>

              {/* Core Information */}
              <div className="form-row-three">
                <div className="form-group">
                  <label>Full Client Name *</label>
                  <input type="text" className="designer-input" name="name" value={clientForm.name} onChange={handleInputChange} required />
                </div>
                <div className="form-group">
                  <label>Email Address</label>
                  <input type="email" className="designer-input" name="email" value={clientForm.email} onChange={handleInputChange} />
                </div>
                <div className="form-group">
                  <label>Mobile Number *</label>
                  <input type="text" className="designer-input" name="phone" value={clientForm.phone} onChange={handleInputChange} required />
                </div>
              </div>

              <div className="form-group">
                <label>Atelier / CRM Internal Notes</label>
                <textarea className="designer-input designer-textarea" name="notes" value={clientForm.notes} onChange={handleInputChange} placeholder="e.g. Likes high necklines, avoids itchy lace, VIP client..."></textarea>
              </div>

              {/* Measurement Logs Form Section */}
              <div className="measurement-form-section">
                <h4 className="font-serif" style={{ fontSize: "18px", marginBottom: "8px" }}>Anatomical Measurements</h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "12px" }}>Enter precise sizing values. All measurements are measured in centimeters (cm).</p>
                
                <h5 style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", marginTop: "24px", color: "var(--accent-hover)", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px" }}>
                  Upper Body & Torso
                </h5>
                <div className="measurement-form-grid">
                  {[
                    { label: "Bust / Chest", name: "bust" },
                    { label: "Waist", name: "waist" },
                    { label: "Hips", name: "hips" },
                    { label: "Shoulder Width", name: "shoulder" },
                    { label: "Arm Length", name: "armLength" },
                    { label: "Bicep Circ.", name: "bicep" },
                    { label: "Wrist Circ.", name: "wrist" },
                    { label: "Neck Circum.", name: "neck" },
                    { label: "Back Width", name: "backWidth" },
                    { label: "Front Bodice L.", name: "frontLength" },
                    { label: "Back Bodice L.", name: "backLength" }
                  ].map(f => (
                    <div key={f.name} className="form-group">
                      <label style={{ fontSize: "10px" }}>{f.label}</label>
                      <div className="measurement-input-wrapper">
                        <input type="number" step="0.1" className="designer-input" name={f.name} value={clientForm.measurements[f.name]} onChange={handleMeasurementChange} />
                        <span className="measurement-unit">cm</span>
                      </div>
                    </div>
                  ))}
                </div>

                <h5 style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", marginTop: "24px", color: "var(--accent-hover)", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px" }}>
                  Lower Body & Legs
                </h5>
                <div className="measurement-form-grid">
                  {[
                    { label: "Low Waist", name: "lowWaist" },
                    { label: "Low Hips", name: "lowerHips" },
                    { label: "Inseam Length", name: "inseam" },
                    { label: "Outseam Length", name: "outseam" },
                    { label: "Body Rise", name: "rise" },
                    { label: "Thigh Circum.", name: "thigh" },
                    { label: "Knee Circum.", name: "knee" },
                    { label: "Calf Circum.", name: "calf" },
                    { label: "Ankle Circum.", name: "ankle" },
                    { label: "Total Skirt L.", name: "totalLength" }
                  ].map(f => (
                    <div key={f.name} className="form-group">
                      <label style={{ fontSize: "10px" }}>{f.label}</label>
                      <div className="measurement-input-wrapper">
                        <input type="number" step="0.1" className="designer-input" name={f.name} value={clientForm.measurements[f.name]} onChange={handleMeasurementChange} />
                        <span className="measurement-unit">cm</span>
                      </div>
                    </div>
                  ))}
                </div>

                <h5 style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px", marginTop: "24px", color: "var(--accent-hover)", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px" }}>
                  Fit Preferences
                </h5>
                <div className="form-row" style={{ marginTop: "16px" }}>
                  <div className="form-group">
                    <label>Sizing / Silhouette fit preference</label>
                    <select className="designer-input" name="fitPreference" value={clientForm.measurements.fitPreference} onChange={handleMeasurementChange}>
                      <option value="Snug Corset Style">Snug Corset Style (Tightly Fitted)</option>
                      <option value="Regular Tailored Fit">Regular Tailored Fit (Fitted Bodice)</option>
                      <option value="Modern Comfort Fit">Modern Comfort Fit (Draped Silhouette)</option>
                      <option value="Oversized Slouchy">Oversized / Slouchy</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Specific Fit Notes</label>
                    <input type="text" className="designer-input" name="generalNotes" value={clientForm.measurements.generalNotes} onChange={handleMeasurementChange} placeholder="e.g. Right shoulder drop, long torso..." />
                  </div>
                </div>
              </div>
            </form>
          ) : activeClient ? (
            /* CLIENT DETAILED VIEW WITH VISUAL BLUEPRINT MAP */
            <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
              <div className="card-title-block">
                <div>
                  <h3 className="font-serif" style={{ fontSize: "32px", fontWeight: "300" }}>{activeClient.name}</h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
                    📁 Client ID: <strong style={{ color: "var(--text-primary)" }}>{activeClient.id}</strong> • ✉️ {activeClient.email || "No email logged"} • 📞 {activeClient.phone}
                  </p>
                </div>
                <div style={{ display: "flex", gap: "12px" }}>
                  <button className="btn-luxury btn-outline btn-sm" onClick={handleStartEdit}>
                    Edit Profile
                  </button>
                  <button className="btn-luxury btn-gold btn-sm" onClick={() => onStartOrderWithClient(activeClient)}>
                    ✨ New Order with Sizes
                  </button>
                </div>
              </div>

              {activeClient.notes && (
                <div style={{ backgroundColor: "var(--bg-secondary)", padding: "16px", borderRadius: "var(--radius)", borderLeft: "3px solid var(--accent)", fontSize: "13px" }}>
                  <strong>Designer Notes:</strong> {activeClient.notes}
                </div>
              )}

              {/* Anatomy Blueprint Map Interactive Area */}
              <div>
                <h4 className="font-serif" style={{ fontSize: "20px", marginBottom: "8px" }}>Interactive Size Guide</h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "13px" }}>Hover over the blueprint dots to locate measurement positions on the body silhouette.</p>
                
                <div className="blueprint-container">
                  
                  {/* Inline Vector Body Silhouette */}
                  <div className="silhouette-graphics">
                    <svg className="svg-silhouette" viewBox="0 0 100 220" xmlns="http://www.w3.org/2000/svg">
                      {/* Stylized vector body line art */}
                      <path d="M 50 15 C 53 15, 54 22, 53 25 C 56 26, 58 29, 58 33 C 58 38, 55 41, 50 41 C 45 41, 42 38, 42 33 C 42 29, 44 26, 47 25 C 46 22, 47 15, 50 15 Z" /> {/* Head */}
                      <path d="M 48 41 L 52 41 L 53 45 L 47 45 Z" /> {/* Neck */}
                      <path d="M 47 45 C 42 45, 33 49, 31 54 C 29 59, 29 65, 29 70 C 30 75, 32 82, 33 90 C 34 98, 35 105, 35 110 C 35 118, 39 126, 43 130 L 43 210 L 49 210 L 49 140 L 51 140 L 51 210 L 57 210 L 57 130 C 61 126, 65 118, 65 110 C 65 105, 66 98, 67 90 C 68 82, 70 75, 71 70 C 71 65, 71 59, 69 54 C 67 49, 58 45, 53 45" /> {/* Body */}
                      <path d="M 33 90 C 34 95, 31 105, 30 115 C 29 125, 27 135, 26 142" /> {/* Left Arm */}
                      <path d="M 67 90 C 66 95, 69 105, 70 115 C 71 125, 73 135, 74 142" /> {/* Right Arm */}
                      {/* Measuring assistance markers */}
                      {/* Shoulder line */}
                      <line x1="31" y1="52" x2="69" y2="52" stroke="var(--accent)" strokeDasharray="1,2" opacity="0.5" />
                      {/* Chest line */}
                      <line x1="33" y1="68" x2="67" y2="68" stroke="var(--accent)" strokeDasharray="1,2" opacity="0.5" />
                      {/* Waist line */}
                      <line x1="36" y1="92" x2="64" y2="92" stroke="var(--accent)" strokeDasharray="1,2" opacity="0.5" />
                      {/* Hips line */}
                      <line x1="34" y1="114" x2="66" y2="114" stroke="var(--accent)" strokeDasharray="1,2" opacity="0.5" />
                    </svg>

                    {/* Interactive dots layered on SVG body points */}
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "18%", left: "48%" }}
                      onMouseEnter={() => setHoveredMeasurement("neck")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Neck"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "24%", left: "30%" }}
                      onMouseEnter={() => setHoveredMeasurement("shoulder")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Shoulder Width"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "31%", left: "44%" }}
                      onMouseEnter={() => setHoveredMeasurement("bust")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Bust"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "42%", left: "47%" }}
                      onMouseEnter={() => setHoveredMeasurement("waist")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Waist"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "52%", left: "46%" }}
                      onMouseEnter={() => setHoveredMeasurement("hips")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Hips"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "48%", left: "26%" }}
                      onMouseEnter={() => setHoveredMeasurement("armLength")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Arm Length"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "72%", left: "43%" }}
                      onMouseEnter={() => setHoveredMeasurement("inseam")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Inseam"
                    />
                    <div 
                      className="blueprint-interactive-dot" 
                      style={{ top: "65%", left: "55%" }}
                      onMouseEnter={() => setHoveredMeasurement("outseam")}
                      onMouseLeave={() => setHoveredMeasurement(null)}
                      title="Outseam"
                    />
                  </div>

                  {/* Sizing values side dashboard */}
                  <div>
                    <h5 style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "12px", borderBottom: "1px solid var(--border-color)", paddingBottom: "4px" }}>
                      Apparel Blueprint Dimensions (cm)
                    </h5>
                    
                    <div className="blueprint-specs-grid">
                      {[
                        { label: "Bust / Chest", val: activeClient.measurements.bust, key: "bust" },
                        { label: "Waist (True)", val: activeClient.measurements.waist, key: "waist" },
                        { label: "Hips (Fullest)", val: activeClient.measurements.hips, key: "hips" },
                        { label: "Shoulder Width", val: activeClient.measurements.shoulder, key: "shoulder" },
                        { label: "Arm Length", val: activeClient.measurements.armLength, key: "armLength" },
                        { label: "Neck Circumf.", val: activeClient.measurements.neck, key: "neck" },
                        { label: "Body Rise", val: activeClient.measurements.rise, key: "rise" },
                        { label: "Leg Inseam", val: activeClient.measurements.inseam, key: "inseam" },
                        { label: "Leg Outseam", val: activeClient.measurements.outseam, key: "outseam" },
                        { label: "Lower Waist", val: activeClient.measurements.lowWaist, key: "lowWaist" },
                        { label: "Lower Hips", val: activeClient.measurements.lowerHips, key: "lowerHips" },
                        { label: "Bicep Circ.", val: activeClient.measurements.bicep, key: "bicep" },
                        { label: "Wrist Circ.", val: activeClient.measurements.wrist, key: "wrist" },
                        { label: "Skirt/Leg L.", val: activeClient.measurements.totalLength, key: "totalLength" }
                      ].map(item => {
                        const isHovered = hoveredMeasurement === item.key;
                        return (
                          <div 
                            key={item.key} 
                            className="blueprint-field"
                            style={{ 
                              borderColor: isHovered ? "var(--accent)" : "var(--border-color)",
                              backgroundColor: isHovered ? "var(--accent-light)" : "var(--bg-card)",
                              transform: isHovered ? "scale(1.03)" : "none",
                              transition: "all 0.25s ease"
                            }}
                          >
                            <span className="blueprint-field-label">{item.label}</span>
                            <span className="blueprint-field-val">{item.val} cm</span>
                          </div>
                        );
                      })}
                    </div>

                    <div style={{ marginTop: "16px", paddingTop: "12px", borderTop: "1px solid var(--border-color)", fontSize: "12px" }}>
                      <div>Fit Styling Preference: <strong style={{ color: "var(--text-primary)" }}>{activeClient.measurements.fitPreference}</strong></div>
                      {activeClient.measurements.generalNotes && (
                        <div style={{ color: "var(--accent-hover)", fontStyle: "italic", marginTop: "2px" }}>* {activeClient.measurements.generalNotes}</div>
                      )}
                    </div>
                  </div>

                </div>
              </div>

              {/* Client Order History */}
              <div>
                <h4 className="font-serif" style={{ fontSize: "20px", marginBottom: "12px" }}>Commission History</h4>
                
                {clientHistory.length === 0 ? (
                  <p style={{ color: "var(--text-muted)", fontSize: "13px", padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)", textAlign: "center" }}>
                    No commissions logged yet for this client.
                  </p>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                    {clientHistory.map(ord => (
                      <div 
                        key={ord.id}
                        className="couture-card"
                        style={{ padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: "var(--bg-hover)", cursor: "pointer" }}
                        onClick={() => onStartOrderWithClient(activeClient)} /* Let them clone/interact */
                      >
                        <div>
                          <strong style={{ color: "var(--accent-hover)" }}>{ord.id}</strong>
                          <span style={{ marginLeft: "12px", fontWeight: "600" }}>{ord.garmentName}</span>
                          <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "4px" }}>
                            Fabric: {ord.fabric} • Target Date: {ord.deadline}
                          </div>
                        </div>
                        
                        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                          <span className={`badge ${ord.currentStage === "Completed & Ready" ? "badge-success" : ord.currentStage === "Fitting Session" ? "badge-info" : "badge-warning"}`}>
                            {ord.currentStage}
                          </span>
                          <strong style={{ fontSize: "14px" }}>${ord.price}</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "64px 0", color: "var(--text-muted)" }}>
              <h3>No Client Selected</h3>
              <p>Please select a customer profile from the left index directory or register a new client.</p>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}
