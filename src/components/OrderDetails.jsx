import { useMemo, useState } from "react";
import { ORDER_STAGES } from "../utils/mockData";

export default function OrderDetails({ order, clients, workers, onUpdateOrder, onClose }) {
  const [newTaskText, setNewTaskText] = useState("");
  const [newFittingNote, setNewFittingNote] = useState("");
  const [selectedNoteAuthor, setSelectedNoteAuthor] = useState("Marie");
  const [whatsappTemplate, setWhatsappTemplate] = useState("fitting");

  const client = useMemo(() => {
    return clients.find(c => c.id === order.clientId);
  }, [clients, order]);

  // Compute remaining days
  const daysRemaining = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const due = new Date(order.deadline);
    due.setHours(0, 0, 0, 0);
    const diffTime = due.getTime() - today.getTime();
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }, [order]);

  // Stage index calculation for timeline progress bar
  const currentStageIndex = ORDER_STAGES.indexOf(order.currentStage);
  const timelineProgressWidth = `${(currentStageIndex / (ORDER_STAGES.length - 1)) * 100}%`;

  // Toggle stage selection
  const handleStageSelect = (stage) => {
    onUpdateOrder({
      ...order,
      currentStage: stage
    });
  };

  // Toggle checklist tasks
  const handleToggleTask = (taskId) => {
    const updatedTasks = order.tasks.map(t => 
      t.id === taskId ? { ...t, completed: !t.completed } : t
    );
    onUpdateOrder({ ...order, tasks: updatedTasks });
  };

  // Add custom new checklist item
  const handleAddTask = (e) => {
    e.preventDefault();
    if (!newTaskText.trim()) return;
    const newTask = {
      id: "cust-" + Date.now(),
      text: newTaskText.trim(),
      completed: false
    };
    onUpdateOrder({
      ...order,
      tasks: [...order.tasks, newTask]
    });
    setNewTaskText("");
  };

  // Append fitting session note
  const handleAddFittingNote = (e) => {
    e.preventDefault();
    if (!newFittingNote.trim()) return;
    const newNote = {
      date: new Date().toISOString().split("T")[0],
      author: selectedNoteAuthor,
      content: newFittingNote.trim()
    };
    onUpdateOrder({
      ...order,
      fittingNotes: [newNote, ...order.fittingNotes]
    });
    setNewFittingNote("");
  };

  // Auto-generate luxury WhatsApp Message Templates
  const customMessage = useMemo(() => {
    if (!client) return "";
    const greetingName = client.name.split(" ")[0];
    
    if (whatsappTemplate === "fitting") {
      return `Hi ${greetingName}! ✨ This is ${selectedNoteAuthor} from Bibiere Atelier. We are delighted to share that your custom ${order.garmentName} is progressing beautifully and is now ready for your fitting session! 🪡 Let us know which day this week suits you best to visit us. With love, Bibiere.`;
    }
    
    if (whatsappTemplate === "sourced") {
      return `Hello ${greetingName}, we hope you are well. ✨ Just a quick note to let you know that we have sourced the exquisite ${order.fabric} for your ${order.garmentName}! Next, we begin drafting your custom pattern block. Warmly, the Bibiere Team.`;
    }
    
    if (whatsappTemplate === "completed") {
      return `Dearest ${greetingName}! ✨ Exciting news—your beautiful ${order.garmentName} has passed its final quality inspections and is fully completed and ready for pick up! We can't wait for you to wear it. Let us know when you would like to drop by. Best regards, Bibiere.`;
    }

    return "";
  }, [client, order, whatsappTemplate, selectedNoteAuthor]);

  const handleCopyMessage = () => {
    navigator.clipboard.writeText(customMessage);
    alert("Concierge message successfully copied to clipboard! You can now paste it into WhatsApp or SMS.");
  };

  return (
    <div className="anim-slide-up" style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      
      {/* Detail Header & Action Panel */}
      <div className="workspace-header">
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <button className="btn-icon" onClick={onClose} title="Back to Dashboard">
            ←
          </button>
          <div>
            <span style={{ fontSize: "11px", letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--accent-hover)", fontWeight: "600" }}>
              Garment Workspace
            </span>
            <h2 className="font-serif" style={{ fontSize: "32px", marginTop: "4px" }}>
              {order.garmentName} ({order.id})
            </h2>
          </div>
        </div>
        <div className="header-actions">
          <button className="btn-luxury btn-outline" onClick={onClose}>
            Back to Dashboard
          </button>
        </div>
      </div>

      {/* Interactive Visual Stepper Timeline */}
      <div className="couture-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 className="font-serif" style={{ fontSize: "20px" }}>Tailoring Stage Progression</h3>
          <span className="badge badge-info">{order.currentStage}</span>
        </div>
        
        <div className="timeline-stages">
          <div className="timeline-progress-bar" style={{ width: timelineProgressWidth }} />
          {ORDER_STAGES.map((stage, idx) => {
            const isCompleted = idx < currentStageIndex;
            const isActive = idx === currentStageIndex;
            return (
              <div 
                key={stage} 
                className={`stage-node ${isCompleted ? "completed" : ""} ${isActive ? "active" : ""}`}
                onClick={() => handleStageSelect(stage)}
              >
                <div className="stage-dot" />
                <span className="stage-label" style={{ transform: "translateX(-50%)" }}>
                  {stage}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Order Details & Sizing Grid Splits */}
      <div className="dashboard-workspace-grid">
        
        {/* Left Side: Specs, Checklist, Fitting Log */}
        <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
          
          {/* Spec Sheet & Financial Balances */}
          <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            <div>
              <h3 className="font-serif" style={{ fontSize: "22px" }}>Garment Specifications</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "2px" }}>Design aesthetics, fabric types and details.</p>
            </div>

            <div className="form-row-three">
              <div style={{ padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)" }}>
                <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "600" }}>Garment Silh.</span>
                <div style={{ fontWeight: "700", fontSize: "16px", marginTop: "4px" }}>{order.garmentType}</div>
              </div>
              <div style={{ padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)" }}>
                <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "600" }}>Principal Fabric</span>
                <div style={{ fontWeight: "700", fontSize: "15px", marginTop: "4px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={order.fabric}>{order.fabric}</div>
              </div>
              <div style={{ padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)" }}>
                <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "600" }}>Inner Lining</span>
                <div style={{ fontWeight: "700", fontSize: "15px", marginTop: "4px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={order.lining}>{order.lining || "None"}</div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "24px", borderTop: "1px solid var(--border-color)", paddingTop: "24px" }}>
              <div>
                <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "600" }}>Couture Trims & Embellishments</span>
                <p style={{ fontSize: "13px", fontWeight: "500", marginTop: "6px" }}>{order.trims || "No embellishments specified."}</p>
              </div>
              <div>
                <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: "600" }}>Atelier General Instructions</span>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)", fontStyle: "italic", marginTop: "6px" }}>{order.notes || "None."}</p>
              </div>
            </div>

            {/* Financial ledger */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", borderTop: "1px dashed var(--border-accent)", paddingTop: "24px", marginTop: "12px" }}>
              <div>
                <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>Total Commission Price</span>
                <div style={{ fontSize: "20px", fontWeight: "700", marginTop: "2px" }}>${order.price.toLocaleString()}</div>
              </div>
              <div>
                <span style={{ fontSize: "11px", color: "var(--success)" }}>Deposit Paid</span>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--success)", marginTop: "2px" }}>${order.deposit.toLocaleString()}</div>
              </div>
              <div>
                <span style={{ fontSize: "11px", color: "var(--error)" }}>Remaining Balance</span>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--error)", marginTop: "2px" }}>${(order.price - order.deposit).toLocaleString()}</div>
              </div>
            </div>
          </div>

          {/* Sub-tasks Workroom Checklist */}
          <div className="couture-card">
            <h3 className="font-serif" style={{ fontSize: "22px", marginBottom: "4px" }}>Assembly Steps Checklist</h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginBottom: "20px" }}>Monitor granular tasks completed for this piece.</p>

            <form onSubmit={handleAddTask} style={{ display: "flex", gap: "12px", marginBottom: "20px" }}>
              <input 
                type="text" 
                className="designer-input" 
                placeholder="e.g. Hand hem lining, sew buttonholes..."
                value={newTaskText}
                onChange={(e) => setNewTaskText(e.target.value)}
              />
              <button type="submit" className="btn-luxury btn-outline btn-sm">
                Add Step
              </button>
            </form>

            <div style={{ display: "flex", flexDirection: "column" }}>
              {order.tasks.map(t => (
                <div 
                  key={t.id} 
                  className={`checklist-item ${t.completed ? "completed" : ""}`}
                  onClick={() => handleToggleTask(t.id)}
                >
                  <div className={`checklist-checkbox ${t.completed ? "checked" : ""}`}></div>
                  <span className={`checklist-text ${t.completed ? "completed" : ""}`}>{t.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Fitting Sessions Revisions Ledger */}
          <div className="couture-card">
            <h3 className="font-serif" style={{ fontSize: "22px", marginBottom: "4px" }}>Fitting Revisions Log</h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginBottom: "20px" }}>Log client fit feedback and corresponding panel adjustments.</p>

            <form onSubmit={handleAddFittingNote} style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: "24px" }}>
              <div style={{ display: "flex", gap: "12px" }}>
                <select 
                  className="designer-input"
                  value={selectedNoteAuthor}
                  onChange={(e) => setSelectedNoteAuthor(e.target.value)}
                  style={{ width: "160px" }}
                >
                  {workers.map(w => (
                    <option key={w.id} value={w.name}>{w.name} ({w.role.split(" ")[0]})</option>
                  ))}
                </select>
                <input 
                  type="text" 
                  className="designer-input" 
                  placeholder="Record client fitting changes (e.g. shorten bodice by 1cm)..."
                  value={newFittingNote}
                  onChange={(e) => setNewFittingNote(e.target.value)}
                />
              </div>
              <button type="submit" className="btn-luxury btn-outline btn-sm" style={{ alignSelf: "flex-end" }}>
                Save Fit Log Note
              </button>
            </form>

            <div style={{ display: "flex", flexDirection: "column", gap: "16px", maxHeight: "300px", overflowY: "auto" }}>
              {order.fittingNotes.length === 0 ? (
                <p style={{ color: "var(--text-muted)", fontSize: "13px", textAlign: "center", padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)" }}>
                  No fitting revisions logged yet.
                </p>
              ) : (
                order.fittingNotes.map((note, index) => (
                  <div key={index} style={{ padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)", borderLeft: "2px solid var(--accent-hover)", fontSize: "13px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)", fontWeight: "600", marginBottom: "6px" }}>
                      <span>📝 Logged by: {note.author}</span>
                      <span>📅 {note.date}</span>
                    </div>
                    <p style={{ color: "var(--text-primary)" }}>{note.content}</p>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>

        {/* Right Side: Size profiles, Worker Assignees & Concierge Messages */}
        <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
          
          {/* Target Deadline Details */}
          <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent)" }}>
            <span style={{ fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-secondary)", fontWeight: "600" }}>Delivery Deadline</span>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span className="font-serif" style={{ fontSize: "28px" }}>
                {new Date(order.deadline).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}
              </span>
              <span className={`badge ${daysRemaining <= 3 ? "badge-error" : daysRemaining <= 7 ? "badge-warning" : "badge-success"}`}>
                {daysRemaining < 0 ? `Overdue ${Math.abs(daysRemaining)}d` : daysRemaining === 0 ? "DUE TODAY" : `${daysRemaining} days left`}
              </span>
            </div>
          </div>

          {/* Worker Assignees Grid */}
          <div className="couture-card">
            <h3 className="font-serif" style={{ fontSize: "20px", marginBottom: "16px" }}>Atelier Staff Assignments</h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {[
                { label: "Pattern Draft & Cutting", role: "cutting" },
                { label: "Construction Stitching", role: "stitching" },
                { label: "Embroidery & Finishing", role: "embroidery" },
                { label: "Master Quality Check", role: "qc" }
              ].map(as => {
                const name = order.workerAssignments[as.role] || "Unassigned";
                const wInfo = workers.find(w => w.name.toLowerCase() === name.toLowerCase());
                return (
                  <div key={as.role} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", border: "1px solid var(--border-color)", borderRadius: "var(--radius)" }}>
                    <div style={{ display: "flex", flexDirection: "column" }}>
                      <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>{as.label}</span>
                      <strong style={{ fontSize: "13px", marginTop: "2px" }}>{name}</strong>
                    </div>
                    <span className="user-avatar" style={{ fontSize: "14px" }}>{wInfo?.avatar || "🪡"}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Custom Sizing Cards display (Silhouette guide) */}
          <div className="couture-card" style={{ padding: "24px" }}>
            <h3 className="font-serif" style={{ fontSize: "20px", marginBottom: "4px" }}>Client Measurement Dimensions</h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "12px", marginBottom: "16px" }}>Core customer anatomical metrics logged for this client profile.</p>

            {client ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                <div style={{ padding: "12px", backgroundColor: "var(--accent-light)", border: "1px solid var(--border-accent)", borderRadius: "var(--radius)", fontSize: "13px" }}>
                  <div>Garment Sizing Profile: <strong>{client.measurements.fitPreference}</strong></div>
                  {client.measurements.generalNotes && <div style={{ color: "var(--accent-hover)", fontStyle: "italic", marginTop: "2px" }}>* {client.measurements.generalNotes}</div>}
                </div>

                <div className="blueprint-specs-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)", gap: "8px" }}>
                  {[
                    { label: "Bust / Chest", val: client.measurements.bust },
                    { label: "Waist (True)", val: client.measurements.waist },
                    { label: "Hips (Fullest)", val: client.measurements.hips },
                    { label: "Shoulder Width", val: client.measurements.shoulder },
                    { label: "Arm Length", val: client.measurements.armLength },
                    { label: "Neck Circumf.", val: client.measurements.neck },
                    { label: "Leg Inseam", val: client.measurements.inseam },
                    { label: "Skirt/Total L.", val: client.measurements.totalLength }
                  ].map((sz, i) => (
                    <div key={i} className="blueprint-field" style={{ padding: "6px 10px", fontSize: "12px" }}>
                      <span className="blueprint-field-label">{sz.label}</span>
                      <strong style={{ color: "var(--accent-hover)" }}>{sz.val} cm</strong>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "12px" }}>No measurements bound to this order.</p>
            )}
          </div>

          {/* Concierge Message Generator */}
          <div className="couture-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div>
              <h3 className="font-serif" style={{ fontSize: "20px", marginBottom: "4px" }}>Client Atelier Messenger</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "12px" }}>Auto-draft highly elegant WhatsApp messages for your client.</p>
            </div>

            <div style={{ display: "flex", gap: "8px" }}>
              {[
                { label: "Fitting Session", val: "fitting" },
                { label: "Fabric Sourced", val: "sourced" },
                { label: "Finished / QC", val: "completed" }
              ].map(opt => (
                <button 
                  key={opt.val}
                  type="button"
                  className={`btn-luxury btn-sm ${whatsappTemplate === opt.val ? "btn-gold" : "btn-outline"}`}
                  style={{ flexGrow: 1, padding: "8px", fontSize: "11px", letterSpacing: "0.5px" }}
                  onClick={() => setWhatsappTemplate(opt.val)}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div style={{ padding: "16px", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius)", border: "1px dashed var(--border-color)" }}>
              <p style={{ fontStyle: "italic", fontSize: "12.5px", color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {customMessage}
              </p>
            </div>

            <button className="btn-luxury btn-sm btn-gold" onClick={handleCopyMessage} style={{ alignSelf: "flex-end" }}>
              Copy Atelier Message
            </button>
          </div>

        </div>

      </div>
    </div>
  );
}
