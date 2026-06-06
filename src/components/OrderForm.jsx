/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from "react";
import { ORDER_STAGES } from "../utils/mockData";

export default function OrderForm({ order, clients, workers, onSave, onCancel }) {
  const [isNewClient, setIsNewClient] = useState(false);
  const [selectedClientId, setSelectedClientId] = useState("");
  
  const [formData, setFormData] = useState({
    clientName: "",
    email: "",
    phone: "",
    garmentType: "Gown",
    garmentName: "",
    fabric: "",
    lining: "",
    trims: "",
    price: 1200,
    deposit: 600,
    deadline: "",
    currentStage: "Order Placed",
    workerAssignments: {
      cutting: "David",
      stitching: "Elena",
      embroidery: "Fatima",
      qc: "Marie"
    },
    notes: ""
  });

  // Pre-fill fields if we are editing an order
  useEffect(() => {
    if (order) {
      setFormData({
        ...order,
        email: clients.find(c => c.id === order.clientId)?.email || "",
        phone: clients.find(c => c.id === order.clientId)?.phone || ""
      });
      setSelectedClientId(order.clientId);
      setIsNewClient(false);
    } else {
      // Set default deadline 2 weeks out
      const defaultDate = new Date();
      defaultDate.setDate(defaultDate.getDate() + 14);
      setFormData(prev => ({
        ...prev,
        deadline: defaultDate.toISOString().split("T")[0]
      }));
      if (clients.length > 0) {
        setSelectedClientId(clients[0].id);
        setIsNewClient(false);
      } else {
        setIsNewClient(true);
      }
    }
  }, [order, clients]);

  // Handle client selection sync
  useEffect(() => {
    if (!isNewClient && selectedClientId && !order) {
      const activeC = clients.find(c => c.id === selectedClientId);
      if (activeC) {
        setFormData(prev => ({
          ...prev,
          clientName: activeC.name,
          email: activeC.email,
          phone: activeC.phone
        }));
      }
    }
  }, [selectedClientId, isNewClient, clients, order]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const parsedValue = name === "price" || name === "deposit" ? parseFloat(value) || 0 : value;
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };

  const handleWorkerChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      workerAssignments: {
        ...prev.workerAssignments,
        [name]: value
      }
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Auto-generate starting tasks list if creating a new order
    let finalTasks = order?.tasks || [];
    if (!order) {
      finalTasks = autoGenerateTasks(formData.garmentType, formData.garmentName);
    }

    const orderPayload = {
      ...formData,
      clientId: isNewClient ? null : selectedClientId, // Will be bound in App.jsx if new client
      tasks: finalTasks,
      fittingNotes: order?.fittingNotes || []
    };

    onSave(orderPayload, isNewClient);
  };

  // Auto-generate customized master tasks based on garment silhouette
  const autoGenerateTasks = (type, name) => {
    const common = [
      { id: "t1", text: `Draft customized block pattern for ${name}`, completed: false },
      { id: "t2", text: "Select, steam and prepare fabric rolls", completed: false },
      { id: "t3", text: "Place pattern templates, trace & execute cutting", completed: false }
    ];

    if (type === "Gown" || type === "Dress") {
      return [
        ...common,
        { id: "t4", text: "Stitch bodice lining & prepare temporary bones", completed: false },
        { id: "t5", text: "Assemble temporary baste for client Fitting 1", completed: false },
        { id: "t6", text: "Conduct fitting session 1 (adjust hemline & shoulder drape)", completed: false },
        { id: "t7", text: "Stitch final luxury lining, attach closures & zippers", completed: false },
        { id: "t8", text: "Hand-hem dress edges, hand-stitch beads/trims", completed: false },
        { id: "t9", text: "Final structural steam, QC inspect & hang", completed: false }
      ];
    }
    
    if (type === "Suit" || type === "Blazer") {
      return [
        ...common,
        { id: "t4", text: "Fuse interlinings to chest canvases, stitch pockets", completed: false },
        { id: "t5", text: "Assemble tuxedo jacket body, pad shoulders & lapels", completed: false },
        { id: "t6", text: "Baste pants rise and legs, organize client Fitting 1", completed: false },
        { id: "t7", text: "Execute post-fitting changes on blazer waist & shoulder pads", completed: false },
        { id: "t8", text: "Stitch arm sleeve silk linings, construct functional buttonholes", completed: false },
        { id: "t9", text: "Final tailor pressing, lapel steam, lint rollers", completed: false }
      ];
    }

    // Default template for jumpsuit/corset/others
    return [
      ...common,
      { id: "t4", text: "Assemble panels, prepare invisible zippers & seams", completed: false },
      { id: "t5", text: "Assemble temporary baste for client Fitting 1", completed: false },
      { id: "t6", text: "Conduct fitting session 1 (check sleeve length & waist wrap)", completed: false },
      { id: "t7", text: "Execute post-fitting adjustments & sew final stitch paths", completed: false },
      { id: "t8", text: "Hand finish necklines, hem cuffs, press creases", completed: false },
      { id: "t9", text: "Examine stitch lines under light boards & package", completed: false }
    ];
  };

  return (
    <div className="anim-slide-up">
      <div className="couture-card" style={{ maxWidth: "1000px", margin: "0 auto" }}>
        
        {/* Form Title & Actions */}
        <div className="card-title-block">
          <div>
            <h3 className="font-serif" style={{ fontSize: "28px" }}>
              {order ? `Modify Commission: ${order.id}` : "Log New Couture Commission"}
            </h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
              Fill in client parameters, silhouette styles, deadlines, and workers.
            </p>
          </div>
          <div style={{ display: "flex", gap: "12px" }}>
            <button type="button" className="btn-luxury btn-outline btn-sm" onClick={onCancel}>
              Cancel
            </button>
            <button type="submit" form="order-atelier-form" className="btn-luxury btn-gold btn-sm">
              {order ? "Save Updates" : "Place Commission"}
            </button>
          </div>
        </div>

        <form id="order-atelier-form" onSubmit={handleSubmit} className="designer-form" style={{ marginTop: "24px" }}>
          
          {/* Client Relationship (CRM Binding) */}
          <div style={{ borderBottom: "1px solid var(--border-color)", paddingBottom: "24px" }}>
            <h4 className="font-serif" style={{ fontSize: "18px", marginBottom: "16px", color: "var(--accent-hover)" }}>Client Assignment</h4>
            
            {!order && (
              <div style={{ display: "flex", gap: "24px", marginBottom: "20px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", cursor: "pointer", fontWeight: "600" }}>
                  <input 
                    type="radio" 
                    name="clientType" 
                    checked={!isNewClient} 
                    onChange={() => setIsNewClient(false)}
                    style={{ accentColor: "var(--accent)" }}
                  />
                  Repeat Client (Roster)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", cursor: "pointer", fontWeight: "600" }}>
                  <input 
                    type="radio" 
                    name="clientType" 
                    checked={isNewClient} 
                    onChange={() => setIsNewClient(true)}
                    style={{ accentColor: "var(--accent)" }}
                  />
                  New Client Registration
                </label>
              </div>
            )}

            {!isNewClient && !order ? (
              <div className="form-group" style={{ maxWidth: "400px" }}>
                <label>Select Repeat Customer</label>
                <select 
                  className="designer-input" 
                  value={selectedClientId} 
                  onChange={(e) => setSelectedClientId(e.target.value)}
                >
                  {clients.map(c => (
                    <option key={c.id} value={c.id}>{c.name} ({c.phone})</option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="form-row-three">
                <div className="form-group">
                  <label>Client Name *</label>
                  <input 
                    type="text" 
                    className="designer-input" 
                    name="clientName" 
                    value={formData.clientName} 
                    onChange={handleInputChange} 
                    required 
                    disabled={!!order}
                  />
                </div>
                <div className="form-group">
                  <label>Email Address</label>
                  <input 
                    type="email" 
                    className="designer-input" 
                    name="email" 
                    value={formData.email} 
                    onChange={handleInputChange} 
                    disabled={!!order}
                  />
                </div>
                <div className="form-group">
                  <label>Telephone Number *</label>
                  <input 
                    type="text" 
                    className="designer-input" 
                    name="phone" 
                    value={formData.phone} 
                    onChange={handleInputChange} 
                    required 
                    disabled={!!order}
                  />
                </div>
              </div>
            )}
            {isNewClient && !order && (
              <div style={{ marginTop: "12px", fontSize: "12px", color: "var(--accent-hover)", fontStyle: "italic" }}>
                * This will automatically create a new customer record inside the Client CRM registry using standard base measurements. You can later customize sizes.
              </div>
            )}
          </div>

          {/* Garment Details */}
          <div style={{ borderBottom: "1px solid var(--border-color)", paddingBottom: "24px" }}>
            <h4 className="font-serif" style={{ fontSize: "18px", marginBottom: "16px", color: "var(--accent-hover)" }}>Garment Spec & Details</h4>
            
            <div className="form-row">
              <div className="form-group">
                <label>Commission Title / Name *</label>
                <input 
                  type="text" 
                  className="designer-input" 
                  name="garmentName" 
                  value={formData.garmentName} 
                  onChange={handleInputChange} 
                  placeholder="e.g. Royal Gold Silk Wedding Dress"
                  required 
                />
              </div>
              <div className="form-group">
                <label>Silhouette Category</label>
                <select 
                  className="designer-input" 
                  name="garmentType" 
                  value={formData.garmentType} 
                  onChange={handleInputChange}
                >
                  <option value="Gown">Gown (Evening Gala / Bridal)</option>
                  <option value="Suit">Suit (Bespoke Jackets & Pants)</option>
                  <option value="Jumpsuit">Jumpsuit (Wide-leg / Structured)</option>
                  <option value="Dress">Dress (Cocktail / Day Wear)</option>
                  <option value="Corset">Corset (Bustiers / Tops)</option>
                  <option value="Skirt">Skirt (A-line / Pencil)</option>
                </select>
              </div>
            </div>

            <div className="form-row-three" style={{ marginTop: "16px" }}>
              <div className="form-group">
                <label>Principal Fabric *</label>
                <input 
                  type="text" 
                  className="designer-input" 
                  name="fabric" 
                  value={formData.fabric} 
                  onChange={handleInputChange} 
                  placeholder="e.g. French Chantilly Lace & Silk Crepe"
                  required 
                />
              </div>
              <div className="form-group">
                <label>Inner Lining Fabric</label>
                <input 
                  type="text" 
                  className="designer-input" 
                  name="lining" 
                  value={formData.lining} 
                  onChange={handleInputChange} 
                  placeholder="e.g. Premium Mulberry Silk Satin"
                />
              </div>
              <div className="form-group">
                <label>Trims & Embellishments</label>
                <input 
                  type="text" 
                  className="designer-input" 
                  name="trims" 
                  value={formData.trims} 
                  onChange={handleInputChange} 
                  placeholder="e.g. Swarovski crystals, hand embroidery..."
                />
              </div>
            </div>
          </div>

          {/* Financial & Timelines */}
          <div style={{ borderBottom: "1px solid var(--border-color)", paddingBottom: "24px" }}>
            <h4 className="font-serif" style={{ fontSize: "18px", marginBottom: "16px", color: "var(--accent-hover)" }}>Timeline & Ledger</h4>
            
            <div className="form-row-three">
              <div className="form-group">
                <label>Delivery Target Deadline *</label>
                <input 
                  type="date" 
                  className="designer-input" 
                  name="deadline" 
                  value={formData.deadline} 
                  onChange={handleInputChange} 
                  required 
                />
              </div>
              <div className="form-group">
                <label>Total Price ($) *</label>
                <div className="measurement-input-wrapper">
                  <input 
                    type="number" 
                    className="designer-input" 
                    name="price" 
                    value={formData.price} 
                    onChange={handleInputChange} 
                    required 
                  />
                  <span className="measurement-unit">USD</span>
                </div>
              </div>
              <div className="form-group">
                <label>Initial Deposit Paid ($) *</label>
                <div className="measurement-input-wrapper">
                  <input 
                    type="number" 
                    className="designer-input" 
                    name="deposit" 
                    value={formData.deposit} 
                    onChange={handleInputChange} 
                    required 
                  />
                  <span className="measurement-unit">USD</span>
                </div>
              </div>
            </div>
            
            {order && (
              <div className="form-group" style={{ maxWidth: "300px", marginTop: "16px" }}>
                <label>Current Shop Stage</label>
                <select 
                  className="designer-input" 
                  name="currentStage" 
                  value={formData.currentStage} 
                  onChange={handleInputChange}
                >
                  {ORDER_STAGES.map(st => (
                    <option key={st} value={st}>{st}</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Worker Assignments */}
          <div>
            <h4 className="font-serif" style={{ fontSize: "18px", marginBottom: "16px", color: "var(--accent-hover)" }}>Staff Assignments</h4>
            
            <div className="form-row">
              <div className="form-group">
                <label>Pattern Drafting & Fabric Cutting</label>
                <select 
                  className="designer-input" 
                  name="cutting" 
                  value={formData.workerAssignments.cutting} 
                  onChange={handleWorkerChange}
                >
                  {workers.map(w => (
                    <option key={w.id} value={w.name}>{w.name} ({w.role})</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Main Body Panel Stitching</label>
                <select 
                  className="designer-input" 
                  name="stitching" 
                  value={formData.workerAssignments.stitching} 
                  onChange={handleWorkerChange}
                >
                  {workers.map(w => (
                    <option key={w.id} value={w.name}>{w.name} ({w.role})</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row" style={{ marginTop: "16px" }}>
              <div className="form-group">
                <label>Embellishments & Finishes</label>
                <select 
                  className="designer-input" 
                  name="embroidery" 
                  value={formData.workerAssignments.embroidery} 
                  onChange={handleWorkerChange}
                >
                  {workers.map(w => (
                    <option key={w.id} value={w.name}>{w.name} ({w.role})</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Senior Quality Check inspector</label>
                <select 
                  className="designer-input" 
                  name="qc" 
                  value={formData.workerAssignments.qc} 
                  onChange={handleWorkerChange}
                >
                  {workers.map(w => (
                    <option key={w.id} value={w.name}>{w.name} ({w.role})</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-group" style={{ marginTop: "24px" }}>
              <label>Special Pattern Design Instructions</label>
              <textarea 
                className="designer-input designer-textarea" 
                name="notes" 
                value={formData.notes} 
                onChange={handleInputChange} 
                placeholder="e.g. Tailor with extra-long sleeves, adjust buttons for left-hand use..."
              />
            </div>
          </div>

        </form>
      </div>
    </div>
  );
}
