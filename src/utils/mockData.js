// Premium Starting Mock Data for Bibiere Atelier Hub

export const INITIAL_WORKERS = [
  { id: "w1", name: "Marie", role: "Lead Designer", avatar: "👗" },
  { id: "w2", name: "David", role: "Pattern Cutter", avatar: "✂️" },
  { id: "w3", name: "Elena", role: "Senior Tailor", avatar: "🪡" },
  { id: "w4", name: "Fatima", role: "Embroidery & Finishing", avatar: "✨" }
];

export const INITIAL_CLIENTS = [
  {
    id: "c1",
    name: "Chloe Laurent",
    email: "chloe.laurent@paris.com",
    phone: "+33 6 1234 5678",
    notes: "VIP Client. Prefers high-contrast linings and gold metal closures.",
    measurements: {
      // Upper Body (in cm)
      bust: 88,
      waist: 64,
      hips: 92,
      shoulder: 38,
      armLength: 59,
      bicep: 26,
      wrist: 14,
      neck: 34,
      backWidth: 35,
      frontLength: 42,
      backLength: 40,
      // Lower Body (in cm)
      lowWaist: 68,
      lowerHips: 94,
      inseam: 78,
      outseam: 104,
      rise: 26,
      thigh: 52,
      knee: 36,
      calf: 33,
      ankle: 21,
      totalLength: 106,
      // Styling notes
      fitPreference: "Snug Corset Style",
      generalNotes: "Waist is sensitive. Avoid high-stiffness fabrics directly on skin."
    }
  },
  {
    id: "c2",
    name: "Alexander Sterling",
    email: "a.sterling@london.co.uk",
    phone: "+44 7911 123456",
    notes: "Savile Row enthusiast. Prefers structured shoulders and functional buttonholes.",
    measurements: {
      // Upper Body (in cm)
      bust: 104,
      waist: 88,
      hips: 102,
      shoulder: 46,
      armLength: 64,
      bicep: 34,
      wrist: 18,
      neck: 41,
      backWidth: 44,
      frontLength: 48,
      backLength: 47,
      // Lower Body (in cm)
      lowWaist: 90,
      lowerHips: 104,
      inseam: 81,
      outseam: 108,
      rise: 29,
      thigh: 60,
      knee: 42,
      calf: 38,
      ankle: 24,
      totalLength: 110,
      // Styling notes
      fitPreference: "Classic Structured Drape",
      generalNotes: "Slightly broader right shoulder (raise padding by 0.5cm)."
    }
  },
  {
    id: "c3",
    name: "Zara Al-Jamil",
    email: "zara.aj@dubai.ae",
    phone: "+971 50 123 4567",
    notes: "Prefers clean minimalist silhouettes. Uses extra-length sleeves for elegant cuffing.",
    measurements: {
      // Upper Body (in cm)
      bust: 92,
      waist: 70,
      hips: 98,
      shoulder: 40,
      armLength: 61,
      bicep: 28,
      wrist: 15,
      neck: 35,
      backWidth: 37,
      frontLength: 44,
      backLength: 42,
      // Lower Body (in cm)
      lowWaist: 74,
      lowerHips: 100,
      inseam: 79,
      outseam: 105,
      rise: 27,
      thigh: 55,
      knee: 38,
      calf: 35,
      ankle: 22,
      totalLength: 107,
      // Styling notes
      fitPreference: "Modern Relaxed Comfort",
      generalNotes: "Loves high-rise pants and fluid silk draping."
    }
  }
];

export const INITIAL_ORDERS = [
  {
    id: "ord-1001",
    clientName: "Chloe Laurent",
    clientId: "c1",
    garmentType: "Gown",
    garmentName: "Gilded Champagne Gala Gown",
    fabric: "Italian Silk Zibeline & Gold Metallic Brocade",
    lining: "Heavyweight Sand Silk Satin",
    trims: "Hand-applied Freshwater Pearls & Gold Threading",
    price: 3200,
    deposit: 1600,
    deadline: (() => {
      const d = new Date();
      d.setDate(d.getDate() + 14); // 2 weeks from now
      return d.toISOString().split("T")[0];
    })(),
    currentStage: "Basting (First Stitch)", // 5th stage
    workerAssignments: {
      cutting: "David",
      stitching: "Elena",
      embroidery: "Fatima",
      qc: "Marie"
    },
    tasks: [
      { id: "t1", text: "Create customized pattern template", completed: true },
      { id: "t2", text: "Cut gold brocade panels & silk backing", completed: true },
      { id: "t3", text: "Stitch base structure & interlining", completed: true },
      { id: "t4", text: "Prepare temporary baste for Client Fitting 1", completed: true },
      { id: "t5", text: "Conduct fitting session 1 (adjust corset panels)", completed: false },
      { id: "t6", text: "Stitch final body panels", completed: false },
      { id: "t7", text: "Apply hand-embossed pearl patterns on collar", completed: false },
      { id: "t8", text: "Press, hand-hem & attach lining", completed: false },
      { id: "t9", text: "Final quality inspect & steam", completed: false }
    ],
    fittingNotes: [
      { date: "2026-05-15", author: "Marie", content: "Client requested gold brocade panels to sit 2cm higher on the waistline for a lengthening effect. Adjusted pattern sheet accordingly." }
    ],
    notes: "To be worn at the Versailles Charity Ball. Absolute showstopper garment."
  },
  {
    id: "ord-1002",
    clientName: "Alexander Sterling",
    clientId: "c2",
    garmentType: "Suit",
    garmentName: "Midnight Velvet Tuxedo Blazer & Trousers",
    fabric: "Premium Cotton Velvet (Midnight Blue) & Silk Satin",
    lining: "Burgundy Jacquard Silk",
    trims: "Silk covered dome buttons, silk satin lapel facing",
    price: 1850,
    deposit: 1000,
    deadline: (() => {
      const d = new Date();
      d.setDate(d.getDate() + 6); // 6 days from now (Approaching!)
      return d.toISOString().split("T")[0];
    })(),
    currentStage: "Fitting Session", // 6th stage
    workerAssignments: {
      cutting: "David",
      stitching: "Elena",
      embroidery: "Fatima",
      qc: "Marie"
    },
    tasks: [
      { id: "t1", text: "Draft classic trouser and blazer block", completed: true },
      { id: "t2", text: "Carefully cut velvet (ensure grain faces down)", completed: true },
      { id: "t3", text: "Fuse canvas interfacing to chest panels", completed: true },
      { id: "t4", text: "Baste blazer body & drape trousers", completed: true },
      { id: "t5", text: "Setup fitting session (scheduled for May 19)", completed: false },
      { id: "t6", text: "Perform structural alterations post-fitting", completed: false },
      { id: "t7", text: "Stitch silk lapels and construct pockets", completed: false },
      { id: "t8", text: "Hand-sew buttonholes & attach canvas lining", completed: false },
      { id: "t9", text: "Finish trouser hems & press tuxedo ensemble", completed: false }
    ],
    fittingNotes: [],
    notes: "Requires extremely delicate pressing. Use velvet needle-board only."
  },
  {
    id: "ord-1003",
    clientName: "Zara Al-Jamil",
    clientId: "c3",
    garmentType: "Jumpsuit",
    garmentName: "Emerald Fluid Crepe Jumpsuit",
    fabric: "Triple-weave Heavy Silk Crepe in Emerald",
    lining: "Ultra-fine Emerald Habotai Silk",
    trims: "Premium invisible back zipper, gold-plated cuff buttons",
    price: 1200,
    deposit: 1200, // Fully paid
    deadline: (() => {
      const d = new Date();
      d.setDate(d.getDate() + 2); // 2 days from now (URGENT!)
      return d.toISOString().split("T")[0];
    })(),
    currentStage: "Finishing & QC", // 8th stage
    workerAssignments: {
      cutting: "David",
      stitching: "Marie",
      embroidery: "Fatima",
      qc: "Fatima"
    },
    tasks: [
      { id: "t1", text: "Draft custom wide-leg jumpsuit pattern", completed: true },
      { id: "t2", text: "Cut crepe and fine silk lining panels", completed: true },
      { id: "t3", text: "Assemble trousers section & pleated bodice", completed: true },
      { id: "t4", text: "Stitch bodice to waistband, insert invisible zipper", completed: true },
      { id: "t5", text: "Conduct fitting session 1 (adjust sleeve length)", completed: true },
      { id: "t6", text: "Execute trouser leg hems & attach silk sleeve linings", completed: true },
      { id: "t7", text: "Stitch bespoke gold cuff buttons", completed: true },
      { id: "t8", text: "Hand finish neck edge facing & clean press", completed: true },
      { id: "t9", text: "Examine seams under lightbox & steam out creases", completed: false }
    ],
    fittingNotes: [
      { date: "2026-05-12", author: "Marie", content: "Fitting successful! Zara loved the fluid drape. Shortened pants hem by 1.5cm so she can wear them perfectly with both stilettos and flats." }
    ],
    notes: "Full payment completed upfront. Client is extremely sweet and expects next-level elegance."
  }
];

export const ORDER_STAGES = [
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
