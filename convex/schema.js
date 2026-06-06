import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  atelierOrders: defineTable({
    id: v.string(),
    data: v.any(),
    updatedAt: v.number()
  }).index("by_id", ["id"]),
  atelierClients: defineTable({
    id: v.string(),
    data: v.any(),
    updatedAt: v.number()
  }).index("by_id", ["id"])
});
