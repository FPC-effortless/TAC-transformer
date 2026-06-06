import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const list = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("atelierOrders").collect();
    return rows.map((row) => row.data);
  }
});

export const upsertFromAdmin = mutation({
  args: {
    order: v.any()
  },
  handler: async (ctx, args) => {
    if (!args.order?.id) throw new Error("Order must include an id.");

    const existing = await ctx.db
      .query("atelierOrders")
      .withIndex("by_id", (q) => q.eq("id", args.order.id))
      .unique();

    const payload = {
      id: args.order.id,
      data: args.order,
      updatedAt: Date.now()
    };

    if (existing) {
      await ctx.db.patch(existing._id, payload);
      return existing._id;
    }

    return await ctx.db.insert("atelierOrders", payload);
  }
});
