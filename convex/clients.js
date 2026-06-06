import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const list = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("atelierClients").collect();
    return rows.map((row) => row.data);
  }
});

export const upsertFromAdmin = mutation({
  args: {
    client: v.any()
  },
  handler: async (ctx, args) => {
    if (!args.client?.id) throw new Error("Client must include an id.");

    const existing = await ctx.db
      .query("atelierClients")
      .withIndex("by_id", (q) => q.eq("id", args.client.id))
      .unique();

    const payload = {
      id: args.client.id,
      data: args.client,
      updatedAt: Date.now()
    };

    if (existing) {
      await ctx.db.patch(existing._id, payload);
      return existing._id;
    }

    return await ctx.db.insert("atelierClients", payload);
  }
});
