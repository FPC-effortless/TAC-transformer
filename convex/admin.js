import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

async function upsertById(ctx, tableName, record) {
  const existing = await ctx.db
    .query(tableName)
    .withIndex("by_id", (q) => q.eq("id", record.id))
    .unique();

  const payload = {
    id: record.id,
    data: record,
    updatedAt: Date.now()
  };

  if (existing) {
    await ctx.db.patch(existing._id, payload);
    return existing._id;
  }

  return await ctx.db.insert(tableName, payload);
}

export const getAtelierData = query({
  args: {},
  handler: async (ctx) => {
    const [orders, clients] = await Promise.all([
      ctx.db.query("atelierOrders").collect(),
      ctx.db.query("atelierClients").collect()
    ]);

    return {
      orders: orders.map((row) => row.data),
      clients: clients.map((row) => row.data)
    };
  }
});

export const saveAtelierData = mutation({
  args: {
    orders: v.array(v.any()),
    clients: v.array(v.any())
  },
  handler: async (ctx, args) => {
    for (const client of args.clients) {
      if (client?.id) await upsertById(ctx, "atelierClients", client);
    }

    for (const order of args.orders) {
      if (order?.id) await upsertById(ctx, "atelierOrders", order);
    }

    return { ok: true };
  }
});
