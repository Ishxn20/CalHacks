import { internalMutation, query } from "./_generated/server";
import { v } from "convex/values";

const item = v.object({ uid: v.string(), instrument: v.string(), name: v.string(), status: v.string() });

// Replace the board with the console's ledger. Only documents that actually changed are written, so
// subscribed dashboards see exactly the instruments that moved.
export const sync = internalMutation({
  args: { items: v.array(item), counts: v.optional(v.any()) },
  handler: async (ctx, { items }) => {
    const existing = new Map((await ctx.db.query("instruments").collect()).map((d) => [d.uid, d]));
    for (const it of items) {
      const cur = existing.get(it.uid);
      if (!cur) await ctx.db.insert("instruments", it);
      else if (cur.status !== it.status || cur.name !== it.name) await ctx.db.patch(cur._id, { status: it.status, name: it.name });
      existing.delete(it.uid);
    }
    for (const gone of existing.values()) await ctx.db.delete(gone._id);
  },
});

export const list = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("instruments").collect();
    const items = rows
      .map((r) => ({ uid: r.uid, instrument: r.instrument, name: r.name, status: r.status }))
      .sort((a, b) => a.uid.localeCompare(b.uid));
    const counts: Record<string, number> = { tray: 0, held: 0, in_field: 0, used: 0 };
    for (const i of items) counts[i.status] = (counts[i.status] ?? 0) + 1;
    return { items, counts };
  },
});
