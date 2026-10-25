import { internalMutation, query } from "./_generated/server";
import { v } from "convex/values";

// Writes are internal: only the console (holding the deploy key) can add events, so the public URL
// cannot be used to forge history. Reads are public live queries the dashboard subscribes to.
export const add = internalMutation({
  args: { kind: v.string(), data: v.any(), seq: v.number(), t: v.number() },
  handler: async (ctx, args) => {
    await ctx.db.insert("events", args);
  },
});

export const recent = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("events").order("desc").take(150);
    return rows.reverse().map((r) => ({ seq: r.seq, t: r.t, kind: r.kind, data: r.data }));
  },
});

const KEEP_MS = 7 * 24 * 3600 * 1000;

export const prune = internalMutation({
  args: {},
  handler: async (ctx) => {
    const old = await ctx.db.query("events").withIndex("by_t", (q) => q.lt("t", (Date.now() - KEEP_MS) / 1000)).take(1000);
    for (const e of old) await ctx.db.delete(e._id);
    return old.length;
  },
});
