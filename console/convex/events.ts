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

// The log view: newest first, only the kinds asked for (utterance, command, arm_event, fault, job, ...).
export const byKind = query({
  args: { kinds: v.array(v.string()), limit: v.optional(v.number()) },
  handler: async (ctx, { kinds, limit }) => {
    const n = Math.max(1, Math.min(limit ?? 100, 300));
    const per = await Promise.all(
      kinds.slice(0, 20).map((k) => ctx.db.query("events").withIndex("by_kind", (q) => q.eq("kind", k)).order("desc").take(n)),
    );
    return per
      .flat()
      .sort((a, b) => b._creationTime - a._creationTime)
      .slice(0, n)
      .map((r) => ({ seq: r.seq, t: r.t, kind: r.kind, data: r.data }));
  },
});
