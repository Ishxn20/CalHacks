import { internalMutation, query } from "./_generated/server";
import { v } from "convex/values";

// The arm's and the console's audit logs are hash chains: editing or deleting a MIDDLE entry breaks the chain,
// but cutting entries off the END leaves a valid (shorter) chain. Publishing each chain head here, somewhere
// the arm's disk cannot reach, closes that gap: a head that moves backwards, or two different heads for the
// same entry number, is recorded as an alert.
export const anchor = internalMutation({
  args: { source: v.string(), seq: v.number(), head: v.string(), ok: v.boolean() },
  handler: async (ctx, args) => {
    const t = Date.now();
    const rows = await ctx.db.query("anchors").withIndex("by_source", (q) => q.eq("source", args.source)).collect();
    const latest = rows.reduce<(typeof rows)[number] | null>((a, r) => (a === null || r.seq > a.seq ? r : a), null);
    if (latest && args.seq < latest.seq) {
      await ctx.db.insert("alerts", {
        source: args.source, kind: "rollback", t,
        detail: `${args.source} log went from ${latest.seq} entries back to ${args.seq}`,
      });
    }
    const same = rows.find((r) => r.seq === args.seq && r.head !== args.head);
    if (same) {
      await ctx.db.insert("alerts", {
        source: args.source, kind: "fork", t,
        detail: `${args.source} entry ${args.seq} was anchored with a different head before`,
      });
    }
    if (!rows.some((r) => r.seq === args.seq && r.head === args.head)) await ctx.db.insert("anchors", { ...args, t });
  },
});

export const latest = query({
  args: {},
  handler: async (ctx) => {
    const out: Record<string, { seq: number; head: string; ok: boolean; t: number }> = {};
    for (const r of await ctx.db.query("anchors").collect()) {
      if (!out[r.source] || r.seq >= out[r.source].seq) out[r.source] = { seq: r.seq, head: r.head, ok: r.ok, t: r.t };
    }
    return out;
  },
});

export const alerts = query({
  args: {},
  handler: async (ctx) => (await ctx.db.query("alerts").order("desc").take(20)).map((a) => ({ source: a.source, kind: a.kind, detail: a.detail, t: a.t })),
});
