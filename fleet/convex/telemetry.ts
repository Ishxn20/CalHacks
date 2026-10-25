import { v } from "convex/values";
import { internalMutation, query } from "./_generated/server";

export const recent = query({
  args: { robotId: v.id("robots"), seconds: v.optional(v.number()) },
  handler: (ctx, { robotId, seconds }) => {
    const since = Date.now() - (seconds ?? 60) * 1000;
    return ctx.db
      .query("telemetry")
      .withIndex("by_robot_t", (q) => q.eq("robotId", robotId).gte("t", since))
      .take(600);
  },
});

/** Telemetry is an operational stream; the training data lives in episodes. Keep 24 h. */
export const prune = internalMutation({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() - 24 * 3600 * 1000;
    const old = await ctx.db
      .query("telemetry")
      .filter((q) => q.lt(q.field("t"), cutoff))
      .take(2000);
    for (const row of old) await ctx.db.delete(row._id);
    return old.length;
  },
});
