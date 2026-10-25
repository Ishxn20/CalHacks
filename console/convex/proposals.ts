import { internalMutation, query } from "./_generated/server";
import { v } from "convex/values";

export const upsert = internalMutation({
  args: {
    id: v.string(), fault: v.string(), status: v.string(), created: v.number(),
    simulated: v.optional(v.boolean()), summary: v.optional(v.string()), diff: v.optional(v.string()),
    changes: v.optional(v.any()), errors: v.optional(v.any()), steps: v.optional(v.any()),
    needs_restart: v.optional(v.boolean()), original_sha: v.optional(v.string()),
    cause: v.optional(v.string()), severity: v.optional(v.string()), safe_to_resume: v.optional(v.boolean()),
    evidence: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const cur = await ctx.db.query("proposals").withIndex("by_proposal_id", (q) => q.eq("id", args.id)).unique();
    if (cur) await ctx.db.replace(cur._id, args);
    else await ctx.db.insert("proposals", args);
  },
});

export const recent = query({
  args: {},
  handler: async (ctx) => (await ctx.db.query("proposals").order("desc").take(5)).map(({ _id, _creationTime, ...p }) => p),
});
