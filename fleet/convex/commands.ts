import { ConvexError, v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { audit, requireOperator } from "./lib";

// Remote operators may ask for these; the arm's own server still decides whether it's safe to run them.
const REMOTE_KINDS = ["handoff", "stage", "release", "return", "home", "stop", "reset"] as const;

export const enqueue = mutation({
  args: {
    operatorKey: v.string(),
    robotId: v.id("robots"),
    kind: v.union(...REMOTE_KINDS.map((k) => v.literal(k))),
    text: v.optional(v.string()),
  },
  handler: async (ctx, { operatorKey, robotId, kind, text }) => {
    const actor = requireOperator(operatorKey);
    if ((kind === "handoff" || kind === "stage") && !text?.trim()) {
      throw new ConvexError(`${kind} needs a request, e.g. "needle holder, 6-0 prolene, backhand"`);
    }
    const id = await ctx.db.insert("commands", {
      robotId,
      kind,
      text: text?.trim() || null,
      status: "queued",
      createdBy: actor,
      createdAt: Date.now(),
      result: null,
    });
    await audit(ctx, actor, `command_${kind}`, robotId, text ?? "");
    return id;
  },
});

export const recent = query({
  args: { robotId: v.id("robots") },
  handler: (ctx, { robotId }) =>
    ctx.db
      .query("commands")
      .filter((q) => q.eq(q.field("robotId"), robotId))
      .order("desc")
      .take(15),
});
