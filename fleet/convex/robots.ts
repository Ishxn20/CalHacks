import { v } from "convex/values";
import { internal } from "./_generated/api";
import { internalAction, internalMutation, internalQuery, mutation, query } from "./_generated/server";
import { audit, requireOperator, sha256Hex } from "./lib";

const OFFLINE_AFTER_MS = 10_000;

/**
 * Register an arm and print its token (shown once; only the hash is stored).
 *   npx convex run robots:register '{"name": "OR-3 arm", "site": "UCSF Parnassus OR-3"}'
 */
export const register = internalAction({
  args: { name: v.string(), site: v.string() },
  handler: async (ctx, { name, site }): Promise<{ robotId: string; token: string }> => {
    const bytes = crypto.getRandomValues(new Uint8Array(24));
    const token = "rbt_" + [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
    const robotId: string = await ctx.runMutation(internal.robots.insert, {
      name,
      site,
      tokenHash: await sha256Hex(token),
    });
    return { robotId, token };
  },
});

export const insert = internalMutation({
  args: { name: v.string(), site: v.string(), tokenHash: v.string() },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("robots").withIndex("by_name", (q) => q.eq("name", args.name)).first();
    if (existing) throw new Error(`a robot named ${args.name} already exists`);
    const id = await ctx.db.insert("robots", {
      ...args,
      lastSeen: null,
      phase: "unknown",
      fault: null,
      enabled: false,
      qDeg: [],
      gripperDeg: null,
      job: null,
      agentVersion: null,
      recording: true,
    });
    await audit(ctx, "admin", "register_robot", id, { name: args.name, site: args.site });
    return id;
  },
});

export const byTokenHash = internalQuery({
  args: { tokenHash: v.string() },
  handler: (ctx, { tokenHash }) =>
    ctx.db.query("robots").withIndex("by_tokenHash", (q) => q.eq("tokenHash", tokenHash)).first(),
});

export const list = query({
  args: {},
  handler: async (ctx) => {
    const robots = await ctx.db.query("robots").collect();
    const now = Date.now();
    return Promise.all(
      robots.map(async ({ tokenHash: _secret, ...r }) => {
        const openIncidents = await ctx.db
          .query("incidents")
          .withIndex("by_robot", (q) => q.eq("robotId", r._id))
          .filter((q) => q.and(q.neq(q.field("status"), "resolved"), q.neq(q.field("status"), "rejected")))
          .collect();
        return {
          ...r,
          online: r.lastSeen !== null && now - r.lastSeen < OFFLINE_AFTER_MS,
          openIncidents: openIncidents.length,
        };
      }),
    );
  },
});

/** Pause/resume dataset recording for one arm (e.g. a patient declined research use). */
export const setRecording = mutation({
  args: { operatorKey: v.string(), robotId: v.id("robots"), recording: v.boolean() },
  handler: async (ctx, { operatorKey, robotId, recording }) => {
    const actor = requireOperator(operatorKey);
    await ctx.db.patch(robotId, { recording });
    await audit(ctx, actor, recording ? "recording_on" : "recording_off", robotId, "");
  },
});

export const byId = internalQuery({
  args: { robotId: v.id("robots") },
  handler: (ctx, { robotId }) => ctx.db.get(robotId),
});
