import { ConvexError, v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { audit, checkChange, requireOperator, requireWorker, TUNABLE } from "./lib";
import { configChange } from "./schema";

export const list = query({
  args: { robotId: v.optional(v.id("robots")) },
  handler: async (ctx, { robotId }) => {
    const rows = robotId
      ? await ctx.db
          .query("incidents")
          .withIndex("by_robot", (q) => q.eq("robotId", robotId))
          .order("desc")
          .take(20)
      : await ctx.db.query("incidents").order("desc").take(20);
    return rows;
  },
});

export const tunables = query({ args: {}, handler: () => TUNABLE });

// ── Droid worker (WORKER_KEY) ───────────────────────────────────────────────

/** Incidents waiting for a diagnosis. The worker subscribes to this and wakes on new faults. */
export const open = query({
  args: { workerKey: v.string() },
  handler: async (ctx, { workerKey }) => {
    requireWorker(workerKey);
    const rows = await ctx.db
      .query("incidents")
      .withIndex("by_status", (q) => q.eq("status", "open"))
      .take(10);
    return Promise.all(
      rows.map(async (i) => ({ ...i, robotName: (await ctx.db.get(i.robotId))?.name ?? "unknown" })),
    );
  },
});

export const claim = mutation({
  args: { workerKey: v.string(), incidentId: v.id("incidents") },
  handler: async (ctx, { workerKey, incidentId }) => {
    const actor = requireWorker(workerKey);
    const inc = await ctx.db.get(incidentId);
    if (!inc || inc.status !== "open") return false; // another worker got it
    await ctx.db.patch(incidentId, { status: "diagnosing" });
    await audit(ctx, actor, "incident_claimed", inc.robotId, { incidentId });
    return true;
  },
});

export const step = mutation({
  args: { workerKey: v.string(), incidentId: v.id("incidents"), kind: v.string(), text: v.string() },
  handler: async (ctx, { workerKey, incidentId, kind, text }) => {
    requireWorker(workerKey);
    const inc = await ctx.db.get(incidentId);
    if (!inc) return;
    const steps = [...inc.steps, { t: Date.now(), kind, text: text.slice(0, 4000) }].slice(-200);
    await ctx.db.patch(incidentId, { steps });
  },
});

export const propose = mutation({
  args: {
    workerKey: v.string(),
    incidentId: v.id("incidents"),
    diagnosis: v.string(),
    severity: v.union(v.literal("low"), v.literal("medium"), v.literal("high")),
    safeToResume: v.boolean(),
    change: v.union(configChange, v.null()),
  },
  handler: async (ctx, { workerKey, incidentId, diagnosis, severity, safeToResume, change }) => {
    const actor = requireWorker(workerKey);
    const inc = await ctx.db.get(incidentId);
    if (!inc) throw new ConvexError("unknown incident");
    // The worker can only propose; bounds are enforced here so an out-of-range idea never reaches a person.
    let note = "";
    if (change) {
      try {
        checkChange(change.key, change.to);
      } catch (e) {
        note = ` (proposed change dropped: ${e instanceof Error ? e.message : e})`;
        change = null;
      }
    }
    await ctx.db.patch(incidentId, {
      status: "proposed",
      diagnosis: diagnosis + note,
      severity,
      safeToResume,
      change,
    });
    await audit(ctx, actor, "incident_proposed", inc.robotId, { incidentId, severity, change });
  },
});

export const fail = mutation({
  args: { workerKey: v.string(), incidentId: v.id("incidents"), error: v.string() },
  handler: async (ctx, { workerKey, incidentId, error }) => {
    const actor = requireWorker(workerKey);
    const inc = await ctx.db.get(incidentId);
    if (!inc) return;
    await ctx.db.patch(incidentId, { status: "failed", diagnosis: `Droid could not diagnose: ${error}` });
    await audit(ctx, actor, "incident_failed", inc.robotId, { incidentId, error });
  },
});

// ── operator decision ───────────────────────────────────────────────────────

export const decide = mutation({
  args: { operatorKey: v.string(), incidentId: v.id("incidents"), approve: v.boolean() },
  handler: async (ctx, { operatorKey, incidentId, approve }) => {
    const actor = requireOperator(operatorKey);
    const inc = await ctx.db.get(incidentId);
    if (!inc || inc.status !== "proposed") throw new ConvexError("incident has no pending proposal");
    const now = Date.now();
    if (!approve) {
      await ctx.db.patch(incidentId, { status: "rejected", decidedBy: actor, decidedAt: now });
      await audit(ctx, actor, "incident_rejected", inc.robotId, { incidentId });
      return;
    }
    if (inc.change) {
      checkChange(inc.change.key, inc.change.to); // re-check: bounds may have tightened since the proposal
      await ctx.db.insert("commands", {
        robotId: inc.robotId,
        kind: "apply_config",
        text: null,
        change: inc.change,
        status: "queued",
        createdBy: actor,
        createdAt: now,
        result: null,
      });
    }
    await ctx.db.patch(incidentId, {
      status: inc.change ? "approved" : "resolved",
      decidedBy: actor,
      decidedAt: now,
    });
    await audit(ctx, actor, "incident_approved", inc.robotId, { incidentId, change: inc.change });
  },
});

export const auditLog = query({
  args: {},
  handler: (ctx) => ctx.db.query("audit").withIndex("by_t").order("desc").take(40),
});
