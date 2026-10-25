// Mutations the robot agent reaches through http.ts (already authenticated to one robot).
import { ConvexError, v } from "convex/values";
import { internal } from "./_generated/api";
import type { Doc, Id } from "./_generated/dataModel";
import { internalMutation } from "./_generated/server";
import { audit } from "./lib";

const sample = v.object({
  t: v.number(),
  phase: v.string(),
  qDeg: v.array(v.union(v.number(), v.null())),
  qCmdDeg: v.array(v.union(v.number(), v.null())),
  gripperDeg: v.union(v.number(), v.null()),
});

function trackingErr(s: { qDeg: (number | null)[]; qCmdDeg: (number | null)[] }): number | null {
  let worst: number | null = null;
  s.qDeg.forEach((q, i) => {
    const c = s.qCmdDeg[i];
    if (q !== null && c !== null && c !== undefined) worst = Math.max(worst ?? 0, Math.abs(q - c));
  });
  return worst;
}

export const heartbeat = internalMutation({
  args: {
    robotId: v.id("robots"),
    agentVersion: v.string(),
    enabled: v.boolean(),
    fault: v.union(v.string(), v.null()),
    job: v.union(v.string(), v.null()),
    samples: v.array(sample),
  },
  handler: async (ctx, { robotId, agentVersion, enabled, fault, job, samples }) => {
    const robot = await ctx.db.get(robotId);
    if (!robot) throw new ConvexError("unknown robot");
    for (const s of samples.slice(-50)) {
      await ctx.db.insert("telemetry", { robotId, ...s, trackingErrMaxDeg: trackingErr(s) });
    }
    const last = samples.at(-1);
    await ctx.db.patch(robotId, {
      lastSeen: Date.now(),
      agentVersion,
      enabled,
      fault,
      job,
      ...(last ? { phase: last.phase, qDeg: last.qDeg, gripperDeg: last.gripperDeg } : {}),
    });
    return { recording: robot.recording };
  },
});

/** Hand queued commands to the robot (oldest first) and mark them sent. */
export const takeCommands = internalMutation({
  args: { robotId: v.id("robots") },
  handler: async (ctx, { robotId }) => {
    const queued = await ctx.db
      .query("commands")
      .withIndex("by_robot_status", (q) => q.eq("robotId", robotId).eq("status", "queued"))
      .order("asc")
      .take(10);
    for (const c of queued) await ctx.db.patch(c._id, { status: "sent" });
    return queued.map((c) => ({ id: c._id, kind: c.kind, text: c.text, change: c.change ?? null }));
  },
});

export const commandResult = internalMutation({
  args: { robotId: v.id("robots"), commandId: v.id("commands"), ok: v.boolean(), result: v.string() },
  handler: async (ctx, { robotId, commandId, ok, result }) => {
    const c = await ctx.db.get(commandId);
    if (!c || c.robotId !== robotId) throw new ConvexError("unknown command");
    await ctx.db.patch(commandId, { status: ok ? "done" : "failed", result });
    if (c.kind === "apply_config" && ok) {
      const inc = await ctx.db
        .query("incidents")
        .withIndex("by_robot", (q) => q.eq("robotId", robotId))
        .filter((q) => q.eq(q.field("status"), "approved"))
        .first();
      if (inc) await ctx.db.patch(inc._id, { status: "resolved" });
    }
    await audit(ctx, "robot", `command_${ok ? "done" : "failed"}`, robotId, { kind: c.kind, result });
  },
});

export const startEpisode = internalMutation({
  args: {
    robotId: v.id("robots"),
    jobId: v.number(),
    kind: v.string(),
    request: v.union(v.string(), v.null()),
    startedAt: v.number(),
    consent: v.boolean(),
  },
  handler: async (ctx, args) => {
    const dup = await ctx.db
      .query("episodes")
      .withIndex("by_robot_job", (q) => q.eq("robotId", args.robotId).eq("jobId", args.jobId))
      .filter((q) => q.eq(q.field("startedAt"), args.startedAt))
      .first();
    if (dup) return dup._id;
    return ctx.db.insert("episodes", {
      ...args,
      endedAt: null,
      outcome: null,
      error: null,
      videoId: null,
      framesId: null,
      keyframeIds: [],
      keyframeTimesS: [],
      frameCount: 0,
      fps: 0,
      facesBlurred: 0,
      labelStatus: "waiting",
      labelError: null,
      labels: null,
      review: "unreviewed",
    });
  },
});

export const uploadUrl = internalMutation({
  args: {},
  handler: (ctx) => ctx.storage.generateUploadUrl(),
});

export const finishEpisode = internalMutation({
  args: {
    robotId: v.id("robots"),
    episodeId: v.id("episodes"),
    endedAt: v.number(),
    outcome: v.string(),
    error: v.union(v.string(), v.null()),
    videoId: v.union(v.id("_storage"), v.null()),
    framesId: v.union(v.id("_storage"), v.null()),
    keyframeIds: v.array(v.id("_storage")),
    keyframeTimesS: v.array(v.number()),
    frameCount: v.number(),
    fps: v.number(),
    facesBlurred: v.number(),
  },
  handler: async (ctx, { robotId, episodeId, ...rest }) => {
    const ep = await ctx.db.get(episodeId);
    if (!ep || ep.robotId !== robotId) throw new ConvexError("unknown episode");
    // No consent or no footage: keep the record for fleet stats, but never label or export it.
    const label: Doc<"episodes">["labelStatus"] =
      ep.consent && rest.videoId !== null && rest.keyframeIds.length > 0 ? "pending" : "skipped";
    await ctx.db.patch(episodeId, { ...rest, labelStatus: label });
    if (label === "pending") await ctx.scheduler.runAfter(0, internal.labeling.labelEpisode, { episodeId });
  },
});

export const openIncident = internalMutation({
  args: { robotId: v.id("robots"), fault: v.string(), phase: v.string(), context: v.string() },
  handler: async (ctx, args): Promise<Id<"incidents">> => {
    // One live incident per fault: a robot re-reporting the same fault doesn't open duplicates.
    const live = await ctx.db
      .query("incidents")
      .withIndex("by_robot", (q) => q.eq("robotId", args.robotId))
      .order("desc")
      .first();
    if (live && live.fault === args.fault && !["resolved", "rejected", "failed"].includes(live.status)) {
      return live._id;
    }
    const id = await ctx.db.insert("incidents", {
      ...args,
      openedAt: Date.now(),
      status: "open",
      steps: [],
      diagnosis: null,
      severity: null,
      safeToResume: null,
      change: null,
      decidedBy: null,
      decidedAt: null,
    });
    await audit(ctx, "robot", "incident_opened", args.robotId, { fault: args.fault });
    return id;
  },
});
