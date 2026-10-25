import { internalMutation, internalQuery, query } from "./_generated/server";
import { internal } from "./_generated/api";
import type { Doc } from "./_generated/dataModel";
import { v } from "convex/values";
import { labels as labelsValidator } from "./schema";

// Writes are internal: only the console (deploy key) records episodes. Public reads return captions and
// flags only, never video links: footage leaves Convex only through the key-protected export (http.ts).

export const uploadUrl = internalMutation({
  args: {},
  handler: (ctx) => ctx.storage.generateUploadUrl(),
});

/** One finished arm job, uploaded by the console. Consented episodes with footage are sent to Claude for labels. */
export const record = internalMutation({
  args: {
    jobId: v.number(),
    kind: v.string(),
    request: v.union(v.string(), v.null()),
    startedAt: v.number(),
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
    consent: v.boolean(),
    simulated: v.boolean(),
  },
  handler: async (ctx, args) => {
    const label = args.consent && args.videoId !== null && args.keyframeIds.length > 0 ? "pending" : "skipped";
    const episodeId = await ctx.db.insert("episodes", {
      ...args,
      labelStatus: label,
      labelError: null,
      labels: null,
      review: "unreviewed",
    });
    if (label === "pending") await ctx.scheduler.runAfter(0, internal.labeling.labelEpisode, { episodeId });
    return episodeId;
  },
});

export const list = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("episodes").order("desc").take(12);
    return rows.map((e) => ({
      id: e._id,
      kind: e.kind,
      request: e.request,
      startedAt: e.startedAt,
      outcome: e.outcome,
      frameCount: e.frameCount,
      facesBlurred: e.facesBlurred,
      simulated: e.simulated,
      labelStatus: e.labelStatus,
      labelError: e.labelError,
      caption: e.labels?.caption ?? null,
      success: e.labels?.success ?? null,
      agreesWithRobot: e.labels?.agreesWithRobot ?? null,
      qualityFlags: e.labels?.qualityFlags ?? [],
      privacyFlags: e.labels?.privacyFlags ?? [],
      review: e.review,
    }));
  },
});

export const stats = query({
  args: {},
  handler: async (ctx) => {
    const all = await ctx.db.query("episodes").collect();
    const accepted = all.filter((e) => e.review === "accepted");
    return {
      total: all.length,
      labeled: all.filter((e) => e.labelStatus === "done").length,
      accepted: accepted.length,
      needsReview: all.filter(needsHuman).length,
      frames: accepted.reduce((n, e) => n + e.frameCount, 0),
      minutes: accepted.reduce((s, e) => s + e.frameCount / Math.max(e.fps, 1), 0) / 60,
    };
  },
});

/** Labels that disagree with the robot or carry quality/privacy flags wait for a person. */
function needsHuman(e: Doc<"episodes">): boolean {
  if (e.review !== "unreviewed" || e.labelStatus !== "done" || !e.labels) return false;
  return !e.labels.agreesWithRobot || e.labels.privacyFlags.length > 0 || e.labels.qualityFlags.length > 0;
}

// ── labelling pipeline (labeling.ts runs the model) ─────────────────────────

export const forLabeling = internalQuery({
  args: { episodeId: v.id("episodes") },
  handler: (ctx, { episodeId }) => ctx.db.get(episodeId),
});

export const setLabelStatus = internalMutation({
  args: {
    episodeId: v.id("episodes"),
    status: v.union(v.literal("labeling"), v.literal("failed"), v.literal("skipped")),
    error: v.union(v.string(), v.null()),
  },
  handler: (ctx, { episodeId, status, error }) => ctx.db.patch(episodeId, { labelStatus: status, labelError: error }),
});

export const saveLabels = internalMutation({
  args: { episodeId: v.id("episodes"), labels: labelsValidator },
  handler: async (ctx, { episodeId, labels }) => {
    const e = await ctx.db.get(episodeId);
    if (!e) return;
    // Clean, agreeing, consented episodes are accepted automatically; the rest wait for a person.
    const clean = labels.agreesWithRobot && labels.privacyFlags.length === 0 && labels.qualityFlags.length === 0;
    await ctx.db.patch(episodeId, { labels, labelStatus: "done", labelError: null, review: clean && e.consent ? "accepted" : "unreviewed" });
  },
});

export const retryFailedLabels = internalMutation({
  args: {},
  handler: async (ctx) => {
    const failed = await ctx.db.query("episodes").withIndex("by_labelStatus", (q) => q.eq("labelStatus", "failed")).take(5);
    for (const e of failed) {
      await ctx.db.patch(e._id, { labelStatus: "pending" });
      await ctx.scheduler.runAfter(0, internal.labeling.labelEpisode, { episodeId: e._id });
    }
  },
});

/** Accepted episodes with download URLs: what `python -m scrub_console export` turns into a training set. */
export const exportManifest = internalQuery({
  args: {},
  handler: async (ctx) => {
    const accepted = await ctx.db.query("episodes").withIndex("by_review", (q) => q.eq("review", "accepted")).collect();
    return Promise.all(
      accepted
        .filter((e) => e.consent && e.videoId && e.framesId && e.labels)
        .map(async (e) => ({
          id: e._id,
          kind: e.kind,
          request: e.request,
          startedAt: e.startedAt,
          fps: e.fps,
          frameCount: e.frameCount,
          outcome: e.outcome,
          labels: e.labels,
          videoUrl: await ctx.storage.getUrl(e.videoId!),
          framesUrl: await ctx.storage.getUrl(e.framesId!),
        })),
    );
  },
});
