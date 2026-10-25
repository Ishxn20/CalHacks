import { ConvexError, v } from "convex/values";
import { internal } from "./_generated/api";
import type { Doc } from "./_generated/dataModel";
import { internalMutation, internalQuery, mutation, query } from "./_generated/server";
import { audit, requireOperator } from "./lib";
import { labels as labelsValidator } from "./schema";

export const list = query({
  args: { robotId: v.optional(v.id("robots")), limit: v.optional(v.number()) },
  handler: async (ctx, { robotId, limit }) => {
    const rows = robotId
      ? await ctx.db
          .query("episodes")
          .withIndex("by_robot", (q) => q.eq("robotId", robotId))
          .order("desc")
          .take(limit ?? 25)
      : await ctx.db
          .query("episodes")
          .order("desc")
          .take(limit ?? 25);
    return Promise.all(
      rows.map(async (e) => ({
        ...e,
        videoUrl: e.videoId ? await ctx.storage.getUrl(e.videoId) : null,
        thumbUrl: e.keyframeIds[0] ? await ctx.storage.getUrl(e.keyframeIds[0]) : null,
      })),
    );
  },
});

export const stats = query({
  args: {},
  handler: async (ctx) => {
    const all = await ctx.db.query("episodes").collect();
    const count = (pred: (e: Doc<"episodes">) => boolean) => all.filter(pred).length;
    return {
      total: all.length,
      labeled: count((e) => e.labelStatus === "done"),
      accepted: count((e) => e.review === "accepted"),
      needsReview: count((e) => needsHuman(e)),
      frames: all.reduce((n, e) => n + (e.review === "accepted" ? e.frameCount : 0), 0),
      hours: all.reduce((s, e) => s + (e.review === "accepted" ? e.frameCount / Math.max(e.fps, 1) : 0), 0) / 3600,
      successRate:
        count((e) => e.labels !== null) === 0
          ? null
          : count((e) => e.labels?.success === true) / count((e) => e.labels !== null),
    };
  },
});

/** Labels Claude isn't sure about, or that disagree with the robot, or that show people, go to a person. */
function needsHuman(e: Doc<"episodes">): boolean {
  if (e.review !== "unreviewed" || e.labelStatus !== "done" || !e.labels) return false;
  return !e.labels.agreesWithRobot || e.labels.privacyFlags.length > 0 || e.labels.qualityFlags.length > 0;
}

export const review = mutation({
  args: { operatorKey: v.string(), episodeId: v.id("episodes"), accept: v.boolean() },
  handler: async (ctx, { operatorKey, episodeId, accept }) => {
    const actor = requireOperator(operatorKey);
    const e = await ctx.db.get(episodeId);
    if (!e) throw new ConvexError("unknown episode");
    if (accept && (!e.consent || e.labels === null)) throw new ConvexError("only consented, labelled episodes can be accepted");
    if (accept && e.labels && e.labels.privacyFlags.length > 0) {
      throw new ConvexError(`can't accept: privacy flags (${e.labels.privacyFlags.join(", ")}); re-run de-identification first`);
    }
    await ctx.db.patch(episodeId, { review: accept ? "accepted" : "rejected" });
    await audit(ctx, actor, accept ? "episode_accepted" : "episode_rejected", e.robotId, { episodeId });
  },
});

// ── labelling pipeline (labeling.ts runs the model) ─────────────────────────

export const forLabeling = internalQuery({
  args: { episodeId: v.id("episodes") },
  handler: async (ctx, { episodeId }) => {
    const e = await ctx.db.get(episodeId);
    if (!e) return null;
    const robot = await ctx.db.get(e.robotId);
    return { episode: e, robotName: robot?.name ?? "unknown" };
  },
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
    await ctx.db.patch(episodeId, {
      labels,
      labelStatus: "done",
      labelError: null,
      review: clean && e.consent ? "accepted" : "unreviewed",
    });
    await audit(ctx, "claude-labeler", "episode_labeled", e.robotId, {
      episodeId,
      success: labels.success,
      autoAccepted: clean && e.consent,
    });
  },
});

export const retryFailedLabels = internalMutation({
  args: {},
  handler: async (ctx) => {
    const failed = await ctx.db
      .query("episodes")
      .withIndex("by_labelStatus", (q) => q.eq("labelStatus", "failed"))
      .take(5);
    for (const e of failed) {
      await ctx.db.patch(e._id, { labelStatus: "pending" });
      await ctx.scheduler.runAfter(0, internal.labeling.labelEpisode, { episodeId: e._id });
    }
  },
});

/** Accepted episodes with download URLs: what `python -m scrub_fleet.export` turns into a training set. */
export const exportManifest = internalQuery({
  args: {},
  handler: async (ctx) => {
    const accepted = await ctx.db
      .query("episodes")
      .withIndex("by_review", (q) => q.eq("review", "accepted"))
      .collect();
    return Promise.all(
      accepted
        .filter((e) => e.consent && e.videoId && e.framesId && e.labels)
        .map(async (e) => ({
          id: e._id,
          robotId: e.robotId,
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
