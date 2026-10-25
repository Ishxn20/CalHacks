"use node";
// Claude turns each recorded episode into training labels for the world-model dataset.
import Anthropic from "@anthropic-ai/sdk";
import { v } from "convex/values";
import { z } from "zod";
import { internal } from "./_generated/api";
import { internalAction } from "./_generated/server";

const MODEL = "claude-opus-5-5";

const Labels = z.object({
  task: z.string(),
  caption: z.string(),
  success: z.boolean(),
  failureMode: z.string().nullable(),
  segments: z.array(
    z.object({ startS: z.number(), endS: z.number(), phase: z.string(), description: z.string() }),
  ),
  instrumentVisible: z.boolean(),
  qualityFlags: z.array(z.string()),
  privacyFlags: z.array(z.string()),
  agreesWithRobot: z.boolean(),
});

const LABEL_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "task",
    "caption",
    "success",
    "failureMode",
    "segments",
    "instrumentVisible",
    "qualityFlags",
    "privacyFlags",
    "agreesWithRobot",
  ],
  properties: {
    task: { type: "string", description: "Imperative task text, e.g. 'hand the Ryder needle holder handle-first, backhand'" },
    caption: { type: "string", description: "One or two sentences describing what the arm did, for a video-language model" },
    success: { type: "boolean" },
    failureMode: {
      type: ["string", "null"],
      description: "null on success; else one of: missed_grasp, dropped, wrong_instrument, collision, timeout, perception, other",
    },
    segments: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["startS", "endS", "phase", "description"],
        properties: {
          startS: { type: "number" },
          endS: { type: "number" },
          phase: { type: "string" },
          description: { type: "string" },
        },
      },
    },
    instrumentVisible: { type: "boolean" },
    qualityFlags: {
      type: "array",
      items: { type: "string" },
      description: "Problems that make the clip worse training data: blurry, dark, occluded, camera_unplugged, gripper_only, clipped",
    },
    privacyFlags: {
      type: "array",
      items: { type: "string" },
      description: "Anything that could identify a person: face, name_badge, patient_wristband, screen_text, tattoo. Empty if none.",
    },
    agreesWithRobot: { type: "boolean", description: "Whether the frames support the robot's own reported outcome" },
  },
} as const;

const SYSTEM = `You label recordings from a robotic surgical scrub assistant for a world-model training set.
Each episode is one arm job: the robot hears a request, picks a 3D-printed surgical instrument with a wrist camera
looking along its gripper (its black jaws fill the bottom centre of every frame), and hands it to the surgeon.

Label only what the frames and the state timeline support. If they don't show whether the instrument was taken,
say so in the caption and set success from the robot's outcome, with agreesWithRobot reflecting whether anything
you can see contradicts it. Segments must cover the episode in order, using the phase timeline's times.

Privacy matters more than completeness: flag anything that could identify a patient or staff member, even partly
visible or blurred badly. These clips are shared with researchers only when the privacy flags are empty.`;

type Frame = { t: number; phase: string; gripperDeg: number | null; holding?: boolean };

/** Collapse the per-frame state table into phase intervals Claude can read. */
function phaseTimeline(frames: Frame[]): string {
  if (frames.length === 0) return "(no state recorded)";
  const t0 = frames[0].t;
  const out: string[] = [];
  let start = frames[0];
  for (let i = 1; i <= frames.length; i++) {
    const f = frames[i];
    if (!f || f.phase !== start.phase) {
      const end = frames[i - 1];
      const g = end.gripperDeg === null ? "?" : end.gripperDeg.toFixed(0);
      out.push(`${(start.t - t0).toFixed(1)}-${(end.t - t0).toFixed(1)} s: ${start.phase} (gripper ${g} deg at end)`);
      if (f) start = f;
    }
  }
  return out.join("\n");
}

export const labelEpisode = internalAction({
  args: { episodeId: v.id("episodes") },
  handler: async (ctx, { episodeId }) => {
    const e = await ctx.runQuery(internal.episodes.forLabeling, { episodeId });
    if (!e || !e.consent || e.keyframeIds.length === 0) return;
    await ctx.runMutation(internal.episodes.setLabelStatus, { episodeId, status: "labeling", error: null });
    try {
      let frames: Frame[] = [];
      if (e.framesId) {
        const blob = await ctx.storage.get(e.framesId);
        if (blob) frames = (await blob.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l) as Frame);
      }
      const content: Anthropic.Beta.BetaContentBlockParam[] = [];
      for (const [i, id] of e.keyframeIds.entries()) {
        const blob = await ctx.storage.get(id);
        if (!blob) continue;
        content.push({ type: "text", text: `Keyframe at ${e.keyframeTimesS[i]?.toFixed(1) ?? "?"} s:` });
        content.push({
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: Buffer.from(await blob.arrayBuffer()).toString("base64") },
        });
      }
      content.push({
        type: "text",
        text: [
          `${e.simulated ? "Simulated arm: the frames are placeholders, not a camera." : "Real arm."} Job kind: ${e.kind}. Spoken request: ${e.request ?? "(none)"}.`,
          `Robot-reported outcome: ${e.outcome ?? "unknown"}${e.error ? ` (error: ${e.error})` : ""}.`,
          `Duration ${e.endedAt ? ((e.endedAt - e.startedAt) / 1000).toFixed(1) : "?"} s, ${e.frameCount} frames at ${e.fps} fps; faces blurred on-device: ${e.facesBlurred}.`,
          "Phase timeline from the arm controller:",
          phaseTimeline(frames),
          "Label this episode.",
        ].join("\n"),
      });

      const client = new Anthropic();
      const response = await client.beta.messages.create({
        model: MODEL,
        max_tokens: 16000,
        betas: ["server-side-fallback-2026-07-01"],
        fallbacks: "default",
        thinking: { type: "adaptive" },
        output_config: { effort: "medium", format: { type: "json_schema", schema: LABEL_SCHEMA } },
        system: SYSTEM,
        messages: [{ role: "user", content }],
      });
      if (response.stop_reason === "refusal") {
        // Not transient: don't let the retry cron resend it. A person can look at the clip instead.
        await ctx.runMutation(internal.episodes.setLabelStatus, {
          episodeId,
          status: "skipped",
          error: `labeler declined (${response.stop_details?.category ?? "no category"})`,
        });
        return;
      }
      if (response.stop_reason === "max_tokens") throw new Error("label output was truncated");
      const text = response.content.find((b) => b.type === "text");
      if (!text || text.type !== "text") throw new Error("no label in response");
      const labels = Labels.parse(JSON.parse(text.text));
      await ctx.runMutation(internal.episodes.saveLabels, {
        episodeId,
        labels: { ...labels, model: response.model },
      });
    } catch (err) {
      // Typed SDK errors (rate limit, 5xx) and validation failures alike: mark failed; the cron retries.
      const msg =
        err instanceof Anthropic.APIError ? `API ${err.status}: ${err.message}` : err instanceof Error ? err.message : String(err);
      await ctx.runMutation(internal.episodes.setLabelStatus, { episodeId, status: "failed", error: msg });
    }
  },
});
