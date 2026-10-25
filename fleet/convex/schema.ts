import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

// Joint vectors are degrees, joint1..joint6, as reported by scrub_arm's /status.
const jointVec = v.array(v.union(v.number(), v.null()));

export const commandKind = v.union(
  v.literal("handoff"),
  v.literal("stage"),
  v.literal("release"),
  v.literal("return"),
  v.literal("home"),
  v.literal("stop"),
  v.literal("reset"),
  v.literal("apply_config"),
);

export const labels = v.object({
  task: v.string(),
  caption: v.string(),
  success: v.boolean(),
  failureMode: v.union(v.string(), v.null()),
  segments: v.array(
    v.object({ startS: v.number(), endS: v.number(), phase: v.string(), description: v.string() }),
  ),
  instrumentVisible: v.boolean(),
  qualityFlags: v.array(v.string()),
  // Anything that could identify a patient or staff member (faces, name badges, screens, wristbands).
  privacyFlags: v.array(v.string()),
  agreesWithRobot: v.boolean(),
  model: v.string(),
});

export const configChange = v.object({
  key: v.string(),
  from: v.union(v.number(), v.null()),
  to: v.number(),
  rationale: v.string(),
});

export default defineSchema({
  robots: defineTable({
    name: v.string(),
    site: v.string(),
    tokenHash: v.string(),
    lastSeen: v.union(v.number(), v.null()),
    phase: v.string(),
    fault: v.union(v.string(), v.null()),
    enabled: v.boolean(),
    qDeg: jointVec,
    gripperDeg: v.union(v.number(), v.null()),
    job: v.union(v.string(), v.null()),
    agentVersion: v.union(v.string(), v.null()),
    recording: v.boolean(),
  })
    .index("by_tokenHash", ["tokenHash"])
    .index("by_name", ["name"]),

  // Downsampled state stream (the agent sends ~2 Hz; full-rate state lives in each episode's file).
  telemetry: defineTable({
    robotId: v.id("robots"),
    t: v.number(),
    phase: v.string(),
    qDeg: jointVec,
    qCmdDeg: jointVec,
    gripperDeg: v.union(v.number(), v.null()),
    trackingErrMaxDeg: v.union(v.number(), v.null()),
  }).index("by_robot_t", ["robotId", "t"]),

  commands: defineTable({
    robotId: v.id("robots"),
    kind: commandKind,
    text: v.union(v.string(), v.null()),
    change: v.optional(configChange),
    status: v.union(v.literal("queued"), v.literal("sent"), v.literal("done"), v.literal("failed")),
    createdBy: v.string(),
    createdAt: v.number(),
    result: v.union(v.string(), v.null()),
  }).index("by_robot_status", ["robotId", "status"]),

  // One episode = one arm job (handoff / stage / return ...), recorded for world-model training:
  // de-identified wrist video + a per-frame state/action table + the spoken request.
  episodes: defineTable({
    robotId: v.id("robots"),
    jobId: v.number(),
    kind: v.string(),
    request: v.union(v.string(), v.null()),
    startedAt: v.number(),
    endedAt: v.union(v.number(), v.null()),
    outcome: v.union(v.string(), v.null()),
    error: v.union(v.string(), v.null()),
    videoId: v.union(v.id("_storage"), v.null()),
    framesId: v.union(v.id("_storage"), v.null()),
    keyframeIds: v.array(v.id("_storage")),
    keyframeTimesS: v.array(v.number()),
    frameCount: v.number(),
    fps: v.number(),
    facesBlurred: v.number(),
    consent: v.boolean(),
    labelStatus: v.union(
      v.literal("waiting"),
      v.literal("pending"),
      v.literal("labeling"),
      v.literal("done"),
      v.literal("failed"),
      v.literal("skipped"),
    ),
    labelError: v.union(v.string(), v.null()),
    labels: v.union(labels, v.null()),
    // Dataset gate: only reviewed, consented, privacy-clean episodes are exported for training.
    review: v.union(v.literal("unreviewed"), v.literal("accepted"), v.literal("rejected")),
  })
    .index("by_robot", ["robotId", "startedAt"])
    .index("by_robot_job", ["robotId", "jobId"])
    .index("by_labelStatus", ["labelStatus"])
    .index("by_review", ["review"]),

  incidents: defineTable({
    robotId: v.id("robots"),
    openedAt: v.number(),
    fault: v.string(),
    phase: v.string(),
    context: v.string(), // JSON: recent telemetry, arm events, config excerpt, last job
    status: v.union(
      v.literal("open"),
      v.literal("diagnosing"),
      v.literal("proposed"),
      v.literal("approved"),
      v.literal("rejected"),
      v.literal("resolved"),
      v.literal("failed"),
    ),
    steps: v.array(v.object({ t: v.number(), kind: v.string(), text: v.string() })),
    diagnosis: v.union(v.string(), v.null()),
    severity: v.union(v.literal("low"), v.literal("medium"), v.literal("high"), v.null()),
    safeToResume: v.union(v.boolean(), v.null()),
    change: v.union(configChange, v.null()),
    decidedBy: v.union(v.string(), v.null()),
    decidedAt: v.union(v.number(), v.null()),
  })
    .index("by_status", ["status"])
    .index("by_robot", ["robotId", "openedAt"]),

  audit: defineTable({
    t: v.number(),
    actor: v.string(),
    action: v.string(),
    robotId: v.union(v.id("robots"), v.null()),
    detail: v.string(),
  }).index("by_t", ["t"]),
});
