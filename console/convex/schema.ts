import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

const nums = v.array(v.union(v.number(), v.null()));

// One arm reading as the arm's control loop takes it (20 Hz), stamped by the console with phase/job/fault.
export const sample = v.object({
  t: v.number(), q: nums, q_cmd: nums, tau: nums, gripper: v.union(v.number(), v.null()),
  moving: v.boolean(), enabled: v.boolean(),
  phase: v.string(), job: v.union(v.number(), v.null()), fault: v.union(v.string(), v.null()),
});

// One second of readings for charts: the worst lag and |torque| per joint keep short spikes visible.
export const point = v.object({
  t: v.number(), q: nums, lag: nums, tau: nums, gripper: v.union(v.number(), v.null()),
  moving: v.boolean(), phase: v.string(), fault: v.union(v.string(), v.null()),
});

// What Claude writes for each recorded episode (labeling.ts).
export const labels = v.object({
  task: v.string(),
  caption: v.string(),
  success: v.boolean(),
  failureMode: v.union(v.string(), v.null()),
  segments: v.array(v.object({ startS: v.number(), endS: v.number(), phase: v.string(), description: v.string() })),
  instrumentVisible: v.boolean(),
  qualityFlags: v.array(v.string()),
  // Anything that could identify a patient or staff member (faces, name badges, screens, wristbands).
  privacyFlags: v.array(v.string()),
  agreesWithRobot: v.boolean(),
  model: v.string(),
});

export default defineSchema({
  // One arm job (handoff / stage / return) recorded for world-model training: face-blurred wrist video,
  // a per-frame state/action table and the spoken request, all in Convex file storage.
  episodes: defineTable({
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
    labelStatus: v.union(v.literal("pending"), v.literal("labeling"), v.literal("done"), v.literal("failed"), v.literal("skipped")),
    labelError: v.union(v.string(), v.null()),
    labels: v.union(labels, v.null()),
    // Dataset gate: only consented, labelled, privacy-clean episodes are exported for training.
    review: v.union(v.literal("unreviewed"), v.literal("accepted"), v.literal("rejected")),
    // Claude's task, caption and failure mode as one string, for full-text search when curating the dataset.
    searchText: v.optional(v.string()),
    // The Action Retrier run labelling this episode, so its completion callback can find it.
    labelRunId: v.optional(v.string()),
  })
    .index("by_labelStatus", ["labelStatus"])
    .index("by_labelRunId", ["labelRunId"])
    .index("by_review", ["review"])
    .searchIndex("search_labels", { searchField: "searchText", filterFields: ["review"] }),

  // Every console event (utterance, intent, command, job, fault, count, ...), newest last by creation time.
  events: defineTable({
    kind: v.string(),
    data: v.any(),
    seq: v.number(),
    t: v.number(),
  })
    .index("by_t", ["t"])
    .index("by_kind", ["kind"]),

  // Raw arm readings, one document per ~5 s batch (~100 samples). Kept for a day; see metrics.ts.
  metrics: defineTable({
    t0: v.number(),
    t1: v.number(),
    source: v.string(), // arm | sim
    samples: v.array(sample),
  }).index("by_t0", ["t0"]),

  // The same batches rolled up to one point per second: what the dashboard charts read.
  metricPoints: defineTable({
    t0: v.number(),
    source: v.string(),
    points: v.array(point),
  }).index("by_t0", ["t0"]),

  // One document per physical instrument: where it is right now.
  instruments: defineTable({
    uid: v.string(),
    instrument: v.string(),
    name: v.string(),
    status: v.string(), // tray | held | in_field | used
  }).index("by_uid", ["uid"]),

  // Audit-log chain heads published off the arm's own disk. See audit.ts.
  anchors: defineTable({
    source: v.string(), // arm | console
    seq: v.number(),
    head: v.string(),
    ok: v.boolean(),
    t: v.number(),
  }).index("by_source", ["source"]),

  // Something that should never happen: a log that went backwards, or two different heads for one entry.
  alerts: defineTable({
    source: v.string(),
    kind: v.string(), // rollback | fork
    detail: v.string(),
    t: v.number(),
  }),

  // Droid's fault-triage proposals and their status.
  proposals: defineTable({
    id: v.string(),
    fault: v.string(),
    status: v.string(),
    created: v.number(),
    simulated: v.optional(v.boolean()),
    summary: v.optional(v.string()),
    diff: v.optional(v.string()),
    changes: v.optional(v.any()),
    errors: v.optional(v.any()),
    steps: v.optional(v.any()),
    needs_restart: v.optional(v.boolean()),
    original_sha: v.optional(v.string()),
    // Droid's structured diagnosis (triage.py DIAGNOSIS_SCHEMA).
    cause: v.optional(v.string()),
    severity: v.optional(v.string()),
    safe_to_resume: v.optional(v.boolean()),
    evidence: v.optional(v.any()),
  }).index("by_proposal_id", ["id"]),
});
