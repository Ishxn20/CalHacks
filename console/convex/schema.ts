import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // Every console event (utterance, intent, command, job, fault, count, ...), newest last by creation time.
  events: defineTable({
    kind: v.string(),
    data: v.any(),
    seq: v.number(),
    t: v.number(),
  }),

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
  }).index("by_proposal_id", ["id"]),
});
