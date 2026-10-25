/// <reference types="vite/client" />
import actionRetrierTest from "@convex-dev/action-retrier/test";
import { convexTest, type TestConvex } from "convex-test";
import { expect, test } from "vitest";
import { api, internal } from "./_generated/api";
import schema from "./schema";

const modules = import.meta.glob("./**/!(*.*.*)*.*s");

function setup(): TestConvex<typeof schema> {
  const t = convexTest(schema, modules);
  actionRetrierTest.register(t);
  return t;
}
const board = (...rows: [string, string, string][]) =>
  rows.map(([uid, instrument, status]) => ({ uid, instrument, name: instrument.toUpperCase(), status }));

test("events come back oldest first and are capped at the newest 150", async () => {
  const t = setup();
  for (let i = 1; i <= 160; i++) await t.mutation(internal.events.add, { kind: "utterance", data: { text: `r${i}` }, seq: i, t: i });
  const rows = await t.query(api.events.recent, {});
  expect(rows).toHaveLength(150);
  expect(rows[0].seq).toBe(11);
  expect(rows[149]).toMatchObject({ seq: 160, kind: "utterance", data: { text: "r160" } });
});

test("writes are internal: the generated public api exposes only the read queries", () => {
  // Each line below must fail to type-check, because these are internalMutations (npm run typecheck proves it).
  // @ts-expect-error events.add is internal
  void api.events.add;
  // @ts-expect-error instruments.sync is internal
  void api.instruments.sync;
  // @ts-expect-error audit.anchor is internal
  void api.audit.anchor;
  // @ts-expect-error proposals.upsert is internal
  void api.proposals.upsert;
  expect(typeof internal.events.add).toBeDefined();
});

test("the board syncs: inserts, moves, and removals; counts follow", async () => {
  const t = setup();
  await t.mutation(internal.instruments.sync, { items: board(["a#1", "resano", "tray"], ["b#1", "potts", "tray"]) });
  expect((await t.query(api.instruments.list, {})).counts).toMatchObject({ tray: 2, in_field: 0 });
  await t.mutation(internal.instruments.sync, { items: board(["a#1", "resano", "in_field"], ["b#1", "potts", "tray"]) });
  const after = await t.query(api.instruments.list, {});
  expect(after.counts).toMatchObject({ tray: 1, in_field: 1 });
  expect(after.items.find((i) => i.uid === "a#1")?.status).toBe("in_field");
  await t.mutation(internal.instruments.sync, { items: board(["a#1", "resano", "used"]) });
  expect((await t.query(api.instruments.list, {})).items).toHaveLength(1);
});

test("an unchanged board writes nothing new", async () => {
  const t = setup();
  const items = board(["a#1", "resano", "tray"]);
  await t.mutation(internal.instruments.sync, { items });
  const before = await t.run((ctx) => ctx.db.query("instruments").collect());
  await t.mutation(internal.instruments.sync, { items });
  const after = await t.run((ctx) => ctx.db.query("instruments").collect());
  expect(after).toEqual(before);
});

test("audit anchors: normal growth is quiet, a log that goes backwards raises an alert", async () => {
  const t = setup();
  for (const seq of [1, 2, 5]) await t.mutation(internal.audit.anchor, { source: "arm", seq, head: `h${seq}`, ok: true });
  expect(await t.query(api.audit.alerts, {})).toHaveLength(0);
  expect((await t.query(api.audit.latest, {})).arm).toMatchObject({ seq: 5, head: "h5" });
  await t.mutation(internal.audit.anchor, { source: "arm", seq: 3, head: "h3", ok: true }); // someone cut the log back
  const alerts = await t.query(api.audit.alerts, {});
  expect(alerts).toHaveLength(1);
  expect(alerts[0]).toMatchObject({ source: "arm", kind: "rollback" });
  expect(alerts[0].detail).toContain("5 entries back to 3");
});

test("two different heads for the same entry is a fork alert; a repeat is ignored", async () => {
  const t = setup();
  await t.mutation(internal.audit.anchor, { source: "console", seq: 4, head: "aaa", ok: true });
  await t.mutation(internal.audit.anchor, { source: "console", seq: 4, head: "aaa", ok: true });
  expect(await t.query(api.audit.alerts, {})).toHaveLength(0);
  await t.mutation(internal.audit.anchor, { source: "console", seq: 4, head: "bbb", ok: true });
  expect((await t.query(api.audit.alerts, {}))[0]).toMatchObject({ source: "console", kind: "fork" });
});

test("sources are tracked separately", async () => {
  const t = setup();
  await t.mutation(internal.audit.anchor, { source: "arm", seq: 9, head: "x", ok: true });
  await t.mutation(internal.audit.anchor, { source: "console", seq: 2, head: "y", ok: false });
  expect(await t.query(api.audit.alerts, {})).toHaveLength(0);
  expect(await t.query(api.audit.latest, {})).toMatchObject({ arm: { seq: 9 }, console: { seq: 2, ok: false } });
});

test("a proposal is updated in place as its status changes", async () => {
  const t = setup();
  const base = { id: "t1", fault: "joint3 lag", created: 1, simulated: true, steps: [{ type: "tool", name: "Read" }] };
  await t.mutation(internal.proposals.upsert, { ...base, status: "running" });
  await t.mutation(internal.proposals.upsert, { ...base, status: "pending", diff: "+x", changes: [{ key: "k", old: 1, new: 2 }] });
  const rows = await t.query(api.proposals.recent, {});
  expect(rows).toHaveLength(1);
  expect(rows[0]).toMatchObject({ id: "t1", status: "pending", diff: "+x", simulated: true });
});

// ── training episodes ───────────────────────────────────────────────────────

const goodLabels = {
  task: "hand the Castroviejo needle holder, backhand", caption: "The arm hands over the needle holder.", success: true,
  failureMode: null, segments: [], instrumentVisible: true, qualityFlags: [], privacyFlags: [], agreesWithRobot: true,
  model: "claude-opus-5-5",
};

async function recorded(t: ReturnType<typeof setup>, opts: { consent: boolean; withVideo: boolean }) {
  const blob = await t.run((ctx) => ctx.storage.store(new Blob(["x"])));
  return t.mutation(internal.episodes.record, {
    jobId: 1, kind: "handoff", request: "castroviejo, backhand", startedAt: 1000, endedAt: 9000, outcome: "done", error: null,
    videoId: opts.withVideo ? blob : null, framesId: opts.withVideo ? blob : null, keyframeIds: opts.withVideo ? [blob] : [],
    keyframeTimesS: opts.withVideo ? [0] : [], frameCount: 80, fps: 10, facesBlurred: 0, consent: opts.consent, simulated: true,
  });
}

test("recording is internal: the public api can read episodes but not write them", () => {
  // @ts-expect-error episodes.record is internal
  void api.episodes.record;
  // @ts-expect-error episodes.uploadUrl is internal
  void api.episodes.uploadUrl;
  expect(typeof api.episodes.list).toBeDefined();
});

test("only consented episodes with footage are sent to Claude", async () => {
  const t = setup();
  await recorded(t, { consent: false, withVideo: true });
  await recorded(t, { consent: true, withVideo: false });
  await recorded(t, { consent: true, withVideo: true });
  const statuses = (await t.query(api.episodes.list, {})).map((e) => e.labelStatus).sort();
  expect(statuses).toEqual(["pending", "skipped", "skipped"]);
});

test("clean labels are accepted automatically; the public list carries no video links", async () => {
  const t = setup();
  const id = await recorded(t, { consent: true, withVideo: true });
  await t.mutation(internal.episodes.saveLabels, { episodeId: id, labels: goodLabels });
  expect(await t.query(api.episodes.stats, {})).toMatchObject({ total: 1, labeled: 1, accepted: 1, needsReview: 0, frames: 80 });
  const [row] = await t.query(api.episodes.list, {});
  expect(row.caption).toBe(goodLabels.caption);
  expect(JSON.stringify(row)).not.toMatch(/https?:/);
  expect(await t.query(internal.episodes.exportManifest, {})).toHaveLength(1);
});

test("privacy or quality flags, or disagreeing with the robot, hold an episode for a person", async () => {
  const t = setup();
  for (const labels of [{ ...goodLabels, privacyFlags: ["face"] }, { ...goodLabels, qualityFlags: ["dark"] },
                        { ...goodLabels, agreesWithRobot: false }]) {
    const id = await recorded(t, { consent: true, withVideo: true });
    await t.mutation(internal.episodes.saveLabels, { episodeId: id, labels });
  }
  expect(await t.query(api.episodes.stats, {})).toMatchObject({ accepted: 0, needsReview: 3 });
  expect(await t.query(internal.episodes.exportManifest, {})).toHaveLength(0);
});

test("labelling runs through the Action Retrier component and the run id is kept on the episode", async () => {
  const t = setup();
  const id = await recorded(t, { consent: true, withVideo: true });
  const e = await t.run((ctx) => ctx.db.get(id));
  expect(e?.labelStatus).toBe("pending");
  expect(e?.labelRunId).toBeTruthy();
});

test("when the retrier gives up, only that episode is marked failed", async () => {
  const t = setup();
  const a = await recorded(t, { consent: true, withVideo: true });
  const b = await recorded(t, { consent: true, withVideo: true });
  const runA = (await t.run((ctx) => ctx.db.get(a)))!.labelRunId!;
  await t.mutation(internal.episodes.labelDone, { runId: runA as never, result: { type: "failed", error: "API 529: overloaded" } });
  const [ea, eb] = await t.run(async (ctx) => [await ctx.db.get(a), await ctx.db.get(b)]);
  expect(ea).toMatchObject({ labelStatus: "failed", labelError: "gave up after retries: API 529: overloaded" });
  expect(eb?.labelStatus).toBe("pending");
});

test("labels are searchable; the accepted-only filter hides episodes still under review", async () => {
  const t = setup();
  const good = await recorded(t, { consent: true, withVideo: true });
  await t.mutation(internal.episodes.saveLabels, { episodeId: good, labels: goodLabels });
  const flagged = await recorded(t, { consent: true, withVideo: true });
  await t.mutation(internal.episodes.saveLabels, {
    episodeId: flagged,
    labels: { ...goodLabels, caption: "The gripper closed on nothing.", success: false, failureMode: "missed_grasp", agreesWithRobot: false },
  });
  expect((await t.query(api.episodes.search, { q: "missed_grasp" })).map((e) => e.id)).toEqual([flagged]);
  expect(await t.query(api.episodes.search, { q: "missed_grasp", acceptedOnly: true })).toHaveLength(0);
  expect((await t.query(api.episodes.search, { q: "needle holder", acceptedOnly: true })).map((e) => e.id)).toEqual([good]);
  expect(await t.query(api.episodes.search, { q: "  " })).toEqual([]);
});

test("events older than a week are pruned; recent ones stay", async () => {
  const t = setup();
  const now = Date.now() / 1000;
  await t.mutation(internal.events.add, { kind: "utterance", data: {}, seq: 1, t: now - 8 * 24 * 3600 });
  await t.mutation(internal.events.add, { kind: "utterance", data: {}, seq: 2, t: now - 60 });
  expect(await t.mutation(internal.events.prune, {})).toBe(1);
  expect((await t.query(api.events.recent, {})).map((e) => e.seq)).toEqual([2]);
});

test("a proposal carries Droid's structured diagnosis", async () => {
  const t = setup();
  await t.mutation(internal.proposals.upsert, {
    id: "t2", fault: "joint3 lag", created: 1, status: "pending", cause: "speed too high", severity: "medium",
    safe_to_resume: true, evidence: [{ time_s: 12.4, joint: "joint3", observation: "lag 31 deg" }],
  });
  expect((await t.query(api.proposals.recent, {}))[0]).toMatchObject({ severity: "medium", safe_to_resume: true });
});
