import { convexTest } from "convex-test";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { api, internal } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import schema from "../convex/schema";

const modules = import.meta.glob("../convex/**/*.ts");
const OP = "op-secret";
const WORKER = "worker-secret";

beforeEach(() => {
  process.env.OPERATOR_KEY = OP;
  process.env.WORKER_KEY = WORKER;
  process.env.EXPORT_KEY = "export-secret";
});
afterEach(() => {
  delete process.env.OPERATOR_KEY;
  delete process.env.WORKER_KEY;
  delete process.env.EXPORT_KEY;
});

async function setup() {
  const t = convexTest(schema, modules);
  const { robotId, token } = await t.action(internal.robots.register, { name: "OR-3 arm", site: "test" });
  const post = (path: string, body: unknown, tok: string | null = token) =>
    t.fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(tok ? { Authorization: `Bearer ${tok}` } : {}) },
      body: JSON.stringify(body),
    });
  return { t, robotId: robotId as Id<"robots">, token, post };
}

const sample = (phase = "idle") => ({
  t: Date.now(),
  phase,
  qDeg: [0, 20, 40, 0, 0, 0],
  qCmdDeg: [0, 20, 45, 0, 0, 0],
  gripperDeg: 259,
});

describe("robot API", () => {
  test("rejects a missing or wrong token", async () => {
    const { post } = await setup();
    expect((await post("/robot/heartbeat", {}, null)).status).toBe(401);
    expect((await post("/robot/heartbeat", {}, "rbt_nope")).status).toBe(401);
  });

  test("heartbeat updates the robot and stores telemetry with tracking error", async () => {
    const { t, post } = await setup();
    const r = await post("/robot/heartbeat", {
      agentVersion: "0.1",
      enabled: true,
      fault: null,
      job: null,
      samples: [sample("moving")],
    });
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ recording: true });
    const [robot] = await t.query(api.robots.list, {});
    expect(robot.online).toBe(true);
    expect(robot.phase).toBe("moving");
    expect("tokenHash" in robot).toBe(false); // never sent to the dashboard
    const tele = await t.query(api.telemetry.recent, { robotId: robot._id });
    expect(tele[0].trackingErrMaxDeg).toBe(5);
  });

  test("the token is stored only as a hash", async () => {
    const { t, token } = await setup();
    const rows = await t.run((ctx) => ctx.db.query("robots").collect());
    expect(rows[0].tokenHash).not.toContain(token);
    expect(rows[0].tokenHash).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe("commands", () => {
  test("moving an arm needs the operator key", async () => {
    const { t, robotId } = await setup();
    await expect(
      t.mutation(api.commands.enqueue, { operatorKey: "wrong", robotId, kind: "handoff", text: "scalpel" }),
    ).rejects.toThrow(/operator key/);
    delete process.env.OPERATOR_KEY;
    await expect(
      t.mutation(api.commands.enqueue, { operatorKey: OP, robotId, kind: "home" }),
    ).rejects.toThrow(/OPERATOR_KEY is not set/);
  });

  test("queued commands are delivered once, oldest first", async () => {
    const { t, robotId, post } = await setup();
    await t.mutation(api.commands.enqueue, { operatorKey: OP, robotId, kind: "handoff", text: "scalpel" });
    await t.mutation(api.commands.enqueue, { operatorKey: OP, robotId, kind: "home" });
    const first = await (await post("/robot/commands", {})).json();
    expect(first.commands.map((c: { kind: string }) => c.kind)).toEqual(["handoff", "home"]);
    expect((await (await post("/robot/commands", {})).json()).commands).toEqual([]);
  });

  test("a handoff needs a request", async () => {
    const { t, robotId } = await setup();
    await expect(t.mutation(api.commands.enqueue, { operatorKey: OP, robotId, kind: "handoff" })).rejects.toThrow(
      /needs a request/,
    );
  });
});

describe("incidents and Droid proposals", () => {
  async function proposed(change: { key: string; from: number | null; to: number; rationale: string } | null) {
    const s = await setup();
    const { incidentId } = await (
      await s.post("/robot/incidents", { fault: "joint3 lagged 31 deg", phase: "moving", context: { x: 1 } })
    ).json();
    expect(await s.t.mutation(api.incidents.claim, { workerKey: WORKER, incidentId })).toBe(true);
    expect(await s.t.mutation(api.incidents.claim, { workerKey: WORKER, incidentId })).toBe(false);
    await s.t.mutation(api.incidents.propose, {
      workerKey: WORKER,
      incidentId,
      diagnosis: "speed too high under load",
      severity: "medium",
      safeToResume: true,
      change,
    });
    return { ...s, incidentId: incidentId as Id<"incidents"> };
  }

  test("a re-reported fault does not open a duplicate incident", async () => {
    const { t, post } = await setup();
    const a = await (await post("/robot/incidents", { fault: "f", phase: "p", context: {} })).json();
    const b = await (await post("/robot/incidents", { fault: "f", phase: "p", context: {} })).json();
    expect(a.incidentId).toBe(b.incidentId);
    expect(await t.query(api.incidents.list, {})).toHaveLength(1);
  });

  test("out-of-bounds changes are dropped before a person sees them", async () => {
    const { t } = await proposed({ key: "motion.speed_scale", from: 0.5, to: 2.0, rationale: "go faster" });
    const [inc] = await t.query(api.incidents.list, {});
    expect(inc.change).toBeNull();
    expect(inc.diagnosis).toMatch(/dropped/);
  });

  test("keys outside the allowlist are dropped", async () => {
    const { t } = await proposed({ key: "joints.0.limits_deg", from: null, to: 170, rationale: "more range" });
    const [inc] = await t.query(api.incidents.list, {});
    expect(inc.change).toBeNull();
  });

  test("approval queues apply_config; the robot's success resolves the incident", async () => {
    const { t, incidentId, post } = await proposed({
      key: "motion.speed_scale",
      from: 0.5,
      to: 0.35,
      rationale: "joint3 lag at speed",
    });
    await expect(t.mutation(api.incidents.decide, { operatorKey: "nope", incidentId, approve: true })).rejects.toThrow();
    await t.mutation(api.incidents.decide, { operatorKey: OP, incidentId, approve: true });
    const { commands } = await (await post("/robot/commands", {})).json();
    expect(commands).toHaveLength(1);
    expect(commands[0].kind).toBe("apply_config");
    expect(commands[0].change.to).toBe(0.35);
    await post("/robot/command-result", { commandId: commands[0].id, ok: true, result: "applied" });
    const [inc] = await t.query(api.incidents.list, {});
    expect(inc.status).toBe("resolved");
  });

  test("the worker can't touch commands or decisions", async () => {
    const { t, incidentId } = await proposed(null);
    await expect(
      t.mutation(api.incidents.decide, { operatorKey: WORKER, incidentId, approve: true }),
    ).rejects.toThrow(/operator key/);
  });
});

describe("episodes and the dataset gate", () => {
  async function episode(opts: { recording: boolean; withVideo: boolean }) {
    const s = await setup();
    if (!opts.recording) {
      await s.t.mutation(api.robots.setRecording, { operatorKey: OP, robotId: s.robotId, recording: false });
    }
    const start = await (
      await s.post("/robot/episodes/start", { jobId: 1, kind: "handoff", request: "scalpel", startedAt: 1000 })
    ).json();
    expect(start.consent).toBe(opts.recording);
    const blob = opts.withVideo
      ? await s.t.run((ctx) => ctx.storage.store(new Blob(["x"], { type: "image/jpeg" })))
      : null;
    await s.post("/robot/episodes/finish", {
      episodeId: start.episodeId,
      endedAt: 9000,
      outcome: "done",
      videoId: blob,
      framesId: blob,
      keyframeIds: blob ? [blob] : [],
      keyframeTimesS: blob ? [0.5] : [],
      frameCount: 80,
      fps: 10,
    });
    return { ...s, episodeId: start.episodeId as Id<"episodes"> };
  }

  test("without consent an episode is never sent for labelling", async () => {
    const { t } = await episode({ recording: false, withVideo: true });
    const [e] = await t.query(api.episodes.list, {});
    expect(e.labelStatus).toBe("skipped");
  });

  test("with consent and footage it is queued for labelling", async () => {
    const { t } = await episode({ recording: true, withVideo: true });
    const [e] = await t.query(api.episodes.list, {});
    expect(e.labelStatus).toBe("pending");
  });

  const goodLabels = {
    task: "hand the scalpel",
    caption: "The arm picks a scalpel and presents it.",
    success: true,
    failureMode: null,
    segments: [{ startS: 0, endS: 8, phase: "picking", description: "picks" }],
    instrumentVisible: true,
    qualityFlags: [],
    privacyFlags: [],
    agreesWithRobot: true,
    model: "claude-opus-5-5",
  };

  test("clean, agreeing labels are accepted automatically and exported", async () => {
    const { t, episodeId } = await episode({ recording: true, withVideo: true });
    await t.mutation(internal.episodes.saveLabels, { episodeId, labels: goodLabels });
    const stats = await t.query(api.episodes.stats, {});
    expect(stats.accepted).toBe(1);
    const r = await t.fetch("/export/manifest", { headers: { Authorization: "Bearer export-secret" } });
    expect((await r.json())).toHaveLength(1);
    expect((await t.fetch("/export/manifest")).status).toBe(401);
  });

  test("privacy flags hold an episode for review and block acceptance", async () => {
    const { t, episodeId } = await episode({ recording: true, withVideo: true });
    await t.mutation(internal.episodes.saveLabels, {
      episodeId,
      labels: { ...goodLabels, privacyFlags: ["face"] },
    });
    expect((await t.query(api.episodes.stats, {})).needsReview).toBe(1);
    await expect(t.mutation(api.episodes.review, { operatorKey: OP, episodeId, accept: true })).rejects.toThrow(
      /privacy/,
    );
  });

  test("a label that disagrees with the robot waits for a person", async () => {
    const { t, episodeId } = await episode({ recording: true, withVideo: true });
    await t.mutation(internal.episodes.saveLabels, {
      episodeId,
      labels: { ...goodLabels, agreesWithRobot: false },
    });
    const [e] = await t.query(api.episodes.list, {});
    expect(e.review).toBe("unreviewed");
  });
});
