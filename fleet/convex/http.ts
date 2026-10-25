// Robot-agent API. Every request carries `Authorization: Bearer <robot token>`.
//
//   POST /robot/heartbeat        {agentVersion, enabled, fault, job, samples[]}  -> {recording}
//   POST /robot/commands         {}                                             -> {commands[]}
//   POST /robot/command-result   {commandId, ok, result}
//   POST /robot/episodes/start   {jobId, kind, request, startedAt}             -> {episodeId, consent}
//   POST /robot/upload-url       {}                                             -> {url}  (then POST the file to it)
//   POST /robot/episodes/finish  {episodeId, endedAt, outcome, ..., videoId, framesId, keyframeIds}
//   POST /robot/incidents        {fault, phase, context}                        -> {incidentId}
//   GET  /export/manifest        Authorization: Bearer <EXPORT_KEY>             -> accepted training episodes
import { httpRouter } from "convex/server";
import { internal } from "./_generated/api";
import type { Id } from "./_generated/dataModel";
import { type ActionCtx, httpAction } from "./_generated/server";
import { sha256Hex } from "./lib";

const http = httpRouter();

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function bearer(req: Request): string | null {
  const h = req.headers.get("Authorization") ?? "";
  return h.startsWith("Bearer ") ? h.slice(7).trim() : null;
}

async function robotFor(ctx: ActionCtx, req: Request): Promise<Id<"robots"> | null> {
  const token = bearer(req);
  if (!token) return null;
  const robot = await ctx.runQuery(internal.robots.byTokenHash, { tokenHash: await sha256Hex(token) });
  return robot?._id ?? null;
}

type Handler = (ctx: ActionCtx, robotId: Id<"robots">, body: any) => Promise<unknown>;

function robotRoute(path: string, handler: Handler) {
  http.route({
    path,
    method: "POST",
    handler: httpAction(async (ctx, req) => {
      const robotId = await robotFor(ctx, req);
      if (!robotId) return json({ error: "unknown or missing robot token" }, 401);
      let body: unknown;
      try {
        body = await req.json();
      } catch {
        return json({ error: "body must be JSON" }, 400);
      }
      try {
        return json((await handler(ctx, robotId, body)) ?? { ok: true });
      } catch (e) {
        return json({ error: e instanceof Error ? e.message : String(e) }, 400);
      }
    }),
  });
}

robotRoute("/robot/heartbeat", (ctx, robotId, b) =>
  ctx.runMutation(internal.ingest.heartbeat, {
    robotId,
    agentVersion: String(b.agentVersion ?? "unknown"),
    enabled: Boolean(b.enabled),
    fault: b.fault ?? null,
    job: b.job ?? null,
    samples: b.samples ?? [],
  }),
);

robotRoute("/robot/commands", async (ctx, robotId) => ({
  commands: await ctx.runMutation(internal.ingest.takeCommands, { robotId }),
}));

robotRoute("/robot/command-result", (ctx, robotId, b) =>
  ctx.runMutation(internal.ingest.commandResult, {
    robotId,
    commandId: b.commandId,
    ok: Boolean(b.ok),
    result: String(b.result ?? ""),
  }),
);

robotRoute("/robot/episodes/start", async (ctx, robotId, b) => {
  const robots = await ctx.runQuery(internal.robots.byId, { robotId });
  const consent = robots?.recording ?? false;
  const episodeId = await ctx.runMutation(internal.ingest.startEpisode, {
    robotId,
    jobId: Number(b.jobId),
    kind: String(b.kind),
    request: b.request ?? null,
    startedAt: Number(b.startedAt),
    consent,
  });
  return { episodeId, consent };
});

robotRoute("/robot/upload-url", async (ctx) => ({ url: await ctx.runMutation(internal.ingest.uploadUrl, {}) }));

robotRoute("/robot/episodes/finish", (ctx, robotId, b) =>
  ctx.runMutation(internal.ingest.finishEpisode, {
    robotId,
    episodeId: b.episodeId,
    endedAt: Number(b.endedAt),
    outcome: String(b.outcome),
    error: b.error ?? null,
    videoId: b.videoId ?? null,
    framesId: b.framesId ?? null,
    keyframeIds: b.keyframeIds ?? [],
    keyframeTimesS: b.keyframeTimesS ?? [],
    frameCount: Number(b.frameCount ?? 0),
    fps: Number(b.fps ?? 0),
    facesBlurred: Number(b.facesBlurred ?? 0),
  }),
);

robotRoute("/robot/incidents", async (ctx, robotId, b) => ({
  incidentId: await ctx.runMutation(internal.ingest.openIncident, {
    robotId,
    fault: String(b.fault),
    phase: String(b.phase ?? "unknown"),
    context: typeof b.context === "string" ? b.context : JSON.stringify(b.context ?? {}),
  }),
}));

http.route({
  path: "/export/manifest",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const expected = process.env.EXPORT_KEY;
    if (!expected || bearer(req) !== expected) return json({ error: "export key required" }, 401);
    return json(await ctx.runQuery(internal.episodes.exportManifest, {}));
  }),
});

export default http;
