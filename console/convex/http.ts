// GET /export/manifest  Authorization: Bearer <EXPORT_KEY>  ->  accepted training episodes with download URLs.
// The only way footage leaves Convex; fails closed when EXPORT_KEY isn't set on the deployment.
import { httpRouter } from "convex/server";
import { internal } from "./_generated/api";
import { httpAction } from "./_generated/server";

const http = httpRouter();

function sameSecret(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

http.route({
  path: "/export/manifest",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const expected = process.env.EXPORT_KEY;
    const got = (req.headers.get("Authorization") ?? "").replace(/^Bearer /, "");
    if (!expected || !sameSecret(got, expected)) {
      return new Response(JSON.stringify({ error: "export key required" }), { status: 401 });
    }
    return new Response(JSON.stringify(await ctx.runQuery(internal.episodes.exportManifest, {})), {
      headers: { "Content-Type": "application/json" },
    });
  }),
});

export default http;
