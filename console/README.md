# scrub_console: the operating-room console for the scrub assistant

The arm picks and hands over instruments. This is the software around it: you talk to it, it understands you,
it tracks where every instrument is, it runs the end-of-case count, it keeps a tamper-evident record of what
happened, and when the arm faults it proposes a fix for a person to approve.

```
 mic ─► Cartesia Ink ─► voice agent ── Claude ──► intent ─┐
                                                          ▼
 dashboard ◄─ live queries ─ Convex ◄── bridge ──► arm agent ─► arm server (token, audit log) ─► arm
     ▲                                  │  ▲                                  │
     └──────── server-sent events ──────┘  └──── eyes agent ◄─ perception ────┘  (camera count)
                                           fault ─► Factory Droid (sandbox copy of arm.yaml) ─► diff ─► you approve
```

## Try it with no hardware

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "arm[kinematics]" -e "console[all]"
python -m scrub_console demo --agents          # simulated arm + console + dashboard, http://127.0.0.1:8780
```

Type (or say) `castroviejo, backhand`, then `got it`, then `done with the castro`. Use **Remove an instrument
(simulation)** then **Count check** to see the console name what's missing. **Simulate a fault** runs fault
triage. Everything on screen is labelled *Simulated arm*. With no keys set it still works: Claude is replaced by
a rule-based parser, voice is off (typing works), and Droid is a clearly labelled rehearsal.

The 8780 dashboard includes **Doctor view** and **Robot view**. Select a camera for each and start its preview
in the browser. Choose different devices for the two views. These previews stay local and are not recorded;
physical arm picking still uses the arm's own camera.

Copy `.env.example` to `.env` (git-ignored) at the repo root and fill in what you have.

## With the real arm

```bash
export SCRUB_API_TOKEN=$(openssl rand -hex 16)         # the arm server refuses to run off-loopback without one
python -m scrub_arm serve                              # terminal 1; writes logs/audit.jsonl
python -m scrub_console serve                          # terminal 2; http://127.0.0.1:8780
python -m scrub_console check                          # which services are configured, is the arm reachable
```

For the count check from the camera, run the perception service too (`python -m scrub_vision.service`); the
console asks its `POST /count`.

## Where each sponsor is used

Every row says what is real, how it was checked here, and what is **not** done. "Fake" means the vendor SDK's
real signatures were checked but the call went to a stand-in, because there was no key or account on the build
machine. None of the paid services have been exercised with a live key yet.

| Sponsor | What it does in the product | Code | Checked how | Not done / limits |
|---|---|---|---|---|
| **Cartesia** | Ink turns the surgeon's recorded speech into text; Sonic reads every reply aloud. The browser records the mic, so nothing is installed locally. | `speech.py`, mic code in `web/index.html` | Unit tests against a fake client plus a test that the real SDK accepts the argument names we send | Never run with a live key. Voice-to-voice latency unmeasured. |
| **Anthropic (Claude)** | Understands messy spoken requests, asks one short question when unsure, never moves the arm itself: every instrument it names must pass the arm's own `/resolve` check. `claude-opus-5-5`, strict tools, refusal fallbacks on. | `intent.py` | Tests with a fake client, including that a hallucinated instrument becomes a question | Never run live. **Named community still to fill in** (see below). |
| **Convex** | Live database behind the dashboard: events, the instrument board, audit-log heads, triage proposals, and the arm's 20 Hz joint readings (see *Arm readings* below). The page subscribes with live queries (the browser client is vendored, not loaded from a CDN). Writes are internal functions only the console can call. | `convex/`, `sinks.py`, `web/index.html` | Deployed to a cloud dev deployment and run end to end (handoff readings stored at 20 Hz, dashboard checked in a browser). 17 `convex-test` tests, a TypeScript check, a live test against a local Convex deployment (writes, public reads, public writes refused), and the dashboard run against it in jsdom | Dev deployment only, not a production one. No cleanup of old events (arm readings are pruned after a day). |
| **Armadin (security)** | The arm's API used to answer any website with `Access-Control-Allow-Origin: *` and had no auth, so a web page open on the same machine could command the arm. Now: bearer token, no wildcard CORS, Origin and Host checks, a hash-chained audit log (HMAC with `SCRUB_AUDIT_KEY`), and chain heads published to Convex so deleting the end of a log is noticed. | `arm/scrub_arm/server.py`, `audit.py`, Convex `audit.ts` | 17 arm tests, Convex rollback/fork tests, a live rollback alert | The demo tamper button edits a *copy* of the log, never the real one. Bearer token, not signed commands. |
| **Factory Droid** | On an arm fault, Droid reads the fault and a sandbox copy of `arm.yaml` and proposes the smallest tuning fix. It can only read and edit that one file; the result is diffed, checked against an allowlist of numeric tuning keys and bounds, and applied only if a person approves. Each step shows on the dashboard. | `triage.py` | 24 tests, including the permission handler built from the SDK's own types and the runner against a faked session | The Droid CLI is not installed on the build machine, so **no real Droid run has happened**. `--rehearse-droid` covers the UI and is labelled "rehearsal, no agent ran". |
| **Fetch.ai** | Voice, arm and eyes are separate uAgents. Every spoken request, arm command and camera count is an actual message between them, shown in the timeline. | `agents.py` | 7 tests with a real local Bureau, a real handoff through the agents, and the full dashboard run with `--agents` | Local only: not registered on Agentverse or the Almanac, no ASI:One chat protocol. |
| **Bitdeer** | Writes the end-of-case report and suggests the next instrument on Model Studio. Each result says what produced it. | `report.py` | Tests against a fake OpenAI-compatible server | Never run live; the default model name may not exist on your account (`BITDEER_MODEL`). Falls back to a plain template. |
| **Microsoft** | The production-minded framing: who it is for, how it fails, what it would take. | `docs/PRODUCTION.md` | Written from what the code does | No measurements yet. |
| **Mana, Figma, Notability** | Not code. | | | The team needs to link their Mana chats/prototype, Figma frames, and Notability notes in the submission. |

Not attempted: Moss, Spacetime (Convex is used instead), Elastic.

### Who it is for (Anthropic track: needs a real answer)

Placeholder from the brief, not validated: surgical technology students and new scrub techs practising
instrument handling and counts. **Replace this with a named community and a real person's account of the
problem before submitting.** Count discrepancies matter because a missed instrument is a patient-safety event.

## Safety choices worth knowing

- The console never moves the arm on Claude's say-so alone: the arm's catalog validates every request.
- Droid never touches the repo, only a throwaway copy; its edit is validated and needs a human approval.
- Joint limits, gripper calibration, motor IDs and poses can never change through triage; the collision guard
  can be loosened within bounds but never switched off.
- Everything listens on loopback only; other websites cannot drive the console or the arm.
- Simulated results (the simulated count, rehearsal Droid) are labelled wherever they appear.

## Tests

```bash
cd arm && python -m pytest tests/test_audit.py tests/test_server_security.py     # arm hardening
cd console && python -m pytest                                                    # 137 tests (+3 opt-in live Convex)
cd console && npm install && npm test && npm run typecheck                        # Convex functions
cd perception && python -m pytest tests/test_service.py                           # /count
# live Convex (opt in): CONVEX_AGENT_MODE=anonymous npx convex dev, then
# CONVEX_LIVE_URL=http://127.0.0.1:3210 CONVEX_LIVE_KEY=<adminKey in .convex/local/default/config.json> python -m pytest tests/test_convex_live.py
```

The dashboard's JavaScript was exercised in jsdom against a live console and simulated arm; nobody has looked
at it in a real browser yet, so check the layout and the microphone there first.

## Arm readings (metrics and logs)

The arm's control loop (200 Hz) keeps a 20 Hz buffer of joint angles, setpoints, torque and the gripper angle.
The console drains it on every poll (`GET /status?since=`, same token as every other call), stamps each reading
with the phase, job and fault, and writes one batch per ~5 s to Convex (`metrics`, raw) along with a
one-point-per-second rollup (`metricPoints`). Each point keeps the *worst* lag and |torque| in its second, so a contact spike still shows on a chart.
Raw and rolled-up readings are pruned after a day (`metrics:prune`, hourly cron).

The dashboard's **Arm readings** section loads the rolled-up history once (10/30/60 min) and then follows only
the newest batch. It shows lag against the collision guard, torque and gripper angle, plus summary tiles and a
per-joint table. A button downloads the last 10 minutes of raw 20 Hz readings as CSV. The **Logs** list there
filters the stored events by kind (`events:byKind`). Readings are kept out of the browser's event stream; they only go to Convex.

The console writes to Convex over its HTTP API (`ConvexHTTP` in `sinks.py`), not the official Python client.
The official client holds the GIL for a whole round trip (~110 ms to the cloud). That froze every thread in the console,
and in the demo it put holes in the simulated arm's 20 Hz readings.

Not done: motor temperature (not read from the bus), longer-term history, per-job tracing, threshold alerts.
The simulated arm reports zero lag and torque, so those charts are only flat lines until a real arm is connected.

## Known gaps

- No live-key run of Cartesia, Claude, Bitdeer or Droid.
- The camera count has only been tested on rendered scenes, not real photos.
- The simulated count assumes the table matches the console's records except what was removed.
- The arm's `pick.mode: ik` handoff needs the perception service; the demo uses slot mode instead.
