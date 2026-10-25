# Hackathon log

- **Project:** Scrub, a robotic surgical scrub assistant
- **Event:** Cal Hacks 13.0
- **What it does:** A robot arm picks surgical instruments by wrist camera and hands them over on voice or gesture requests, with a Convex-backed console that tracks instruments and logs every event.
- **Live app:** not deployed
- **Repo:** https://github.com/Ishxn20/CalHacks
- **Frontend:** not deployed (served locally by the Python console)
- **Convex deployment:** not deployed
- **Components:** none
- **Convex features:** schema, indexes, queries, mutations, actions, HTTP actions, crons, scheduled functions, file storage, realtime subscriptions (`ConvexClient.onUpdate`)
- **Auth:** none
- **AI models:** claude-opus-5-5, claude-sonnet-5-5, gpt-6.1-sol, gpt-6-luna, Whisper base.en / small.en, yolo11n-seg
- **Started:** 2026-10-24T03:15:29Z
- **Last updated:** 2026-10-25T16:09:03Z

## How we use Convex

- **Instrument board:** each instrument's status (tray, held, in field, used) lives in the `instruments` table. Proof: `console/convex/instruments.ts`.
- **Event timeline and logs:** every utterance, command, job and fault is saved by a mutation and read by indexed queries. Proof: `console/convex/events.ts`.
- **Live arm charts:** 20 Hz readings are stored in batches, rolled up per second, and pruned by an hourly cron. Proof: `console/convex/metrics.ts`, `console/convex/crons.ts`.
- **Training episodes:** blurred wrist video goes to file storage. A scheduled action has Claude label each episode, and a cron retries failures. Proof: `console/convex/episodes.ts`, `console/convex/labeling.ts`.
- **Tamper-evident audit:** arm and console audit-log heads are anchored; a mutation records rollback or fork alerts. Proof: `console/convex/audit.ts`.
- **Dataset export:** the `/export/manifest` HTTP action returns accepted episodes with storage URLs behind an export key. Proof: `console/convex/http.ts`.

**Updates live without a refresh:** on the **Live** tab, the instrument board, event timeline and latest arm readings. On the **Robot data** tab, the log view. All use `onUpdate` in `console/scrub_console/web/index.html`.

## Log

All dates are UTC.

- 10-24 `fcd2681a` reBot B601-RS arm control (arm/)
- 10-24 `9388e9a1` Whisper voice agent (voice/)
- 10-24 `3d53257b` FK/IK and top-down grasp solver (arm/scrub_arm)
- 10-24 `f60bace3` IK pick with contact-sensing descent (arm/scrub_arm)
- 10-24 `6a27a99f` instrument table and STL templates (perception/)
- 10-24 `26de1591` synthetic YOLO-seg dataset (perception/scrub_vision)
- 10-24 `a90a60b3` /plan /verify vision HTTP service (perception/scrub_vision)
- 10-24 `cd489d36` MuJoCo pile simulator (sim/)
- 10-24 `3af36412` declutter stress test (sim/, perception/)
- 10-25 `95d63084` video predictor for scissors hand-over (predict/watch.py)
- 10-25 `abce74e3` real-arm bring-up and camera-guided picking (arm/, perception/)
- 10-25 `62fec978` identify our 8 red parts by shape (perception/)
- 10-25 `fe893686` per-tool gripper widths (arm/, config/tray.yaml)
- 10-25 `0a7c4980` pick.py: type a tool, arm picks it (tools/pick.py)
- 10-25 `45ef9835` camera-watched hand-off (tools/)
- 10-25 `5ae25bca` camera gesture demo (gesture/)
- 10-25 `fc206e55` console and Convex backend (console/convex, console/scrub_console)
- 10-25 `212d38a2` locked-down arm API and audit log (arm/scrub_arm)
- 10-25 `e2469425` POST /count instrument count (perception/scrub_vision)
- 10-25 `37afa3a3` 20 Hz readings for live charts (arm/scrub_arm)
- 10-25 `2aabd10b` recalibration after the camera was knocked (config/, perception/, tools/)
- 10-25 `3399bfbc` two-tab dashboard with live feeds (console/scrub_console)
- 10-25 `63bcbe1d` server picks and hands off by wrist camera (arm/, tools/)
- 10-25 `4c42a221` integration tests (console/scrub_console, tools/demo.py)
- 10-25 `63f38c9f` better error messages (arm/scrub_arm)
- 10-25 `4c246931` hackathon log (hackathon.md, README.md)
- 10-25 `18e63204` full technical README (README.md)
