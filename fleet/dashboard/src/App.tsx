import { useMutation, useQuery } from "convex/react";
import type { FunctionReturnType } from "convex/server";
import { useEffect, useState } from "react";
import { api } from "../../convex/_generated/api";
import type { Doc, Id } from "../../convex/_generated/dataModel";

type Robot = FunctionReturnType<typeof api.robots.list>[number];
const GUARD_DEG = 25; // arm.yaml motion.tracking_error_limit_deg: lag past this freezes the arm

function useOperatorKey(): [string, (k: string) => void] {
  const [key, setKey] = useState(() => {
    try {
      return sessionStorage.getItem("operatorKey") ?? "";
    } catch {
      return "";
    }
  });
  const save = (k: string) => {
    setKey(k);
    try {
      sessionStorage.setItem("operatorKey", k);
    } catch {
      /* private window: keep it in memory only */
    }
  };
  return [key, save];
}

function ago(t: number | null): string {
  if (t === null) return "never";
  const s = Math.round((Date.now() - t) / 1000);
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
}

export default function App() {
  const robots = useQuery(api.robots.list);
  const [selected, setSelected] = useState<Id<"robots"> | null>(null);
  const [operatorKey, setOperatorKey] = useOperatorKey();
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (!selected && robots?.length) setSelected(robots[0]._id);
  }, [robots, selected]);
  const robot = robots?.find((r) => r._id === selected) ?? null;

  return (
    <div className="shell">
      <header className="top">
        <h1>Scrub fleet</h1>
        <label className="key">
          <span>Operator key</span>
          <input
            type="password"
            value={operatorKey}
            onChange={(e) => setOperatorKey(e.target.value)}
            placeholder="needed to move arms or approve fixes"
            autoComplete="off"
          />
        </label>
      </header>
      <aside className="fleet" aria-label="Arms">
        {robots === undefined && <p className="quiet">Connecting…</p>}
        {robots?.length === 0 && (
          <p className="quiet">
            No arms yet. Register one with <code>npx convex run robots:register</code> and start its agent.
          </p>
        )}
        <ul>
          {robots?.map((r) => (
            <li key={r._id}>
              <button className={`arm-row ${r._id === selected ? "on" : ""}`} onClick={() => setSelected(r._id)}>
                <span className={`lamp ${r.fault ? "fault" : r.online ? "ok" : "off"}`} aria-hidden />
                <span className="arm-name">{r.name}</span>
                <span className="arm-site">{r.site}</span>
                <span className="arm-state">
                  {!r.online ? `offline, seen ${ago(r.lastSeen)}` : r.fault ? "faulted" : r.job ?? r.phase}
                </span>
                {r.openIncidents > 0 && <span className="badge">{r.openIncidents}</span>}
              </button>
            </li>
          ))}
        </ul>
        <Dataset operatorKey={operatorKey} />
      </aside>
      <main className="detail">{robot ? <ArmDetail robot={robot} operatorKey={operatorKey} /> : null}</main>
    </div>
  );
}

function ArmDetail({ robot, operatorKey }: { robot: Robot; operatorKey: string }) {
  const telemetry = useQuery(api.telemetry.recent, { robotId: robot._id, seconds: 30 });
  const last = telemetry?.at(-1);
  return (
    <>
      <section className="arm-head">
        <div>
          <h2>{robot.name}</h2>
          <p className="quiet">
            {robot.site}. {robot.online ? `Online, agent ${robot.agentVersion}.` : `Offline since ${ago(robot.lastSeen)}.`}
          </p>
        </div>
        <Recording robot={robot} operatorKey={operatorKey} />
      </section>
      {robot.fault && <p className="fault-line">Stopped: {robot.fault}</p>}
      <section aria-label="Joints">
        <h3>Joints, measured against commanded</h3>
        <Joints q={last?.qDeg ?? robot.qDeg} cmd={last?.qCmdDeg ?? []} />
        <p className="quiet small">
          Shading is the lag between where a joint is and where it was told to be. Past {GUARD_DEG}° the collision guard
          freezes the arm. Gripper {robot.gripperDeg === null ? "unknown" : `${robot.gripperDeg.toFixed(0)}°`}.
        </p>
      </section>
      <Commands robot={robot} operatorKey={operatorKey} />
      <Incidents robotId={robot._id} operatorKey={operatorKey} />
      <Episodes robotId={robot._id} operatorKey={operatorKey} />
    </>
  );
}

function Joints({ q, cmd }: { q: (number | null)[]; cmd: (number | null)[] }) {
  const span = 180;
  const x = (deg: number) => 50 + (Math.max(-span, Math.min(span, deg)) / span) * 50;
  return (
    <ol className="joints">
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const m = q[i] ?? null;
        const c = cmd[i] ?? null;
        const lag = m !== null && c !== null ? Math.abs(m - c) : null;
        const level = lag === null ? "" : lag > GUARD_DEG * 0.6 ? "hot" : lag > 3 ? "warm" : "";
        return (
          <li key={i} className={level}>
            <span className="jn">J{i + 1}</span>
            <span className="track">
              {m !== null && c !== null && (
                <span className="lag" style={{ left: `${Math.min(x(m), x(c))}%`, width: `${Math.abs(x(m) - x(c))}%` }} />
              )}
              {c !== null && <span className="cmd" style={{ left: `${x(c)}%` }} title="commanded" />}
              {m !== null && <span className="meas" style={{ left: `${x(m)}%` }} title="measured" />}
            </span>
            <span className="num">{m === null ? "—" : `${m.toFixed(1)}°`}</span>
            <span className="num dim">{lag === null ? "" : `Δ ${lag.toFixed(1)}°`}</span>
          </li>
        );
      })}
    </ol>
  );
}

function Recording({ robot, operatorKey }: { robot: Robot; operatorKey: string }) {
  const set = useMutation(api.robots.setRecording);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="recording">
      <button
        className={robot.recording ? "rec on" : "rec"}
        onClick={() =>
          set({ operatorKey, robotId: robot._id, recording: !robot.recording }).then(
            () => setErr(null),
            (e: Error) => setErr(e.message),
          )
        }
      >
        {robot.recording ? "Recording for research" : "Recording paused"}
      </button>
      {err && <span className="err">{err}</span>}
    </div>
  );
}

function Commands({ robot, operatorKey }: { robot: Robot; operatorKey: string }) {
  const enqueue = useMutation(api.commands.enqueue);
  const recent = useQuery(api.commands.recent, { robotId: robot._id });
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const send = (kind: "handoff" | "stage" | "release" | "return" | "home" | "stop" | "reset", t?: string) =>
    enqueue({ operatorKey, robotId: robot._id, kind, text: t }).then(
      () => {
        setErr(null);
        if (t) setText("");
      },
      (e: Error) => setErr(e.message),
    );
  return (
    <section aria-label="Commands">
      <h3>Ask for an instrument</h3>
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          send("handoff", text);
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="needle holder, 6-0 prolene, backhand" />
        <button type="submit" disabled={!text.trim()}>
          Hand it over
        </button>
        <button type="button" disabled={!text.trim()} onClick={() => send("stage", text)}>
          Stage it
        </button>
      </form>
      <div className="row">
        <button onClick={() => send("release")}>Release</button>
        <button onClick={() => send("return")}>Put back</button>
        <button onClick={() => send("home")}>Go home</button>
        <button onClick={() => send("reset")}>Clear stop</button>
        <button className="stop" onClick={() => send("stop")}>
          Stop arm
        </button>
      </div>
      {err && <p className="err">{err}</p>}
      {recent && recent.length > 0 && (
        <ul className="log">
          {recent.slice(0, 5).map((c) => (
            <li key={c._id}>
              <span className={`st ${c.status}`}>{c.status}</span> {c.kind}
              {c.text ? ` “${c.text}”` : ""}
              {c.change ? ` ${c.change.key} → ${c.change.to}` : ""}
              {c.result && c.status !== "sent" ? <span className="quiet"> {c.result.slice(0, 120)}</span> : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Incidents({ robotId, operatorKey }: { robotId: Id<"robots">; operatorKey: string }) {
  const incidents = useQuery(api.incidents.list, { robotId });
  const decide = useMutation(api.incidents.decide);
  const [err, setErr] = useState<string | null>(null);
  if (!incidents?.length) return null;
  return (
    <section aria-label="Incidents">
      <h3>Faults and Droid diagnoses</h3>
      {incidents.slice(0, 3).map((inc) => (
        <article key={inc._id} className={`incident ${inc.status}`}>
          <p className="inc-fault">
            {inc.fault} <span className="quiet">({ago(inc.openedAt)}, {inc.status})</span>
          </p>
          {inc.steps.length > 0 && (
            <details open={inc.status === "diagnosing"}>
              <summary>What Droid did ({inc.steps.length} steps)</summary>
              <ol className="steps">
                {inc.steps.map((s, i) => (
                  <li key={i} className={s.kind}>
                    <span className="sk">{s.kind}</span> {s.text}
                  </li>
                ))}
              </ol>
            </details>
          )}
          {inc.diagnosis && <p className="diag">{inc.diagnosis}</p>}
          {inc.severity && (
            <p className="quiet small">
              Severity {inc.severity}. {inc.safeToResume ? "Safe to resume after a reset." : "Needs a person to inspect the arm."}
            </p>
          )}
          {inc.change && (
            <p className="change">
              Proposed: <strong>{inc.change.key}</strong> {inc.change.from ?? "?"} → {inc.change.to}. {inc.change.rationale}
            </p>
          )}
          {inc.status === "proposed" && (
            <div className="row">
              <button
                className="approve"
                onClick={() => decide({ operatorKey, incidentId: inc._id, approve: true }).then(() => setErr(null), (e: Error) => setErr(e.message))}
              >
                {inc.change ? "Approve and send to arm" : "Mark resolved"}
              </button>
              <button onClick={() => decide({ operatorKey, incidentId: inc._id, approve: false }).then(() => setErr(null), (e: Error) => setErr(e.message))}>
                Reject
              </button>
            </div>
          )}
        </article>
      ))}
      {err && <p className="err">{err}</p>}
    </section>
  );
}

function Episodes({ robotId, operatorKey }: { robotId: Id<"robots">; operatorKey: string }) {
  const episodes = useQuery(api.episodes.list, { robotId, limit: 8 });
  const review = useMutation(api.episodes.review);
  const [err, setErr] = useState<string | null>(null);
  if (!episodes?.length) return null;
  return (
    <section aria-label="Episodes">
      <h3>Recorded episodes</h3>
      <ul className="episodes">
        {episodes.map((e) => (
          <li key={e._id}>
            {e.thumbUrl ? <img src={e.thumbUrl} alt="" /> : <span className="nothumb">no video</span>}
            <div>
              <p>
                <strong>{e.request ?? e.kind}</strong> <span className="quiet">{e.outcome ?? "running"}</span>
              </p>
              <EpisodeLabel e={e} />
              {e.review === "unreviewed" && e.labelStatus === "done" && (
                <div className="row">
                  <button onClick={() => review({ operatorKey, episodeId: e._id, accept: true }).then(() => setErr(null), (x: Error) => setErr(x.message))}>
                    Add to dataset
                  </button>
                  <button onClick={() => review({ operatorKey, episodeId: e._id, accept: false }).then(() => setErr(null), (x: Error) => setErr(x.message))}>
                    Leave out
                  </button>
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
      {err && <p className="err">{err}</p>}
    </section>
  );
}

function EpisodeLabel({ e }: { e: Doc<"episodes"> }) {
  if (!e.consent) return <p className="quiet small">Not recorded: research recording was paused.</p>;
  if (e.labelStatus === "waiting") return <p className="quiet small">Recording…</p>;
  if (e.labelStatus === "pending" || e.labelStatus === "labeling") return <p className="quiet small">Claude is labelling…</p>;
  if (e.labelStatus === "failed" || e.labelStatus === "skipped")
    return <p className="quiet small">Not labelled{e.labelError ? `: ${e.labelError}` : ""}.</p>;
  const l = e.labels!;
  const flags = [...l.privacyFlags.map((f) => `privacy: ${f}`), ...l.qualityFlags, ...(l.agreesWithRobot ? [] : ["disagrees with robot"])];
  return (
    <>
      <p className="caption">{l.caption}</p>
      <p className="small">
        <span className={l.success ? "ok" : "bad"}>{l.success ? "Success" : `Failed: ${l.failureMode ?? "unknown"}`}</span>
        {e.review === "accepted" ? ", in the dataset" : e.review === "rejected" ? ", left out" : ", waiting for review"}
        {flags.length > 0 && <span className="flags">. Flagged: {flags.join(", ")}</span>}
      </p>
    </>
  );
}

function Dataset({ operatorKey: _ }: { operatorKey: string }) {
  const s = useQuery(api.episodes.stats);
  if (!s) return null;
  return (
    <section className="dataset" aria-label="World-model dataset">
      <h3>World-model dataset</h3>
      <p className="big">{s.accepted}</p>
      <p className="quiet small">
        accepted episodes of {s.total} recorded, {(s.hours * 60).toFixed(1)} min of de-identified wrist video,{" "}
        {s.frames.toLocaleString()} state/action frames.
        {s.needsReview > 0 ? ` ${s.needsReview} waiting for a person.` : ""}
        {s.successRate !== null ? ` Labelled success rate ${(s.successRate * 100).toFixed(0)}%.` : ""}
      </p>
    </section>
  );
}
