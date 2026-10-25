# From prototype to production: scrub assistant console

A hackathon prototype, not a medical device. This note says who it is for, how it fails, and what a real
deployment would need. It states what the code does today; nothing here has been clinically evaluated.

## Who it is for and the problem

Hypothesis, to be validated with real people before it is claimed: scrub techs and surgical-technology students
spend attention on handing instruments and keeping the count, and a missed instrument is a patient-safety
event. The console takes over two chores a person does under pressure: fetching a named instrument and keeping
an auditable record of where every instrument is. **Open item:** talk to a working or training scrub tech and
replace this paragraph with their account.

## What "useful" would be measured by (not measured yet)

| Question | Measure | Today |
|---|---|---|
| Does it understand what's asked? | Share of spoken requests resolved on the first try, by speaker and noise level | Unmeasured; unit-tested on the parser only |
| Is the right instrument handed over? | Wrong-instrument rate on a real tray | Unmeasured; arm picking is unproven on real instruments |
| Does the count catch a missing instrument? | Detection rate with an instrument removed, on camera | Only on rendered scenes and the simulated count |
| Is it faster than asking a person? | Seconds from request to presented instrument | Unmeasured |

## How it fails (and what happens)

| Failure | Behaviour |
|---|---|
| Claude unreachable, refuses, or no key | Rule-based parser takes over; the dashboard says which parser answered |
| Claude names an instrument that isn't on the tray | The arm's own catalog rejects it; the console asks a spoken question; nothing moves |
| Request is ambiguous ("scissors") | One short question naming the choices |
| Speech service down or no key | Typing and on-screen replies keep working |
| Convex down | Events are mirrored from a background thread; failures are logged and the arm is unaffected |
| Arm faults (collision guard) | The arm freezes in place; the fault is logged, shown, and sent to triage |
| Droid proposes something outside the allowlist or bounds | Marked invalid; it cannot be approved |
| Config changed after a proposal was made | The proposal goes stale and is not applied |
| An audit log is edited or its tail is cut | Verification fails at the edited line; Convex raises an alert if a head moves backwards |
| A web page tries to drive the arm or console | Refused: bearer token, no CORS, Origin and Host checks |
| Count is off | The console names the missing or outstanding instrument aloud and on screen |

## Responsible design

- **A person decides** anything with consequences: config changes need an explicit approval, and the count
  check reports; it does not "fix" the ledger.
- **Fail closed.** Unknown, ambiguous or refused inputs produce a question or a refusal, never a motion.
- **Honest labels.** Simulated counts and rehearsal agents are marked as such wherever they appear, and every
  generated text says what generated it.
- **Privacy.** No patient data is collected or stored. Audio goes to Cartesia and request text to Anthropic;
  keys live in the environment, never in the repo. The audit log records utterances (truncated), so a
  deployment must decide retention.
- **Accessibility.** Every action works by keyboard and typing as well as by voice; contrast and focus
  states were designed for, but not audited with assistive technology.

## What production would need

1. **Regulatory path.** A robot that handles surgical instruments is a medical device; this needs a
   risk analysis and the relevant clearance. None is claimed.
2. **Hardware safety.** A hardware e-stop, force/torque limits verified on the actual arm, and sterile-field
   practice. Today the safety net is the software collision guard and the gripper torque cap.
3. **Identification validated on real instruments.** Segmentation was trained on rendered scenes only.
4. **A real count source.** Camera count plus a second independent check (for example per-instrument tags),
   because a single vision system should not be the only safeguard.
5. **Identity and access.** The bearer token is a single shared secret; production needs per-user
   authentication, roles (who may approve a config change), and signed commands.
6. **Operations.** Hosted Convex with backups and retention, log shipping, alerting on audit-chain alerts,
   a deployment pipeline, and monitoring of speech and model latency.
7. **Evaluation.** A study with scrub techs: task time, errors, trust, and workload, before any claim of benefit.
