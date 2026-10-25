# Take-back: the arm takes a tool back from the surgeon and sets it down in a clear spot

Design, 2026-10-25 (revised after a builder check and a consistency review). Builds on the camera-watched
hand-off in `tools/pick.py` and the pick/place motions in `arm/scrub_arm/handoff.py`. Nothing here changes how
the arm picks (user: the pick motion stays as it is).

## Intent

After the arm has handed a tool out, the surgeon gives it back. On a command (how the surgeon gives it is out of
scope; for now it is typed), the arm goes to the hand-off pose with its jaws open, waits for the tool to be put
between them, holds it, finds a clear spot on the table, lays it down parallel to the other tools and returns to
rest. The same wrist camera that today decides "the tool has been taken" decides "the tool is here".

Why: a scrub assistant has to take instruments back as well as hand them out, and the returned tool must end up
where the next camera pick can find and grab it, not on top of another tool.

## Decisions made by the user

1. The tool coming back is always the one most recently handed out. So its identity, length, footprint on the
   table, thickness and jaw width are known before it arrives.
2. Any clear spot on the table will do. No fixed return zone, no slots.
3. It is laid parallel to the other tools.
4. The jaws close softly on the tool while the surgeon is still holding it, the arm pauses one second so they
   can let go, then moves off slowly.
5. The command is `take` in `tools/pick.py`; the same job kind exists on `ScrubAssistant` so the voice/HTTP
   route can call it later.
6. Once the tool is seen between the jaws it must stay there one second before the jaws close.

## Goals

- Works for every tool in `config/tray.yaml` `in_set` (the 8 printed parts).
- No new calibration ritual: the presence threshold comes from the same tool's own hand-off (the red count the
  camera saw while presenting it), and the geometry from what is already calibrated (camera.yaml, FK, the paper
  polygon `mat_mm`).
- The returned tool lands where the next `pick.py` pick works unchanged: parallel to the row (within a few
  degrees), clear of its neighbours by a margin, on the paper, where the arm reaches.
- The operator can always step in: Enter grips now, Enter releases at a stuck point, `q` abandons.

## Non-goals

- Identifying an unknown tool at the jaws (decision 1 makes this unnecessary).
- Tidying or moving other tools to make room.
- Taking back several tools in one go, or a tool the arm never handed out.
- The voice/HTTP trigger itself (the job kind is added; wiring a camera into the service is a follow-up).
- A new wrist orientation for laying a tool flat (the two orientations the arm already places with are used).

## Anti-goals (a failure even if it "works")

- Closing hard, or closing twice, on the surgeon's hand. One soft close per offered tool; a confirmed-empty
  grasp reopens once and waits again.
- Dropping the tool anywhere but at the set-down: not from the hand-off pose, not from the hover.
- Setting it down on top of, or touching, another tool, or outside what the camera can see.
- Fast motion near the surgeon: the first move after the grip is slow.
- Opening the jaws anywhere except (a) at set-down after the descent, or (b) right after a grasp the camera
  confirms caught nothing. (Today's rule, with (b) as the one new case.)
- Commanding the gripper past its closed limit (the hard limit stays; `set_gripper` clamps).

## Constraints from what exists

- Gripper: `ArmDriver.gripper_grasp()` closes softly to the stall (grip_kp 6, settle 0.6 s), reads the position
  once, then squeezes 4 deg more at squeeze_kp 20; its "holding" return is unreliable for thin tools (fingers
  flex), so the camera, not the gripper, decides whether the grasp caught the tool. `gripper_width(mm)` opens to a
  gap; the width table covers 0-95.5 mm.
- Camera: wrist camera index 0 at 1920x1080, opened by `pick.py` at import; frames are sometimes empty
  (`frame()` retries and reopens). Runs need macOS camera permission, so they go through the terminal panel.
- Hand-off pose: `poses.yaml` `handoff` (re-taught by `tools/handoff_calib.py` where the camera sees the held
  tool). `tools/handoff_calib.json`: `min_px` 20000, `gone_fraction`, `gone_px`, measured with the #20 blade.
- Hand-off release today (`pick.py wait_until_taken`): red pixels near the jaws in the lower-middle of the frame
  (`held_red()`), fingers masked by `perception/config/finger_mask.png` (open fingers); "taken" when the count
  drops below `gone_fraction` of the held count for 0.5 s; a held count under `GONE_MIN_PX` means the camera
  can't see the tool and the release is Enter-only. The held count (`base`) is local to that function today and
  the detection that was picked is local to `pick()`.
- Table canvas (`scrub_vision/red_parts.py`): metric top-down map at 0.5 mm/px over table x 150-750, y
  -350-350 mm; every detection carries `table_mask` on it; the pixels the camera cannot see (fingers, frame edge
  minus `UNSEEN_MARGIN_PX`) are known inside `detect()` (`seen`/`valid`) but not returned today. The paper's
  corners in table mm are `mat_mm` in `config/camera.yaml` (written by `calibrate.py`, used as a region by the
  service); the real paper sits around x 300-550.
- Place motion: `ScrubAssistant._place_at(x, y, yaw, thickness, sense_contact)` = hover at `pick.hover_mm`
  (100 mm) -> descend (optionally `_descend_until_contact`, which itself moves to the hardcoded hover first and
  steps down; contact on the first step raises TaskFailed, which `_place_at` swallows and lets go at the hover)
  -> `_let_go` (opens to `self.holding`'s `jaw_open_mm`, or fully if nothing is recorded as held) -> hover. It uses
  `_tilt_for` (the 45 deg pick tilt where it reaches, else straight down) and `_solve` with `allow_flip=True`
  (the solver may turn the jaws 180 deg). `_pick_side` shows how `_pick_at` pins the yaw instead (tries both
  ways, then solves with `allow_flip=False`). `_pick_at` records `_picked_from` (grasp x, y, yaw, thickness: the
  grab point, not the tool's centre) and `_flipped` (one value, overwritten by every pick).
- Pick orientation: `arm.yaml pick.tilt_pose: scan`: grasps and places copy the scan pose's tilt, the gripper
  45 deg from vertical with the wrist camera looking straight down; it reaches ~100 mm further than straight
  down. Both orientations are reachable through `_solve(..., tilt=True/False)`.
- Geometry: `Kinematics.fk(q)` gives the TCP (between the fingertips) and its rotation; `tool_axis_local` is the
  approach axis (fingers point along it), `jaw_axis_local` the direction the fingers travel. `Camera.pose(q)`
  gives the camera centre and rotation in the base frame; `cv2.undistortPoints` + that rotation turn pixels into
  rays. The scan pose repeats to a fraction of a degree, so a frame taken there earlier lines up with a later one.
- `pick.py look()` moves to the scan pose and detects with the jaws as they are; `pick()` opens them first.

Terms used below: a = approach axis, j = jaw (closing) axis, k = a x j (across the jaws). The printed parts are
flat plates clamped by their edges, so a held tool lies in the plane through the TCP with normal j, at an angle
theta from k toward -a (away from the fingertips): u0 = cos(theta) k - sin(theta) a. In the tilted pick
orientation both a and k point 45 deg down (checked numerically with `Kinematics.set_pick_tilt` on the taught
scan pose: a 45.2 deg from vertical, k 45.1 deg below horizontal), so a tool picked flat off the table has
theta = 45 deg; straight down, k is horizontal and a flat tool has theta = 0.

## The flow

0. Precondition and look first. `pick.py` keeps a stack of hand-offs, most recent first; each entry: the tool
   (`Resolved`), the red count the camera saw while presenting it and whether the camera saw it taken
   (`camera_ok`: count >= `GONE_MIN_PX` and released by the camera, not Enter), the wrist orientation it was
   presented with (`_flipped` at that time), where it lay before the pick (centre, `axis_deg`) and its footprint
   there (length L and width W of the oriented box round its `table_mask`, from the detection that was picked).
   Empty stack -> "nothing was handed out" and stop. Then, before presenting, open the jaws and `look()` (scan
   pose) and keep one frame: it gives the occupancy map for step 6 and the background for step 5. The table is
   assumed not to change while the surgeon hands the tool back.
1. Present open jaws. Move to the `handoff` pose with the entry's wrist orientation; open the jaws to
   TAKE_OPEN_MM (50 mm: a wide target for the surgeon, but a shorter, softer close than from fully open; my
   call).
2. Wait for the tool (camera). Baseline = median `held_red()` of 5 frames with the jaws empty. Threshold =
   PRESENT_FRACTION (0.5) x this tool's own hand-off count, never below `GONE_MIN_PX`. The tool is present when
   the count is at or above the threshold; it must stay present for SETTLE_S = 1.0 s continuously (user); a
   count that drops resets the timer. Enter grips now. Nothing for TIMEOUT_S = 60 s -> rest pose, report
   "nothing offered".
   Camera trust: if `camera_ok` is false for this tool, or the empty-jaw baseline is already at the threshold
   (something red in view on arrival), the camera is not trusted for this take: it says so, waits for Enter,
   and step 3 skips its camera verdict (jaws stay closed; the operator confirms). Logged, mirroring today's
   Enter-only hand-off.
3. Grip. `gripper_grasp(settle_s=TAKE_SETTLE_S)` with TAKE_SETTLE_S = 1.2 s (the close from 50 mm travels about
   twice a pick's; the stall must be reached before the squeeze; to be checked on the bench, see Tests). Pause
   LETGO_PAUSE_S = 1.0 s (user) for the surgeon to let go. Then look: count still at or above the threshold ->
   held: set `a.holding` to the entry's tool (so `_let_go` opens to its jaw width and the REPL guard works).
   Otherwise the jaws closed on nothing (the surgeon kept it, or it fell): reopen to TAKE_OPEN_MM and go back to
   step 2 within the same 60 s, at most MAX_GRIPS = 3 grips in one take. Camera not trusted: held by the
   operator's word, `a.holding` set the same way.
4. Move off slowly. `move_pose("scan", speed=SLOW)` with SLOW = 0.2 (user: slowly); normal speeds after that.
5. Measure how it sits in the jaws (camera, scan pose; see Geometry). Held pixels = red now, minus the step-0
   red mask dilated by BACKGROUND_DILATE_PX = 15, minus the finger mask, in the lower-middle region `held_red()`
   uses; the component touching or nearest the finger mask. Outputs: u (the unit direction from the jaws toward
   the visible end, in the gripper frame; theta follows from it) and v (the visible side's length, mm from the TCP
   to that end). Visible side shorter than MIN_VISIBLE_MM = 30, or its far end on the frame border ->
   measurement unavailable: theta = 45 deg (as picked), v unknown.
   Lost: with the camera trusted and fewer than MIN_HELD_PX = 2000 held pixels -> "lost it": clear `a.holding`
   (the jaws stay closed, nothing is in them), rest pose, stack unchanged; the next `take` starts again at step 0.
6. Find a clear spot. Inputs from the step-0 frame: occupied = the union of every red component's `table_mask`
   (complete or cut off, known part or not) plus everything the camera could not see (the unseen region warped
   onto the canvas), dilated once by MARGIN_MM = 15 (the footprint itself carries no margin). Row direction psi =
   circular median (mod 180) of `axis_deg` over the complete detections; none -> this tool's own axis before it
   was picked. Allowed area: the `mat_mm` polygon (the paper; my call, widen later if the paper proves too
   tight).
   Two footprints, one per place orientation (tilted, straight down), each from Geometry: tilt, which end is low,
   l_low (the held length on the low side), hang = l_low sin|tilt|, and the rectangle: the tool stands on its
   low end at the contact point C = jaw point + l_low cos(tilt) along psi toward the low side and, released,
   falls back toward the high side, so its footprint along psi is the full L starting at C (flat: C is the low-
   side end and the same rule gives the ordinary rectangle). Across psi: W_fall = 3 W (a tool handed in on its
   face stands on edge and falls onto one side; my call). v unknown (step 5 fallback): 2L along psi centred on
   the jaw point; that usually fits nowhere and the operator decides (below).
   `clear_spots` (a pure function on canvas arrays) runs once per footprint: candidates are footprint positions on
   a GRID_MM = 5 mm grid inside the polygon, for both jaw yaws psi+90 and psi+270 (mirrored along psi: the same
   physical end is low, pointing the other way), free when the rectangle meets no occupied pixel (rotate the
   occupied mask by -psi once, one box filter); each returns the jaw point, yaw and score. Ranking: the flatter
   orientation's candidates first; within one, nearest to where this tool lay before (keeps the layout stable),
   then the farthest from occupied (distance transform). Then, in `takeback.py`, walk the ranked list and keep
   the first candidate where `_solve(allow_flip=False)` reaches both the raised hover (hover + hang) and the
   set-down floor in that orientation. None -> hold at the scan pose and ask: Enter puts it back where it was
   picked from (the safe place below with `_picked_from`'s x, y, yaw and `sense_contact=True`), `q` keeps holding
   (see Operator state).
7. Place. `_place_at(jaw_x, jaw_y, yaw, thickness, sense_contact=True, tilt=<chosen>, allow_flip=False,
   raise_mm=hang, drop_if_blocked=False)`: the hover and the descent floor both rise by `hang` (threaded into
   `_descend_until_contact` too, which otherwise returns to the hardcoded hover) so the low end clears the table
   at the hover and meets it at the floor; contact sensing stops the descent when the low end touches. Contact on
   the first step (something in the way at the hover) does not let go: lift, mark that candidate occupied, try
   the next (at most 2 more), else the no-candidate path. `_let_go` opens to the tool's `jaw_open_mm` and clears
   `a.holding`. Back to hover, then `move_pose("rest")`; pop the stack.
8. Confirm. One more scan: a detection within CONFIRM_MM = 25 mm of the planned centre -> report it and what the
   detector calls it (log only; a different name is not a failure, see Known limits). Nothing there -> report
   "set it down but can't see it"; no retry.

## Geometry: u, v, theta, tilt, the low end and hang

- At the scan pose, the TCP p and rotation R come from `kin.fk(q_measured)`; a = R tool_axis_local,
  j = R jaw_axis_local, k = a x j. The clamped plate lies in the plane through p with normal j.
- From the held component: its principal axis; the far end E (farthest from the finger mask) and a second point
  M halfway from the finger mask to E. Each pixel -> a ray (`cv2.undistortPoints` with K, dist; rotate by the
  camera rotation from `Camera.pose(q)`); intersect with the plane (p, j) -> 3-D points. u = normalize(E - M)
  points at the visible end; v = |E - p|, which must be <= L (else the measurement is rejected). In the gripper
  frame theta = atan2(-u . a, u . k). The tool's centre is at p + (v - L/2) u; its ends are E and p + (v - L) u.
- At a place orientation R_place (the tilted pick rotation turned to the candidate's yaw, or the straight-down
  one), the tool's direction is u_w = R_place u_local and tilt = asin(u_w . z): theta - 45 tilted, theta
  straight down (both checked numerically with the taught scan pose for psi = -85 and +30, within 0.3 deg).
  The low end is the end of u_w with negative z: the visible end E if tilt < 0, else the other end; l_low = v
  if E is low, else L - v. hang = l_low sin|tilt|.
- In both orientations the horizontal part of u_w lies along psi (the vertical plane through a and k contains
  the row direction), so the jaw yaw for a parallel landing is psi + 90 whatever theta is. Exact straight down;
  tilted, the scan pose's jaw axis is 2 deg off horizontal, so the landed axis is off psi by up to ~2 deg over
  |tilt| <= 45 (measured numerically; accepted, see Goals).
- Which orientation: tilted lays a tool flat at theta 45 (as picked), straight down at theta 0 (handed in square
  across the jaws); a tool pointing out of the fingers (theta 90) tilts 45 deg in the tilted orientation and
  hangs vertical straight down, so the tilted one is chosen and it is set down by contact and tips over. Prefer
  the orientation with the smaller |tilt|; the other if the first reaches no candidate.
- Self-check without a surgeon (also the acceptance test for this measurement): pick a tool with `pick.py`,
  go to the scan pose, measure. Expect theta = 45 +/- 5 deg and the centre p + (v - L/2) u within 10 mm of the
  grab point's offset from the tool's centre along its axis (known from the detection that was picked). The
  signs of theta, u and the low-end rule are fixed with this check, not by inspection.

## Operator state

- `pick` is refused while `a.holding` is set (it offers to put the tool back); today it would open the jaws
  first and drop the tool.
- `take` is re-entrant only from the holding-at-scan state (after `q`, or after the no-candidate hold): it skips
  to step 5. After "lost it" or a set-down nothing is held and it starts at step 0.
- The hand-off stack is popped only by a successful set-down. Two tools handed out are taken back most recent
  first; taking back several in one go stays a non-goal.

## Parameters

Constants at the top of `arm/scrub_arm/takeback.py` to start (like GONE_FRACTION in pick.py), moved into
`arm/config/arm.yaml` under `takeback:` once tuned on the arm: TAKE_OPEN_MM 50, TAKE_SETTLE_S 1.2, SETTLE_S 1.0,
PRESENT_FRACTION 0.5, LETGO_PAUSE_S 1.0, SLOW 0.2, MAX_GRIPS 3, TIMEOUT_S 60, MARGIN_MM 15, GRID_MM 5,
CONFIRM_MM 25, MIN_HELD_PX 2000, MIN_VISIBLE_MM 30, BACKGROUND_DILATE_PX 15.

## Where the code goes (my calls)

- `arm/scrub_arm/takeback.py`: the flow (`take_back(...)`), importable without hardware: it takes the app, the
  frame/red-count functions and the hand-off stack as arguments, so a test can drive it with scripted counts.
  Also the u/v measurement, the two footprints and the reachability walk.
- `tools/pick.py`: the `take` command (REPL and argv), the hand-off stack (pushed in `after_pick` when the tool
  is handed over), the REPL guards above; passes its camera functions to `take_back`. To fill a stack entry:
  `wait_until_taken` returns the held count with its verdict, `pick()` hands back the detection it picked (not
  just a bool), and `after_pick` reads `a._flipped`. `tools/handoff_calib.py` (`import pick as P`) keeps working
  (it only uses `pick()`'s truthiness and `_present`).
- `perception/scrub_vision/free_spot.py`: `clear_spots(occupied, allowed_polygon, footprint, psi_deg, prefer_xy)
  -> [(jaw_x_mm, jaw_y_mm, yaw_deg, score)]`, a pure function on canvas arrays (the footprint carries its own
  jaw-point offset along psi). `detect()` grows a way to get its unseen mask on the canvas (an attribute or a
  second return, builder's choice).
- `arm/scrub_arm/handoff.py`: `_place_at` grows `tilt`, `allow_flip`, `raise_mm`, `drop_if_blocked` keyword
  arguments and `_descend_until_contact` takes the raised hover; defaults keep today's behaviour for
  `_return_held`. `ScrubAssistant` gets `take_back: Callable | None` and a `take` job kind that calls it or fails
  with "no camera". No other change to the motion code.
- `arm/scrub_arm/driver.py`: `gripper_grasp(settle_s=None)` overriding `gripper.settle_s` when given.

## Tests

- `perception/tests`: `clear_spots` on synthetic canvases: fits / does not fit; prefers the old spot; never
  overlaps the occupied-or-unseen mask or leaves the polygon; respects psi; mirrored jaw points for the two yaws.
- `perception/tests`: u/v recovery from a synthetic projection of a segment at known theta and hold point
  (project with the Camera model, recover within 2 deg / 3 mm), plus the tilt / low-end / hang rule for both
  orientations against the numeric values above.
- `arm/tests`: `take_back` with scripted red counts in the sim app: the only jaw opening outside set-down is
  after a camera-confirmed empty grasp; the camera-untrusted path never reopens; a blocked descent never lets go;
  `a.holding` is set after the grip and clear after set-down or "lost it"; the gripper is never commanded past
  closed (existing test keeps passing).
- On the bench: the close from 50 mm with TAKE_SETTLE_S reaches the stall before the squeeze (watch the
  gripper position settle); then the self-check above for one tool; then a take-back of each of the 8 tools with
  the operator acting as the surgeon; each lands parallel and clear, and the next `pick.py` pick of it works
  first try.

## Known limits

- Anything red near the jaws reads as the tool (red gloves or sleeves).
- A tool handed in pointing out of the fingers (theta near 90) tilts 45 deg at the set-down; it is set down by
  contact and tips over, so the landing is less tidy.
- The landed axis can be ~2 deg off the row in the tilted orientation (the taught scan pose's jaw axis is not
  quite horizontal).
- The confirm scan cannot tell the returned tool from an interchangeable twin (the two micro scissors share one
  class).
- The step-0 frame is the picture of the table for the whole take: a tool moved during the hand-back is missing
  from the map (the contact-sensing descent is the safeguard; a re-check before descending is a cheap later
  addition), and if it slid under the held tool's streak it corrupts the measurement.
- `mat_mm` is where the paper was when `calibrate.py` last ran; if the paper moves, re-run it (a knob, not a
  code change).
- The 60 s wait and the thresholds are tuned for a single operator at the bench, not for an OR.

## Left to the builder

- The exact lower-middle region and how the held component is chosen among several (touching the finger mask
  wins; else nearest).
- Whether `detect()` exposes the unseen mask as an attribute or a second return; how `_place_at` and
  `_descend_until_contact` take the new arguments; how the job kind reaches `pick.py`'s wiring.
- How `pick()` returns the detection without disturbing its callers' truthiness checks.
