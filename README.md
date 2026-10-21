# AGV / AMR route simulator

Kinematic + quasi-static simulation of a differential-drive AGV/AMR on a factory floor plan.
It answers: *How long is one cycle? How tight does it really turn? How much floor does it sweep?
Does the towed/lurked cart hit anything? Are the motors, gearboxes, tyres and battery sized right?*

## Easy mode (recommended)

**Mac:** double-click `Start Simulator.command` (first time: right-click → Open, because it was downloaded).
**Windows:** double-click `Start Simulator (Windows).bat`.
**Any system:** `pip install -r requirements.txt` then `python3 app.py` (or just `python3 main.py`).

Your browser opens the simulator. The first start takes a minute to install; keep the terminal window open while you work.

**Windows PC without Python** (or where installing is not allowed): use `AGV_Simulator_Windows.zip` - the
simulator with its own portable Python (64-bit Windows 10/11, no installer, no admin rights). Unzip it
(right-click → Extract All), double-click `Start Simulator.bat`. Make a fresh one after changes with
`python3 tools/make_windows_portable.py` (needs `tools/windows_portable/python-win-runtime.zip`, which
holds Python 3.12 + numpy 2.1, matplotlib, pillow, pyyaml, flask for Windows).

1. **Route** tab: pick or upload your floor plan, press **📏 Measure scale** and click two points you know the distance of.
   Click on the plan to add route points; select a point to make it a stop, a sharp corner, reverse, etc.
2. **Vehicle** tab: all AGV settings as a form, with a live top-view drawing.
3. **▶ Run simulation**: cycle time, turning radii, motor/battery load, grip, tip-over and collisions in the
   **Results** tab, and the cycle plays on the map. Result files also land in `results/<route name>/`.
4. **Turning** tab: compare corner radii against turning on the spot - real axle / inner-wheel radius, swept
   floor with the cart, manoeuvre time, motor torque, grip, cart angle - and find the smallest radius that works.

5. **Route → Tools**: 📐 **Ruler** (also the 📐 button next to the zoom buttons) - click two or more points
   to measure; it snaps to route points and to the path (hold **Alt** for a free point), shows each segment,
   the total and the angle; **Esc** or right-click ends it. **Block an area** - drag a rectangle over the plan
   for a no-go zone the auto route and the collision check treat as a wall (list below it, ✕ removes one).
   **↩ Reverse branch** (Route → Tools): a spur the AGV backs into. Click the junction on the route line,
   then click along the branch - the last click is the end. The tool adds: a **REVERSE** stop (RFID tag)
   *past the junction* (default: front-to-rear sensor distance + 0.5 m), the reverse legs into the branch
   (rear sensor, curve radius at the junction), a stop at the end (wait, name, hitch / unhitch trolleys),
   and the forward legs out of the branch back onto the route (with the speed mode it had before). The
   route must pass the junction driving *away* from the direction the branch points to (backing up may
   bend at most 80° into it) - otherwise the tool says so. Ctrl+Z undoes it; Run simulation to check the
   tape error in reverse. REVERSE tags also show in the Combined tab's RFID tags (grape circle).
   **Auto route keeps reverse branches exactly as drawn** (REVERSE tag, junction curve, branch points, end
   stop and the way back to the junction - no sharp corners, no speed tags inside); in *Plan a new path*
   the planned lane arrives on the junction's line 2 m before it and leaves 1 m past the REVERSE tag.
   **RFID marks** tab (7): a marking plan - where the RFID tags go on the floor, for the people who lay the
   tape and tags. **Marking only**: routes and simulation are not changed. Tick the routes to draw underneath;
   drag a **new tag** (APPROACH … MAX, LEFT, RIGHT, STOP, HOME, REVERSE - it asks for the tag ID, e.g. 0100)
   or a **used tag** (an ID already in your list - places it again, same ID = same command) onto the map
   (snaps to the route line; or click a chip, then the map). Drag a mark to move it, right-click / Delete
   removes it, Ctrl+Z undoes. The tag list edits ID (all its marks follow), command and note; *Tags from the
   routes* adds a mark at every tag the routes already have (one ID per command - then type your real IDs).
   **Custom tags**: *+ Custom tag* adds your own command (e.g. PIN UP, CHARGE, CALL) with its own colour and
   note; it appears with the new tags to drag onto the map. Rename it (its tag IDs follow) or change the
   colour in the list; it can be removed once no tag ID uses it.
   Saved as `rfid/<name>.yaml` (the previous version goes to `rfid/.backup/`). **⬇ Save image** = the
   picture with the tag IDs, labels and a key (ID, command, how many); **⬇ List (CSV)** = every mark + the IDs.
6. **Route → Auto route**: see *Auto route* below.
   **Several points at once:** ⌘/Ctrl+click adds / removes a point, Shift+drag selects a box, ⌘/Ctrl+A (or
   Tools → *Select all points*) selects all. Drag one of them to move them all; arrow keys move the selection
   5 cm (Shift: 50 cm); the yellow panel moves them by an exact x / y or deletes them; Esc clears.

7. **Save a video**: after a run, set the playback speed (e.g. 10× or 20×), zoom / pan the map to frame it and press
   **⏺ Save video** in the player bar. The whole cycle plays and the video lands in your Downloads folder (MP4 in
   Safari / recent Chrome, else WebM). The video shows the floor plan, tape path, AGV + trolleys, swept area and
   the stops with their names - no route points or helper lines. ⏹ stops early and saves what was recorded.

8. **Sequence** tab: one AGV drives several routes one after another, each with **its own vehicle** (e.g. UBC
   route + UBC vehicle, then ITG, OCC, T6, Piston). Add steps (route, vehicle, × repeat, wait after), order them
   with ↑ ↓, set a target for the whole sequence and press **▶ Run sequence**: a table per step (cycle, start →
   end, average speed, tape error, status), the total against the target, sequences per hour, distance, energy,
   and warnings (a step that loses the tape, a gap between where one step ends and the next starts). **▶ Play
   the whole sequence** plays it step by step on the map; ⏺ Save video records all of it. Saved as
   `sequences/<name>.yaml`; `python main.py sequence --file sequences/<name>.yaml` prints the same table.

9. **Combined** tab: several **saved** routes on one floor plan (e.g. ITG / T6, UBC, OCC, Piston) - viewing
   only, nothing is changed. Tick the routes; options: colour per route (or one tape colour, so a shared tape
   is one line), side by side (routes on the same tape drawn next to each other on screen), direction arrows,
   floor plan on / off, and **stop areas** (yellow boxes S1, S2 ...: stops of different routes closer than
   *merge* m are one area; the table says who stops there and what it does - click a row to zoom to it, click
   a box on the map to see it). Click a route name to highlight it, ✎ opens it in the Route tab, **⬇ Save
   image** downloads the view as PNG. Each route is smoothed with its own vehicle (`configs/<name>.yaml`,
   found from the route file name). The same picture from the command line:
   `python tools/combined_map.py` (options `--no-stops`, `--colors`, `--map`, `--full`, or list route files).
   **Follow a route in an area** (same tab): when one route changed in one place (e.g. ITG got a new home
   spur), make other routes drive the same way there. Choose the route to follow, **▭ Draw area** (drag a
   rectangle around the part to change - each route must go in and out of it once), tick the routes to
   change, **Preview**: inside the area the path, curves, speed modes, stops and trolley actions are copied
   from the reference (the route starts where the reference starts if its old start was in the area);
   outside it nothing changes. *Keep each route's own wait times* keeps the start stop's wait / name and
   the wait of an attach / detach stop with the same action. The preview lists the stops removed / added,
   length and - with *Check with the simulation* - cycle time before → after, target, tape error and
   TAPE LOST. **💾 Save** writes the routes; the old files go to `routes/.backup/<date> before follow ...`.
   **RFID tags** (same tab), four kinds, all stored in the route points:
   *speed* APPROACH / SLOW / MED / HIGH / MAX (coloured squares) = the point's speed mode, read by the
   closed-loop simulation; *track select* LEFT / RIGHT (teal diamond L / R) = which branch to follow at the
   next fork (route point field `fork: left|right`, also in the Route tab - for the layout guide, the
   simulation follows the drawn path); *stop* STOP (crimson octagon) = a stop point, asks for the wait time;
   *homepost* HOME (dark house) = the first point, where the cycle starts. Tags of different routes closer
   than 0.35 m are one physical tag (T1, T2 ...); red = different speeds or LEFT and RIGHT at one spot.
   Drag a chip onto a route line (or click a chip, then the line); drop it on a tag to change that tag
   (STOP: its wait time). Drag a tag to move it (HOME: the route now starts there - closed-loop routes;
   the old start stays a stop only if it attaches / detaches trolleys); right-click it (or select +
   Delete) to remove it (not HOME, not trolley stops - those are changed in the Route tab). A new tag goes
   on every route whose tape passes there - click a route name first to put it on that route only.
   Nothing is written until **💾 Save to routes**: a tag on a straight part becomes a pass point (moved
   out of a curve if dropped in one), on an existing point it is added to that point; a removed tag
   deletes a point that then carries nothing. Old files go to `routes/.backup/<date> tags/`. *Show*
   picks which kinds are drawn; labels show what the tag does / its number / both / none. **⬇ Save image**
   gives the picture with tags, labels and key as a guide for laying the tape and tags; **⬇ Tag list
   (CSV)** lists every tag (type, value, x / y, routes, stop wait). Auto route keeps LEFT / RIGHT tags.

The command line below is still there for scripting and parameter sweeps.

## Git / GitHub

The folder is a git repository (branch `main`). `.gitignore` keeps out what is made again or too big:
the two zips, `tools/windows_portable/` (the 40 MB portable Windows Python - keep a copy, it is needed by
`tools/make_windows_portable.py`), `results/`, the app's `.backup/` folders, caches and `.DS_Store`.
Your routes, vehicles, maps, sequences and RFID plans *are* in git. Tests use their own files in
`tests/data/`, so they pass whatever you keep in `configs/` and `maps/`.

## Install (command line)

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt                         # numpy, matplotlib, pillow, pyyaml
python examples/make_demo_map.py                        # builds maps/demo_factory.png
```

## Workflow

```bash
# 1. Check the vehicle definition (top + side view, sensor/hook/wheel positions, drivetrain)
python main.py agv  --agv configs/agv2.yaml

# 2. Draw a route on your own floor plan (PNG/JPG). Press 'c' first to calibrate the scale.
python main.py edit --map maps/my_factory.png --route routes/line1.yaml --agv configs/agv2.yaml

# 3. Simulate one cycle -> report.txt, summary.json, timeseries.csv, report.png in --out
python main.py run  --agv configs/agv2.yaml --route routes/line1.yaml --out results --animate
python main.py run  ... --save-anim results/cycle.gif --speed 8 --no-show     # headless / CI

# 4. Turning-radius study (arc radii vs pivot, swept envelope, loads) -> results/turning/turning.txt + .png
python main.py turn --agv configs/agv2.yaml --radii 0.75 1 1.25 1.5 2 3 --angle 90      # add --no-cart to compare

# 5. What-if: vary one parameter and compare
python main.py sweep --agv configs/agv2.yaml --route routes/line1.yaml \
                     --param performance.v_curve --values 0.25 0.35 0.45 --csv sweep.csv

# Any value can be overridden without editing the YAML:
python main.py run ... --set performance.accel=0.4 --set cart.mass_kg=250 --set hook.type=none
```

Demo routes: `routes/demo_route.yaml` (looped tow route with two stations),
`routes/demo_tow_delivery.yaml` (speed modes, a 3 % ramp, drop the full cart / pick up an empty one) and
`routes/demo_pivot_reverse.yaml` (lurking AMR: pivot on a sharp corner, reverse docking in DOCKING mode, lift a cart).

## Coordinate conventions

* **World**: metres, origin = lower-left corner of the map image, +x right, +y up, heading 0 = +x (CCW positive).
* **Body frame** (everything in the AGV YAML): origin = centre of the drive axle, +x forward, +y left, +z up.
  So a sensor 1.15 m ahead of the drive wheels is `x: 1.15`; a tow hitch 45 cm behind them is `x: -0.45`.
* **Map**: pixels darker than `obstacle_threshold` (0-255 grey) are obstacles; everything outside the image is an obstacle.
  Clean the floor plan so aisles are light and walls/racks/machines are dark. Set `meters_per_pixel` or calibrate with `c` in the editor.

## Route file

```yaml
name: Line 1
map: {image: ../maps/my_factory.png, meters_per_pixel: 0.02, obstacle_threshold: 100}
loop: true               # last waypoint connects back to the first
default_radius: 2.0      # fillet radius applied at every corner unless overridden
default_mode: ""         # speed mode for legs without their own (''/default = the vehicle's performance values)
waypoints:
  - {x: 6.0, y: 3.0, name: Home, action: stop, dwell: 20}   # stop + wait
  - {x: 12.0, y: 3.0, grade: 3}                               # leg to the next point climbs 3 %
  - {x: 21.0, y: 3.0}                                         # plain corner, default radius
  - {x: 21.0, y: 7.75, radius: 1.2, speed: 0.3, mode: SLOW}   # own radius, speed cap, speed mode for the next leg
  - {x: 12.0, y: 7.75, cart: detach, dwell: 10}               # stop, unhitch / lower the cart
  - {x: 9.0,  y: 7.75, cart: attach, cart_mass: 40}           # stop, hitch / lift a (40 kg) cart
  - {x: 5.0,  y: 7.75, action: pivot, radius: 0}              # stop and spin in place
  - {x: 3.0,  y: 7.75, direction: reverse}                    # leg leaving this point is driven backwards
```

`action`: `pass` (default), `stop`, `pivot`. `radius: 0` makes a sharp corner (forces a stop and pivot).
`speed`, `mode`, `grade` and `direction` belong to the leg that LEAVES the waypoint.
`cart: attach | detach | swap` implies a stop; the handling time is `hook.attach_time` / `hook.detach_time`
(swap = drop the trolleys and hitch other ones: both times). At `attach` / `swap`, `cart_load: full | empty`
chooses what is picked up: full = `cart.masses` (the loaded weights), empty = `cart.empty_mass_kg` per trolley;
`cart_mass: <kg>` overrides both. Route key `start_load: full | empty` = trolleys on the hook at the start.
In the app: select a point → *Trolleys at this point* (+cart / −cart / ⇄ swap) → *Picks up* Full / Empty / Own
weight; the timeline shows the kg at every attach / detach.
The cycle starts with the cart on the hook unless the first cart event of the route is an `attach`.
Corners whose fillet does not fit the leg lengths are clamped and reported as warnings.

## Closed-loop tape following with your own PID (line_follow)

**Speeds with line following on:** HIGH, SLOW and APPROACH run at `line_follow.speed_high / speed_slow /
speed_approach` with `accel_rate`, `decel_rate` (`approach_decel_rate` into APPROACH) - exactly like modes.py
on the real AGV - not at the `speed_modes` rows of the same name. The app keeps those rows equal to the
line_follow values (marked ⇄; editing either changes both); a YAML where they differ gets a warning in the
report. Other modes (e.g. MED) use their own row. Top speed is capped by the 0-5 V output (≈0.98 m/s for AGV2).

With `line_follow.enabled: true` (it is on in `configs/agv2.yaml`) every drive run is simulated closed-loop:
the MGS1600 is modelled as a 160 mm line across the vehicle that reports where the tape crosses it
(integer mm, noise, CAN delay, tape lost beyond +-80 mm), and **your controller** decides the wheel commands
every 10 ms. `agv_sim/kayaba_pid.py` is a 1:1 port of `core/pid.py` (checked bit-identical over 5000 random
cycles) plus the auto_mode / reverse_mode logic of `modes.py`: speed_mode HIGH / SLOW / APPROACH, linear
ACCEL/DECEL ramps, SENSOR_MAX_MM / SENSOR_MAX_STEP_MM checks, speed reduction, curvature feed-forward,
rpm -> AO volt with the 0-5 V clamp and 12-bit DAC, brake on tape loss. The drive is BLDC speed-controlled
wheels with a first-order response (`motor_time_constant`) and `brake_decel` when the brake is on.

Route points behave like RFID tags (read by the `rfid` sensor): a point's `mode` switches speed_mode there,
`speed` caps the speed, and at a stop the AGV brakes when the tag is read - so it stops past the tag by its
braking distance, like the real one. Put an extra point on a straight where your RFID tag really is.
The tape is assumed to continue 1 m past dead ends (stops before reversing, route ends).

Results: tape position under the sensor over time, largest error and where it happens (orange ring),
margin left to the sensor edge, **TAPE LOST** markers (the run stops there, as on the real AGV), stop
overshoot after each tag, and a tape-error column in the curve table and in the Turning study.
`front_sensor_positive` sets which side a positive `left_mm` means; with `right` your
ERROR_SIGN_FORWARD = +1 steers correctly (and -1 loses the tape), matching what you found on the AGV.
Copy new tuning from the robot with `python main.py import-profile --json .../profiles/agv2_kim.json --agv configs/agv2.yaml`
or the **Import from AGV profile** button in the Vehicle tab.

Note found while porting: `modes.py` only knows HIGH, SLOW and APPROACH. A `set_speed` of any other name
(e.g. EXTRA_SLOW) makes auto_mode drive at **HIGH speed** with SLOW gains, and AUTO_TARGET_EXTRA_SLOW_SPEED
is never read by config.py. The simulator reproduces this and warns.

## PID auto-tune

Vehicle → Line following → **🎯 Auto-tune PID on this route** (or
`python main.py tune-pid --agv configs/agv2.yaml --route routes/my.yaml [--apply]`).
It drives the route open in the Route tab again and again with your controller and changes **Kp and Td** of
every gain set the route uses - HIGH (`kp`, `td`), SLOW (`kp_slow`, `td_slow`; also MED and every mode modes.py does
not know), APPROACH, REVERSE - each judged only on the stretches it drives. N and Ti stay as they are.

* "Best" = balanced: small tape error (RMS + 99 %), little wobble (error changing side), calm steering (no
  jitter from noise), no saturation, far from the sensor edge, never losing the tape.
* Every try runs under slightly harsher conditions than set (≥1 mm sensor noise, +10 ms delay, 30 % slower
  motors, one wheel 3 % off), Kp / Td stay within 1/3 … 3 × the current values, and at the end 2 × the new Kp
  must still hold the tape calmly (gain margin), otherwise Kp is lowered.
* Result: old → new Kp / Td with tape RMS / 99 % / max / wobble before and after, the values to copy into
  agv2_kim.json ("pid_tuning": KP, TD, KP_SLOW, …) and **Apply to vehicle** (then *Save vehicle*).
* The simulation is not the real AGV - try the new values at SLOW first or go half-way, and watch the tape error.
  Takes about 1-5 minutes (several tries run in parallel).

## Trolley trains and loop speed

`cart.count` = number of trolleys the AGV pulls, `cart.masses` = kg of each one (with load), e.g.
`cart: {count: 3, masses: [170, 120, 80], coupling_behind_axle: 0.3, ...}` (Vehicle tab: *Number of trolleys
pulled* and one weight box per trolley). Every trolley is a single fixed-axle trailer hooked `coupling_behind_axle`
behind the axle of the one in front; the report gives the angle of every joint (jackknife check), the pull on
the AGV hitch summed through the whole train, swept floor / collisions of every trolley, and warns if the
trolleys would overlap. From the command line: `--set cart.count=3 --set cart.masses=170,120,80`.

**Speed with and without trolleys - the "Trolley load calc" switch** (top bar next to *Run simulation*, also
Vehicle → Hook & cart; `simulation.load_effects` in the YAML, `--set simulation.load_effects=true` on the
command line). Off (default): the speed profile does not depend on the trolley weight. On: the drive is
force-limited (`agv_sim/drive.py`) - drive force = min(motor peak torque through the gearbox, `friction_coeff` x
load on the drive wheels); the trolleys add mass and rolling drag and have no brakes; slopes add M·g·sin. So the
AGV speeds up slower than `accel_rate`, rolls further past a stop tag (`brake_decel` = bare AGV) and may not start
on a slope. Results then show *With and without trolleys* (speed-up, time to top speed, distance past the stop
tag, steepest slope for 0 … n trolleys) and split *Speed per speed mode* into pulling / empty.

Loop speed: the report and the Results tab split the run into loops (leaving the start point until leaving
it again) and give time, distance and average speed of each loop - with and without the stops.

## Auto route

Makes the route faster *and* checks every try with the full simulation (tape following with your PID,
swept floor of the AGV and every trolley):

* **Keep my path, optimize corner radii and speed tags** (`--mode keep`): your corner points stay; the tool re-fits every corner radius to
  its legs, picks the fastest speed mode for each straight (only if it saves time after its ramps), and
  places the RFID speed tags so the AGV is already slow before each curve and speeds up only after the
  curve (and after the trolleys have left it). Then it runs the simulation and fixes what goes wrong:
  collision at a corner → other radius; tape error above the limit → slow down earlier, bigger radius,
  longer straight after the curve before speeding up, or APPROACH speed. Finally it tightens radii while
  everything stays clean. Example: `AGV_Kayaba (OCC)` went from 696.9 s to 587.9 s per cycle, tape max 56 mm.
* **Plan a new path through my stops** (`--mode plan`): only your stops are used (name / stop / dwell / trolley events are
  kept). It finds the shortest lane with few corners through the aisles on a grid (orthogonal, or any angle
  with *style any*), keeping half the vehicle width + margin away from obstacles and leaving/arriving at
  stops straight; then the same radius / tag / verify loop. Collisions from trolleys cutting inside a
  corner move that lane section away from the hit spot, or re-plan around it; the best try is kept.
  **Only dark pixels are walls** (`obstacle_threshold`, default 100 of 255): light-grey CAD lines are
  *not* obstacles - draw no-go zones with *Block an area* (saved as `blocked:` in the route file) or the
  lane may cut through rooms.

**Trolley-safe slowing down** (on by default when the AGV tows; option *Allowed trolley push*, default 10 % of
the AGV weight): the trolleys have no brakes, so when the AGV slows down faster than their own rolling friction
(`cart.rolling_resistance` x g, ≈0.2 m/s²) they push it - with force = trolley kg x (deceleration − that).
Auto route knows the trolley weight on every stretch (full / empty / none from the +cart / −cart / swap points)
and (1) steps down through a gentler speed mode when the direct ramp would push more than allowed (e.g. HIGH →
SLOW at DECEL_RATE → APPROACH), (2) comes into every stop at APPROACH speed, so the hard stop brake only lasts a
moment, and (3) tells you which ramp setting of the AGV is too steep for that weight ("set APPROACH_DECEL_RATE
<= 0.30 m/s2"). The ramp rates themselves are settings of the AGV (DECEL_RATE, APPROACH_DECEL_RATE, the Brake
of a mode like MED) - a route can only choose which ramps are used. Results show *Trolley push (max)*.

**One tape used in both directions.** RFID tags are physical: a speed tag is read every time the AGV drives
over it, whichever way and whichever leg it was drawn for. The simulation does the same (Results note "Shared
tape … also reads …"). Auto route checks every leg against the tags of all the others: a speed-UP tag that another
leg would read where it has to be slow (e.g. just before its curve or stop) is moved further along its own leg,
or gets the slower mode if that is not possible; where another leg's SLOW tag would slow a leg down for nothing,
a tag to speed up again is added right after it (only where no other leg reads it). The log reports what it changed.

**Target cycle time** (Route settings → *Target cycle time*, route key `target_cycle_s`, or `--target` for
`main.py autoroute`). Results shows the cycle time against it (red = too slow, with where the time goes). Auto
route first makes the fastest clean route. If that is above the target it takes the **curves faster**: all curves
one speed mode up (e.g. SLOW → MED), then - where a bigger step loses the tape - one curve at a time, longest
first; then it trims the safety distances around the curves to 0.1 m. Every step is checked with the full
simulation. If the target is still out of reach it says what it would take (e.g. "add a speed mode between MED
0.40 and HIGH 0.98 m/s for the curves", what switching off trolley-safe slowing would give, allowed tape error,
waits). If it is faster than needed it drives calmer to land just under the target: the fastest mode only on the longest straights
(binary search on the straight length), then not at all, then the next mode - less tape error, trolley push and
wear for the same output.

The log says what each try found. If a collision is left, it names the spot: the floor is too tight there
for that train (e.g. a 180° turn between two aisles 3.75 m apart with a 1.6 m tongue) - give that turn more
room, pull fewer trolleys or move the stop. In the app the auto route replaces the one on the map
(↶ Undo brings yours back); press **Save as…** to keep it.

```bash
python main.py autoroute --agv configs/agv2.yaml --route "routes/AGV_Kayaba (OCC).yaml" --mode keep
python main.py autoroute --agv configs/agv2.yaml --route routes/stations.yaml --mode plan --margin 0.15
```

## Speed modes and S-curve ramps (vehicle YAML)

```yaml
performance:
  v_straight: 0.70   # default (NORMAL) mode
  ...
  jerk: 0.5          # m/s3: S-curve ramps. 0 = linear ramps (trapezoid)
  pivot_jerk_dps3: 60
speed_modes:
  - {name: SLOW,       v_straight: 0.40, v_curve: 0.30, v_reverse: 0.20, accel: 0.25, decel: 0.25}
  - {name: EXTRA_SLOW, v_straight: 0.15, v_curve: 0.15, v_reverse: 0.10, accel: 0.15, decel: 0.15}
```

With `jerk > 0` the profile generator works like a motion controller with look-ahead: the acceleration ramps
at `jerk`, and braking starts early enough to reach every lower limit ahead (curve, slower mode, stop) exactly at
its start with zero acceleration. Speeds, ramps and limits apply to the **drive axle** (the speed the motor
controller commands), also for tape tracking where the sensor point runs faster than the axle in curves.

## Tracking mode (important for tape AGVs)

`tracking.forward` / `tracking.reverse` say **which point of the vehicle follows the drawn line**:

* `axle` — the drive-axle centre follows the path (typical for a natural-navigation AMR driving its own planned path).
* a sensor name, e.g. `tape_front` — that sensor follows the tape and the body trails behind it (tractrix).
  The axle therefore cuts **inside** every curve; the report shows the drawn radius (`R_path`) next to the radius
  the axle and inner wheel actually drive (`R_axle`, `R_inWhl`) and the swept radii of the body.
  With a sensor far ahead of the axle, a tape radius smaller than about the sensor offset becomes impossible
  and is flagged.

## What the report contains

* Cycle time split into driving / pivoting / dwelling, cycles per hour, per-segment timeline.
* Per curve: path radius, real axle radius, inner-wheel radius, outer/inner swept radii, swept width,
  curve speed and time, trailer off-tracking.
* Pivot swept radius (clear circle needed to spin in place), swept floor area.
* Tip-over safety factor of the AGV (support polygon of drive wheels + castors, CG height) and of a towed cart
  (wheel track, CG height). 1.0 = tips, the report warns below 2.
* Towed cart: articulation angle over time and jackknife warning (`cart.max_articulation_deg`).
* Drivetrain per motor: peak rpm vs max, peak torque vs peak, RMS torque vs rated (thermal load),
  peak electrical power, tractive force, grip usage (longitudinal + lateral tyre force / (µ·N)), drawbar pull,
  castor swivel force, energy per cycle, cycles and hours per charge.
* Design checks: motor+gearbox fit in the chassis width (coaxial layout), lurking hook stroke vs cart clearance,
  tow hitch vs drawbar height, sensors outside the outline.
* Map clearance: collisions (body, and cart) and places closer than `simulation.safety_margin`.

## Model — and its limits (read before trusting numbers)

* **Motion, two modes.**
  * *Closed loop* (`line_follow.enabled: true`, all AGV_Kayaba vehicles): the drawn route is the tape. Every
    10 ms the simulation reads an MGS1600 model (1 mm steps, `sensor_noise_mm`, a delay of exactly
    `sensor_latency_s`, rounded to 10 ms with a note), runs your controller (`kayaba_pid.py`, a 1:1 port of
    modes.py / core/pid.py) and drives a per-wheel plant: AO volts → rpm map → first-order motor lag
    `motor_time_constant` (integrated exactly, stable for any value), `brake_decel` when the brake is on.
    Speed modes and stops come from RFID tags (read in both directions on a shared tape). Tape error, tape loss
    and the overshoot past a stop tag are real outputs of this loop.
  * *Kinematic* (line following off): unicycle model with a jerk-limited S-curve or trapezoid speed profile
    (forward/backward pass) limited by the mode speeds, `v_curve`, lateral acceleration `sqrt(a_lat / κ)` and
    the motor rpm limit.
* **Trolley weight and speed.** With `simulation.load_effects: false` (the default, *Trolley load calc* off)
  the trolley / cart weight does **not** change speed or timing - it only enters the forces, torque, grip and
  energy (the report says so). With it on (`drive.py`) the drive is force-limited: per drive wheel by its
  motor and by its grip, the grip shrinking with the cornering side force (friction circle); the trolleys add
  mass and rolling drag and have no brakes; slopes add M·g·sin.
* **Loads** (loads.py) are computed *after* the motion, from it: rigid-body force and yaw balance about the
  drive axle - chassis and carried-cart inertia incl. centripetal / yaw terms (CG from `drive_load_fraction`),
  rolling resistance of drive wheels and of each castor along its own heading, **castor swivel** (tyre twist /
  trail spike at the start of a pivot or after reversing), drive-tyre twist, **drawbar force** through the
  whole trolley train (each trolley's speed taken from the point it hangs on), **floor slopes** (`grade`, also
  in the tip-over check: on a ramp gravity shifts the tipping margin). In closed loop the AGV's speed, yaw
  rate and acceleration come straight from the plant (no double differentiation of noisy positions). Yaw
  acceleration is filtered by `yaw_response_time`; `wheels.scrub_factor` scales the tyre-twist effects.
* **Grip.** Needing more than the available grip (μ·N) is **reported, not simulated**: warning with the
  spots (red *SLIP* rings on the map). The motion still assumes the full ramps unless *Trolley load calc* is on,
  so a slipping run is optimistic. Tyre slip itself is not modelled.
* Not modelled: load transfer under braking, floor joints, regenerative braking, trolley yaw inertia, trailer
  slip (single-axle kinematic trailers, tractrix of the hitch point), bumpers / lidar fields / people (the
  collision check is body and trolley rectangles against dark obstacle pixels and no-go areas).
* **Uncalibrated inputs.** Values marked `# TODO` in the vehicle YAML (and sensors whose note says TODO) are
  listed at the end of the report and in Results - they are placeholders; the numbers depend on them.

Run `python -m pytest tests` (or `python tests/test_physics.py`; add `-m "not slow"` / `--fast` to skip the
auto-route and auto-tune tests) to check the model against hand formulas. `python tools/baseline.py --save
base.json` records cycle time, tape error, stop overshoot, torque, energy, tip-over, articulation, collisions,
grip and trolley push for every route; `--compare base.json` shows what a change moved.

"Perfect" comes from calibration, not from the model: run the real AGV on a measured stretch
(stopwatch or logged encoder/CAN data), compare the time and speed trace with `timeseries.csv`,
then tune `accel`, `decel`, `v_curve`, `stop_settle_time` and the pivot values until they match.
After that the simulator predicts new layouts and parameter changes reliably.

## Values to measure for AGV2 (marked TODO in `configs/agv2.yaml`)

For the closed-loop run the most important ones are: `line_follow.motor_time_constant` (log the wheel speed
after an AO step), `line_follow.brake_decel` (measure how far past a stop tag the AGV stops), the RFID antenna
position (sensor `rfid`), `line_follow.sensor_noise_mm` (std-dev of left_mm while standing on the tape) and
whether a positive left_mm means tape right or left (`front_sensor_positive`). Then: mass incl. battery,
weight split and CG height, castor y position and swivel offset, trolley geometry (pin -> trolley fixed
axle = `tongue_length`), motor peak torque, battery capacity, tyre friction.

Check the model against a real run: compare `results/<route>/timeseries.csv` (tape_pv_mm, pid_left_rpm,
pid_right_rpm) with the RunRecorder CSV in `analysis_plot/` for the same stretch of tape and speed.

## Package layout

```
main.py                 CLI (agv | edit | run | sweep | turn | autoroute | import-profile)
agv_sim/config.py       dataclasses + strict YAML loader (unknown keys are errors), set_param("a.b", v)
agv_sim/factory_map.py  PNG -> occupancy
agv_sim/drive.py        drive force limits: grip / motor torque vs mass, rolling drag, slope, brake
agv_sim/sequence.py     sequences: several route + vehicle steps in a row
agv_sim/combined.py     several routes on one map: tape + stop areas (Combined tab, tools/combined_map.py)
agv_sim/pidtune.py      PID auto-tune (Kp / Td per gain set) on a route
agv_sim/autoroute.py    auto route: lane planning on a grid, radii / speed tags, verify-and-fix loop
agv_sim/path.py         waypoints -> runs with fillet arcs, pivots, reverse legs
agv_sim/planner.py      speed limits, trapezoid and jerk-limited S-curve profiles, pivot profile
agv_sim/simulate.py     timeline, axle/sensor tracking, speed modes, cart attach/detach, trailer / lurking cart
agv_sim/loads.py        wheel forces: inertia, castors (rolling + swivel), drawbar, slopes, grip, tip-over
agv_sim/turning.py      turning-radius study (arc radii vs pivot)
agv_sim/linefollow.py   closed-loop tape following: MGS1600 model, RFID tags, BLDC drive plant
agv_sim/kayaba_pid.py   your controller (core/pid.py + modes.py logic), ported 1:1
agv_sim/profile_import.py  profiles/<agv>.json -> line_follow
agv_sim/analysis.py     swept path, curve table, collisions, drivetrain & battery checks, text report
agv_sim/visual.py       spec sheet, report figure, animation (GIF/MP4)
agv_sim/editor.py       interactive matplotlib route editor
configs/                agv2.yaml (AGV2 KIM2A, 84 cm, closed-loop PID on), tow_tractor_example.yaml
                        (earlier 157 cm layout), lurking_amr_example.yaml
tests/test_physics.py   sanity checks against hand calculations
```

Python API:

```python
from agv_sim import load_agv, load_route, run_simulation, format_report, set_param
agv = load_agv("configs/agv2.yaml"); set_param(agv, "performance.v_curve", 0.4)
res, summary = run_simulation(agv, load_route("routes/demo_route.yaml"))
print(format_report(summary)); print(summary["cycle_time_s"])
```
