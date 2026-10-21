"""Configuration: AGV specification, route and map. Loaded from / saved to YAML.

Coordinate conventions used everywhere in this package
------------------------------------------------------
Body frame (all AGV geometry in the AGV YAML):
    origin = centre point between the two drive wheels (drive-axle centre)
    +x = forward (towards the front bumper), +y = left, +z = up from the floor
World frame (map and route):
    metres, origin at the LOWER-LEFT corner of the map image, +x right, +y up
"""
from __future__ import annotations

import math
import os
import typing
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional

import yaml

G = 9.81  # m/s^2


# =============================================================================
# AGV specification
# =============================================================================
@dataclass
class Chassis:
    length: float = 1.50              # m, bumper to bumper
    width: float = 0.55               # m
    height: float = 0.35              # m, top of chassis above floor
    ground_clearance: float = 0.03    # m, underside of chassis above floor
    axle_from_rear: float = 0.30      # m, rear bumper -> drive-axle centre
    mass_kg: float = 120.0            # empty AGV incl. battery
    drive_load_fraction: float = 0.6  # share of AGV weight on the two drive wheels
    cg_height: float = 0.20           # m, centre of gravity above floor (tip-over check)


@dataclass
class Castor:
    x: float = 1.0          # body frame, m
    y: float = 0.0
    diameter: float = 0.10
    width: float = 0.05
    trail: float = 0.035    # m, swivel offset (kingpin -> wheel axle); 0 = ignore castor swivel


@dataclass
class Wheels:
    diameter: float = 0.18        # drive wheel
    width: float = 0.06
    track: float = 0.39           # centre-to-centre distance of the drive wheels
    friction_coeff: float = 0.5   # tyre/floor, PU on concrete ~0.5-0.7
    scrub_factor: float = 1.0     # tyre twist (bore) torque multiplier: drive wheels + castor swivel; 0 = off
    castors: List[Castor] = field(default_factory=list)


@dataclass
class Drivetrain:
    motor_rated_power_w: float = 200.0
    motor_rated_rpm: float = 3000.0
    motor_max_rpm: float = 3000.0
    motor_peak_torque_factor: float = 3.0   # peak / rated torque (motor datasheet)
    motor_efficiency: float = 0.85
    gear_ratio: float = 30.0
    gear_efficiency: float = 0.90
    layout: str = "right_angle"   # coaxial | right_angle | parallel (drawing + fit check)
    motor_diameter: float = 0.08
    motor_length: float = 0.12
    gearbox_length: float = 0.08
    battery_voltage: float = 24.0
    battery_capacity_ah: float = 40.0
    battery_usable_fraction: float = 0.8
    aux_power_w: float = 40.0     # controller, sensors, PC, lights

    @property
    def rated_torque(self) -> float:
        """Continuous motor torque, N*m (from rated power and rated speed)."""
        return self.motor_rated_power_w / (self.motor_rated_rpm * 2 * math.pi / 60)

    @property
    def peak_torque(self) -> float:
        return self.rated_torque * self.motor_peak_torque_factor


@dataclass
class Performance:
    v_straight: float = 0.8        # m/s, cruise speed forward on straights
    v_curve: float = 0.4           # m/s, speed cap on any curve
    v_reverse: float = 0.3         # m/s, cap when driving in reverse
    accel: float = 0.3             # m/s^2
    decel: float = 0.3             # m/s^2
    max_lateral_accel: float = 0.5  # m/s^2, also limits curve speed: v <= sqrt(a_lat * R)
    min_turn_radius: float = 0.5   # m, curves tighter than this raise a warning
    pivot_speed_dps: float = 30.0  # deg/s, spin-in-place
    pivot_accel_dps2: float = 30.0  # deg/s^2
    jerk: float = 0.0              # m/s^3, S-curve ramps; 0 = plain trapezoid (instant accel steps)
    pivot_jerk_dps3: float = 0.0   # deg/s^3 for spin-in-place; 0 = trapezoid
    stop_settle_time: float = 0.0  # s added at every stop (brake, positioning)


@dataclass
class SpeedMode:
    """Named speed mode (like NORMAL / SLOW / EXTRA_SLOW on the real controller).

    A route leg uses it with `mode: SLOW` on the waypoint the leg starts from."""
    name: str = "SLOW"
    v_straight: float = 0.4
    v_curve: float = 0.3
    v_reverse: float = 0.2
    accel: float = 0.25
    decel: float = 0.25


@dataclass
class Sensor:
    name: str = "sensor"
    type: str = "generic"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    note: str = ""


@dataclass
class Hook:
    type: str = "none"      # none | tow | lurking
    x: float = -0.45        # body frame, m (negative = behind drive axle)
    y: float = 0.0
    z: float = 0.20
    lift_stroke: float = 0.0  # lurking: lift height of the pin/plate
    attach_time: float = 0.0  # s to couple / lift a cart at a station with 'cart: attach'
    detach_time: float = 0.0  # s to uncouple / lower a cart at a station with 'cart: detach'


@dataclass
class Cart:
    length: float = 1.2
    width: float = 0.8
    height: float = 0.8
    mass_kg: float = 0.0            # total towed / carried mass (cart + load)
    rolling_resistance: float = 0.02
    tongue_length: float = 1.2      # tow: hitch eye -> cart FIXED-wheel axle
    axle_from_rear: float = 0.2     # tow: fixed axle measured from cart rear edge
    hitch_height: float = 0.20      # tow: drawbar eye height
    underside_clearance: float = 0.40  # lurking: free height under the cart
    max_articulation_deg: float = 80.0
    cg_height: float = 0.60         # m, loaded cart centre of gravity (tip-over check)
    wheel_track: float = 0.70       # m, cart wheel spacing across (tip-over check)
    count: int = 1                  # tow: number of trolleys in the train (all the same size)
    masses: List[float] = field(default_factory=list)  # tow: kg of each trolley incl. load (missing -> mass_kg)
    coupling_behind_axle: float = 0.30  # tow: fixed axle -> rear coupling where the next trolley hooks on
    empty_mass_kg: float = 60.0     # kg of ONE trolley / cart without load ('empty' at a +cart / swap point)


@dataclass
class Tracking:
    """Which point of the AGV follows the drawn path.

    'axle'        -> the drive-axle centre follows the path exactly (QR / SLAM / lidar AMR)
    <sensor name> -> that sensor follows the path (magnetic tape: the tape IS the path);
                     the axle then cuts the inside of every curve like a real tape AGV.
    """
    forward: str = "axle"
    reverse: str = "axle"


@dataclass
class LineFollow:
    """Closed-loop tape following: MGS1600 sensor model + the AGV's own PID code (core/pid.py,
    modes.py auto_mode / reverse_mode of 'AGV Kayaba (2 Sensor)') + motor/driver model.

    Values mirror profiles/<agv>.json (pid_tuning, speeds, hardware, can_sensor, feedforward).
    Import them with:  python main.py import-profile --json <agv2_kim.json> --agv configs/agv2.yaml
    """
    enabled: bool = False
    # ---- PID gains (pid_tuning) ------------------------------------------------------------
    kp: float = 1.4                  # HIGH
    td: float = 0.12
    n: float = 20.0
    kp_slow: float = 2.2             # SLOW (and every mode that is not HIGH / APPROACH)
    td_slow: float = 0.14
    n_slow: float = 12.0
    kp_approach: float = 1.6
    td_approach: float = 0.12
    n_approach: float = 12.0
    kp_reverse: float = 2.2
    td_reverse: float = 0.14
    n_reverse: float = 12.0
    ti: float = 20.0
    ti_deadband: float = 25.0        # mm: integrate only while |e| < deadband
    ti_max: float = 500.0
    dt: float = 0.01                 # s, control loop period (= sensor frame period)
    v_red_coef: float = 4.0
    v_red_coef_slow: float = 2.0
    v_red_coef_approach: float = 0.0
    v_red_coef_reverse: float = 2.0
    output_clamp_rpm: float = 800.0
    output_clamp_into_bar_approach: float = 800.0
    output_clamp_away_approach: float = 800.0
    sr_alpha: float = 0.15
    sr_cap: float = 0.55
    error_sign_forward: float = 1.0
    error_sign_reverse: float = -1.0  # hard-coded in reverse_mode
    # ---- speeds (speeds) ---------------------------------------------------------------
    speed_high: float = 1.0          # AUTO_TARGET_HIGH_SPEED
    speed_slow: float = 0.2          # AUTO_TARGET_SLOW_SPEED
    speed_approach: float = 0.1      # AUTO_TARGET_APPROACH_SPEED
    speed_reverse: float = 0.15      # AUTO_TARGET_REVERSE_SPEED
    accel_rate: float = 0.8
    decel_rate: float = 0.2
    approach_decel_rate: float = 0.4
    start_mode: str = "SLOW"         # speed_mode after START / RESET (state.py default)
    # ---- motor map (hardware) + AO -------------------------------------------------------
    rpm_per_volt_left: float = 646.59
    rpm_volt_offset_left: float = -101.2
    rpm_per_volt_right: float = 646.59
    rpm_volt_offset_right: float = -101.2
    ao_max_volt: float = 5.0         # motion.rpm_to_voltage clamps to [0, 5] V
    ao_range_volt: float = 10.0      # V_RANGE
    dac_res: int = 4095              # DAC_RES
    # ---- sensor (can_sensor + MGS1600 datasheet) ----------------------------------------
    sensor_max_mm: float = 100.0     # frames beyond this are skipped (hold last command)
    sensor_max_step_mm: float = 40.0
    sensor_range_mm: float = 80.0    # tape centre detected within +-range (160 mm sensing width)
    sensor_noise_mm: float = 0.0     # std-dev of measurement noise
    sensor_latency_s: float = 0.01   # CAN + scheduling delay
    front_sensor_positive: str = "right"  # which side a POSITIVE left_mm means on the front sensor
                                          # (rear sensor is the same unit facing backwards = mirrored)
    # ---- curvature feed-forward (feedforward) -------------------------------------------
    ff_enabled: bool = False
    ff_track_width: float = 0.46
    ff_curve_radius: float = 1.5
    ff_direction: str = "left"
    ff_curve_modes: List[str] = field(default_factory=lambda: ["SLOW"])
    ff_alpha: float = 0.15
    ff_exit_alpha: float = 0.15
    ff_scale: float = 1.0
    # ---- drive plant (not in the profile: measure / tune) --------------------------------
    motor_time_constant: float = 0.12  # s, BLDC driver speed response to an AO step
    brake_decel: float = 1.5           # m/s2 when the brake DO is on (tape loss, stop tag)
    plant_scale_left: float = 1.0      # real / calibrated rpm (1.0 = calibration is exact)
    plant_scale_right: float = 1.0
    lost_abort_s: float = 1.5          # stopped without tape this long -> operator needed, cycle aborted


@dataclass
class Simulation:
    dt: float = 0.02                # s, output time step
    ds: float = 0.01                # m, path sampling
    safety_margin: float = 0.10     # m, clearance required to obstacles
    rolling_resistance: float = 0.015
    collision_spacing: float = 0.05  # m, footprint sample grid for collision checks
    yaw_response_time: float = 0.3   # s, smooths yaw-rate steps (line->arc) for the inertia load
    load_effects: bool = False       # True: trolley mass / grip limit speeding up and braking (drive.py)


@dataclass
class AGVConfig:
    name: str = "AGV"
    chassis: Chassis = field(default_factory=Chassis)
    wheels: Wheels = field(default_factory=Wheels)
    drivetrain: Drivetrain = field(default_factory=Drivetrain)
    performance: Performance = field(default_factory=Performance)
    speed_modes: List[SpeedMode] = field(default_factory=list)
    sensors: List[Sensor] = field(default_factory=list)
    hook: Hook = field(default_factory=Hook)
    cart: Cart = field(default_factory=Cart)
    tracking: Tracking = field(default_factory=Tracking)
    line_follow: LineFollow = field(default_factory=LineFollow)
    simulation: Simulation = field(default_factory=Simulation)

    # ---- derived values -------------------------------------------------
    @property
    def wheel_radius(self) -> float:
        return self.wheels.diameter / 2

    @property
    def max_wheel_speed(self) -> float:
        """Linear wheel speed at motor_max_rpm, m/s."""
        d = self.drivetrain
        return d.motor_max_rpm / d.gear_ratio * math.pi * self.wheels.diameter / 60

    @property
    def x_rear(self) -> float:
        return -self.chassis.axle_from_rear

    @property
    def x_front(self) -> float:
        return self.chassis.length - self.chassis.axle_from_rear

    @property
    def hook_type(self) -> str:
        return (self.hook.type or "none").lower()

    def mode_params(self, name: str = "") -> dict:
        """Speeds / ramps of a named speed mode ('' or 'default' -> the performance section)."""
        p = self.performance
        base = {"name": "default", "v_straight": p.v_straight, "v_curve": p.v_curve,
                "v_reverse": p.v_reverse, "accel": p.accel, "decel": p.decel}
        key = (name or "").strip()
        if not key or key.lower() == "default":
            return base
        for m in self.speed_modes:
            if m.name.strip().lower() == key.lower():
                out = {"name": m.name, "v_straight": m.v_straight, "v_curve": m.v_curve,
                       "v_reverse": m.v_reverse, "accel": m.accel, "decel": m.decel}
                for k in ("v_straight", "v_reverse", "accel", "decel"):
                    if out[k] <= 0:
                        raise ValueError(f"Speed mode '{m.name}': {k} must be > 0")
                return out
        raise ValueError(f"Unknown speed mode '{name}'. Defined modes: "
                         f"{['default'] + [m.name for m in self.speed_modes]}")

    def train_mass(self, load: str = "full") -> float:
        """kg of the whole train / cart: 'full' = the loaded weights, 'empty' = empty_mass_kg each."""
        ms = self.train_masses()
        if (load or "full").strip().lower() == "empty":
            return len(ms) * float(self.cart.empty_mass_kg)
        return float(sum(ms))

    def train_masses(self) -> List[float]:
        """kg of every towed trolley (1 for lurking / a single cart)."""
        c = self.cart
        n = max(1, int(c.count)) if self.hook_type == "tow" else 1
        ms = [float(m) for m in (c.masses or [])][:n]
        return ms + [float(c.mass_kg)] * (n - len(ms))

    @property
    def cg_x(self) -> float:
        """CG position along x, from the load split between drive wheels (x=0) and castors."""
        cs = self.wheels.castors
        if not cs:
            return 0.0
        xc = sum(c.x for c in cs) / len(cs)
        return (1.0 - self.chassis.drive_load_fraction) * xc

    def sensor(self, name: str) -> Sensor:
        for s in self.sensors:
            if s.name == name:
                return s
        raise ValueError(f"No sensor named '{name}'. Known sensors: {[s.name for s in self.sensors]}")

    def tracking_offset(self, direction: int) -> float:
        """Distance (m) from drive axle to the tracking point, measured along the travel direction."""
        name = self.tracking.forward if direction > 0 else self.tracking.reverse
        if not name or name.lower() == "axle":
            return 0.0
        s = self.sensor(name)
        d = s.x if direction > 0 else -s.x
        if d <= 0.01:
            which = "forward" if direction > 0 else "reverse"
            raise ValueError(f"Tracking sensor '{name}' for {which} travel must be ahead of the drive "
                             f"axle in that travel direction (x={s.x}).")
        return d


# =============================================================================
# Route / map
# =============================================================================
@dataclass
class MapConfig:
    image: str = ""                 # PNG/JPG; empty -> blank 30 x 20 m floor
    meters_per_pixel: float = 0.02
    obstacle_threshold: int = 100   # grey value (0-255) below which a pixel is an obstacle


@dataclass
class Waypoint:
    x: float = 0.0
    y: float = 0.0
    name: str = ""
    action: str = "pass"            # pass | stop | pivot
    dwell: float = 0.0              # s, standing time at a stop
    radius: Optional[float] = None  # corner radius here; None -> route default; 0 -> sharp corner (pivot)
    speed: Optional[float] = None   # speed cap for the leg LEAVING this waypoint
    direction: str = "forward"      # direction of the leg LEAVING this waypoint
    mode: str = ""                  # speed mode of the leg LEAVING this waypoint ('' = route default)
    grade: float = 0.0              # % floor slope of the leg LEAVING this waypoint (+ = uphill towards next point)
    cart: str = ""                  # '' | attach | detach | swap (drop the trolleys, hitch others)  (implies a stop)
    cart_load: str = ""             # attach / swap: full | empty ('' = full)
    cart_mass: Optional[float] = None  # kg of the cart(s) picked up here - overrides cart_load
    fork: str = ""                  # '' | left | right: RFID track-selection tag here (which branch to follow
                                    #   at the next fork) - for the layout / installation guide; the drawn path decides


@dataclass
class RouteConfig:
    name: str = "route"
    map: MapConfig = field(default_factory=MapConfig)
    loop: bool = True               # last waypoint connects back to the first
    default_radius: float = 1.0
    default_mode: str = ""          # speed mode for legs without their own 'mode'
    waypoints: List[Waypoint] = field(default_factory=list)
    blocked: List[List[float]] = field(default_factory=list)   # no-go rectangles [x0, y0, x1, y1] (m)
    start_load: str = "full"        # trolleys already on the hook at the start: full | empty
    target_cycle_s: float = 0.0     # your target cycle time (0 = none); auto route aims for it


# =============================================================================
# YAML helpers
# =============================================================================
def _convert(tp, value, where: str):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if value is None:
        if is_dataclass(tp):
            return tp()
        if origin in (list, List):
            return []
        return None
    if origin is typing.Union:
        inner = [a for a in args if a is not type(None)][0]
        return _convert(inner, value, where)
    if origin in (list, List):
        if not isinstance(value, list):
            raise ValueError(f"'{where[:-1]}' must be a list")
        return [_convert(args[0], v, where) for v in value]
    if is_dataclass(tp):
        return _from_dict(tp, value, where)
    if tp is float:
        return float(value)
    if tp is int:
        return int(value)
    if tp is bool:
        return bool(value)
    if tp is str:
        return str(value)
    return value


def _from_dict(cls, data, where: str = ""):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"'{where[:-1] or cls.__name__}' must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in known:
            raise ValueError(f"Unknown setting '{where}{key}'. Valid keys here: {sorted(known)}")
        kwargs[key] = _convert(hints[key], val, f"{where}{key}.")
    return cls(**kwargs)


def load_agv(path: str) -> AGVConfig:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    agv = _from_dict(AGVConfig, yaml.safe_load(text) or {})
    agv._todo_fields = todo_fields(text)          # not a dataclass field: never saved back
    return agv


def todo_fields(text: str) -> List[str]:
    """Values marked '# TODO' in a vehicle YAML (placeholders still to be measured), as 'section.key: note'.
    The YAML loader drops comments, so the report would not know about them otherwise."""
    out = []
    section = ""
    for ln in text.splitlines():
        m0 = re.match(r"^([A-Za-z_][\w]*)\s*:", ln)
        if m0:
            section = m0.group(1)
        if "TODO" not in ln or ln.lstrip().startswith("#") or "note:" in ln:
            continue
        code, _, comment = ln.partition("#")
        m = re.match(r"^\s*-?\s*\{?\s*([A-Za-z_][\w]*)\s*:", code)
        if not m:
            continue
        key = m.group(1)
        name = f"{section}.{key}" if section and key != section else key
        out.append(f"{name}: {comment.replace('TODO', '').strip() or 'placeholder'}")
    return out


def load_route(path: str) -> RouteConfig:
    with open(path, "r", encoding="utf-8") as f:
        route = _from_dict(RouteConfig, yaml.safe_load(f) or {})
    img = route.map.image
    if img and not os.path.isabs(img):
        route.map.image = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), img))
    return route


def save_route(route: RouteConfig, path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    img = route.map.image
    if img:
        try:
            img = os.path.relpath(img, folder)
        except ValueError:  # different drive on Windows
            pass
    default_wp = Waypoint()
    wps = []
    for wp in route.waypoints:
        d = {"x": round(float(wp.x), 3), "y": round(float(wp.y), 3)}
        for f in fields(Waypoint):
            if f.name in ("x", "y"):
                continue
            val = getattr(wp, f.name)
            if val != getattr(default_wp, f.name):
                d[f.name] = round(val, 3) if isinstance(val, float) else val
        wps.append(d)
    data = {
        "name": route.name,
        "map": {"image": img.replace("\\", "/"),
                "meters_per_pixel": float(route.map.meters_per_pixel),
                "obstacle_threshold": int(route.map.obstacle_threshold)},
        "loop": bool(route.loop),
        "default_radius": float(route.default_radius),
        **({"default_mode": route.default_mode} if route.default_mode else {}),
        "waypoints": wps,
        **({"blocked": [[round(float(v), 3) for v in b] for b in route.blocked]} if route.blocked else {}),
        **({"start_load": route.start_load} if (route.start_load or "full") != "full" else {}),
        **({"target_cycle_s": float(route.target_cycle_s)} if route.target_cycle_s else {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, width=120)


def set_param(cfg, dotted: str, value):
    """Set e.g. 'performance.v_curve' on a config object (used by --set and the sweep)."""
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise ValueError(f"Unknown parameter '{dotted}'")
        obj = getattr(obj, p)
    last = parts[-1]
    if not hasattr(obj, last):
        raise ValueError(f"Unknown parameter '{dotted}'")
    cur = getattr(obj, last)
    if isinstance(cur, list) and isinstance(value, str):
        value = [float(v) for v in value.replace(";", ",").split(",") if v.strip()]
    elif isinstance(cur, bool):
        value = str(value).lower() in ("1", "true", "yes", "on")
    elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
        value = type(cur)(float(value))
    setattr(obj, last, value)


def to_dict(cfg) -> dict:
    return asdict(cfg)
