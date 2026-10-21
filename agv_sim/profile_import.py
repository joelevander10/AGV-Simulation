"""Import controller settings from the AGV's own profile JSON (profiles/<agv>.json of
'AGV Kayaba (2 Sensor)') into the simulator's line_follow section.

    python main.py import-profile --json ".../profiles/agv2_kim.json" --agv configs/agv2.yaml
"""
from __future__ import annotations

import json
import re

# (json section, json key) -> line_follow field ; defaults follow config.py of the AGV code
MAP = [
    ("pid_tuning", "KP", "kp"), ("pid_tuning", "TD", "td"), ("pid_tuning", "N", "n"),
    ("pid_tuning", "KP_SLOW", "kp_slow"), ("pid_tuning", "TD_SLOW", "td_slow"), ("pid_tuning", "N_SLOW", "n_slow"),
    ("pid_tuning", "KP_APPROACH", "kp_approach"), ("pid_tuning", "TD_APPROACH", "td_approach"),
    ("pid_tuning", "N_APPROACH", "n_approach"),
    ("pid_tuning", "KP_REVERSE", "kp_reverse"), ("pid_tuning", "TD_REVERSE", "td_reverse"),
    ("pid_tuning", "N_REVERSE", "n_reverse"),
    ("pid_tuning", "TI", "ti"), ("pid_tuning", "TI_DEADBAND", "ti_deadband"), ("pid_tuning", "TI_MAX", "ti_max"),
    ("pid_tuning", "DT", "dt"), ("pid_tuning", "V_RED_COEF", "v_red_coef"),
    ("pid_tuning", "V_RED_COEF_SLOW", "v_red_coef_slow"), ("pid_tuning", "V_RED_COEF_APPROACH", "v_red_coef_approach"),
    ("pid_tuning", "V_RED_COEF_REVERSE", "v_red_coef_reverse"),
    ("pid_tuning", "OUTPUT_CLAMP_RPM", "output_clamp_rpm"),
    ("pid_tuning", "OUTPUT_CLAMP_INTO_BAR_APPROACH", "output_clamp_into_bar_approach"),
    ("pid_tuning", "OUTPUT_CLAMP_AWAY_APPROACH", "output_clamp_away_approach"),
    ("pid_tuning", "SR_ALPHA", "sr_alpha"), ("pid_tuning", "SR_CAP", "sr_cap"),
    ("pid_tuning", "ERROR_SIGN_FORWARD", "error_sign_forward"),
    ("speeds", "AUTO_TARGET_HIGH_SPEED", "speed_high"), ("speeds", "AUTO_TARGET_SLOW_SPEED", "speed_slow"),
    ("speeds", "AUTO_TARGET_APPROACH_SPEED", "speed_approach"),
    ("speeds", "AUTO_TARGET_REVERSE_SPEED", "speed_reverse"),
    ("speeds", "ACCEL_RATE", "accel_rate"), ("speeds", "DECEL_RATE", "decel_rate"),
    ("speeds", "APPROACH_DECEL_RATE", "approach_decel_rate"),
    ("hardware", "V_RANGE", "ao_range_volt"), ("hardware", "DAC_RES", "dac_res"),
    ("can_sensor", "SENSOR_MAX_MM", "sensor_max_mm"), ("can_sensor", "SENSOR_MAX_STEP_MM", "sensor_max_step_mm"),
    ("feedforward", "ENABLED", "ff_enabled"), ("feedforward", "TRACK_WIDTH_M", "ff_track_width"),
    ("feedforward", "CURVE_RADIUS_M", "ff_curve_radius"), ("feedforward", "CURVE_DIRECTION", "ff_direction"),
    ("feedforward", "CURVE_SPEED_MODES", "ff_curve_modes"), ("feedforward", "FF_ALPHA", "ff_alpha"),
    ("feedforward", "FF_EXIT_ALPHA", "ff_exit_alpha"), ("feedforward", "FF_SCALE", "ff_scale"),
]


def profile_to_line_follow(data: dict) -> dict:
    """Return {line_follow_field: value} for everything present in the profile."""
    out = {}
    for sec, key, fld in MAP:
        d = data.get(sec) or {}
        if key in d:
            v = d[key]
            if fld == "ff_enabled":
                v = bool(v)
            out[fld] = v
    pt = data.get("pid_tuning") or {}
    # config.py fallbacks: reverse gains default to the SLOW gains
    for fld, base in (("kp_reverse", "KP_SLOW"), ("td_reverse", "TD_SLOW"), ("n_reverse", "N_SLOW"),
                      ("v_red_coef_reverse", "V_RED_COEF_SLOW")):
        if fld not in out and base in pt:
            out[fld] = pt[base]
    hw = data.get("hardware") or {}
    k = hw.get("RPM_PER_VOLT", 646.59)
    o = hw.get("RPM_VOLT_OFFSET", -101.2)
    out["rpm_per_volt_left"] = hw.get("RPM_PER_VOLT_LEFT", k)
    out["rpm_volt_offset_left"] = hw.get("RPM_VOLT_OFFSET_LEFT", o)
    out["rpm_per_volt_right"] = hw.get("RPM_PER_VOLT_RIGHT", k)
    out["rpm_volt_offset_right"] = hw.get("RPM_VOLT_OFFSET_RIGHT", o)
    return out


def extras(data: dict) -> dict:
    """Other vehicle values the profile knows (dotted AGV paths)."""
    kin = data.get("kinematics") or {}
    out = {}
    if "WHEEL_DIAMETER" in kin:
        out["wheels.diameter"] = kin["WHEEL_DIAMETER"]
    if "GEAR_RATIO" in kin:
        out["drivetrain.gear_ratio"] = kin["GEAR_RATIO"]
    return out


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def update_yaml_text(text: str, values: dict, section: str = "line_follow") -> str:
    """Replace `key: value` lines inside a top-level YAML section, keeping comments/layout.
    Keys that are not in the file are appended to the section."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{section}:\s*(#.*)?$", ln)), None)
    if start is None:
        lines += ["", f"{section}:"] + [f"  {k}: {_fmt(v)}" for k, v in values.items()]
        return "\n".join(lines) + "\n"
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1
    done = set()
    for i in range(start + 1, end):
        m = re.match(r"^(\s+)([A-Za-z_]\w*):\s*([^#]*?)(\s*#.*)?$", lines[i])
        if m and m.group(2) in values:
            k = m.group(2)
            lines[i] = f"{m.group(1)}{k}: {_fmt(values[k])}{m.group(4) or ''}"
            done.add(k)
    add = [f"  {k}: {_fmt(v)}" for k, v in values.items() if k not in done]
    lines[end:end] = add
    return "\n".join(lines) + "\n"
