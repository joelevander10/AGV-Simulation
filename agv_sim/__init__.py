"""AGV route simulator: cycle time, turning radius, swept path, drivetrain loads."""
import math

from .config import AGVConfig, RouteConfig, load_agv, load_route, save_route, set_param
from .factory_map import FactoryMap
from .path import build_plan
from .simulate import simulate
from .analysis import analyze, format_report

__all__ = ["AGVConfig", "RouteConfig", "load_agv", "load_route", "save_route", "set_param",
           "FactoryMap", "build_plan", "simulate", "analyze", "format_report", "run_simulation"]


def run_simulation(agv, route, fmap=None, steady_state=True):
    """Plan + simulate + analyze. Returns (result, summary).

    For a looped route with a towed cart, the cart usually has not straightened out when the
    AGV arrives back home. With steady_state=True a first lap is simulated only to get the
    cart angle at the end of the cycle, and the reported lap starts from that angle.
    """
    if fmap is None:
        fmap = FactoryMap.for_route(route)
    plan = build_plan(route, agv)
    res = simulate(agv, route, plan)
    if steady_state and route.loop and res.articulation_deg is not None \
            and bool(res.attached[0]) and bool(res.attached[-1]) \
            and not (res.lf is not None and res.lf["aborted"]):
        a_end = [float(tr["art"][-1]) for tr in res.train]
        if all(math.isfinite(a) for a in a_end) and max(abs(a) for a in a_end) > 0.5:
            res = simulate(agv, route, plan, cart_articulation0_deg=a_end)
    summary = analyze(res, fmap)
    return res, summary
