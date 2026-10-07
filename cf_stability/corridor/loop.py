"""Control loop of the corridor (docs/m5_contract.md, section 3): SUMO moves the vehicles, the law sets
their speeds.

Importing this module imports ``libsumo`` first (section 0): import it before pandas and torch. The
loop reads no parquet: the demand comes from ``vehicles_truth.npz`` of the scenario.

Per step, after ``simulationStep`` (the state then belongs to the time of that step, ``k * dt``):

* new vehicles: speed mode 0, subscription of speed, lane, lane position, road and of the leader
  within ``leader_range``; the member was drawn before the simulation, in the order of the planned
  departures, so it does not depend on what happens in the simulation;
* teleported vehicles are removed (``teleported`` marks those that had not passed ``x_out``, so that
  ``n_inserted = n_exited + n_in_network + n_teleports``); SUMO's collisions are collected as distinct
  pairs (with ``--collision.action warn`` SUMO reports an overlap every step);
* vehicles on the edges of the section (and on an off-ramp): state ``(s, dv, v)`` from the leader
  subscription (without a leader within range ``s = leader_range, dv = 0``; at the end of the auxiliary
  edge a standing virtual leader replaces a leader further away for every vehicle whose lane does not
  lead on along its route: on the auxiliary lane unless the route ends on the off-ramp that leaves it
  (I-80: it has no successor at all), on the other lanes when the route ends there); the history (filled
  with the first state of a new vehicle) goes to the law, the acceleration is clipped and
  ``setSpeed(v + a dt)`` (at least 0): SUMO's Euler update then moves the vehicle by ``v_next dt``, the
  integration scheme of the project;
* contact instead of removal: ``s <= 0`` begins a collision episode of the follower (counted once; a new
  one begins only after the gap has been above ``contact_gap`` in between; also for a vehicle that enters
  the buffer in contact). The vehicle stays: while in contact its commanded speed is capped,
  ``v_next = max(0, min(v_law, v_lead + s / dt))``, so that it cannot penetrate further. Removing it would
  drain the queue that the laws run into, and the section would show free flow instead of the collapse;
* vehicles on the buffer edges approach the target speed ``v_b(lane, t) * g(t)`` (``v_b``: speed of the
  data in that lane, nearest second) by at most ``max_decel`` / ``max_accel`` per second, so that a
  vehicle does not jump when it enters the buffer, but not faster than keeps ``min_gap`` to the vehicle
  ahead (the buffer vehicles do not follow each other otherwise), and without lane changes there unless
  ``lane_changes``. With ``feedback`` the gain
  ``g = clip(1 - gain * dN, g_min, g_max)`` follows ``dN(t)`` = simulated passages of ``x_out`` minus those
  of the data (``boundary.npz`` ``exits``, linear in time; the data passages of the vehicles that were
  removed from the simulation (teleports) are taken out of that count): a prescribed speed alone lets out the law's own
  flow at that speed, not the observed one, and the queue of the data drains or spills back; without
  feedback ``g = 1``. The settings are those of the scenario (``scenario.json["boundary"]``).

Positions are recorded at whole seconds for ``x_in <= x <= x_out`` on the main line (not on an off-ramp);
the passage of ``x_out`` is interpolated between two steps; every collision episode is recorded (time,
vehicle, position, lane of its beginning). The time mean, minimum and maximum of ``g`` and ``dN`` inside
the analysis window of the scenario go to the statistics of the run. A vehicle whose route ends on the
off-ramp leaves the network at its end (``n_offramp``): ``n_inserted = n_exited + n_offramp + n_in_network +
n_teleports``.
"""

from __future__ import annotations

from cf_stability.corridor.sumo_env import import_libsumo, short_path

libsumo = import_libsumo()  # before pandas and torch (docs/m5_contract.md, section 0)

import dataclasses  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Mapping  # noqa: E402

import numpy as np  # noqa: E402

from cf_stability.corridor.scenario import (  # noqa: E402
    COLLISION_DTYPES,
    TRAJECTORY_DTYPES,
    VEHICLE_DTYPES,
    BoundaryConfig,
    cast,
    exit_difference,
)
from cf_stability.utils import read_json  # noqa: E402

CONSTANTS = libsumo.constants
SUBSCRIBED = (CONSTANTS.VAR_SPEED, CONSTANTS.VAR_LANE_INDEX, CONSTANTS.VAR_LANEPOSITION, CONSTANTS.VAR_ROAD_ID)
# SUMO's leader search stops at the end of the lane when that lies beyond its look-ahead, so a leader whose back
# is within leader_range but whose front is already on the next edge (up to one vehicle length, 24 m) would be
# missed; the subscription looks further and the loop applies leader_range to the gap
LOOKAHEAD_MARGIN = 50.0
Observer = Callable[..., None]


@dataclasses.dataclass(frozen=True)
class SimConfig:
    """``run_corridor.sim`` (part of the config hash of a run)."""

    step_length: float = 0.1
    time_to_teleport: float = 300.0
    leader_range: float = 50.0  # m; without a leader within it: s = leader_range, dv = 0 (also the virtual leader)
    tail_s: float = 120.0  # simulated time after the last planned departure
    a_min: float = -8.0
    a_max: float = 4.0
    contact_gap: float = 1.0  # m: a collision episode ends when the gap is above it; below it the speed is capped
    sumo_options: tuple[str, ...] = ()  # further options of SUMO

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SimConfig:
        values = dict(raw or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown sim keys {sorted(unknown)}")
        if "sumo_options" in values:
            values["sumo_options"] = tuple(str(o) for o in values["sumo_options"] or ())
        return cls(**values)


@dataclasses.dataclass
class Scenario:
    """The files of a scenario directory that the loop needs."""

    directory: Path
    meta: dict[str, Any]
    demand: dict[str, np.ndarray]
    boundary_speed: np.ndarray  # [lanes, seconds]
    boundary_t: np.ndarray  # [seconds]
    exits: np.ndarray  # [seconds] data vehicles that have passed x_out by then

    @classmethod
    def load(cls, directory: str | Path) -> Scenario:
        directory = Path(directory)
        meta = read_json(directory / "scenario.json")
        with np.load(directory / "vehicles_truth.npz") as data:
            demand = {key: data[key] for key in data.files}
        with np.load(directory / "boundary.npz") as data:
            if "exits" not in data.files:
                raise ValueError(f"{directory / 'boundary.npz'} has no exits: rebuild the scenario")
            speed, t, exits = (np.asarray(data[key], dtype=np.float64) for key in ("speed", "t", "exits"))
        return cls(directory, meta, demand, speed, t, exits)

    @property
    def n_vehicles(self) -> int:
        return len(self.demand["vehicle_id"])

    @property
    def boundary(self) -> BoundaryConfig:
        return BoundaryConfig.from_mapping(self.meta.get("boundary"))


def sumo_command(scenario: Scenario, seed: int, sim: SimConfig) -> list[str]:
    """Options of the contract; warnings of emergency braking are off (they change nothing): the clip of the
    laws and the speeds prescribed on the buffer trigger them by design, thousands per run; beyond 20
    warnings of one type (vehicles in contact overlap for many steps) SUMO prints a count instead."""
    return [
        "sumo", "-n", short_path(scenario.directory / "net.net.xml"),
        "-r", short_path(scenario.directory / "routes.rou.xml"),
        "--step-length", f"{sim.step_length:g}", "--seed", str(int(seed)), "--collision.action", "warn",
        "--collision.mingap-factor", "0", "--time-to-teleport", f"{sim.time_to_teleport:g}", "--no-step-log", "true",
        "--emergencydecel.warning-threshold", "1000", "--aggregate-warnings", "20", *sim.sumo_options,
    ]  # fmt: skip


def sumo_version() -> str:
    return str(libsumo.simulation.getVersion()[1]) if libsumo.simulation.isLoaded() else str(libsumo.__version__)


def run_simulation(
    scenario: Scenario, law: Any, seed: int, sim: SimConfig, observer: Observer | None = None
) -> dict[str, Any]:
    """One run: ``{"trajectories", "vehicles", "collisions", "stats", "timing", "sumo_version"}`` (arrays in the
    formats of section 1 of the contract, ``collisions``: the beginning of every collision episode; ``stats``
    holds the counts of ``run.json``).

    ``observer(t, ids, states, accelerations, histories, speeds)`` is called every step with the controlled
    vehicles (SUMO ids, ``(s, dv, v)``, clipped accelerations, the histories handed to the law, the commanded
    speeds after the contact cap) while SUMO is at that state (tests query libsumo from it).
    """
    geometry = scenario.meta["geometry"]
    edges = geometry["edges"]
    edge_code = {e["id"]: i for i, e in enumerate(edges)}
    edge_x0 = np.array([e["x0"] for e in edges], dtype=np.float64)
    edge_lanes = np.array([e["lanes"] for e in edges], dtype=np.int64)
    controlled_edge = np.array([e["id"] in geometry["controlled_edges"] for e in edges])
    ramp = geometry.get("offramp_edge")  # one-lane edge from the end of the auxiliary lane (None: I-80)
    ramp_code, ramp_lane = (edge_code[ramp], int(geometry["offramp_lane"])) if ramp else (-1, 0)
    early_end = float(geometry.get("offramp_early_end") or 0.0) if ramp else 0.0
    main_line = np.array([e["id"] != ramp for e in edges])  # positions recorded, x_out passed: not on the off-ramp
    aux_code, aux_index = edge_code[geometry["aux_edge"]], int(geometry["aux_index"])
    aux_end = float(geometry["aux_end"])
    x_in, x_out = float(geometry["x_in"]), float(geometry["x_out"])
    boundary = scenario.boundary_speed
    n_seconds = boundary.shape[1]
    feedback = scenario.boundary
    window_start, window_end = (float(w) for w in scenario.meta.get("analysis_window", (0.0, 0.0)))
    tracked = {"g": [], "dN": []}  # values of the steps inside the analysis window

    n = scenario.n_vehicles
    dt = sim.step_length
    per_second = int(round(1.0 / dt))
    if abs(per_second * dt - 1.0) > 1e-9:
        raise ValueError(f"step_length {dt} must divide one second")
    planned = np.asarray(scenario.demand["depart_planned"], dtype=np.float64)
    last_step = int(round((planned.max() + sim.tail_s) * per_second))
    t_end = last_step / per_second

    member = law.assign(n, np.random.default_rng(seed))  # before the simulation: independent of it
    window = int(law.window)
    history = np.zeros((n, window, 3), dtype=np.float64)
    depart, exit_t = np.full(n, np.nan), np.full(n, np.nan)
    exit_lane = np.full(n, -1, dtype=np.int64)
    teleported, exited, alive, fresh, on_buffer, in_contact = (np.zeros(n, dtype=bool) for _ in range(6))
    bound, left_by_ramp = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)  # route ends on the off-ramp; left by it
    n_contacts = np.zeros(n, dtype=np.int32)  # collision episodes of every vehicle as follower
    x_prev, lane_prev = np.full(n, np.nan), np.full(n, -1, dtype=np.int64)
    records: dict[str, list[np.ndarray]] = {key: [] for key in TRAJECTORY_DTYPES}
    episodes: list[tuple[float, int, float, int]] = []  # beginning of every collision episode: t, row, x, lane
    collision_pairs: set[tuple[str, str]] = set()
    vanished = 0
    vehicle_steps = 0
    n_passed = 0  # simulated vehicles that have passed x_out
    data_exit = np.asarray(scenario.demand["exit_t"], dtype=np.float64)
    gone: list[float] = []  # data passages of x_out of the vehicles removed from the simulation
    timing = {"sumo": 0.0, "read": 0.0, "law": 0.0, "write": 0.0}

    def gain(t: float) -> float:
        """``g(t)`` of the buffer; records ``g`` and ``dN`` inside the analysis window."""
        dn = exit_difference(t, n_passed, scenario.boundary_t, scenario.exits, gone)
        g = feedback.gain_of(dn)
        if window_start <= t < window_end:
            tracked["g"].append(g)
            tracked["dN"].append(dn)
        return g

    libsumo.start(sumo_command(scenario, seed, sim))
    version = sumo_version()
    t_loop = time.perf_counter()
    try:
        for k in range(last_step + 1):
            t = k / per_second
            c0 = time.perf_counter()
            libsumo.simulationStep()
            c1 = time.perf_counter()
            timing["sumo"] += c1 - c0

            for vid in libsumo.simulation.getDepartedIDList():
                row = int(vid)
                depart[row], alive[row], fresh[row] = t, True, True
                libsumo.vehicle.setSpeedMode(vid, 0)
                libsumo.vehicle.subscribe(vid, SUBSCRIBED)
                libsumo.vehicle.subscribeLeader(vid, sim.leader_range + LOOKAHEAD_MARGIN)
                if ramp:
                    bound[row] = libsumo.vehicle.getRoute(vid)[-1] == ramp
            removed: set[str] = set()
            for vid in libsumo.simulation.getStartingTeleportIDList():
                row = int(vid)
                teleported[row] = not exited[row]  # beyond x_out it has left the section already
                if alive[row]:
                    _remove(vid)
                    alive[row] = False
                    removed.add(vid)
                    if teleported[row] and np.isfinite(data_exit[row]):
                        gone.append(data_exit[row])
            for vid in libsumo.simulation.getArrivedIDList():
                row = int(vid)
                if alive[row] and not exited[row]:
                    if bound[row]:  # the end of its route: the end of the off-ramp
                        left_by_ramp[row] = True
                    else:
                        vanished += 1  # left the network without passing x_out, a collision or a teleport
                alive[row] = False
            for collision in libsumo.simulation.getCollisions():
                collision_pairs.add((collision.collider, collision.victim))

            results = libsumo.vehicle.getAllSubscriptionResults()
            ids, rows, road, lane_index, pos, speed, leader, gap = [], [], [], [], [], [], [], []
            speed_of: dict[str, float] = {}
            for vid, r in results.items():
                v = r[CONSTANTS.VAR_SPEED]
                speed_of[vid] = v
                if vid in removed:
                    continue
                lead = r[CONSTANTS.VAR_LEADER]
                ids.append(vid)
                rows.append(int(vid))
                road.append(edge_code.get(r[CONSTANTS.VAR_ROAD_ID], -1))
                lane_index.append(r[CONSTANTS.VAR_LANE_INDEX])
                pos.append(r[CONSTANTS.VAR_LANEPOSITION])
                speed.append(v)
                leader.append(lead[0] if lead else "")
                gap.append(lead[1] if lead else -1.0)
            rows_a = np.array(rows, dtype=np.int64)
            timing["read"] += time.perf_counter() - c1
            if not len(rows_a):
                gain(t)
                continue
            c1 = time.perf_counter()
            code = np.array(road, dtype=np.int64)
            if (code < 0).any():
                bad = [ids[i] for i in np.flatnonzero(code < 0)]
                raise RuntimeError(f"t={t:.1f}: vehicles {bad} are not on an edge of the corridor")
            v_a = np.array(speed, dtype=np.float64)
            lane_idx = np.array(lane_index, dtype=np.int64)
            x = edge_x0[code] + np.array(pos, dtype=np.float64)
            lane = edge_lanes[code] - lane_idx
            if ramp:
                lane = np.where(code == ramp_code, ramp_lane, lane)
            vehicle_steps += len(rows_a)

            # passage of x_out, interpolated between the previous and this step
            passing = ~exited[rows_a] & (x >= x_out) & main_line[code]
            if passing.any():
                r = rows_a[passing]
                xp, xc = x_prev[r], x[passing]
                ok = np.isfinite(xp) & (xp < x_out) & (xc > xp)
                frac = np.where(ok, (x_out - np.where(ok, xp, 0.0)) / np.where(ok, xc - xp, 1.0), 1.0)
                exit_t[r] = t - dt + dt * frac
                exit_lane[r] = np.where(lane_prev[r] > 0, lane_prev[r], lane[passing])
                exited[r] = True
                n_passed += len(r)
            x_prev[rows_a], lane_prev[rows_a] = x, lane
            if k % per_second == 0:
                keep = (x >= x_in) & (x <= x_out) & main_line[code]
                for key, values in (("t", np.full(int(keep.sum()), t)), ("vehicle", rows_a[keep]), ("x", x[keep]),
                                    ("lane", lane[keep]), ("v", v_a[keep])):  # fmt: skip
                    records[key].append(values)

            # state of the controlled vehicles
            ctrl = controlled_edge[code]
            gap_a = np.array(gap, dtype=np.float64)
            has_leader = np.array([bool(name) for name in leader], dtype=bool) & (gap_a <= sim.leader_range)
            v_lead = np.array([speed_of.get(name, np.nan) if name else np.nan for name in leader], dtype=np.float64)
            s = np.where(has_leader, gap_a, sim.leader_range)
            dv = np.where(has_leader, v_a - np.where(has_leader, v_lead, 0.0), 0.0)
            # the end of the auxiliary edge stops a vehicle whose lane does not lead on along its route: the
            # auxiliary lane unless the route ends on the off-ramp, the other lanes when it does; for the latter
            # `early_end` m earlier while it is upstream of that point (else two vehicles that need each other's lane
            # stop side by side at the same end and block each other until a teleport)
            on_aux = (code == aux_code) & (lane_idx == aux_index)
            blocked = (code == aux_code) & (on_aux != bound[rows_a])
            s_wall = aux_end - x
            if early_end > 0.0:
                s_wall = np.where(blocked & ~on_aux & (s_wall > early_end), s_wall - early_end, s_wall)
            closer = blocked & (s_wall < s)
            s, dv = np.where(closer, s_wall, s), np.where(closer, v_a, dv)
            # contact episodes of the followers of the law, and of a vehicle of the law that runs into the queue
            # on the buffer in the very step it passes x_out; an episode ends when the gap is above contact_gap
            entering = ~controlled_edge[code] & ~on_buffer[rows_a]
            touch = (ctrl | entering) & (s <= 0.0) & ~in_contact[rows_a]
            for i in np.flatnonzero(touch):
                n_contacts[rows_a[i]] += 1
                episodes.append((t, int(rows_a[i]), float(x[i]), int(lane[i])))
            in_contact[rows_a[touch]] = True
            in_contact[rows_a[in_contact[rows_a] & (s > sim.contact_gap)]] = False

            idx = np.flatnonzero(ctrl)
            if len(idx):
                rc = rows_a[idx]
                state = np.stack((s[idx], dv[idx], v_a[idx]), axis=1)
                slot = k % window
                history[rc, slot] = state
                new = fresh[rc]
                if new.any():
                    history[rc[new]] = state[new][:, None, :]
                order = (np.arange(window) + slot + 1) % window
                inputs = history[rc[:, None], order[None, :]]
                c2 = time.perf_counter()
                timing["read"] += c2 - c1
                acc = np.asarray(law.accelerations(rc, inputs), dtype=np.float64)
                c3 = time.perf_counter()
                timing["law"] += c3 - c2
                if not np.all(np.isfinite(acc)):
                    raise RuntimeError(f"t={t:.1f}: the law returned non-finite accelerations")
                acc = np.clip(acc, sim.a_min, sim.a_max)
                v_next = np.maximum(state[:, 2] + acc * dt, 0.0)
                capped = in_contact[rc]  # in contact means a gap of at most contact_gap
                if capped.any():  # no further penetration: the gap after the step is >= 0 if the leader keeps its speed
                    v_ahead = state[capped, 2] - state[capped, 1]  # v - dv
                    v_next[capped] = np.maximum(np.minimum(v_next[capped], v_ahead + state[capped, 0] / dt), 0.0)
                if observer is not None:
                    observer(t, [ids[i] for i in idx], state, acc, inputs, v_next)
                for i, value in zip(idx, v_next.tolist()):
                    libsumo.vehicle.setSpeed(ids[i], value)
                fresh[rc] = False
            else:
                c3 = time.perf_counter()
                timing["read"] += c3 - c1
            c5 = c3
            g = gain(t)
            buffer = ~controlled_edge[code]
            if buffer.any():
                second = min(max(int(round(t)), 0), n_seconds - 1)
                target = g * boundary[np.clip(lane[buffer] - 1, 0, boundary.shape[0] - 1), second]
                if feedback.max_decel is not None:
                    v_now = v_a[buffer]
                    target = np.clip(target, v_now - feedback.max_decel * dt, v_now + feedback.max_accel * dt)
                index = np.flatnonzero(buffer)
                target = np.maximum(target, 0.0)
                if feedback.min_gap is not None:
                    # the buffer vehicles do not follow each other: without this bound a vehicle that enters
                    # faster than the queue on the buffer and brakes at max_decel only runs into it. From the
                    # front to the back, so that every vehicle sees the new speed of the one ahead (Euler step:
                    # new gap = gap + (v_ahead - v) dt >= min_gap)
                    new_speed: dict[str, float] = {}
                    for j in np.argsort(-x[index]):
                        i, name = index[j], leader[index[j]]
                        if name:
                            ahead = new_speed.get(name, speed_of.get(name, 0.0) - (feedback.max_decel or 0.0) * dt)
                            target[j] = max(min(target[j], ahead + (gap_a[i] - feedback.min_gap) / dt), 0.0)
                        new_speed[ids[i]] = float(target[j])
                for i, value in zip(index, target.tolist()):
                    libsumo.vehicle.setSpeed(ids[i], value)
                if not feedback.lane_changes:
                    for i in index[~on_buffer[rows_a[index]]]:
                        libsumo.vehicle.setLaneChangeMode(ids[i], 0)  # a cut-in on the buffer defeats min_gap
                on_buffer[rows_a[index]] = True
            timing["write"] += time.perf_counter() - c5
    finally:
        libsumo.close()
    loop_wall = time.perf_counter() - t_loop

    inserted = np.isfinite(depart)
    in_network = alive & ~exited
    known = ("vehicle_id", "depart_planned", "entry_x", "entry_lane", "length", "v_class")  # from the demand
    vehicles = {
        **{key: scenario.demand[key] for key in known}, "depart": depart, "exit_t": exit_t, "exit_lane": exit_lane,
        "member": member, "n_collisions": n_contacts, "teleported": teleported,
    }  # fmt: skip
    trajectories = {key: np.concatenate(values) if values else np.zeros(0) for key, values in records.items()}
    collisions = {key: np.array([e[j] for e in episodes]) for j, key in enumerate(COLLISION_DTYPES)}
    stats = {
        "n_planned": int(n),
        "n_inserted": int(inserted.sum()),
        "n_exited": int(exited.sum()),
        "n_offramp": int(left_by_ramp.sum()),
        "n_in_network": int(in_network.sum()),
        "n_collisions": int(n_contacts.sum()),
        "n_sumo_collisions": len(collision_pairs),
        "n_teleports": int(teleported.sum()),
        "n_not_inserted": int(n - inserted.sum()),
        "mean_depart_delay_s": float(np.mean(depart[inserted] - planned[inserted])) if inserted.any() else None,
        "sim_time_s": float(t_end),
        "steps": last_step + 1,
        "vehicle_steps": int(vehicle_steps),
        "vehicle_steps_per_second": float(vehicle_steps / loop_wall) if loop_wall > 0 else None,
    }
    for name, values in tracked.items():  # time mean, minimum and maximum inside the analysis window
        stats.update({f"{name}_{what}": float(fn(values)) if values else None
                      for what, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))})  # fmt: skip
    timing.update(loop=loop_wall, vanished=vanished)
    return {
        "trajectories": cast(trajectories, TRAJECTORY_DTYPES),
        "vehicles": cast(vehicles, VEHICLE_DTYPES),
        "collisions": cast(collisions, COLLISION_DTYPES),
        "stats": stats,
        "timing": timing,
        "sumo_version": version,
    }


def _remove(vid: str) -> None:
    """Remove a teleported vehicle from the network. Its subscriptions go first: SUMO would evaluate them in
    the next step and fail on the unknown vehicle."""
    try:
        libsumo.vehicle.unsubscribe(vid)
        libsumo.vehicle.remove(vid, CONSTANTS.REMOVE_VAPORIZED)
    except libsumo.TraCIException:  # already gone (e.g. a teleport that ended beyond the last edge)
        pass
