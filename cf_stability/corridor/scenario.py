"""Scenario of the corridor (docs/m5_contract.md, section 2): network, demand, boundary speeds, ground truth.

Build process only (pandas and pyarrow; ``netconvert`` runs as a child process). Files of a scenario
directory::

    net.nod.xml  net.edg.xml  net.con.xml   plain XML sources of the network
    net.net.xml                             netconvert output
    routes.rou.xml                          vehicle types, routes, one vehicle per track (id = row)
    boundary.npz                            t [T] (s), speed [main lanes, T] (m/s): rows = NGSIM lanes 1..6
                                            (I-80), 1..5 (US-101); exits [T]: data vehicles that have passed
                                            x_out by t
    ground_truth.npz                        trajectories of the data, as trajectories.npz of a run
    vehicles_truth.npz                      vehicles of the demand, as vehicles.npz of a run
    scenario.json                           geometry, period, counts, config, config_hash (and variant,
                                            overrides, metrics where given)

The rows of ``vehicles_truth.npz`` are the vehicles of the demand in the order of the planned
departures; row ``i`` is the SUMO vehicle ``"i"`` and row ``i`` of the ``vehicles.npz`` of every run.
Time is counted from the first frame of the period in the data (``frame_origin``).

One code for every site (``configs/corridor/<site>.yaml``): I-80 (D94) has an auxiliary lane without
successor; US-101 (D112) has data lanes that the network models as its auxiliary lane (``lane_aliases``)
and an off-ramp edge that leaves the end of that lane (``offramp``): the vehicles whose track ends on the
off-ramp get a route that ends on it, and their samples beyond the end of the auxiliary lane are not part
of the section (neither in the ground truth nor in the runs). A variant of a scenario (D113) is the same
period built with other settings: ``<site>_p<period>_<variant>``.

pandas is imported inside the functions that need it: the control loop takes the file formats from
here and must not load pandas before libsumo (docs/m5_contract.md, section 0).
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np

from cf_stability.corridor.sumo_env import run_netconvert
from cf_stability.utils import config_hash, git_revision, to_plain, write_json

if TYPE_CHECKING:
    import pandas as pd

FRAMES_PER_SECOND = 10  # NGSIM: 10 Hz
TRACK_COLUMNS = ("track_id", "period", "vehicle_id", "frame_id", "lane_id", "x", "v", "length", "v_class")
TRAJECTORY_DTYPES = {"t": np.float32, "vehicle": np.int32, "x": np.float32, "lane": np.int8, "v": np.float32}
VEHICLE_DTYPES = {
    "vehicle_id": np.int32, "depart_planned": np.float64, "depart": np.float64, "entry_x": np.float64,
    "entry_lane": np.int8, "exit_t": np.float64, "exit_lane": np.int8, "length": np.float64, "v_class": np.int8,
    "member": np.int8, "n_collisions": np.int32, "teleported": np.bool_,
}  # fmt: skip
COLLISION_DTYPES = {"t": np.float32, "vehicle": np.int32, "x": np.float32, "lane": np.int8}  # collisions.npz
NO_LANE = -1  # exit_lane of a vehicle that did not pass x_out; member of the ground truth
EXTRAPOLATION_MIN_SPEED = 0.1  # m/s: slower last samples give no extrapolated passage of x_out (at most 10 s per m)
HOV_CLASS = "hov"  # SUMO vehicle class of the vehicles that use the HOV lane in the data
RAMP_OFFSET = 10.0  # m, the end node of an off-ramp lies this far to the right of the main line (drawing only)
UNHASHED_KEYS = ("periods", "enabled", "metrics")  # keys of a site config that do not change the simulation


@dataclasses.dataclass(frozen=True)
class Edge:
    """Edge ``[x0, x1]`` of the corridor with the NGSIM lanes ``1..lanes`` (SUMO index = ``lanes - lane``)."""

    id: str
    x0: float
    x1: float
    lanes: int

    @property
    def length(self) -> float:
        return self.x1 - self.x0

    def index(self, lane: int | np.ndarray) -> Any:
        return self.lanes - lane

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Corridor:
    """Geometry of a corridor (``configs/corridor/<site>.yaml``).

    ``edges`` is the main line, contiguous from ``x_in`` to the end of the buffer; ``offramp`` (optional) is a
    one-lane edge that leaves the end of the auxiliary lane (``x0`` = that end, ``x1`` = ``x0`` + its length,
    lane positions as the main line's) and carries the NGSIM number ``offramp_lane``. ``lane_aliases`` maps
    lanes of the data to the lane of the network that models them (US-101: the on-ramp lane 7 and the off-ramp
    lane 8 are the auxiliary lane 6 where the main line has it)."""

    x_in: float
    x_out: float
    edges: tuple[Edge, ...]
    aux_lane: int
    lane_width: float
    speed: float
    hov_lane: int | None = None  # NGSIM lane reserved to the vehicle class hov on the edges of the section
    lane_aliases: tuple[tuple[int, int], ...] = ()  # (lane of the data, lane of the network) where they differ
    offramp: Edge | None = None  # one-lane edge from the end of the auxiliary lane; not part of the main line
    offramp_lane: int | None = None  # NGSIM number of the off-ramp: the last lane of the tracks that leave by it
    offramp_early_end: float = 0.0  # m: a vehicle bound for the off-ramp that is not on the auxiliary lane sees the end
    # of its lane this far before the end of the auxiliary lane (two vehicles that need each other's lane cannot stop
    # side by side at the same end)
    aux_class: str | None = None  # the auxiliary lane admits only this SUMO class: the vehicles that use it in the data

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Corridor:
        edges = tuple(Edge(str(e["id"]), float(e["x0"]), float(e["x1"]), int(e["lanes"])) for e in raw["edges"])
        hov = raw.get("hov_lane")
        aliases = tuple(sorted((int(k), int(v)) for k, v in (raw.get("lane_aliases") or {}).items()))
        corridor = cls(
            x_in=float(raw["x_in"]), x_out=float(raw["x_out"]), edges=edges, aux_lane=int(raw["aux_lane"]),
            lane_width=float(raw["lane_width"]), speed=float(raw["speed"]), hov_lane=None if hov is None else int(hov),
            lane_aliases=aliases, aux_class=None if raw.get("aux_class") is None else str(raw["aux_class"]),
        )  # fmt: skip
        corridor.check()
        ramp = raw.get("offramp")
        if ramp:
            end = corridor.aux_edge.x1
            corridor = dataclasses.replace(
                corridor, offramp=Edge(str(ramp["id"]), end, end + float(ramp["length"]), 1),
                offramp_lane=int(ramp["lane"]), offramp_early_end=float(ramp.get("early_end") or 0.0),
            )  # fmt: skip
            corridor.check()
        return corridor

    def permissions(self) -> dict[tuple[str, int], str]:
        """``{(edge, SUMO lane index): allowed classes}`` of the restricted lanes: the HOV lane on the edges of the
        section (not on the buffer, where the speeds of the data are prescribed) and the auxiliary lane with
        ``aux_class``."""
        out = {} if self.hov_lane is None else {(e.id, e.index(self.hov_lane)): HOV_CLASS for e in self.section}
        if self.aux_class is not None:
            out[(self.aux_edge.id, self.aux_edge.index(self.aux_lane))] = self.aux_class
        return out

    def check(self) -> None:
        """Contiguous edges from ``x_in``, one of them starting at ``x_out``, one edge with the auxiliary lane; an
        off-ramp leaves that edge inside the section and has a number of its own."""
        if self.edges[0].x0 != self.x_in or any(a.x1 != b.x0 for a, b in zip(self.edges[:-1], self.edges[1:])):
            raise ValueError("the edges must be contiguous and start at x_in")
        if not any(e.x0 == self.x_out for e in self.edges):
            raise ValueError("one edge (the buffer) must start at x_out")
        if len([e for e in self.edges if e.lanes >= self.aux_lane]) != 1:
            raise ValueError(f"exactly one edge must have the auxiliary lane {self.aux_lane}")
        if self.offramp is not None:
            if self.offramp.id in {e.id for e in self.edges} or self.aux_edge.x1 >= self.x_out:
                raise ValueError("the off-ramp needs an id of its own and must leave the auxiliary lane in the section")
            if self.offramp_lane is None or self.offramp_lane <= self.aux_edge.lanes:
                raise ValueError(f"the off-ramp needs an NGSIM number beyond the lanes of {self.aux_edge.id}")

    @property
    def main_lanes(self) -> int:
        """Lanes present on every edge of the main line (NGSIM 1..main_lanes)."""
        return min(e.lanes for e in self.edges)

    @property
    def aux_edge(self) -> Edge:
        return next(e for e in self.edges if e.lanes >= self.aux_lane)

    @property
    def section(self) -> tuple[Edge, ...]:
        """Edges of the main line inside the section: the positions recorded, the passage of ``x_out``."""
        return tuple(e for e in self.edges if e.x1 <= self.x_out)

    @property
    def controlled(self) -> tuple[Edge, ...]:
        """Edges where the law drives: the section and the off-ramp."""
        return self.section + (() if self.offramp is None else (self.offramp,))

    @property
    def buffer(self) -> tuple[Edge, ...]:
        """Edges downstream of ``x_out``, where the speeds of the data are prescribed."""
        return tuple(e for e in self.edges if e.x0 >= self.x_out)

    @property
    def all_edges(self) -> tuple[Edge, ...]:
        return self.edges + (() if self.offramp is None else (self.offramp,))

    def network_lane(self, lane: Any) -> Any:
        """Lane of the network that models the data lane(s) ``lane`` (``lane_aliases``; the others unchanged)."""
        if np.ndim(lane) == 0:
            return dict(self.lane_aliases).get(int(lane), int(lane))
        lane = np.asarray(lane)
        out = lane.copy()
        for data, network in self.lane_aliases:
            out[lane == data] = network
        return out

    def on_network(self, x: np.ndarray, lane: np.ndarray) -> np.ndarray:
        """Samples that lie on a lane of the main line inside the section (the on-ramp upstream of the
        auxiliary lane is not modelled; data lanes are taken as the lanes of the network that model them)."""
        x, lane = np.asarray(x, dtype=np.float64), self.network_lane(np.asarray(lane))
        inside = (x >= self.x_in) & (x <= self.x_out) & (lane >= 1)
        on_lane = np.zeros(len(x), dtype=bool)
        for edge in self.edges:
            on_lane |= (x >= edge.x0) & (x <= edge.x1) & (lane <= edge.lanes)
        return inside & on_lane

    def locate(self, x: float, lane: int) -> tuple[Edge, float]:
        """Edge and lane position of the front bumper at ``x`` in the data lane ``lane`` (half-open edges; a
        position at the very end of the last edge with that lane stays on it)."""
        network = self.network_lane(lane)
        candidates = [e for e in self.edges if e.x0 <= x <= e.x1 and network <= e.lanes]
        if not candidates:
            raise ValueError(f"x = {x} m in lane {lane} is not on the network")
        edge = next((e for e in candidates if x < e.x1), candidates[-1])
        return edge, min(x - edge.x0, edge.length - 1e-3)

    def sim_geometry(self) -> dict[str, Any]:
        """What the control loop needs (``scenario.json["geometry"]``); the off-ramp keys only where there is one."""
        aux = self.aux_edge
        geometry = {
            "x_in": self.x_in, "x_out": self.x_out, "edges": [e.as_dict() for e in self.all_edges],
            "controlled_edges": [e.id for e in self.controlled], "buffer_edges": [e.id for e in self.buffer],
            "aux_lane": self.aux_lane, "aux_edge": aux.id, "aux_index": aux.index(self.aux_lane), "aux_end": aux.x1,
            "main_lanes": self.main_lanes, "lane_width": self.lane_width, "speed": self.speed,
        }  # fmt: skip
        if self.offramp is not None:
            geometry.update(
                offramp_edge=self.offramp.id, offramp_lane=self.offramp_lane, offramp_early_end=self.offramp_early_end,
                section_edges=[e.id for e in self.section],
            )
        return geometry


# ------------------------------------------------------------------------------------------------ data


def read_tracks(path: str | Path, periods: Sequence[int] | None = None) -> pd.DataFrame:
    """Reconstructed trajectories (``TRACK_COLUMNS``), sorted by track and frame; ``track_id`` as str."""
    import pandas as pd

    filters = [("period", "in", list(periods))] if periods is not None else None
    df = pd.read_parquet(path, columns=list(TRACK_COLUMNS), filters=filters)
    df["track_id"] = df["track_id"].astype(str)
    return df.sort_values(["track_id", "frame_id"], kind="stable").reset_index(drop=True)


def frame_seconds(frame: np.ndarray | pd.Series, origin: int) -> np.ndarray:
    """Seconds from the start of the period."""
    return (np.asarray(frame, dtype=np.int64) - origin) / FRAMES_PER_SECOND


def build_demand(tracks: pd.DataFrame, corridor: Corridor, origin: int) -> pd.DataFrame:
    """One vehicle per track: its first sample on the network gives the time, lane, position and speed of the
    insertion (tracks that begin inside the section are inserted where they begin; tracks that begin
    upstream of ``x_in`` or on the on-ramp, where they reach the network). ``hov``: the track uses the HOV lane
    at some sample. ``offramp``: the track ends on the off-ramp lane and enters upstream of its end (its route
    ends on the off-ramp edge; ``offramp_unreachable``: it ends there but enters downstream of the auxiliary
    edge, and keeps the route of the main line). ``aux_user`` (with ``aux_class``): the track uses the
    auxiliary lane (a data lane that the network models as it, on the auxiliary edge), or leaves by the
    off-ramp. Sorted by planned departure, downstream first at equal times; the row is the SUMO vehicle id."""
    import pandas as pd

    tracks = tracks.sort_values(["track_id", "frame_id"], kind="stable")
    on = corridor.on_network(tracks["x"].to_numpy(), tracks["lane_id"].to_numpy())
    entries = tracks[on].groupby("track_id", sort=False).head(1)
    located = [corridor.locate(x, lane) for x, lane in zip(entries["x"], entries["lane_id"])]
    hov_lane = tracks["lane_id"] == corridor.hov_lane if corridor.hov_lane is not None else None
    hov_tracks = set() if hov_lane is None else set(tracks.loc[hov_lane, "track_id"])
    leaves = np.zeros(len(entries), dtype=bool)
    reachable = np.ones(len(entries), dtype=bool)
    if corridor.offramp is not None:
        last_lane = tracks.groupby("track_id", sort=False)["lane_id"].last()
        leaves = (entries["track_id"].map(last_lane) == corridor.offramp_lane).to_numpy(bool)
        order = {e.id: k for k, e in enumerate(corridor.edges)}
        reachable = np.array([order[edge.id] <= order[corridor.aux_edge.id] for edge, _ in located], dtype=bool)
    aux_tracks: set = set()
    if corridor.aux_class is not None:
        aux, x = corridor.aux_edge, tracks["x"].to_numpy(np.float64)
        network = corridor.network_lane(tracks["lane_id"].to_numpy())
        aux_tracks = set(tracks.loc[(network == corridor.aux_lane) & (x >= aux.x0) & (x <= aux.x1), "track_id"])
    demand = pd.DataFrame(
        {
            "track_id": entries["track_id"].to_numpy(),
            "vehicle_id": entries["vehicle_id"].to_numpy(np.int64),
            "entry_frame": entries["frame_id"].to_numpy(np.int64),
            "depart_planned": frame_seconds(entries["frame_id"], origin),
            "entry_x": entries["x"].to_numpy(np.float64),
            "entry_lane": entries["lane_id"].to_numpy(np.int64),
            "entry_v": entries["v"].to_numpy(np.float64),
            "length": entries["length"].to_numpy(np.float64),
            "v_class": entries["v_class"].to_numpy(np.int64),
            "hov": [track in hov_tracks for track in entries["track_id"]],
            "offramp": leaves & reachable,
            "offramp_unreachable": leaves & ~reachable,
            "aux_user": np.array([track in aux_tracks for track in entries["track_id"]], dtype=bool)
            | (leaves & reachable & (corridor.aux_class is not None)),
            "edge": [edge.id for edge, _ in located],
            "pos": [pos for _, pos in located],
            "index": [edge.index(corridor.network_lane(lane)) for (edge, _), lane in zip(located, entries["lane_id"])],
        }
    )
    demand = demand.sort_values(["depart_planned", "entry_x", "vehicle_id"], ascending=[True, False, True])
    return demand.reset_index(drop=True)


def truth_arrays(
    tracks: pd.DataFrame, demand: pd.DataFrame, corridor: Corridor, origin: int, extrapolate_m: float = 0.0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """``ground_truth.npz`` (whole seconds, samples from the entry on with ``x_in <= x <= x_out``; the samples of
    a vehicle that leaves by the off-ramp end at the end of the auxiliary lane, where it leaves the main line),
    ``vehicles_truth.npz`` and the number of extrapolated exits. ``exit_t`` is interpolated between the
    10 Hz samples around ``x_out``; a track that ends less than ``extrapolate_m`` before ``x_out`` (the
    tracks end at 499.6-499.99 m) passes it at the speed of its last sample, if that is at least
    ``EXTRAPOLATION_MIN_SPEED``. A vehicle that leaves by the off-ramp never passes ``x_out``."""
    import pandas as pd

    tracks = tracks.sort_values(["track_id", "frame_id"], kind="stable")  # consecutive samples of a track
    row = pd.Series(np.arange(len(demand)), index=demand["track_id"].to_numpy())
    entry_frame = pd.Series(demand["entry_frame"].to_numpy(), index=demand["track_id"].to_numpy())
    known = tracks["track_id"].isin(row.index).to_numpy()
    df = tracks[known]
    vehicle = row.loc[df["track_id"]].to_numpy()
    after_entry = df["frame_id"].to_numpy() >= entry_frame.loc[df["track_id"]].to_numpy()
    offset = df["frame_id"].to_numpy(np.int64) - origin
    x, lane, v = df["x"].to_numpy(np.float64), df["lane_id"].to_numpy(), df["v"].to_numpy(np.float64)
    keep = after_entry & (offset % FRAMES_PER_SECOND == 0) & (x >= corridor.x_in) & (x <= corridor.x_out)
    leaves = np.zeros(len(x), dtype=bool)  # samples of the vehicles that leave by the off-ramp
    if corridor.offramp is not None and "offramp" in demand:
        leaves = demand["offramp"].to_numpy(bool)[vehicle]
        keep &= ~(leaves & (x > corridor.aux_edge.x1))  # on the off-ramp: not in the section
    order = np.lexsort((vehicle[keep], offset[keep]))  # by time, then vehicle
    trajectories = {
        "t": offset[keep][order] / FRAMES_PER_SECOND, "vehicle": vehicle[keep][order], "x": x[keep][order],
        "lane": lane[keep][order], "v": v[keep][order],
    }  # fmt: skip

    # first passage of x_out after the entry (consecutive samples of one track, frames 0.1 s apart)
    same = (vehicle[1:] == vehicle[:-1]) & after_entry[:-1] & ~leaves[:-1]
    cross = same & (x[:-1] < corridor.x_out) & (x[1:] >= corridor.x_out)
    i = np.flatnonzero(cross)
    first = np.unique(vehicle[i], return_index=True)[1]
    i = i[first]
    dt = (offset[i + 1] - offset[i]) / FRAMES_PER_SECOND
    exit_t = np.full(len(demand), np.nan)
    exit_lane = np.full(len(demand), NO_LANE)
    exit_t[vehicle[i]] = offset[i] / FRAMES_PER_SECOND + dt * (corridor.x_out - x[i]) / (x[i + 1] - x[i])
    exit_lane[vehicle[i]] = lane[i]

    # tracks that end just before x_out (the camera's view) without passing it; a standing vehicle has no
    # passage time (at 1e-9 m/s it would pass after decades)
    samples = np.flatnonzero(after_entry)[::-1]
    last = samples[np.unique(vehicle[samples], return_index=True)[1]]  # last sample after the entry of every track
    gap = corridor.x_out - x[last]
    near = np.isnan(exit_t[vehicle[last]]) & (gap > 0.0) & (gap <= extrapolate_m) & (v[last] >= EXTRAPOLATION_MIN_SPEED)
    near &= ~leaves[last]
    j = last[near]
    exit_t[vehicle[j]] = offset[j] / FRAMES_PER_SECOND + gap[near] / v[j]
    exit_lane[vehicle[j]] = lane[j]
    vehicles = demand_vehicles(demand)
    vehicles.update(depart=vehicles["depart_planned"].copy(), exit_t=exit_t, exit_lane=exit_lane)
    return cast(trajectories, TRAJECTORY_DTYPES), cast(vehicles, VEHICLE_DTYPES), int(near.sum())


def demand_vehicles(demand: pd.DataFrame) -> dict[str, np.ndarray]:
    """Columns of ``vehicles.npz`` known before the simulation; the others empty (not inserted, not exited)."""
    n = len(demand)
    return {
        "vehicle_id": demand["vehicle_id"].to_numpy(), "depart_planned": demand["depart_planned"].to_numpy(),
        "depart": np.full(n, np.nan), "entry_x": demand["entry_x"].to_numpy(),
        "entry_lane": demand["entry_lane"].to_numpy(), "exit_t": np.full(n, np.nan),
        "exit_lane": np.full(n, NO_LANE), "length": demand["length"].to_numpy(),
        "v_class": demand["v_class"].to_numpy(), "member": np.full(n, NO_LANE), "n_collisions": np.zeros(n, np.int32),
        "teleported": np.zeros(n, dtype=bool),
    }  # fmt: skip


def cast(arrays: Mapping[str, np.ndarray], dtypes: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The arrays of ``dtypes`` in that order and type (the file formats of the contract, section 1)."""
    return {key: np.asarray(arrays[key]).astype(dtype) for key, dtype in dtypes.items()}


def boundary_speeds(
    tracks: pd.DataFrame, origin: int, n_seconds: int, lanes: Sequence[int], x_range: Sequence[float],
    half_window_s: float, fallback: float,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``t [T]``, ``speed [len(lanes), T]``: per lane and whole second the mean speed of the samples with
    ``x_range[0] <= x <= x_range[1]`` in that lane within ``half_window_s`` of the second. A second without
    samples takes the last value before it (the first value at the start; ``fallback`` for a lane without
    any sample)."""
    lanes = list(lanes)
    x = tracks["x"].to_numpy(np.float64)
    zone = (x >= x_range[0]) & (x <= x_range[1]) & tracks["lane_id"].isin(lanes).to_numpy()
    lane_row = np.searchsorted(lanes, tracks["lane_id"].to_numpy()[zone])
    offset = tracks["frame_id"].to_numpy(np.int64)[zone] - origin
    v = tracks["v"].to_numpy(np.float64)[zone]
    half = int(round(half_window_s * FRAMES_PER_SECOND))
    k_low = -((half - offset) // FRAMES_PER_SECOND)  # ceil((offset - half) / 10) with integers
    k_high = (offset + half) // FRAMES_PER_SECOND
    sums, counts = np.zeros((len(lanes), n_seconds)), np.zeros((len(lanes), n_seconds))
    for step in range(2 * half // FRAMES_PER_SECOND + 1):
        k = k_low + step
        ok = (k <= k_high) & (k >= 0) & (k < n_seconds)
        np.add.at(sums, (lane_row[ok], k[ok]), v[ok])
        np.add.at(counts, (lane_row[ok], k[ok]), 1.0)
    with np.errstate(invalid="ignore"):
        speed = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    for row in speed:
        valid = np.flatnonzero(np.isfinite(row))
        if not len(valid):
            row[:] = fallback
            continue
        last = np.maximum.accumulate(np.where(np.isfinite(row), np.arange(n_seconds), -1))
        row[:] = row[np.where(last >= 0, last, valid[0])]
    return np.arange(n_seconds, dtype=np.float64), speed


@dataclasses.dataclass(frozen=True)
class BoundaryConfig:
    """Downstream boundary of a scenario (``scenario.json["boundary"]``, from ``configs/corridor/<site>.yaml``):
    a vehicle on the buffer approaches ``v_b(lane, t) * g(t)``, ``g = clip(1 - gain * dN, g_min, g_max)`` with
    ``feedback`` (else 1), by at most ``max_decel`` / ``max_accel``, keeping ``min_gap`` to the vehicle ahead."""

    feedback: bool = False
    gain: float = 0.02  # per vehicle
    g_min: float = 0.2
    g_max: float = 1.5
    max_decel: float | None = None  # m/s^2; None: the speed of a buffer vehicle jumps to its target
    max_accel: float | None = None
    min_gap: float | None = None  # m: a buffer vehicle does not come closer than this to the one ahead
    lane_changes: bool = True  # False: no lane changes on the buffer (lane change mode 0 from its first step there)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> BoundaryConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in (raw or {}).items() if key in names})

    def gain_of(self, dn: float) -> float:
        return min(max(1.0 - self.gain * dn, self.g_min), self.g_max) if self.feedback else 1.0


def exit_difference(
    t: float, n_passed: int, boundary_t: np.ndarray, exits: np.ndarray, gone: Sequence[float] = ()
) -> float:
    """``dN(t)``: simulated passages of ``x_out`` minus those of the data (``exits`` of ``boundary.npz``, linear
    in time). ``gone``: data passage times of the vehicles removed from the simulation (collision, teleport);
    they never pass ``x_out`` there, so their passages leave the data's count too, else the feedback would read
    the removals as a queue that lets out too little and drain it."""
    data = float(np.interp(t, boundary_t, exits)) - np.count_nonzero(np.asarray(gone, dtype=np.float64) <= t)
    return float(n_passed - data)


# --------------------------------------------------------------------------------------------- SUMO files


def network_sources(corridor: Corridor) -> tuple[str, str, str]:
    """Plain XML node, edge and connection files. Consecutive edges connect the lanes with the same NGSIM
    number, so the auxiliary lane (absent downstream) has no successor on the main line; an off-ramp is its
    only successor; the HOV lane of the section allows the class hov only (:meth:`Corridor.permissions`)."""
    xs = [corridor.edges[0].x0] + [e.x1 for e in corridor.edges]
    node = {x: f"x{x:g}" for x in xs}
    nodes = ["<nodes>"] + [f'    <node id="{node[x]}" x="{x:.2f}" y="0.00" type="priority"/>' for x in xs]
    ramp = corridor.offramp
    if ramp is not None:  # its length is set on the edge; the node only draws it to the right of the main line
        nodes.append(f'    <node id="{ramp.id}_end" x="{ramp.x1:.2f}" y="{-RAMP_OFFSET:.2f}" type="priority"/>')
    nodes.append("</nodes>")
    restricted = corridor.permissions()
    ends = {e.id: (node[e.x0], node[e.x1]) for e in corridor.edges}
    if ramp is not None:
        ends[ramp.id] = (node[ramp.x0], f"{ramp.id}_end")
    edges = ["<edges>"]
    for e in corridor.all_edges:
        head = (
            f'    <edge id="{e.id}" from="{ends[e.id][0]}" to="{ends[e.id][1]}" numLanes="{e.lanes}" '
            f'speed="{corridor.speed:.2f}" width="{corridor.lane_width:.2f}" length="{e.length:.2f}"'
        )
        lanes = [f'        <lane index="{i}" allow="{rule}"/>' for (on, i), rule in restricted.items() if on == e.id]
        edges += [head + ">", *lanes, "    </edge>"] if lanes else [head + "/>"]
    edges.append("</edges>")
    connections = ["<connections>"]
    for a, b, from_lane, to_lane in _connections(corridor):
        connections.append(f'    <connection from="{a}" to="{b}" fromLane="{from_lane}" toLane="{to_lane}"/>')
    connections.append("</connections>")
    return "\n".join(nodes) + "\n", "\n".join(edges) + "\n", "\n".join(connections) + "\n"


def _connections(corridor: Corridor) -> list[tuple[str, str, int, int]]:
    """``(from, to, fromLane, toLane)`` of the network in the order of the connection file: consecutive edges of
    the main line by NGSIM number, then the auxiliary lane to the off-ramp."""
    out = [
        (a.id, b.id, a.index(lane), b.index(lane))
        for a, b in zip(corridor.edges[:-1], corridor.edges[1:]) for lane in range(1, min(a.lanes, b.lanes) + 1)
    ]  # fmt: skip
    if corridor.offramp is not None:
        aux = corridor.aux_edge
        out.append((aux.id, corridor.offramp.id, aux.index(corridor.aux_lane), 0))
    return out


def sumo_class(v_class: int, hov: bool, classes: Mapping[int, str], aux_class: str | None = None) -> str:
    """SUMO vehicle class of a vehicle of the demand: ``hov`` for the users of the HOV lane, ``aux_class`` for the
    users of a restricted auxiliary lane, else by NGSIM class (one class per vehicle: a vehicle of both kinds
    cannot be represented)."""
    if hov and aux_class is not None:
        raise ValueError(f"a vehicle uses the HOV lane and the auxiliary lane of class {aux_class}")
    return HOV_CLASS if hov else aux_class if aux_class is not None else classes[int(v_class)]


def vehicle_classes(demand: pd.DataFrame, corridor: Corridor, classes: Mapping[int, str]) -> list[str]:
    """:func:`sumo_class` of every vehicle of the demand."""
    aux = demand["aux_user"] if "aux_user" in demand else [False] * len(demand)
    return [
        sumo_class(c, h, classes, corridor.aux_class if a else None)
        for c, h, a in zip(demand["v_class"], demand["hov"], aux)
    ]


def vtype_id(sumo_vclass: str, length: float) -> str:
    return f"{sumo_vclass}_{int(round(length * 1000))}"


def route_id(start: str, offramp: str | None = None) -> str:
    """Route from the edge ``start`` to the end of the main line, or to the off-ramp ``offramp``."""
    return f"from_{start}" if offramp is None else f"from_{start}_{offramp}"


def routes_xml(
    demand: pd.DataFrame, corridor: Corridor, classes: Mapping[int, str], vtype: Mapping[str, Any],
    insertion_checks: str | None = None, class_vtype: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:  # fmt: skip
    """Vehicle types (one per SUMO class and length, attributes ``vtype``, for a class of ``class_vtype`` with
    its attributes over them), one route per edge where vehicles start and destination (the end of the main
    line, or the off-ramp for the vehicles with ``offramp``), and the vehicles in the order of the demand
    (sorted by departure, as SUMO requires), with the attribute ``insertionChecks`` when given (``collision``:
    SUMO refuses an insertion only when the vehicle would overlap another one; without it, also when its own
    car-following model finds the gap unsafe)."""
    checks = f' insertionChecks="{insertion_checks}"' if insertion_checks else ""
    lines = ["<routes>"]
    vclass = vehicle_classes(demand, corridor, classes)
    for name, length in sorted(set(zip(vclass, demand["length"]))):
        merged = {**vtype, **((class_vtype or {}).get(name) or {})}
        attributes = " ".join(f"{key}={quoteattr(_xml_value(value))}" for key, value in merged.items())
        lines.append(f'    <vType id="{vtype_id(name, length)}" vClass="{name}" length="{length:.3f}" {attributes}/>')
    ids = [e.id for e in corridor.edges]
    ramp = corridor.offramp
    leaves = [bool(o) for o in demand["offramp"]] if "offramp" in demand and ramp is not None else [False] * len(demand)
    for start, off in dict.fromkeys(zip(demand["edge"], leaves)):
        if off:
            edges = ids[ids.index(start) : ids.index(corridor.aux_edge.id) + 1] + [ramp.id]
            lines.append(f'    <route id="{route_id(start, ramp.id)}" edges="{" ".join(edges)}"/>')
        else:
            lines.append(f'    <route id="{route_id(start)}" edges="{" ".join(ids[ids.index(start):])}"/>')
    for row, (d, name, off) in enumerate(zip(demand.itertuples(index=False), vclass, leaves)):
        route = route_id(d.edge, ramp.id if off else None)
        lines.append(
            f'    <vehicle id="{row}" type="{vtype_id(name, d.length)}" route="{route}" '
            f'depart="{d.depart_planned:.1f}" departLane="{d.index}" departPos="{d.pos:.4f}" '
            f'departSpeed="{max(d.entry_v, 0.0):.4f}"{checks}/>'
        )
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def _xml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def read_network(path: str | Path) -> dict[str, Any]:
    """Lanes (``{edge: [(index, length, speed), ...]}``), connections (``(from, fromLane, to, toLane)``) and lane
    permissions (``{(edge, index): allow}`` of the lanes with an ``allow`` list) of a ``.net.xml``; internal
    (junction) edges listed apart."""
    root = ET.parse(path).getroot()
    lanes, internal, permissions = {}, [], {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            internal.append(edge.get("id"))
            continue
        lanes[edge.get("id")] = [
            (int(lane.get("index")), float(lane.get("length")), float(lane.get("speed")))
            for lane in edge.findall("lane")
        ]
        for lane in edge.findall("lane"):
            if lane.get("allow") is not None or lane.get("disallow") is not None:
                permissions[(edge.get("id"), int(lane.get("index")))] = lane.get("allow")
    connections = sorted(
        (c.get("from"), int(c.get("fromLane")), c.get("to"), int(c.get("toLane"))) for c in root.findall("connection")
        if not c.get("from", "").startswith(":")
    )  # fmt: skip
    return {"lanes": lanes, "connections": connections, "internal": internal, "permissions": permissions}


def expected_connections(corridor: Corridor) -> list[tuple[str, int, str, int]]:
    return sorted((a, from_lane, b, to_lane) for a, b, from_lane, to_lane in _connections(corridor))


def build_network(corridor: Corridor, out_dir: Path, options: Sequence[str]) -> dict[str, Any]:
    """Plain XML sources and ``net.net.xml`` (netconvert, child process); the network read back must have the
    lengths of the edges (the lane position is the corridor coordinate minus ``x0``), no internal lanes and
    exactly the connections and lane permissions of :func:`network_sources`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"net.{name}.xml" for name in ("nod", "edg", "con", "net")}
    for name, text in zip(("nod", "edg", "con"), network_sources(corridor)):
        paths[name].write_text(text, encoding="utf-8")
    proc = run_netconvert(paths["nod"], paths["edg"], paths["con"], paths["net"], options)
    net = read_network(paths["net"])
    problems = [] if not net["internal"] else [f"internal edges {net['internal']}"]
    for edge in corridor.all_edges:
        lanes = net["lanes"].get(edge.id, [])
        if sorted(i for i, _, _ in lanes) != list(range(edge.lanes)):
            problems.append(f"{edge.id}: lanes {lanes}")
        problems += [
            f"{edge.id} lane {i}: length {length}" for i, length, _ in lanes if abs(length - edge.length) > 0.01
        ]
    if net["connections"] != expected_connections(corridor):
        problems.append(f"connections {net['connections']}")
    if net["permissions"] != corridor.permissions():
        problems.append(f"lane permissions {net['permissions']}")
    if problems:
        raise RuntimeError(f"network {paths['net']} differs from the corridor: {'; '.join(problems)}")
    return {"netconvert": (proc.stdout + proc.stderr).strip(), **net}


# ------------------------------------------------------------------------------------------------ scenario


def scenario_name(site: str, period: int, variant: str | None = None) -> str:
    """``<site>_p<period>``, or ``<site>_p<period>_<variant>`` for a variant (D113)."""
    return f"{site}_p{int(period)}" + (f"_{variant}" if variant else "")


def scenario_config(corridor_cfg: Mapping[str, Any], period: int, variant: str | None = None) -> dict[str, Any]:
    """The hashed config of one scenario: the corridor section (without the keys that do not change the
    simulation: ``UNHASHED_KEYS``), the period and, for a variant, its name (the base scenario has no such key,
    so its hash does not change)."""
    cfg = {k: v for k, v in to_plain(dict(corridor_cfg)).items() if k not in UNHASHED_KEYS}
    cfg = {**cfg, "period": int(period)}
    if variant:
        cfg["variant"] = str(variant)
    return cfg


def scenario_metrics(corridor_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Settings of the corridor metrics of a site (``metrics``: detectors, lanes of the wave field), written to
    ``scenario.json`` for :mod:`cf_stability.corridor.macro`; None for a site that keeps the defaults (I-80)."""
    metrics = to_plain(dict(corridor_cfg)).get("metrics")
    return dict(metrics) if metrics else None


def build_scenario(
    tracks: pd.DataFrame, corridor_cfg: Mapping[str, Any], period: int, out_dir: str | Path,
    variant: str | None = None, overrides: Mapping[str, Any] | None = None,
) -> dict:  # fmt: skip
    """All files of the scenario of ``period`` (``tracks``: the samples of that period) in ``out_dir``;
    returns the content of ``scenario.json``, which is written last. ``variant``: name of a variant (D113)
    whose settings ``corridor_cfg`` holds already; ``overrides``: the settings that make it (recorded only)."""
    out_dir = Path(out_dir)
    cfg = scenario_config(corridor_cfg, period, variant)
    corridor = Corridor.from_mapping(cfg)
    classes = {int(k): str(v) for k, v in cfg["classes"].items()}
    tracks = tracks[tracks["period"] == period]
    if tracks.empty:
        raise ValueError(f"no samples of period {period}")
    origin = int(tracks["frame_id"].min())
    demand = build_demand(tracks, corridor, origin)
    unknown = sorted(set(demand["v_class"]) - set(classes))
    if unknown:
        raise ValueError(f"vehicle classes {unknown} have no SUMO class in `classes`")
    trajectories, vehicles, extrapolated = truth_arrays(
        tracks, demand, corridor, origin, float(cfg.get("exit_extrapolation_m") or 0.0)
    )
    t_data_end = float(frame_seconds(tracks["frame_id"].max(), origin))
    exit_times = np.sort(vehicles["exit_t"][np.isfinite(vehicles["exit_t"])])
    n_seconds = int(math.ceil(max(t_data_end, exit_times[-1] if len(exit_times) else 0.0))) + 1
    bound = cfg["boundary"]
    t, speed = boundary_speeds(
        tracks, origin, n_seconds, range(1, corridor.main_lanes + 1), bound["x_range"], bound["half_window_s"],
        corridor.speed,
    )  # fmt: skip
    exits = np.searchsorted(exit_times, t, side="right").astype(np.float64)  # data vehicles past x_out by t

    network = build_network(corridor, out_dir, cfg.get("netconvert_options") or ())
    routes = routes_xml(demand, corridor, classes, cfg["vtype"], cfg.get("insertion_checks"), cfg.get("class_vtype"))
    (out_dir / "routes.rou.xml").write_text(routes, encoding="utf-8")
    np.savez(out_dir / "boundary.npz", t=t, speed=speed, exits=exits)
    np.savez(out_dir / "ground_truth.npz", **trajectories)
    np.savez(out_dir / "vehicles_truth.npz", **vehicles)

    n_tracks = int(tracks["track_id"].nunique())
    exited = np.isfinite(vehicles["exit_t"])
    extra: dict[str, Any] = {}
    if variant:
        extra.update(variant=str(variant), overrides=to_plain(dict(overrides or {})))
    metrics = scenario_metrics(corridor_cfg)
    if metrics is not None:
        extra["metrics"] = metrics
    payload = {
        "scenario": scenario_name(cfg["name"], period, variant),
        "site": cfg["name"],
        "period": int(period),
        **extra,
        "source": str(cfg.get("source")),
        "frame_origin": origin,
        "geometry": corridor.sim_geometry(),
        "analysis_window": [float(w) for w in cfg["analysis_window"]],
        "t_first_depart": float(demand["depart_planned"].min()),
        "t_last_depart": float(demand["depart_planned"].max()),
        "t_data_end": t_data_end,
        "boundary": {**to_plain(dict(bound)), "n_seconds": n_seconds, "lanes": list(range(1, corridor.main_lanes + 1))},
        "counts": {
            "tracks": n_tracks,
            "vehicles": len(demand),
            "tracks_off_network": n_tracks - len(demand),
            "entry_lane": _counts(demand["entry_lane"]),
            "entry_edge": _counts(demand["edge"]),
            "v_class": _counts(demand["v_class"]),
            "entered_downstream_of_x_in": int((demand["entry_x"] > corridor.x_in + 5.0).sum()),
            "hov": int(demand["hov"].sum()),
            "offramp": int(demand["offramp"].sum()),  # routes that end on the off-ramp
            "offramp_unreachable": int(demand["offramp_unreachable"].sum()),
            "aux_users": int(demand["aux_user"].sum()),  # admitted to a restricted auxiliary lane (aux_class)
            "exited": int(exited.sum()),
            "exits_extrapolated": extrapolated,
            "vehicle_types": len(set(zip(vehicle_classes(demand, corridor, classes), demand["length"]))),
            "truth_rows": int(len(trajectories["t"])),
        },
        "lane_permissions": {f"{edge}_{index}": allow for (edge, index), allow in corridor.permissions().items()},
        "netconvert": network["netconvert"],
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file() and p.name != "scenario.json"),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_revision": git_revision(),
    }
    write_json(out_dir / "scenario.json", payload)
    return payload


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(n) for k, n in sorted(values.value_counts().items())}
