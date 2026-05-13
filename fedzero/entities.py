import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import vessim as vs

from fedzero.config import BATCH_SIZE, TIMESTEP_IN_MIN
from fedzero.forecast import Forecast


class Client:
    def __init__(self, name: str, zone: str, batches_per_timestep: float, energy_per_batch: float):
        self.name = name
        self.zone = zone
        self.batches_per_timestep = batches_per_timestep
        self.energy_per_batch = energy_per_batch # Ws

        self.participated_rounds = 0
        self.participated_batches = 0
        self.num_samples = 0.0
        self._statistical_utilities: Dict = {}

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.num_samples / BATCH_SIZE)

    def __repr__(self):
        return f"Client({self.name})"

    def __lt__(self, other):  # Sortable as we use instances of this class for DataFrame indexing
        return self.name < other.name

    def record_usage(self, computed_batches: int) -> None:
        if computed_batches > 0:
            self.participated_rounds += 1
            self.participated_batches += computed_batches

    def record_statistical_utility(self, server_round: int, utility: float) -> None:
        self._statistical_utilities[server_round] = utility

    def statistical_utility(self) -> float:
        if len(self._statistical_utilities) == 0:
            return (self.num_samples)  # by convention (copied from the original Oort code)
        return list(self._statistical_utilities.values())[-1]

    def participated_in_last_round(self, round_number) -> bool:
        try:
            return list(self._statistical_utilities.keys())[-1] == round_number - 1
        except IndexError:
            return False


class ClientLoadApi:
    """Per-client GPU-load adapter for the FL scheduler.

    Wraps one `vs.Trace` per client (the "fraction GPU busy with non-FL
    workload" signal) plus an optional archived `Forecast`. Translates
    `(now, client_name)` queries into "available batches per timestep",
    the unit the LP and runtime simulator operate in.
    """

    def __init__(
        self,
        clients: List[Client],
        actuals: Dict[str, vs.Trace],
        sim_start: datetime,
        forecast: Optional[Forecast] = None,
        unconstrained: Union[bool, List[str]] = False,
    ):
        self._actuals = actuals
        self._forecast = forecast
        self._sim_start = pd.Timestamp(sim_start)
        self._clients = {c.name: c for c in clients}
        if isinstance(unconstrained, list):
            self._unconstrained = [c.name for c in clients if c.zone in unconstrained]
        elif unconstrained:
            self._unconstrained = list(self._clients.keys())
        else:
            self._unconstrained = []

    def get_clients(self, zones: Optional[List[str]] = None) -> List[Client]:
        """Returns the clients present in one of the zones as list."""
        if zones is None:
            return list(self._clients.values())
        return [c for c in self._clients.values() if c.zone in zones]

    def actual(self, dt: datetime, client_name: str) -> float:
        """Returns the actual amount of batches than can be computed during the next timestep."""
        bpt = self._clients[client_name].batches_per_timestep
        if client_name in self._unconstrained:
            return bpt
        elapsed = (pd.Timestamp(dt) - self._sim_start).total_seconds()
        return (1 - self._actuals[client_name].at(elapsed)) * bpt

    def forecast(self, now: datetime, duration_in_timesteps: int, client_name: str) -> pd.Series:
        """Returns the forecasted amount of batches than can be computed during the next timesteps."""
        bpt = self._clients[client_name].batches_per_timestep
        step = timedelta(minutes=TIMESTEP_IN_MIN)
        end = now + step * duration_in_timesteps
        ts = pd.date_range(start=pd.Timestamp(now) + step, end=pd.Timestamp(end), freq=step)
        if client_name in self._unconstrained:
            return pd.Series(bpt, index=ts)
        if self._forecast is not None:
            load = self._forecast.window(now, start=now, end=end, freq=step, column=client_name, fill="bfill")
        else:
            offsets = [(t - self._sim_start).total_seconds() for t in ts]
            load = pd.Series([self._actuals[client_name].at(o) for o in offsets], index=ts)
        return (1 - load) * bpt


class PowerDomainApi:
    """Per-zone solar adapter for the FL scheduler.

    Wraps one `vs.Trace` per zone plus an optional archived `Forecast`.
    Translates `(now, zone)` queries into "Ws available per timestep", the
    energy unit the LP operates in (`power_W * 60 * TIMESTEP_IN_MIN`).
    """

    _UNCONSTRAINED_VALUE = 1_000_000_000_000.0

    def __init__(
        self,
        actuals: Dict[str, vs.Trace],
        sim_start: datetime,
        forecast: Optional[Forecast] = None,
        unconstrained: Union[bool, List[str]] = False,
    ):
        self._actuals = actuals
        self._forecast = forecast
        self._sim_start = pd.Timestamp(sim_start)
        if isinstance(unconstrained, list):
            self._unconstrained = unconstrained
        elif unconstrained:
            self._unconstrained = self.zones
        else:
            self._unconstrained = []

    @property
    def zones(self) -> List[str]:
        return list(self._actuals.keys())

    def actual(self, dt: datetime, zone: str) -> float:
        """Returns the actual Ws available during the next timestep."""
        if zone in self._unconstrained:
            return self._UNCONSTRAINED_VALUE
        elapsed = (pd.Timestamp(dt) - self._sim_start).total_seconds()
        return self._actuals[zone].at(elapsed) * 60 * TIMESTEP_IN_MIN

    def forecast(self, start_time: datetime, duration_in_timesteps: int, zone: str) -> pd.Series:
        """Returns the forecasted Ws available during the next timesteps."""
        step = timedelta(minutes=TIMESTEP_IN_MIN)
        end = start_time + step * duration_in_timesteps
        ts = pd.date_range(start=pd.Timestamp(start_time) + step, end=pd.Timestamp(end), freq=step)
        if zone in self._unconstrained:
            return pd.Series(self._UNCONSTRAINED_VALUE, index=ts)
        if self._forecast is not None:
            power = self._forecast.window(start_time, start=start_time, end=end, freq=step, column=zone, fill="bfill")
        else:
            offsets = [(t - self._sim_start).total_seconds() for t in ts]
            power = pd.Series([self._actuals[zone].at(o) for o in offsets], index=ts)
        return power * 60 * TIMESTEP_IN_MIN
