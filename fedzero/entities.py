import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import pandas as pd
from vessim import TimeSeriesApi

from fedzero.config import BATCH_SIZE, TIMESTEP_IN_MIN


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


class ClientLoadApi(TimeSeriesApi):
    def __init__(self, clients: List[Client], actual: pd.DataFrame, forecast: Optional[pd.DataFrame] = None):
        super().__init__(actual, forecast, fill_method="bfill")
        self._clients = {c.name: c for c in clients}

    def get_clients(self, zones: Optional[List[str]] = None) -> List[Client]:
        """Returs the names of clients present in one of the zones as list."""
        if zones is None:
            return list(self._clients.values())
        return [client for client in self._clients.values() if client.zone in zones]

    def actual(self, dt: datetime, client_name: str) -> float:
        """Returns the actual amount of batches than can be computed during the next timestep."""
        # TODO fedzero crashes with "Cannot retrieve actual data at '2022-06-15 00:00:00' in zone '56_Hong Kong_mid'."
        return (1 - super().actual(dt, zone=client_name)) * self._clients[client_name].batches_per_timestep

    def forecast(self, now: datetime, duration_in_timesteps: int, client_name: str) -> pd.Series:
        """Returns the forecasted amount of batches than can be computed during the next timesteps."""
        return (1 - super().forecast(now, now + timedelta(minutes=TIMESTEP_IN_MIN * duration_in_timesteps),
                                     zone=client_name, frequency=f"{TIMESTEP_IN_MIN}T",
                                     resample_method="bfill")) * self._clients[client_name].batches_per_timestep


class PowerDomainApi(TimeSeriesApi):
    def actual(self, dt: datetime, zone: str) -> float:
        """Returns the actual Ws available during the next timestep."""
        return super().actual(dt, zone=zone) * 60 * TIMESTEP_IN_MIN

    def forecast(self, start_time: datetime, duration_in_timesteps: int, zone: str) -> pd.Series:
        """Returns the forecasted Ws available during the next timesteps."""
        return (super().forecast(start_time, start_time + timedelta(minutes=TIMESTEP_IN_MIN * duration_in_timesteps),
                zone=zone, frequency=f"{TIMESTEP_IN_MIN}T", resample_method="bfill") * 60 * TIMESTEP_IN_MIN)
