import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union

import pandas as pd
from vessim.signal import HistoricalSignal

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


class ClientLoadApi:
    def __init__(self, clients: List[Client], signal: HistoricalSignal, unconstrained: Union[bool, List[str]] = False):
        self.signal = signal
        self._clients = {c.name: c for c in clients}
        if isinstance(unconstrained, list):
            self._unconstrained = [client.name for client in clients if client.zone in unconstrained]
        elif unconstrained:
            self._unconstrained = [client.name for client in clients]
        else:
            self._unconstrained = []
        self.signal = signal

    def get_clients(self, zones: Optional[List[str]] = None) -> List[Client]:
        """Returs the names of clients present in one of the zones as list."""
        if zones is None:
            return list(self._clients.values())
        return [client for client in self._clients.values() if client.zone in zones]

    def actual(self, dt: datetime, client_name: str) -> float:
        """Returns the actual amount of batches than can be computed during the next timestep."""
        if client_name in self._unconstrained:
            return self._clients[client_name].batches_per_timestep
        return (1 - self.signal.at(dt, column=client_name)) * self._clients[client_name].batches_per_timestep

    def forecast(self, now: datetime, duration_in_timesteps: int, client_name: str) -> pd.Series:
        """Returns the forecasted amount of batches than can be computed during the next timesteps."""
        forecast = (1 - self.signal.forecast(now, now + timedelta(minutes=TIMESTEP_IN_MIN * duration_in_timesteps),
                                     column=client_name, frequency=f"{TIMESTEP_IN_MIN}T",
                                     resample_method="bfill")) * self._clients[client_name].batches_per_timestep
        if client_name in self._unconstrained:
            forecast[:] = self._clients[client_name].batches_per_timestep
        return forecast


class PowerDomainApi:
    def __init__(self, signal: HistoricalSignal, unconstrained: Union[bool, List[str]] = False):
        self.signal = signal
        if isinstance(unconstrained, list):
            self._unconstrained = unconstrained
        elif unconstrained:
            self._unconstrained = self.zones
        else:
            self._unconstrained = []

    @property
    def zones(self) -> List[str]:
        return self.signal.columns()

    def actual(self, dt: datetime, zone: str) -> float:
        """Returns the actual Ws available during the next timestep."""
        if zone in self._unconstrained:
            return 1000000000000.0
        return self.signal.at(dt, column=zone) * 60 * TIMESTEP_IN_MIN
    
    def forecast(self, start_time: datetime, duration_in_timesteps: int, zone: str) -> pd.Series:
        """Returns the forecasted Ws available during the next timesteps."""
        forecast = (self.signal.forecast(start_time, start_time + timedelta(minutes=TIMESTEP_IN_MIN * duration_in_timesteps),
                column=zone, frequency=f"{TIMESTEP_IN_MIN}T", resample_method="bfill") * 60 * TIMESTEP_IN_MIN)
        if zone in self._unconstrained:
            forecast[:] = 1000000000000.0
        return forecast
