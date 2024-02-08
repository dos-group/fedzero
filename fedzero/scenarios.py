from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Union

import pandas as pd
from vessim.signal import HistoricalSignal

from fedzero.config import TIMESTEP_IN_MIN, SOLAR_SIZE, MAX_TIME_IN_DAYS, BATCH_SIZE
from fedzero.entities import Client, ClientLoadApi, PowerDomainApi

_GLOBAL_START = pd.to_datetime("2022-06-08 00:00:00")
_GLOBAL_END = _GLOBAL_START + timedelta(days=MAX_TIME_IN_DAYS, minutes=-TIMESTEP_IN_MIN)

_GERMANY_START = pd.to_datetime("2022-07-15 00:00:00")
_GERMANY_END = _GERMANY_START + timedelta(days=MAX_TIME_IN_DAYS, minutes=-TIMESTEP_IN_MIN)


@dataclass
class Scenario:
    power_domain_api: PowerDomainApi
    client_load_api: ClientLoadApi
    start_date: datetime
    end_date: datetime
    solar_scenario: str
    forecast_error: str
    unconstrained: Union[bool, List[str]]
    imbalanced_scenario: bool

def get_scenario(solar_scenario: str,
                 net_arch_size_factor: float,
                 forecast_error: str = "error",
                 imbalanced_scenario: bool = False) -> Scenario:
    client_sizes = get_client_sizes(net_arch_size_factor)

    # Print information on sizing
    max_solar_per_timestep = SOLAR_SIZE * 60 * TIMESTEP_IN_MIN
    print(f"Max solar energy/timestep: {max_solar_per_timestep:.0f}  Ws")
    for size, props in client_sizes.items():
        batches_per_timestep = props["batches_per_timestep"]
        energy_per_batch = props["energy_per_batch"]
        max_power_per_timestep = batches_per_timestep * energy_per_batch
        print(
            f"{size}: {batches_per_timestep:.2f} batches/timestep;"
            f"{energy_per_batch*1000:.0f} mWs/batch; "
            f"max power/timestep: {max_power_per_timestep:.0f} Ws"
        )

    unconstrained: Union[bool, List[str]] = False
    if solar_scenario == "unconstrained":
        solar_scenario = "global"
        unconstrained = True
    if imbalanced_scenario:
        unconstrained = ["Berlin"]
    start_date, end_date = _load_start_end_date(solar_scenario)

    print("Load solar data...")
    dataset = f"solcast2022_{solar_scenario}"
    power_domain_api = PowerDomainApi(
        HistoricalSignal.from_dataset(dataset, params={"scale":SOLAR_SIZE, "use_forecast":(forecast_error != "no_error")}), unconstrained=unconstrained)

    print("Load client load data...")
    clients_time_series = _load_client_time_series_api(start_date, end_date, client_sizes, power_domain_api.zones, forecast_error,
                                                        unconstrained, imbalanced_scenario)

    return Scenario(power_domain_api=power_domain_api,
                    client_load_api=clients_time_series,
                    start_date=start_date,
                    end_date=end_date,
                    solar_scenario=solar_scenario,
                    forecast_error=forecast_error,
                    unconstrained=unconstrained,
                    imbalanced_scenario=imbalanced_scenario)

def _load_client_time_series_api(start_date: datetime, end_date: datetime,
                                 client_sizes: Dict[str, Dict[str, float]],
                                 power_domain_zones: List[str],
                                 forecast_error: str = "no_error",
                                 unconstrained: Union[bool, List[str]] = False,
                                 imbalanced_scenario: bool = False) -> ClientLoadApi:
    # Load Client information
    clients_data = pd.read_csv("data/clients.csv")
    client_names = clients_data.apply(lambda row: f"{row.name}_{power_domain_zones[row['power_domain']]}_{row['size']}", axis=1)

    # Initialize Clients
    clients = []
    for i, client in clients_data.iterrows():
        zone = power_domain_zones[client["power_domain"]]
        capacity_factor = 1000000000000 if imbalanced_scenario and zone == "Berlin" else 1

        clients.append(Client(name=client_names[i], zone=zone,
                              batches_per_timestep=client_sizes[client["size"]]["batches_per_timestep"] * capacity_factor,
                              energy_per_batch=client_sizes[client["size"]]["energy_per_batch"]))

    # Load actual data
    index = pd.date_range(start_date, end_date, freq=f"{TIMESTEP_IN_MIN}T")
    client_load = pd.read_csv("data/client_load_gpu_used.csv", nrows=len(index)) / 100
    client_load.set_index(index, inplace=True)
    client_load = client_load.set_axis(client_names, axis=1)
    
    # Load forecast data
    if forecast_error == "no_error":
        client_load_reserved = None
    else:
        client_load_reserved = (pd.read_csv("data/client_load_gpu_reserved.csv", nrows=len(index)) / 100)
        client_load_reserved.set_index(index, inplace=True)
        client_load_reserved = client_load_reserved.set_axis(client_names, axis=1)
        if forecast_error == "error_no_load_fc":
            client_load_reserved[:] = 0

    return ClientLoadApi(clients, HistoricalSignal(client_load, client_load_reserved, fill_method="bfill"), unconstrained=unconstrained)


def _load_start_end_date(dataset: str):
    if dataset == "global":
        return _GLOBAL_START, _GLOBAL_END
    elif dataset == "germany":
        return _GERMANY_START, _GERMANY_END
    else:
        raise ValueError("Unknown scenario")


def get_client_sizes(net_arch_size_factor) -> Dict[str, Dict[str, float]]:
    # ResNet benchmarks
    # T4: 70W, 183 images/sec
    # V100: 300W, 639 images/sec
    # A100: 400W, 1237 images/sec
    return {
        "small": {
            "batches_per_timestep": 183 * 60 / BATCH_SIZE / 100 / net_arch_size_factor,
            "energy_per_batch": 70 / 183 * BATCH_SIZE * 100 * net_arch_size_factor,
        },
        "mid": {
            "batches_per_timestep": 639 * 60 / BATCH_SIZE / 100 / net_arch_size_factor,
            "energy_per_batch": 300 / 639 * BATCH_SIZE * 100 * net_arch_size_factor,
        },
        "large": {
            "batches_per_timestep": 1237 * 60 / BATCH_SIZE / 100 / net_arch_size_factor,
            "energy_per_batch": 700 / 1237 * BATCH_SIZE * 100 * net_arch_size_factor,
        },
    }
