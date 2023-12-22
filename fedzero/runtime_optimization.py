import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import gurobipy as grb
import numpy as np
import pandas as pd

from fedzero.config import TIMESTEP_IN_MIN, MAX_ROUND_IN_MIN, MIN_LOCAL_EPOCHS, MAX_LOCAL_EPOCHS, \
    CLIENTS_PER_ROUND, GUROBI_ENV
from fedzero.entities import PowerDomainApi, ClientLoadApi, Client

EPSILON = 0.0001


def execute_round(power_domain_api: PowerDomainApi,
                  client_load_api: ClientLoadApi,
                  selection: pd.DataFrame,
                  min_epochs: float,
                  max_epochs: float) -> Tuple[Dict[str, int], timedelta]:
    """Simulates the execution of a training round."""
    selection = _extend_selection_df(selection)
    time_iterator = [_execute_power_domain_round(power_domain_api, client_load_api, zone, p_selection, max_epochs)
                     for zone, p_selection
                     in selection.groupby(lambda c: c.zone)]

    # We progress in all energy domains until <CLIENTS_PER_ROUND> clients have reached `min_epochs`
    participation = {}
    for p_timestep_results in zip(*time_iterator):
        n_clients_above_min_epochs = 0
        for p_participation, now in p_timestep_results:
            for client, part in p_participation.items():
                participation[client] = part
                if participation[client] >= client.batches_per_epoch * min_epochs:
                    n_clients_above_min_epochs += 1
        if n_clients_above_min_epochs >= CLIENTS_PER_ROUND:
            break

    for client, client_participation in participation.items():
        client.record_usage(client_participation)
    round_duration = now - selection.columns[0]

    computed_batches = {}
    for c, p in participation.items():
        minimum = c.batches_per_epoch * MIN_LOCAL_EPOCHS
        if math.floor(p) >= minimum:
            print(f"{c.name} computes {math.floor(p)} (above {minimum})")
            computed_batches[c.name] = math.floor(p)
        else:
            print(f"{c.name} computes {math.floor(p)} (BELOW {minimum})")

    return computed_batches, round_duration


def _execute_power_domain_round(power_domain_api: PowerDomainApi,
                                client_load_api: ClientLoadApi,
                                zone: str,
                                selection: pd.DataFrame,
                                max_epochs: float) -> Tuple[Dict[Client, int], datetime]:
    """Simulates the execution of a training round within a power domain.

    Returns:
        Dict that maps clients to actually computed batches
    """
    first_round = True
    participation: Dict[Client, float] = {c: 0.0 for c in selection.index}
    for now in selection.columns:
        # yield aggregated participation after every completed time step
        if not first_round:
            yield {c: np.floor(p) for c, p in participation.items()}, now

        clients_below_max = [c for c, p in participation.items() if p < c.batches_per_epoch * MAX_LOCAL_EPOCHS]
        if len(clients_below_max) == 0:  # no more training
            continue

        # Minimum of how much the client can compute on excess capacity and until it reaches it max local epochs
        max_batches = {c: int(min(client_load_api.actual(now, c.name), c.batches_per_epoch * max_epochs - participation[c])) 
                       for c in participation.keys()}

        participation = _execute_power_domain_timestep(clients=clients_below_max,
                                                       participation=participation,
                                                       available_energy=power_domain_api.actual(now, zone),
                                                       max_batches=max_batches)
        first_round = False


def _execute_power_domain_timestep(clients: List[Client],
                                   participation: Dict[Client, float],
                                   available_energy: float,
                                   max_batches: Dict[Client, int]) -> Dict[Client, float]:
    """Simulates the execution of a training round for one timestep within a power domain."""
    if len(clients) == 1:
        c = clients[0]
        participation[c] += min(available_energy / c.energy_per_batch, max_batches[c])
    else:
        # First attribute energy to clients that haven't reached MIN_LOCAL_EPOCHS
        participation1, remaining_energy = _attribute_power(MIN_LOCAL_EPOCHS, participation, available_energy,
                                                            max_batches)
        for c, p in participation1.items():
            participation[c] += p
            max_batches[c] -= p

        # Attribute the remaining power by how much energy is still required to reach MAX_LOCAL_EPOCHS
        if remaining_energy > 0:
            participation2, _ = _attribute_power(MAX_LOCAL_EPOCHS, participation, available_energy, max_batches)
            for c, p in participation2.items():
                participation[c] += p

    return participation


def _attribute_power(required_epochs, participation, available_energy, max_batches):
    """Attributes power to all clients below <required_epochs>."""
    missing_batches = {c: c.batches_per_epoch * required_epochs - p for c, p in participation.items()}
    clients = [c for c in participation.keys() if missing_batches[c] > 0]
    weighting = {c: missing_batches[c] * c.energy_per_batch for c in clients}

    model = grb.Model(name="Runtime power attribution model", env=GUROBI_ENV)

    # capping the available_energy to the max of possible usage allows us to use an equality constraint in (1)
    _available_energy = min(available_energy, sum([max_batches[c] * c.energy_per_batch for c in clients]))

    m = {c: model.addVar(lb=0, ub=max_batches[c]) for c in clients}
    y = {c: model.addVar(vtype=grb.GRB.BINARY) for c in clients}
    x = model.addVar(lb=0, ub=_available_energy/EPSILON)

    model.addConstr(grb.quicksum(m[c] * c.energy_per_batch for c in clients) <= _available_energy)
    for c in clients:
        model.addGenConstrIndicator(y[c], False, m[c] == x * weighting[c])
        model.addGenConstrIndicator(y[c], True, m[c] >= max_batches[c])
        model.addGenConstrIndicator(y[c], True, x * weighting[c] >= max_batches[c])

    model.ModelSense = grb.GRB.MAXIMIZE
    model.setObjective(x)
    model.optimize()

    if model.Status == grb.GRB.OPTIMAL:
        participation = {c: xc.X for c, xc in m.items()}
        remaining_energy = available_energy - sum(p * c.energy_per_batch for c, p in participation.items())
        return participation, 0 if np.isclose(remaining_energy, 0) else remaining_energy
    elif model.Status == grb.GRB.INFEASIBLE:
        raise RuntimeError("INFEASIBLE")
    elif model.Status == grb.GRB.INF_OR_UNBD:
        raise RuntimeError("INF_OR_UNBD")
    else:
        raise Exception(model.Status)


def _extend_selection_df(selection: pd.DataFrame):
    start = selection.columns[0]
    time = pd.date_range(start=start, end=start + timedelta(minutes=MAX_ROUND_IN_MIN),
                         freq=f"{TIMESTEP_IN_MIN}min")
    return selection.reindex(columns=time, fill_value=1)
