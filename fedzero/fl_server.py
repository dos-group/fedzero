# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""
import json
import time
from datetime import datetime, timedelta
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Parameters, Scalar
from flwr.common.logger import log
from flwr.server import Server, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import fit_clients, FitResultsAndFailures
from flwr.server.strategy import Strategy
from torch.utils.tensorboard import SummaryWriter

from fedzero.config import STOPPING_CRITERIA
from fedzero.runtime_optimization import execute_round
from fedzero.scenarios import Scenario
from fedzero.selection_strategy import SelectionStrategy


class FedZeroClientManager(SimpleClientManager):
    """For this client manager the next sample is set manually on each round in the FedZeroServer."""

    def set_next_sample(self, next_sample: List[str]):
        self.next_sample = [self.clients[cid] for cid in next_sample]

    def sample(self, num_clients, min_num_clients=None, criterion=None) -> List[ClientProxy]:
        return self.next_sample


class FedZeroServer(Server):
    """Flower server adapted to support discrete event fedzero over time series data."""

    def __init__(self, *,
                 scenario: Scenario,
                 selection_strategy: SelectionStrategy,
                 min_epochs: float,
                 max_epochs: float,
                 strategy: Strategy,
                 writer: SummaryWriter,
                 ) -> None:
        self.power_domain_api = scenario.power_domain_api
        self.client_load_api = scenario.client_load_api
        self.selection_strategy = selection_strategy
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.start_time = scenario.start_date
        self.end_time = scenario.end_date
        self.writer = writer
        super(FedZeroServer, self).__init__(client_manager=FedZeroClientManager(), strategy=strategy)

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO, f"initial parameters (loss, other metrics): {res[0]}, {res[1]}")
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
            self.writer.add_scalar("timestamp", 0, global_step=0, walltime=self.start_time.timestamp())
            self.writer.add_scalar("val_loss", res[0], global_step=0, walltime=self.start_time.timestamp())
            self.writer.add_scalar("accuracy", res[1]["accuracy"], global_step=0, walltime=self.start_time.timestamp())

        # Run federated learning for num_rounds
        now = self.start_time
        best_accuracy = 0
        rounds_without_accuracy_improvement = 0
        log(INFO, f"FL starting at {now}")
        for current_round in range(1, num_rounds + 1):
            start_time = time.time()
            # Train model and replace previous global model
            while True:
                start_time_fit = time.time()
                res_fit = self.fit_round_ra(server_round=current_round, now=now, timeout=timeout)
                tb_props = dict(global_step=current_round, walltime=now.timestamp())
                if res_fit:
                    print(f'Select & fit time: {time.time() - start_time_fit:.1f} s')
                    parameters, metrics, _, participation, new_now = res_fit  # fit_metrics_aggregated
                    duration = new_now - now
                    if parameters:
                        self.parameters = parameters
                    self.writer.add_scalar("train_loss", metrics.get("local_train_loss", np.nan), **tb_props)
                    self.writer.add_scalar("train_accuracy", metrics.get("local_train_acc", np.nan), **tb_props)
                    break
                now += timedelta(minutes=5)  # wait for 5 min and try again

            now = new_now
            # Evaluate model using strategy implementation
            # We don't do client side evaluation!
            start_time_eval = time.time()
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                if metrics_cen["accuracy"] > best_accuracy:
                    best_accuracy = metrics_cen["accuracy"]
                    rounds_without_accuracy_improvement = 0
                else:
                    rounds_without_accuracy_improvement += 1

                tb_props = dict(global_step=current_round, walltime=now.timestamp())

                log(INFO, f"fit progress: ({current_round}, {loss_cen}, {metrics_cen}, {now})")
                self.writer.add_scalar("timestamp", now.timestamp() - self.start_time.timestamp(), **tb_props)
                self.writer.add_scalar("val_loss", loss_cen, **tb_props)
                self.writer.add_scalar("accuracy", metrics_cen["accuracy"], **tb_props)
            print(f'Eval time: {time.time() - start_time_eval:.1f} s')

            # Report round duration
            round_duration_in_min = int(duration.seconds / 60)
            self.writer.add_scalar("round_duration", round_duration_in_min, **tb_props)

            # Report energy usage
            used_energy = _ws_to_kwh(sum(client.participated_batches * client.energy_per_batch for client in self.client_load_api.get_clients()))
            self.writer.add_scalar("energy/total", used_energy, **tb_props)

            # Report round energy usage
            round_energy = sum(_ws_to_kwh(c.energy_per_batch) * participation[c.name] for c in self.client_load_api.get_clients()
                               if c.name in participation)
            self.writer.add_scalar("energy/round", round_energy, **tb_props)

            # Report energy per domain
            round_energy_per_domain_round = {zone: 0.0 for zone in self.power_domain_api.zones()}
            for zone in self.power_domain_api.zones():
                clients_in_zone = [client for client in self.client_load_api.get_clients() if client.zone == zone]
                self.writer.add_scalar(f"energy_per_domain/{zone}", _ws_to_kwh(sum(client.participated_batches * client.energy_per_batch for client in clients_in_zone)), **tb_props)
                for c in clients_in_zone:
                    if c.name in participation:
                        round_energy_per_domain_round[zone] += _ws_to_kwh(c.energy_per_batch) * participation[c.name]
            for domain_name, energy in round_energy_per_domain_round.items():
                self.writer.add_scalar(f"energy_per_domain_round/{domain_name}", energy, **tb_props)

            # Report participation per client
            for c in self.client_load_api.get_clients():
                client_participation = participation[c.name] if c.name in participation else 0
                self.writer.add_scalar(f"client_participation/{c.name}", client_participation, **tb_props)

            if STOPPING_CRITERIA is not None and rounds_without_accuracy_improvement >= STOPPING_CRITERIA:
                log(INFO, f"STOPPING no progress since {STOPPING_CRITERIA} rounds.: Best acc: {best_accuracy}")
                break
            if now >= self.end_time:
                log(INFO, "STOPPING max time reached before model converged.")
                break
            print(f'Round time: {time.time() - start_time:.1f} s')

        log(INFO, "FL finished.")
        return history

    def fit_round_ra(self, server_round: int, now: datetime, timeout: Optional[float]) -> Optional[
        Tuple[Optional[Parameters], Dict, FitResultsAndFailures, Dict[str, int], datetime]]:
        """Perform a single round of federated averaging."""
        selection = self.selection_strategy.select(self.power_domain_api, self.client_load_api, round_number=server_round, now=now)
        if selection is None:
            log(INFO, f"fit_round {server_round} ({now}) no clients selected, cancel")
            return None

        expected_duration = len(selection.columns)
        participation, round_duration = execute_round(self.power_domain_api, self.client_load_api, selection, self.min_epochs, self.max_epochs)
        log(DEBUG, f"Round {server_round} ({now}) training {int(round_duration.seconds/60)} min ({expected_duration} min expected) "
                   f"on {len(participation)} clients: {participation}")
        if len(participation) == 0:
            log(INFO, f"fit_round {server_round} ({now}) no clients reached min epochs.")
            return None, {}, None, participation, now + round_duration

        # Get clients and their respective instructions from strategy
        self._client_manager: FedZeroClientManager
        self._client_manager.set_next_sample(list(participation.keys()))
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        for _, fit_ins in client_instructions:
            # We send the full participation dict to all clients
            fit_ins.config["participation_dict"] = json.dumps(participation)  # needs to be str for grcp

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        if failures:
            raise RuntimeError(f"Round {server_round} ({now}) received {len(results)} results and {len(failures)} failures")

        training_losses = {client_proxy.cid: result.metrics["local_loss"] for client_proxy, result in results}
        training_accs = {client_proxy.cid: result.metrics["local_acc"] for client_proxy, result in results}
        statistical_utilities = {client_proxy.cid: result.metrics["statistical_utility"] for client_proxy, result in results}

        agg_local_train_loss = np.mean([loss for loss in training_losses.values()])
        agg_local_train_acc = np.mean([acc for acc in training_accs.values()])

        for client in self.client_load_api.get_clients():
            if client.name in training_losses:
                client.record_statistical_utility(server_round, statistical_utilities[client.name])

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        metrics_aggregated["local_train_loss"] = agg_local_train_loss
        metrics_aggregated["local_train_acc"] = agg_local_train_acc
        return parameters_aggregated, metrics_aggregated, (results, failures), participation, now + round_duration


def _ws_to_kwh(ws: float) -> float:
    return ws / 3600 / 1000
