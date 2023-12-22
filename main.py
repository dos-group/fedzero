import os
from dataclasses import dataclass
from typing import Dict, Optional

import click
import flwr
import pandas as pd
import torch
from flwr.client import NumPyClient
from flwr.server import ServerConfig
from torch.utils.tensorboard import SummaryWriter

from fedzero.config import NUM_CLIENTS, BATCH_SIZE, CLIENTS_PER_ROUND, MIN_LOCAL_EPOCHS, MAX_LOCAL_EPOCHS, \
    MAX_ROUNDS, RAY_CLIENT_RESOURCES, RAY_INIT_ARGS, SAVE_TRAINED_MODELS
from fedzero.datasets import get_dataloaders
from fedzero.fl_client import flwr_get_parameters, flwr_set_parameters, test, FedZeroClient, FedZeroClientMock
from fedzero.fl_server import FedZeroServer
from fedzero.models import create_model
from fedzero.scenarios import get_scenario, Scenario
from fedzero.selection_strategy import SelectionStrategy, RandomSelectionStrategy, FedZeroSelectionStrategy, \
    OortSelectionStrategy
from fedzero.utility import StaticJudge, StatUtilityJudge


@dataclass
class Experiment:
    scenario: Scenario
    selection_strategy: SelectionStrategy
    overselect: float
    net_arch: str
    optimizer: str
    opt_args: Dict
    beta: Optional[float]
    proximal_mu: float
    dataset: str

    @property
    def name(self):
        if self.proximal_mu:
            aggregation_strategy = f"FedProx_{self.proximal_mu}"
        else:
            aggregation_strategy = "FedAvg"
        iid_str = "noniid" if self.beta is None else f"b={self.beta:.1f}"
        scenario_str = self.scenario.solar_scenario if self.scenario.constraints() != "no_constr" else "no_constr"
        imbalanced_str = "_imbalanced" if self.scenario.imbalanced_scenario else ""
        overselect_str = f"_{self.overselect:.1f}K" if self.overselect > 1 else ""
        error_str = ""
        if "fedzero" in str(self.selection_strategy) and self.scenario.forecast_error != "error":
            error_str = f",{self.scenario.forecast_error}"

        experiment_name = (f"{scenario_str}{imbalanced_str},"
                           f"{self.dataset},{iid_str},{self.net_arch},"
                           f"{aggregation_strategy},"
                           f"{self.selection_strategy}{overselect_str}{error_str}")

        i = 0
        while os.path.exists(f"runs/{experiment_name},i={i}"):
            i += 1
        return experiment_name + f",i={i}"


def get_model_and_hyperparameters(dataset, iid):
    optimizer = "SGD"
    if dataset == "cifar10":
        net_arch = 'resnet18'
        net_arch_size_factor = 1
        opt_args = {'lr': 0.001, 'weight_decay': 5e-4, 'momentum': 0.9}
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    elif dataset == "cifar100":
        net_arch = 'densenet121'
        net_arch_size_factor = 1.5
        opt_args = {'lr': 0.001, 'weight_decay': 5e-4, 'momentum': 0.9}
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    elif dataset == "tiny_imagenet":
        net_arch = 'efficientnet_b1'
        net_arch_size_factor = 1.4
        opt_args = {'lr': 0.001}
        optimizer = "Adam"
        if iid:
            proximal_mu = 0
            beta = 1
        else:
            proximal_mu = 0.1
            beta = 0.5
    elif dataset == "shakespeare":
        net_arch = 'SimpleLSTM'
        net_arch_size_factor = 1.5
        opt_args = {'lr': 0.8}
        # iid is not implemented for Shakespeare
        proximal_mu = 0.001
        beta = None
    elif dataset == "kwt":
        optimizer = "AdamW"
        net_arch = 'kwt-1'
        net_arch_size_factor = 1.9
        opt_args = {'lr': 0.001, 'weight_decay': 0.1}
        # iid is not implemented for Google Speech
        proximal_mu = 0.001
        beta = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return net_arch, net_arch_size_factor, optimizer, opt_args, proximal_mu, beta


def simulate_fl_training(experiment: Experiment, device: torch.device, mock: bool) -> None:
    print(f"Starting experiment {experiment.name} ...")
    writer = SummaryWriter(log_dir="runs/"+experiment.name)

    os.makedirs(f'trained_models/{experiment.name}/', exist_ok=True)

    trainloaders, testloader, num_classes = get_dataloaders(
        dataset=experiment.dataset,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        beta=experiment.beta
    )

    print(f"Sample distribution: {pd.Series([len(t.batch_sampler.sampler) for t in trainloaders]).describe()}")

    # Initialize 1 model for initial params
    model = create_model(model_arch=experiment.net_arch, num_classes=num_classes, device=device)
    initial_params = flwr_get_parameters(model)

    for i, (c, trainloader) in enumerate(zip(experiment.scenario.client_load_api.get_clients(), trainloaders)):
        c.num_samples = len(trainloader) * BATCH_SIZE
        required_time = c.num_samples / (c.batches_per_timestep * BATCH_SIZE)
        # if required_time <= 5 or required_time >= 55:
        print(f"{i+1:>3}: {required_time:.0f} mins ({len(trainloader)} batches at {c.batches_per_timestep:.1f} batches/min)")

    def client_fn(client_name) -> NumPyClient:
        client_id = int(client_name.split('_')[0])
        if mock:
            return FedZeroClientMock(client_name=client_name)
        else:
            return FedZeroClient(client_name=client_name,
                                 net=model,
                                 trainloader=trainloaders[client_id],
                                 optimizer=experiment.optimizer,
                                 opt_args=experiment.opt_args,
                                 proximal_mu=experiment.proximal_mu,
                                 device=device)

    # The `evaluate` function will be by Flower called after every round
    def server_eval_fn(server_round: int, parameters: flwr.common.NDArrays, config: Dict[str, flwr.common.Scalar]):
        net = create_model(model_arch=experiment.net_arch, num_classes=num_classes, device=device)
        flwr_set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, testloader, device=device)
        net_state_dict = net.state_dict()
        if SAVE_TRAINED_MODELS and net_state_dict is not None:
            torch.save(net_state_dict, f"trained_models/{experiment.name}/round_{server_round}")
        print(f"Server-side evaluation, round: {server_round},  loss: {loss},  accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}

    # Pass parameters to the Strategy for server-side parameter initialization
    strategy = flwr.server.strategy.FedAvg(
        fraction_fit=NUM_CLIENTS / CLIENTS_PER_ROUND,
        fraction_evaluate=0,  # we only do server side evaluation
        initial_parameters=flwr.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=server_eval_fn
    )

    server = FedZeroServer(scenario=experiment.scenario,
                           selection_strategy=experiment.selection_strategy,
                           min_epochs=MIN_LOCAL_EPOCHS,
                           max_epochs=MAX_LOCAL_EPOCHS,
                           strategy=strategy,
                           writer=writer)

    flwr.fedzero.start_simulation(
        client_fn=client_fn,
        clients_ids=[c.name for c in experiment.scenario.client_load_api.get_clients()],
        server=server,
        config=ServerConfig(num_rounds=MAX_ROUNDS),
        client_resources=RAY_CLIENT_RESOURCES,
        ray_init_args=RAY_INIT_ARGS,
        keep_initialised=True
    )
    print("Simulation finished successfully.")


@click.command()
@click.option('--scenario', type=click.Choice(["unconstrained", "global", "germany"]), required=True)
@click.option('--dataset', type=click.Choice(["cifar10", "cifar100", "tiny_imagenet", "shakespeare", "kwt"]), required=True)
@click.option('--approach', type=str, required=True)  # fedzero_a{alpha}_e{exclusion_factor}, fedzero_static, random, random_fc, oort, oort_fc
@click.option('--overselect', type=float, default=1)  # K
@click.option('--forecast_error', type=click.Choice(["error", "no_error", "error_no_load_fc"]), default="error")
@click.option('--imbalanced_scenario', is_flag=True, default=False)
@click.option('--mock', is_flag=True, default=False)
@click.option('--seed', type=int, default=None)
def main(scenario: str, dataset: str, approach: str, overselect: float, forecast_error: str,
         imbalanced_scenario: bool, mock: bool, seed: Optional[int]):
    assert overselect >= 1
    clients_per_round = int(CLIENTS_PER_ROUND * overselect)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    net_arch, net_arch_size_factor, optimizer, opt_args, proximal_mu, beta = get_model_and_hyperparameters(dataset, iid=False)

    if "fedzero" not in approach:
        forecast_error = "no_error"  # performance optimization, no other approach makes use of forecasts
    scenario = get_scenario(scenario,
                            net_arch_size_factor=net_arch_size_factor,
                            forecast_error=forecast_error,
                            imbalanced_scenario=imbalanced_scenario)
    if approach == "random":
        selection_strategy = RandomSelectionStrategy(clients_per_round=clients_per_round, seed=seed)
    elif approach == "random_fc":
        selection_strategy = RandomSelectionStrategy(clients_per_round=clients_per_round, seed=seed,
                                                     use_forecasts=True, min_epochs=MIN_LOCAL_EPOCHS)
    elif approach == "fedzero_static":
        selection_strategy = FedZeroSelectionStrategy(
            clients_per_round=clients_per_round,
            utility_judge=StaticJudge(scenario.client_load_api.get_clients()),
            alpha=0,
            exclusion_factor=0,
            min_epochs=MIN_LOCAL_EPOCHS,
            max_epochs=MAX_LOCAL_EPOCHS,
            seed=seed,
        )
    elif "fedzero" in approach:
        split = approach.split("_")
        assert len(split) == 3, ("Invalid approach format: FedZero has the format fedzero_{alpha}_{exclusion_factor}, "
                                 "e.g. fedzero_1_1")
        selection_strategy = FedZeroSelectionStrategy(
            clients_per_round=clients_per_round,
            utility_judge=StatUtilityJudge(scenario.client_load_api.get_clients()),
            alpha=float(split[1]),
            exclusion_factor=float(split[2]),
            min_epochs=MIN_LOCAL_EPOCHS,
            max_epochs=MAX_LOCAL_EPOCHS,
            seed=seed,
        )
    elif approach == "oort":
        selection_strategy = OortSelectionStrategy(clients_per_round=clients_per_round, seed=seed)
    elif approach == "oort_fc":
        selection_strategy = OortSelectionStrategy(clients_per_round=clients_per_round, use_forecasts=True, seed=seed)
    else:
        raise click.ClickException(f"Unknown approach: {approach}")

    experiment = Experiment(scenario=scenario,
                            selection_strategy=selection_strategy,
                            overselect=overselect,
                            net_arch=net_arch,
                            optimizer=optimizer,
                            opt_args=opt_args,
                            beta=beta,
                            proximal_mu=proximal_mu,
                            dataset=dataset)
    simulate_fl_training(experiment, device, mock)


if __name__ == "__main__":
    main()
