import copy
import json
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from flwr.client import NumPyClient


class FedZeroClient(NumPyClient):
    def __init__(self, client_name, net, trainloader, optimizer, opt_args, proximal_mu, device):
        self.client_name = client_name
        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.proximal_mu = proximal_mu
        self.device = device

    def get_parameters(self, config):
        return flwr_get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f'Fitting client: {self.client_name}')
        flwr_set_parameters(self.net, parameters)
        participation_dict = json.loads(config["participation_dict"])
        expected_batches = participation_dict[self.client_name]
        local_round_loss, local_round_acc, statistical_utility = train(
            self.net, self.trainloader, batches=expected_batches, optimizer=self.optimizer,
            opt_args=self.opt_args, proximal_mu=self.proximal_mu, device=self.device
        )
        # print(f'Client {self.client_name} local acc is {local_round_acc}')
        return flwr_get_parameters(self.net), len(self.trainloader), {'local_loss': local_round_loss,
                                                                      'local_acc': local_round_acc,
                                                                      'statistical_utility': statistical_utility}

    def evaluate(self, parameters, config):
        flwr_set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.trainlloader, self.device)
        return float(loss), len(self.trainlloader), {"accuracy": float(accuracy)}


def flwr_get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def flwr_set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    try:
        net.load_state_dict(state_dict, strict=True)
    except:
        pass


def train(net, trainloader, batches, optimizer, opt_args, proximal_mu, device):
    """Train the network on the training set."""
    # print(f"Client {client_name} starts training")
    net.train()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = getattr(torch.optim, optimizer)(net.parameters(), **opt_args)

    sample_loss = []
    local_round_loss, local_round_acc, total = 0.0, 0.0, 0
    if proximal_mu:
        global_model_parameters = copy.deepcopy(list(net.parameters()))
        # for param in net.parameters():
        #     global_model_parameters.append(copy.deepcopy(param.detach()))

    for i in range(batches):
        # print(f"Client {client_name}, batch {i}")
        X, Y = next(iter(trainloader))
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = net(X)
        individual_sample_loss = criterion(outputs, Y)  # required for Oort statistical utility
        sample_loss.extend(individual_sample_loss)
        ce_loss = torch.mean(individual_sample_loss)

        _, predicted = torch.max(outputs.data, 1)
        total += Y.size(0)
        local_round_acc += (predicted == Y).sum().item()

        if proximal_mu:
            tmp_model_parameters = list(net.parameters())
            proximal_term = sum([torch.sum((global_model_parameters[i]-tmp_model_parameters[i])**2) for i in range(len(global_model_parameters))])
            proximal_term = (proximal_mu / 2) * proximal_term
            loss = ce_loss + proximal_term
        else:
            loss = ce_loss
        loss.backward()
        local_round_loss += loss.item()
        optimizer.step()
    
    local_round_loss /= total
    local_round_acc /= total

    return local_round_loss, local_round_acc, statistical_utility(sample_loss)


def statistical_utility(sample_loss: list):
    """Statistical utility as defined in Oort"""
    squared_sum = sum([torch.square(l) for l in sample_loss]).item()
    return len(sample_loss) * np.sqrt(1/len(sample_loss) * squared_sum)


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    net.eval()
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss, accuracy, total = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            X, Y = data[0].to(device), data[1].to(device)
            outputs = net(X)
            loss += criterion(outputs, Y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            accuracy += (predicted == Y).sum().item()
    loss /= total
    accuracy /= total
    return loss, accuracy


class FedZeroClientMock(NumPyClient):
    """Only used in simulations that do not perform any training."""

    def __init__(self, client_name):
        self.client_name = client_name

    def fit(self, parameters, config):
        participation_dict = json.loads(config["participation_dict"])
        expected_batches = participation_dict[self.client_name]
        print(f"Client {self.client_name}: {expected_batches}")
        return parameters, 0, {'local_loss': 0, 'local_acc': 0, 'statistical_utility': 0}
