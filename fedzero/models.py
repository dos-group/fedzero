import torch
import torch.nn as nn
import torchvision

from fedzero.kwt.utils.misc import get_kwt_model, count_params


def create_model(model_arch, num_classes, device):
    if model_arch == 'SimpleLSTM':
        model = SimpleLSTM(num_classes, device=device)
    elif model_arch == 'densenet121':
        model = torchvision.models.densenet121(weights='DEFAULT')
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_arch == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_arch == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_arch in ["kwt-1", "kwt-2", "kwt-3"]:
        model = get_kwt_model(dict(name=model_arch))
    else:
        raise NotImplementedError(f"Model '{model_arch}' not implemented")
    print(f"Created model with {count_params(model)} parameters.")
    return model.to(device)


class SimpleLSTM(nn.Module):
    """Simple LSTM for next character prediction.

    As in:
        Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith.
        "Federated Optimization for Heterogeneous Networks." MLSys. 2020.
    """

    def __init__(self, num_classes, hidden_dim=100, n_layers=2, embedding_dim=8, device="cpu"):
        super().__init__()
        self.device = device
        self.embedding = torch.nn.Embedding(num_classes, embedding_dim)
        self.hidden_dim = hidden_dim if hidden_dim > 0 else embedding_dim
        self.n_layers = n_layers

        # shape of input/output tensors: (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, num_classes)

        # hidden state and cell state (cell state is in LSTM only)
        self.h0 = torch.zeros(self.n_layers, 10, self.hidden_dim).requires_grad_().to(self.device)
        self.c0 = torch.zeros(self.n_layers, 10, self.hidden_dim).requires_grad_().to(self.device)

    def forward(self, X_batch):
        x = self.embedding(X_batch)  # word embedding
        if self.h0.size(1) == x.size(0):
            self.h0.data.zero_()
            self.c0.data.zero_()
        else:
            # resize hidden vars
            self.h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
            self.c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, _ = self.lstm(x, (self.h0.detach(), self.c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def zero_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim)
