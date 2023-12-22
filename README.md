# FedZero

Philipp Wiesner, Ramin Khalili, Dennis Grinwald, Pratik Agrawal, Lauritz Thamsen, and Odej Kao.
"[FedZero: Leveraging Renewable Excess Energy in Federated Learning.](https://arxiv.org/pdf/2305.15092.pdf)"
15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy). 2023.

## Installation

To run FedZero you need a Gurobi license. You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

Then run:
```
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Instructions on how to download and integrate the datasets will follow.

## Usage

Options:
```
  --scenario [unconstrained|global|germany]  [required]
  --dataset [cifar10|cifar100|tiny_imagenet|shakespeare|kwt]  [required]
  --approach TEXT  [required]
  --overselect FLOAT
  --forecast_error [error|no_error|error_no_load_fc]
  --imbalanced_scenario
  --mock
  --seed INTEGER
  --help                                        Show this message and exit.
```

Example:
```
python main.py --scenario global --dataset cifar10 --approach random
```

## Bibtex

```
@proceedings{wiesner2024fedzero,
    title={FedZero: Leveraging Renewable Excess Energy in Federated Learning}, 
    author={Wiesner, Philipp and Khalili, Ramin and Grinwald, Dennis and Agrawal, Pratik and Thamsen, Lauritz and Kao, Odej},
    booktitle={Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy)},
    year={2024},
    publisher={ACM}
}
```
