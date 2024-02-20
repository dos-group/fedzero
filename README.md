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

## Datasets

- Download of CIFAR and TinyImagenets should work automatically through [torchvision](https://pytorch.org/vision/stable/index.html).

- For the shakespeare experiments, run the following in the base directory:
  ```
  git clone git@github.com:TalwalkarLab/leaf.git
  cd leaf/data/shakespeare
  ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8
  ```

- For the Google speech dataset, run the following in the base directory:
  ```
  cd data/speech
  mkdir data
  sh download_gspeech_v2.sh data
  python make_data_list.py -v data/validation_list.txt -t data/testing_list.txt -d data -o data
  ```

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
