import gurobipy

NIID_DATA_SEED = 42  # controls how the data is split across clients
SAVE_TRAINED_MODELS = False
GUROBI_ENV = gurobipy.Env(params={"OutputFlag": 0})

TIMESTEP_IN_MIN = 1  # minutes
MAX_ROUND_IN_MIN = 60  # minutes
MAX_ROUNDS = 10000
MAX_TIME_IN_DAYS = 7  # currently 11 max
STOPPING_CRITERIA = None  # rounds without improved accuracy

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 10
MIN_LOCAL_EPOCHS = 1
MAX_LOCAL_EPOCHS = 5

SOLAR_SIZE = 800  # W

# Flower
RAY_CLIENT_RESOURCES = {
    "num_cpus": 1,  # CPU threads assigned to each client
    # "num_gpus": 1 / 3
}
RAY_INIT_ARGS = {
    # "num_cpus": 8,  # Number of physically accessible CPUs
    # "num_gpus": 1,  # Number of physically accessible GPUs
    "ignore_reinit_error": True,
    "include_dashboard": False,
}
