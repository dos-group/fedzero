import functools
import json
import multiprocessing
import os.path
import random
import urllib
import zipfile

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
from fedzero.kwt.utils.dataset import get_loader
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm

from fedzero.config import NIID_DATA_SEED

ALL_LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def get_dataloaders(dataset: str, num_clients: int, batch_size: int, beta: float):
    np.random.seed(NIID_DATA_SEED)
    random.seed(NIID_DATA_SEED)
    torch.manual_seed(NIID_DATA_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(NIID_DATA_SEED)
    print(f'NIID data seed: {NIID_DATA_SEED}')

    if dataset in ['cifar10', 'cifar100']:
        trainloaders, testloader = load_cifar(dataset.upper(), num_clients, batch_size, beta)
        num_classes = len(np.unique(testloader.dataset.targets))
    elif dataset == 'shakespeare':
        trainloaders, testloader = load_shakespeare(
            train_data_dir='leaf/data/shakespeare/data/train',
            test_data_dir='leaf/data/shakespeare/data/test',
            batch_size=batch_size
        )
        num_classes = len(ALL_LETTERS)
    elif dataset == 'kwt':
        trainloaders, testloader, num_classes = load_speech(num_clients, batch_size)
    elif dataset == 'tiny_imagenet':
        trainloaders, testloader, num_classes = load_tiny_imagenet(num_clients, batch_size, beta)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")
    return trainloaders, testloader, num_classes


def load_cifar(cifar_type: str, num_clients: int, batch_size: int, beta: float):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Mean and Standard deviation of CIFAR10: https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = getattr(torchvision.datasets, cifar_type)(
        "./data", train=True, download=True, transform=train_transforms
    )
    testset = getattr(torchvision.datasets, cifar_type)(
        "./data", train=False, download=True, transform=test_transforms
    )
    trainloaders = []
    if 0.0 < beta < 1.0:
        client_to_data_ids = _get_niid_client_data_ids(trainset, num_clients, beta)
        for client_id in client_to_data_ids:
            tmp_client_img_ids = client_to_data_ids[client_id]
            tmp_train_sampler = SubsetRandomSampler(tmp_client_img_ids)
            _append_to_dataloaders(trainset, batch_size, trainloaders, tmp_train_sampler)
    else:
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        for dataset in datasets:
            _append_to_dataloaders(dataset, batch_size, trainloaders)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, testloader


def load_shakespeare(train_data_dir, test_data_dir, batch_size=10):
    """Loading 100 clients (== speakers) from the Shakespeare dataset"""
    train_data = _load_shakespeare_dataset(train_data_dir)
    test_data = _load_shakespeare_dataset(test_data_dir)
    with open("data/100speakers.json") as f:
        speakers = json.load(f)
    trainloaders = []
    for speaker in speakers:
        train_data_x = _preprocess_shakespeare_data_x(train_data[speaker]['x'])
        train_data_y = _preprocess_shakespeare_data_y(train_data[speaker]['y'])
        trainloader = _build_dataloader(train_data_x, train_data_y, batch_size)
        trainloaders.append(trainloader)
    testset_x = None
    testset_y = None
    for speaker in speakers:
        test_data_x = _preprocess_shakespeare_data_x(test_data[speaker]['x'])
        test_data_y = _preprocess_shakespeare_data_y(test_data[speaker]['y'])
        if testset_x is None:
            testset_x = torch.LongTensor(test_data_x)
            testset_y = torch.LongTensor(test_data_y)
        else:
            testset_x = torch.cat((testset_x, torch.LongTensor(test_data_x)), dim=0)
            testset_y = torch.cat((testset_y, torch.LongTensor(test_data_y)), dim=0)
    testloader = _build_dataloader(testset_x, testset_y, batch_size)
    return trainloaders, testloader


def load_speech(num_clients, batch_size):
    with open("data/kwt/data/training_list.txt", "r") as f:
        train_list = f.read().rstrip().split("\n")
    with open("data/kwt/data/testing_list.txt", "r") as f:
        test_list = f.read().rstrip().split("\n")
    with open("data/kwt/data/label_map.json", "r") as f:
        label_map = json.load(f)

    speakers = sorted(set([_sample_to_speaker(s) for s in train_list]))
    rng = np.random.default_rng(0)
    p = softmax(np.linspace(1, 3, num_clients))
    rng.shuffle(p)
    speaker_to_client = {speaker: rng.choice(range(num_clients), p=p) for speaker in speakers}
    _, counts = np.unique(list(speaker_to_client.values()), return_counts=True)
    print(f"Speakers per client: {min(counts)} min; {max(counts)} max")
    client_samples = {i: [] for i in range(num_clients)}
    for s in train_list:
        client = speaker_to_client[_sample_to_speaker(s)]
        client_samples[client].append(s)

    config = dict(
        data_root="data/kwt/data",
        exp=dict(cache=2,  # cache specs
                 n_workers=0, pin_memory=torch.cuda.is_available()),
        hparams=dict(
            batch_size=batch_size,
            audio=dict(sr=16000, n_mels=40, n_fft=480, win_length=480, hop_length=160, center=False),
            augment=dict(spec_aug=dict(n_time_masks=2, time_mask_width=25, n_freq_masks=2, freq_mask_width=7))
        )
    )

    trainloaders = []
    loader_fn = functools.partial(get_loader, label_map=label_map, config=config, train=True)
    pool = multiprocessing.Pool(processes=8)
    for dataloader in tqdm(pool.imap(func=loader_fn, iterable=client_samples.values()), total=len(client_samples),
                           desc="Initializing clients"):
        trainloaders.append(dataloader)
    pool.close()
    pool.join()

    print("Loading test dataset...")
    testloader = get_loader(test_list, label_map, config, train=False, n_cache_workers=8)
    return trainloaders, testloader, len(label_map)


def load_tiny_imagenet(num_clients: int, batch_size: int, beta: float):
    if not os.path.exists('data/tiny-imagenet-200/'):
        download_and_unzip_tiny_imagenet()
    # Data Transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Mean and Standard deviation of CIFAR10: https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # Data and Creating Train/Test Split
    trainset = torchvision.datasets.ImageFolder(
        root='data/tiny-imagenet-200/train',
        transform=train_transform
    )
    testset = torchvision.datasets.ImageFolder(
        root='data/tiny-imagenet-200/val',
        transform=test_transform
    )
    trainloaders = []
    if 0.0 < beta < 1.0:
        client_to_data_ids = _get_niid_client_data_ids(trainset, num_clients, beta)
        for client_id in client_to_data_ids:
            tmp_client_img_ids = client_to_data_ids[client_id]
            tmp_train_sampler = SubsetRandomSampler(tmp_client_img_ids)
            _append_to_dataloaders(trainset, batch_size, trainloaders, tmp_train_sampler)
    else:
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        for dataset in datasets:
            _append_to_dataloaders(dataset, batch_size, trainloaders)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_classes = len(np.unique(testset.targets))
    return trainloaders, testloader, num_classes


def _sample_to_speaker(s):
    return s.split("_")[0].split("/")[-1]


def _append_to_dataloaders(trainset, batch_size, trainloaders, random_sampler=None):
    if random_sampler is None:
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
    else:
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, sampler=random_sampler))


def _get_niid_client_data_ids(dataset: Dataset, num_clients: int, beta: float):
    labels = np.array(dataset.targets)
    client_to_data_ids = {k: [] for k in range(num_clients)}
    for label_id in range(len(np.unique(labels))):
        idx_batch = [[] for _ in range(num_clients)]
        label_ids = np.where(labels == label_id)[0]
        label_proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        label_proportions = np.cumsum(label_proportions * len(label_ids)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(label_ids, label_proportions))]
        for client_id in range(num_clients):
            client_to_data_ids[client_id] += idx_batch[client_id]
    return client_to_data_ids


def _load_shakespeare_dataset(data_dir):
    """Returns list client data"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    assert len(files) == 1
    f = files[0]
    file_path = os.path.join(data_dir, f)
    print(f"Reading train file {file_path}")
    with open(file_path, 'r') as inf:
        data = json.load(inf)
    return data["user_data"]


def _build_dataloader(dataset_x, dataset_y, batch_size):
    dataset = torch.utils.data.TensorDataset(torch.LongTensor(dataset_x), torch.LongTensor(dataset_y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _letter_to_idx(letter):
    """Returns index of given letter."""
    return max(0, ALL_LETTERS.find(letter))  # treating ' ' as unknown character


def _preprocess_shakespeare_data_x(raw_x_batch):
    x_batch = [[_letter_to_idx(l) for l in x] for x in raw_x_batch]
    return x_batch


def _preprocess_shakespeare_data_y(raw_y_batch):
    y_batch = [_letter_to_idx(c) for c in raw_y_batch]
    return y_batch


def download_and_unzip_tiny_imagenet():
    print('Beginning dataset download with urllib2')
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    path = "data/tiny-imagenet-200.zip"
    urllib.request.urlretrieve(url, path)
    print("Dataset downloaded")
    path_to_zip_file = "data/tiny-imagenet-200.zip"
    directory_to_extract_to = 'data/'
    print("Extracting zip file: %s" % path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print("Extracted at: %s" % directory_to_extract_to)
