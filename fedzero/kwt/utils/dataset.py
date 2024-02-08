import functools
import multiprocessing as mp
import os

import librosa
import numpy as np
import torch
from audiomentations import AddBackgroundNoise
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fedzero.kwt.utils.augment import time_shift, resample, spec_augment


class GoogleSpeechDataset(Dataset):
    """Dataset wrapper for Google Speech Commands V2."""
    
    def __init__(self, data_list: list, audio_settings: dict, label_map: dict = None, aug_settings: dict = None,
                 cache: int = 0, data_root="", n_cache_workers=1):
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings = aug_settings
        self.cache = cache

        if cache:
            self.data_list = init_cache(data_list, audio_settings["sr"], cache, audio_settings, n_cache_workers)
        else:
            self.data_list = data_list
            
        # labels: if no label map is provided, will not load labels. (Use for inference)
        if label_map is not None:
            self.label_list = []
            label_2_idx = {v: int(k) for k, v in label_map.items()}
            for path in data_list:
                self.label_list.append(label_2_idx[path.split("/")[-2]])
        else:
            self.label_list = None

        if aug_settings is not None:
            if "bg_noise" in self.aug_settings:
                self.bg_adder = AddBackgroundNoise(sounds_path=os.path.join(data_root,
                                                                            aug_settings["bg_noise"]["bg_folder"]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.cache:
            x = self.data_list[idx]
        else:
            x = librosa.load(self.data_list[idx], self.audio_settings["sr"])[0]

        x = self.transform(x)

        if self.label_list is not None:
            label = self.label_list[idx]
            return x, label
        else:
            return x


    def transform(self, x):
        """Applies necessary preprocessing to audio.

        Args:
            x (np.ndarray) - Input waveform; array of shape (n_samples, ).
        
        Returns:
            x (torch.FloatTensor) - MFCC matrix of shape (n_mfcc, T).
        """

        sr = self.audio_settings["sr"]

        ###################
        # Waveform 
        ###################

        if self.cache < 2:
            if self.aug_settings is not None:
                if "bg_noise" in self.aug_settings:
                    x = self.bg_adder(samples=x, sample_rate=sr)

                if "time_shift" in self.aug_settings:
                    x = time_shift(x, sr, **self.aug_settings["time_shift"])

                if "resample" in self.aug_settings:
                    x, _ = resample(x, sr, **self.aug_settings["resample"])
            
            x = librosa.util.fix_length(x, size=sr)

            ###################
            # Spectrogram
            ###################
        
            x = librosa.feature.melspectrogram(y=x, **self.audio_settings)        
            x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=self.audio_settings["n_mels"])

        if self.aug_settings is not None:
            if "spec_aug" in self.aug_settings:
                x = np.array(x)  # make a copy to own the memory
                x = spec_augment(x, **self.aug_settings["spec_aug"])

        x = torch.from_numpy(x).float().unsqueeze(0)
        return x


def cache_item_loader(path: str, sr: int, cache_level: int, audio_settings: dict) -> np.ndarray:
    x = librosa.load(os.path.join("data", "kwt", path), sr=sr)[0]
    if cache_level == 2:
        x = librosa.util.fix_length(x, size=sr)
        x = librosa.feature.melspectrogram(y=x, **audio_settings)        
        x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
    return x


def init_cache(data_list: list, sr: int, cache_level: int, audio_settings: dict, n_cache_workers: int) -> list:
    """Loads entire dataset into memory for later use.

    Args:
        data_list (list): List of data items.
        sr (int): Sampling rate.
        cache_level (int): Cache levels, one of (1, 2), caching wavs and spectrograms respectively.
        n_cache_workers (int): Number of workers.

    Returns:
        cache (list): List of data items.
    """
    cache = []
    if n_cache_workers > 1:
        loader_fn = functools.partial(cache_item_loader, sr=sr, cache_level=cache_level, audio_settings=audio_settings)
        pool = mp.Pool(n_cache_workers)
        for audio in tqdm(pool.imap(func=loader_fn, iterable=data_list, chunksize=10), total=len(data_list)):
            cache.append(audio)
        pool.close()
        pool.join()
    else:
        for item in data_list:
            cache.append(cache_item_loader(item, sr=sr, cache_level=cache_level, audio_settings=audio_settings))
    return cache


def get_loader(data_list, label_map, config, train=True, n_cache_workers=1):
    """Creates dataloaders for training, validation and testing.

    Args:
        config (dict): Dict containing various settings for the training run.
        train (bool): Training or evaluation mode.
        
    Returns:
        dataloader (DataLoader): DataLoader wrapper for training/validation/test data.
    """
    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=label_map,
        audio_settings=config["hparams"]["audio"],
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"],
        data_root=config["data_root"],
        n_cache_workers=n_cache_workers,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True if train else False
    )

    return dataloader

    