import os
from collections import defaultdict
from tqdm import tqdm
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns

sns.set()


from ctt.data_loading.loader import ContactDataset


def extract_infectiousnesses(data_path):
    dataset = ContactDataset(path=data_path, preload=True)

    samples = defaultdict(list)

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        for key in sample:
            samples[key].append(sample[key])

    for key in samples:
        if key.startswith("encounter"):
            continue
        samples[key] = torch.stack(samples[key]).numpy()

    def filter_thresh(x, thresh=0.0):
        x = np.asarray(x)
        x = x[x > thresh]
        return x

    all_infectiousnesses = filter_thresh(samples["infectiousness_history"][:, 0, 0])
    return all_infectiousnesses


def extract_all_infectiousnesses(data_path):
    data_paths = glob.glob(os.path.join(data_path, "*.zip"))
    all_infectiousnesses = []
    for data_path in tqdm(data_paths):
        all_infectiousnesses.append(extract_infectiousnesses(data_path))
    return np.concatenate(all_infectiousnesses, axis=0)


if __name__ == "__main__":
    pass
