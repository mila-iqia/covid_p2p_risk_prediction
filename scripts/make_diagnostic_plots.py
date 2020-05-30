import os
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

sns.set()


from ctt.data_loading.loader import ContactDataset


def dump_plots(data_path, plot_path):
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

    def get_plot_path(key):
        os.makedirs(os.path.join(plot_path, key), exist_ok=True)
        return os.path.join(
            plot_path, key, os.path.basename(data_path.replace(".zip", ".png"))
        )

    plt.figure()
    plt.hist(filter_thresh(samples["infectiousness_history"][:, :, 0]), bins=20)
    plt.savefig(get_plot_path("inf_hist"))

    valid_encounter_messages = [
        msg for msg in samples["encounter_message"] if msg.numel() > 0
    ]

    plt.figure()
    plt.title("Number of Elements in the Message")
    plt.hist([msg.numel() for msg in valid_encounter_messages], bins=30, log=True)
    plt.savefig(get_plot_path("msg_numel"))

    plt.figure()
    plt.title("Mean Message")
    plt.hist(
        ([msg.mean().item() for msg in valid_encounter_messages]), bins=30, log=True
    )
    plt.savefig(get_plot_path("msg_mean"))

    plt.figure()
    plt.title("Max Message")
    plt.hist([msg.max().item() for msg in valid_encounter_messages], bins=30, log=True)
    plt.savefig(get_plot_path("msg_max"))

    plt.figure()
    plt.title("Min Message")
    plt.hist([msg.min().item() for msg in valid_encounter_messages], bins=30, log=True)
    plt.savefig(get_plot_path("msg_min"))

    infectiousness_and_max_message = [
        (
            samples["infectiousness_history"][idx, :, 0].max(),
            samples["encounter_message"][idx].max().item(),
        )
        for idx in range(len(samples["encounter_message"]))
        if samples["encounter_message"][idx].numel() > 0
    ]

    infectiousness, max_message = zip(*infectiousness_and_max_message)
    infectiousness, max_message = np.asarray(infectiousness), np.asarray(max_message)

    plt.figure()
    plt.scatter(infectiousness, max_message, alpha=0.1)
    plt.xlabel("Infectiousness")
    plt.ylabel("Max Message")
    plt.savefig("inf_vs_max_msg")

    contagions_and_messages = [
        (
            samples["encounter_is_contagion"][idx][:, 0],
            samples["encounter_message"][idx][:, 0],
        )
        for idx in range(len(samples["encounter_is_contagion"]))
        if samples["encounter_is_contagion"][idx].numel() > 0
    ]

    contagions, messages = zip(*contagions_and_messages)
    contagions, messages = torch.cat(contagions), torch.cat(messages)

    plt.figure()
    plt.hist(
        messages[contagions == 1],
        log=True,
        bins=20,
        alpha=0.5,
        label="Messages when Contagion",
    )
    plt.hist(
        messages[contagions == 0],
        log=True,
        bins=20,
        alpha=0.5,
        label="Messages when not Contagion",
    )
    plt.legend()
    plt.savefig("msg_at_contagion")


if __name__ == '__main__':
    import argparse
    import glob

    parsey = argparse.ArgumentParser()
    parsey.add_argument("--data-path", type=str)
    parsey.add_argument("--plot-path", type=str)
    args = parsey.parse_args()

    data_paths = glob.glob(os.path.join(args.data_path, "*.zip"))
    for data_path in tqdm(data_paths):
        dump_plots(data_path, args.plot_path)
