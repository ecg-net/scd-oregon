from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from utils.data import ECGDataset
from utils.models import EffNet
from utils.training_models import BinaryClassificationModel


# LIME-for-ECG code originally written by J Weston Hughes

ecg_len = 1250


def dist_kernel(x, x_mod, p_keep=0.7):
    # needs batch
    m, s = np.mean(x, axis=(1, 2)), np.std(x, axis=(1, 2))
    m = np.expand_dims(np.expand_dims(m, -1), -1)
    s = np.expand_dims(np.expand_dims(s, -1), -1)
    x = (x - m) / s
    x_mod = (x_mod - m) / s
    l2 = np.sum((x - x_mod) ** 2, axis=(1, 2))
    kernel = np.exp(-l2 / (ecg_len * 12 * p_keep * (1 - p_keep)))
    return kernel


def lime(
    x,
    cnn_model,
    cutoffs=None,
    n=10,
    target=1,
    step=50,
    smoothing=40,
    p_keep=0.7,
    return_coefs=False,
):
    # Divide input in time
    if cutoffs is None:
        cutoffs = np.arange(0, ecg_len + 1, step)
        if cutoffs[-1] != ecg_len:
            cutoffs = np.append(cutoffs, ecg_len)

    # Divide input accross leads
    num_segments = len(cutoffs) * 12

    # Make n "samples" with segments removed
    sample_maps = stats.bernoulli.rvs(p=p_keep, size=(n, num_segments))

    signals = np.zeros((n, ecg_len, 12))
    med = np.median(x, axis=0)

    for j in range(n):
        signals[j] = x
        sample_map = sample_maps[j]
        current = 0

        for k in range(len(cutoffs) - 1):
            for l in range(12):
                if not sample_map[current]:
                    signals[j, cutoffs[k] : cutoffs[k + 1], l] = med[l]

                current += 1
    # The linear models are weighted by distance from the original
    weights = dist_kernel(signals, np.array(x), p_keep)

    # Get our original model's prections on each input
    new_preds = batch_predict(cnn_model, signals)

    new_preds = np.concatenate([new_preds[j][:] for j in range(len(new_preds))])

    # We want to select `target` features to highlight, so we have to tune the L1 regularization
    alpha = 1.0
    coefs = 0

    # First find the magnitude of alpha
    while coefs < target:
        alpha /= 10
        lasso = sklearn.linear_model.Lasso(alpha=alpha)
        X = sample_maps * np.expand_dims(weights**0.5, -1)
        y = new_preds * weights**0.5
        lasso.fit(list(X), y)
        coefs = np.count_nonzero(lasso.coef_)
    saved_coefs = lasso.coef_
    bounds = [alpha, 10 * alpha]

    j = 0

    # Then use binary search to find an exact alpha value
    while coefs != target:
        alpha = (bounds[0] + bounds[1]) / 2
        lasso = sklearn.linear_model.Lasso(alpha=alpha)
        X = sample_maps * np.expand_dims(weights, -1)
        y = new_preds * weights
        lasso.fit(list(X), y)
        coefs = np.count_nonzero(lasso.coef_)
        if coefs < target:
            bounds = [bounds[0], alpha]
        else:
            bounds = [alpha, bounds[1]]
        j += 1
        if j > 10:
            break

    if coefs != 0:
        saved_coefs = lasso.coef_
    if return_coefs:
        return saved_coefs

    colors = np.zeros((12, ecg_len))
    j = 0

    for k in range(len(cutoffs) - 1):
        for i in range(12):
            colors[i, cutoffs[k] : cutoffs[k + 1]] = saved_coefs[i + k * 12]
    for i in range(12):
        colors[i] = np.convolve(
            colors[i],
            np.hanning(smoothing) / np.sum(np.hanning(smoothing)),
            mode="same",
        )
    return colors


import matplotlib

cdict = {
    "red": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "green": [[0.0, 1.0, 1.0], [0.25, 0.5, 0.5], [1.0, 0.0, 0.0]],
    "blue": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
}
cmap = matplotlib.colors.LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)


def batch_predict(model, images):
    images = np.transpose(images, (0, 2, 1))
    model.eval()
    batch = torch.Tensor(images)

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def plot_example(waveform, lime_colors=None, save_path=None):
    # Grid Parameters

    n_boxes_between = 4
    margin_boxes = n_boxes_between / 2
    n_boxes = 2 * margin_boxes + 11 * n_boxes_between

    # Titling
    fig = plt.figure(figsize=(25, n_boxes / 2), dpi=80, facecolor="w", edgecolor="k")

    # Grid
    plt.xlim(0, ecg_len)
    plt.xticks(np.arange(0, ecg_len + 1, 50), "")
    plt.yticks(
        ticks=np.arange(0, 12) * (-5) * 4,
        labels=[
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        fontsize=25,
    )
    plt.minorticks_on()

    # Plot the signal
    spacing = 5
    for i in range(12):
        plt.plot(
            waveform[:, i] - i * spacing * n_boxes_between,
            alpha=0.9,
            color=["tab:blue", "tab:brown"][i % 2],
        )

    do_lime = True
    if do_lime:
        # Plot LIME results
        if lime_colors is not None:
            # x bounds for every component
            scatter_x = np.array([j for j in range(100) for i in range(12)]) * 25
            colors = np.zeros((ecg_len, int(n_boxes * spacing)))
            for i, (s_x, l_c) in enumerate(zip(scatter_x, lime_colors)):
                if l_c == 0:
                    continue

                # y bounds are different for each compoenent, depending on signal values
                # s_y is the actual y coordinates
                s_y = waveform[s_x : s_x + 25, i % 12]
                s_y = (i % 12) * spacing * n_boxes_between
                s_y = 5 * margin_boxes - s_y

                # Fill in color in this area
                colors[
                    s_x : s_x + 25, (int(np.min(s_y)) - 5) : (int(np.max(s_y)) + 5)
                ] += (l_c * 1000)
            colors = colors.T
            # Apply a gaussian filter to soften the edges of the explaining box
            colors = (
                ndimage.gaussian_filter(colors[::2, ::2], sigma=(1, 10))
                .repeat(2, axis=0)
                .repeat(2, axis=1)
            )
            # Plot explanation over signal
            plt.imshow(
                colors,
                aspect="auto",
                cmap=cmap,
                extent=[
                    0,
                    ecg_len,
                    -spacing * (n_boxes - margin_boxes),
                    spacing * margin_boxes,
                ],
                alpha=0.6,
                clim=(0, np.max(colors)),
            )
    if save_path is not None:
        plt.savefig(save_path)

    return True


backbone = EffNet(output_neurons=1)
model = BinaryClassificationModel(backbone)

print(model.load_state_dict(torch.load("/path/to/weights")))

project = Path("/path/to/project/dir")
repo = Path("/path/to/repo/dir")

if not (repo / "lime").is_dir():
    (repo / "lime").mkdir()

data_path = project / "train_normed_npy"
manifest_path = repo / "manifest.csv"

test_set = pd.read_csv(manifest_path)
test_set = test_set[test_set["split"] == "ex_test"]
np.random.seed(42)
positives = test_set[test_set["case"] == 1].sample(20)
positives.to_csv(repo / "lime" / "positive_manifest.csv", index=False)
negatives = test_set[test_set["case"] == 0].sample(20)
negatives.to_csv(repo / "lime" / "negative_manifest.csv", index=False)

for polarity in ["positive", "negative"]:
    test_ds = ECGDataset(
        split="ex_test",
        data_path=data_path,
        manifest_path=repo / "lime" / f"{polarity}_manifest.csv",
        labels="case",
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )

    for fname, waveform, label in tqdm(test_dl):
        c = 0
        waveform = waveform.squeeze().T * 50

        print(waveform[:, 1].shape)
        for i in range(12):
            waveform[:, i] = waveform[:, i] - waveform[:, i].mean()

        lime_colors = lime(
            waveform,
            model,
            n=1000,
            target=10,
            step=25,
            p_keep=0.7,
            smoothing=200,
            return_coefs=True,
        )

        if not (repo / "lime" / polarity).is_dir():
            (repo / "lime" / polarity).mkdir()

        save_path = repo / "lime" / polarity / f"{fname[0]}.png"
        plot_example(waveform, lime_colors, save_path=save_path)
