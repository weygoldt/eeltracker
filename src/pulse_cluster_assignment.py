"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
from audioio import load_audio
from scipy.signal import find_peaks, savgol_filter
from scipy.signal.windows import tukey
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from style import set_light_style
from pathlib import Path
from rich.progress import track
from umap import UMAP

set_light_style()
overwrite = False


def collect_snippets(files):
    all_snippets = []
    file_indices = np.arange(len(files))
    for file_idx, file in track(
        zip(file_indices, files), description="Loading files", total=len(files)
    ):
        data, rate = load_audio(str(file))
        time = np.arange(0, len(data)) / rate
        peak_window = int(np.round(0.001 * rate))
        min_channels_with_peaks = 8

        data_smooth = savgol_filter(data.T, 51, 3)
        data_smooth = data_smooth.T
        data = savgol_filter(data.T, 21, 3)
        data = data.T

        peaks = [
            find_peaks(np.abs(data_smooth), prominence=0.002)[0]
            for data_smooth in data_smooth.T
        ]

        # Group peaks across channels that are close in time
        peak_groups = []
        for c in range(len(peaks)):
            for p in peaks[c]:
                if len(peak_groups) == 0:
                    peak_groups.append([p])
                else:
                    for group in peak_groups:
                        if np.abs(p - group[-1]) < peak_window:
                            group.append(p)
                            break
                    else:
                        peak_groups.append([p])

        # Filter out groups with less than min_channels_with_peaks channels
        peak_groups = [
            group for group in peak_groups if len(group) >= min_channels_with_peaks
        ]

        # Get the center of each group
        peak_groups = [int(np.round(np.mean(group))) for group in peak_groups]

        # Extract the snippets around the peaks
        peak_window = 10 * peak_window
        snippets = np.zeros((len(peak_groups), peak_window))
        for i, p in enumerate(peak_groups):
            indexer = np.arange(p - peak_window // 2, p + peak_window // 2)
            snippet = np.abs(data[indexer, :])
            snippet = np.mean(snippet, axis=1)

            start_baseline = np.mean(snippet[: peak_window // 4])
            snippet -= start_baseline
            snippet = (snippet + np.min(snippet)) / (np.max(snippet) - np.min(snippet))

            if abs(start_baseline) > 0.05:
                continue

            if np.min(snippet) < -0.05:
                continue

            # check if new center is not approx the middle of the snippet
            new_center = np.argmax(snippet)
            if new_center not in range(peak_window // 2 - 20, peak_window // 2 + 20):
                continue
            snippet = np.roll(snippet, peak_window // 2 - new_center)

            # add tukey window to smooth the roll
            snippet *= tukey(len(snippet), 0.5)

            peak_snippet = np.zeros_like(snippet)
            peak_snippet[snippet > 0.2] = snippet[snippet > 0.2]
            peaks = find_peaks(peak_snippet)[0]
            if len(peaks) != 1:
                continue

            snippets[i, :] = snippet

        all_snippets.append(snippets)

    snippets = np.vstack(all_snippets)

    # remove all zero snippets
    zero_idx = np.sum(snippets, axis=1) == 0
    snippets = snippets[~zero_idx]
    # all_snippet_file_idx = np.array(all_snippet_file_idx)[~zero_idx]
    # all_snippet_peak_idx = np.array(all_snippet_peak_idx)[~zero_idx]
    return snippets


def main():
    files = Path("data/morning").glob("*.wav")
    files = list(files)[:30]

    if Path("eel_snippets.npy").exists() and not overwrite:
        snippets = np.load("eel_snippets.npy")
    else:
        snippets = collect_snippets(files)
        np.save("eel_snippets.npy", snippets)

    scaler = StandardScaler()
    snippets_scaled = scaler.fit_transform(snippets)
    print("Doing UMAP")
    umap = UMAP(n_components=3, n_neighbors=10, min_dist=0.1, metric="euclidean")

    decomp_snippets = umap.fit_transform(snippets_scaled)

    hdb = HDBSCAN(min_cluster_size=50, min_samples=10)
    y = hdb.fit_predict(decomp_snippets)
    unique_labels = np.unique(y)

    fig, ax = plt.subplots(1, 2)
    colors = plt.cm.viridis(np.linspace(0.5, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:
            color = "grey"
        else:
            color = colors[i]
            ax[1].plot(
                np.mean(snippets[y == label, :], axis=0), lw=2, c="k", zorder=10000
            )
        ax[0].plot(
            decomp_snippets[y == label, 0], decomp_snippets[y == label, 1], "o", c=color
        )
        ax[1].plot(snippets[y == label, :].T, lw=1, alpha=0.1, c=color)

    plt.show()

    exit()


if __name__ == "__main__":
    main()
