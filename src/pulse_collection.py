import matplotlib.pyplot as plt
from audioio import load_audio
from scipy.signal import find_peaks, savgol_filter
import numpy as np
from style import set_light_style
from pathlib import Path
from rich.progress import track
import seaborn as sns
from tetratools import (
    make_tetra_corner_coordinates,
    move_tetra_to_coordinate,
    estimate_direction_vector,
    plot_tetra_with_vector,
)

set_light_style()
cmap = np.array(sns.color_palette("tab10"))
cmap = cmap[np.array([1, 2, 4, 6])]


def collect_pulses(files):
    file_indices = np.arange(len(files))

    for file_idx, file in track(
        zip(file_indices, files),
        description="Loading files",
        total=len(files),
    ):
        try:
            data, rate = load_audio(str(file))
        except EOFError:
            continue

        time = np.arange(len(data)) / rate
        time += file_idx * time[-1]

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

        # fig, ax = plt.subplots()
        # ax.plot(data)
        # ax.plot(peak_groups, data[peak_groups, 0], "ro")
        # plt.show()
        # continue

        # Extract the snippets around the peaks
        peak_window = 5 * peak_window
        all_snippets = []
        all_snippet_amps = []
        all_snippet_signs = []
        all_snippet_indices = []
        all_snippet_times = []
        for i, p in enumerate(peak_groups):
            indexer = np.arange(p - peak_window // 2, p + peak_window // 2)

            snippet = data[indexer, :]
            snippet_signs = [
                1 if s[np.argmax(np.abs(s))] > 0 else -1 for s in snippet.T
            ]

            snippet_signs = np.array(snippet_signs)
            snippet_amp = np.max(np.abs(snippet), axis=0)

            # Get groups of four amplitudes
            snippet = snippet.T
            snippet = snippet.reshape(4, 4, snippet.shape[-1])
            snippet_amp = snippet_amp.reshape(-1, 4)
            snippet_signs = snippet_signs.reshape(-1, 4)

            print(snippet.shape)
            print(snippet_amp.shape)
            print(snippet_signs.shape)

            # Resort electrodes in each group
            indexer = [3, 1, 0, 2]
            snippet = snippet[:, indexer, :]
            snippet_amp = snippet_amp[:, indexer]
            snippet_signs = snippet_signs[:, indexer]

            # Resort the groups the same way
            snippet = snippet[indexer]
            snippet_amp = snippet_amp[indexer]
            snippet_signs = snippet_signs[indexer]

            snippet = snippet.reshape(16, snippet.shape[-1])
            snippet_amp = snippet_amp.reshape(-1)
            snippet_signs = snippet_signs.reshape(-1)
            snippet_time = time[p]

            print(snippet.shape)
            print(snippet_amp.shape)
            print(snippet_signs.shape)

            all_snippets.append(snippet)
            all_snippet_amps.append(snippet_amp)
            all_snippet_signs.append(snippet_signs)
            all_snippet_indices.append(p)
            all_snippet_times.append(snippet_time)

        all_snippets = np.array(all_snippets)
        all_snippet_amps = np.array(all_snippet_amps)
        all_snippet_signs = np.array(all_snippet_signs)
        all_snippet_indices = np.array(all_snippet_indices)
        all_snippet_times = np.array(all_snippet_times)

        np.savez(
            f"../data/processed/pulses_{file_idx}.npz",
            all_snippets=all_snippets,
            all_snippet_amps=all_snippet_amps,
            all_snippet_signs=all_snippet_signs,
            all_snippet_indices=all_snippet_indices,
            all_snippet_times=all_snippet_times,
            data_file=file,
        )


def main():
    # lab()
    files = Path("../data/morning").glob("*.wav")
    files = list(files)
    collect_pulses(files)


if __name__ == "__main__":
    main()
