import matplotlib.pyplot as plt
from audioio import load_audio
from scipy.signal import find_peaks, savgol_filter
import numpy as np
from style import cm, set_light_style
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
        data, rate = load_audio(str(file))
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
        peak_window = 5 * peak_window
        for i, p in enumerate(peak_groups):
            indexer = np.arange(p - peak_window // 2, p + peak_window // 2)

            snippet = data[indexer, :]
            snippet_signs = [
                1 if s[np.argmax(np.abs(s))] > 0 else -1 for s in snippet.T
            ]

            snippet_signs = np.array(snippet_signs)
            snippet_amp = np.max(np.abs(snippet), axis=0)

            # Get groups of four amplitudes
            snippet_amp = snippet_amp.reshape(-1, 4)
            snippet_sings = snippet_signs.reshape(-1, 4)

            meta_tetra_positions = make_tetra_corner_coordinates(1000)
            meta_tetra_positions[:, 2] += 1000
            meta_tetra_positions[:, 0] -= 1000 / 2
            meta_tetra_positions[:, 1] -= 1000 / 2

            corner_tetras = [make_tetra_corner_coordinates(250) for i in range(4)]

            positioned_corner_tetras = [
                move_tetra_to_coordinate(corner_tetra, meta_tetra_position)
                for corner_tetra, meta_tetra_position in zip(
                    corner_tetras, meta_tetra_positions
                )
            ]

            # Resort the electrode tetrahedrons
            indexer = [3, 1, 0, 2]
            positioned_corner_tetras = [positioned_corner_tetras[i] for i in indexer]

            # Resort each electrode in each sub tetrahedron
            for i in range(4):
                positioned_corner_tetras[i] = positioned_corner_tetras[i][indexer]
            positioned_corner_tetras = np.array(positioned_corner_tetras)

            print(positioned_corner_tetras)
            print(np.shape(positioned_corner_tetras))
            exit()
            direction_vectors = [
                estimate_direction_vector(ct, sa, sgns)
                for ct, sa, sgns in zip(
                    positioned_corner_tetras, snippet_amp, snippet_sings
                )
            ]

            fig = plt.figure(figsize=(16, 9), constrained_layout=True)
            ax1 = fig.add_subplot(1, 1, 1, projection="3d")
            # ax2 = fig.add_subplot(1, 2, 2)

            # Plot metatetra
            meta_corners = [np.mean(pct, axis=0) for pct in positioned_corner_tetras]
            meta_corners = np.array(meta_corners)
            plot_tetra_with_vector(
                ax1,
                meta_corners,
                np.array([np.nan, np.nan, np.nan]),
                np.array([0, 0, 0, 0]),
                facecolors=["k", "k", "k", "k"],
                edgecolors=["k", "k", "k", "k"],
            )

            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            colors = cmap
            dotsizes = snippet_amp / np.max(snippet_amp) * 100
            for i, (ct, dv, ds, sgns) in enumerate(
                zip(
                    positioned_corner_tetras, direction_vectors, dotsizes, snippet_sings
                )
            ):
                fcs = ["tab:red" if s > 0 else "tab:blue" for s in sgns]
                plot_tetra_with_vector(
                    ax1,
                    ct,
                    dv,
                    ds,
                    facecolors=fcs,
                    edgecolors=[colors[i]] * 4,
                )

            ax1.set_xlim(-1000, 1000)
            ax1.set_ylim(-1000, 1000)
            ax1.set_zlim(0, 1200)
            ax1.set_aspect("equal")

            # Only show 3d axis from above
            ax1.view_init(elev=90, azim=0)

            # Hide axis
            ax1.set_axis_off()

            # Hide grid
            ax1.grid(False)

            # for i, sa in enumerate(snippet_amp):
            #     ax2.plot(sa, label=f"E{i+1}", color=colors[i])
            # ax2.set_xticks(np.arange(0, len(snippet_amp), 1))
            # ax2.set_xticklabels([f"E{i+1}" for i in range(4)])
            # ax2.set_ylabel("Amplitude on electrode")

            plt.show()


def lab():
    meta_edge_length = 1000  # mm
    sub_edge_length = 250  # mm

    meta_tetra_positions = make_tetra_corner_coordinates(meta_edge_length)

    meta_tetra_positions[:, -1] += meta_edge_length

    corner_tetras = [make_tetra_corner_coordinates(sub_edge_length) for i in range(4)]

    positioned_corner_tetras = [
        move_tetra_to_coordinate(corner_tetra, meta_tetra_position)
        for corner_tetra, meta_tetra_position in zip(
            corner_tetras, meta_tetra_positions
        )
    ]

    center_meta_tetra = [np.mean(pct, axis=0) for pct in positioned_corner_tetras]
    center_meta_tetra = np.array(center_meta_tetra)
    test_pos = np.mean(center_meta_tetra, axis=0)

    test_positions = np.array(
        [
            [2000, 2000, 500],
            [0, 0, 0],
            [1000, 2000, 0],
            [-200, 1000, 1000],
        ]
    )

    for test_pos in test_positions:
        test_pos_distances = [
            np.linalg.norm(ct - test_pos, axis=1) for ct in positioned_corner_tetras
        ]

        # amplitude is proportional to 1/d^2
        snippet_amp = 1 / np.array(test_pos_distances) ** 2

        signs = np.array([1, 1, 1, 1])

        # Compute direction vectors
        direction_vectors = [
            estimate_direction_vector(ct, sa, sgns)
            for ct, sa, sgns in zip(positioned_corner_tetras, snippet_amp, signs)
        ]

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.scatter(test_pos[0], test_pos[1], test_pos[2], marker="o", color="k")

        plot_tetra_with_vector(
            ax1,
            center_meta_tetra,
            np.array([np.nan, np.nan, np.nan]),
            np.array([0, 0, 0, 0]),
            facecolors=["k", "k", "k", "k"],
            edgecolors=["k", "k", "k", "k"],
        )

        dotsizes = snippet_amp / np.max(snippet_amp) * 100
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, (ct, dv, ds) in enumerate(
            zip(positioned_corner_tetras, direction_vectors, dotsizes)
        ):
            plot_tetra_with_vector(
                ax1,
                ct,
                dv,
                ds,
                facecolors=["tab:red", "tab:blue", "tab:red", "tab:blue"],
                edgecolors=[colors[i]] * 4,
            )
            centroid = np.mean(ct, axis=0)
            ax1.scatter(
                centroid[0], centroid[1], centroid[2], marker="o", color=colors[i]
            )
            # plot line between centroid and test position
            ax1.plot(
                [centroid[0], test_pos[0]],
                [centroid[1], test_pos[1]],
                [centroid[2], test_pos[2]],
                color=colors[i],
                alpha=0.2,
            )

        ax1.set_zlim(0, 1200)
        ax1.set_aspect("equal")
        ax1.view_init(elev=90, azim=0)

        for i, sa in enumerate(snippet_amp):
            ax2.plot(sa, label=f"E{i+1}", color=colors[i])
        ax2.set_xticks(np.arange(0, len(snippet_amp), 1))
        ax2.set_xticklabels([f"E{i+1}" for i in range(4)])
        ax2.set_ylabel("Amplitude on electrode")

        plt.show()


def main():
    # lab()
    files = Path("data/morning").glob("*.wav")
    files = list(files)[15:20]
    collect_pulses(files)


if __name__ == "__main__":
    main()
