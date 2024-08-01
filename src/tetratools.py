import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_tetra_corner_coordinates(edgelength, flipped=True):
    flipper = -1 if flipped else 1
    sensor_1 = [0, 0, 0]
    sensor_2 = [edgelength, 0, 0]
    sensor_3 = [edgelength / 2, np.sqrt(3) * edgelength / 2, 0]
    sensor_4 = [
        edgelength / 2,
        np.sqrt(3) / 6 * edgelength,
        flipper * np.sqrt(6) / 3 * edgelength,
    ]
    return np.array([sensor_1, sensor_2, sensor_3, sensor_4])


def move_tetra_to_coordinate(corner_coordinates, new_origin, handle="corner"):
    if handle == "corner":
        shift_vector = new_origin - corner_coordinates[0]
        shifted_coordinates = corner_coordinates + shift_vector
    if handle == "centroid":
        centroid = np.mean(corner_coordinates, axis=0)
        shift_vector = new_origin - centroid
        shifted_coordinates = corner_coordinates + shift_vector
    return shifted_coordinates


def estimate_direction_vector(sensor_positions, amplitudes, signs):
    distances = 1 / np.sqrt(amplitudes)

    if len(np.unique(signs)) != 1:
        print("Signs are not unique")
        return np.array([np.nan, np.nan, np.nan])

    norm_distances = distances / np.sum(distances)
    weighted_centroid = np.dot(sensor_positions.T, norm_distances)
    true_centroid = np.mean(sensor_positions, axis=0)
    direction_vector = true_centroid - weighted_centroid
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
    return direction_vector_normalized


def plot_tetra_with_vector(
    ax, sensor_positions, direction_vector, dotsizes, facecolors, edgecolors
):
    # Plot the vertices
    for i in range(len(sensor_positions)):
        ax.scatter(
            sensor_positions[i, 0],
            sensor_positions[i, 1],
            sensor_positions[i, 2],
            marker="o",
            color=facecolors[i],
            edgecolor=edgecolors[i],
            s=dotsizes[i],
        )

    # Annotate sensor positions
    for i, pos in enumerate(sensor_positions):
        ax.text(pos[0], pos[1], pos[2], f"E{i+1}", color="k")

    # Plot the edges of the tetrahedron
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),  # Edges from the origin to other vertices
        (1, 2),
        (2, 3),
        (3, 1),  # Edges among the top three vertices
    ]

    for edge in edges:
        p1, p2 = sensor_positions[edge[0]], sensor_positions[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=edgecolors[0])

    # Plot the direction vector
    # origin is the center of the tetrahedron
    origin = np.mean(sensor_positions, axis=0)
    edge_length = np.linalg.norm(sensor_positions[0] - sensor_positions[1])

    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        direction_vector[0],
        direction_vector[1],
        direction_vector[2],
        color="grey",
        lw=1,
        length=edge_length * 4,
        normalize=True,
    )

    # Set plot labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add legend
    ax.legend()
