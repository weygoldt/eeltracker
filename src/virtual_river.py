import numpy as np
import matplotlib.pyplot as plt
from tetratools import make_tetra_corner_coordinates, move_tetra_to_coordinate

space_bounds = (400, 400, 400)
meta_tetra_edgelen = 100
sub_tetra_edgelen = 25


def simulate_eel(length, nbends, nsamples=1000):
    """
    Simulate an eel by a simple line. Bends are simulated
    by circles with a given radius.
    """
    x = np.linspace(0, length, length * 100)

    # create a circle to curve the body of the eel
    theta = np.linspace(0, 2 * np.pi, 10000)
    min_radius = length / nbends / 4
    max_radius = length / nbends * 2
    radius = np.random.uniform(min_radius, max_radius)
    x_circle = radius * np.cos(theta) + length / 2
    y_circle = radius * np.sin(theta)

    # extract eel as a part of the circles outline with the correct length
    circumference = 2 * np.pi * radius
    nb_points = int(length / circumference * 10000)
    x_eel = x_circle[:nb_points]
    y_eel = y_circle[:nb_points]
    x_eel = x_eel.copy()
    y_eel = y_eel.copy()

    # Get the tangent of the circle at the start and end point
    tangent_start = np.array([x_eel[1] - x_eel[0], y_eel[1] - y_eel[0]])
    tangent_end = np.array([x_eel[-1] - x_eel[-2], y_eel[-1] - y_eel[-2]])

    # plt.plot(x_circle, y_circle)
    # plt.plot(x_eel, y_eel)
    # plt.scatter(x_eel[0], y_eel[0], color="red")
    # plt.scatter(x_eel[-1], y_eel[-1], color="green")
    # plt.quiver(x_eel[0], y_eel[0], tangent_start[0], tangent_start[1], color="red")
    # plt.quiver(x_eel[-1], y_eel[-1], tangent_end[0], tangent_end[1], color="green")
    # plt.axis("equal")
    # plt.show()
    #
    # Turn the eel so the end tangent is parallel to the x-axis
    angle = np.arctan2(tangent_start[1], tangent_start[0])
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    x_eel, y_eel = np.dot(rotation_matrix, np.array([x_eel, y_eel]))
    tangent_start = np.dot(rotation_matrix, tangent_start)
    tangent_end = np.dot(rotation_matrix, tangent_end)

    # Move first point to origin
    x_eel = x_eel - np.mean(x_eel)
    y_eel = y_eel - y_eel[0]

    # Interpolate to get nsamples points
    x_eel = np.interp(np.linspace(0, 1, nsamples), np.linspace(0, 1, len(x_eel)), x_eel)
    y_eel = np.interp(np.linspace(0, 1, nsamples), np.linspace(0, 1, len(y_eel)), y_eel)

    x_eel = x_eel[::-1]
    y_eel = y_eel[::-1]
    print(x_eel[0], y_eel[0])

    # plt.plot(x_eel, y_eel)
    # plt.scatter(x_eel[0], y_eel[0], color="red")
    # plt.scatter(x_eel[-1], y_eel[-1], color="green")
    # plt.axhline(0, color="black", lw=0.5)
    # plt.axvline(0, color="black", lw=0.5)
    # plt.axis("equal")
    # plt.show()
    # exit()

    return x_eel, y_eel


def virtual_river(space_dims):
    x = np.linspace(-space_dims[0], space_dims[0], space_dims[0] * 2)
    y = np.linspace(-space_dims[1], space_dims[1], space_dims[1] * 2)
    z = np.linspace(-space_dims[2], space_dims[2], space_dims[2] * 2)
    xx, yy = np.meshgrid(x, y)

    ## Tetrahedra
    construct_tetra = make_tetra_corner_coordinates(meta_tetra_edgelen)
    sub_tetras = [make_tetra_corner_coordinates(sub_tetra_edgelen) for _ in range(4)]
    sub_tetras = [
        move_tetra_to_coordinate(sub_tetra, construct_corner)
        for sub_tetra, construct_corner in zip(sub_tetras, construct_tetra)
    ]
    help_tetra = np.array([np.mean(sub_tetra, axis=0) for sub_tetra in sub_tetras])
    help_tetra = move_tetra_to_coordinate(help_tetra, (0, 0, 0), handle="centroid")
    sub_tetras = [
        move_tetra_to_coordinate(sub_tetra, help_corner, handle="centroid")
        for sub_tetra, help_corner in zip(sub_tetras, help_tetra)
    ]

    ## Eel
    xeel, yeel = simulate_eel(200, 1, 10)
    eel_len = np.sqrt(np.diff(xeel) ** 2 + np.diff(yeel) ** 2)
    print(f"Length of eel: {np.sum(eel_len)}")

    # random flip eel
    flipper = np.random.choice([-1, 1])
    xeel = xeel * flipper

    # random rotate eel
    angle = np.random.uniform(0, 2 * np.pi)
    xeel, yeel = (
        xeel * np.cos(angle) - yeel * np.sin(angle),
        xeel * np.sin(angle) + yeel * np.cos(angle),
    )

    # Randomly move eel
    xeel = xeel + np.random.uniform(0, space_dims[0])
    yeel = yeel + np.random.uniform(0, space_dims[1])

    # Random depth of eel
    eel_depth = np.random.uniform(-space_dims[2], space_dims[2])
    z_eel = np.zeros(len(xeel)) + eel_depth

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for sub_tetra in sub_tetras:
        ax.scatter(sub_tetra[:, 0], sub_tetra[:, 1], sub_tetra[:, 2], color="blue")

    ax.scatter(help_tetra[:, 0], help_tetra[:, 1], help_tetra[:, 2], color="red")

    ax.plot(xeel, yeel, z_eel, color="grey", lw=2)

    # Plot zero lines for all dimensions
    ax.plot([0, 0], [-space_dims[1], space_dims[1]], [0, 0], color="black", lw=0.5)
    ax.plot([-space_dims[0], space_dims[0]], [0, 0], [0, 0], color="black", lw=0.5)
    ax.plot([0, 0], [0, 0], [-space_dims[2], space_dims[2]], color="black", lw=0.5)

    ax.set_xlim(-space_dims[0], space_dims[0])
    ax.set_ylim(-space_dims[1], space_dims[1])
    ax.set_zlim(-space_dims[2], space_dims[2])
    ax.set_aspect("equal")
    plt.show()
    return x, y, z


def plot_virtual_river(x, y, z):
    pass


def main():
    x, y, z = virtual_river(space_bounds)


if __name__ == "__main__":
    main()
