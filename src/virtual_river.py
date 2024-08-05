import numpy as np
from tetratools import make_tetra_corner_coordinates, move_tetra_to_coordinate
from efield import make_eel, make_monopoles, epotential
from rich.progress import track

space_bounds = (800, 800, 800)
meta_tetra_edgelen = 100
sub_tetra_edgelen = 25


def resample_line(x, y, z, num_samples):
    # Ensure the first and last points are kept
    num_samples = max(2, num_samples)  # Ensure at least two samples

    # Calculate the distance between consecutive points
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Determine the new sample positions
    new_positions = np.linspace(0, cumulative_distances[-1], num_samples)

    # Interpolate the new x, y, z coordinates
    new_x = np.interp(new_positions, cumulative_distances, x)
    new_y = np.interp(new_positions, cumulative_distances, y)
    new_z = np.interp(new_positions, cumulative_distances, z)

    return new_x, new_y, new_z


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
    sub_tetras = np.array(sub_tetras)

    ## Eel
    size = np.random.uniform(80, 240)
    x, y, z = make_eel(size, 1000)
    # Randomly move eel
    x = x + np.random.uniform(-space_dims[0] / 2, space_dims[0] / 2)
    y = y + np.random.uniform(-space_dims[1] / 2, space_dims[1] / 2)

    # random flip eel
    flipper = np.random.choice([-1, 1])
    x = x * flipper

    # random rotate eel
    angle = np.random.uniform(0, 2 * np.pi)
    x, y = (
        x * np.cos(angle) - y * np.sin(angle),
        x * np.sin(angle) + y * np.cos(angle),
    )

    # Random depth of eel
    eel_depth = np.random.uniform(-100, 100)
    z = z + eel_depth

    poles = make_monopoles(x, y, z, size, 10)

    pots = []
    for sub_tetra in sub_tetras:
        tetrapod = []
        for point in sub_tetra:
            point = point.reshape(1, 3)
            pot = epotential(point, poles)
            tetrapod.append(pot)

        tetrapod = np.ravel(tetrapod)
        pots.append(tetrapod)

    pots = np.array(pots)
    amps = np.abs(pots)

    # mz = 0.65
    # sqpot = [squareroot_transform(pot / 200, mz) for pot in pots]

    # Compute direction vectors
    signs = np.sign(pots)

    """
    direction_vectors = [
        estimate_direction_vector(ct, sa, sgns)
        for ct, sa, sgns in zip(sub_tetras, amps, signs)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    markersizes = amps / np.max(amps) * 100
    for points, size, sign, dirvec in zip(
        sub_tetras, markersizes, signs, direction_vectors
    ):
        color = ["tab:red" if s > 0 else "tab:blue" for s in sign]
        plot_tetra_with_vector(
            ax,
            points,
            dirvec,
            size,
            color,
            ["k", "k", "k", "k"],
        )

    ax.scatter(help_tetra[:, 0], help_tetra[:, 1], help_tetra[:, 2], color="grey")

    ax.plot(x, y, z, color="grey", lw=2)

    # Plot zero lines for all dimensions
    ax.plot([0, 0], [-space_dims[1], space_dims[1]], [0, 0], color="black", lw=0.5)
    ax.plot([-space_dims[0], space_dims[0]], [0, 0], [0, 0], color="black", lw=0.5)
    ax.plot([0, 0], [0, 0], [-space_dims[2], space_dims[2]], color="black", lw=0.5)

    ax.set_xlim(-space_dims[0], space_dims[0])
    ax.set_ylim(-space_dims[1], space_dims[1])
    ax.set_zlim(-space_dims[2], space_dims[2])
    # Top down view
    ax.view_init(elev=90, azim=0)
    ax.set_aspect("equal")
    plt.show()
    """
    return x, y, z, size, amps, signs, sub_tetras


def main():
    data = dict(
        x=[],
        y=[],
        z=[],
        size=[],
        amps=[],
        signs=[],
        sub_tetras=[],
    )
    for i in track(range(10000), description="Generating virtual rivers"):
        x, y, z, size, amps, signs, sub_tetras = virtual_river(space_bounds)
        x, y, z = resample_line(x, y, z, 5)
        data["x"].append(x)
        data["y"].append(y)
        data["z"].append(z)
        data["size"].append(size)
        data["amps"].append(amps)
        data["signs"].append(signs)
        data["sub_tetras"].append(sub_tetras)

    # Turn into np arrays
    for key in data.keys():
        data[key] = np.array(data[key])

    # Save data
    np.savez("simulations.npz", **data)


if __name__ == "__main__":
    main()
