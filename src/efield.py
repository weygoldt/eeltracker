import numpy as np
import matplotlib.pyplot as plt
from virtual_river import simulate_eel
import seaborn as sns
from thunderfish.efield import squareroot_transform
from style import transparent_fade_colormap


def simulate_eel(length, nbends, bend_angles, nsamples=1000):
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

    plt.plot(x_circle, y_circle)
    plt.plot(x_eel, y_eel)
    plt.scatter(x_eel[0], y_eel[0], color="red")
    plt.scatter(x_eel[-1], y_eel[-1], color="green")
    plt.quiver(x_eel[0], y_eel[0], tangent_start[0], tangent_start[1], color="red")
    plt.quiver(x_eel[-1], y_eel[-1], tangent_end[0], tangent_end[1], color="green")
    plt.axis("equal")
    plt.show()

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

    plt.plot(x_eel, y_eel)
    plt.scatter(x_eel[0], y_eel[0], color="red")
    plt.scatter(x_eel[-1], y_eel[-1], color="green")
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)
    plt.axis("equal")
    plt.show()

    return x_eel, y_eel


def efish_monopoles(pos=(0, 0, 0), direction=(1, 0, 0), size=200.0, nneg=10):
    n = int(10 * size)
    npos = n - nneg
    pos = np.asarray(pos)
    charges = np.ones(n)
    charges[:nneg] = -float(npos) / nneg
    xeel, yeel = np.array(simulate_eel(size, 1, n))
    zeel = np.zeros_like(xeel)
    poles = np.array([xeel, yeel, zeel]).T

    return poles, charges


def epotential(pos, *args):
    """Simulation of electric field potentials.

    Parameters
    ----------
    pos: 2D array of floats
        Each row contains the coordinates (2D or 3D)
        for which the potential should be computed.
    args: list of tuples
        Each tuple contains as the first argument the position of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    pot: 1D array of float
        The potential for each position in `pos`.
    """
    pos = np.asarray(pos)
    pot = np.zeros(len(pos))
    for poles, charges in args:
        for p, c in zip(poles, charges):
            r = pos - p
            rnorm = np.linalg.norm(r, axis=1)
            rnorm[np.abs(rnorm) < 1e-12] = 1.0e-12
            pot += c / rnorm
    return pot


def epotential_meshgrid(xx, yy, zz, *args):
    """Simulation of electric field potentials on a mesh grid.

    This is a simple wrapper for epotential().

    Parameters
    ----------
    xx: 2D array of floats
        Range of x coordinates as returned by numpy.meshgrid().
    yy: 2D array of floats
        Range of y coordinates as returned by numpy.meshgrid().
    zz: None or 2D array of floats
        z coordinates on the meshgrid defined by xx and yy.
        If provided, poles in args must be 3D.
        If None then treat it as a 2D problem with poles in args providing 2D coordinate.
    args: list of tuples
        Each tuple contains as the first argument the position (2D or 3D) of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    pot: 2D array of floats
        The potential for the mesh grid defined by xx and yy and evaluated
        at (xx, yy, zz).

    Example
    -------
    ```
    fig, ax = plt.subplots()
    maxx = 30.0
    maxy = 27.0
    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = object_monopoles((-6, 0), 1.0, -0.5, poles1, poles2)
    allpoles = (poles1, poles2, poles3)
    # potential:
    pot = epotential_meshgrid(xx, yy, None, *allpoles)
    thresh = 0.65
    zz = squareroot_transform(pot/200, thresh)
    levels = np.linspace(-thresh, thresh, 16)
    ax.contourf(x, y, -zz, levels, cmap='RdYlBu')
    ax.contour(x, y, -zz, levels, zorder=1, colors='#707070',
               linewidths=0.1, linestyles='solid')
    plt.show()
    ```
    """
    pos = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    pot = epotential(pos, *args)
    return pot.reshape(xx.shape)


aspect_ratio = 16 / 9
maxx = 400
maxy = maxx / aspect_ratio
x = np.linspace(-maxx, maxx, 50)
y = np.linspace(-maxy, maxy, 50)
z = np.linspace(-maxx, 0, 50)

xx, yy, zz = np.meshgrid(x, y, z)


size, nneg = 200, 10
fish = ((0, 0, 0), (0, 0), size, nneg)
poles = efish_monopoles(*fish)
pot = epotential_meshgrid(xx, yy, zz, poles)
mz = 0.65
sqpot = squareroot_transform(pot / 200, mz)

cmap = sns.color_palette("magma", as_cmap=True)
# cmap = transparent_fade_colormap(cmap, mode="lin")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xx, yy, zz, s=30, c=-sqpot, cmap=cmap, lw=0)
plt.show()

# Only plot slice, lower half of the z-axis
