import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from thunderfish.efield import squareroot_transform
from style import transparent_fade_colormap
from scipy.signal import resample


def cubic_bezier(t, p0, p1, p2, p3):
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def make_eel(eellen=200, npoints=1000):
    length = 1
    t = np.linspace(0, length, npoints)
    p0 = np.array([0, 0])
    p3 = np.array([length, 0])

    p1_xvals = np.linspace(0, length / 2, npoints)
    p2_xvals = np.linspace(length / 2, length, npoints)
    p1_yvals = np.linspace(-length / 2, length / 2, npoints)
    p2_yvals = np.linspace(-length / 2, length / 2, npoints)

    p1x = np.random.choice(p1_xvals)
    p1y = np.random.choice(p1_yvals)
    p2x = np.random.choice(p2_xvals)
    p2y = np.random.choice(p2_yvals)
    p1 = np.array([p1x, p1y])
    p2 = np.array([p2x, p2y])
    x, y = [], []
    for ti in t:
        xi, yi = cubic_bezier(ti, p0, p1, p2, p3)
        x.append(xi)
        y.append(yi)
    x = np.array(x)
    y = np.array(y)

    # Estimate length of the line
    length_estimate = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    # Scale the line so that the lenght is 1
    x = x / length_estimate
    y = y / length_estimate
    # now scale up so that line is eellen long
    x = x * eellen
    y = y * eellen
    length_estimate = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Shift the head to 0
    x = x - x[-1]
    y = y - y[-1]
    # Add z coordinates
    z = np.zeros_like(x)
    return x, y, z


def make_monopoles(x, y, z, size=200.0, nneg=10):
    n = int(10 * size)
    npos = n - nneg
    charges = np.ones(n)
    charges[:nneg] = -float(npos) / nneg
    x = resample(x, n)
    y = resample(y, n)
    z = resample(z, n)
    poles = np.array([x, y, z]).T
    return poles, charges


def epotential(pos, *args):
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
    pos = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    print(np.shape(pos))
    pot = epotential(pos, *args)
    return pot.reshape(xx.shape)


if __name__ == "__main__":
    aspect_ratio = 16 / 9
    maxx = 200
    maxy = maxx / aspect_ratio
    x = np.linspace(-maxx, maxx, 50)
    y = np.linspace(-maxy, maxy, 50)
    z = np.linspace(-maxx, -1, 50)
    xx, yy, zz = np.meshgrid(x, y, z)

    size = 200
    x, y, z = make_eel(size, 1000)
    x = x + 100
    poles = make_monopoles(x, y, z, size, 10)
    pot = epotential_meshgrid(xx, yy, zz, poles)
    mz = 0.65
    sqpot = squareroot_transform(pot / 200, mz)

    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap = transparent_fade_colormap(cmap, mode="abs")
    cmap = cmap.reversed()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xx, yy, zz, s=30, c=-sqpot, cmap=cmap, lw=0)
    ax.plot(x, y, z, c="black", lw=2, zorder=1000)
    plt.show()
