import numpy as np
import matplotlib.pyplot as plt


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
    print(f"Before scaling: {length_estimate}")

    # Scale the line so that the lenght is 1
    x = x / length_estimate
    y = y / length_estimate

    length_estimate = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    print(f"After scaling: {length_estimate}")

    # now scale up so that line is eellen long
    x = x * eellen
    y = y * eellen
    length_estimate = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    print(f"After scaling to {eellen}: {length_estimate}")
    return x, y


if __name__ == "__main__":
    for i in range(100):
        length = np.random.uniform(100, 200)
        x, y = make_eel(length)
        plt.plot(x, y)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()
