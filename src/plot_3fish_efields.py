import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

from thunderfish.fakefish import wavefish_eods
from thunderfish.efield import (
    efish_monopoles,
    epotential_meshgrid,
    squareroot_transform,
)
from thunderfish.fishshapes import plot_fish
from thunderlab.powerspectrum import psd, decibel
import seaborn as sns


def plot_threewavefish(ax):
    aspect_ratio = 16 / 9
    maxx = 60.0
    maxy = maxx / aspect_ratio
    x = np.linspace(-maxx, maxx, 1000)
    y = np.linspace(-maxy, maxy, 1000)
    xx, yy = np.meshgrid(x, y)

    fish1 = (("Alepto", "top"), (-12, -12), (1, 0.7), 18.0, 10)
    fish2 = (("Alepto", "top"), (14, 2), (0.6, 1), 20.0, -15)
    fish3 = (("Alepto", "top"), (-8, 10), (1, -0.4), 16.0, -12)

    poles1 = efish_monopoles(*fish1[1:])
    poles2 = efish_monopoles(*fish2[1:])
    poles3 = efish_monopoles(*fish3[1:])

    pot = epotential_meshgrid(xx, yy, None, poles1, poles2, poles3)
    mz = 0.65
    zz = squareroot_transform(pot / 200, mz)
    levels = np.linspace(-mz, mz, 16)

    cmap = sns.color_palette("mako", as_cmap=True)

    ax.contourf(
        x,
        y,
        -zz,
        levels,
        cmap=cmap,
        zorder=-999,
        alpha=0.9,
        antialiased=True,
    )
    ax.contour(
        x,
        y,
        -zz,
        levels,
        cmap=cmap,
        linewidths=0.5,
        zorder=-1,
        alpha=1,
        antialiased=True,
    )
    # ax.imshow(-zz, extent=[-maxx, maxx, -maxy, maxy], cmap=cmap, origin="lower")

    bodykwargs = dict(lw=1, edgecolor="white", facecolor="white")
    bodykwargs = dict(lw=1, edgecolor="k", facecolor="k")
    finkwargs = dict(lw=1, edgecolor="white", facecolor="white")
    finkwargs = dict(lw=1, edgecolor="k", facecolor="k")

    plot_fish(ax, *fish1, bodykwargs=bodykwargs, finkwargs=finkwargs)
    plot_fish(ax, *fish2, bodykwargs=bodykwargs, finkwargs=finkwargs)
    plot_fish(ax, *fish3, bodykwargs=bodykwargs, finkwargs=finkwargs)

    # plot_fish(ax, ("Alepto", "top"))
    # ax.xscalebar(0.99, 0.06, 5, 'cm', ha='right')


if __name__ == "__main__":
    # plot_style()
    cm = 1 / 2.54
    figsize = (3 * 16 * cm, 3 * 9 * cm)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    plot_threewavefish(ax)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.margins(0)
    fmts = ["svg", "png", "pdf"]
    for fmt in fmts:
        plt.savefig(
            f"reports/figures/plot_3fish_efields.{fmt}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()
