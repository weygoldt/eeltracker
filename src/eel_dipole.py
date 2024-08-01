import numpy as np
import matplotlib.pyplot as plt
from thunderfish.efield import (
    efish_monopoles,
    epotential_meshgrid,
    squareroot_transform,
    epotential,
)

aspect_ratio = 16 / 9
maxx = 200
maxy = maxx / aspect_ratio
x = np.linspace(-maxx, maxx, 100)
y = np.linspace(-maxy, maxy, 100)
xx, yy = np.meshgrid(x, y)

size, nneg, bend = 200, 20, 0
fish = ((0, 0), (0, 0), size, bend, nneg)
poles = efish_monopoles(*fish)

fig, ax = plt.subplots()
pot = epotential_meshgrid(xx, yy, None, poles)
mz = 0.65
sqpot = squareroot_transform(pot / 200, mz)
ax.imshow(-sqpot, extent=[-maxx, maxx, -maxy, maxy], cmap="viridis", origin="lower")
ax.set_aspect("equal")
plt.show()

# Test in 3D
maxz = maxx / aspect_ratio
x = np.linspace(-maxx, maxx, 10)
y = np.linspace(-maxy, maxy, 10)
z = np.linspace(-maxz, maxz, 10)
xx, yy, zz = np.meshgrid(x, y, z)

pos = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
pot = epotential(pos, poles)
mz = 0.65
sqpot = squareroot_transform(pot, mz)

print(len(np.unique(sqpot)))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xx, yy, zz, s=10, c=-sqpot, cmap="viridis")
ax.set_zlim(-maxz, maxz)
ax.set_aspect("equal")
plt.show()
