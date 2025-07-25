import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a grid of spherical coordinates (phi, theta)
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
phi_grid, theta_grid = np.meshgrid(phi, theta)

# Convert spherical to Cartesian using your formula
x = np.cos(theta_grid) * np.sin(phi_grid)
y = np.sin(theta_grid)
z = np.cos(theta_grid) * np.cos(phi_grid)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=5, cstride=5, alpha=0.3, color='cyan', edgecolor='gray')

# Plot axes
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Spherical to Cartesian Coordinate System')
ax.legend()

plt.tight_layout()
plt.show()
