import numpy as np
import matplotlib.pyplot as plt


def plotCylinder(VORT, nx, ny):
    plt.figure()
    VORT = np.clip(VORT, -5, 5)  # cutoff at [-5, 5]

    # Plot vorticity field
    plt.imshow(VORT.reshape(nx, ny), cmap='coolwarm', origin='lower')
    plt.colorbar()

    # Add contour lines
    plt.contour(VORT.reshape(nx, ny), levels=np.arange(-5.5, 6, 0.5), colors='k', linestyles=['--', '-'])

    # Draw the cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 49 + 25 * np.sin(theta)
    y = 99 + 25 * np.cos(theta)
    plt.fill(x, y, color='gray')
    plt.plot(x, y, 'k-', linewidth=1.2)

    # Clean up axes
    plt.xticks([1, 50, 100, 150, 200, 250, 300, 350, 400, 449], ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
    plt.yticks([1, 50, 100, 150, 199], ['2', '1', '0', '-1', '-2'])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
