import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig
from scipy.io import loadmat

# Load the .mat file
data = loadmat('C:/Users/HUAWEI/PycharmProjects/data_driven_feature/Project2/CYLINDER_ALL.mat')

# Extract 'VORTALL' matrix from the loaded data
VORTALL = data['VORTALL']

# Prepare the data for DMD
X = VORTALL[:, :-1]
X2 = VORTALL[:, 1:]

# Compute SVD
U, S, Vh = svd(X, full_matrices=False)
V = Vh.T

# Truncate at 21 modes
r = 21
U_r = U[:, :r]
S_r = np.diag(S[:r])
V_r = V[:, :r]

# Compute Atilde
Atilde = U_r.T @ X2 @ V_r @ np.linalg.inv(S_r)

# Eigen decomposition
W, eigs = eig(Atilde)
Phi = X2 @ V_r @ np.linalg.inv(S_r) @ W

# Check the shape of Phi
print(f"Shape of Phi: {Phi.shape}")  # Add this line for debugging

# Reshape Phi if necessary
if Phi.ndim == 1:
    Phi = Phi.reshape(-1, 1)  # Reshape Phi into a column if it is 1D

# Grid dimensions
nx, ny = 199, 449

# Plotting function
def plotCylinder(VORT, nx, ny):
    plt.figure()
    VORT = np.clip(VORT, -5, 5)  # cutoff
    plt.imshow(VORT.reshape(nx, ny), extent=[-1, 8, -2, 2], cmap='coolwarm')
    plt.colorbar()
    plt.show()

# Plot DMD modes
for i in range(10, min(21, Phi.shape[1]), 2):
    plotCylinder(np.real(Phi[:, i]), nx, ny)
    plotCylinder(np.imag(Phi[:, i]), nx, ny)

# Plot DMD spectrum
plt.figure()
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--')  # unit circle
plt.scatter(eigs.real, eigs.imag, marker='o')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
