import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.io import loadmat

# Load the data from a MATLAB .mat file
data = loadmat('C:/Users/HUAWEI/PycharmProjects/data_driven_feature/Project2/CYLINDER_ALL.mat')
VORTALL = data['VORTALL']  # Access the 'VORTALL' field

# Define grid dimensions (adjust as necessary)
nx, ny = 199, 449

# Prepare the data matrix and augment with mirror images to enforce symmetry/anti-symmetry
X = VORTALL
Y = np.hstack([X, np.zeros_like(X)])  # Initialize Y with extra space for mirrored data

for k in range(X.shape[1]):
    xflip = np.flipud(X[:, k].reshape(nx, ny)).reshape(nx * ny, 1)
    Y[:, k + X.shape[1]] = -xflip.flatten()  # Append the negative flipped version

# Compute the mean of the augmented data and subtract it from the original
VORTavg = np.mean(Y, axis=1)

# Plot function to visualize the cylinder wake
def plotCylinder(VORT, nx, ny, title=''):
    plt.figure()
    VORT = np.clip(VORT, -5, 5)  # Clip the vorticity values between -5 and 5
    plt.imshow(VORT.reshape(nx, ny), extent=[-1, 8, -2, 2], cmap='coolwarm')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Plot the average vorticity field (wake)
plotCylinder(VORTavg, nx, ny, title='Average Wake')

# Perform Proper Orthogonal Decomposition (POD)
PSI, S, Vh = svd(Y - VORTavg[:, None], full_matrices=False)

# Plot the singular values (which represent the energy content of each mode)
plt.figure()
plt.semilogy(S / np.sum(S), 'o-')  # Normalized singular values
plt.xlim([0, 50])
plt.title('Singular Values (POD Modes)')
plt.show()

# Plot the first four POD modes
for k in range(4):
    plotCylinder(PSI[:, k], nx, ny, title=f'POD Mode {k + 1}')
