import numpy as np

# Grid dimensions
nx, ny = 199, 449

# Create space for 150 snapshots
VORTALL = np.zeros((nx * ny, 150))

# Load the data for 150 snapshots
for count in range(150):
    num = (count + 1) * 10
    fname = f'ibpm{num:05d}.plt'
    X, Y, U, V, VORT = loadIBPM(fname, nx, ny)
    VORTALL[:, count] = VORT.flatten()

# Save the VORTALL array to file
np.save('CYLINDER_ALL.npy', {'VORTALL': VORTALL})
