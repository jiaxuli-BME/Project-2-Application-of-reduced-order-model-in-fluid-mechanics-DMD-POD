import numpy as np


def loadIBPM(fname, nx, ny):
    with open(fname) as file:
        # Skip the first 6 lines of text
        for _ in range(6):
            file.readline()

        # Load the rest of the file
        data = np.loadtxt(file)

    # Reshape the data
    X = data[0].reshape(nx, ny).T
    Y = data[1].reshape(nx, ny).T
    U = data[2].reshape(nx, ny).T
    V = data[3].reshape(nx, ny).T
    VORT = data[4].reshape(nx, ny).T

    return X, Y, U, V, VORT
