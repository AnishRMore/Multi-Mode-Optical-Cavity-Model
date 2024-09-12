import numpy as np

# Define parameters
N = 3  # Number of modes
omega = np.array([1.0, 1.5, 2.0])  # Angular frequencies for each mode

# Define creation and annihilation operators
def creation_operator(i):
    ket = np.zeros(N)
    ket[i] = 1
    return np.sqrt(0.5) * (np.roll(ket, -1) + (-1)**i * ket)

def annihilation_operator(i):
    ket = np.zeros(N)
    ket[i] = 1
    return np.sqrt(0.5) * (np.roll(ket, 1) + (-1)**i * ket)

# Define Hamiltonian
def Hamiltonian():
    H = np.zeros((N, N))
    for i in range(N):
        H += omega[i] * (np.outer(creation_operator(i), annihilation_operator(i)) + 0.5 * np.eye(N))
    return H

# Calculate eigenstates and eigenvalues
eigenvalues, eigenstates = np.linalg.eigh(Hamiltonian())

# Print eigenvalues and eigenstates
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenstates:")
print(eigenstates)