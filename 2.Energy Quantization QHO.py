import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1.0  # Reduced Planck's constant
m = 1.0      # Particle mass
omega = 1.0  # Oscillator frequency

# Potential function for harmonic oscillator
def V(x):
    return 0.5 * m * omega**2 * x**2

# Function to solve Schrödinger equation using Finite Difference Method
def solve_schrodinger_eq(xmin, xmax, N):
    # Discretization parameters
    x = np.linspace(xmin, xmax, N)
    dx = (xmax-xmin)/N

    # Kinetic energy operator (second derivative)
    T = np.diag(-2.0*np.ones(N)) + np.diag(np.ones(N-1), k=1) + np.diag(np.ones(N-1), k=-1)
    T /= dx**2

    # Potential energy operator
    V_mat = np.diag(V(x))

    # Hamiltonian matrix
    H = -(h_bar**2 / (2.0 * m)) * T + V_mat

    # Solve eigenvalue problem
    energies, wavefunctions = np.linalg.eigh(H)

    # Normalize wavefunctions
    wavefunctions /= np.sqrt(dx)

    return energies, wavefunctions, x

# Parameters
xmin = -5.0
xmax = 5.0
N = 1000

# Solve Schrödinger equation
energies, wavefunctions, x = solve_schrodinger_eq(xmin, xmax, N)

# Plot potential
plt.plot(x, V(x), color='black', label='Harmonic Potential')

# Plot wavefunctions (first few)
for i in range(5):
    plt.plot(x, energies[i] + np.abs(wavefunctions[:, i])**2, label=f'$E_{i}$={energies[i]:.2f}')

plt.title("Quantum Harmonic Oscillator")
plt.xlabel("x")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()