import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
hbar = 1.0  # Reduced Planck constant
m = 1.0     # Mass of the particle
omega = 1.0 # Angular frequency of the oscillator
num_points = 1000

# Function to calculate the wavefunction for position. This basically gives the discrete wavefunction solutions
def wavefunction_position(n, x):
    prefactor = 1.0 / np.sqrt(2**n * math.factorial(n)) * (m * omega / (np.pi * hbar)) ** 0.25
    return prefactor * np.exp(-m * omega * x**2 / (2 * hbar)) * np.polynomial.hermite.hermval(np.sqrt(m * omega / hbar) * x, np.eye(n + 1)[-1])
#hermite.hermval returns the solutions of the wavefunctions. More efficient than LU Decomposition.

# Function to calculate the probability distribution for position
def probability_distribution_position(n, x_values):
    psi = wavefunction_position(n, x_values)
    return np.abs(psi)**2

# Function to calculate the probability distribution for momentum
def probability_distribution_momentum(n, x_values):
    psi_p = np.fft.fftshift(np.fft.fft(wavefunction_position(n, x_values)))
    return np.abs(psi_p)**2

# Define range of x and p values
x_values = np.linspace(-5, 5, num_points)
p_values = np.linspace(-5, 5, num_points)

# Calculate and plot probability distribution for position
plt.figure(figsize=(10, 6))
for n in range(4):  # Plot for the first 4 energy levels
    plt.plot(x_values, probability_distribution_position(n, x_values), label=f'n={n}')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Probability Distribution for Position')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot probability distribution for momentum
plt.figure(figsize=(10, 6))
for n in range(4):  # Plot for the first 4 energy levels
    plt.plot(p_values, probability_distribution_momentum(n, p_values), label=f'n={n}')
plt.xlabel('Momentum')
plt.ylabel('Probability Density')
plt.title('Probability Distribution for Momentum')
plt.legend()
plt.grid(True)
plt.show()
