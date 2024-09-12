import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Define coherent state |alpha>
def coherent_state(alpha, n_mean):
    alpha_abs_sq = np.abs(alpha) ** 2
    coherent = np.zeros(n_mean+1, dtype=np.complex128)
    for n in range(n_mean+1):
        coherent[n] = (np.exp(-0.5 * alpha_abs_sq) * alpha ** n) / np.sqrt(factorial(n))
    return coherent

# Define number states |n>
def number_state(n, n_mean):
    number = np.zeros(n_mean+1)
    number[n] = 1.0
    return number

# Define photon number probability distribution P_n(alpha)
def photon_prob_distribution(alpha, n_mean):
    alpha_abs_sq = np.abs(alpha) ** 2
    prob_distribution = np.zeros(n_mean+1)
    for n in range(n_mean+1):
        prob_distribution[n] = np.abs((np.exp(-0.5 * alpha_abs_sq) * alpha ** n) / np.sqrt(factorial(n))) ** 2
    return prob_distribution

# Parameters
alpha = 1.5 + 0.5j  # Complex value of alpha
n_mean = 10          # Mean photon number

# Calculate coherent state |alpha>
coherent = coherent_state(alpha, n_mean)

# Calculate photon number probability distribution P_n(alpha)
prob_distribution = photon_prob_distribution(alpha, n_mean)

# Plot results
plt.bar(range(n_mean+1), prob_distribution)
plt.xlabel('Photon Number (n)')
plt.ylabel('Probability Density')
plt.title('Coherent State Photon Number Probability Distribution')
plt.show()
