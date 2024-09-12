import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, fock, qeye, destroy, mesolve

times = np.linspace(0.0, 10.0, 200)
psi0 = tensor(fock(2,0), fock(10, 5))
a = tensor(qeye(2), destroy(10)) #Lowering operator for the optical cavity
sm = tensor(destroy(2), qeye(10)) #Sigma vector for the atom (from ground to excited state)
#Hamiltonian Equation
H = 2 * np.pi * a.dag() * a + 2 * np.pi * sm.dag() * sm + 2 * np.pi * 0.25 * (sm * a.dag() + sm.dag() * a) 
#Solves the Hamiltonian equation to give discretized energy states functions
result = mesolve(H, psi0, times, [np.sqrt(0.1)*a], e_ops=[a.dag()*a, sm.dag()*sm]) 

#Plotting
plt.figure()
plt.plot(times, result.expect[0],'r')
plt.plot(times, result.expect[1],'g')
plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.grid()
plt.legend(("cavity photon number", "atom excitation probability"))
plt.show()