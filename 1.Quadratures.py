import numpy as np
import matplotlib.pyplot as plt

#Initial conditions:
L = 1
N = 2 #State number
k = np.pi*N/L
c = 1
omega = k*c

def SHO(q,t): #Defined a function to give the values of the position and velocity for the SHO
  x = q[0] #Initial x position
  v = q[1] #Initial velocity
  fx = v #We dissolved the given SHO equation into two first order equations dx/dt=v and dv/dt=-omega^2*x
  fv = -omega**2*x
  return np.array([fx,fv],float)

def leapfrog(function, init_cond,a,b,n): #
  h = (b-a)/n #Spacing between 2 points
  t_points = np.arange(a,b+h,h) #Time scale
  x_points = [init_cond[0]]
  v_points = [init_cond[1]]
  half_values = init_cond + 0.5 * h * function(init_cond,t_points[0]) #introduced a new 2D array 
  #to store the initial value of the function at half-step interval
  for i in range(1,len(t_points)):
    init_cond += h * function(half_values,t_points[i]+0.5*h)
    x_points.append(init_cond[0]), v_points.append(init_cond[1])
    half_values += h * function(init_cond,t_points[i]+h) #Updated the next half-integral 
    #value using the previous half-integral value

  return x_points, v_points, t_points

#Initial Conditions:
a = 0.0
b = 10.0
n = 10000
h = (b-a)/n
init_cond_1 = [0, c] #Initial conditions for part a

soln   = leapfrog(SHO,init_cond_1,a,b,n)

time, q, v = soln[2],soln[0],soln[1]

def central_diff(f):
  der = []
  for i in range(len(f)):
    if i==0:
      der.append((f[i+1]-f[i])/h)
    elif 1 <= i < len(f)-1:
      der.append((f[i+1]-f[i-1])/(2*h))
    else:
      der.append((f[i]-f[i-1])/h)
  return der

qdot = central_diff(q)

plt.plot(time,qdot,label='p(t)')
plt.plot(time,q, label='q(t)')
plt.xlabel("time")
plt.ylabel("Position/Momentum")
plt.legend(loc='best')
plt.title("Quadratures")
plt.show()