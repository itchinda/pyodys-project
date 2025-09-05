import pyodys as ode

import numpy as np


class systeme(ode.EDOs):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)
    
    def evalue(self, t, u):
        x, y = u
        return np.array([-x + y, -y])

# Analytical solution
def solution_analytique(t):
    x = np.exp(-t) * (1 +  t)
    y =  np.exp(-t)
    return np.column_stack((x, y))  

f = systeme(0, 5, np.array([1.0, 1.0]))
# time points and exact solution
xi = np.linspace(0, 2, 5)
yi = solution_analytique(xi)

print(xi)
print(yi)
xnew = np.linspace(0, 2, 50)
ynew = ode.hermite_interpolate(xi, yi, f, xnew)

import matplotlib.pyplot as plt
plt.plot(xnew, solution_analytique(xnew), 'k', label="true")
plt.plot(xi, yi, 'o', label="data")
plt.plot(xnew, ynew, '--', label="hermiteSoln")
plt.legend()
plt.show()
