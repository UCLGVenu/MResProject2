import numpy as np
import matplotlib.pyplot as plt

def get_n_eff(theta):
    n_ord = 1.5
    n_ext = 1.8
    n_eff = (n_ord * n_ext) / (((n_ext ** 2 * np.sin(theta) ** 2) + (n_ord ** 2 * np.cos(theta) ** 2)) ** 0.5)
    return n_eff

thetas = np.linspace(0, 0.5 * np.pi, 100)

ns = []
n_ons = []

for i in thetas:
    n_eff = get_n_eff(i)
    n_on = 1.5
    ns.append(n_eff)
    n_ons.append(n_on)

difference = [x - y for x, y in zip(ns, n_ons)]

plt.plot(thetas, ns)
plt.plot(thetas, n_ons)
plt.plot(thetas, difference)
plt.xlabel('Angle of Linear Polarisation / rad')
plt.ylabel('Refractive Index')
plt.legend(['0V Applied' , 'Large V Applied' , 'Difference'])
plt.title('Refractive index in unswitched/fully switched LC states \n against Angle of Polarisation')
plt.show()