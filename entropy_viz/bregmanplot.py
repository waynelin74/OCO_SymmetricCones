"""
PARAMETERS
"""

ref = np.array([0.3*0.7, 0.4*0.7, 0.5]) # Reference point y

"""--------------------------------------------------------------------------"""
"""
Packages
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from SOC import *


"""--------------------------------------------------------------------------"""
"""
Main
"""
density = 10**3 # Grid point density
x_axis = np.linspace(-0.5, 0.5, density)
y_axis = np.linspace(-0.5, 0.5, density)
X, Y = np.meshgrid(x_axis, y_axis)

def point(x,y):
    return np.array([x, y, 0.5])
pointv = np.vectorize(point, otypes=[object])
lnv = np.vectorize(ln, otypes=[object])
jordanprodv = np.vectorize(jordanprod, otypes=[object])
jordanprodhalfv = np.vectorize(jordanprod, otypes=[object], excluded=['y'])
trv = np.vectorize(tr, otypes=[object])
points = pointv(X,Y)
H = trv(jordanprodv(points, lnv(points)) - jordanprodhalfv(x=points, y=ln(ref)))

conplot = plt.figure().add_subplot(aspect='equal')
cons = plt.contour(X, Y, H, levels=40)
# Feasible disc
theta = np.linspace(0, 2*np.pi, 10**4)
x0 = 0.5 * np.cos(theta)
x1 = 0.5 * np.sin(theta)
x2 = list(map(lambda x: 0.5, theta))
plt.plot(x0, x1, label='Feasible disc')
#point
plt.plot(ref[0], ref[1], marker="o", markersize=10, markeredgecolor="red",
markerfacecolor="red")
cbar = plt.colorbar(cons)
#plot
plt.tight_layout()
plt.show()
