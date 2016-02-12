#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import sys
import json
import pickle
import scipy.ndimage
import scipy.interpolate

u = 2.0
Q = .37
H = 40.


x_std_dev = .1
pi = math.pi
	
def calc_sigma_y(x):
	return .32*x*((1.+.0004*x)**-.5)
	
def calc_sigma_z(x):
	sigma_z = .24*x*((1+.001*x)**.5)
	if sigma_z == H/((2.)**5):
		print x
		sys.exit()
	return sigma_z

def calc_concentration(x,y,z):

	y_std_dev = calc_sigma_y(x)
	z_std_dev = calc_sigma_z(x)
	
	if x == 0:
		g1 = 1.0
	else:
		g1 = math.exp((-.5 * (y**2))/(y_std_dev**2))
	
	if z_std_dev == 0:
		g2_1 = 0
		g2_2 = 0
	else:
		g2_1 = -.5 * ((H-z)**2)/(z_std_dev**2)
		g2_2 = -.5 * ((H+z)**2)/(z_std_dev**2)

	g2 = math.exp(g2_1) + math.exp(g2_2)
	
	if y_std_dev == 0 or z_std_dev == 0:
		conc = 0
	else:

		conc = Q * (1/u)
		conc = conc * g1
		conc = conc / ((2*pi)**.5)
		conc = conc / y_std_dev
		conc = conc * g2
		conc = conc / ((2*pi)**.5)
		conc = conc / z_std_dev

	if np.isnan(conc):
		print conc
		print x,y,z
		print g1, g2, y_std_dev, z_std_dev
		sys.exit()
	return conc * (10**4)
	
#Define the limits
step = 1

x_max = 1000.
xs = list(np.linspace(0.,x_max,(x_max/step)+1))

y_max = 1000
ys = list(np.linspace(-1000.,1000.,(y_max/step)+1))

z_max = 500.
zs = list(np.linspace(0.,z_max,(z_max/step)+1))

#for diagnostic purposes
start_time = time.time()

#calc concentration and append as tuple (x,y,z,c) into concentration list
concentration = []
zi = np.empty([len(xs), len(ys)])
min_conc = 1000
max_conc = 0
y_slice = 100
zs = [y_slice]
for i,z in enumerate(zs):
	for j,y in enumerate(ys):
		for k,x in enumerate(xs):
			temp_conc = calc_concentration(x,y,z)
			if z == y_slice:
				zi[j][k] = temp_conc
			if temp_conc > max_conc:
				max_conc = temp_conc
			if temp_conc < min_conc:
				min_conc = temp_conc
			
			concentration.append((x,y,z,temp_conc))

print "min, max", min_conc, max_conc
end_time = time.time() - start_time
print end_time, step
plt.imshow(zi, vmin = min([min(v) for v in zi]), vmax = max([max(v) for v in zi]), origin = 'lower', extent = [xs[0], xs[-1], ys[0], ys[-1]])
#print xs[0], xs[-1]
#plt.imshow([[1,0],[2,3]], vmin = 0, vmax = 3, origin = 'lower', extent = [0,1,0,1])
plt.colorbar()
		
plt.show()
print time.time() - start_time
