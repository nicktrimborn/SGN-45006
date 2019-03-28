#from pylab import *
from linefitlsq import linefitlsq
import numpy as np
import matplotlib.pyplot as plt

# Load and plot points
data = np.load('points.npy')
x, y = data[0,:], data[1,:]
plt.figure(1, (10,10))
plt.plot(x, y, 'kx')
plt.axis('scaled')
# plt.show()

# m is the number of data points
m = np.size(x)*1.0
# s is the size of the random sample
s = 2
# t is the inlier distance threshold
t = np.sqrt(3.84)*2
# e is the expected outlier ratio
e = 0.8
# at least one random sample should be free 
# from outliers with probability p
p = 0.999
# required number of samples
N_estimated = np.log(1-p) / np.log(1-(1-e)**s)
print("Estimated number of samples: %.1f" % N_estimated)

############## RANSAC loop ######################

# First initialize some variables
N = np.inf
sample_count = 0
max_inliers = 0
best_line = np.zeros((3,1))

# Data points in homogeneous coordinates
points_h = np.vstack((x,y,np.ones((int(m)))))
while N > sample_count:
    # Pick two random samples
    samples = np.random.choice(np.arange(len(x)), 2, replace=False)
    a = samples[0]  # sample id 1
    b = samples[1]  # sample id 2
    
    # Determine the line crossing the points with the cross product 
    # of the points (in homogeneous coordinates).
    
    ##-your-code-starts-here-##
    l = np.cross(points_h[:,a],points_h[:,b])
    ##-your-code-ends-here-##
        
    # Determine inliers by finding the indices for the line and data point dot
    # products that are less than inlier distance threshold t.

    inliers = []
    ##-your-code-starts-here-##
    for i in range(int(m)):
        dist = np.abs(np.dot(l, points_h[:,i]))
        if (dist <= t):
            inliers.append(i)
    ##-your-code-ends-here-##

    # Store the line in best_line and update max_inliers if the number of 
    # inliers is the best so far
    inlier_count = np.size(inliers)
    if inlier_count > max_inliers:
        best_line = l
        max_inliers = inlier_count

    # Update the estimate of the outlier ratio
    e = 1-inlier_count/m
    # Update also the estimate for the required number of samples
    N = np.log(1-p)/np.log(1-(1-e)**s)
    
    sample_count += 1

# Least squares fitting to the inliers of the best hypothesis, i.e
# find the inliers similarly as above but this time for the best line.
inliers.clear()
##-your-code-starts-here-##
for i in range(int(m)):
    dist = np.abs(np.dot(best_line, points_h[:,i]))
    if (dist <= t):
        inliers.append(i)

x_inliers = x[inliers]
y_inliers = y[inliers]

# Fit a line to the above-given points (non-homogeneous)
l = linefitlsq(x_inliers, y_inliers)
print(l)
#l = best_line
# Plot the resulting line and the inliers
k = -l[0] / l[1]
b = -l[2] / l[1]
plt.plot(np.arange(1,101), k*np.arange(1,101)+b, 'm-')
plt.plot(x[inliers], y[inliers], 'ro', markersize=7)
plt.show()

