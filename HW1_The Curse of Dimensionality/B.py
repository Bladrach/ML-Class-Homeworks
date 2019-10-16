import numpy as np 
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import random
from statistics import stdev
from statistics import mean
from math import sqrt
N=int(1e6)
d= range(1,10) 
std = []
avg = []
ratio = []
for i in d:
    unif = np.random.uniform(0, 1, size=(N, i))
    point1 = random.choice(unif)
    point2 = random.choice(unif)
    point3 = random.choice(unif)
    point4 = random.choice(unif)
    point5 = random.choice(unif)
    point6 = random.choice(unif)
    distance1 = euclidean(point2,point1)
    distance2 = euclidean(point4,point3)
    distance3 = euclidean(point6,point5)
    distance = [distance1, distance2, distance3]
    std.append(stdev(distance))
    avg.append(mean(distance))

    end_index = len(std)
    for k in range(end_index):
        ratio.append(std[k]/avg[k])
    
plt.plot(ratio)
plt.xlabel("Dimension d")
plt.ylabel("Ratio of standard deviation to average distance")
plt.show()
