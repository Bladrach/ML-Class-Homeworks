import numpy as np 
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def calculate(N=int(1e6), r=1, d=2):
    unif = np.random.uniform(0, 1, size=(N, d))

    count = 0
    for i in range(N):
        if euclidean(np.zeros((1,d)), unif[i]) < r:
            count += 1

    cube_volume = (2*r)**d 

    print("Volume of hypercube = {}".format(cube_volume))
    print("Percentage of points inside hypersphere = {}".format(count / float(N)))
    print("-----------------------------------------")

    return count / float(N)

D = range(1, 31)

x = []
for d in D:
    print("d = {}".format(d))
    perc = calculate(d=d)
    x.append(perc)

plt.plot(x)
plt.xlabel("Dimension d")
plt.ylabel("Percentage of inside points ")
plt.show()