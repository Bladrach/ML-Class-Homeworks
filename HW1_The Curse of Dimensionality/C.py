import numpy as np 
import matplotlib.pyplot as plt
import random
N=int(1e6)
d= range(1,101) 
x = []
for i in d:
    unif = np.random.uniform(0, 1, size=(N, i))
    point1 = random.choice(unif)
    point2 = random.choice(unif)
    point3 = random.choice(unif)
    point4 = random.choice(unif)
    v1 = point2 - point1
    v2 = point4 - point3
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    angle_r = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degrees = angle_r*180/np.pi
    x.append(angle_degrees)
    
plt.plot(x,'bo')
plt.xlabel("Dimension d")
plt.ylabel("Angle degrees for corresponding dimension ")
plt.show()