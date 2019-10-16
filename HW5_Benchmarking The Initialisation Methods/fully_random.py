from copy import deepcopy
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

center_1 = np.array([4,4])
center_2 = np.array([4,8])
center_3 = np.array([4,12])
center_4 = np.array([10,4])
center_5 = np.array([10,8])
center_6 = np.array([10,12])
center_7 = np.array([16,4])
center_8 = np.array([16,8])
center_9 = np.array([16,12])

data_1 = np.random.randn(200,2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3
data_4 = np.random.randn(200,2) + center_4
data_5 = np.random.randn(200,2) + center_5
data_6 = np.random.randn(200,2) + center_6
data_7 = np.random.randn(200,2) + center_7
data_8 = np.random.randn(200,2) + center_8
data_9 = np.random.randn(200,2) + center_9

dataset = np.concatenate((data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9), axis = 0)

k = 9
n = dataset.shape[0]
c = dataset.shape[1]

mean = np.mean(dataset, axis = 0)
std = np.std(dataset, axis = 0)
centres = np.random.randn(k,c)*std + mean

old_centres = np.zeros(centres.shape) 
new_centres = deepcopy(centres)

dataset.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(new_centres - old_centres)

while error != 0:
    for i in range(k):
        distances[:,i] = np.linalg.norm(dataset - centres[i], axis=1)
    
    clusters = np.argmin(distances, axis = 1)
    old_centres = deepcopy(new_centres)
    for i in range(k):
        new_centres[i] = np.mean(dataset[clusters == i], axis=0)
    error = np.linalg.norm(new_centres - old_centres)

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#FF9FE2', '#FFC433' ]
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([dataset[j] for j in range(len(dataset)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=15, c=colors[i], alpha=0.5)
ax.scatter(new_centres[:, 0], new_centres[:, 1], marker='+', s=200, c='#4E3900')
plt.title('Fully Random k-means Clustering')
plt.show()