import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn import metrics
#from statistics import stdev
#from fractions import Fraction as fr

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

k=9
n = dataset.shape[0]
c = dataset.shape[1]

df = DataFrame(dataset,columns=['n','c'])
  
kmeans = KMeans(n_clusters=9, init='random').fit(df)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
print("The Davies-Bouldin score is :", metrics.davies_bouldin_score(df, labels))
print("The Silhouette score is :", metrics.silhouette_score(df, labels, metric='euclidean'))

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#FF9FE2', '#FFC433' ]
for i in range(k):
        points = np.array([dataset[j] for j in range(len(dataset)) if labels[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=15, c=colors[i], alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=200, c='#4E3900')
plt.title('Random Partition k-means Clustering')
plt.show()

#This part is used to calculate 5 runs which are done in report.
"""
sample_DB = (0.6431120315177721, 0.6199829505429216, 0.6173394480445622, 0.6102921888061947, 0.6072543770476985)
sample_S = (0.5273424559377415, 0.5361342915595941, 0.5377324272779415, 0.5434021069046533, 0.5392468049797844)

print("The Standard Deviation of Davies-Bouldin scores is % s" 
                              %(stdev(sample_DB))) 
                                
print("The Standard Deviation of Silhouette scores is % s" 
                              %(stdev(sample_S))) 
"""