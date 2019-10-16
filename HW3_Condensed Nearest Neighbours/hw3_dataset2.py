import numpy as np
from numpy import concatenate
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pylab import subplot, title 
from matplotlib.colors import ListedColormap
from imblearn.under_sampling import CondensedNearestNeighbour

X1, y1 = make_blobs(n_samples=150, centers=4, n_features=2,random_state=19)
X2, y2 = make_gaussian_quantiles(mean=(7,2),cov=3., n_samples=150, n_features=2, n_classes=3, random_state=7)
X3, y3 = make_gaussian_quantiles(mean=(1,3),cov=4., n_samples=150, n_features=2, n_classes=2, random_state=37)

X = concatenate([X1,X2,X3])
y = concatenate([y1,y2,y3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=35)

cnn = CondensedNearestNeighbour(random_state=0)  #random_state is used to get the same result for every run
X_res1, y_res1 = cnn.fit_sample(X, y)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_res1, y_res1, test_size=0.25, random_state=35)    

h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFA500'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFA500'])

clf1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
clf2 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
clf1.fit(X_train, y_train)
clf2.fit(X_train1,y_train1)
pred1 = clf1.predict(X_test)

pred2 = clf2.predict(X_test1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))          

x_min1, x_max1 = X_res1[:, 0].min() - 1, X_res1[:, 0].max() + 1
y_min1, y_max1 = X_res1[:, 1].min() - 1, X_res1[:, 1].max() + 1
xx1, yy1 = np.meshgrid(np.arange(x_min1, x_max1, h),np.arange(y_min1, y_max1, h))    

Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)

Z2 = clf2.predict(np.c_[xx1.ravel(), yy1.ravel()])      
Z2 = Z2.reshape(xx1.shape)

accuracy1 = accuracy_score(y_test, pred1) * 100
print('\nThe accuracy of k=1, majority voting, Euclidean metric classifier is %d%%' % accuracy1)

accuracy2 = accuracy_score(y_test1, pred2) * 100
print('\nThe accuracy of k=1, majority voting, Euclidean metric classifier (CNN randomly chosing samples) is %d%%' % accuracy2)

subplot(1,2,1)
plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title('KNN Classifier k 1, weights Uniform, metric Euclidean')

subplot(1,2,2)
plt.pcolormesh(xx1, yy1, Z2, cmap=cmap_light)
plt.scatter(X_res1[:, 0], X_res1[:, 1], c=y_res1, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(yy1.min(), yy1.max())
title('KNN Classifier k 1, weights Uniform, metric Euclidean, (CNN randomly chosing samples)')

plt.show()