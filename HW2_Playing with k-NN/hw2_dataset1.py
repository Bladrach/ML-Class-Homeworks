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

X1, y1 = make_blobs(n_samples=150, centers=4, n_features=2,random_state=21)
X2, y2 = make_gaussian_quantiles(mean=(2,2),cov=3., n_samples=150, n_features=2, n_classes=3, random_state=9)
X3, y3 = make_gaussian_quantiles(mean=(5,5),cov=5., n_samples=150, n_features=2, n_classes=2, random_state=15)

X = concatenate([X1,X2,X3])
y = concatenate([y1,y2,y3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=35)

h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#8B008B'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#8B008B'])

clf1 = KNeighborsClassifier(n_neighbors=1, weights='uniform')
clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)

clf2 = KNeighborsClassifier(n_neighbors=5, weights='uniform')
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)

clf3 = KNeighborsClassifier(n_neighbors=5, weights='distance')
clf3.fit(X_train, y_train)
pred3 = clf3.predict(X_test)

clf4 = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='mahalanobis', metric_params={'V': np.cov(X_train,rowvar=False)})
clf4.fit(X_train, y_train)
pred4 = clf4.predict(X_test)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)

Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)

Z3 = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
Z3 = Z3.reshape(xx.shape)

Z4 = clf4.predict(np.c_[xx.ravel(), yy.ravel()])
Z4 = Z4.reshape(xx.shape)

accuracy1 = accuracy_score(y_test, pred1) * 100
print('\nThe accuracy of k=1, majority voting, Euclidean metric classifier is %d%%' % accuracy1)

accuracy2 = accuracy_score(y_test, pred2) * 100
print('\nThe accuracy of k=5, majority voting, Euclidean metric classifier is %d%%' % accuracy2)

accuracy3 = accuracy_score(y_test, pred3) * 100
print('\nThe accuracy of k=5, weighted distance voting, Euclidean metric classifier is %d%%' % accuracy3)

accuracy4 = accuracy_score(y_test, pred4) * 100
print('\nThe accuracy of k=1, majority voting, Mahalanobis metric classifier is %d%%' % accuracy4)

subplot(2,2,1)
plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title('KNN Classifier k 1, weights Uniform, metric Euclidean')
subplot(2,2,2)
plt.pcolormesh(xx, yy, Z2, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title('KNN Classifier k 5, weights Uniform, metric Euclidean')
subplot(2,2,3)
plt.pcolormesh(xx, yy, Z3, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title('KNN Classifier k 5, weights Distance, metric Euclidean')
subplot(2,2,4)
plt.pcolormesh(xx, yy, Z4, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
title('KNN Classifier k 1, weights Uniform, metric Mahalanobis')

plt.show()