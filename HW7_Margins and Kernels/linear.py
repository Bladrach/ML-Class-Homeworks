import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import sys
import os
sys.path.append(os.path.abspath("../"))
from utils import *

# Color maps
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Read data
X, y = read_data("points_class_0.txt", "points_class_1.txt")
X = StandardScaler().fit_transform(X)

# Split data to train and test on 60-40 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=100)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
percentage = 100 * (1 - score)
print('Percentage of points on the wrong side of the separating hyperplane = ', percentage)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVM (C= 100)')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
# plot decision boundary and margins
plot_boundaries(ax, clf, xx, yy, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=cm_bright, s=40)

# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 40,
            linewidth=1.5, facecolors='none', edgecolors = 'g', label='Support Vectors')
# To write 'percentage of points on the wrong side of the seperating hyperplane' on the graph 
ax.text(xx.max() - .3, yy.min() + .3, ('%.4f' % percentage).lstrip('0'),
                size=15, horizontalalignment='right')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()