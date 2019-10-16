import numpy as np 
import matplotlib.pyplot as plt 

def read_points_file(filename):
  pts = []
  with open(filename, "r") as f:
    for pt in f:
      pt = pt.strip("\n").split()
      pts.append([float(pt[0]), float(pt[1])])
  return pts

def read_data(class_0_file, class_1_file):
  pts_0 = read_points_file(class_0_file)
  pts_1 = read_points_file(class_1_file)

  X = pts_0 + pts_1
  y = [0] * len(pts_0) + [1] * len(pts_1)
  X = np.array(X)
  return (X, y)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_boundaries(ax, clf, xx, yy, **params):
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = clf.decision_function(xy).reshape(xx.shape)
    out = ax.contour(xx, yy, Z, **params)
    return out

