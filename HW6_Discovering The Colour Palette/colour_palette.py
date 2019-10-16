from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import warnings; warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn import metrics

road = np.array(Image.open('road.jpg'))

dataset = road / 255.0
dataset = dataset.reshape(360 * 640, 3)

def pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)

pixels(dataset, title='Input Colours')
plt.show()

kmeans = MiniBatchKMeans(16)
kmeans.fit(dataset)
new_colors = kmeans.cluster_centers_[kmeans.predict(dataset)]

labels = kmeans.labels_
print("The Davies-Bouldin score is :", metrics.davies_bouldin_score(dataset, labels))

pixels(dataset, colors=new_colors, title="Reduced to 16 - Colours")
plt.show()

twoD_recolouredroad = new_colors.reshape(dataset.shape)
recolouredroad = new_colors.reshape(road.shape)

pca = PCA(n_components=2)
pc_original = pca.fit_transform(dataset)
pc_recoloured = pca.fit_transform(twoD_recolouredroad)

N=250000
rng = np.random.RandomState(0)
i = rng.permutation(dataset.shape[0])[:N]
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
ax[0].scatter(pc_original[:, 0], pc_original[:, 1], color=dataset[i], marker='.', alpha=0.2)
ax[0].axis('equal')
ax[0].set(xlabel='1st Component', ylabel='2nd Component',
          title='Principal Components of Original Image',
          xlim=(-5, 5), ylim=(-3, 3.1))

ax[1].scatter(pc_recoloured[:, 0], pc_recoloured[:, 1], color=new_colors, marker='.', alpha=0.2)
ax[1].axis('equal')
ax[1].set(xlabel='1st Component', ylabel='2nd Component',
          title='Principal Components of Recoloured Image',
          xlim=(-5, 5), ylim=(-3, 3.1))
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(road)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(recolouredroad)
ax[1].set_title('16-Colours Image', size=16)
plt.show()