from sklearn.neighbors import KNeighborsClassifier
import numpy as np

mnist_embedding = np.load('mnist_features.npy')
mnist_labels = np.load('mnist_labels.npy')
svhn_embedding = np.load('svhn_features.npy')
svhn_labels = np.load('svhn_labels.npy')

print(mnist_embedding.shape, mnist_labels.shape)

clf_1 = KNeighborsClassifier(n_neighbors=1)
clf_1.fit(svhn_embedding, svhn_labels)
acc = clf_1.score(mnist_embedding, mnist_labels)
print(acc)
