import numpy as np
from scipy.linalg import norm, pinv
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


class RBF:
    def __init__(self, input_dim, num_centers, out_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.out_dim = out_dim
        self.centers = np.zeros((self.num_centers, 33))
        self.W = np.random.random((self.num_centers, self.out_dim))
        self.rbf_gaussian_sigma = 0

    def nonlinear(self, X, sigma):
        # Calculate the nonlinear mapping of each center vectors with each input vectors
        # Totally 330 input vectors, hidden layer output dimension is 330 * num_centers
        O = np.zeros((X.shape[0], self.num_centers), dtype=float)
        for centeridx, center in enumerate(self.centers):
            # for each center, calculate the nonlinear mapping of each input
            for xidx, x in enumerate(X):
                O[xidx, centeridx] = np.exp(-1 / (2 * sigma ** 2) * norm(x - center) ** 2)
        return O

    def random_sel(self, X):
        # select center vectors randomly
        rnd_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]

    def kmeans_sel(self, X):
        estimator = KMeans(n_clusters=self.num_centers, n_init=100)  # 构造聚类器
        estimator.fit(X)  # 聚类
        label_pred = estimator.labels_  # 获取聚类标签
        self.centers = estimator.cluster_centers_  # 获取聚类中心

    def train(self, X, Y):
        # self.random_sel(X)
        self.kmeans_sel(X)

        # calculate rbf_gaussian_sigma
        distances = pdist(self.centers)
        max_distance = distances.max()
        self.rbf_gaussian_sigma = max_distance / np.sqrt(2 * self.num_centers)
        # print("rbf_gaussian_sigma: {}".format(rbf_gaussian_sigma))

        # Nonlinear Mapping
        O = self.nonlinear(X, self.rbf_gaussian_sigma)

        # Linear Classifier - linear least square estimate
        phi = np.insert(O, 0, 1, axis=1)
        self.W = np.dot(np.dot(pinv(np.dot(phi.T, phi)), phi.T), Y)
        np.save("./checkpoints/RBF_Weight.npy", self.W)
        # np.save("./checkpoints/RBF_Phi.npy", phi)

        Y = np.dot(phi, self.W)
        return Y

    def predict(self, X):
        # Nonlinear transformation
        O = self.nonlinear(X, self.rbf_gaussian_sigma)

        # Add the bias term
        phi = np.insert(O, 0, 1, axis=1)

        self.W = np.load("./checkpoints/RBF_Weight.npy")

        Y_pred = np.dot(phi, self.W)
        return Y_pred
