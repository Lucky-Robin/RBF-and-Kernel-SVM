import numpy as np
from getdata import train_data, train_label, test_data
from models.RBF import RBF

# Initialization parameter
# RBF
num_centers = 10
seed = 973
# seed = 20

# SVM
svm_seed = 313

if __name__ == '__main__':
    max = 0.1
    for seed in range(10000):
        np.random.seed(seed)
        # Pick center vectors randomly
        rnd_idx = np.random.permutation(train_data.shape[0])[:4]
        centers = [train_data[i, :] for i in rnd_idx]

        # Construct RBF network and Train, Test
        rbf = RBF(33, num_centers, 1)
        y = rbf.train(train_data, train_label)
        y = np.where(y > 0, 1, -1)

        count = 0
        for i in range(len(y)):
            if y[i] == train_label[i]:
                count += 1
        acc = count / len(y)
        if acc > max:
            max = acc
            print("seed = {}".format(seed), "acc = {}".format(max))
