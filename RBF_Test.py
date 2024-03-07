import numpy as np
from getdata import train_data, train_label, test_data
from models.RBF2 import RBF
from para_init import num_centers, seed

np.random.seed(seed)

# Pick center vectors randomly
rnd_idx = np.random.permutation(train_data.shape[0])[:num_centers]
centers = [train_data[i, :] for i in rnd_idx]

# Construct RBF network and Train, Test
rbf = RBF(33, num_centers, 1)
y_pred = rbf.predict(test_data)
y_pred = np.where(y_pred > 0, 1, -1)
print(y_pred.T)

# count = 0
# for i in range(len(y)):
#     if y[i] == train_label[i]:
#         count += 1
# acc = count / len(y)
# acc_percentage = "{:.3%}".format(acc)
#
print("Num_Centers: {}".format(num_centers))
# print("RBF Training Accuracy: {}".format(acc_percentage))
