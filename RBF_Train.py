import numpy as np
from getdata import train_data, train_label, test_data
from models.RBF import RBF
from para_init import num_centers, seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(seed)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label.ravel(), test_size=0.3)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Construct RBF network and Train, Test
rbf = RBF(33, num_centers, 1)
y = rbf.train(x_train, y_train)
y = np.where(y > 0, 1, -1)

y_pred = rbf.predict(x_test)
y_pred = np.where(y_pred > 0, 1, -1)

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        count += 1
acc = count / len(y_pred)
acc_percentage = "{:.3%}".format(acc)

print("Num_Centers: {}".format(num_centers))
print("RBF Training Accuracy: {}".format(acc_percentage))

test_pred = rbf.predict(test_data)
test_pred = np.where(test_pred > 0, 1, -1)
print(test_pred.T)
