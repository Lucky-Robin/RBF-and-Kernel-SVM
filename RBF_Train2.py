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

# 初始化存储结果的列表
num_centers_list = []
accuracy_list = []

for k in range(2, 201):
    # Construct RBF network and Train, Test
    rbf = RBF(33, k, 1)
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

    # 将结果添加到列表中
    num_centers_list.append(k)
    accuracy_list.append(acc)

    print("Num_Centers: {}".format(k))
    print("RBF Training Accuracy: {}".format(acc_percentage))

# 绘制图表
plt.plot(num_centers_list, accuracy_list)
plt.xlabel("Number of Center Vectors")
plt.ylabel("RBF Training Accuracy")
plt.ylim(0, 1)
plt.title("RBF Training Accuracy vs Number of Center Vectors")
plt.grid(True)
plt.show()
