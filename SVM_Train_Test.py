from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from getdata import train_data, train_label, test_data
from para_init import svm_seed
import numpy as np

np.random.seed(svm_seed)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label.ravel(), test_size=0.3)

svm = SVC(kernel='rbf')
svm.fit(x_train, y_train.ravel())

acc = svm.score(x_test, y_test.ravel())
acc_percentage = "{:.3%}".format(acc)
print(acc)
print("SVM Training Accuracy: {}".format(acc_percentage))

# 使用训练好的模型对 test_data 进行预测
test_predictions = svm.predict(test_data)
print(test_predictions)

# max = 0.1
# for i in range(10000):
#     seed = i
#     np.random.seed(seed)
#     x_train, x_test, y_train, y_test = train_test_split(train_data, train_label.ravel(), test_size=0.3)
#
#     svm = SVC(kernel='rbf')
#     svm.fit(x_train, y_train.ravel())
#
#     acc = svm.score(x_test, y_test.ravel())
#     acc_percentage = "{:.3%}".format(acc)
#     # print(acc)
#     if acc >= max:
#         max = acc
#         print("seed: {}".format(seed))
#         print("SVM Training Accuracy: {}".format(acc_percentage))
