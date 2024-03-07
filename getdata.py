from scipy.io import loadmat

# Retrieve dataset from given .mat file
train_dataset = loadmat('dataset/data_train.mat')
train_labelset = loadmat('dataset/label_train.mat')
test_dataset = loadmat('dataset/data_test.mat')

train_data = train_dataset['data_train']
# print(train_data.shape)     # (330, 33)
train_label = train_labelset['label_train']
# print(train_label.shape)    # (330, 1)
test_data = test_dataset['data_test']
# print(test_data.shape)      # (21, 33)
