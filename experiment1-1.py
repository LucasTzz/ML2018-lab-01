from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.linalg import inv
from io import BytesIO
import requests

# download the dataset
r = requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale''')
X, y = load_svmlight_file(f=BytesIO(r.content), n_features=13)

# preprocess
X = X.toarray()
n_samples, n_features = X.shape
X = np.column_stack((X, np.ones((n_samples, 1))))
y = y.reshape((-1, 1))

# split dataset to the training set and the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

# initialize with random numbers
w = np.random.random((n_features + 1, 1))

# calculate the initial loss
Y_init = np.dot(X_train, w)
Loss = np.average(np.abs(Y_init - y_train))

# closed-form solution
w_update = inv(np.dot(X_train.transpose(), X_train)).dot(X_train.transpose()).dot(y_train)
Y_predict = np.dot(X_train, w_update)

# loss on training set
loss_train = np.average(np.abs(Y_predict - y_train))

# loss on validation set
Y_predict_val = X_val.dot(w_update)
loss_val = np.average(np.abs(Y_predict_val - y_val))

# print losses
print(Loss, loss_train, loss_val)
