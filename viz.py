import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train.csv").to_numpy()
X_train = X_train[:,1:]  #remove the id column
y_train = pd.read_csv("y_train.csv").to_numpy()
y_train = y_train[:,1]
X_test = pd.read_csv("X_test.csv").to_numpy()
X_test = X_test[:,1:]  #remove the id column


print('X_train[0].size()', X_train[0].size)
# make histogram plot of X_train[0]
plt.hist(X_train[0], bins=10)
plt.show()
plt.hist(X_train[100], bins=10)
plt.show()
plt.hist(X_train[200], bins=10)
plt.show()
plt.hist(X_train[300], bins=10)
plt.show()