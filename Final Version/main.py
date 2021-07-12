# This is a sample Python script.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from pydataset import data
from matplotlib import pyplot as plt
import time


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# linear regression using "mini-batch" gradient descent
# function to compute hypothesis / predictions
def hypothesis(X, theta):
    return np.dot(X, theta)


# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad


# function to compute the error for current values of theta
def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))
    J /= 2
    return J[0]


# function to create a list containing mini-batches
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


# function to perform mini-batch gradient descent
def gradientDescent(X, y, learning_rate=0.00001, batch_size=4):

    theta = np.ones((X.shape[1], 1))
    error_list = []
    max_iters = 10
    for itr in range(max_iters):
        mini_batches = create_mini_batches(X, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta = theta - learning_rate * gradient(X_mini, y_mini, theta)
            error_list.append(cost(X_mini, y_mini, theta))

    return theta, error_list


def predict(y, theta1, theta2):
    return y * theta1 + theta2


def vanilla_gradient(x_train,y_train):
    eta = 0.001  # learning rate
    n_iterations = 40000
    y = y_train.values.reshape(-1, 1)
    m = len(x_train.values)
    X = x_train.values.reshape(-1, 1)
    X_b = np.c_[np.ones((len(x_train.values), 1)), X]
    theta = np.ones((2,1))

    iteration =0
    while iteration < n_iterations :
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)

        theta = theta - eta * gradients
        iteration+=1
    return theta


def learning_schedule(t):
    t0, t1 = 1, 1000
    return t0 / (t + t1)


def stochastic_gradient_descent(x_train, y_train):
    n_epochs = 200
    X = x_train.values.reshape(-1, 1)
    y = y_train.values.reshape(-1, 1)
    X_b = np.c_[np.ones((len(x_train), 1)), X]
    theta = np.ones((2,1))# random initialization
    m = len(x_train.values)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
    return theta

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pima = data('Pima.tr')
    # test train split
    x_train, x_test, y_train, y_test = train_test_split(pima.skin, pima.bmi)
    plt.scatter(x_test, y_test, label="Test Data", color="b", alpha=.7)
    plt.scatter(x_train, y_train, label="Training data", color="g", alpha=.7)
    plt.legend()
    plt.show()


    # My implementation of the vanilla
    t0 = time.time();
    theta = vanilla_gradient(x_train,y_train)
    X_new_b = np.c_[np.ones((len(x_test.values), 1)), x_test.values.reshape(-1, 1)]
    y_predict = X_new_b.dot(theta)
    t1 = time.time() - t0

    t0 = time.time();
    y = y_train.values.reshape(-1, 1)
    sgd_reg = SGDRegressor(max_iter=100, tol=1e-15, penalty=None, eta0=0.0001)
    sgd_reg.fit(x_train.values.reshape(-1,1), y.ravel())
    thetino = sgd_reg.intercept_, sgd_reg.coef_
    y2_predict = X_new_b.dot(thetino)
    t2 = time.time() - t0

    t3 = time.time()

    al, bl= gradientDescent(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

    y_pred = hypothesis(x_test.values.reshape(-1, 1), al)
    t4 =  time.time() -t3 
    print(al)
    plt.plot(bl)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

    LR = LinearRegression()
    LR.fit(x_train.values.reshape(-1, 1), y_train.values)
    prediction = LR.predict(x_test.values.reshape(-1, 1))
    plt.title(t1)
    plt.plot(x_test, prediction, label="LinearRegression", color='b')
    plt.plot(x_test, y_predict, label="Vanilla Gradient batch time :" + str(t1), color='r', alpha=.5)
    plt.plot(x_test, y2_predict, label="Stochastic Gradient batch time :" + str(t2), color='y', alpha=.5)
    plt.plot(x_test, y_pred, label="Minibatch Gradient batch time :" + str(t4), color='pink', alpha=.5)
    plt.scatter(x_test, y_test, label="Testing data", color="g", alpha=.7)
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
