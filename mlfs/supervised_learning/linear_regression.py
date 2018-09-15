import numpy as np
import pandas as pd

class LinearRegression():
    """ Linear Regression.
    Args:
        alpha (float): The learning rate.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    def __init__(self, alpha=0.1, batch=False):
        self.alpha = alpha

    def computeCost(self, X, y, theta, m, n):
        # hypothesis: h(X) = X * theta = t1 * x1 + ... + tn * xn + t0 * x0
        hypothesis = pd.DataFrame(X.multiply(theta).sum(axis=1), columns=y.columns.tolist())
        # error
        E = (hypothesis - y)
        # cost function
        J = (1/ (2 * m)) * ((hypothesis - y) ** 2).sum(axis=0)
        # update E data frame to be able to multiply it by the features in X
        for i in range(1, n):
            E[i] = E[0]
        # gradient
        theta_grad = theta - self.alpha * (1/ m) * ((X * E).sum(axis=0)).values

        return J.values[0], theta_grad

    def train(self, X, y, num_iters=1000):
        # rename the columns of X and y to be numbers
        m, n = np.shape(X)
        X.columns = [x for x in range( n )]
        y.columns = [x for x in range( 1 )]

        # init to theta to a random value
        J_history = []
        theta = np.random.rand(n)
        for _ in range(num_iters):
            #print('iteration: ' + str(i))
            J, theta = self.computeCost(X, y, theta, m, n)
            J_history.append(J)

        return J_history, theta

"""
    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred
"""