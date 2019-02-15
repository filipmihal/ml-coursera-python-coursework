import os

from abstract import AbstractRegression
import numpy as np
from matplotlib import pyplot
from scipy import optimize


class LinearRegression(AbstractRegression):
    """Linear regression class"""

    def plot_data(self):

        index = 0
        for feature in self.get_features().transpose():
            index += 1
            pyplot.plot(feature, self.labels, 'ro', ms=10, mec='k')
            pyplot.ylabel('labels')
            pyplot.xlabel('features ' + str(index))
            pyplot.show()

    def cost_function(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute cost for linear regression. Computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y.
        :param theta
        :return:
        """
        matrix = np.subtract(np.dot(self.features, theta), self.labels)
        J = np.dot(matrix.transpose(), matrix) / (2 * self.labels.size)
        return J

    def gradient(self, theta: np.ndarray, alpha: float) -> np.ndarray:
        # compute gradient for the constants
        # compute the rest of gradients for all features
        output = np.zeros(self.features.shape[1])
        #add to the first row of the output
        k = 0
        output[k] = alpha * np.sum(np.dot(self.features, theta) - self.labels) / self.labels.size

        for feature in self.get_features().transpose():
            k += 1
            output[k] = alpha * np.dot(np.dot(self.features, theta) - self.labels, feature) / self.labels.size

        return output

    def minimize_cost(self, theta: np.ndarray):
        # number of iterations
        output = optimize.minimize(self.cost_function, theta, jac=True, method='TNC', options={'maxiter': 400})
        return output.x

    def predict(self, theta: np.ndarray):
        print("ahoj")


data = np.loadtxt(os.path.join('../Exercise1/Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
linear_regression = LinearRegression(X.T, y)
linear_regression.plot_data()
linear_regression.gradient(np.array([0, 0]), 0.01)
