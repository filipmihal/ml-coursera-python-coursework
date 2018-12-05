# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

data = np.genfromtxt('beer_consumption_data.csv', delimiter=',')
TEMP_MED, TEMP_MIN, TEMP_MAX, PRECIPITATION, WEEKEND, CONSUMPTION = data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]

M = CONSUMPTION.size  # number of training examples

# Add a column of ones to X. The numpy function stack joins arrays along a given axis.
# The first axis (axis=0) refers to rows (training examples)
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(M), TEMP_MED], axis=1)


def plot_data(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    fig = pyplot.figure()
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')


def gradient_descent(x, y, theta, alpha, num_iterations):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    x : array_like
        The input data set of shape (m x n+1).

    y : array_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iterations : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    error = 0
    for i in range(num_iterations):
        temp_slope_0 = alpha * np.sum(np.dot(x, theta) - y) / m
        temp_slope_1 = alpha * np.dot(np.dot(x, theta) - y, x[:, 1]) / m
        theta[0] = theta[0] - temp_slope_0
        theta[1] = theta[1] - temp_slope_1
        error = (np.sum(np.dot(x, theta) - y) ** 2) / (2*m)
    print("error: " + str(error))
    return theta


# initialize fitting parameters
THETA = np.zeros(2)

# some gradient descent settings
ITERATIONS = 1500
ALPHA = 0.001

THETA = gradient_descent(X, CONSUMPTION, THETA, ALPHA, ITERATIONS)

# plot the linear fit
plot_data(X[:, 1], CONSUMPTION)
pyplot.plot(X[:, 1], np.dot(X, THETA), '-')
pyplot.legend(['Training data', 'Linear regression'])
pyplot.show()

PREDICTION = np.dot([1, 25], THETA)
print('For temperature = 25C, we predict a beer consumption of {:.2f} liters\n'.format(PREDICTION))

