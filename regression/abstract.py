from abc import ABC, abstractmethod
import numpy as np
from typing import List


class AbstractRegression(ABC):
    """ Abstract regression class """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        initialize the class
        :param features: np.array
        :param labels: np.array
        """
        self.precondition(features, labels)
        self.features = self.add_ones_to_matrix(features)
        self.labels = labels

    def get_features(self):
        """
        As the self.features contains also ones that are needed for gradient descent and cost function,
        it needs to be removed
        :return: np.ndarray
        """
        return self.features[:, 1:]

    def get_labels(self):
        return self.labels

    @staticmethod
    def add_ones_to_matrix(features: np.ndarray) -> np.ndarray:
        """
        add ones to a matrix for calculating with constants
        :param features: np.ndarray
        :return features with ones at beginning of each row
        """
        try:
            features.shape[1]
            empty_vector = np.ones((features.shape[0], 1))
            features = np.concatenate([empty_vector, features], axis=1)
        except IndexError:
            features = np.stack([np.ones(features.shape[0]), features], axis=1)

        return features

    @staticmethod
    def precondition(features: np.ndarray, labels: np.ndarray):
        if features.shape[0] != labels.shape[0]:
            raise ValueError("The number of features is not the same as the number of labels")
        # TODO: add condition if labels is not a vector

    @abstractmethod
    def plot_data(self):
        """
        plot features and labels on the graph
        """
        pass

    @abstractmethod
    def cost_function(self, theta: np.ndarray):
        """
        cost function
        :return: np.array()
        """
        pass

    @abstractmethod
    def gradient(self, theta: np.ndarray, alpha: float):
        """
        cost function
        :return: np.array()
        """
        pass

    @abstractmethod
    def minimize_cost(self, theta: np.ndarray):
        """
        minimize the cost function using gradient descent algorithm
        :return: np.array()
        """
        pass

    @abstractmethod
    def predict(self, theta: np.ndarray):
        """
        minimize the cost function using gradient descent algorithm
        :return: np.array()
        """
        pass

