from unittest import TestCase
from abstract import AbstractRegression
import numpy as np


class TestAbstractRegression(TestCase):
    def test_add_ones_to_matrix(self):
        input_vector = np.array([[1], [2], [3]])
        output_vector = np.array([[1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(AbstractRegression.add_ones_to_matrix(input_vector), output_vector)

        # # TODO: this should  not raise an error
        input_vector2 = np.array([1, 2, 7])
        np.testing.assert_raises(ValueError, AbstractRegression.add_ones_to_matrix, input_vector2)

        input_matrix = np.array([[1, 3], [2, 6], [3, 5]])
        output_matrix = np.array([[1, 1, 3], [1, 2, 6], [1, 3, 5]])
        np.testing.assert_array_equal(AbstractRegression.add_ones_to_matrix(input_matrix), output_matrix)

    def test_precondition(self):
        self.assertRaises(ValueError, AbstractRegression.precondition, np.array([[1], [2]]), np.array([[1], [2], [3]]))
        # self.assertRaises(Warning, AbstractRegression.precondition, np.array([[1], [2]]), np.array([[1, 2], [2, 2]]))
