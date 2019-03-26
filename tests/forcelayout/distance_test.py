from forcelayout.distance import euclidean
import numpy as np
import math

mock_int_vector1 = np.array([1, 2, 3, 4, 5])
mock_int_vector2 = np.array([6, 7, 8, 9, 10])

mock_float_vector1 = np.array([0.3, 0.4, 0.5])
mock_float_vector2 = np.array([0.6, 0.7, 0.8])


def test_euclidean_distance_returns_expected_with_positive_int():
    expected = math.sqrt(125)
    actual = euclidean(mock_int_vector1, mock_int_vector2)
    assert actual == expected


def test_euclidean_distance_returns_expected_with_negative_int():
    expected = math.sqrt(125)
    actual = euclidean(-mock_int_vector1, -mock_int_vector2)
    assert actual == expected


def test_euclidean_distance_returns_expected_with_positive_float():
    expected = math.sqrt(0.27)
    actual = euclidean(mock_float_vector1, mock_float_vector2)
    np.testing.assert_almost_equal(actual, expected)


def test_euclidean_distance_returns_expected_with_negative_float():
    expected = math.sqrt(0.27)
    actual = euclidean(mock_float_vector1, mock_float_vector2)
    np.testing.assert_almost_equal(actual, expected)
