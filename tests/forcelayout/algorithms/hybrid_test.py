from forcelayout import Hybrid
from numpy.testing import assert_almost_equal
import numpy as np
import math

dataset = np.array([
    [0, 1],
    [0, 0],
    [1, 0],
    [2, 0],
    [1, 1],
])


def set_node_positions(algorithm):
    for node in algorithm.nodes:
        node.x, node.y = node.datapoint


def test_find_parent_gets_the_parent_with_minimum_distance():
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=[1, 2, 3])
    test_node = algorithm.nodes[0]
    parent_index = algorithm._find_parent(test_node)[0]

    assert parent_index == 0  # sample node 0 is [0, 0]


def test_create_error_fn_returns_function_that_returns_expected():
    sample_indexes = [1, 2, 3]
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=sample_indexes,
                       random_sample_size=1)
    set_node_positions(algorithm)

    source = algorithm.nodes[0]
    distances = [algorithm.distance(source, algorithm.nodes[t]) for t in sample_indexes]
    error_fn = algorithm._create_error_fn(0, distances)

    assert error_fn(90) == 0
    assert error_fn(270) == 0  # sample is 3 nodes in a line so either side works
    assert error_fn(180) == 6 - sum(distances)
    assert error_fn(0) == sum(distances) - 2


def test_find_circle_quadrant_finds_expected_angles():
    sample_indexes = [1, 2, 3]
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=sample_indexes,
                       random_sample_size=1)
    set_node_positions(algorithm)

    # source: [0, 1] with parent [0, 0]
    source = algorithm.nodes[0]
    distances = [algorithm.distance(source, algorithm.nodes[t]) for t in sample_indexes]
    error_fn = algorithm._create_error_fn(0, distances)

    lower_angle, upper_angle = algorithm._find_circle_quadrant(error_fn)
    assert lower_angle == 90
    assert upper_angle == 180

    # source: [1, 1] with parent [0, 0]
    source = algorithm.nodes[4]
    distances = [algorithm.distance(source, algorithm.nodes[t]) for t in sample_indexes]
    error_fn = algorithm._create_error_fn(0, distances)

    lower_angle, upper_angle = algorithm._find_circle_quadrant(error_fn)
    assert lower_angle == 0
    assert upper_angle == 90


def test_binary_search_angle_finds_best_angle():
    sample_indexes = [1, 2, 3]
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=sample_indexes,
                       random_sample_size=1)
    set_node_positions(algorithm)

    # source: [0, 1] with parent [0, 0]
    source = algorithm.nodes[0]
    distances = [algorithm.distance(source, algorithm.nodes[t]) for t in sample_indexes]
    error_fn = algorithm._create_error_fn(0, distances)
    lower_angle, upper_angle = (90, 180)

    best_angle = algorithm._binary_search_angle(lower_angle, upper_angle, error_fn)
    assert abs(best_angle - 90) <= 4  # must be within 4 degrees of best angle

    # source: [1, 1] with parent [0, 0]
    source = algorithm.nodes[4]
    distances = [algorithm.distance(source, algorithm.nodes[t]) for t in sample_indexes]
    error_fn = algorithm._create_error_fn(0, distances)
    lower_angle, upper_angle = (0, 90)

    best_angle = algorithm._binary_search_angle(lower_angle, upper_angle, error_fn)
    assert abs(best_angle - 45) <= 4  # must be within 4 degrees of best angle


def test_sample_distances_sum_returns_correct_sum():
    sample_indexes = [1, 2, 3]
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=sample_indexes,
                       random_sample_size=1)
    set_node_positions(algorithm)

    assert algorithm._sample_distances_sum(3, 0) == 6
    assert algorithm._sample_distances_sum(-1, 0) == 6
    assert algorithm._sample_distances_sum(1, 0) == 2
    one_one_distance = 1 + 2 * math.sqrt(2)
    assert_almost_equal(algorithm._sample_distances_sum(1, 1), one_one_distance)
