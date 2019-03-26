from forcelayout.utils import (jiggle, random_sample_set, flexible_intersection, sample,
                               point_on_circle, get_size)
from numpy.testing import assert_almost_equal
from sys import getsizeof
import random
import math


def test_jiggle_returns_100_numbers_within_boundary():
    for i in range(100):
        test_value = jiggle()
        assert -1e-6 <= test_value <= 1e-6
        assert test_value != 0


def test_random_sample_set_returns_100_expected_ranges():
    limit = 100
    set_size = 10
    for i in range(100):
        exclude = {random.randint(0, limit-1) for i in range(5)}
        result = random_sample_set(set_size, limit, exclude.copy())
        assert len(result) == set_size
        assert all(map(lambda x: x not in exclude, result))
        assert all(map(lambda x: 0 <= x < limit, result))


def test_sample_returns_100_expected_numbers():
    limit = 10
    for i in range(100):
        # set of up to 5 numbers to exclude
        exclude = {random.randint(0, limit-1) for i in range(5)}
        result = sample(limit, exclude)
        assert 0 <= result < limit
        assert result not in exclude


def test_point_on_circle_gives_correct_points_at_90_degree_angles():
    expected = [
        (2, 0),
        (0, 2),
        (-2, 0),
        (0, -2),
        (2, 0),
    ]

    radius = 2
    angles = [0, 90, 180, 270, 360]
    result = list(map(lambda a: point_on_circle(0, 0, a, radius), angles))

    assert_almost_equal(result, expected)


def test_point_on_circle_gives_correct_points_at_45_degree_angles():
    expected = [
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ]

    radius = math.sqrt(2)
    angles = [45, 135, 225, 315]
    result = list(map(lambda a: point_on_circle(0, 0, a, radius), angles))

    assert_almost_equal(result, expected)


def test_flexible_intersection_handles_normal_intersections():
    expected = {1, 2}
    test_sets = [
        {0, 1, 2, 3},
        {0, 1, 2},
        {1, 2, 3},
    ]
    result = flexible_intersection(*test_sets)
    assert result == expected


def test_flexible_intersection_always_returns_most_common_intersection():
    expected = {1, 2, 3}
    test_sets = [
        {1, 2, 3},
        {1},
        {2},
        {3},
    ]
    result = flexible_intersection(*test_sets)
    assert result == expected


def list_size(size):
    return getsizeof([None] * size)


def test_get_size_returns_expected_size_for_basic_values():
    basic_values = [1, 'hello', set(), dict()]
    for basic in basic_values:
        assert get_size(basic) == getsizeof(basic)


def test_get_size_doesnt_count_same_instances_multiple_times():
    instance = dict()
    multiple_references = [instance] * 5

    # Only expect the size of one instance and list overhead
    # for length 5
    expected_size = list_size(5) + getsizeof(instance)
    assert get_size(multiple_references) == expected_size
