from forcelayout.algorithms import SpringForce
from numpy.testing import assert_almost_equal
from ..data.data import load_iris
import numpy as np

dataset = load_iris()


def test_calling_distance_caches_result():
    algorithm = SpringForce(dataset=dataset, enable_cache=True)

    source = algorithm.nodes[0]
    target = algorithm.nodes[1]
    key = frozenset({source, target})

    algorithm.distance(source, target, cache=False)
    assert key not in algorithm.distances

    distance = algorithm.distance(source, target, cache=True)
    assert key in algorithm.distances
    assert algorithm.distances[key] == distance


def test_calling_distance_doesnt_cache_result_when_cache_disabled():
    algorithm = SpringForce(dataset=dataset, enable_cache=False)

    source = algorithm.nodes[0]
    target = algorithm.nodes[1]
    algorithm.distance(source, target, cache=True)
    assert not hasattr(algorithm, 'distances')


def test_spring_layout_return_after_performs_correct_number_of_iterations():
    algorithm = SpringForce(dataset=dataset)

    for i in range(2, 10, 2):
        algorithm.spring_layout(return_after=2)
        assert algorithm.current_iteration() == i


def test_spring_layout_performs_all_iterations():
    algorithm = SpringForce(dataset=dataset, iterations=20)
    algorithm.spring_layout()
    assert algorithm.current_iteration() == 20


def test_spring_force_builds_nodes_correctly():
    spring_force = SpringForce(dataset=dataset)
    assert np.array_equal(spring_force.nodes[0].datapoint, dataset[0, :])
    assert np.array_equal(spring_force.nodes[1].datapoint, dataset[1, :])


def test_get_stress_returns_expected_stress_with_perfect_layout():
    # Data that should result in 0 stress layout
    # with euclidean distance
    mock_dataset = np.array([
        [0, 0],
        [0, 1],
    ])

    algorithm = SpringForce(dataset=mock_dataset, iterations=100)
    algorithm.spring_layout()

    expected_stress = 0
    assert_almost_equal(algorithm.get_stress(), expected_stress)


def test_get_stress_returns_expected_stress_with_bad_layout():
    mock_dataset = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    # Distance function that does not allow for a zero
    # stress layout with given data
    def bad_distance(a, b):
        return 1

    algorithm = SpringForce(dataset=mock_dataset, distance_fn=bad_distance, iterations=100)
    algorithm.spring_layout()

    # precalculated nearly optimal stress for this case
    expected_stress = 0.1715728
    assert_almost_equal(algorithm.get_stress(), expected_stress)
