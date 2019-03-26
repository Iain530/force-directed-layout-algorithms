from forcelayout import Pivot
import numpy as np

dataset = np.array([
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [3, 1],
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 2],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
])


def set_node_positions(algorithm):
    for node in algorithm.nodes:
        node.x, node.y = node.datapoint


def test_find_parent_gets_a_parent_close_to_minimum_distance():
    algorithm = Pivot(dataset=dataset,
                      preset_sample=[0, 5, 10, 15])
    for pivot in algorithm.pivots:
        algorithm._preprocess_pivot(pivot)
    print(algorithm.pivots)
    print(algorithm.pivot_buckets)
    source = algorithm.nodes[2]  # [2, 0]
    parent = algorithm._find_parent(source)
    assert list(parent.datapoint) not in ([2, 2], [3, 3])  # can be [0, 0] or [1, 1]


def test_create_error_fn_returns_function_that_returns_expected():
    sample_indexes = [0, 5, 10, 15]
    algorithm = Pivot(dataset=dataset,
                      preset_sample=sample_indexes,
                      random_sample_size=1)
    set_node_positions(algorithm)
    source = algorithm.nodes[4]  # [0, 1]
    distances = [algorithm.distance(source, target) for target in algorithm.pivots]
    distance_sum = sum(distances)
    parent = algorithm.nodes[5]
    radius = algorithm.distance(source, parent)
    error_fn = algorithm._create_error_fn(parent, radius, distance_sum)

    assert error_fn(180) == 0


def test_pivot_distances_sum_returns_correct_sum():
    algorithm = Pivot(dataset=dataset)
    set_node_positions(algorithm)
    source = algorithm.nodes[3]
    assert algorithm._pivot_distances_sum(3, 0) == sum(
        [algorithm.distance(source, p) for p in algorithm.pivots]
    )
