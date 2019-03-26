from forcelayout import forcelayout
from .data.data import load_iris

dataset = load_iris()


def mock_distance_fn(x, y):
    return 1


def test_create_params_correctly_ignores_none_values():
    expected = dict()
    params = forcelayout._create_params(a=None, b=None, c=None)
    assert params == expected


def test_create_params_adds_defined_values():
    expected = dict(a=1, b="hello", c=[1, 2, 3])
    params = forcelayout._create_params(a=1, b="hello", c=[1, 2, 3])
    assert params == expected


def test_create_params_ignores_none_and_adds_correct_values():
    expected = dict(a=1, b="hello", c=[1, 2, 3])
    params = forcelayout._create_params(a=1, z=None, b="hello", c=[1, 2, 3], y=None)
    assert params == expected


def test_create_algorithm_creates_spring_force_algorithm_correctly_no_params():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.SpringForce)

    assert type(algorithm) is forcelayout.SpringForce
    assert len(algorithm.nodes) == len(dataset)


def test_create_algorithm_creates_spring_force_algorithm_correctly():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.SpringForce,
                                              iterations=10,
                                              distance=mock_distance_fn)
    assert type(algorithm) is forcelayout.SpringForce
    assert len(algorithm.nodes) == len(dataset)
    assert algorithm.iterations == 10
    assert algorithm.distance_fn is mock_distance_fn


def test_create_algorithm_creates_neighbour_sampling_algorithm_correctly_no_params():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.NeighbourSampling)

    assert type(algorithm) is forcelayout.NeighbourSampling
    assert len(algorithm.nodes) == len(dataset)


def test_create_algorithm_creates_neighbour_sampling_algorithm_correctly():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.NeighbourSampling,
                                              iterations=10,
                                              neighbour_set_size=3,
                                              sample_set_size=2,
                                              distance=mock_distance_fn)
    assert type(algorithm) is forcelayout.NeighbourSampling
    assert len(algorithm.nodes) == len(dataset)
    assert algorithm.iterations == 10
    assert algorithm.neighbour_set_size == 3
    assert algorithm.sample_set_size == 2
    assert algorithm.distance_fn is mock_distance_fn


def test_create_algorithm_creates_hybrid_algorithm_correctly_no_params():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.Hybrid)
    assert type(algorithm) is forcelayout.Hybrid
    assert len(algorithm.nodes) == len(dataset)


def test_create_algorithm_creates_hybrid_algorithm_correctly():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.Hybrid,
                                              iterations=10,
                                              hybrid_remainder_layout_iterations=11,
                                              hybrid_refine_layout_iterations=12,
                                              sample_set_size=3)
    assert type(algorithm) is forcelayout.Hybrid
    assert len(algorithm.nodes) == len(dataset)
    assert algorithm.sample_layout_iterations == 10
    assert algorithm.remainder_layout_iterations == 11
    assert algorithm.refine_layout_iterations == 12
    assert algorithm.random_sample_size == 3


def test_create_algorithm_creates_pivot_algorithm_correctly_no_params():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.Pivot)
    assert type(algorithm) is forcelayout.Pivot
    assert len(algorithm.nodes) == len(dataset)


def test_create_algorithm_creates_pivot_algorithm_correctly():
    algorithm = forcelayout._create_algorithm(dataset=dataset,
                                              algorithm=forcelayout.Pivot,
                                              iterations=10,
                                              hybrid_remainder_layout_iterations=11,
                                              hybrid_refine_layout_iterations=12,
                                              sample_set_size=4)
    assert type(algorithm) is forcelayout.Pivot
    assert len(algorithm.nodes) == len(dataset)
    assert algorithm.sample_layout_iterations == 10
    assert algorithm.remainder_layout_iterations == 11
    assert algorithm.refine_layout_iterations == 12
    assert algorithm.random_sample_size == 4
