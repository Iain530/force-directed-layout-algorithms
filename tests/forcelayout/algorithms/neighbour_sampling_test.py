from forcelayout import NeighbourSampling
from ..data.data import load_iris

dataset = load_iris()


def test_get_neighbours_returns_correct_size_set():
    algorithm = NeighbourSampling(dataset=dataset, neighbour_set_size=5)

    neighbour_set = algorithm._get_neighbours(0)
    assert len(neighbour_set) == 5


def test_get_neighbours_saves_neighbours_after_first_call():
    algorithm = NeighbourSampling(dataset)
    assert 0 not in algorithm.neighbours
    algorithm._get_neighbours(0)
    assert 0 in algorithm.neighbours


def test_get_neighbours_returns_sorted_list_of_neighbours():
    algorithm = NeighbourSampling(dataset)
    neighbour_set = algorithm._get_neighbours(0)

    source = algorithm.nodes[0]
    previous = -1
    for i in neighbour_set:
        distance = algorithm.distance(source, algorithm.nodes[i])
        assert distance >= previous
        previous = distance


def test_update_neighbours_doesnt_change_neighbour_set_size():
    algorithm = NeighbourSampling(dataset,
                                  neighbour_set_size=3,
                                  sample_set_size=3)
    for i in range(10):
        sample_set = algorithm._get_sample_set(0)
        algorithm._update_neighbours(0, sample_set)
        neighbour_set = algorithm._get_neighbours(0)
        assert len(neighbour_set) == 3


def test_update_neighbours_keeps_sorted_neighbour_set():
    algorithm = NeighbourSampling(dataset,
                                  neighbour_set_size=5,
                                  sample_set_size=1)
    source = algorithm.nodes[0]

    for i in range(10):
        sample_set = algorithm._get_sample_set(0)
        algorithm._update_neighbours(0, sample_set)
        neighbour_set = algorithm._get_neighbours(0)

        # Check neighbour set is sorted
        current_distance = -1
        for j in neighbour_set:
            target = algorithm.nodes[j]
            distance = algorithm.distance(source, target)
            assert distance >= current_distance
            current_distance = distance


def test_get_sample_set_excludes_neighbours_and_current_node():
    algorithm = NeighbourSampling(dataset,
                                  neighbour_set_size=3,
                                  sample_set_size=3)
    for i in range(10):
        sample_set = set(algorithm._get_sample_set(0))
        neighbour_set = set(algorithm._get_neighbours(0))
        excluded = {0}.union(neighbour_set)
        empty_set = set()
        assert sample_set.intersection(excluded) == empty_set
