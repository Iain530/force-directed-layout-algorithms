from forcelayout.algorithms import Node
import numpy as np

mock_datapoint = np.array([1, 2, 3, 4, 5])


def test_add_velocity_sums_correctly():
    node = Node(mock_datapoint, vx=0, vy=0)

    node.add_velocity(10, 20)
    assert node.vx == 10 and node.vy == 20
    node.add_velocity(5, 10)
    assert node.vx == 15 and node.vy == 30


def test_apply_velocity_moves_node_correctly():
    node = Node(mock_datapoint, x=0, y=0, vx=10, vy=20)
    node.apply_velocity()

    assert node.x == 10 and node.y == 20


def test_apply_velocity_clears_velocity_after_moving_node():
    node = Node(mock_datapoint, vx=10, vy=10)
    node.apply_velocity()

    assert node.vx == 0 and node.vy == 0
