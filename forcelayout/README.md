# Force Directed Layout Algorithms for Python

This package provides 3 force directed layout algorithms implemented in Python and a brute force implementation.

- [Chalmers' 1996 algorithm](https://ieeexplore.ieee.org/document/567787) is implemented in `algorithms.neighbour_sampling`
- [Hybrid Layout algorithm](https://ieeexplore.ieee.org/document/1173161) is implemented in `algorithms.hybrid`
- [Pivot Layout algorithm](https://ieeexplore.ieee.org/document/1249012) is implemented in `algorithms.pivot`

## Basic Usage

These algorithms are designed for the visualisation of high-dimensional datasets using [force-directed graph drawing methods](https://en.wikipedia.org/wiki/Force-directed_graph_drawing). Datasets are expected to be a numpy array with shape `(N, D)` where `N` is the size of the dataset and `D` is the number of dimensions.


```
# other imports
import matplotlib.pyplot as plt
import numpy as np

# import the package
import forcelayout as fl

# Need a dataset to visualise
dataset = np.array([[1, 1],
                    [1, 3],
                    [2, 2]])

# Need to use the brute force algorithm on a dataset this small
# (not recommended for larger datasets)
layout = fl.draw_spring_layout(dataset=dataset, algorithm=fl.SpringForce)

plt.show()
```

## Functions

### `fl.draw_spring_layout()`:
Draw a spring layout diagram of the dataset. Returns the instance of the algorithm.

- `dataset: np.ndarray`: 2d array of the data to lay out
- `algorithm: fl.BaseSpringLayout`: The algorithm class to draw the spring layout diagram. One of the following:
    - `fl.SpringForce`
    - `fl.NeighbourSampling`
    - `fl.Hybrid`
    - `fl.Pivot`
- `iterations: int`: number of force iterations to perform on the layout (when using the Hybrid algorithm this is the number of force iterations performed on the original sample).
- `hybrid_remainder_layout_iterations: int`: number of force iterations to perform on each child of the remainder subset when using the Hybrid or Pivot algorithm.
- `hybrid_refine_layout_iterations: int`: number of force iterations to perform on the whole layout after the remainder layout stage.
- `pivot_set_size: int`: number of pivot nodes if `algorithm=fl.Pivot`.
- `sample_set_size: int`: size of random sample set.
- `neighbour_set_size: int`: size of neighbour set for neighbour sampling stages.
- `distance: Callable[[np.ndarray, np.ndarray], float]`: function to determine the high dimensional distance between two datapoints  of shape `(1, D)` and return a float representing the dissimilarity.
- `alpha: float`: opacity of nodes in the diagram in range `[0-1]`.
- `size: float`: size of node to draw passed to matplotlib.pyplot.scatter(s=size).
- `color_by: Callable[[np.ndarray], float]`: function to colour nodes by converting a `(1, D)` datapoint to a float to be expressed through the `color_map`.
- `color_map: str`: string name or matplotlib colour map to use with color_by function.
- `annotate: Callable[[Node, int], str]`: callback function for annotating nodes when hovered, given the node being hovered and the index in dataset.
- `algorithm_highlights: bool`: allow on click of data point to highlight important nodes for the algorithm.
    - `fl.NeighbourSampling`: neighbour nodes.
    - `fl.Hybrid`: sample set nodes.
    - `fl.Pivot`: pivot nodes.

### `fl.draw_spring_layout_animated()`:
Draw a spring layout diagram of the data using the given algorithm. Returns a `matplotlib.animation.FuncAnimation`. Call `plt.show()` to view the animation. Parameters are the same as `fl.draw_spring_layout()` with one extra:
- `interval: int`: milliseconds between each frame. If this is set low it will be limited by the run time of the algorithm.

### `fl.spring_layout()`:
Create an instance of a spring layout algorithm without running it. Uses the same parameters as `fl.draw_spring_layout` without ones specific to drawing.
