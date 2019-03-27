# Force Directed Layout Algorithms for Python

This package provides 3 force-directed layout algorithms implemented in Python and a brute force implementation.

- [Chalmers' 1996 algorithm](https://ieeexplore.ieee.org/document/567787) is implemented in `algorithms.neighbour_sampling`
- [Hybrid Layout algorithm](https://ieeexplore.ieee.org/document/1173161) is implemented in `algorithms.hybrid`
- [Pivot Layout algorithm](https://ieeexplore.ieee.org/document/1249012) is implemented in `algorithms.pivot`

## Structure

```
forcelayout
├── algorithms
│   ├── base_spring_layout.py
│   ├── hybrid.py
│   ├── neighbour_sampling.py
│   ├── node.py
│   ├── pivot.py
│   ├── spring_force.py
│   └── __init__.py
├── distance.py
├── draw.py
├── forcelayout.py
├── metrics.py
├── utils.py
└── __init__.py
```

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

More complex examples can be seen in the `examples/` folder.

# Documentation

Throughout this documentation these notations are used:
- `<parameter name>: <type> = <default value>`
- `<function name>() -> <return type>`

Also it is assumed that the package is import by:

```
import forcelayout as fl
```

## Functions

#### `fl.draw_spring_layout() -> [SpringForce | NeighbourSampling | Hybrid | Pivot ]`:
Draw a spring layout diagram of the dataset. Returns the instance of the algorithm.

- `dataset: np.ndarray`: 2D array of the data to lay out
- `algorithm: fl.BaseSpringLayout = fl.NeighbourSampling`: The algorithm class to draw the spring layout diagram. One of the following:
    - `fl.SpringForce`
    - `fl.NeighbourSampling`
    - `fl.Hybrid`
    - `fl.Pivot`
- `iterations: int = (SpringForce=50 | NeighbourSampling=50 | Hybrid=75 | Pivot=75)`: Number of force iterations to perform on the layout. When using the Hybrid or Pivot algorithm this is the number of force iterations performed on the sample layout.
- `hybrid_remainder_layout_iterations: int = 5`: Number of force iterations to perform on each child of the remainder subset when using the Hybrid or Pivot algorithm.
- `hybrid_refine_layout_iterations: int = 5`: Number of force iterations to perform on the whole layout after the remainder layout stage.
- `pivot_set_size: int = 10`: Number of pivot nodes if `algorithm=fl.Pivot`.
- `sample_set_size: int = 10`: Size of random sample set.
- `neighbour_set_size: int = 5`: Size of neighbour set for neighbour sampling stages.
- `distance: Callable[[np.ndarray, np.ndarray], float] = fl.distances.euclidean`: Function to determine the high-dimensional distance between two datapoints  of shape `(1, D)` and return a float representing the dissimilarity.
- `alpha: float = 1.0`: Opacity of nodes in the diagram in range `[0-1]`.
- `size: float = 40`: Size of node to draw passed to matplotlib.pyplot.scatter(s=size).
- `color_by: Callable[[np.ndarray], float] = None`: Function to colour nodes by converting a `(1, D)` datapoint to a float to be expressed through the `color_map`.
- `color_map: str = 'viridis'`: String name or matplotlib colour map to use with color_by function.
- `annotate: Callable[[Node, int], str] = None`: Callback function for annotating nodes when hovered, given the node being hovered and the index in dataset.
- `algorithm_highlights: bool = False`: Allow on click of data point to highlight important nodes for the algorithm.
    - `fl.NeighbourSampling`: Neighbour nodes to the selected node.
    - `fl.Hybrid`: Sample set nodes.
    - `fl.Pivot`: Pivot nodes.

#### `fl.draw_spring_layout_animated() -> matplotlib.animation.FuncAnimation`:
Draw a spring layout diagram of the data using the given algorithm. Returns a `matplotlib.animation.FuncAnimation`. Call `plt.show()` to view the animation. Parameters are the same as `fl.draw_spring_layout()` with one extra:
- `interval: int = 10`: Milliseconds between each frame. If this is set low it will be limited by the run time of the algorithm.

#### `fl.spring_layout() -> [SpringForce | NeighbourSampling | Hybrid | Pivot]`:
Create an instance of a spring layout algorithm without running it. Uses the same parameters as `fl.draw_spring_layout` without ones specific to drawing.


## Algorithm Interface

```
from forcelayout.algorithms import (BaseSpringLayout,
                                    SpringForce,
                                    NeighbourSampling,
                                    Hybrid,
                                    Pivot)
```

### `BaseSpringLayout`:

This is the spring layout class that all algorithms extend and provides the main interface for them. This is an abstract class and cannot be instantiated however these functions are available on all algorithms.

#### `.spring_layout() -> np.ndarray`:

This will run the algorithm until the maximum number of iterations is reach or the `target_node_speed` is reached if set. Returns a `(N, 2)` array of final the positions for each point in `self.dataset` where the values are `(x, y)`.

- `return_after: int = None`: if this is set then the algorithm will return after the set number of iterations however will not run past the maximum set on `self.iterations`.

#### `.get_positions() -> np.ndarray`:

Returns a `(N, 2)` array of the current positions for each point in `self.dataset` where the values are `(x, y)`.

#### `.set_positions() -> None`:

Sets the positions of each node given an `(N, 2)` array.

- `positions: np.ndarray`: the `(N, 2)` array required containing the positions for nodes.

#### `.get_stress() -> float`:

Returns a float representing the current stress of the layout. The stress tends to 0 as the layout quality increases. This is an `O(N^2)` time complexity and is very expensive on large datasets.

#### `.average_speed() -> float`:

An approximation metric for the stress of the layout. Runs in `O(N)`.

#### `.get_memory() -> int`:

Returns the current memory usage of the algorithm.

#### `.current_iteration() -> int`:

Returns the current iteration number of the algorithm.

## Drawing interface

```
from forcelayout.draw import DrawLayout
```

### `DrawLayout`:

This class contains the functionality for rendering layouts using matplotlib.

#### `.__init__()`:
- `dataset: np.ndarray`: dataset used by the layout instance.
- `layout: BaseSpringLayout`: BaseSpringLayout instance.

#### `.draw() -> None`:

Draw the layout created.

- `alpha: float = 1.0`: Opacity of nodes in the diagram in range `[0-1]`.
- `size: float = 40`: Size of node to draw passed to matplotlib.pyplot.scatter(s=size).
- `color_by: Callable[[np.ndarray], float] = None`: Function to colour nodes by converting a `(1, D)` datapoint to a float to be expressed through the `color_map`.
- `color_map: str = 'viridis'`: String name or matplotlib colour map to use with color_by function.
- `annotate: Callable[[Node, int], str] = None`: Callback function for annotating nodes when hovered, given the node being hovered and the index in dataset.
- `algorithm_highlights: bool = False`: Allow on click of data point to highlight important nodes for the algorithm.
    - `fl.NeighbourSampling`: Neighbour nodes to the selected node.
    - `fl.Hybrid`: Sample set nodes.
    - `fl.Pivot`: Pivot nodes.

#### `.draw_animated() -> matplotlib.animation.FuncAnimation`:

Draw a spring layout diagram of the data using the given algorithm. Returns a `matplotlib.animation.FuncAnimation`. Call `plt.show()` to view the animation. All parameters on `.draw()` are available.

- `interval: int = 10`: Milliseconds between each frame. If this is set low it will be limited by the run time of the algorithm.
- `draw_every: int = 1`: Number of iterations between frames.

#### `.save() -> None`:

Save the current layout. Wraps `matplotlib.pyplot.savefig()` and all `*args` and `**kwargs` are passed on.

- `path: str`: path to save the image to


## Metrics Interface

```
from forcelayout.metrics import LayoutMetrics
```

### `LayoutMetrics`:

Class containing the performance metrics of the algorithms.

#### `.__init__()`

- `spring_layout`: BaseSpringLayout instance to measure the metrics of.

#### `.measure() -> None`:

Run the algorithm and collect metrics.

#### `.get_stress() -> float`:

Get the current stress of the layout

#### `get_avg_speed() -> float`:

Get the average speed of the algorithm at a given iteration.

- `i: int`: iteration number to get the average speed at.

#### `.get_memory() -> int`:

Get the memory usage of the algorithm.

- `i: int`: iteration to check the memory of.

#### `.get_max_memory() -> int`:

Get the maximum memory usage of the algorithm.

#### `.get_average_iteration_time() -> float`

Get the mean time per iteration of the algorithm.

#### `.get_csv_header() -> str`:

Get a header of the form `"N","Total time",...` that corresponds to `.get_csv()` and `.to_list`.

#### `.to_list() -> List[int | float]`:

Return a list of the metrics measured in the order specifies by `.get_csv_header()`

#### `.to_csv() -> str`:

Returns a string representation of `.to_list()` separated by commas.
