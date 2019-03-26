# Force Directed Layout Algorithms for Python
Iain Cattermole 2189570c

## Setup

Before this package can be used, dependencies must be installed.

```bash
python3 -m pip install forcelayout
```

or

```bash
python3 -m pip install -r requirements.txt
```

### Submission Structure

The implementation of the algorithms and functionality described in the dissertation:

```

```

Examples of usage of the algorithms are included in the `examples` folder.

### Examples

Examples are contained in the `examples/` folder. Three command line scripts are included and a jupyter notebook.

##### `examples/poker_hands_layout.py`

This script will create a static layout of the poker hands dataset rendering using matplotlib. This shows the colouring functionality and annotations for each node when hovered. Nodes can be clicked to highlight other nodes based on the algorithm:
- Chalmers' 1996: Highlight the nodes in the neighbour set for the selected node.
- Hybrid: Highlight the nodes in the sample layout
- Pivot: Highlight the pivot nodes

The size of dataset can be changed to any available size in `datasets/poker/` and the algorithms available are: `brute`, `chalmers96`, `hybrid` and `pivot`.

Run it using the following commands:
```bash
cd examples/
# python3 poker_hands_layout.py <dataset size> <algorithm>
python3 poker_hands_layout.py 500 pivot
```

##### `examples/animated_poker_hands_layout.py`

This is similar to the above example but will show an animated version of the layout.

```bash
cd examples/
# python3 poker_hands_layout.py <dataset size> <algorithm>
python3 animated_poker_hands_layout.py 500 pivot
```

##### `examples/iris_layout.py`

This will show an animated layout of the iris dataset using Chalmers' 1996 algorithm, also an image will be saved of the final layout.

```bash
cd examples/
python3 iris_layout.py
```


### Usage

Documentation on full usage of the `forcelayout` package is available in the `forcelayout/README.md` [here](forcelayout/README.md)

### Testing

In order to run tests, `pytest` is required

```bash
python3 -m pip install pytest
```

To run the test cases run the following command from the root directory:

```bash
python3 -m pytest forcelayout/ tests/forcelayout
```
