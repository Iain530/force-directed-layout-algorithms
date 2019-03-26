import forcelayout as fl
import matplotlib.pyplot as plt
import numpy as np

iris_data = None
with open(f'../datasets/iris/iris.csv', encoding='utf8') as iris_file:
    iris_data = np.loadtxt(
        iris_file,
        skiprows=1,
        usecols=(0, 1, 2, 3),
        delimiter=',',
        comments='#'
    )

iris_classes = None
with open(f'../datasets/iris/iris.csv', encoding='utf8') as iris_file:
    iris_classes = np.loadtxt(
        iris_file,
        skiprows=1,
        usecols=(4,),
        delimiter=',',
        dtype='str',
        comments='#'
    )

colors = dict()
ind = 0
col = 0


def classed_iris(iris):
    global ind, col
    if iris_classes[ind] not in colors:
        colors[iris_classes[ind]] = col
        col += 1
    class_col = colors[iris_classes[ind]]
    ind += 1
    return class_col


ani = fl.draw_spring_layout_animated(dataset=iris_data,
                                     algorithm=fl.NeighbourSampling,
                                     alpha=0.5,
                                     algorithm_highlights=True,
                                     color_by=classed_iris)

plt.savefig('iris.png')

plt.show()
