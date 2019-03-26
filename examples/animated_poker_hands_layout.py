from poker_utils import load_poker, poker_distance, annotate_poker
import matplotlib.pyplot as plt
import forcelayout as fl
import time
import sys

if len(sys.argv) < 2:
    print(f'\nusage: python3 animated_poker_hands_layout.py <dataset size> <algorithm>')
    print('\tSizes: see datasets/poker')
    print('\tAvailable algorithms: brute, chalmers96, hybrid, pivot')
    exit(1)


data_set_size = int(sys.argv[1])
poker_hands = load_poker(data_set_size)
algorithm = sys.argv[2].lower() if len(sys.argv) > 2 else 'pivot'
print(f"Creating Layout of {len(poker_hands)} poker hands using {algorithm} algorithm")

algorithm = {
    'brute': fl.SpringForce,
    'chalmers96': fl.NeighbourSampling,
    'hybrid': fl.Hybrid,
    'pivot': fl.Pivot,
}[algorithm]

plt.figure(figsize=(8.0, 8.0))
ani = fl.draw_spring_layout_animated(dataset=poker_hands,
                                     distance=poker_distance,
                                     algorithm=algorithm,
                                     alpha=0.7,
                                     color_by=lambda d: d[10],
                                     annotate=annotate_poker,
                                     algorithm_highlights=True)
plt.show()
