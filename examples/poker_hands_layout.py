from poker_utils import load_poker, poker_distance, annotate_poker
import matplotlib.pyplot as plt
import forcelayout as fl
import time
import sys

algorithms = {
    'brute': fl.SpringForce,
    'chalmers96': fl.NeighbourSampling,
    'hybrid': fl.Hybrid,
    'pivot': fl.Pivot,
}

if len(sys.argv) < 2:
    print(f'\nusage: python3 poker_hands_layout.py <dataset size> <algorithm>')
    print('\tSizes: see datasets/poker')
    print('\tAvailable algorithms: brute, chalmers96, hybrid, pivot')
    exit(1)

if len(sys.argv) > 2 and sys.argv[2] not in algorithms:
    print('\tAvailable algorithms: brute, chalmers96, hybrid, pivot')
    exit(1)

data_set_size = int(sys.argv[1])
poker_hands = load_poker(data_set_size)
algorithm = sys.argv[2].lower() if len(sys.argv) > 2 else 'pivot'
print(f"Creating Layout of {len(poker_hands)} poker hands using {algorithm} algorithm")

algorithm = algorithms[algorithm]

plt.figure(figsize=(8.0, 8.0))

start = time.time()

spring_layout = fl.draw_spring_layout(dataset=poker_hands,
                                      distance=poker_distance,
                                      algorithm=algorithm,
                                      alpha=0.7,
                                      color_by=lambda d: d[10],
                                      annotate=annotate_poker,
                                      algorithm_highlights=True)

total = time.time() - start
print(f'\nLayout time: {"%.2f" % total}s ({"%.1f" % (total / 60)} mins)')

plt.show()
