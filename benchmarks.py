from test_correctness import dp_matrix_cost
from matrix_chain import optimal_matrix_chain_cost
import csv
from hwcounter import Timer
import numpy as np
from tqdm import tqdm

rng = np.random.default_rng(seed=42)

# clock speed in Hz
# I'm testing on a 2.6 GHz Intel
CLOCK_SPEED = 2.6e9

with open('benchmarks.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(('algorithm', 'size', 'time'))
    
    n = [5, 10, 20, 50, 100, 200, 500]
    post_n = [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
    
    for size in tqdm(n, 'Comparison'):
        for _ in range(20):
            dims = rng.integers(10, 10_000, size=size)
            for (method, name) in (
                (dp_matrix_cost, 'Dynamic Programming'),
                (optimal_matrix_chain_cost, 'Hu-Shing')
            ):
                with Timer() as t:
                    method(dims)

                time = t.cycles / CLOCK_SPEED
                writer.writerow((name, size, time))

    for size in tqdm(post_n, 'Above and Beyond'):
        for _ in range(20):
            dims = rng.integers(10, 10_000, size=size)
            with Timer() as t:
                optimal_matrix_chain_cost(dims)

            time = t.cycles / CLOCK_SPEED
            writer.writerow(('Hu-Shing', size, time))

print('Done!')