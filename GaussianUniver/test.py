import itertools
import time

def generate_degree_combinations_v1(k, d):
    for dividers in itertools.combinations(range(k + d - 1), d - 1):
        partition = [0] * d
        prev = -1
        for i, divider in enumerate(dividers):
            partition[i] = divider - prev - 1
            prev = divider
        partition[-1] = k + d - 1 - prev - 1
        yield tuple(partition)

def generate_degree_combinations_v2(k, d):
    if d == 1:
        yield (k,)
        return
    for i in range(k + 1):
        for tail in generate_degree_combinations_v2(k - i, d - 1):
            yield (i,) + tail

k, d = 15, 10  # Example values, you can adjust these to test different scenarios

# Measure runtime for version 1
start_time = time.time()
degree_combinations_v1 = list(generate_degree_combinations_v1(k, d))
v1_runtime = time.time() - start_time

# Measure runtime for version 2
start_time = time.time()
degree_combinations_v2 = list(generate_degree_combinations_v2(k, d))
v2_runtime = time.time() - start_time

print(f"Runtime for version 1: {v1_runtime:.6f} seconds")
print(f"Runtime for version 2: {v2_runtime:.6f} seconds")