import os
import sys
from test_gen import gen_one_testcase
import random

random.seed(0)

benchmarks_dir='benchmarks'
testcases = [
    ('example_6.txt', 2, 99, 0.4),
    ('example_7.txt', 10, 1, 0.25),
    ('example_8.txt', 27, 3, 0.3),
    ('example_9.txt', 50, 20, 0.32),
    ('example_10.txt', 100, 500, 0.41),
    ('example_11.txt', 128, 1280, 0.44),
    ('example_12.txt', 156, 871, 0.24),
    ('example_13.txt', 177, 1899, 0.34),
    ('example_14.txt', 200, 996, 0.42),
    ('example_15.txt', 217, 2024, 0.4),
    ('example_16.txt', 255, 5000, 0.2),
    ('example_17.txt', 512, 10000, 0.4),
    ('example_18.txt', 777, 7777, 0.3),
    ('example_19.txt', 999, 9999, 0.45),
    ('example_20.txt', 1000, 10000, 0.35),
]

for (filename, n_nodes, n_nets, ratio) in testcases:
    path = os.path.join(benchmarks_dir, filename)
    gen_one_testcase(
        filename=path, n_nodes=n_nodes, n_nets=n_nets, ratio=ratio
    )

