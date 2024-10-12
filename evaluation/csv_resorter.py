# Imports
import csv
import math
import os
import matplotlib
import numpy
import matplotlib.pyplot as plot
from scipy.stats import chi2

# Global definitions
SEQUENTIAL_ALGORITHMS = ["sequential_two_lattice", "sequential_two_step", "sequential_swap", "sequential_shift"]
PARALLEL_ALGORITHMS = ["parallel_two_lattice", "parallel_two_lattice_framework", "parallel_two_step", "parallel_swap", "parallel_shift"]
ACCESS_PATTERNS = ["collision", "stream", "bundle"]
TESTS = ["strong_scaling", "weak_scaling"]

current_runtimes = []

print("Resorting .csv files...")

for test in TESTS:

    read_content = []
    future_content = [["algorithm", "access_pattern", "cores", "runtime[s]"]]

    if os.path.isfile("../runtimes/" + test + "_readable.csv"):
        continue

    print("Currently working on", test, "test")

    with open("../runtimes/" + test + "_results.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            read_content.append(line)

    print("Sequential algorithms...")
    for algo in SEQUENTIAL_ALGORITHMS:
        for access_pattern in ACCESS_PATTERNS:
            ready_line = [algo, access_pattern, "1"]
            for line in read_content:
                if line[0] == algo and line[1] == access_pattern:
                    ready_line.append(line[-1])
            future_content.append(ready_line)

    max_core_count = int(read_content[-1][2])
    max_pow_core_count = int(math.log2(max_core_count))
    core_ticks = [2 ** i for i in range(1, max_pow_core_count+1)]

    print("Parallel algorithms...")
    for algo in PARALLEL_ALGORITHMS:
        for access_pattern in ACCESS_PATTERNS:
            for core_count in core_ticks:
                ready_line = [algo, access_pattern, str(core_count)]
                for line in read_content:
                    if line[0] == algo and line[1] == access_pattern and line[2] == str(core_count):
                        ready_line.append(line[-1])
                future_content.append(ready_line)
    
    with open("../runtimes/" + test + "_readable.csv", "w") as new_file:
        writer = csv.writer(new_file)
        writer.writerows(future_content)

print("Done resorting .csv files.")