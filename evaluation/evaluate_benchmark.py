# Imports
import csv
import math
import os
import matplotlib
import numpy
import matplotlib.pyplot as plot
from scipy.stats import chi2

# Matplotlib configurations
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'errorbar.capsize': 5})
matplotlib.rcParams['figure.dpi'] = 300

# Global definitions
ACCESS_PATTERNS = ["collision", "stream", "bundle"]
ALGORITHMS = ["two_step", "swap", "two_lattice_framework", "two_lattice", "shift"]
CONTENTS = ["averages", "lower_bounds", "upper_bounds"]
COLORS = ['r', 'g', 'k', 'y', 'b']
MARKERS = ['o', 's', 'D', '*', 'p']


def confidence_error(standard_deviation: float, n: int, confidence:float=0.95) -> list[float]:
    """
    Determines the confidence error for the specified confidence, standard deviation and number of samples.

    Args:
        standard_deviation (float): Standard deviation of the test
        n (int): number of samples
        confidence (float, optional): The confidence used. Defaults to 0.95.

    Returns:
        list[float]: A list containing the lower error bound and the upper error bound in this order.
    """

    alpha = 1.0 - confidence
    lower_limit = standard_deviation * numpy.sqrt(n / chi2.ppf(1 - (alpha / 2), df=n))
    upper_limit = standard_deviation * numpy.sqrt(n / chi2.ppf(alpha / 2, df=n))
    return [lower_limit, upper_limit]



def determine_algorithm(algorithm_name: str) -> list[str]:
    """
    Determines to which data set the algorithm with the given name belongs.
    This is necessary to group the results of the corresponding sequential and parallel algorithms together.
    Notice that the sequential two-lattice algorithm contributes to two data sets, 
    namely "two_lattice" and "two_lattice_framework".

    Returns:
        list[str]: A list containing the string representations of the data sets this algorithm contributes to.
    """
    #print("Working on algorithm: ", algorithm_name )
    if algorithm_name == "parallel_two_lattice":
        return ["two_lattice"]
    elif line[0] == "parallel_two_lattice_framework":
        return ["two_lattice_framework"]
    elif line[0] == "parallel_two_step" or line[0] == "sequential_two_step":
        return ["two_step"]
    elif line[0] == "parallel_swap" or line[0] == "sequential_swap":
        return ["swap"]
    elif line[0] == "parallel_shift" or line[0] == "sequential_shift":
        return ["shift"]
    elif line[0] == "sequential_two_lattice":
        return ["two_lattice", "two_lattice_framework"]
    else:
        print("Unknown algorithm name:", algorithm_name)
    

def extract_runtimes_from_line(line: list[str]) -> list[float]:
    """
    Extracts all runtimes in a given line and returns them.

    Args:
        line (list[str]): The line in question

    Returns:
        list[float]: A list containing all runtimes extracted from the given line
    """
    result = []
    for i in range(3, len(line)):
        result.append(float(line[i]))
    return result


def set_result(
        results: numpy.ndarray, 
        algorithm: str, 
        access_pattern: str, 
        content: str, 
        core_tick: int, 
        value: float
    ) -> None:
    """
    Sets a value at the specified result data structure.
    The indices are determined using the remaining parameters.

    Args:
        results (numpy.ndarray): the data structure storing the runtimes
        algorithm (str): string representation of an algorithm family
        access_pattern (str): string representation of the access pattern used
        content (str): either "averages" or "lower_bounds" or "upper_bounds"
        core_tick (list[int]): a list containing the surveyed amount of cores in ascending order
        value (float): the value that is to be set
    """

    access_pattern_index = 0
    algorithm_index = 0
    content_index = 0

    for i in range(0, len(ALGORITHMS)):
        if algorithm == ALGORITHMS[i]:
            algorithm_index = i
            break

    for i in range(0, len(ACCESS_PATTERNS)):
        if access_pattern == ACCESS_PATTERNS[i]:
            access_pattern_index = i
            break
            
    for i in range(0, len(CONTENTS)):
        if content == CONTENTS[i]:
            content_index = i
            break

    results[algorithm_index][access_pattern_index][content_index][core_tick] = value


def get_result(
        results: numpy.ndarray, 
        algorithm: str, 
        access_pattern: str, 
        content: str, 
        core_tick: int, 
    ) -> float:
    """
    Returns a value read from the specified result data structure.
    The indices are determined using the remaining parameters.

    Args:
        results (numpy.ndarray): the data structure storing the runtimes
        algorithm (str): string representation of an algorithm family
        access_pattern (str): string representation of the access pattern used
        content (str): either "averages" or "lower_bounds" or "upper_bounds"
        core_tick (list[int]): a list containing the surveyed amount of cores in ascending order

     Returns:
        float: The content of the respective test run
    """

    access_pattern_index = 0
    algorithm_index = 0
    content_index = 0

    for i in range(0, len(ALGORITHMS)):
        if algorithm == ALGORITHMS[i]:
            algorithm_index = i
            break

    for i in range(0, len(ACCESS_PATTERNS)):
        if access_pattern == ACCESS_PATTERNS[i]:
            access_pattern_index = i
            break
            
    for i in range(0, len(CONTENTS)):
        if content == CONTENTS[i]:
            content_index = i
            break

    return results[algorithm_index][access_pattern_index][content_index][core_tick]


def set_scaling_result(scaling_results: numpy.ndarray, algorithm: str, access_pattern: str, core_tick: list[int], value: float) -> None:
    """
    Sets a value at the specified data structure containing the scaling results.
    The indices are determined using the remaining parameters.

    Args:
        scaling_results (numpy.ndarray): the data structure storing the scaling factors
        algorithm (str): string representation of an algorithm family
        core_tick (list[int]): a list containing the surveyed amount of cores in ascending order
        value (float): the value that is to be set
    """

    algorithm_index = 0
    access_pattern_index = 0

    for i in range(0, len(ALGORITHMS)):
        if algorithm == ALGORITHMS[i]:
            algorithm_index = i
            break

    for i in range(0, len(ACCESS_PATTERNS)):
        if access_pattern == ACCESS_PATTERNS[i]:
            access_pattern_index = i
            break
    
    scaling_results[algorithm_index][access_pattern_index][core_tick] = value


def get_scaling_result(scaling_results: numpy.ndarray, algorithm: str, access_pattern: str, core_tick: list[int]) -> float:
    """
    Returns a value read from the specified result data structure.
    The indices are determined using the remaining parameters.

    Args:
        scaling_results (numpy.ndarray): the data structure storing the scaling factors
        algorithm (str): string representation of an algorithm family
        core_tick (list[int]): a list containing the surveyed amount of cores in ascending order

     Returns:
        float: The content of the respective test run
    """

    algorithm_index = 0
    access_pattern_index = 0

    for i in range(0, len(ALGORITHMS)):
        if algorithm == ALGORITHMS[i]:
            algorithm_index = i

    for i in range(0, len(ACCESS_PATTERNS)):
        if access_pattern == ACCESS_PATTERNS[i]:
            access_pattern_index = i
            break
    
    return scaling_results[algorithm_index][access_pattern_index][core_tick]

'''
main
'''
if __name__ == "__main__":

    # Make results readable
    os.system("python3 csv_resorter.py")

    # Initializations
    core_ticks = []
    read_content_weak = []
    read_content_strong = []

    # Store content of file with weak scaling results
    with open("../runtimes/weak_scaling_readable.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            read_content_weak.append(line)

    # Store content of file with strong scaling results   
    with open("../runtimes/strong_scaling_readable.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            read_content_strong.append(line)

    # Determine core ticks
    max_core_count = 0
    for i in range(1, len(read_content_weak)):
        if int(read_content_weak[i][2]) > max_core_count:
            core_ticks.append(int(read_content_weak[i][2]))
            max_core_count = int(read_content_weak[i][2])
        elif int(read_content_weak[i][2]) == max_core_count:
            max_core_count = int(read_content_weak[i][2])
        else:
            break
    print("Core ticks are: ", core_ticks)

    ### Weak scaling
    print("Creating weak scaling plots...")

    runtimes = []
    lower_bound = 0
    upper_bound = 0
    average = 0
    standard_deviation = 0
    algorithm = ""
    results = numpy.zeros(shape=(5,3,3,len(core_ticks)), dtype=float)

    ## Assign times to corresponding data set
    for i in range(1, len(read_content_weak)):
        line = read_content_weak[i]
        algorithm = determine_algorithm(line[0])
        length = len(line)
        runtimes = extract_runtimes_from_line(line)
        average = numpy.average(runtimes)

        for current_algorithm in algorithm:
            set_result(results, current_algorithm, line[1], "averages", core_ticks.index(int(line[2])), average)

        standard_deviation = numpy.std(runtimes)
        lower, upper = confidence_error(standard_deviation, len(runtimes))

        for current_algorithm in algorithm:
            set_result(results, current_algorithm, line[1], "lower_bounds", core_ticks.index(int(line[2])), lower)
            set_result(results, current_algorithm, line[1], "upper_bounds", core_ticks.index(int(line[2])), upper)


    ## Determine scaling factors and plot results
    scaling_results = numpy.zeros(shape=(5,3,len(core_ticks)), dtype=float)
    efficiency_results = numpy.zeros(shape=(5,3,len(core_ticks)), dtype=float)

    for access_pattern in ACCESS_PATTERNS:
        
        # Determine weak scaling factor based on results
        for i in range(0, len(core_ticks)):
            core_count = core_ticks[i]

            for current_algo in ALGORITHMS: 

                set_scaling_result(efficiency_results, current_algo, access_pattern, i, 
                                   (get_result(results, current_algo, access_pattern, "averages", 0)) / 
                                   get_result(results, current_algo, access_pattern, "averages", i)) 
                
                set_scaling_result(scaling_results, current_algo, access_pattern, i, 
                                   (core_count * get_result(results, current_algo, access_pattern, "averages", 0)) / 
                                   get_result(results, current_algo, access_pattern, "averages", i)) 

        # Plot runtime results
        fig, ax = plot.subplots()

        global_min_y = math.inf
        global_max_y = -math.inf

        for i in range(0,5):
            y_values = [get_result(results, ALGORITHMS[i], access_pattern, "average", j) for j in range(0, len(core_ticks))]
            error_bounds = [[get_result(results, ALGORITHMS[i], access_pattern, content, j) 
                             for j in range(0, len(core_ticks))] 
                             for content in ["lower_bounds", "upper_bounds"]]
            ax.errorbar(core_ticks, y_values, yerr=error_bounds, capsize=5, elinewidth=2, markeredgewidth=2, color=COLORS[i])

            current_max_y = numpy.max([error_bounds[1][j] + y_values[j] for j in range(0, len(core_ticks))])
            current_min_y = numpy.min([y_values[j] - error_bounds[0][j] for j in range(0, len(core_ticks))])

            if current_max_y > global_max_y:
                global_max_y = current_max_y
            
            if current_min_y < global_min_y:
                global_min_y = current_min_y

        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(["Two-step", "Swap", "Framework two-lattice", "Non-framework two-lattice ", "Shift",], loc='best')
        #plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, runtimes with $\displaystyle\gamma=95\%$, " + access_pattern + " layout")
        plot.xlabel('cores')
        plot.ylabel('runtime [s]')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim([math.floor(global_min_y / 10.0) * 10, math.ceil(global_max_y / 10.0)* 10+10])
        plot.xticks(core_ticks)
        plot.yticks(range(math.floor(global_min_y / 10.0) * 10, math.ceil(global_max_y / 10.0)* 10+10, 10))
        plot.savefig("../images/runtimes/runtimes_weak_scaling_" + access_pattern + ".pdf", format="pdf", bbox_inches="tight")
        plot.close()

        # Plot efficiency results
        fig, ax = plot.subplots()

        global_min_y = math.inf
        global_max_y = -math.inf

        for i in range(0,5):
            y_values = [get_scaling_result(efficiency_results, ALGORITHMS[i], access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=COLORS[i], linestyle='-', marker=MARKERS[i])
            pass
        
        plot.plot(core_ticks, [1 for i in range(0, len(core_ticks))], color='k', linestyle='--')

        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(["Two-step", "Swap", "Framework two-lattice", "Non-framework two-lattice ", "Shift", 'Ideal'], loc='best')
        #plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, " + access_pattern + " layout")
        plot.xlabel('cores')
        plot.ylabel('efficiency')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        ax.set_ylim([0.2, 1.1])
        plot.yticks(numpy.linspace(0.2,1,9, endpoint=True))
        plot.savefig("../images/efficiency/weak_scaling_" + access_pattern + "_efficiency.pdf", format="pdf", bbox_inches="tight")
        plot.close()

    ## Determine best algorithms for weak scaling and compare access patterns
    # algorithm_ranking_two_lattice = []
    # algorithm_ranking_two_lattice_framework = []
    # algorithm_ranking_two_step = []
    # algorithm_ranking_swap = []
    # algorithm_ranking_shift = []

    # for access_pattern in ACCESS_PATTERNS:
    #     algorithm_ranking_two_lattice.append(("two_lattice", access_pattern, get_scaling_result(scaling_results, "two_lattice", access_pattern, len(core_ticks) - 1)))
    #     algorithm_ranking_two_lattice_framework.append(("two_lattice_framework", access_pattern, get_scaling_result(scaling_results, "two_lattice_framework", access_pattern, len(core_ticks) - 1)))
    #     algorithm_ranking_two_step.append(("two_step", access_pattern, get_scaling_result(scaling_results, "two_step", access_pattern, len(core_ticks) - 1)))
    #     algorithm_ranking_swap.append(("swap", access_pattern, get_scaling_result(scaling_results, "swap", access_pattern, len(core_ticks) - 1)))
    #     algorithm_ranking_shift.append(("shift", access_pattern, get_scaling_result(scaling_results, "shift", access_pattern, len(core_ticks) - 1)))

    # algorithm_ranking_two_lattice.sort(key = lambda tuple : tuple[2], reverse=True)
    # algorithm_ranking_two_lattice_framework.sort(key = lambda tuple : tuple[2], reverse=True)
    # algorithm_ranking_two_step.sort(key = lambda tuple : tuple[2], reverse=True)
    # algorithm_ranking_swap.sort(key = lambda tuple : tuple[2], reverse=True)
    # algorithm_ranking_shift.sort(key = lambda tuple : tuple[2], reverse=True)

    # all_minima = [algorithm_ranking_two_lattice[0], algorithm_ranking_two_lattice_framework[0], algorithm_ranking_two_step[0], algorithm_ranking_swap[0], algorithm_ranking_shift[0]]
    # all_minima.sort(key = lambda tuple : tuple[2], reverse=True)
    # global_optimum = all_minima[0]
    # second_optimum = all_minima[1]
    # print("The best weak scaling result is achieved by the", global_optimum[0], "algorithm using the", global_optimum[1], "access pattern with scaling factor", global_optimum[2])
    # print("The second best weak scaling result is achieved by the", second_optimum[0], "algorithm using the", second_optimum[1], "access pattern with scaling factor", second_optimum[2])

    # line_styles_access_patterns = ['dotted', '-', '-.']

    # for algo in [global_optimum[0], second_optimum[0]]:
        
    #     fig, ax = plot.subplots()
    #     new_colors = ['r', 'g', 'b']
    #     algorithm_name = ""

    #     for access_pattern in ACCESS_PATTERNS:
    #         y_values = [get_scaling_result(scaling_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
    #         plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)

    #         current_max_y = numpy.max([y_values[j] for j in range(0, len(core_ticks))])
    #         current_min_y = numpy.min([y_values[j] for j in range(0, len(core_ticks))])

    #         if current_max_y > global_max_y:
    #             global_max_y = current_max_y
            
    #         if current_min_y < global_min_y:
    #             global_min_y = current_min_y
                
    #     plot.plot(core_ticks, core_ticks, color='k', linestyle='--')
    #     plot.grid(color='#808080', linestyle='--', linewidth=0.5)
    #     plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

    #     if algo == "two_lattice":
    #         algorithm_name = "non-framework two-lattice"
    #     elif algo == "two_lattice_framework":
    #         algorithm_name = "framework two-lattice"
    #     else:
    #         algorithm_name = algo

    #     #plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, " + algorithm_name + " algorithm")
    #     plot.xlabel('cores')
    #     plot.ylabel('speedup')
    #     ax.set_xscale('log', base=2)
    #     ax.set_yscale('log', base=10)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     plot.xticks(core_ticks)
    #     plot.yticks([1,2,4,8,10,15,20,30,40,50])
    #     ax.set_ylim([1, global_max_y+10])

    #     plot.savefig("../images/best_algorithms/weak_scaling_" + algo + "_speedup.pdf", format="pdf", bbox_inches="tight")
    #     plot.close()

    ## Determine best algorithms for strong scaling and compare access patterns
    algorithm_ranking_two_lattice = []
    algorithm_ranking_two_lattice_framework = []
    algorithm_ranking_two_step = []
    algorithm_ranking_swap = []
    algorithm_ranking_shift = []

    for access_pattern in ACCESS_PATTERNS:
        algorithm_ranking_two_lattice.append(("two_lattice", access_pattern, get_scaling_result(efficiency_results, "two_lattice", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_lattice_framework.append(("two_lattice_framework", access_pattern, get_scaling_result(efficiency_results, "two_lattice_framework", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_step.append(("two_step", access_pattern, get_scaling_result(efficiency_results, "two_step", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_swap.append(("swap", access_pattern, get_scaling_result(efficiency_results, "swap", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_shift.append(("shift", access_pattern, get_scaling_result(efficiency_results, "shift", access_pattern, len(core_ticks) - 1)))

    algorithm_ranking_two_lattice.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_lattice_framework.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_step.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_swap.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_shift.sort(key = lambda tuple : tuple[2], reverse=True)

    all_minima = [algorithm_ranking_two_lattice[0], algorithm_ranking_two_lattice_framework[0], algorithm_ranking_two_step[0], algorithm_ranking_swap[0], algorithm_ranking_shift[0]]
    all_minima.sort(key = lambda tuple : tuple[2], reverse=True)
    global_optimum = all_minima[0]
    second_optimum = all_minima[1]
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")
    print("The best weak scaling result is achieved by the", global_optimum[0], "algorithm using the", global_optimum[1], "access pattern with scaling efficiency", global_optimum[2])
    print("The second best weak scaling result is achieved by the", second_optimum[0], "algorithm using the", second_optimum[1], "access pattern with scaling efficiency", second_optimum[2])
    print("-------------------------------------------------------------------------------------------")
    print("For the", global_optimum[0], "algorithm using the collision access pattern, the scaling efficiency is ", get_scaling_result(efficiency_results, global_optimum[0], "collision", len(core_ticks) - 1))
    print("For the", second_optimum[0], "algorithm using the collision access pattern, the scaling efficiency is ", get_scaling_result(efficiency_results, second_optimum[0], "collision", len(core_ticks) - 1))
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")

    line_styles_access_patterns = ['dotted', '-', '-.']

    for algo in [global_optimum[0], second_optimum[0]]:

        fig, ax = plot.subplots()

        new_colors = ['r', 'g', 'b']
        algorithm_name = ""

        for access_pattern in ACCESS_PATTERNS:
            y_values = [get_scaling_result(efficiency_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)
            
        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

        if algo == "two_lattice":
            algorithm_name = "non-framework two-lattice"
        elif algo == "two_lattice_framework":
            algorithm_name = "two-lattice-framework"
        else:
            algorithm_name = algo

        #plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, " + algorithm_name + " algorithm")
        plot.xlabel('cores')
        plot.ylabel('efficiency')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        plot.yticks(numpy.linspace(0,1,11, endpoint=True))
        ax.set_ylim([0.3, 1.1])
        plot.savefig("../images/best_algorithms/weak_scaling_" + algo + "_efficiency.pdf", format="pdf", bbox_inches="tight")
        plot.close()

        ## Difference in efficiency to collision layout
        fig, ax = plot.subplots()

        new_colors = ['r', 'g', 'b']
        algorithm_name = ""

        y_values_collision = [get_scaling_result(efficiency_results, algo, "collision", j) for j in range(0, len(core_ticks))]
        for access_pattern in ACCESS_PATTERNS:
            y_values = [get_scaling_result(efficiency_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
            diff_y_values = [y_values[k] - y_values_collision[k] for k in range(0, len(y_values))]
            plot.plot(core_ticks, diff_y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)
            
        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

        if algo == "two_lattice":
            algorithm_name = "non-framework two-lattice"
        elif algo == "two_lattice_framework":
            algorithm_name = "two-lattice-framework"
        else:
            algorithm_name = algo

        #plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, " + algorithm_name + " algorithm")
        plot.xlabel('cores')
        plot.ylabel('difference in efficiency to collision layout')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)

        if algo == "shift":
            plot.yticks([-0.012, -0.008, -0.004, 0, 0.004, 0.008])
        #plot.yticks(numpy.linspace(0,1,11, endpoint=True))
        #ax.set_ylim([0.3, 1.1])
        plot.savefig("../images/best_algorithms/weak_scaling_" + algo + "_efficiency_diff_to_col.pdf", format="pdf", bbox_inches="tight")
        plot.close()


        # Plot runtime scatter
        for algo in [global_optimum[0], second_optimum[0]]:
            fig, ax = plot.subplots()

            for access_pattern in ACCESS_PATTERNS:
                error_bounds = [[get_result(results, algo, access_pattern, content, j) 
                                for j in range(0, len(core_ticks))] 
                                for content in ["lower_bounds", "upper_bounds"]]
                y_values = [error_bounds[1][i] + error_bounds[0][i] for i in range(0,len(core_ticks))]
                plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)

            if algo == "two_lattice":
                algorithm_name = "non-framework two-lattice"
            elif algo == "two_lattice_framework":
                algorithm_name = "two-lattice-framework"
            else:
                algorithm_name = algo
            
            plot.grid(color='#808080', linestyle='--', linewidth=0.5)
            plot.legend(['Collision', 'Stream', 'Bundle'], loc='best')
            ##plot.title(r"Weak scaling: $\displaystyle 512 \times 512$ nodes per core, " + algorithm_name + " algorithm")
            plot.xlabel('cores')
            plot.ylabel(r'width of 95\% confidence interval [s]')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=10)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            plot.xticks(core_ticks)
            if algo == "two_lattice":
                plot.yticks([0.2, 0.5, 1,1.5,2,2.5,3])
            elif algo == "shift":
                plot.yticks([0.1, 0.2, 0.5, 1,2,5])
            plot.savefig("../images/runtime_confidence_interval/weak_scaling_" + algo + "_confidence_int.pdf", format="pdf", bbox_inches="tight")
            plot.close()

    ### Strong scaling
    print("")
    print("Creating strong scaling plots...")

    runtimes = []
    lower_bound = 0
    upper_bound = 0
    average = 0
    standard_deviation = 0
    algorithm = ""
    results = numpy.zeros(shape=(5,3,3,len(core_ticks)), dtype=float)

    # Assign times to corresponding data set
    for i in range(1, len(read_content_strong)):
        line = read_content_strong[i]
        algorithm = determine_algorithm(line[0])
        length = len(line)
        runtimes = extract_runtimes_from_line(line)
        average = numpy.average(runtimes)

        for current_algorithm in algorithm:
            set_result(results, current_algorithm, line[1], "averages", core_ticks.index(int(line[2])), average)

        standard_deviation = numpy.std(runtimes)
        lower, upper = confidence_error(standard_deviation, len(runtimes))

        for current_algorithm in algorithm:
            set_result(results, current_algorithm, line[1], "lower_bounds", core_ticks.index(int(line[2])), lower)
            set_result(results, current_algorithm, line[1], "upper_bounds", core_ticks.index(int(line[2])), upper)

   ## Determine scaling factors and plot results
    scaling_results = numpy.zeros(shape=(5,3,len(core_ticks)), dtype=float)
    efficiency_results = numpy.zeros(shape=(5,3,len(core_ticks)), dtype=float)

    for access_pattern in ACCESS_PATTERNS:

        # Determine strong scaling factor based on results
        for i in range(0, len(core_ticks)):
            core_count = core_ticks[i]

            for current_algo in ALGORITHMS: 
                set_scaling_result(scaling_results, current_algo, access_pattern, i, 
                                   (get_result(results, current_algo, access_pattern, "averages", 0)) / 
                                   get_result(results, current_algo, access_pattern, "averages", i)) 
                
                set_scaling_result(efficiency_results, current_algo, access_pattern, i, 
                                   (get_result(results, current_algo, access_pattern, "averages", 0)) / 
                                   (core_count * get_result(results, current_algo, access_pattern, "averages", i))) 

        # Plot runtime results
        fig, ax = plot.subplots()
        global_min_y = math.inf
        global_max_y = -math.inf
        for i in range(0,5):
            y_values = [get_result(results, ALGORITHMS[i], access_pattern, "average", j) for j in range(0, len(core_ticks))]
            error_bounds = [[get_result(results, ALGORITHMS[i], access_pattern, content, j) 
                             for j in range(0, len(core_ticks))] 
                             for content in ["lower_bounds", "upper_bounds"]]
            
            ax.errorbar(core_ticks, y_values, yerr=error_bounds, capsize=5, elinewidth=2, markeredgewidth=2, color=COLORS[i])

            current_max_y = numpy.max([error_bounds[1][j] + y_values[j] for j in range(0, len(core_ticks))])
            current_min_y = numpy.min([y_values[j] - error_bounds[0][j] for j in range(0, len(core_ticks))])

            if current_max_y > global_max_y:
                global_max_y = current_max_y
            
            if current_min_y < global_min_y:
                global_min_y = current_min_y

        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(["Two-step", "Swap", "Framework two-lattice", "Non-framework two-lattice ", "Shift",], loc='best')
        #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, runtimes with $\displaystyle\gamma=95\%$, " + access_pattern + " layout")
        plot.xlabel('cores')
        plot.ylabel('runtime [s]')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim([math.floor(global_min_y), math.ceil(global_max_y / 10.0)* 10+10])
        plot.xticks(core_ticks)
        plot.yticks([3,4,5,10,20,30,40,50,60,80,100,140])
        plot.savefig("../images/runtimes/runtimes_strong_scaling_" + access_pattern + ".pdf", format="pdf", bbox_inches="tight")
        plot.close()

        # Plot speedup results
        fig, ax = plot.subplots()
        global_max_y = -math.inf

        for i in range(0,5):
            y_values = [get_scaling_result(scaling_results, ALGORITHMS[i], access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=COLORS[i], linestyle='-', marker=MARKERS[i])

            current_max_y = numpy.max([y_values[j] for j in range(0, len(core_ticks))])

            if current_max_y > global_max_y:
                global_max_y = current_max_y

        plot.plot(core_ticks, core_ticks, color='k', linestyle='--')

        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(["Two-step", "Swap", "Framework two-lattice", "Non-framework two-lattice ", "Shift", 'Ideal'], loc='best')
        #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, " + access_pattern + " layout")
        plot.xlabel('cores')
        plot.ylabel('speedup')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        plot.yticks([1,2,4,8,10,15,20,30,40,50])
        ax.set_ylim([1, global_max_y+10])
        plot.savefig("../images/speedup/strong_scaling_" + access_pattern + "_speedup.pdf", format="pdf", bbox_inches="tight")
        plot.close()

        # Plot efficiency results
        fig, ax = plot.subplots()

        for i in range(0,5):
            y_values = [get_scaling_result(efficiency_results, ALGORITHMS[i], access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=COLORS[i], linestyle='-', marker=MARKERS[i])

        plot.plot(core_ticks, [1 for i in range(0, len(core_ticks))], color='k', linestyle='--')

        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(["Two-step", "Swap", "Framework two-lattice", "Non-framework two-lattice ", "Shift", 'Ideal'], loc='best')
        #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, " + access_pattern + " layout")
        plot.xlabel('cores')
        plot.ylabel('efficiency')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        plot.yticks(numpy.linspace(0,1,11, endpoint=True))
        ax.set_ylim([0.15, 1.1])
        plot.savefig("../images/efficiency/strong_scaling_" + access_pattern + "_efficiency.pdf", format="pdf", bbox_inches="tight")
        plot.close()

    ## Determine best algorithms for strong scaling and compare access patterns
    algorithm_ranking_two_lattice = []
    algorithm_ranking_two_lattice_framework = []
    algorithm_ranking_two_step = []
    algorithm_ranking_swap = []
    algorithm_ranking_shift = []

    for access_pattern in ACCESS_PATTERNS:
        algorithm_ranking_two_lattice.append(("two_lattice", access_pattern, get_scaling_result(scaling_results, "two_lattice", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_lattice_framework.append(("two_lattice_framework", access_pattern, get_scaling_result(scaling_results, "two_lattice_framework", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_step.append(("two_step", access_pattern, get_scaling_result(scaling_results, "two_step", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_swap.append(("swap", access_pattern, get_scaling_result(scaling_results, "swap", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_shift.append(("shift", access_pattern, get_scaling_result(scaling_results, "shift", access_pattern, len(core_ticks) - 1)))

    algorithm_ranking_two_lattice.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_lattice_framework.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_step.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_swap.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_shift.sort(key = lambda tuple : tuple[2], reverse=True)

    all_minima = [algorithm_ranking_two_lattice[0], algorithm_ranking_two_lattice_framework[0], algorithm_ranking_two_step[0], algorithm_ranking_swap[0], algorithm_ranking_shift[0]]
    all_minima.sort(key = lambda tuple : tuple[2], reverse=True)
    global_optimum = all_minima[0]
    second_optimum = all_minima[1]
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")
    print("The best strong scaling result is achieved by the", global_optimum[0], "algorithm using the", global_optimum[1], "access pattern with scaling factor", global_optimum[2])
    print("The second best strong scaling result is achieved by the", second_optimum[0], "algorithm using the", second_optimum[1], "access pattern with scaling factor", second_optimum[2])
    print("-------------------------------------------------------------------------------------------")
    print("For the", global_optimum[0], "algorithm using the collision access pattern, the scaling factor is ", get_scaling_result(scaling_results, global_optimum[0], "collision", len(core_ticks) - 1))
    print("For the", second_optimum[0], "algorithm using the collision access pattern, the scaling factor is ", get_scaling_result(scaling_results, second_optimum[0], "collision", len(core_ticks) - 1))
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")

    line_styles_access_patterns = ['dotted', '-', '-.']

    for algo in [global_optimum[0], second_optimum[0]]:

        fig, ax = plot.subplots()

        new_colors = ['r', 'g', 'b']
        algorithm_name = ""

        for access_pattern in ACCESS_PATTERNS:
            y_values = [get_scaling_result(scaling_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)

            current_max_y = numpy.max([error_bounds[1][j] + y_values[j] for j in range(0, len(core_ticks))])
            
        plot.plot(core_ticks, core_ticks, color='k', linestyle='--')
        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

        if algo == "two_lattice":
            algorithm_name = "non-framework two-lattice"
        elif algo == "two_lattice_framework":
            algorithm_name = "two-lattice-framework"
        else:
            algorithm_name = algo

        #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, " + algorithm_name + " algorithm")
        plot.xlabel('cores')
        plot.ylabel('speedup')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        plot.yticks([1,2,4,8,12,16,20,24,28,36])
        ax.set_ylim([1, numpy.ceil(global_optimum[2]+2)])

        plot.savefig("../images/best_algorithms/strong_scaling_" + algo + "_speedup.pdf", format="pdf", bbox_inches="tight")
        plot.close()
        
    ## Determine best algorithms for strong scaling and compare access patterns
    algorithm_ranking_two_lattice = []
    algorithm_ranking_two_lattice_framework = []
    algorithm_ranking_two_step = []
    algorithm_ranking_swap = []
    algorithm_ranking_shift = []

    for access_pattern in ACCESS_PATTERNS:
        algorithm_ranking_two_lattice.append(("two_lattice", access_pattern, get_scaling_result(efficiency_results, "two_lattice", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_lattice_framework.append(("two_lattice_framework", access_pattern, get_scaling_result(efficiency_results, "two_lattice_framework", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_two_step.append(("two_step", access_pattern, get_scaling_result(efficiency_results, "two_step", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_swap.append(("swap", access_pattern, get_scaling_result(efficiency_results, "swap", access_pattern, len(core_ticks) - 1)))
        algorithm_ranking_shift.append(("shift", access_pattern, get_scaling_result(efficiency_results, "shift", access_pattern, len(core_ticks) - 1)))

    algorithm_ranking_two_lattice.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_lattice_framework.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_two_step.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_swap.sort(key = lambda tuple : tuple[2], reverse=True)
    algorithm_ranking_shift.sort(key = lambda tuple : tuple[2], reverse=True)

    all_minima = [algorithm_ranking_two_lattice[0], algorithm_ranking_two_lattice_framework[0], algorithm_ranking_two_step[0], algorithm_ranking_swap[0], algorithm_ranking_shift[0]]
    all_minima.sort(key = lambda tuple : tuple[2], reverse=True)
    global_optimum = all_minima[0]
    second_optimum = all_minima[1]
    print("The best strong scaling result is achieved by the", global_optimum[0], "algorithm using the", global_optimum[1], "access pattern with scaling efficiency", global_optimum[2])
    print("The second best strong scaling result is achieved by the", second_optimum[0], "algorithm using the", second_optimum[1], "access pattern with scaling efficiency", second_optimum[2])
    print("-------------------------------------------------------------------------------------------")
    print("For the", global_optimum[0], "algorithm using the collision access pattern, the scaling efficiency is ", get_scaling_result(efficiency_results, global_optimum[0], "collision", len(core_ticks) - 1))
    print("For the", second_optimum[0], "algorithm using the collision access pattern, the scaling efficiency is ", get_scaling_result(efficiency_results, second_optimum[0], "collision", len(core_ticks) - 1))
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")

    line_styles_access_patterns = ['dotted', '-', '-.']

    for algo in [global_optimum[0], second_optimum[0]]:

        fig, ax = plot.subplots()

        new_colors = ['r', 'g', 'b']
        algorithm_name = ""

        for access_pattern in ACCESS_PATTERNS:
            y_values = [get_scaling_result(efficiency_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
            plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)
            
        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

        if algo == "two_lattice":
            algorithm_name = "non-framework two-lattice"
        elif algo == "two_lattice_framework":
            algorithm_name = "two-lattice-framework"
        else:
            algorithm_name = algo

        #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, " + algorithm_name + " algorithm")
        plot.xlabel('cores')
        plot.ylabel('efficiency')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        plot.yticks(numpy.linspace(0,1,11, endpoint=True))
        ax.set_ylim([0.2, 1.1])
        plot.savefig("../images/best_algorithms/strong_scaling_" + algo + "_efficiency.pdf", format="pdf", bbox_inches="tight")
        plot.close()

        ## Difference in efficiency to collision layout
        fig, ax = plot.subplots()

        new_colors = ['r', 'g', 'b']
        algorithm_name = ""

        y_values_collision = [get_scaling_result(efficiency_results, algo, "collision", j) for j in range(0, len(core_ticks))]
        for access_pattern in ACCESS_PATTERNS:
            y_values = [get_scaling_result(efficiency_results, algo, access_pattern, j) for j in range(0, len(core_ticks))]
            diff_y_values = [y_values[k] - y_values_collision[k] for k in range(0, len(y_values))]
            plot.plot(core_ticks, diff_y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)
            
        plot.grid(color='#808080', linestyle='--', linewidth=0.5)
        plot.legend(['Collision', 'Stream', 'Bundle', 'Ideal'], loc='best')

        if algo == "two_lattice":
            algorithm_name = "non-framework two-lattice"
        elif algo == "two_lattice_framework":
            algorithm_name = "two-lattice-framework"
        else:
            algorithm_name = algo

        plot.xlabel('cores')
        plot.ylabel('difference in efficiency to collision layout')
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.xticks(core_ticks)
        #plot.yticks(numpy.linspace(0,1,11, endpoint=True))
        #ax.set_ylim([0.3, 1.1])
        plot.savefig("../images/best_algorithms/strong_scaling_" + algo + "_efficiency_diff_to_col.pdf", format="pdf", bbox_inches="tight")
        plot.close()

        # Plot runtime scatter
        for algo in [global_optimum[0], second_optimum[0]]:
            fig, ax = plot.subplots()

            for access_pattern in ACCESS_PATTERNS:
                error_bounds = [[get_result(results, algo, access_pattern, content, j) 
                                for j in range(0, len(core_ticks))] 
                                for content in ["lower_bounds", "upper_bounds"]]
                y_values = [error_bounds[1][i] + error_bounds[0][i] for i in range(0,len(core_ticks))]
                plot.plot(core_ticks, y_values, color=new_colors[ACCESS_PATTERNS.index(access_pattern)], linestyle='-', marker='x', markersize=5)

            if algo == "two_lattice":
                algorithm_name = "non-framework two-lattice"
            elif algo == "two_lattice_framework":
                algorithm_name = "two-lattice-framework"
            else:
                algorithm_name = algo
            
            plot.grid(color='#808080', linestyle='--', linewidth=0.5)
            plot.legend(['Collision', 'Stream', 'Bundle'], loc='best')
            #plot.title(r"Strong scaling: $\displaystyle 1024 \times 1024$ nodes, " + algorithm_name + " algorithm")
            plot.xlabel('cores')
            plot.ylabel(r'width of 95\% confidence interval [s]')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=10)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            plot.xticks(core_ticks)
            if algo == "two_lattice":
                plot.yticks([0.05, 0.1, 0.2, 0.5, 1,2])
            elif algo == "shift":
                plot.yticks([0.1, 0.2, 0.5, 1,2])
            plot.savefig("../images/runtime_confidence_interval/strong_scaling_" + algo + "_confidence_int.pdf", format="pdf", bbox_inches="tight")
            plot.close()

    print("All done.")

'''
/main
'''