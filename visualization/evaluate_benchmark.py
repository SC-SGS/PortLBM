#######################################################################################################################
# file:         evaluate_benchmark.py                                                                                 #
#                                                                                                                     #  
# author:       Marcel Graf                                                                                           #
#                                                                                                                     #
# description:  This Python file is used to generate plots for evaluating the lattice Boltzmann algorithms.           #
#                                                                                                                     #
# version:      1.0                                                                                                   #
#                                                                                                                     #
# date:         March 2025                                                                                            #
#                                                                                                                     #
# copyright:    Marcel Graf                                                                                           #
#######################################################################################################################

# IMPORTS #############################################################################################################
import json
import glob
import math
import os
import matplotlib
import numpy
import matplotlib.pyplot as plot
from scipy.stats import chi2

# MATPLOTLIB CONFIGURATIONS ###########################################################################################
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'errorbar.capsize': 5})
matplotlib.rcParams['figure.dpi'] = 300

# Global definitions
DATA_LAYOUTS = ["collision", "stream", "bundle"]
ALGORITHMS = ["gpu-two-lattice", "gpu-two-lattice-linear", "gpu-two-lattice-buffered", "gpu-swap"]
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


def load_json_from_file(path: str) -> dict:
    """
    Loads a JSON file at the specified path and extracts a dictionary from it.

    Args:
        path (str): The path to the JSON file

    Returns:
        dict: A dictionary built from the contents of the JSON file
    """

    with open(path, "r") as file:
        data = json.load(file)

    return data


def plot_runtimes(
        work_group_sizes: list[str], 
        plot_data: dict[str, float], 
        algorithm: str,
        data_layout: str, 
        device: str
    ) -> None:

    fig, ax = plot.subplots()
    bottom = numpy.zeros(len(work_group_sizes))
    
    for time, time_data in plot_data.items():
        p = ax.bar(work_group_sizes, time_data, 0.8, label=time, bottom=bottom)
        bottom += time_data

    leg = ax.legend()
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0.0)
    ax.grid(axis='y', color='#808080', linestyle='--', linewidth=0.5)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0.0)
    plot.xlabel('work-group size')
    plot.ylabel('time [s]')

    filename = "../plots/phase1/" + device + "/" + algorithm + "/stacked/" + "results_phase1_" + algorithm + "_" + data_layout + "_" + device + ".pdf"
    plot.savefig(filename, format="pdf", bbox_inches="tight")
    plot.close()


def plot_runtimes_error(
        work_group_sizes: list[str], 
        plot_data: dict[str, float], 
        algorithm: str,
        data_layout: str, 
        device: str
    ) -> None:
    fig, ax = plot.subplots()
    for time, time_data in plot_data.items():
        ax.errorbar(work_group_sizes, time_data, yerr=error_data[time], marker='s', capsize=5, elinewidth=1, markeredgewidth=1, linestyle='none', label=time)

    ax.grid(axis='y', color='#808080', linestyle='--', linewidth=0.5)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0.0)
    plot.xlabel('work-group size')
    plot.ylabel('time [s]')
    filename = "../plots/phase1/" + device + "/" + algorithm + "/error/" + "results_phase1_error_" + algorithm + "_" + data_layout + "_" + device + ".pdf"
    plot.savefig(filename, format="pdf", bbox_inches="tight")
    plot.close()


'''
main
'''
if __name__ == "__main__":

    # Set the default color cycle
    plot.rcParams['axes.prop_cycle'] = plot.cycler(color=["tab:blue", "tab:green", "tab:purple", "tab:olive", "tab:gray", "tab:pink"]) 


    result_files = glob.glob("../benchmarks/phase1/*.json")

    if not os.path.isdir("../plots/"):
        os.system("mkdir ../plots/")

    if not os.path.isdir("../plots/phase1"):
        os.system("mkdir ../plots/phase1")
    
    global_comparison = {}

    for algorithm in ALGORITHMS:
        global_comparison[algorithm] = {}
        global_comparison[algorithm] = {}
        global_comparison[algorithm] = {}
        global_comparison[algorithm] = {}

    for algorithm in ALGORITHMS:
        global_comparison[algorithm]["optimal_setup"] = ["collision", "auto"]
        global_comparison[algorithm]["optimal_times"] = [math.inf, math.inf, math.inf]
        global_comparison[algorithm]["optimal_init_errors"] = [0, 0]
        global_comparison[algorithm]["optimal_run_errors"] = [0, 0]

    for result_file in result_files:

        data = load_json_from_file(result_file)

        local_comparison = {}

        if not os.path.isdir("../plots/phase1/" + data["device"]):
            os.system("mkdir \"../plots/phase1/" + data["device"] + "\"")

        if not os.path.isdir("../plots/phase1/" + data["device"] + "/" + data["algorithm"]):
            os.system("mkdir \"../plots/phase1/" + data["device"] + "/" + data["algorithm"] + "\"")

        if not os.path.isdir("../plots/phase1/" +  data["device"] + "/" + data["algorithm"] + "/stacked"):
            os.system("mkdir \"../plots/phase1/" +  data["device"] + "/" + data["algorithm"] + "/stacked\"")

        if not os.path.isdir("../plots/phase1/" +  data["device"] + "/" + data["algorithm"] + "/error"): 
            os.system("mkdir \"../plots/phase1/" +  data["device"] + "/" + data["algorithm"] + "/error\"")

        for data_layout in DATA_LAYOUTS:

            local_comparison[data_layout] = {}

            local_comparison[data_layout]["optimal_work_group_size"] = "auto"
            local_comparison[data_layout]["optimal_times"] = [math.inf, math.inf, math.inf]
            local_comparison[data_layout]["optimal_init_errors"] = [0, 0]
            local_comparison[data_layout]["optimal_run_errors"] = [0, 0]

            work_group_sizes = []

            plot_data = {
                'Initialization times': [],
                'Runtimes': []
            }

            error_data = {
                'Initialization times': [],
                'Runtimes': []
            }

            lower_errors_init = []
            upper_errors_init = []

            lower_errors_run = []
            upper_errors_run = []

            for test_row in data[data_layout]:
                work_group_sizes.append(str(test_row["workGroupSize"]))

                average_init_time = numpy.average(test_row["initializationTimes"])
                average_runtime = numpy.average(test_row["runtimes"])

                plot_data["Initialization times"].append(average_init_time)
                plot_data["Runtimes"].append(average_runtime)
                average_total = average_init_time + average_runtime

                sdi = numpy.std(test_row["initializationTimes"])
                sdr = numpy.std(test_row["runtimes"])
                cei = confidence_error(sdi, len(test_row["initializationTimes"]))
                cer = confidence_error(sdr, len(test_row["initializationTimes"]))
                lower_errors_init.append(cei[0])
                upper_errors_init.append(cei[1])
                lower_errors_run.append(cer[0])
                upper_errors_run.append(cer[1])

                if average_total < global_comparison[data["algorithm"]]["optimal_times"][0]:
                    global_comparison[data["algorithm"]]["optimal_times"] = [average_total, average_init_time, average_runtime]
                    global_comparison[data["algorithm"]]["optimal_setup"] = [data_layout, test_row["workGroupSize"]]
                    global_comparison[data["algorithm"]]["optimal_init_errors"] = cei
                    global_comparison[data["algorithm"]]["optimal_run_errors"] = cer

                if average_total < local_comparison[data_layout]["optimal_times"][0]:
                    local_comparison[data_layout]["optimal_times"] = [average_total, average_init_time, average_runtime]
                    local_comparison[data_layout]["optimal_work_group_size"] = test_row["workGroupSize"]
                    local_comparison[data_layout]["optimal_init_errors"] = cei
                    local_comparison[data_layout]["optimal_run_errors"] = cer

            error_data["Initialization times"].append(lower_errors_init)
            error_data["Initialization times"].append(upper_errors_init)
            error_data["Runtimes"].append(lower_errors_run)
            error_data["Runtimes"].append(upper_errors_run)

            fig, ax = plot.subplots()
            bottom = numpy.zeros(len(local_comparison))
            names = []
            local_plot_data = {
                'Initialization times': [],
                'Runtimes': []
            }
            lower_errors_init = []
            upper_errors_init = []
            lower_errors_run = []
            upper_errors_run = []

            # Plot runtime results
            plot_runtimes(work_group_sizes, plot_data, data["algorithm"], data_layout, data["device"])
            plot_runtimes_error(work_group_sizes, plot_data, data["algorithm"], data_layout, data["device"])

        for key, value in local_comparison.items():
            names.append(key[:3] + "/" + str(value["optimal_work_group_size"]))
            lower_errors_init.append(value["optimal_init_errors"][0])
            upper_errors_init.append(value["optimal_init_errors"][1])
            lower_errors_run.append(value["optimal_run_errors"][0])
            upper_errors_run.append(value["optimal_run_errors"][1])
            local_plot_data["Initialization times"].append(value["optimal_times"][1])
            local_plot_data["Runtimes"].append(value["optimal_times"][2])

        for category, vals in local_plot_data.items():
            p = ax.bar(names, vals, 0.8, label=category, bottom=bottom)
            bottom += vals

        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
        leg.get_frame().set_alpha(0)
        leg.get_frame().set_linewidth(0.0)
        ax.grid(axis='y', color='#808080', linestyle='--', linewidth=0.5)
        plot.xticks(rotation=45)
        plot.ylabel('runtime [s]')
        filename = "../plots/phase1/" + data["device"] + "/local_" + data["algorithm"] + ".pdf"
        plot.savefig(filename, format="pdf", bbox_inches="tight")
        plot.close() 

        print("Best configuration for algorithm " + data["algorithm"] + ": " + str(global_comparison[data["algorithm"]]["optimal_setup"]) + " with average total time " + str(global_comparison[data["algorithm"]]["optimal_times"][0]))

    #print(global_comparison)
    fig, ax = plot.subplots()
    bottom = numpy.zeros(len(global_comparison))
    #numpy.seterr('raise')
    names = []
    global_plot_data = {
        'Initialization times': [],
        'Runtimes': []
    }
    lower_errors_init = []
    upper_errors_init = []
    lower_errors_run = []
    upper_errors_run = []

    for key, value in global_comparison.items():
        names.append(key[4:] + "/" + value["optimal_setup"][0][:3] + "/" + str(value["optimal_setup"][1]))
        lower_errors_init.append(value["optimal_init_errors"][0])
        upper_errors_init.append(value["optimal_init_errors"][1])
        lower_errors_run.append(value["optimal_run_errors"][0])
        upper_errors_run.append(value["optimal_run_errors"][1])
        global_plot_data["Initialization times"].append(value["optimal_times"][1])
        global_plot_data["Runtimes"].append(value["optimal_times"][2])

    for category, vals in global_plot_data.items():
        p = ax.bar(names, vals, 0.8, label=category, bottom=bottom)
        bottom += vals

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0.0)
    ax.grid(axis='y', color='#808080', linestyle='--', linewidth=0.5)
    plot.xticks(rotation=45)
    plot.ylabel('runtime [s]')

    filename = "../plots/global.pdf"
    plot.savefig(filename, format="pdf", bbox_inches="tight")
    plot.close() 

    print("All done.")

'''
! main
'''