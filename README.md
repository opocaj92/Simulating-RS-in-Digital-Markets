# Simulating RS in Digital Markets
Experimental setup for simulating the impact of recommender systems in digital markets

## Prerequisites
To be able to run the code provided in this repository it is necessary to install the extended t-recs library available at [https://github.com/opocaj92/t-recs](https://github.com/opocaj92/t-recs).

Please follow the instructions on the project page on how to install and configure the framework.

## Usage
The provided code consists of three parts:
1. The running script `main.py`, used to run experiments,
2. The running and plotting functions, contained in `utils.py`,
3. The parameter files in `params` subfolders, that are used to provide the configuration parameters to the execution script.

A given param file (experiment configuration) can be run with:

`python3 main.py -d subdir_of_param_file -p param_filename_without_py`

Other available options are:
- `-o output_subdir` (default is a new folder with the name of the param file)
- `-cbo` to only execute content-based RS
- `-hybrids` to also execute hybrid RSs
- `-plots` to save plots on the tracked metrics
- `-debug` to save debug info (like the users/items values etc...)
- `-more` to track an additional set of metrics
- `-skip` used for big configs like `paper_experiments/combined_all.py`, it allows to skip configurations that will results in the same setting

## Sweep Param Files
In order to simplify the execution of multiple parameter settings, we allowed for "sweep" param files, that contains a list of value for a given (or multiple) parameters. This will results in (sequentially) executing multiple configurations, one after the other, where the value of that parameter(s) takes different values in each execution.

As an example, if the parameter file is:

`params = {
  a = 1,
  b = [1, 2],
}`

This will execute two different experiments, one where `a = 1, b = 1` and the other where `a = 1, b = 2`.
