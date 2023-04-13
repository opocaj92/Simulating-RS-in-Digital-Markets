import copy
import argparse
from tqdm import tqdm

from utils import *

VERBOSE = True

def main(params, models, running_setting):
    # Initialise dicts that will hold results
    results = {}
    full_tmp_results = {}

    # Loop over recommender models
    for model_name in models:
        if VERBOSE:
            print()
            print("Simulating the " + model_name + " RS model")

        # Initialise results sub containers
        results[model_name] = {}
        full_tmp_results[model_name] = {}
        tmp_results = []

        # Do the experiment run-many times and aggregate (mean)
        for r in range(params["runs"]):
            # Run the experiment
            tmp_results = run_experiment(params, model_name, tmp_results, running_setting)

            # Adjust data in tmp_results in an experiment specific way
            tmp_results = process_tmp_results(params, tmp_results, running_setting)

        # Update results and results_std from tmp_results
        results, full_tmp_results = results_from_tmp_results(results, full_tmp_results, model_name, tmp_results)

    return results, full_tmp_results


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description = "main")
    parser.add_argument("--output_dir", "-o", required = False, default = "results")
    parser.add_argument("--params_subdir", "-d", required = True)
    parser.add_argument("--params_name", "-p", required = True) # fname without .py
    parser.add_argument("--cb_only", "-cbo", action = "store_true")
    parser.add_argument("--with_hybrid_rs", "-hybrid", action = "store_true")
    parser.add_argument("--save_pdf_plots", "-plots", action = "store_true")
    parser.add_argument("--save_debug_info", "-debug", action = "store_true")
    parser.add_argument("--more_computation", "-more", action = "store_true")
    parser.add_argument("--skip_repeated", "-skip", action = "store_true")
    args = parser.parse_args()
    params_name = args.params_name

    # Load params
    import importlib
    tmp = f"{args.params_subdir}." if args.params_subdir is not None else ""
    module = "params.%s%s" % (tmp, params_name)
    params_module = importlib.import_module(module, package = None)
    params = params_module.params

    # Output path depending on params
    savepath = args.output_dir # Top level subdir
    # Low level subdir determined by params_to_folder
    if args.params_subdir is not None:
        savepath = os.path.join(savepath, args.params_subdir, params_name)
    else:
        savepath = os.path.join(savepath, params_name)
    os.makedirs(savepath, exist_ok = True)

    # Copy params file to experiment folder
    import shutil
    if args.params_subdir is not None:
        src = os.path.join("params", args.params_subdir, f"{params_name}.py")
    else:
        src = os.path.join("params", f"{params_name}.py")
    dst = os.path.join(savepath, "params.py")
    shutil.copyfile(src, dst)

    # Define models
    if args.cb_only:
        models = ["content_based"]
    else:
        models = ["popularity_recommender", "content_based", "collaborative_filtering", "random_recommender", "ideal_recommender"]

    if args.with_hybrid_rs:
        # models.append("ensemble_hybrid")
        models.append("mixed_hybrid")

    running_setting = {
        "save_pdf_plots": args.save_pdf_plots,
        "save_debug_info": args.save_debug_info,
        "more_computation": args.more_computation
    }

    # Convert the "sweep" params file into a list of single experiment params
    all_params = process_mixed_dict(params)

    valid_params = []
    # If the setting is "illogical" (i.e. a duplicate of another different setting but with non-coherent parameters combination), we skip it
    for exp_params in all_params:
        if args.skip_repeated and (len(all_params) > 1 and ((exp_params["random_newly_created"] and (exp_params["random_items_per_iter"] == 0 or exp_params["num_creators"] == 0)) or ((exp_params["num_forced_items"] > 0 and exp_params["forced_period"] == 0) or (exp_params["num_forced_items"] == 0 and exp_params["forced_period"] > 0)))):
            continue
        valid_params.append(exp_params)

    # Keep a single csv that maps experiment numbers (as used in the config_N subdirs) to the corresponding parameters
    lst_for_index_csv = []

    # The main loop over experiments
    for i, exp_params in tqdm(enumerate(valid_params), total = len(valid_params)):
        # For the index csv
        tmp_params_dict = copy.deepcopy(exp_params)
        tmp_params_dict["exp_index"] = (i + 1)
        lst_for_index_csv.append(tmp_params_dict)

        # Allow for new items or not depending on if there are creators and set p accordingly
        if exp_params["num_creators"] == 0:
            exp_params["new_items"] = False
        else:
            exp_params["new_items"] = True
        exp_params["random_items_per_iter"] = exp_params["random_items_per_iter"] if exp_params["random_items_per_iter"] > 1 else int(exp_params["random_items_per_iter"] * exp_params["num_items_per_iter"])

        # Use subdirs when there are multiple experiments
        exp_savepath = os.path.join(savepath, "config_%03d" % (i + 1)) if len(valid_params) > 1 else savepath
        # And then write the exp params to the subdir
        if len(valid_params) > 1:
            os.makedirs(exp_savepath, exist_ok = True)
            write_params_file(exp_params, exp_savepath)

        # Do the simulations
        results, tmp_results = main(exp_params, models, running_setting)
        # Plotting
        plot_tmp_results(exp_params, tmp_results, exp_savepath, running_setting)
        plot_results(exp_params, results, exp_savepath, running_setting)

    # Create csv with experiment index and their config
    index_df = pd.DataFrame.from_dict(lst_for_index_csv)
    index_df = index_df.set_index("exp_index")
    index_df.to_csv(os.path.join(savepath, "Params_Index.csv"))
    master_csv(savepath, lst_for_index_csv, running_setting)