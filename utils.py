import os
from typing import Dict, List, Any
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import sparse
import pandas as pd

from trecs.components import Users, Items, Creators
from trecs.models import PopularityRecommender, ContentFiltering, ImplicitMF, RandomRecommender
from trecs.metrics import InteractionMeasurement, MSEMeasurement
from trecs.random import Generator

from trecs_plus.models import IdealRecommender, EnsembleHybrid, MixedHybrid
from trecs_plus.metrics import RecommendationMeasurement, InteractionMetric, RecommendationMetric, ScoreMetric, CorrelationMeasurement, RecommendationRankingMetric, InteractionRankingMetric, VaryingInteractionAttributesSimilarity, VaryingInteractionAttrJaccard, VaryingRecSimilarity, VaryingRecAttributesSimilarity, VaryingRecSummedAttributesSimilarity, most_similar_users_pair, least_similar_users_pair, all_users_pairs, NNLSCoefficientsxMetric, NNLSCoefficientsAMetric, NNLSCoefficientsbMetric

models = {
    "popularity_recommender": PopularityRecommender,
    "content_based": ContentFiltering,
    "collaborative_filtering": ImplicitMF,
    "random_recommender": RandomRecommender,
    "ideal_recommender": IdealRecommender,
    "ensemble_hybrid": EnsembleHybrid,
    "mixed_hybrid": MixedHybrid
}

# Specific to num_creators > 0
def adjust_results(results):
    offsets = np.array([x.shape[0] for x in results])
    new_results = np.zeros((len(results), np.max(offsets)))
    for i in range(len(results)):
        new_results[i, :offsets[i]] = results[i]
    return new_results

def adjust_results_2d(results):
    offsets = np.array([x.shape[0] for x in results])
    new_results = np.zeros((len(results), np.max(offsets), results[0].shape[1]))
    for i in range(len(results)):
        new_results[i, :offsets[i], :] = results[i]
    return new_results

def adjust_results_2d_opposite(results):
    offsets = np.array([x.shape[1] for x in results])
    new_results = np.zeros((len(results), results[0].shape[0], np.max(offsets)))
    for i in range(len(results)):
        new_results[i, :, :offsets[i]] = results[i]
    return new_results

def get_entering_times(results):
    offsets = np.array([x.shape[0] for x in results])
    times = np.zeros(np.max(offsets), dtype = int)
    prev_enter = 0
    for i in range(len(results)):
        new_enter = results[i].shape[0]
        if new_enter > prev_enter:
            times[prev_enter:new_enter] = i
            prev_enter = new_enter
    return times


# Create initial experiment parameters
def get_initialisation_args(params):
    init_args = dict()
    init_args["real_profiles"] = np.random.randint(params["min_preference_per_attribute"], params["max_preference_per_attribute"], (params["num_users"], params["num_attributes"]))

    if params["horizontally_differentiated_only"]:
        num_ones = np.random.randint(params["num_attributes"])
        base_vector = np.concatenate([np.ones(num_ones), np.zeros(params["num_attributes"] - num_ones)])
        init_args["real_attributes"] = np.array([np.random.permutation(base_vector) for _ in range(params["num_items"])]).T
    else:
        init_args["real_attributes"] = Generator().binomial(n = 1, p = 0.5, size = (params["num_attributes"], params["num_items"]))

    init_args["real_types"] = np.random.random((params["num_creators"], params["num_attributes"])) if params["num_creators"] > 0 else None

    if params["num_forced_items"] > 0 and params["forced_period"] > 0:
        init_args["forced_items"] = np.argsort(-np.sum(init_args["real_attributes"], axis = 0))[:(int(params["shuffle_forced_items"]) + 1) * params["num_forced_items"]]
        #init_args["forced_items"] = np.random.randint(0, params["num_items"], size = params["num_forced_items"])
    return init_args


# Main execution function
def run_experiment(params, model_name, tmp_results, running_setting):
    # Get representations
    init_args = get_initialisation_args(params)

    # Create the users profiles
    actual_user_representation = Users(actual_user_profiles = init_args["real_profiles"],
                                       num_users = params["num_users"],
                                       size = (params["num_users"], params["num_attributes"]),
                                       attention_exp = params["attention_exp"],
                                       drift = 0,
                                       individual_rationality = params["individual_rationality"]
                                       )
    # Create the items profiles
    actual_item_representation = Items(item_attributes = init_args["real_attributes"],
                                       size = (params["num_attributes"], params["num_items"])
                                       )
    # Create the creators profiles, if any
    creator = Creators(actual_creator_profiles = init_args["real_types"],
                       creation_probability = params["p_creation"],
                       size = (params["num_creators"], params["num_attributes"])
                       ) if params["num_creators"] > 0 else None

    # Instantiate the recommender system
    if model_name == "content_based" or model_name == "ensemble_hybrid" or model_name == "mixed_hybrid":
        rec = models[model_name](num_attributes = params["num_attributes"],
                                 actual_user_representation = actual_user_representation,
                                 item_representation = actual_item_representation.get_component_state()["items"][0],
                                 actual_item_representation = actual_item_representation,
                                 creators = creator,
                                 num_items_per_iter = params["num_items_per_iter"],
                                 probabilistic_recommendations = params["probabilistic_recommendations"],
                                 random_newly_created = params["random_newly_created"],
                                 num_forced_items = params["num_forced_items"],
                                 forced_items = init_args["forced_items"] if params["num_forced_items"] > 0 else None,
                                 forced_period = params["forced_period"],
                                 )
    else:
        rec = models[model_name](actual_user_representation = actual_user_representation,
                                 actual_item_representation = actual_item_representation,
                                 creators = creator,
                                 num_items_per_iter = params["num_items_per_iter"],
                                 probabilistic_recommendations = params["probabilistic_recommendations"] if model_name != "random_recommender" else False,
                                 random_newly_created = params["random_newly_created"],
                                 num_forced_items = params["num_forced_items"],
                                 forced_items = init_args["forced_items"] if params["num_forced_items"] > 0 else None,
                                 forced_period = params["forced_period"],
                                 )

    # Add metrics
    rec.add_metrics(InteractionMeasurement(),
                    VaryingInteractionAttrJaccard(most_similar_users_pair, actual_user_representation, name = "interaction_similarity_most"),
                    VaryingInteractionAttrJaccard(least_similar_users_pair, actual_user_representation, name = "interaction_similarity_least"),
                    VaryingRecSimilarity(most_similar_users_pair, actual_user_representation, name = "rec_similarity_most"),
                    VaryingRecSimilarity(least_similar_users_pair, actual_user_representation, name = "rec_similarity_least"),
                    MSEMeasurement(),
                    ScoreMetric(),
                    CorrelationMeasurement(),
                    RecommendationRankingMetric(actual_user_representation),
                    InteractionRankingMetric(actual_user_representation)
                    )

    if running_setting["save_debug_info"]:
        rec.add_metrics(RecommendationMeasurement(),
                        InteractionMetric(),
                        RecommendationMetric()
                        )
        if model_name == "content_based":
            rec.add_metrics(NNLSCoefficientsxMetric(),
                            NNLSCoefficientsAMetric(),
                            NNLSCoefficientsbMetric()
                            )

    if running_setting["more_computation"]:
        rec.add_metrics(VaryingInteractionAttrJaccard(all_users_pairs, actual_user_representation, name = "interaction_similarity_all"),
                        VaryingInteractionAttributesSimilarity(most_similar_users_pair, actual_user_representation, name = "interaction_attr_similarity_most"),
                        VaryingInteractionAttributesSimilarity(least_similar_users_pair, actual_user_representation, name = "interaction_attr_similarity_least"),
                        VaryingInteractionAttributesSimilarity(all_users_pairs, actual_user_representation, name = "interaction_attr_similarity_all"),
                        VaryingRecSimilarity(all_users_pairs, actual_user_representation, name = "rec_similarity_all"),
                        VaryingRecAttributesSimilarity(most_similar_users_pair, actual_user_representation, name = "rec_attr_similarity_most"),
                        VaryingRecAttributesSimilarity(least_similar_users_pair, actual_user_representation, name = "rec_attr_similarity_least"),
                        VaryingRecAttributesSimilarity(all_users_pairs, actual_user_representation, name = "rec_attr_similarity_all"),
                        VaryingRecSummedAttributesSimilarity(most_similar_users_pair, actual_user_representation, name = "rec_summed_attr_similarity_most"),
                        VaryingRecSummedAttributesSimilarity(least_similar_users_pair, actual_user_representation, name = "rec_summed_attr_similarity_least"),
                        VaryingRecSummedAttributesSimilarity(all_users_pairs, actual_user_representation, name = "rec_summed_attr_similarity_all")
                        )

    # Pre-train the RS with random interactions
    if params["training"] > 0:
        rec.startup_and_train(timesteps = params["training"], no_new_items = True)
        if running_setting["save_pdf_plots"] and (model_name == "content_based" or model_name == "collaborative_filtering"):
            all_interactions_after_startup = rec.all_interactions

    # Do not pass in disable_tqdm for ImplicitMF
    if model_name == "collaborative_filtering":
        extra_args = dict(reset_interactions = False)
    else:
        extra_args = dict(disable_tqdm = True)

    # Do the simulation
    rec.users.drift = params["drift"]
    rec.run(timesteps = params["timesteps"],
            train_between_steps = params["train_between_steps"],
            random_items_per_iter = params["random_items_per_iter"],
            repeated_items = params["repeated_items"],
            no_new_items = not params["new_items"],
            **extra_args
            )

    # Store interaction matrices for ContentFiltering and ImplicitMF
    tmp_results.append(rec.get_measurements())

    if params["num_forced_items"] > 0 and params["forced_period"] > 0:
        tmp_results[-1]["forced_items"] = init_args["forced_items"]
    if running_setting["save_debug_info"]:
        # Store information on users, scores, items and creators
        tmp_results[-1]["real_profiles"] = rec.actual_user_profiles
        tmp_results[-1]["real_scores"] = rec.actual_user_item_scores
        tmp_results[-1]["real_attributes"] = np.transpose(rec.actual_item_attributes)
        if params["num_creators"] > 0:
            tmp_results[-1]["real_types"] = rec.creators.actual_creator_profiles
        if params["num_forced_items"] > 0 and params["forced_period"] > 0:
            tmp_results[-1]["forced_real_attributes"] = np.transpose(rec.actual_item_attributes)[init_args["forced_items"], :]

    if running_setting["save_pdf_plots"]:
        if model_name == "content_based" :
            tmp_results[-1]["all_interactions"] = rec.all_interactions.toarray()
            if params["training"] > 0:
                tmp_results[-1]["all_interactions_after_startup"] = all_interactions_after_startup.toarray()
        elif model_name == "collaborative_filtering":
            tmp_results[-1]["all_interactions"] = dataframe_to_sparse(rec.all_interactions, rec.num_users, rec.num_items).toarray()
            if params["training"] > 0:
                tmp_results[-1]["all_interactions_after_startup"] = dataframe_to_sparse(all_interactions_after_startup, rec.num_users, rec.num_items).toarray()

    return tmp_results


# Adjust results to be valid for plotting
def process_tmp_results(params, tmp_results, running_setting):
    tmp_results[-1]["interaction_histogram"][0] = np.zeros(params["num_items"])
    entering_times = get_entering_times(tmp_results[-1]["interaction_histogram"])
    items_at_step = np.zeros(np.array(tmp_results[-1]["interaction_histogram"]).shape[0])
    np.add.at(items_at_step, entering_times, 1)
    items_at_step = np.cumsum(items_at_step)

    tmp_results[-1]["interaction_histogram"] = adjust_results(tmp_results[-1]["interaction_histogram"])
    tmp_results[-1]["interaction_similarity_most"][0] = 0
    tmp_results[-1]["interaction_similarity_least"][0] = 0
    tmp_results[-1]["rec_similarity_most"][0] = 0
    tmp_results[-1]["rec_similarity_least"][0] = 0
    tmp_results[-1]["score"][0] = 0

    if running_setting["more_computation"]:
        tmp_results[-1]["interaction_similarity_all"][0] = 0
        tmp_results[-1]["interaction_attr_similarity_most"][0] = 0
        tmp_results[-1]["interaction_attr_similarity_least"][0] = 0
        tmp_results[-1]["interaction_attr_similarity_all"][0] = 0
        tmp_results[-1]["rec_similarity_all"][0] = 0
        tmp_results[-1]["rec_attr_similarity_most"][0] = 0
        tmp_results[-1]["rec_attr_similarity_least"][0] = 0
        tmp_results[-1]["rec_attr_similarity_all"][0] = 0
        tmp_results[-1]["rec_summed_attr_similarity_most"][0] = 0
        tmp_results[-1]["rec_summed_attr_similarity_least"][0] = 0
        tmp_results[-1]["rec_summed_attr_similarity_all"][0] = 0

    if running_setting["save_debug_info"]:
        tmp_results[-1]["recommendation_histogram"][0] = np.zeros(params["num_items"])
        tmp_results[-1]["recommendation_histogram"] = adjust_results(tmp_results[-1]["recommendation_histogram"])
        tmp_results[-1]["interaction_history"][0] = -np.ones(params["num_users"])
        tmp_results[-1]["recommendation_history"][0] = -np.ones((params["num_users"], params["num_items_per_iter"]))

    modified_ih = np.cumsum(np.array(tmp_results[-1]["interaction_histogram"]), axis = 0)
    modified_ih[0] = modified_ih[0] + 1e-32
    modified_ih = np.array([modified_ih[t] - modified_ih[params["training"]] if t > params["training"] else modified_ih[t] for t in range(modified_ih.shape[0])])
    windowed_modified_ih = np.array([modified_ih[t] - modified_ih[t - 10] if t - 10 > params["training"] else modified_ih[t] for t in range(modified_ih.shape[0])])
    sorted_ms = -np.sort(-(windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True)))
    tmp_results[-1]["cn"] = np.array([np.sum(ms[:int(items_at_step[i] * 0.1)]) for i, ms in enumerate(sorted_ms)])
    tmp_results[-1]["cn_entrant"] = np.sum(windowed_modified_ih[:, params["num_items"]:] / np.sum(windowed_modified_ih, axis = 1, keepdims = True), axis = 1)
    tmp_results[-1]["hhi"] = np.sum(np.square(windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True)), axis = 1)

    if np.isnan(tmp_results[-1]["correlation"]).all():
        tmp_results[-1]["recommendation_quality"] = np.mean(np.concatenate([np.expand_dims(tmp_results[-1]["recommendation_ranking"], 1),
                                                                            np.expand_dims(tmp_results[-1]["score"], 1),
                                                                            np.expand_dims(tmp_results[-1]["interaction_ranking"], 1)], axis = 1), axis = 1)
    else:
        tmp_results[-1]["correlation"] = np.nan_to_num(tmp_results[-1]["correlation"])
        tmp_results[-1]["recommendation_quality"] = np.mean(np.concatenate([np.expand_dims(tmp_results[-1]["correlation"], 1),
                                                                            np.expand_dims(tmp_results[-1]["score"], 1),
                                                                            np.expand_dims(tmp_results[-1]["recommendation_ranking"], 1),
                                                                            np.expand_dims(tmp_results[-1]["interaction_ranking"], 1)], axis = 1), axis = 1)

    if params["num_forced_items"] > 0 and params["forced_period"] > 0:
        tmp_results[-1]["forced_interaction_histogram"] = tmp_results[-1]["interaction_histogram"][:, tmp_results[-1]["forced_items"]]
        if running_setting["save_debug_info"]:
            tmp_results[-1]["forced_recommendation_histogram"] = tmp_results[-1]["recommendation_histogram"][:, tmp_results[-1]["forced_items"]]
        forced_modified_ih = windowed_modified_ih[:, tmp_results[-1]["forced_items"]]
        tmp_results[-1]["forced_cn"] = np.sum(forced_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True), axis = 1)
    else:
        tmp_results[-1]["forced_cn"] = np.zeros_like(tmp_results[-1]["cn"])

    return tmp_results


# Main plotting function
def plot_results(params, results, savepath, running_setting, has_std = True):
    training = params["training"]
    timesteps = params["timesteps"]

    os.makedirs(savepath, exist_ok = True)

    plot_modes = {"": [training + timesteps, 1, 0],
                   "_no_startup": [timesteps + 1, -timesteps, 1]}

    if running_setting["save_pdf_plots"]:
        for mode in plot_modes.keys():
            # Plot the market shares of the newly created items, if any
            if params["num_creators"] > 0:
                grid = plt.GridSpec(int(math.ceil(len(results.keys()) / 2)), 4, wspace = 0.8, hspace = 0.9)
                plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(len(results.keys()) / 2)) - 1)], [])
                if len(results.keys()) % 2 == 0:
                    plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 2:])]
                elif len(results.keys()) == 1:
                    plots = [plt.subplot(grid[0, 0:4])]
                else:
                    plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 1:3])]
                for i, key in enumerate(results.keys()):
                    # plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:], color = "C0")
                    plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn_entrant"][plot_modes[mode][1]:], color = "C1")
                    plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:], color = "C2")
                    if has_std:
                        # plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:] - results[key]["cn_std"][plot_modes[mode][1]:], results[key]["cn"][plot_modes[mode][1]:] + results[key]["cn_std"][plot_modes[mode][1]:], color = "C0", alpha = 0.3)
                        plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn_entrant"][plot_modes[mode][1]:] - results[key]["cn_entrant_std"][plot_modes[mode][1]:], results[key]["cn_entrant"][plot_modes[mode][1]:] + results[key]["cn_entrant_std"][plot_modes[mode][1]:], color = "C1", alpha = 0.3)
                        plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:] - results[key]["recommendation_quality_std"][plot_modes[mode][1]:], results[key]["recommendation_quality"][plot_modes[mode][1]:] + results[key]["recommendation_quality_std"][plot_modes[mode][1]:], color = "C2", alpha = 0.3)
                    plots[i].set_title(key)
                    plots[i].set_ylim(-0.05, 1.05)
                    plots[i].set_xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
                if len(results.keys()) % 2 == 0:
                    plots[-1].set_xlabel("Timestep")
                    plots[-2].set_xlabel("Timestep")
                else:
                    plots[-1].set_xlabel("Timestep")
                for i in range(int(math.ceil(len(results.keys()) / 2))):
                    plots[2 * i].set_ylabel("Measure Value")
                first_leg = mpatches.Patch(color = "C0", label = "Incumbents MS")
                second_leg = mpatches.Patch(color = "C1", label = "Entrants MS")
                third_leg = mpatches.Patch(color = "C2", label = "Rec. Quality")
                plt.legend(handles = [second_leg, third_leg], loc = "best", bbox_to_anchor = (1, 0.5))
                # plt.show()
                plt.savefig(os.path.join(savepath, "Entry" + mode + ".pdf"), bbox_inches = "tight")
                plt.clf()

            # Plot the market shares of the forcefully shown items, if any
            if params["num_forced_items"] > 0 and params["forced_period"] > 0:
                grid = plt.GridSpec(int(math.ceil(len(results.keys()) / 2)), 4, wspace = 0.8, hspace = 0.9)
                plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(len(results.keys()) / 2)) - 1)], [])
                if len(results.keys()) % 2 == 0:
                    plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 2:])]
                elif len(results.keys()) == 1:
                    plots = [plt.subplot(grid[0, 0:4])]
                else:
                    plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 1:3])]
                for i, key in enumerate(results.keys()):
                    plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["forced_cn"][plot_modes[mode][1]:], color = "C1")
                    plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:], color = "C2")
                    if has_std:
                        plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["forced_cn"][plot_modes[mode][1]:] - results[key]["forced_cn_std"][plot_modes[mode][1]:], results[key]["forced_cn"][plot_modes[mode][1]:] + results[key]["forced_cn_std"][plot_modes[mode][1]:], color = "C1", alpha = 0.3)
                        plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:] - results[key]["recommendation_quality_std"][plot_modes[mode][1]:], results[key]["recommendation_quality"][plot_modes[mode][1]:] + results[key]["recommendation_quality_std"][plot_modes[mode][1]:], color = "C2", alpha = 0.3)
                    plots[i].set_title(key)
                    plots[i].set_ylim(-0.05, 1.05)
                    plots[i].set_xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
                if len(results.keys()) % 2 == 0:
                    plots[-1].set_xlabel("Timestep")
                    plots[-2].set_xlabel("Timestep")
                else:
                    plots[-1].set_xlabel("Timestep")
                for i in range(int(math.ceil(len(results.keys()) / 2))):
                    plots[2 * i].set_ylabel("Measure Value")
                first_leg = mpatches.Patch(color = "C1", label = "Forced Items MS")
                second_leg = mpatches.Patch(color = "C2", label = "Rec. Quality")
                plt.legend(handles = [first_leg, second_leg], loc = "best", bbox_to_anchor = (1, 0.5))
                # plt.show()
                plt.savefig(os.path.join(savepath, "Forced_Items_Concentration" + mode + ".pdf"), bbox_inches = "tight")
                plt.clf()

            measures = {"interaction": "Interaction",
                        "rec": "Recommendation"}
            int_homogeneities = {"_similarity": ["Jaccard Similarity", "_Jaccard_Homogeneity"]}
            rec_homogeneities = {"_similarity": ["Jaccard Similarity", "_Jaccard_Homogeneity"]}
            pairs = {"_most": "_Most_Similar_Pairs",
                     "_least": "_Least_Similar_Pairs"}

            if running_setting["more_computation"]:
                int_homogeneities["_attr_similarity"] = ["Attributes Similarity", "_Attr_Homogeneity"]
                rec_homogeneities["_attr_similarity"] = ["Attributes Similarity", "_Attr_Homogeneity"]
                rec_homogeneities["_summed_attr_similarity"] = ["Attributes Similarity", "_Summed_Attr_Homogeneity"]
                pairs["_all"] = "_All_Pairs"

            for m in measures.keys():
                for p in pairs.keys():
                    if m == "rec":
                        homogeneities = rec_homogeneities
                    else:
                        homogeneities = int_homogeneities
                    for h in homogeneities.keys():
                        for i, key in enumerate(results.keys()):
                            plt.plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key][m + h + p][plot_modes[mode][1]:], color = "C" + str(i), label = key)
                            if has_std:
                                plt.fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key][m + h + p][plot_modes[mode][1]:] - results[key][m + h + p + "_std"][plot_modes[mode][1]:], results[key][m + h + p][plot_modes[mode][1]:] + results[key][m + h + p + "_std"][plot_modes[mode][1]:], color = "C" + str(i), alpha = 0.3)
                        plt.ylim(-0.05, 1.05)
                        plt.xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
                        plt.xlabel("Timestep")
                        plt.ylabel(homogeneities[h][0])
                        plt.legend()
                        # plt.show()
                        plt.savefig(os.path.join(savepath, measures[m] + homogeneities[h][1] + pairs[p] + mode + ".pdf"), bbox_inches = "tight")
                        plt.clf()

                    if running_setting["more_computation"]:
                        # Plot comparison of the various homogeneity measures
                        grid = plt.GridSpec(int(math.ceil(len(results.keys()) / 2)), 4, wspace = 0.8, hspace = 0.9)
                        plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(len(results.keys()) / 2)) - 1)], [])
                        if len(results.keys()) % 2 == 0:
                            plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 2:])]
                        elif len(results.keys()) == 1:
                            plots = [plt.subplot(grid[0, 0:4])]
                        else:
                            plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 1:3])]
                        for i, key in enumerate(results.keys()):
                            for j, h in enumerate(homogeneities.keys()):
                                plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key][m + h + p][plot_modes[mode][1]:], color = "C" + str(j))
                                if has_std:
                                    plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key][m + h + p][plot_modes[mode][1]:] - results[key][m + h + p + "_std"][plot_modes[mode][1]:], results[key][m + h + p][plot_modes[mode][1]:] + results[key][m + h + p + "_std"][plot_modes[mode][1]:], color = "C" + str(j), alpha = 0.3)
                            plots[i].set_title(key)
                            plots[i].set_ylim(-0.05, 1.05)
                            plots[i].set_xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
                        if len(results.keys()) % 2 == 0:
                            plots[-1].set_xlabel("Timestep")
                            plots[-2].set_xlabel("Timestep")
                        else:
                            plots[-1].set_xlabel("Timestep")
                        for i in range(int(math.ceil(len(results.keys()) / 2))):
                            plots[2 * i].set_ylabel("Measure Value")
                        first_leg = mpatches.Patch(color = "C0", label = "Jaccard")
                        second_leg = mpatches.Patch(color = "C1", label = "Attributes")
                        if m == "rec":
                            third_leg = mpatches.Patch(color = "C2", label = "Summed Attributes")
                            plt.legend(handles = [first_leg, second_leg, third_leg], loc = "best", bbox_to_anchor = (1, 0.5))
                        else:
                            plt.legend(handles = [first_leg, second_leg], loc = "best", bbox_to_anchor = (1, 0.5))
                        # plt.show()
                        plt.savefig(os.path.join(savepath, "Compare_" + measures[m] + "_Homogeneities" + pairs[p] + mode + ".pdf"), bbox_inches = "tight")
                        plt.clf()

            # Plot concentration
            for i, key in enumerate(results.keys()):
                plt.plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:], color = "C" + str(i), label = key)
                if has_std:
                    plt.fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:] - results[key]["cn_std"][plot_modes[mode][1]:], results[key]["cn"][plot_modes[mode][1]:] + results[key]["cn_std"][plot_modes[mode][1]:], color = "C" + str(i), alpha = 0.3)
            plt.ylim(-0.05, 1.05)
            plt.xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
            plt.ylabel("Market share of the 10% largest items")
            plt.xlabel("Timestep")
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(savepath, "Concentration" + mode + ".pdf"), bbox_inches = "tight")
            plt.clf()

            # Plot HHI
            for i, key in enumerate(results.keys()):
                plt.plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["hhi"][plot_modes[mode][1]:], color = "C" + str(i), label = key)
                if has_std:
                    plt.fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["hhi"][plot_modes[mode][1]:] - results[key]["hhi_std"][plot_modes[mode][1]:], results[key]["hhi"][plot_modes[mode][1]:] + results[key]["hhi_std"][plot_modes[mode][1]:], color = "C" + str(i), alpha = 0.3)
            plt.ylim(-0.05, 1.05)
            plt.xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
            plt.ylabel("HHI")
            plt.xlabel("Timestep")
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(savepath, "HHI" + mode + ".pdf"), bbox_inches = "tight")
            plt.clf()

            # Plot quality of recommendation for each RS
            grid = plt.GridSpec(int(math.ceil(len(results.keys()) / 2)), 4, wspace = 0.8, hspace = 0.9)
            plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(len(results.keys()) / 2)) - 1)], [])
            if len(results.keys()) % 2 == 0:
                plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 2:])]
            elif len(results.keys()) == 1:
                plots = [plt.subplot(grid[0, 0:4])]
            else:
                plots = plots + [plt.subplot(grid[int(math.ceil(len(results.keys()) / 2)) - 1, 1:3])]
            for i, key in enumerate(results.keys()):
                plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:], color = "C0")
                plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["rec_similarity_most"][plot_modes[mode][1]:], color = "C1")
                plots[i].plot(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:], color = "C2")
                if has_std:
                    plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["cn"][plot_modes[mode][1]:] - results[key]["cn_std"][plot_modes[mode][1]:], results[key]["cn"][plot_modes[mode][1]:] + results[key]["cn_std"][plot_modes[mode][1]:], color = "C0", alpha = 0.3)
                    plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["rec_similarity_most"][plot_modes[mode][1]:] - results[key]["rec_similarity_most_std"][plot_modes[mode][1]:],  results[key]["rec_similarity_most"][plot_modes[mode][1]:] + results[key]["rec_similarity_most_std"][plot_modes[mode][1]:], color = "C1", alpha = 0.3)
                    plots[i].fill_between(np.arange(plot_modes[mode][2], plot_modes[mode][0]), results[key]["recommendation_quality"][plot_modes[mode][1]:] - results[key]["recommendation_quality_std"][plot_modes[mode][1]:], results[key]["recommendation_quality"][plot_modes[mode][1]:] + results[key]["recommendation_quality_std"][plot_modes[mode][1]:], color = "C2", alpha = 0.3)
                plots[i].set_title(key)
                plots[i].set_ylim(-0.05, 1.05)
                plots[i].set_xticks([plot_modes[mode][2]] + list(range(0, plot_modes[mode][0], 20))[1:])
                if len(results.keys()) % 2 == 0:
                    plots[-1].set_xlabel("Timestep")
                    plots[-2].set_xlabel("Timestep")
                else:
                    plots[-1].set_xlabel("Timestep")
                for j in range(int(math.ceil(len(results.keys()) / 2))):
                    plots[2 * j].set_ylabel("Measure Value")
            first_leg = mpatches.Patch(color = "C0", label = "C10%")
            second_leg = mpatches.Patch(color = "C1", label = "Jaccard (Most Similar Pairs)")
            third_leg = mpatches.Patch(color = "C2", label = "Rec. Quality")
            plt.legend(handles = [first_leg, second_leg, third_leg], loc = "best", bbox_to_anchor = (1, 0.5))
            # plt.show()
            plt.savefig(os.path.join(savepath, "Quality" + mode + ".pdf"), bbox_inches = "tight")
            plt.clf()

        # Plot MSE and other metrics
        for key in results.keys():
            fig, axs = plt.subplots(5, 2)
            axs[0, 0].plot(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["mse"][plot_modes[""][1]:])
            axs[0, 0].set_title("All timeperiods")
            axs[0, 0].set_ylabel("MSE")
            axs[0, 1].plot(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["mse"][plot_modes["_no_startup"][1]:])
            axs[0, 1].set_title("After startup")
            axs[0, 1].set_xticks([plot_modes["_no_startup"][2]] + list(range(0, plot_modes["_no_startup"][0], 25))[1:])

            axs[1, 0].plot(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["correlation"][plot_modes[""][1]:])
            axs[1, 0].set_ylabel("Corr. Coeff.")
            axs[1, 1].plot(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["correlation"][plot_modes["_no_startup"][1]:])
            axs[1, 1].set_xticks([plot_modes["_no_startup"][2]] + list(range(0, plot_modes["_no_startup"][0], 25))[1:])

            axs[2, 0].plot(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["recommendation_ranking"][plot_modes[""][1]:])
            axs[2, 0].set_ylabel("Rec. Rank.")
            axs[2, 1].plot(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["recommendation_ranking"][plot_modes["_no_startup"][1]:])
            axs[2, 1].set_xticks([plot_modes["_no_startup"][2]] + list(range(0, plot_modes["_no_startup"][0], 25))[1:])

            axs[3, 0].plot(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["interaction_ranking"][plot_modes[""][1]:])
            axs[3, 0].set_ylabel("Int. Rank.")
            axs[3, 1].plot(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["interaction_ranking"][plot_modes["_no_startup"][1]:])
            axs[3, 1].set_xticks([plot_modes["_no_startup"][2]] + list(range(0, plot_modes["_no_startup"][0], 25))[1:])

            axs[4, 0].plot(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["score"][plot_modes[""][1]:])
            axs[4, 0].set_ylabel("Score ratio")
            axs[4, 0].set_xlabel("Timestep")
            axs[4, 1].plot(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["score"][plot_modes["_no_startup"][1]:])
            axs[4, 1].set_xlabel("Timestep")
            axs[4, 1].set_xticks([plot_modes["_no_startup"][2]] + list(range(0, plot_modes["_no_startup"][0], 25))[1:])
            if has_std:
                axs[0, 0].fill_between(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["mse"][plot_modes[""][1]:] - results[key]["mse_std"][plot_modes[""][1]:], results[key]["mse"][plot_modes[""][1]:] + results[key]["mse_std"][plot_modes[""][1]:], alpha = 0.3)
                axs[0, 1].fill_between(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["mse"][plot_modes["_no_startup"][1]:] - results[key]["mse_std"][plot_modes["_no_startup"][1]:], results[key]["mse"][plot_modes["_no_startup"][1]:] + results[key]["mse_std"][plot_modes["_no_startup"][1]:], alpha = 0.3)
                axs[1, 0].fill_between(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["correlation"][plot_modes[""][1]:] - results[key]["correlation_std"][plot_modes[""][1]:], results[key]["correlation"][plot_modes[""][1]:] + results[key]["correlation_std"][plot_modes[""][1]:], alpha = 0.3)
                axs[1, 1].fill_between(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["correlation"][plot_modes["_no_startup"][1]:] - results[key]["correlation_std"][plot_modes["_no_startup"][1]:], results[key]["correlation"][plot_modes["_no_startup"][1]:] + results[key]["correlation_std"][plot_modes["_no_startup"][1]:], alpha = 0.3)
                axs[2, 0].fill_between(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["recommendation_ranking"][plot_modes[""][1]:] - results[key]["recommendation_ranking_std"][plot_modes[""][1]:], results[key]["recommendation_ranking"][plot_modes[""][1]:] + results[key]["recommendation_ranking_std"][plot_modes[""][1]:], alpha = 0.3)
                axs[2, 1].fill_between(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["recommendation_ranking"][plot_modes["_no_startup"][1]:] - results[key]["recommendation_ranking_std"][plot_modes["_no_startup"][1]:], results[key]["recommendation_ranking"][plot_modes["_no_startup"][1]:] + results[key]["recommendation_ranking_std"][plot_modes["_no_startup"][1]:], alpha = 0.3)
                axs[3, 0].fill_between(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["interaction_ranking"][plot_modes[""][1]:] - results[key]["interaction_ranking_std"][plot_modes[""][1]:], results[key]["interaction_ranking"][plot_modes[""][1]:] + results[key]["interaction_ranking_std"][plot_modes[""][1]:], alpha = 0.3)
                axs[3, 1].fill_between(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["interaction_ranking"][plot_modes["_no_startup"][1]:] - results[key]["interaction_ranking_std"][plot_modes["_no_startup"][1]:], results[key]["interaction_ranking"][plot_modes["_no_startup"][1]:] + results[key]["interaction_ranking_std"][plot_modes["_no_startup"][1]:], alpha = 0.3)
                axs[4, 0].fill_between(np.arange(plot_modes[""][2], plot_modes[""][0]), results[key]["score"][plot_modes[""][1]:] - results[key]["score_std"][plot_modes[""][1]:], results[key]["score"][plot_modes[""][1]:] + results[key]["score_std"][plot_modes[""][1]:], alpha = 0.3)
                axs[4, 1].fill_between(np.arange(plot_modes["_no_startup"][2], plot_modes["_no_startup"][0]), results[key]["score"][plot_modes["_no_startup"][1]:] - results[key]["score_std"][plot_modes["_no_startup"][1]:], results[key]["score"][plot_modes["_no_startup"][1]:] + results[key]["score_std"][plot_modes["_no_startup"][1]:], alpha = 0.3)

            fig.subplots_adjust(hspace = 0.5)
            fig.suptitle(f"Learning measures over time for {key}")
            plt.savefig(os.path.join(savepath, "Measures_" + key + ".pdf"), figsize = (8, 12), bbox_inches = "tight")
            plt.clf()

        # Plot the interaction matrix for the right RS
        for key in results.keys():
            if key == "content_based" or key == "collaborative_filtering":
                if np.any(results[key]["all_interactions"]):
                    fig, axs = plt.subplots(2 if training > 0 else 1)
                    yticks = ["User " + str(i + 1) for i in range(params["num_users"])]
                    xticks = [str(i) for i in np.where(np.any(results[key]["all_interactions"], axis = 0))[0]]
                    sns.heatmap(results[key]["all_interactions"][:, np.any(results[key]["all_interactions"], axis = 0)], ax = axs[0], cmap = "YlOrRd", yticklabels = yticks, xticklabels = xticks, vmin = 0)
                    axs[0].set_title("All timesteps")
                    axs[0].set_ylabel("Users")

                if training > 0 and np.any(results[key]["all_interactions_after_startup"]):
                    xticks = [str(i) for i in np.where(np.any(results[key]["all_interactions_after_startup"], axis = 0))[0]]
                    sns.heatmap(results[key]["all_interactions_after_startup"][:, np.any(results[key]["all_interactions_after_startup"], axis = 0)], ax = axs[1], cmap = "YlOrRd", yticklabels = yticks, xticklabels = xticks, vmin = 0)
                    axs[1].set_title("After startup")
                    axs[1].set_ylabel("Users")
                    axs[1].set_xlabel("Items")

                if np.any(results[key]["all_interactions"]):
                    fig.subplots_adjust(hspace = 0.5)
                    plt.savefig(os.path.join(savepath, "Interactions_" + key + ".pdf"), figsize = (8, 12), bbox_inches = "tight")
                    plt.clf()
            plt.close()

    # Save all the metrics and measures into a csv file for later use
    if running_setting["more_computation"]:
        measures = ["cn", "cn_entrant", "forced_cn", "hhi", "interaction_similarity_most", "interaction_similarity_all", "interaction_similarity_least", "interaction_attr_similarity_most", "interaction_attr_similarity_all", "interaction_attr_similarity_least", "rec_similarity_most", "rec_similarity_all", "rec_similarity_least", "rec_attr_similarity_most", "rec_attr_similarity_all", "rec_attr_similarity_least", "rec_summed_attr_similarity_most", "rec_summed_attr_similarity_all", "rec_summed_attr_similarity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]
        labels = ["concentration", "entry", "forced_items_concentration", "hhi", "interaction_homogeneity_most", "interaction_homogeneity_all", "interaction_homogeneity_least", "interaction_attr_homogeneity_most", "interaction_attr_homogeneity_all", "interaction_attr_homogeneity_least", "rec_homogeneity_most", "rec_homogeneity_all", "rec_homogeneity_least", "rec_attr_homogeneity_most", "rec_attr_homogeneity_all", "rec_attr_homogeneity_least", "rec_summed_attr_homogeneity_most", "rec_summed_attr_homogeneity_all", "rec_summed_attr_homogeneity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]
    else:
        measures = ["cn", "cn_entrant", "forced_cn", "hhi", "interaction_similarity_most", "interaction_similarity_least", "rec_similarity_most", "rec_similarity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]
        labels = ["concentration", "entry", "forced_items_concentration", "hhi", "interaction_homogeneity_most", "interaction_homogeneity_least", "rec_homogeneity_most", "rec_homogeneity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]

    if has_std:
        measures = sum([[m, m + "_std"] for m in measures], [])
        labels = sum([[m, m + "_std"] for m in labels], [])

    mux = pd.Index(results.keys(), name = "recommender_system")

    for mode in plot_modes.keys():
        data = np.stack([np.concatenate([np.expand_dims(v["timesteps"], 1)[plot_modes[mode][2]:plot_modes[mode][0]], np.concatenate([np.expand_dims(v[m], 1)[plot_modes[mode][1]:] for m in measures], axis = 1)], axis = 1) for v in results.values()])
        raw_sample = []
        for i in range(data.shape[0]):
            raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
        df = pd.concat(raw_sample, keys = mux)
        df.reset_index(level = 1, drop = True, inplace = True)
        # print(df)
        df.to_csv(os.path.join(savepath, "Data" + mode + ".csv"))

    if running_setting["save_debug_info"]:
        if params["runs"] == 1 or not has_std:
            csv_modes = {"": ""}
            if params["num_forced_items"] > 0 and params["forced_period"] > 0:
                csv_modes["forced_"] = "Forced_Items_"

            for new_mode in csv_modes:
                # And saving complete history of execution
                # For interactions
                tot_items = np.max([results[k][new_mode + "interaction_histogram"].shape[1] for k in results.keys()])
                labels = [new_mode + "item_" + str(i + 1) for i in range(tot_items)]

                for mode in plot_modes:
                    data = np.stack([np.concatenate([np.expand_dims(v["timesteps"], 1)[plot_modes[mode][2]:plot_modes[mode][0]], np.concatenate([np.expand_dims(v[new_mode + "interaction_histogram"][:, i], 1) for i in range(v[new_mode + "interaction_histogram"].shape[1])], axis = 1)[plot_modes[mode][1]:]], axis = 1) for v in results.values()])
                    raw_sample = []
                    for i in range(data.shape[0]):
                        raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                    df = pd.concat(raw_sample, keys = mux)
                    df.reset_index(level = 1, drop = True, inplace = True)
                    # print(df)
                    df.to_csv(os.path.join(savepath, csv_modes[new_mode] + "Histogram_Interactions" + mode + ".csv"))

                # And for recommendations
                for mode in plot_modes:
                    data = np.stack([np.concatenate([np.expand_dims(v["timesteps"], 1)[plot_modes[mode][2]:plot_modes[mode][0]], np.concatenate([np.expand_dims(v[new_mode + "recommendation_histogram"][:, i], 1) for i in range(v[new_mode + "recommendation_histogram"].shape[1])], axis = 1)[plot_modes[mode][1]:]], axis = 1) for v in results.values()])
                    raw_sample = []
                    for i in range(data.shape[0]):
                        raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                    df = pd.concat(raw_sample, keys = mux)
                    df.reset_index(level = 1, drop = True, inplace = True)
                    # print(df)
                    df.to_csv(os.path.join(savepath, csv_modes[new_mode] + "Histogram_Recommendations" + mode + ".csv"))

            # Finally, saving complete history of execution per each user
            # For interactions
            labels = ["user_" + str(i + 1) + "_interaction" for i in range(params["num_users"])]

            for mode in plot_modes:
                data = np.stack([np.concatenate([np.expand_dims(v["timesteps"], 1)[plot_modes[mode][2]:plot_modes[mode][0]], v["interaction_history"][plot_modes[mode][1]:]], axis = 1) for v in results.values()])
                raw_sample = []
                for i in range(data.shape[0]):
                    raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                df = pd.concat(raw_sample, keys = mux)
                df.reset_index(level = 1, drop = True, inplace = True)
                # print(df)
                df.to_csv(os.path.join(savepath, "History_Interactions" + mode + ".csv"))

            # And for recommendations
            labels = sum([["user_" + str(i + 1) + "_recommendation_" + str(j + 1) for j in range(params["num_items_per_iter"])] for i in range(params["num_users"])], [])

            for mode in plot_modes:
                data = np.stack([np.concatenate([np.expand_dims(v["timesteps"], 1)[plot_modes[mode][2]:plot_modes[mode][0]], np.array(v["recommendation_history"])[plot_modes[mode][1]:].reshape(-1, params["num_users"] * params["num_items_per_iter"])], axis = 1) for v in results.values()])
                raw_sample = []
                for i in range(data.shape[0]):
                    raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                df = pd.concat(raw_sample, keys = mux)
                df.reset_index(level = 1, drop = True, inplace = True)
                # print(df)
                df.to_csv(os.path.join(savepath, "History_Recommendations" + mode + ".csv"))


# Plotting of each run separately
def plot_tmp_results(params, tmp_results, savepath, running_setting):
    for r in range(params["runs"]):
        new_tmp_results = {key: tmp_results[key][r] for key in tmp_results.keys()}
        tmp_savepath = os.path.join(savepath, "run_%02d" % (r + 1))  if params["runs"] > 1 else savepath

        plot_results(params, new_tmp_results, tmp_savepath, running_setting, False)

        if running_setting["save_debug_info"]:
            # Save info on users, items etc
            for k in tmp_results.keys():
                suffix = "_" + k if params["num_creators"] > 0 else ""
                df = pd.DataFrame(new_tmp_results[k]["real_profiles"], columns = ["attr_" + str(i + 1) for i in range(params["num_attributes"])])
                df.index.name = "users"
                df.to_csv(os.path.join(tmp_savepath, "Users_Profiles" + suffix + ".csv"))

                df = pd.DataFrame(new_tmp_results[k]["real_scores"], columns = ["item_" + str(i + 1) for i in range(new_tmp_results[k]["real_attributes"].shape[0])])
                df.index.name = "users"
                df.to_csv(os.path.join(tmp_savepath, "Users_Scores" + suffix + ".csv"))

                df = pd.DataFrame(new_tmp_results[k]["real_attributes"], columns = ["attr_" + str(i + 1) for i in range(params["num_attributes"])])
                df.index.name = "items"
                df.to_csv(os.path.join(tmp_savepath, "Items_Attributes" + suffix + ".csv"))

                if params["num_creators"] > 0:
                    df = pd.DataFrame(new_tmp_results[k]["real_types"], columns = ["attr_" + str(i + 1) for i in range(params["num_attributes"])])
                    df.index.name = "creators"
                    df.to_csv(os.path.join(tmp_savepath, "Creators_Profiles" + suffix + ".csv"))

                if params["num_forced_items"] > 0 and params["forced_period"] > 0:
                    df = pd.DataFrame(np.concatenate([np.expand_dims(new_tmp_results[k]["forced_items"], 1), new_tmp_results[k]["forced_real_attributes"]], axis = 1), columns = ["idx"] + ["attr_" + str(i + 1) for i in range(params["num_attributes"])])
                    df.index.name = "forced_items"
                    df.to_csv(os.path.join(tmp_savepath, "Forced_Items_Attributes" + suffix + ".csv"))

                if k == "content_based":
                    labels = ["attr_" + str(i + 1) for i in range(params["num_attributes"])]
                    data = np.stack([np.concatenate([np.expand_dims(new_tmp_results[k]["timesteps"], 1), np.array(new_tmp_results[k]["nnls_coefficients_x"])[:, i]], axis = 1) for i in range(params["num_users"])])
                    mux = pd.Index(["user_" + str(i + 1) for i in range(params["num_users"])], name = "users")

                    raw_sample = []
                    for i in range(data.shape[0]):
                        raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                    df = pd.concat(raw_sample, keys = mux)
                    df.reset_index(level = 1, drop = True, inplace = True)
                    # print(df)
                    df.to_csv(os.path.join(tmp_savepath, "NNLS_Coefficient_x.csv"))

                    data = np.stack([np.concatenate([np.expand_dims(new_tmp_results[k]["timesteps"], 1), np.array(adjust_results_2d(new_tmp_results[k]["nnls_coefficients_A"]))[:, i]], axis = 1) for i in range(new_tmp_results[k]["real_attributes"].shape[0])])
                    mux = pd.Index(["item_" + str(i + 1) for i in range(new_tmp_results[k]["real_attributes"].shape[0])], name = "items")

                    raw_sample = []
                    for i in range(data.shape[0]):
                        raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                    df = pd.concat(raw_sample, keys = mux)
                    df.reset_index(level = 1, drop = True, inplace = True)
                    # print(df)
                    df.to_csv(os.path.join(tmp_savepath, "NNLS_Coefficient_A.csv"))

                    labels = ["int_item_" + str(i + 1) for i in range(new_tmp_results[k]["real_attributes"].shape[0])]
                    data = np.stack([np.concatenate([np.expand_dims(new_tmp_results[k]["timesteps"], 1), np.array(adjust_results_2d_opposite(new_tmp_results[k]["nnls_coefficients_b"]))[:, i]], axis = 1) for i in range(params["num_users"])])
                    mux = pd.Index(["user_" + str(i + 1) for i in range(params["num_users"])], name = "users")

                    raw_sample = []
                    for i in range(data.shape[0]):
                        raw_sample.append(pd.DataFrame(data[i], columns = ["timesteps"] + labels))
                    df = pd.concat(raw_sample, keys = mux)
                    df.reset_index(level = 1, drop = True, inplace = True)
                    # print(df)
                    df.to_csv(os.path.join(tmp_savepath, "NNLS_Coefficient_b.csv"))


# Combine results from multiple runs into mean and std results
def results_from_tmp_results(results, full_tmp_results, key, tmp_results):
    for k in tmp_results[-1].keys():
        if k != "nnls_coefficients_x" and k != "nnls_coefficients_A" and k != "nnls_coefficients_b":
            concat = np.concatenate([np.expand_dims(t[k], -1) for t in tmp_results], axis = -1)
            results[key][k] = np.mean(concat, axis = -1)
            results[key][k + "_std"] = np.std(concat, axis = -1)

    full_tmp_results[key] = tmp_results

    return results, full_tmp_results


# Convert the dataframe used by ImplicitMF to represent all_interactions to a sparse matrix
def dataframe_to_sparse(df, num_users, num_items):
    vals = np.array(df.values)
    rows = vals[:, 0]
    cols = vals[:, 1]
    tmp = np.ones(rows.shape)
    return sparse.coo_matrix((tmp, (rows, cols)), shape = (num_users, num_items)).tocsr()


# Used to sweep over entire config files with list of values
def non_list_values_to_lists(input_dict: Dict) -> Dict:
    """
    Takes a dictionary where some values may not be lists and makes them all lists
    """

    for k, v in input_dict.items():
        if not isinstance(v, list):
            input_dict[k] = [v]

    return input_dict

def dict_of_lists_to_list_of_dicts(dict_of_lists: Dict[Any, List[Any]]) -> List[Dict[Any, Any]]:
    """
    Assumes that all values of the input dict are lists

    These lists can have different lengths

    The output is a list of dicts where all values are not lists, and we have
    one list element for every element in the Cartesian product of the lists of
    values in the original input dict
    """

    keys = list(dict_of_lists.keys())
    list_of_values = [dict_of_lists[key] for key in keys]
    product = list(itertools.product(*list_of_values))

    return [dict(zip(keys, product_elem)) for product_elem in product]

def process_mixed_dict(input_dict: Dict) -> Dict:
    tmp = non_list_values_to_lists(input_dict)
    return dict_of_lists_to_list_of_dicts(tmp)


# Used to create a params file with sweep configs
def write_params_file(input_dict, savepath):
    with open(os.path.join(savepath, "params.py"), "w+") as f:
        f.write("params = dict(\n")
        for k in input_dict.keys():
            if type(input_dict[k]) == str:
                txt = "'" + input_dict[k] + "'"
            else:
                txt = str(input_dict[k])
            f.write("    " + str(k) + " = " + txt + ",\n")
        f.write(")")


# Master CSV file with all runs stacked together
def master_csv(savepath, all_params, running_setting):
    mux = pd.Index([p["exp_index"] for p in all_params], name = "exp_index")
    if running_setting["more_computation"]:
        labels = ["recommender_system", "timesteps", "concentration", "entry", "forced_items_concentration", "hhi", "interaction_homogeneity_most", "interaction_homogeneity_all", "interaction_homogeneity_least", "interaction_attr_homogeneity_most", "interaction_attr_homogeneity_all", "interaction_attr_homogeneity_least", "rec_homogeneity_most", "rec_homogeneity_all", "rec_homogeneity_least", "rec_attr_homogeneity_most", "rec_attr_homogeneity_all", "rec_attr_homogeneity_least", "rec_summed_attr_homogeneity_most", "rec_summed_attr_homogeneity_all", "rec_summed_attr_homogeneity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]
    else:
        labels = ["recommender_system", "timesteps", "concentration", "entry", "forced_items_concentration", "hhi", "interaction_homogeneity_most", "interaction_homogeneity_least", "rec_homogeneity_most", "rec_homogeneity_least", "mse", "score", "correlation", "recommendation_ranking", "interaction_ranking", "recommendation_quality"]

    for app in ["", "_no_startup"]:
        all_sample = []
        for i in range(len(all_params)):
            raw_sample = []
            runs = all_params[i]["runs"]
            mux2 = pd.Index(range(1, runs + 1), name = "run")
            for r in range(runs):
                data = pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(savepath, "config_%03d" % (i + 1) if len(all_params) > 1 else "")), "run_%02d" % (r + 1) if runs > 1 else ""), "Data" + app + ".csv"), usecols = labels, header = 0)
                raw_sample.append(pd.DataFrame(data))
            df = pd.concat(raw_sample, keys = mux2)
            all_sample.append(df)
        df2 = pd.concat(all_sample, keys = mux)
        df2.reset_index(level = 2, drop = True, inplace = True)
        # print(df2)
        df2.to_csv(os.path.join(savepath, "Params_Measures" + app + ".csv"))