params = dict(
    num_users = 100,
    num_items = 200,
    num_creators = 10,
    p_creation = 1.,
    attention_exp = [-1.0, 0],
    drift = 0,
    training = 10,
    timesteps = 100,
    train_between_steps = True,
    num_attributes = 30,
    min_preference_per_attribute = [-50, -100],
    max_preference_per_attribute = 100,
    runs = 30,
    num_items_per_iter = 10,
    random_newly_created = False,
    random_items_per_iter= [0, 0.5],
    repeated_items = True,
    probabilistic_recommendations = [True, False],
    horizontally_differentiated_only = [True, False],
    num_forced_items = 0,
    forced_period = 0,
    softmax_mult_for_forced_items = 1,
    individual_rationality = True,
    sort_rec_per_popularity = [False, True],
    prop_items_for_concentration_metric = 0.1,
    cf_num_latent_factors = 30,
)