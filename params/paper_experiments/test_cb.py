params = dict(
    num_users = 5,
    num_items = 20,
    num_creators = 1,
    p_creation = 1.,
    attention_exp = 0,
    drift = 0,
    training = 10,
    timesteps = 20,
    train_between_steps = True,
    num_attributes = 10,
    min_preference_per_attribute = -50,
    max_preference_per_attribute = 150,
    runs = 2,
    num_items_per_iter = 1,
    random_newly_created = False,
    random_items_per_iter= 0,
    repeated_items = True,
    probabilistic_recommendations = False,
    horizontally_differentiated_only = True,
    num_forced_items = 0,
    forced_period = 0,
    softmax_mult_for_forced_items = 1,
    individual_rationality = True,
    sort_rec_per_popularity = False,
    prop_items_for_concentration_metric = 0.1,
    cf_model_params = dict(iterations = 20, reg = 0.1, weight = 40),
)