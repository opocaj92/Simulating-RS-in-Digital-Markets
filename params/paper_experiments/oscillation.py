params = dict(
    num_users = 100,
    num_items = 200,
    num_creators = 10,
    p_creation = 1.,
    attention_exp = -1.0,
    drift = 0,
    training = 10,
    timesteps = 100,
    train_between_steps = True,
    num_attributes = 200,
    min_preference_per_attribute = -50,
    max_preference_per_attribute = 150,
    runs = 10,
    num_items_per_iter = 10,
    random_newly_created = False,
    random_items_per_iter= 0,
    repeated_items = True,
    probabilistic_recommendations = False,
    horizontally_differentiated_only = False,
    num_forced_items = 1,
    forced_period = 10,
    softmax_mult_for_forced_items = 1,
    individual_rationality = True,
    sort_rec_per_popularity = True,
    prop_items_for_concentration_metric = 0.1,
)