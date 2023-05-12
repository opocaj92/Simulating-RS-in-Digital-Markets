params = dict(
    num_users = 10,
    num_items = 200,
    num_creators = 5,
    p_creation = 1.,
    attention_exp = 1,
    drift = 0,
    training = 10,
    timesteps = 500,
    train_between_steps = True,
    num_attributes = 500,
    min_preference_per_attribute = 0,
	max_preference_per_attribute = 5,
    runs = 1,
    num_items_per_iter = 50,
    random_newly_created = False,
    random_items_per_iter = 0,
    repeated_items = True,
    probabilistic_recommendations = False,
    horizontally_differentiated_only = False,
    num_forced_items = 0,
    forced_period = 0,
    shuffle_forced_items = False,
    individual_rationality = False,
    sort_rec_per_popularity = False,
    prop_items_for_concentration_metric = 0.1,
)