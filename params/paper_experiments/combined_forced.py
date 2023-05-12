params = dict(
    num_users = 100,
    num_items = 200,
    num_creators = 10,
    p_creation = 1.,
    attention_exp = 0,
    drift = [0, 0.5],
    training = 10,
    timesteps = 100,
    train_between_steps = True,
    num_attributes = 5,
    min_preference_per_attribute = 0,
	max_preference_per_attribute = 5,
    runs = 10,
    num_items_per_iter = 10,
    random_newly_created = False,
    random_items_per_iter= 0,
    repeated_items = True,
    probabilistic_recommendations = False,
    horizontally_differentiated_only = [False, True],
    num_forced_items = [5, 10],
    forced_period = [5, 10],
    shuffle_forced_items = False,
    individual_rationality = False,
    sort_rec_per_popularity = False,
    prop_items_for_concentration_metric = 0.1,
)