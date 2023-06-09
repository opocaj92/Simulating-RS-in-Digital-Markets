params = dict(
    num_users = 100,
    num_items = 200,
    num_creators = 0,
    p_creation = 1.,
    attention_exp = [-1.0, 0],
    drift = 0,
    training = [0, 10, 1000],
    timesteps = 100,
    train_between_steps = True,
    num_attributes = [5, 10, 100],
    min_preference_per_attribute = 0,
	max_preference_per_attribute = 5,
    runs = 10,
    num_items_per_iter = 10,
    random_newly_created = False,
    random_items_per_iter= 0,
    repeated_items = [False, True],
    probabilistic_recommendations = [False, True],
    horizontally_differentiated_only = False,
    num_forced_items = 0,
    forced_period = 0,
    softmax_mult_for_forced_items = 5,
    individual_rationality = False,
    sort_rec_per_popularity = False,
    prop_items_for_concentration_metric = 0.1,
)