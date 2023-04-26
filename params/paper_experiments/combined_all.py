params = dict(
    num_users = 100,
    num_items = 200,
    num_creators = [0, 5, 10],
    p_creation = 1.,
    attention_exp = [-1.0, 0],
    drift = 1, #[0, 0.5, 1],
    training = [0, 10], #, 100],
    timesteps = 100,
    train_between_steps = True,
    num_attributes = [10, 100],
    min_preference_per_attribute = 0,
	max_preference_per_attribute = 5,
    max_item_score = 10,
    runs = 10,
    num_items_per_iter = [5, 10],
    random_newly_created = [False, True],
    random_items_per_iter= [0., 0.5],
    repeated_items = True,
    probabilistic_recommendations = [False, True],
    horizontally_differentiated_only = [False, True],
    num_forced_items = 0,
    forced_period = 0,
    shuffle_forced_items = False,
    individual_rationality = False,
    prop_items_for_concentration_metric = 0.1,
)