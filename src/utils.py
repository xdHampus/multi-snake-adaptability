
import math

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def game_parameter_combinations(limits):
    combinations = []

    for map_size in limits['map_size']:
        for food_chance in limits['food_chance']:
            for snake_start_len in limits['snake_start_len']:
                for food_total_max in limits['food_total_max']:
                    for walls_enabled in limits['walls_enabled']:
                        if not walls_enabled:
                            combinations.append({
                                'map_size': map_size,
                                'food_chance': food_chance,
                                'snake_start_len': snake_start_len,
                                'food_total_max': food_total_max,
                                'walls_enabled': False,
                                'walls_max': 0,
                                'walls_chance': 0
                            })
                        else:
                            for walls_max in limits['walls_max']:
                                for walls_chance in limits['walls_chance']:
                                    combinations.append({
                                        'map_size': map_size,
                                        'food_chance': food_chance,
                                        'snake_start_len': snake_start_len,
                                        'food_total_max': food_total_max,
                                        'walls_enabled': True,
                                        'walls_max': walls_max,
                                        'walls_chance': walls_chance
                                    })
    # Remove duplicates from combinations
    combinations = list(combinations)
    combinations = list(set(tuple(sorted(d.items())) for d in combinations))
    combinations = [dict(x) for x in combinations]

    # Order combinations by difficulty
    combinations.sort(key=game_parameter_difficulty_estimator)

    return combinations
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def game_parameter_difficulty_estimator(params):
    # Define weights for each parameter
    weights = {
        'map_size': 0.3,  # Increase weight for map_size
        'food_chance': -0.2,
        'snake_start_len': 0.2,
        'food_total_max': -0.1,
        'walls_enabled': 0.6,
        'walls_max': 0.5,
        'walls_chance': 0.5
    }

    # If walls are disabled, reduce the weights for wall-related parameters
    if not params['walls_enabled']:
        weights['walls_max'] = 0
        weights['walls_chance'] = 0
        # Small map size should still be harder than large map size when walls are not enabled
        weights['map_size'] = -0.1

    # Calculate the difficulty score
    difficulty_score = sum(params[key] * weights[key] for key in params.keys())

    # Normalize the difficulty score using sigmoid function
    normalized_difficulty = sigmoid(difficulty_score)

    return normalized_difficulty