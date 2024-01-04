import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import game_parameter_combinations
from train import training_limits

combos = game_parameter_combinations(training_limits())
for i, combo in enumerate(combos):
    combo['id'] = i
combos = sorted(combos, key=lambda x: (int(x["food_total_max"]), int(x["map_size"]), int(x["walls_max"])))

combo_mapping = {}
for i, combo in enumerate(combos):
    combo_mapping[combo['id']] = i


# Load the CSV file
data = pd.read_csv('experiment/results.csv')

# Map the combo column to the id column using the combo_mapping dictionary
data['combo'] = data['combo'].map(combo_mapping)
data['model'] = data['model'].map(combo_mapping)


# Replace the combo column with the id column

# Column keys
column_keys = {
    'avg_total_reward': 'Average Total Reward',
    'avg_snake_size': 'Average Snake Size',
    'avg_food_eaten': 'Average Food Eaten',
    'avg_moves': 'Average Moves',
    'max_snake_size': 'Max Snake Size',
    'max_food_eaten': 'Max Food Eaten',
    'max_moves': 'Max Moves',
    'max_total_reward': 'Max Total Reward'
}

for key, value in column_keys.items():
    correlation_matrix = data.pivot_table(index='model', columns='combo', values=key, aggfunc='mean')

    fig, ax = plt.subplots(figsize = (36, 36))
    al = list(map(lambda x: str(x), range(1, 37)))
    sns.heatmap(correlation_matrix, ax=ax, annot=False, cmap="YlGnBu", linewidths=0.5, xticklabels=al, yticklabels=al, cbar_kws={'label': value})
    b, t = plt.ylim()
    for i in range(0, 36):
        if i % 4 == 0:
            ax.hlines(y = i, xmin = b, xmax = t, colors = 'blue', linewidths=0.5)
            ax.vlines(x = i, ymin = b, ymax = t, colors = 'blue', linewidths=0.5)
    plt.title(f'{value} Correlation Matrix')
    plt.xlabel('Game Parameter Combination')
    plt.ylabel('Model')

    fig.set_size_inches(12, 10)
    fig.savefig(f'experiment/{key}.png', dpi=300, bbox_inches='tight')






