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



# Replace the combo column with the id column
def create_heatmaps():

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

def create_barchart():
    # For each model average all the values across different game parameter combinations such that we can see how well a model performs across all game parameter combinations
    model_results = data.groupby('model').mean()

    # Bar chart X, Y bounds per column, if None dont set bounds
    bounds = {
        'avg_total_reward': None,
        'avg_snake_size': [1, 1.065],
        'avg_food_eaten': None,
        'avg_moves': [5, 9],
        'max_snake_size': [2, 3],
        'max_food_eaten': [1, 2],
        'max_moves': [12, 23],
        'max_total_reward': [30, 53]
    }

    # Now create a chart that contains multiple barcharts, one for all the columns keys in column_keys in one figure.
    # There should be 8 barcharts in total.
    # It should respect the bounds in the bounds dictionary UNLESS bounds is None
    fig, axes = plt.subplots(4, 2, figsize = (36, 36))
    axes = axes.flatten()
    al = list(map(lambda x: str(x), range(1, 37)))
    for i, (key, value) in enumerate(column_keys.items()):
        if bounds[key] is not None:
            model_results[key].plot.bar(ax=axes[i], ylim=bounds[key])
        else:
            model_results[key].plot.bar(ax=axes[i])
        axes[i].set_xticklabels(al)
        # Set Y label
        axes[i].set_ylabel(value)
        # if last in column, set xlabel
        if i in [6, 7]:
            axes[i].set_xlabel('Model')
        else:
            axes[i].set_xlabel('')


    fig.set_size_inches(16, 12)
    fig.savefig(f'experiment/all_barchart.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":    
    create_heatmaps()
    create_barchart()




