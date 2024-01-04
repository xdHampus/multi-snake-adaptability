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

# Create a correlation matrix
correlation_matrix = data.pivot_table(index='model', columns='combo', values='avg_total_reward', aggfunc='mean')

#print(correlation_matrix)

# Visualize it instead as a heatmap, disable display of the matrix values in a gray scale format acceptable for an IEEE publication going between 4 distinct colors. 
# Make it as a grid with cells separate so values are easier to see
# Put lines between every 5th cells
fig, ax = plt.subplots(figsize = (36, 36))
al = list(map(lambda x: str(x), range(1, 37)))
sns.heatmap(correlation_matrix, ax=ax, annot=False, cmap="YlGnBu", linewidths=0.5, xticklabels=al, yticklabels=al)
b, t = plt.ylim()
for i in range(0, 36):
    if i % 4 == 0:
        ax.hlines(y = i, xmin = b, xmax = t, colors = 'blue', linewidths=0.5)
        ax.vlines(x = i, ymin = b, ymax = t, colors = 'blue', linewidths=0.5)
plt.title('Correlation Matrix')
plt.xlabel('Combo')
plt.ylabel('Model')
plt.show()


# Create a plot to visualize max values for each model in different combos
# This is the data. Come up with data visualisations for this in the context of me report. The combo is the id of the game environment and model 
# avg_total_reward,avg_snake_size,avg_food_eaten,avg_moves,max_snake_size,max_food_eaten,max_moves,max_total_reward,model,combo
# -11.3434,0.643,0.025,10.3855,3,2,28,54.3,0,0
# Start now.




