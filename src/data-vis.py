import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the CSV file
data = pd.read_csv('experiment/results.csv')

# Create a correlation matrix
correlation_matrix = data.pivot_table(index='model', columns='combo', values='avg_snake_size', aggfunc='mean')

#print(correlation_matrix)

# Visualize it instead as a heatmap, disable display of the matrix values in a gray scale format acceptable for an IEEE publication going between 4 distinct colors
sns.heatmap(correlation_matrix, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Average Snake Size'})
plt.show()


