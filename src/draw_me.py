# Representing the 2D array as a 1D array
grid_A_2d = [
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27],
    [28, 29, 30, 31, 32, 33, 34],
    [35, 36, 37, 38, 39, 40, 41],
    [42, 43, 44, 45, 46, 47, 48],
]
grid_A_2d_dists = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 3, 2, 1],
    [1, 2, 3, 4, 3, 2, 1],
    [1, 2, 3, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
# Flatten the 2D array to a 1D array
grid_A_1d = [element for row in grid_A_2d for element in row]

# 7x8 grid
grid_B_2d_7x8 = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31],
    [32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53, 54, 55],
]
grid_B_2d_7x8_dists = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
# Flatten the 2D array to a 1D array
grid_B_7x8_1d = [element for row in grid_B_2d_7x8 for element in row]

def generate_grid_2d(width, height):
    return [[x + y * width for x in range(width)] for y in range(height)]

def generate_grid_2d_dists(width, height):
    return [[min(x, y, width - x - 1, height - y - 1) + 1 for x in range(width)] for y in range(height)]


def print_grid_2d(grid_2d):
    for row in grid_2d:
        print(row)

def print_grid_1d(grid_1d, width):
    for i in range(len(grid_1d)):
        print(grid_1d[i], end=" ")
        if (i + 1) % width == 0:
            print()

# Function to convert 2D array to 1D array
def flatten_2d_array(grid_2d):
    return [element for row in grid_2d for element in row]

# Function to from one point get dist to edge of grid for 1D array
def dist_to_edge_1d(width, height, point):
    x = point % width
    y = point // width
    return min(x, y, width - x - 1, height - y - 1) + 1

# Function to validate that dist_to_edge is correct for all points in any grid
def validate_dist_to_edge(grid_1d, width, height):
    for point in range(len(grid_1d)):
        if dist_to_edge_1d(grid_1d, width, height, point) != grid_1d[point]:
            print(f'point {point} is wrong should be {dist_to_edge_1d(grid_1d, width, height, point)} but is {grid_1d[point]}')
            return False
    return True



even_grid_2d = generate_grid_2d(6, 6)
even_grid_2d_dists = generate_grid_2d_dists(6, 6)
even_grid_1d = flatten_2d_array(even_grid_2d)
even_grid_1d_dists = flatten_2d_array(even_grid_2d_dists)
# print even grids
print()
print()
print_grid_2d(even_grid_2d)
print()
print_grid_2d(even_grid_2d_dists)
print()
print_grid_1d(even_grid_1d, 6)

unbalanced_grid_2d = generate_grid_2d(9, 8)
unbalanced_grid_2d_dists = generate_grid_2d_dists(9, 8)
unbalanced_grid_1d = flatten_2d_array(unbalanced_grid_2d)
unbalanced_grid_1d_dists = flatten_2d_array(unbalanced_grid_2d_dists)
# print unbalanced grids
print()
print()
print_grid_2d(unbalanced_grid_2d)
print()
print_grid_2d(unbalanced_grid_2d_dists)
print()
print_grid_1d(unbalanced_grid_1d, 9)

# validate even grid
print()
print()
print('even grid dists validation')
print(validate_dist_to_edge(even_grid_1d_dists, 6, 6))



# validate unbalanced grid
print()
print()
print('unbalanced grid validation')
print(validate_dist_to_edge(unbalanced_grid_1d_dists, 9, 8))

