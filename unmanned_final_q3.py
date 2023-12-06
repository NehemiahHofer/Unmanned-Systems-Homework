from itertools import permutations
import math as m
import time
import tspastar
import matplotlib.pyplot as plt

waypoints = [(0,0), (9,4), (4,4), (1,9), (9,7), (6,14)]
other_waypoints = {}
cost_dictionary = {}


def euclidean_distance(point1, point2):
    return m.dist(point1, point2)


def compute_total_distance(waypoints):
    total_distance = 0.0
    for i in range(len(waypoints) - 1):
        distance_between_points = euclidean_distance(waypoints[i], waypoints[i + 1])
        total_distance += distance_between_points

    return total_distance


# Compute all the costs from some waypoint to all other waypoints
compute_times = 0
for wp in waypoints:
    for other_wp in waypoints:
        if wp == other_wp:
            continue
        if (wp, other_wp) in cost_dictionary:
            continue
        if (other_wp, wp) in cost_dictionary:
            continue
        else:
            compute_times += 1
            ## this would be where you plug in astar
            goal = [other_wp[0], other_wp[1]]
            start = [wp[0], wp[1]]
            path = tspastar.Astar(goal, start)
            # total distance = sum of all the waypoints in the path
            total_distance = compute_total_distance(path)
            # a->c
            cost_dictionary[(wp, other_wp)] = total_distance
            # c->a
            cost_dictionary[(other_wp, wp)] = total_distance

# Compute all the possible paths with fixed starting point (1,1)
paths = list(permutations(waypoints[1:], len(waypoints) - 1))
paths = [(waypoints[0],) + path for path in paths]

total_cost = []
print("Number of compute times: ", compute_times)
print("Number of paths: ", len(paths))

n_iterations = 100

start_time = time.time()
for j, path in enumerate(paths):
    sum_cost = 0
    if j % n_iterations == 0:
        print("Iteration: ", j)

    for i in range(len(path) - 1):
        sum_cost += cost_dictionary[path[i], path[i + 1]]

    total_cost.append(sum_cost)
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# get best path
min_total_cost = min(total_cost)
min_total_cost_index = total_cost.index(min_total_cost)
best_path = paths[min_total_cost_index]

print("Best Path:", best_path)
print("Minimum Total Cost:", min_total_cost)

# Plot the best path
best_path_x, best_path_y = zip(*best_path)
#plt.plot(best_path_x, best_path_y, marker='o', linestyle='-', color='blue', label='Best Path')

# Annotate each point on the best path with its order
for i, point in enumerate(best_path):
    plt.annotate(str(i + 1), (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')

# Customize plot
plt.title(f'Best Path Visualization\nMinimum Total Cost: {min_total_cost}')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)

# Show the plot
plt.show()
