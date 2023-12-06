# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:46:45 2023

@author: nhkm8
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
import time
import tspastar
init_time = time.time()
waypoints = [(0,0), (9, 4), (4, 4), (1, 9), (9, 7), (6, 14)]
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
            goal = [other_wp[0], other_wp[1]]
            start = [wp[0], wp[1]]
            path = tspastar.Astar(goal, start)
            total_distance = compute_total_distance(path)
            cost_dictionary[(wp, other_wp)] = total_distance
            cost_dictionary[(other_wp, wp)] = total_distance


cities = list(range(len(waypoints)))
adjacency_mat = np.zeros((len(waypoints), len(waypoints)))

for i in range(len(waypoints)):
    for j in range(len(waypoints)):
        if i != j:
            adjacency_mat[i, j] = cost_dictionary[(waypoints[i], waypoints[j])]


class Population():
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat

def init_population(cities, adjacency_mat, n_population):
    initial_chromosome = np.arange(len(cities))  # Include all cities
    populations = [np.roll(initial_chromosome, shift=i) for i in range(n_population)]
    return Population(
        np.asarray(populations),
        adjacency_mat
    )

pop = init_population(cities, adjacency_mat, 5)

def fitness(self, chromosome):
    return sum(
        [
            self.adjacency_mat[chromosome[i], chromosome[i + 1]]
            for i in range(len(chromosome) - 1)
        ]
    )

Population.fitness = fitness

def evaluate(self):
    distances = np.asarray(
        [self.fitness(chromosome) for chromosome in self.bag]
    )
    self.score = np.min(distances)
    self.best = self.bag[distances.tolist().index(self.score)]
    self.parents.append(self.best)
    if False in (distances[0] == distances):
        distances = np.max(distances) - distances
    return distances / np.sum(distances)
    
Population.evaluate = evaluate

def select(self, k=4):
    fit = self.evaluate()
    while len(self.parents) < k:
        idx = np.random.randint(0, len(fit))
        if fit[idx] > np.random.rand():
            self.parents.append(self.bag[idx])
    self.parents = np.asarray(self.parents)

Population.select = select

def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome

def crossover(self, p_cross=0.1):
    children = []
    count, size = self.parents.shape
    for _ in range(len(self.bag)):
        if np.random.rand() > p_cross:
            children.append(
                list(self.parents[np.random.randint(count, size=1)[0]])
            )
        else:
            parent1, parent2 = self.parents[
                np.random.randint(count, size=2), :
            ]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children

Population.crossover = crossover

def mutate(self, p_cross=0.1, p_mut=0.1):
    next_bag = []
    children = self.crossover(p_cross)
    for child in children:
        if np.random.rand() < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag
    
Population.mutate = mutate


def genetic_algorithm(
    cities,
    adjacency_mat,
    n_population=500,
    n_iter=2000,
    selectivity=0.15,
    p_cross=0.5,
    p_mut=0.1,
    print_interval=100,
    return_history=True,
    verbose=True,
):
    pop = init_population(cities, adjacency_mat, n_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.adjacency_mat)
    
    
    if return_history:
        return best, history[-1]
    return best, history[-1]


best_path, min_cost = genetic_algorithm(cities, adjacency_mat, verbose=True)


best_path = list(best_path)
index_of_starting_point = best_path.index(0)
best_path = best_path[index_of_starting_point:] + best_path[:index_of_starting_point]


final_order = [waypoints[i] for i in best_path]
print(f"Order of waypoints in the final generation: {final_order}")


total_cost = 0.0
for i in range(len(final_order) - 1):
    current_waypoint = final_order[i]
    next_waypoint = final_order[i + 1]
    cost = cost_dictionary[(current_waypoint, next_waypoint)]
    total_cost += cost
    print(f"Cost from waypoint {i + 1} to waypoint {i + 2}: {cost}")


print(f"Total cost of the path: {total_cost}")

plt.title(f'Best Path Visualization\nMinimum Total Cost: {total_cost}')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')


for i, waypoint in enumerate(waypoints):
    plt.text(waypoint[0], waypoint[1], str(i + 1), color='red', fontsize=8, ha='center', va='center')

path_coordinates = final_order + [final_order[0]]
#plt.plot(*zip(*path_coordinates), marker='o', linestyle='-', color='blue', label='Best Path')


plt.grid(True)
plt.show()
print("time diff is", time.time() - init_time)
