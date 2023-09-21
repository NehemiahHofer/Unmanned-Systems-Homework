"""
Created on Sat Sep 16 17:05:54 2023

@author: nhkm8
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class treeNode():
    def __init__(self, locationX, locationY, agent_radius):
        self.locationX = locationX
        self.locationY = locationY
        self.agent_radius = agent_radius
        self.kids = []
        self.parent = None

class RRT_A():
    def __init__(self, start, goal, numIT, gs, stepSize, agent_radius, obstacle_radius):
        self.randomTree = treeNode(start[0], start[1], agent_radius)
        self.goal = treeNode(goal[0], goal[1], agent_radius)
        self.nearestNode = None
        self.itterations = min(numIT, 1000)
        self.grid_spacing = gs
        self.rho = stepSize
        self.agent_radius = agent_radius
        self.obstacle_radius = obstacle_radius
        self.nearestDist = 10000
        self.numWaypoints = 0
        self.wp = []

        self.obstacle_positions = [(2, 2), (2, 3), (2, 4), (5, 5), (5, 6), (6, 6), (7, 3), (7, 4), (7, 5), (7, 6), (8, 6)]

        self.grid = np.zeros((int(10 / gs), int(10 / gs)))

        for obstacle in self.obstacle_positions:
            x_idx = int(obstacle[0] / gs)
            y_idx = int(obstacle[1] / gs)
            self.grid[y_idx, x_idx] = 1

        self.obstacle_points = []
        for obstacle in self.obstacle_positions:
            x_idx = int(obstacle[0] / gs)
            y_idx = int(obstacle[1] / gs)
            x_center = x_idx * gs + gs / 2
            y_center = y_idx * gs + gs / 2
            self.obstacle_points.append((x_center, y_center))

    def KID(self, locationX, locationY):
        if locationX == self.goal.locationX:
            self.nearestNode.kids.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tempnode = treeNode(locationX, locationY, self.agent_radius)
            self.nearestNode.kids.append(tempnode)
            tempnode.parent = self.nearestNode

    def CheckMove(self):
        while True:
            x = random.uniform(0, 10)
            y = random.uniform(0, 10)
            randiMove = np.array([x, y]) 

            agent_radius_with_obstacle = self.agent_radius + self.obstacle_radius
            for obstacle_point in self.obstacle_points:
                if np.linalg.norm(randiMove - obstacle_point) <= agent_radius_with_obstacle:
                    break
            else:
                return randiMove

    def Direction(self, locationStart, locationEnd):
        v = np.array([locationEnd[0] - locationStart[0], locationEnd[1] - locationStart[1]])
        mag = np.linalg.norm(v)
        if mag < self.rho:
            return locationEnd
        else:
            v = (v / mag) * self.rho
            return np.array([locationStart[0] + v[0], locationStart[1] + v[1]])

    def Obstacle(self, locationStart, locationEnd):
        num_samples = int(np.ceil(np.linalg.norm(np.array(locationEnd) - np.array(locationStart)) / self.rho))
        if num_samples == 0:
            return False

        delta = np.array(locationEnd) - np.array(locationStart)
        delta /= num_samples

        for i in range(num_samples + 1):
            test_point = np.array(locationStart) + i * delta

            for obstacle_point in self.obstacle_points:
                if np.linalg.norm(test_point - obstacle_point) <= (self.agent_radius + self.obstacle_radius):
                    return True

        return False

    def WhoClose(self, root, point):
        if not root:
            return
        dist = np.sqrt((root.locationX - point[0]) ** 2 + (root.locationY - point[1]) ** 2)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        for kid in root.kids:
            self.WhoClose(kid, point)

    def distance(self, node1, point):
        dist = np.sqrt((node1.locationX - point[0]) ** 2 + (node1.locationY - point[1]) ** 2)
        return dist

    def goalFound(self, point):
        if self.distance(self.goal, point) <= self.rho:
            return True

    def val_reset(self):
        self.nearestNode = None
        self.nearestDist = 10000

    def resetpath(self, goal):
        if goal is None:
            return
        if goal.locationX == self.randomTree.locationX:
            return
        self.numWaypoints += 1

        currentPoint = np.array([goal.locationX, goal.locationY])
        self.wp.append(currentPoint)
        self.resetpath(goal.parent)

gs = 0.5
start = np.array([1.0, 1.0])
goal = np.array([9.0, 8.0])
numIT = 1000
stepSize = 0.5
agent_radius = 0.5
obstacle_radius = 0.25

rrt = RRT_A(start, goal, numIT, gs, stepSize, agent_radius, obstacle_radius)

fig, ax = plt.subplots()
ax.grid(True)

for y in range(rrt.grid.shape[0]):
    for x in range(rrt.grid.shape[1]):
        if rrt.grid[y, x] == 1:
            ax.add_patch(plt.Circle((x * gs + gs / 2, y * gs + gs / 2), rrt.obstacle_radius, color='red'))

for i in range(rrt.itterations):
    rrt.val_reset()
    randiMove = rrt.CheckMove()  
    rrt.WhoClose(rrt.randomTree, randiMove)
    new = rrt.Direction([rrt.nearestNode.locationX, rrt.nearestNode.locationY], randiMove)
    testin = rrt.Obstacle([rrt.nearestNode.locationX, rrt.nearestNode.locationY], new)
    if testin == False:
        rrt.KID(new[0], new[1])
        ax.plot([rrt.nearestNode.locationX, new[0]], [rrt.nearestNode.locationY, new[1]], 'go', linestyle="--")
        if rrt.goalFound(new):
            rrt.KID(goal[0], goal[1])
            break
rrt.resetpath(rrt.goal)
rrt.wp.insert(0, start)

for i in range(len(rrt.wp) - 1):
    ax.plot([rrt.wp[i][0], rrt.wp[i + 1][0]], [rrt.wp[i][1], rrt.wp[i + 1][1]], 'ro', linestyle="--")

plt.show
