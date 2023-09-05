# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:08:15 2023

@author: nhkm8
"""
import math as m
import numpy as np
current_position= (2,1)
goal_position=(3,2)
current_node_parent_cost= (0)
cost = current_node_parent_cost + m.dist(current_position, goal_position)
print("the cost to move from",current_position,"to",goal_position, "is",cost)