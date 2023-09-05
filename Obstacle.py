# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:51:43 2023

@author: nhkm8
"""

import numpy as np
import matplotlib.pyplot as plt


class Obstacle():
    def __init__(self, x_pos:float, y_pos:float, radius:float)-> None:
        self.x_pos=x_pos
        self.y_pos=y_pos
        self.radius=radius
    
    def is_inside(self,curr_x:float, curr_y:float, robot_radius:float=0):
        
    
        dist_from = np.sqrt((curr_x-self.x_pos)**2 +(curr_y-self.y_pos**2))
        
        if dist_from > self.radius + robot_radius:
            return False
        return True
    
    '''
    check if inside or near obstiels
    check if ouside the boundry
        check x position
    compare to xmin and max 
        if x_min > x_pos
        return true
    if x_max < x pos
        return true
do the same for postion y

    '''
def is_not_valid(obstacle_list:list, x_min:int, y_min:int, y_max:int, x_max:int,
                 x_curr:float, y_curr:float, agent_radius:float=0.0):
    
    for obs in obstacle_list:
        if obs.is_inside(x_curr,y_curr,agent_radius):
            print("youre dead at", obs.x_pos, obs.y_pos)
        return True
    if x_min > x_curr:
        return True
    if x_max < x_curr:
        return True
    if y_min > y_curr:
        return True
    if y_max < y_curr:
        return True
    
def get_all_moves(node_position:list) -> list:
    pass 
     
'''
for loop through y space bounds
    for loop through x space bounds
        calculate 
    get cost from curr node to nighbor node 
    
'''

if __name__=='__main__':
    obstacle_positions = [(1,1),(4,4),(3,4),(5,0),(5,1),(5,1),(0,7),(1,7),(2,7),(3,7)]
    obstacle_list=[]
    obstacle_radius= 0.25  
    
    for obs_pos in obstacle_positions:
        print("obstacle_positions", obs_pos)
        obstacle= Obstacle(obs_pos[0], obs_pos[1], obstacle_radius)
        obstacle_list.append(obstacle)
        
    agent_x=2
    agent_y=2
    
    for obs in obstacle_list:
        print("This obsticales position is", obs.x_pos, obs.y_pos)
        if obs.is_inside(agent_x, agent_y):
            print("youre dead at", obs.x_pos, obs.y_pos)
        else:
            print("youre safe at",agent_x,agent_y)