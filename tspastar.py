
"""
Created on Wed Sep  6 08:58:14 2023
 
@author: nehem
"""
 
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time
 
start_time = time.time()
class Node:
    def __init__(self,x,y,parent_cost,index, parent_position):
        # attributes
        self.x=x
        self.y=y 
        self.parent_cost=parent_cost
        self.index= int(index)
        self.parent_position = parent_position
 
 
class Obstacle():
    def __init__(self, x_pos:float, y_pos:float, radius:float)-> None:
        self.x_pos=x_pos
        self.y_pos=y_pos
        self.radius=radius
    def is_inside(self,curr_x:float, curr_y:float, robot_radius:float=0):
        dist_from = np.sqrt((curr_x-self.x_pos)**2 +(curr_y-self.y_pos)**2)
        if dist_from >= self.radius + robot_radius:
            return False
        return True
def is_not_valid(obstacle_list:list, x_min:int, y_min:int, y_max:int, x_max:int,
                 x_curr:float, y_curr:float, agent_radius:float=0.0,
                 grid_radius:float=0.0,
                 check_obstacle=True):
    
    if check_obstacle == True:
        for obs in obstacle_list:
            if obs.is_inside(x_curr,y_curr,agent_radius):
                #print("youre dead at", obs.x_pos, obs.y_pos, x_curr, y_curr)
                return True
    if x_min+grid_radius > x_curr:
        return True
    if x_max-grid_radius < x_curr:
        return True
    if y_min+grid_radius > y_curr:
        return True
    if y_max-grid_radius < y_curr:
        return True

 
def compute_index(x_min:int, x_max:int, y_min:int, y_max:int, gs:int, x_curr:int, y_curr:int ) ->int:
    index=((x_curr-x_min)/gs)+(((y_curr-y_min)/gs)*((x_max+gs)-x_min)/gs)
    return index
 
        
def get_all_moves(node_position:list) -> list:
    pass      
 
 
def Astar(goal_position,start):
    # INTIALIZE HERE
    min_x=0
    max_x=15
    min_y=0
    max_y=15
    gs=0.5
    x_curr=1
    y_curr=1
    x_start = start[0]
    wp_list=[]
    counter=(0)
    y_start = start[1]
    agent_radius  = .5
    #goal_position = [7.0,13.0] 
    #goal_position =[xgoal,ygoal]
    obstacle_positions = []
    obstacle_positions = [(2, 2), (2, 3), 
                          (2, 4), (2, 5), (0, 5),
                          (1, 5), (2, 5), (3, 5), (4, 5),
                          (5, 5), (5, 2), (5, 3), (5, 4), 
                          (5, 5), (8, 2), (9, 2), (10, 2),
                          (11, 2), (12, 2), (13, 2), (8, 3), 
                          (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
                          (8, 9), (2, 7), (3, 7), (4, 7), (5, 7),
                          (6, 7), (7, 7), (9, 6), (10, 6), (11, 6),
                          (12, 6), (13, 6), (14, 6), (15, 6), (2, 8),
                          (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), 
                          (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), 
                          (5, 14), (5, 15), (6, 12), (7, 12), (8, 12), 
                          (9, 12), (10, 12), (11, 12), (12, 8), (12, 9),
                          (12, 10), (12, 11), (12, 12)]
    obstacle_list=[]
    obstacle_radius= .5
    for obs_pos in obstacle_positions:
        # print("obstacle_positions", obs_pos)
        obstacle= Obstacle(obs_pos[0], obs_pos[1], obstacle_radius)
        obstacle_list.append(obstacle)
 
    
    obstacle_radius_range = [-obstacle_radius, 0, obstacle_radius]
    inflated_list = []
   
    for obs_position in obstacle_positions:
        inflated_list.append(obs_position)
        for i in obstacle_radius_range:
            for j in obstacle_radius_range:
                new_obs_x = obs_position[0] + i
                new_obs_y = obs_position[1] + j
                new_pos = (new_obs_x, new_obs_y)
                inflated_list.append(new_pos)
             
    obstacle_set = set(inflated_list)
    
    ### THIS IS DJIKSTRAS
    unvisited ={}        
    visited={}
    # Node(x,y, parent_cost,index,parent_position)
    current_node=Node(x_start, y_start, 0, -1, [-1,-1])
    current_index = compute_index(min_x, max_x, min_y, max_y, gs, current_node.x,current_node.y)
    current_position = [current_node.x, current_node.y]
    unvisited[current_index] = current_node
 
    while [current_node.x, current_node.y] != goal_position:
    # Check if unvisited is empty
        if not unvisited:
            # Handle the case where no path is found
            print("No path found to goal")
            break
    
        current_index = min(unvisited, key=lambda x:unvisited[x].parent_cost)
        current_node = unvisited[current_index]

        if [current_node.x, current_node.y] == goal_position:    
            #return 
            wp_list = []
            wp_node = current_node
            wp_list.append([wp_node.x, wp_node.y])
            while (wp_node.index != -1):
                wp_node = visited[wp_node.index]
                wp_list.append([wp_node.x, wp_node.y])
                #print("position is", wp_node.x, wp_node.y)
                if wp_node.index == -1:
                    wp_list= wp_list[::-1]    
                    break

 
        
        visited[current_index] = current_node
        del unvisited[current_index]
 
        gs_bounds = np.arange(-gs, gs+gs, gs)
        move_list = []
        # get all moves
        for y_step in gs_bounds:        
            for x_step in gs_bounds:
 
                move_x = x_step+current_node.x
                move_y = y_step+current_node.y
                if move_x == current_node.x and move_y == current_node.y:
                    continue
                move_list.append([move_x, move_y])
        # once you're here filter out all moves
        filtered_moves = []
        for move in move_list:
            if (is_not_valid(obstacle_list, min_x, min_y, max_x, max_y,
                             move[0], move[1],agent_radius, obstacle_radius,
                             check_obstacle = False) == True):
                continue
            
            
            move_set = (move[0], move[1])
            
            if move_set in obstacle_set:
                continue
        
            filtered_moves.append(move)

        # loop through all filtered moves and put into unvisited
        for move in filtered_moves:
            # compute the cost to get to this filtered move FROM our current node
            current_position = [current_node.x, current_node.y]

 
            cost = current_node.parent_cost + m.dist(current_position, move)
            +m.dist(move,[goal_position[0],goal_position[1]])
 
            # calculate the index of this filtered move
            index = int(compute_index(min_x, max_x, min_y, max_y, gs, move[0],move[1]))
            # make sure its not in visited 
            if index in visited:
                continue
            # update cost
            if index in unvisited:
                if current_position == [0.5, 0.0]:
                    print("cost for ", move ,"is ", cost, "this index cost is", 
                          unvisited[index].parent_cost)
                if current_position == [0.0, 0.0]:
                    print("cost from origin to  ", move ,"is ", cost)
                # compare the cost 
                if unvisited[index].parent_cost > cost:
                    # print("updating cost", unvisited[index].x, unvisited[index].y)
                    unvisited[index].parent_cost = cost
                    unvisited[index].index = current_node.index
                    unvisited[index].parent_position = current_position
                continue
 
                # if lower than update
            # make a temp node
            temp_node=Node(move[0], move[1], cost, current_index, 
                           [current_node.x, current_node.y])
 
            unvisited[index] = temp_node
 
    
    ### PLOT STUFF
    x_list = []
    y_list = []
    for wp in wp_list:
        x_list.append(wp[0])
        y_list.append(wp[1])
    for obs in obstacle_list:
        plt.scatter(obs.x_pos, obs.y_pos, c='r')
    plt.plot(x_list, y_list, '-')
    plt.grid(which = "both")
    plt.minorticks_on()  
    plt.xlim([min_x,max_x+gs])
    plt.ylim([min_y,max_y+gs])        
    plt.scatter(goal_position[0], goal_position[1], c='black', marker='^', s=100, label='Goal Position')  # Yellow triangle marker
    #print(cost)
    end_time = time.time()
    execution_time = end_time - start_time
    #print(f"Execution time: {execution_time} seconds")
    # print(cost)
    return wp_list
    