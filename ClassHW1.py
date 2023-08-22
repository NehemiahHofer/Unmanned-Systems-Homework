# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:34:11 2023

@author: nhkm8
"""
class Node:
    def __init__(self,x,y,parent_cost,index):
        # attributes
        self.x=x
        self.y=y 
        self.parent_cost=parent_cost
        self.index=index
        

class Engine():
    def __init__(self, name:str):
        self.name = name
        
    def engine_type(self):
        print("Engine is ", self.name)
        
    def burn_fuel(self):
        return 2 + 2
        
class Airplane():
    # engine 
    def __init__(self, weight:float, name:str, engine:Engine):
        # attributes of an aircraft 
        self.weight = weight
        self.name = name 
        self.engine = engine

    def takeoff(self):
        print(self.name + " im taking off")



engine1 = Engine("cessna engine")
cessna = Airplane(1000, "Cessna", engine1) 

engine2 = Engine("blackbird engine")
blackbird = Airplane(5000, "SR71", engine2)               
#user= Node(1,2,3,"node 1")

#print(user.x, user.y)