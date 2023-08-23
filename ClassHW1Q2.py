# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:34:42 2023

@author: nhkm8
"""
import matplotlib.pyplot as plt
import numpy as np

def compute_index(x_min:int, x_max:int, y_min:int, y_max:int, gs:int, x_curr:int, y_curr:int, ) ->int:
    index=((x_curr-x_min)/gs)+(((y_curr-y_min)/gs)*((x_max+gs)-x_min)/gs)
    return index

min_x=0
max_x=10
min_y=0
max_y=10
gs=0.5



#index= compute_index(min_x, max_x, min_y, max_y, gs, x_curr,y_curr)
x_bounds = np.arange(min_x, max_x+gs, gs)
y_bounds = np.arange(min_y, max_y+gs, gs)

for y in y_bounds:
    for x in x_bounds:
        index= compute_index(min_x, max_x, min_y, max_y, gs, x,y)
        print(x,y,index)
        plt.text(x, y, str(int(index)), color="red", fontsize=8)
        
        
# for i in range(0, int(length_vec*length_vec)+1):
#     print(i)

# for i in range(0, int(length_vec)):
#     x_val = i*gs
#     print("verbose", x_val)


# for y in y_bounds:
#     for x in x_bounds:
#         print(x,y)
    
# for j in range(len(y_bounds)):
#     for i in range(len(x_bounds)):
        

# for i in range(x):
#     for j

    
plt.grid(which = "both")
plt.minorticks_on()
plt.xlim([min_x,max_x+gs])
plt.ylim([min_y,max_y+gs])
plt.show()