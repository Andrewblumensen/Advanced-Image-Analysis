# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:17:16 2022

@author: danie
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

#%% input
#ANGIV INPUT MATRIX!
I = np.array([[7,1,2,2,4,1],[6,6,5,4,5,1],[5,2,6,4,4,2],[1,5,7,2,2,6],[2,4,3,6,7,7]])


plt.imshow(I, cmap='gray')

#%% one line
#ANGIV DELTA!
delta = 2

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

plt.imshow(I, cmap='gray')
plt.plot(segmentation_line, 'r')
plt.set_title(f'delta = {delta}')