# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:36:22 2022

@author: hujo8
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder


#I = skimage.io.imread('C:/Users/hujo8/OneDrive/Advanced image analysis/7 Week/peaks_image.png').astype(np.int32)

I = np.array([[2,4,3,6,2,7],
     [1,5,3,2,2,6],
     [10,12,14,9,15,7],
     [16,16,15,11,10,17],
     [6,11,5,12,11,8]])

delta = 0

fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap='gray')


layers = [slgbuilder.GraphObject(0*I)] # no on-surface cost
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)

# Adding regional costs, 
# the region in the middle is bright compared to two darker regions.
helper.add_layered_region_cost(layers[0], 20-I, I)

# Adding geometric constrains
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta = delta, wrap=False)  


# Cut
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

# Visualization
ax[1].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[1].plot(line, 'r')