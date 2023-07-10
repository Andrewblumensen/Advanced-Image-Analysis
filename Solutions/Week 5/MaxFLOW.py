# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:24:06 2022

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import maxflow

#%% QUESTION 12 *
# Solution based on:
# https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week05/week05_pycode/dtu_binary.py
#I = skimage.io.imread('C:/Users/danie/OneDrive/Desktop/02506/bony.png').astype(float)
I = np.array([[1,2,6,4,10,8],[4,1,3,5,9,6],[5,2,3,5,4,7]])
mu = np.array([2, 5, 10], dtype=float)
beta  = 10

# Graph with internal and external edges
U = (I.reshape(I.shape+(1,)) - mu.reshape(1,1,-1))**2
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(I.shape)
g.add_grid_edges(nodeids, beta)
g.add_grid_tedges(nodeids, U[:,:,1], U[:,:,0])
g.maxflow()
S = g.get_grid_segments(nodeids)

S = S.astype(int)

prior = beta * ((S[1:,:]!=S[:-1,:]).sum() + (S[:,1:]!=S[:,:-1]).sum()) 
likelihood = int(((mu[S]-I)**2).sum())


# Visualization
fig, ax = plt.subplots()
ax.imshow(S)
#print(f'Question 12: {S.sum()}')
print(f'Answer: {prior + likelihood}')