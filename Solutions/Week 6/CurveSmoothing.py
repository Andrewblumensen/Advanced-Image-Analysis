# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:21:52 2022

@author: danie
"""

import numpy as np
import scipy.interpolate
import scipy.linalg
import skimage.draw
import simple_snake as sis

#%% Forward

alpha = 0.05
beta = 0.1

S = np.array([[0.1, 1.2, 3.3, 6.2, 3.5, 1.4],[2.9, 5.4, 7.1, 0.9, 0.2, 1.1]])


B = sis.regularization_matrix(len(S[1]), -alpha, -beta)

B = np.linalg.inv(B)


Result = np.matmul(B,S.T)

#%% Backward

alpha = 0.05
beta = 0.1

S = np.array([[0.1, 1.2, 3.3, 6.2, 3.5, 1.4],[2.9, 5.4, 7.1, 0.9, 0.2, 1.1]])


B = sis.regularization_matrix(len(S[1]), alpha, beta)



Result = np.matmul(B,S.T)
