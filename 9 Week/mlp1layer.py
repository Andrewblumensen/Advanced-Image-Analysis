# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:17:04 2022

@author: hujo8
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:14:19 2022

@author: hujo8
"""


import make_data
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import scipy.io


#%%

ti = []
g = 4000
mat = scipy.io.loadmat('MNIST_target_train.mat')

mat = mat['target_train']

mat = np.squeeze(mat)

gt = mat[:g]


t = np.zeros((g,10))


for j in range(0,g):
    
    
    t[j,mat[j]] = 1


for i in range(0,g): 
     
    if i > 9999:
    
        ti.append(skimage.io.imread ('C:/Users/hujo8/Desktop/aia pic/MNIST_images_train/image_train_'+ str(i) +'.png').reshape((1,784)))
        
    if i > 999 and i < 10000:
        
        ti.append(skimage.io.imread ('C:/Users/hujo8/Desktop/aia pic/MNIST_images_train/image_train_0'+ str(i) +'.png').reshape((1,784)))
    
    if i > 99 and i < 1000:
        
        ti.append(skimage.io.imread ('C:/Users/hujo8/Desktop/aia pic/MNIST_images_train/image_train_00'+ str(i) +'.png').reshape((1,784)))
    
    if i > 9 and i < 100:
        
        ti.append(skimage.io.imread ('C:/Users/hujo8/Desktop/aia pic/MNIST_images_train/image_train_000'+ str(i) +'.png').reshape((1,784)))

    if i < 10:
    
        ti.append(skimage.io.imread ('C:/Users/hujo8/Desktop/aia pic/MNIST_images_train/image_train_0000'+ str(i) +'.png').reshape((1,784)))


ti = np.array(ti).reshape(1,g,784)

ti = (np.squeeze(ti)/255)


#%%

#%& Generate and display data
n = 1000
example_nr = 3
noise = 0.65

X, T, x, dim = make_data.make_data(example_nr, n, noise)

# Standardize
m = np.mean(X,axis=0)
s = np.std(X, axis=0)
Xc = (X-m)/s
xc = (x-m)/s


# fig, ax = plt.subplots(1)
# ax.plot(Xc[:n,0],Xc[:n,1],'r.', markersize=10,alpha=0.3)
# ax.plot(Xc[n:,0],Xc[n:,1],'g.', markersize=10,alpha=0.3)
# ax.set_aspect('equal')

# Function for simple forward pass
# def simple_forward(x, W):
# 	z = np.c_[x,np.ones((x.shape[0]))]@W[0]
# 	h = np.maximum(z,0)
# 	yh = np.c_[h,np.ones((x.shape[0]))]@W[1]
# 	y = np.exp(yh)/np.sum(np.exp(yh),axis=1,keepdims=True)
# 	return y, h


def simple_forward(x,W):
    z = np.c_[x,np.ones((x.shape[0]))]@W[0]
    h1 = np.maximum(z,0)
    yh = np.c_[h1,np.ones((z.shape[0]))]@W[1]
    y = np.exp(yh)/np.sum(np.exp(yh),axis=1,keepdims=True)
    return y,h1,

# # Function for simple backpropagation
# def simple_backward(x, W, t, learning_rate=0.1):
 	# y, h = simple_forward(x,W)
 	# L= -np.sum(t*np.log(y + 10e-10))/x.shape[0]
 	# # print(L)
 	# d1 = y - t
 	# q1 = np.c_[h,np.ones((x.shape[0]))].T@d1/y.shape[0]
 	# d0 = (h>0)*(d1@W[1].T)[:,:-1]
 	# q0 = np.c_[x,np.ones((x.shape[0]))].T@d0/y.shape[0]
 	# W[0] -= learning_rate*q0
 	# W[1] -= learning_rate*q1
# 	return W, L

def simple_backward(x, W, t, learning_rate=0.1):
    y, h1 = simple_forward(x,W)
    L= -np.sum(t*np.log(y + 10e-10))/x.shape[0]
    # print(L)
    d1 = y - t
    q1 = np.c_[h1,np.ones((x.shape[0]))].T@d1/y.shape[0]
    d0 = (h1>0)*(d1@W[1].T)[:,:-1]
    q0 = np.c_[x,np.ones((x.shape[0]))].T@d0/y.shape[0]
    W[0] -= learning_rate*q0
    W[1] -= learning_rate*q1
    return W, L, y



# Function for simple weight initializaion
# def simple_init_weights(n):
#  	W = []
#  	W.append(np.random.randn(785,n)*np.sqrt(2/3))
#  	W.append(np.random.randn(n+1,10)*np.sqrt(2/(n+1)))
#  	return W

def simple_init_weights(n):
    W = []
    W.append(np.random.randn(785,n)*np.sqrt(2/3))
    W.append(np.random.randn(n+1,10)*np.sqrt(2/(n+1)))
    return W




W = simple_init_weights(30)


fig, ax = plt.subplots(1)
n_iter = 50
L = np.zeros((n_iter))
i_rng = np.arange(0,n_iter)


learning_rate = 0.1

Xc = ti

T = t


for i in range(0,n_iter):
	W, L[i], y = simple_backward(Xc,W,T,learning_rate = 0.5)  
	ax.cla()
	ax.plot(i_rng,L,'k')
	ax.set_title('Loss')
	plt.pause(0.001)
	plt.show()
     
guess = []
for j in range(0,g):
    guess.append(np.argmax(y[j,:]))

guess = np.array(guess)

correctm = guess == gt.T

correct = np.sum(correctm)/g


