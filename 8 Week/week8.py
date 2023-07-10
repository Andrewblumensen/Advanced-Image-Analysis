# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:47:39 2022

@author: hujo8
"""


import make_data
import numpy as np
import matplotlib.pyplot as plt

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


fig, ax = plt.subplots(1)
ax.plot(Xc[:n,0],Xc[:n,1],'r.', markersize=10,alpha=0.3)
ax.plot(Xc[n:,0],Xc[n:,1],'g.', markersize=10,alpha=0.3)
ax.set_aspect('equal')


#%% Forward simple model

# Function for simple forward pass
def simple_forward(x, W):
	z = np.c_[x,np.ones((x.shape[0]))]@W[0]
	h = np.maximum(z,0)
	yh = np.c_[h,np.ones((x.shape[0]))]@W[1]
	y = np.exp(yh)/np.sum(np.exp(yh),axis=1,keepdims=True)
	return y, h

# Function for simple backpropagation
def simple_backward(x, W, t, learning_rate=0.1):
	y, h = simple_forward(x,W)
	L= -np.sum(t*np.log(y + 10e-10))/x.shape[0]
	# print(L)
	d1 = y - t
	q1 = np.c_[h,np.ones((x.shape[0]))].T@d1/y.shape[0]
	d0 = (h>0)*(d1@W[1].T)[:,:-1]
	q0 = np.c_[x,np.ones((x.shape[0]))].T@d0/y.shape[0]
	W[0] -= learning_rate*q0
	W[1] -= learning_rate*q1
	return W, L

# Function for simple weight initializaion
def simple_init_weights(n):
	W = []
	W.append(np.random.randn(3,n)*np.sqrt(2/3))
	W.append(np.random.randn(n+1,2)*np.sqrt(2/(n+1)))
	return W

W = simple_init_weights(30)


fig, ax = plt.subplots(1)
n_iter = 50
L = np.zeros((n_iter))
i_rng = np.arange(0,n_iter)
for i in range(0,n_iter):
	W, L[i] = simple_backward(Xc,W,T,learning_rate = 0.5)
	ax.cla()
	ax.plot(i_rng,L,'k')
	ax.set_title('Loss')
	plt.pause(0.001)
	plt.show()

y = simple_forward(xc,W)[0]

# Display the result
Y = y.reshape((100,100,2))
fig,ax = plt.subplots(1)
ax.imshow(Y[:,:,1],cmap='pink')
ax.plot (X[:n,0],X[:n,1],'r.',markersize=10,alpha=0.3)
ax.plot (X[n:,0],X[n:,1],'g.',markersize=10,alpha=0.3)
ax.set_aspect('equal')

#%% Forward simple model



# x = np.array([120,1])

# t = [1,0]

# w1 = np.array([[0.1,-0.01],[0,10]])

# w2 = np.array([[0.2,0],[-0.01,0.05],[0,4]])

# z = x@w1

# h = np.append(np.maximum(z,0),1)

# y_hat = h@w2

# y = np.exp(y_hat)/np.sum(np.exp(y_hat))

# L = -np.sum(t*np.log(y))

# der = (y[0]-t[0])*h[0]


