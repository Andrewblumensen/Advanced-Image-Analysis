# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:11:24 2022

@author: hujo8
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
import cv2
import sklearn.cluster
import local_features as lf
import scipy.ndimage


def ind2labels(ind):
    return np.unique(ind, return_inverse=True)[1].reshape(ind.shape)


training_image = skimage.io.imread('Bone/bone_train.png').astype(np.float)
training_labels = skimage.io.imread('Bone/bone_train_labels.png')
training_labels = ind2labels(training_labels)
nr_labels = np.max(training_labels)+1

fig, ax = plt.subplots(1,2)
ax[0].imshow(training_image, cmap=plt.cm.gray)
ax[0].set_title('training image')
ax[1].imshow(training_labels)
ax[1].set_title('labels for training image')

sigma = [1,2,4,8]
features = lf.get_gauss_feat_multi(training_image, sigma)
features = features.reshape((features.shape[0],features.shape[1]*features.shape[2]))
labels = training_labels.ravel()

labels = training_labels.ravel()

nr_keep = 15000
keep_indices = np.random.permutation(np.arange(features.shape[0]))[:nr_keep]

features_subset =features[keep_indices,:]
labels_subset = labels[keep_indices]

# fig, ax = plt.subplots(1)
# ax.imshow(features_subset)
# ax.axis('auto')

nr_clusters = 1000
#kmeans = MinibatchKMeans(n_clusters=100)
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=nr_clusters, batch_size=2*nr_clusters)
kmeans.fit(features_subset)
assignment = kmeans.labels_

edges = np.arange(nr_clusters+1)-0.5
hist = np.zeros((nr_clusters, nr_labels))

for l in range(nr_labels):
    hist[:,l] = np.histogram(assignment[labels_subset==l],bins=edges)[0]
sum_hist = np.sum(hist,axis=1)
cluster_probabilities = hist/(sum_hist.reshape(-1,1))

# fig, ax = plt.subplots(1,2)
# legend_label = [f'label {x}' for x in range(nr_labels)]

# ax[0].plot(hist,'.')
# ax[0].set_xlabel('cluster id')
# ax[0].set_ylabel('number of features in cluster')
# ax[0].legend (legend_label)
# ax[0].set_title('features in clusters per label')
# ax[1].plot(cluster_probabilities,'.')
# ax[1].set_xlabel('cluster id')
# ax[1].set_ylabel('label probability for cluster')
# ax[1].legend (legend_label)
# ax[1].set_title('cluster probabilities ')

#%%

testing_image = skimage.io.imread ('Bone/bone_test_1.png')
testing_image = testing_image.astype(np.float)
testing_gt = skimage.io.imread ('Bone/bone_test_1_labels.png')

features_testing = lf.get_gauss_feat_multi(testing_image, sigma)
features_testing = features_testing.reshape((features_testing.shape[0], features_testing.shape[1]*features_testing.shape[2]))
labels = training_labels.ravel()

assignment_testing = kmeans.predict(features_testing)

probability_image = np.zeros((assignment_testing.size, nr_labels))
for l in range(nr_labels):
    probability_image[:,l] = cluster_probabilities[assignment_testing, l]
probability_image = probability_image.reshape(testing_image.shape + (nr_labels,))

P_rgb = np.zeros (probability_image.shape[0:2]+(3,))
k = min(nr_labels,3)
P_rgb[:,:,:k] = probability_image[:,:,:k]

fig, ax = plt.subplots(1,2)
ax[0].imshow(testing_image, cmap=plt.cm.gray)
ax[0]. set_title('testing image')
ax[1]. imshow(P_rgb)
ax[1]. set_title('probabilities for testing image as RGB')

#%%

s = 2

seg_im_max = np.argmax(P_rgb,axis = 2)
c = np.eye(P_rgb.shape[2])
P_rgb_max = c[seg_im_max]

probability_smooth = np.zeros (probability_image. shape)
for i in range(0,probability_image.shape[2]):
    probability_smooth[:,:,i] = scipy.ndimage.gaussian_filter(probability_image[:,:,i],s,order=0)
seg_im_smooth = np.argmax(probability_smooth, axis=2)

probability_smooth_max = c[seg_im_smooth]

P_rgb_smooth = np.zeros(probability_smooth_max.shape [0:2]+(3,))
k = min(nr_labels,3)
P_rgb_smooth[:,:,:k] = probability_smooth[:,:,:k]
P_rgb_smooth_max = np.zeros(probability_smooth_max.shape[0:2]+(3,))
P_rgb_smooth_max[:,:,:k] = probability_smooth_max[:,:,:k]

fig, ax = plt.subplots(2,4,sharex=True, sharey=True)
ax[0][0].imshow(P_rgb[:,:,0])
ax[0][1].imshow(P_rgb[:,:,1])
ax[0][2].imshow(testing_gt, cmap = 'gray')
ax[0][3].imshow(P_rgb_max)
ax[1][0].imshow(P_rgb_smooth[:,:,0])
ax[1][1].imshow(P_rgb_smooth[:,:,1])
ax[1][2].imshow(testing_gt, cmap = 'gray')
ax[1][3].imshow(P_rgb_smooth_max)

#%%

thres = 120
im_thres = (testing_image>thres).astype(np.float)
seg_rgb = np.zeros((P_rgb.shape))

seg_rgb[:,:,0] = im_thres*seg_im_smooth
seg_rgb[:,:,2] = im_thres*(1-seg_im_smooth)
seg_rgb[:,:,1] = (1-im_thres)

fig, ax = plt.subplots(1,3,sharex=True, sharey=True)
ax[0].imshow(im_thres, cmap = 'gray')
ax[1].imshow(1-seg_im_smooth, cmap = 'gray')
ax[2].imshow(seg_rgb)

#%%   

def boundary_length(binim, mask=1):
    bim = np.zeros(binim.shape)
    bim[:-1,:] = (binim[:-1] - binim[1:])!=0
    bim[:,:-1] += (binim[:,:-1] - binim[:,1:])!=0
    return np.sum(bim*mask)

mean_boundary_length = boundary_length(im_thres)/np.sum(im_thres)
fine_boundary_length = boundary_length(im_thres,(1-seg_im_smooth))/np.sum(im_thres*(1-seg_im_smooth))
coarse_boundary_length = boundary_length(im_thres,seg_im_smooth)/np.sum(im_thres*seg_im_smooth)
print(f'Mean boundary length: \t\t\t\t{mean_boundary_length:0.2f}')
print(f'Boundary length of fine structure: \t{fine_boundary_length:0.2f}')
print(f'Boundary length of coarse structure: \t{coarse_boundary_length:0.2f}')

#%% RELATIVE TO GROUND TRUTH

testing_gt_bin = testing_gt == 255
mean_boundary_length = boundary_length(im_thres)/np.sum(im_thres)
fine_boundary_length = boundary_length(im_thres,(1-testing_gt_bin))/np.sum(im_thres*(1-testing_gt_bin))
coarse_boundary_length = boundary_length(im_thres, testing_gt_bin)/np.sum(im_thres*testing_gt_bin)
print(f'Mean_boundary_length: \t\t\t\t{mean_boundary_length:0.2f}')
print(f'Boundary length of fine structure: \t{fine_boundary_length:0.2f}')
print(f'Boundary length of coarse structure: \t{coarse_boundary_length:0.2f}')








