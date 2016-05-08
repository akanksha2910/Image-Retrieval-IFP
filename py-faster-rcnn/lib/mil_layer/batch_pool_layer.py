"""
Layer: BatchPoolLayer

For Multiple Instance Learning

This layer implements feature pooling on batch level,
either max pooling or average pooling

INPUT: N x C x W x H
Output: 1 x C x W x H

Copyright @ Xianming Liu, University of Illinois, Urbana-Champaign
"""

import caffe
import numpy as np
import yaml
from scipy.stats import multivariate_normal


__author__ = ['Xianming Liu (liuxianming@gmail.com)']


class BatchPoolLayer(caffe.Layer):
    """
    Outputs pooled features on batch level,
    i.e., merge all features of samples within a batch into a single one
    This layer is utilized in MIL
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        # pooling method, by default is "MAX"
        self._pooling = layer_params.get('PoolMethod', 'MAX')
        self._max_ids = None
        # use _pooling_mask storing the weight matrix
        self._pooling_mask = None

    def forward(self, bottom, top):
        """Forward: either using max pooling or average pooling
        """
        num_ = bottom[0].data.shape[0]
        bottom_data_ = bottom[0].data.reshape(num_, -1)
        # pooling mask is of the same size as input
        # then backward could be calculated as matrix multiplication
        self._pooling_mask = np.zeros(bottom[0].data.shape)
        # max pooling
        if self._pooling == 'MAX':
            # perform max pooling
            pooled_data_ = np.max(bottom_data_, axis=0)
            self._max_ids = np.argmax(bottom_data_, axis=0).reshape(
                bottom[0].shape[1:])
            # set the pooling_mask
            for i in range(len(self._max_ids)):
                self._pooling_mask[self._max_ids[i], i] = 1.0
        elif self._pooling == 'AVE':
            # perform average pooling
            pooled_data_ = np.mean(bottom_data_, axis=0)
            self._pooling_mask = np.ones(self._pooling_mask.shape)
            self._pooling_mask *= (1.0 / float(num_))
        elif self._pooling == 'MED':
            # using Median pooling, which is used to eliminate outlier points
            pooled_data_ = np.median(bottom_data_, axis=0)
            self._median_ids = np.where(bottom_data_ == pooled_data_)[0]
            # set the pooling_mask
            for i in range(len(self._median_ids)):
                self._pooling_mask[self._median_ids[i], i] = 1.0
        elif self._pooling == 'GAUSSIAN':
            # Fit a gaussian distribution
            bottom_mean_ = np.mean(bottom_data_, axis=0)
            bottom_var_ = np.cov(bottom_data_.transpose())
            self._gaussian_weights = multivariate_normal.pdf(
                bottom_data_, bottom_mean_, bottom_var_, allow_singular=True)
            self._gaussian_weights /= self._gaussian_weights.sum()
            self._pooling_mask = np.ones(self._pooling_mask.shape)
            pooled_data_ = bottom_data_ * self._gaussian_weights[:, np.newaxis]
            pooled_data_ = pooled_data_.sum(axis=0)
            for i in range(len(self._gaussian_weights)):
                self._pooling_mask[i, ...] *= self._gaussian_weights[i]
        else:
            print("WRONG POOLING TYPE")
            pooled_data_ = np.zeros((1, bottom_data_.shape[1]))
        top[0].data[0, ...] = pooled_data_.reshape(bottom[0].data.shape[1:])

    def backward(self, top, propagate_down, bottom):
        """Calculate backward gradients
        """
        top_diff_ = top[0].diff.flatten()
        bottom_diff_ = top_diff_ * self._pooling_mask
        bottom[0].diff[...] += bottom_diff_.reshape(bottom[0].diff.shape)

    def reshape(self, bottom, top):
        top[0].reshape(1, *(bottom[0].shape[1:]))
