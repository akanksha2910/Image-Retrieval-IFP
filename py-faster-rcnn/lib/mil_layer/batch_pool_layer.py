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

    def forward(self, bottom, top):
        """Forward: either using max pooling or average pooling
        """
        num_ = bottom[0].data.shape[0]
        bottom_data_ = bottom[0].data.reshape(num_, -1)
        # max pooling
        if self._pooling == 'MAX':
            # perform max pooling
            pooled_data_ = np.max(bottom_data_, axis=0)
            self._max_ids = np.argmax(bottom_data_, axis=0).reshape(
                bottom[0].shape[1:])
        if self._pooling == 'AVE':
            # perform average pooling
            pooled_data_ = np.mean(bottom_data_, axis=0)
        top[0].data[0, ...] = pooled_data_.reshape(bottom[0].data.shape[1:])

    def backward(self, top, propagate_down, bottom):
        """Calculate backward gradients
        """
        num_ = bottom[0].data.shape[0]
        top_diff_ = top[0].diff.flatten()
        if self._pooling == 'MAX':
            if self._max_ids is not None:
                mask_ = self._max_ids.flatten()
                bottom_diff_ = np.zeros(bottom[0].diff.shape)
                bottom_diff_ = bottom_diff_.reshape(num_, -1)
                for i in range(len(top_diff_)):
                    pos = mask_[i]
                    bottom_diff_[pos, i] = top_diff_[i]
            else:
                print("Max pooling mask is empty")
                raise
        if self._pooling == 'AVE':
            # For average pooling, just pass down the top gradient
            bottom_diff_ = top_diff_[np.newaxis, :] / num_
        bottom[0].diff[...] += bottom_diff_.reshape(bottom[0].diff.shape)

    def reshape(self, bottom, top):
        top[0].reshape(1, *(bottom[0].shape[1:]))
