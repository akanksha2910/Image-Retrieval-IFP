#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    img = im.astype(np.float32, copy=True)
    img -= cfg.PIXEL_MEANS
    return img.transpose(2, 0, 1), np.array(
        [img.shape[0], img.shape[1], 1]).reshape(1, 3)


def vis_proposals(net, image_name):
    """Draw region proposals."""
    im = cv2.imread(image_name)
    blob, im_info = _get_image_blob(im)
    net.blobs['data'].reshape(1, *(blob.shape))
    net.blobs['data'].data[...] = blob
    net.blobs['im_info'].data[...] = im_info

    net.forward()
    rois = net.blobs['rois'].data
    # if it exists, output the score
    if 'rois_scores' in net.blobs.keys():
        rois_scores = net.blobs['rois_scores'].data
        print rois_scores
    # start drawing
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(rois)):
        bbox = rois[i, 1:5]
        print bbox
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2)
            )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Visualization of Region Proposals')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--network', help='Network protofile path',
                        required=True, dest='network')
    parser.add_argument('--model', help='Network pretrained model path',
                        required=True, dest='model')
    parser.add_argument('--num_proposal', help='Number of proposals to generate',
                        dest='num_proposal', type=int, default=128)
    parser.add_argument('--image', help='Image to visualize',
                        dest='image', required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    prototxt = args.network
    model = args.model

    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found.').format(prototxt))
    if not os.path.isfile(model):
        raise IOError(('{:s} not found.').format(model))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, model, caffe.TEST)
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.RPN_POST_NMS_TOP_N = args.num_proposal
    cfg.TEST.RPN_MIN_SIZE = 16

    print '\n\nLoaded network {:s}'.format(model)

    print('Visualizing the region proposals for image {}'.format(args.image))
    vis_proposals(net, args.image)
    plt.show()
