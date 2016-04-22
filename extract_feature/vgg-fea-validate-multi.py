
# coding: utf-8
from __future__ import division
import numpy as np
from PIL import Image
import os
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import time
if len(sys.argv)<3:
    device_id = 0
else:
    device_id = int(sys.argv[2])
caffe.set_device(device_id)
caffe.set_mode_gpu()
#net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
#net = caffe.Net('models/VGG_ILSVRC_16_layers_256_deploy.prototxt', 'models/vgg_finetune_half_90k_v2_iter_60000.caffemodel', caffe.TEST)
#net = caffe.Net('models/VGG_ILSVRC_16_layers_256_deploy.prototxt', 'models/vgg_finetune_all_v3_iter_60000.caffemodel', caffe.TEST)
net = caffe.Net('models/VGG_16_layers_4096_deploy.prototxt', 'models/vgg_finetune_all_v4_iter_100000.caffemodel', caffe.TEST)

im_dir = '/mnt/disk1/ALISC/data/query_image'
if len(sys.argv)<2:
    id_ = 'valid_image.txt'
else:
    id_ = sys.argv[1]
imlist_path = '/mnt/disk1/ALISC/data/eval_tags/'+id_
print imlist_path
with open(imlist_path,'r') as f:
    impath = map(str.strip,f.readlines())
    impath = map(lambda x:x.split(',')[0],impath)
batchSize = 128
nBatch = int(len(impath)/batchSize)
print nBatch
fea_pool1 = np.zeros((len(impath),64),dtype=np.float32)
fea_pool2 = np.zeros((len(impath),128),dtype=np.float32)
fea_pool3 = np.zeros((len(impath),256),dtype=np.float32)
fea_pool4_ave = np.zeros((len(impath),512),dtype=np.float32)
fea_pool4_max = np.zeros((len(impath),512),dtype=np.float32)
fea_pool5_max = np.zeros((len(impath),512),dtype=np.float32)

#save_dir = '/mnt/disk4/szm/share/ALISC/feature/vgg16/valid_image'
save_dir_pool1 = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool1/valid_image'
save_dir_pool2 = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool2/valid_image'
save_dir_pool3 = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool3/valid_image'
save_dir_pool4_ave = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool4-ave/valid_image'
save_dir_pool4_max = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool4-max/valid_image'
save_dir_pool5_max = '/mnt/disk1/ALISC/feature/vgg16-tuned-all-v4-pool5-max/valid_image'
if not os.path.exists(save_dir_pool1):
    os.makedirs(save_dir_pool1)
if not os.path.exists(save_dir_pool2):
    os.makedirs(save_dir_pool2)
if not os.path.exists(save_dir_pool3):
    os.makedirs(save_dir_pool3)
if not os.path.exists(save_dir_pool4_ave):
    os.makedirs(save_dir_pool4_ave)
if not os.path.exists(save_dir_pool4_max):
    os.makedirs(save_dir_pool4_max)
if not os.path.exists(save_dir_pool5_max):
    os.makedirs(save_dir_pool5_max)

net.blobs['data'].reshape(batchSize,3,224,224)
transformer = caffe.io.Transformer({'data': (1,3,256,256)})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('alisc_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order 
start = time.time()
for ii in xrange(nBatch):
    if ii%20 == 0:
        print ii
    for jj in xrange(batchSize):
        #impath = ['horse.jpg']
        idx = ii*batchSize+jj
        #print idx
        im = caffe.io.load_image(os.path.join(im_dir,impath[idx]+'.jpg'))
        #print 'load im size',im.shape, im[100,100]
        im = transformer.preprocess('data',im)
        #print 'after transform',im.shape,im[:,100,100]
        in_ = im[:,16:240,16:240]
        net.blobs['data'].data[jj] = in_[None,:,:,:]
        #net.blobs['data'].data[...] = np.concatenate((in_[None,:,:,:], in_[None,:,:,:]),axis=0)
    #net.blobs['data'].data[...] = np.concatenate((net.blobs['data'].data[...],in_[None,:,:,:]),axis=0)
    net.forward()
    out_pool1 = net.blobs['pool1-ave'].data
    out_pool2 = net.blobs['pool2-ave'].data
    out_pool3 = net.blobs['pool3-ave'].data
    out_pool4_ave = net.blobs['pool4-ave'].data
    out_pool4_max = net.blobs['pool4-max'].data
    out_pool5_max = net.blobs['pool5-max'].data
    #print 'data:', net.blobs['fc7'].data.shape
    #print 'out:',out_pool5.shape,type(out_pool5)
    #import code; code.interact(local=locals())
    #print out[980:]
    #print type(net.blobs['fc7']),dir(net.blobs['fc7'])
    #raw_input()
    fea_pool1[ii*batchSize:(ii+1)*batchSize] = out_pool1.reshape(out_pool1.shape[0],out_pool1.shape[1])
    fea_pool2[ii*batchSize:(ii+1)*batchSize] = out_pool2.reshape(out_pool2.shape[0],out_pool2.shape[1])
    fea_pool3[ii*batchSize:(ii+1)*batchSize] = out_pool3.reshape(out_pool3.shape[0],out_pool3.shape[1])
    fea_pool4_ave[ii*batchSize:(ii+1)*batchSize] = out_pool4_ave.reshape(out_pool4_ave.shape[0],out_pool4_ave.shape[1])
    fea_pool4_max[ii*batchSize:(ii+1)*batchSize] = out_pool4_max.reshape(out_pool4_max.shape[0],out_pool4_max.shape[1])
    fea_pool5_max[ii*batchSize:(ii+1)*batchSize] = out_pool5_max.reshape(out_pool5_max.shape[0],out_pool5_max.shape[1])
    
for jj in range(len(impath)-nBatch*batchSize):
        idx = jj + nBatch*batchSize
        im = caffe.io.load_image(os.path.join(im_dir,impath[idx]+'.jpg'))
        #print 'load im size',im.shape, im[100,100]
        im = transformer.preprocess('data',im)
        #print 'after transform',im.shape,im[:,100,100]
        in_ = im[:,16:240,16:240]
        if jj == 0: 
            net.blobs['data'].reshape(len(impath)-nBatch*batchSize, *in_.shape)
        net.blobs['data'].data[jj] = in_[None,:,:,:]
        #net.blobs['data'].data[...] = np.concatenate((in_[None,:,:,:], in_[None,:,:,:]),axis=0)
    #net.blobs['data'].data[...] = np.concatenate((net.blobs['data'].data[...],in_[None,:,:,:]),axis=0)
net.forward()
#out_pool5 = net.blobs['pool5-ave'].data#[0]#.argmax(axis=0)
#print 'data:', net.blobs['fc7'].data.shape
out_pool1 = net.blobs['pool1-ave'].data
out_pool2 = net.blobs['pool2-ave'].data
out_pool3 = net.blobs['pool3-ave'].data
out_pool4_ave = net.blobs['pool4-ave'].data
out_pool4_max = net.blobs['pool4-max'].data
out_pool5_max = net.blobs['pool5-max'].data
#print 'out:',out.shape,type(out)
#print out[980:]
#print type(net.blobs['fc7']),dir(net.blobs['fc7'])
#raw_input()
fea_pool1[nBatch*batchSize:len(impath)] = out_pool1.reshape(out_pool1.shape[0],out_pool1.shape[1])
fea_pool2[nBatch*batchSize:len(impath)] = out_pool2.reshape(out_pool2.shape[0],out_pool2.shape[1])
fea_pool3[nBatch*batchSize:len(impath)] = out_pool3.reshape(out_pool3.shape[0],out_pool3.shape[1])
fea_pool4_ave[nBatch*batchSize:len(impath)] = out_pool4_ave.reshape(out_pool4_ave.shape[0],out_pool4_ave.shape[1])
fea_pool4_max[nBatch*batchSize:len(impath)] = out_pool4_max.reshape(out_pool4_max.shape[0],out_pool4_max.shape[1])
fea_pool5_max[nBatch*batchSize:len(impath)] = out_pool5_max.reshape(out_pool5_max.shape[0],out_pool5_max.shape[1])

fout_pool1 = file(os.path.join(save_dir_pool1,'valid-fea-tuned-vgg16-pool1'),'wb')
fea_pool1 = np.asfortranarray(fea_pool1).astype(np.float32)
np.save(fout_pool1,fea_pool1)
fout_pool1.close()

fout_pool2 = file(os.path.join(save_dir_pool2,'valid-fea-tuned-vgg16-pool2'),'wb')
fea_pool2 = np.asfortranarray(fea_pool2).astype(np.float32)
np.save(fout_pool2,fea_pool2)
fout_pool2.close()


fout_pool3 = file(os.path.join(save_dir_pool3,'valid-fea-tuned-vgg16-pool3'),'wb')
fea_pool3 = np.asfortranarray(fea_pool3).astype(np.float32)
np.save(fout_pool3,fea_pool3)
fout_pool3.close()


fout_pool4_ave = file(os.path.join(save_dir_pool4_ave,'valid-fea-tuned-vgg16-pool4-ave'),'wb')
fea_pool4_ave = np.asfortranarray(fea_pool4_ave).astype(np.float32)
np.save(fout_pool4_ave,fea_pool4_ave)
fout_pool4_ave.close()


fout_pool4_max = file(os.path.join(save_dir_pool4_max,'valid-fea-tuned-vgg16-pool4-max'),'wb')
fea_pool4_max = np.asfortranarray(fea_pool4_max).astype(np.float32)
np.save(fout_pool4_max,fea_pool4_max)
fout_pool4_max.close()


fout_pool5_max = file(os.path.join(save_dir_pool5_max,'valid-fea-tuned-vgg16-pool5-max'),'wb')
fea_pool5_max = np.asfortranarray(fea_pool5_max).astype(np.float32)
np.save(fout_pool5_max,fea_pool5_max)
fout_pool5_max.close()


end = time.time()
print 'elapsed time:', end-start

