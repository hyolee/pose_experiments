import caffe
import os
caffe_root = "/om/user/hyo/src/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')

from optparse import OptionParser
import cPickle
import h5py
import numpy as np
import math

# ---- caffe setup ---
caffe.set_device(0)
caffe.set_mode_gpu()

#layer = 'fc7'
#train_val = "/om/user/hyo/src/caffe/models/3785162f95cd2d5fee77-master/VGG_ILSVRC_19_layers_deploy.prototxt"
#caffemodel = "/om/user/hyo/src/caffe/models/3785162f95cd2d5fee77-master/VGG_ILSVRC_19_layers.caffemodel"
#output_file = "/om/user/hyo/caffe/features/vgg19.p"

print "parsing options..."
parser = OptionParser()
parser.add_option("-t", "--train-val", dest="train_val")
parser.add_option("-c", "--caffemodel", dest="caffemodel")
parser.add_option("-o", "--output", dest="output_file")
parser.add_option("-l", "--layer", dest="layer", default="fc7")
(options, args) = parser.parse_args()
train_val = options.train_val
caffemodel = options.caffemodel
output_file = options.output_file
layer = options.layer

net = caffe.Net(train_val, caffemodel, caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (0,1,2))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Load HvM data
path = "/om/user/hyo/.skdata/HvMWithDiscfade_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/"
dtype, shape = cPickle.load(open(path+"header.pkl"))
f = h5py.File(path + "data.raw",'r')
d = f.require_dataset('data', shape=shape, dtype='uint8')
shape_net_data = net.blobs['data'].data.shape
net_data_size = shape_net_data[2]
border = (256 - net_data_size) / 2
data = np.empty((shape[0], 3, net_data_size, net_data_size))
for i in range(shape[0]):
    data[i] = d[i].reshape(3, 256, 256)[:, border:(border+net_data_size), border:(border+net_data_size)]

# Extract features
print "...Extract features"
no = 0
nBatch = int(math.ceil(shape[0] / shape_net_data[0]))
while no < shape[0]:
    _in = np.empty(net.blobs['data'].data.shape)
    for i in range(shape_net_data[0]):
        _in[i] = transformer.preprocess('data', data[no])
        no = no + 1
        if no == shape[0]:
            break
    net.blobs['data'].data[...] = _in
    out = net.forward()

    if no < shape_net_data[0] + 1:
        features = net.blobs[layer].data
    else:
        features = np.append(features, net.blobs[layer].data, axis=0)
features = features.reshape(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3])

with open(output_file, "w") as f:
    cPickle.dump(features, f)

