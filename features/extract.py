import numpy as np

caffe_root = "/om/user/hyo/src/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import os 
import cPickle

from optparse import OptionParser
print "parsing options..."
parser = OptionParser()
#parser.add_option("-t", "--train-val", dest="train_val")
#parser.add_option("-c", "--caffemodel", dest="caffemodel")
#parser.add_option("-i", "--img-path", dest="img_path")
#parser.add_option("-s", "--start", dest="start", type="int", default=0)
parser.add_option("-b", "--nBatch", dest="nBatch", type="int", default=23)
#parser.add_option("-o", "--output", dest="output_path")

parser.add_option("-n", "--net", dest="net")
parser.add_option("-c", "--nClass", dest="nClass")
parser.add_option("-d", "--nDim", dest="nDim")
parser.add_option("-l", "--lr", dest="lr")
parser.add_option("-i", "--iter", dest="iter")
parser.add_option("-e", "--ext_data", dest="ext_data")

(options, args) = parser.parse_args()
#train_val = options.train_val
#caffemodel = options.caffemodel
#img_path = options.img_path
#start = options.start
nBatch = options.nBatch
#output_path = options.output_path

net = options.net
nClass = options.nClass
nDim = options.nDim
lr = options.lr
iter = options.iter
ext_data = options.ext_data

#net = 'big'
#nClass = '1novar'
#nDim = '3'
#lr ='3'
#iter = '20000'
#ext_data = 'simple1_novar'
layer = 'fc_pose'

ext = 'reg_' + net + '_class' + nClass + '_' + nDim + 'd'
train_val = "/om/user/hyo/caffe/train_val/train_val_" + ext + ".prototxt"
caffemodel = "/om/user/hyo/caffe/snapshot/caffe_rotation" + nDim + "d_" + ext + "_lr" + str(lr) + "_iter_" + iter + ".caffemodel"
img_path = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/"
output_path = "/om/user/hyo/caffe/features/" + ext + "_lr" + str(lr) + "_iter_" + iter + '_rosch' + ext_data + '_' + layer

# retrieve train/test range
meta_path = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl"
meta = cPickle.load(open(meta_path))
size = len(meta)
train_range = size - size/256/11 * 256
perm = np.random.RandomState(0).permutation(len(meta))
meta_p = meta[perm]


# ------- setup ------- #
mode = 'gpu'
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_device(0)
    caffe.set_mode_gpu()
net = caffe.Net(train_val, caffemodel, caffe.TEST)
#net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
#                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
#                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)
# ------------------------#

#img_path = "/om/user/hyo/.skdata/genthor/RoschDataset3_simple_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/"
#start = 792320
def get_features(start, nBatch):
    for batchnum in range(nBatch):
        for iter in range(50):
            no = start + batchnum*256 + iter
            img_file = img_path + str(no) + ".jpeg"
            if not os.path.isfile(img_file):
                break    
            net.blobs['data'].data[iter] = transformer.preprocess('data', caffe.io.load_image(img_file))
        out = net.forward()
        if batchnum == 0:
            features = np.append( net.blobs[layer].data, net.blobs['label_vec'].data, axis=1)
        else: 
            f = np.append( net.blobs[layer].data, net.blobs['label_vec'].data, axis=1)
            features = np.append(features, f, axis=0)
    return features

# train examples
features_train = get_features(0, nBatch)
features_train = np.concatenate( (features_train, meta_p['ryz1'][:nBatch*50].reshape(nBatch*50,1,1,1), meta_p['rxy1'][:nBatch*50].reshape(nBatch*50,1,1,1), meta_p['rxz1'][:nBatch*50].reshape(nBatch*50,1,1,1)), axis=1)
# test examples
features_test = get_features(train_range, nBatch)
features_test = np.concatenate( (features_test, meta_p['ryz1'][train_range:train_range+nBatch*50].reshape(nBatch*50,1,1,1), meta_p['rxy1'][train_range:train_range+nBatch*50].reshape(nBatch*50,1,1,1), meta_p['rxz1'][train_range:train_range+nBatch*50].reshape(nBatch*50,1,1,1)), axis=1)
    
import cPickle
#output_path = "/om/user/hyo/caffe/features/catInet_roschSimple74testFewer_fc7.p"
with open(output_path + "_train.p", "wb") as f:
    cPickle.dump(features_train, f)
with open(output_path + "_test.p", "wb") as f:
    cPickle.dump(features_test, f)
