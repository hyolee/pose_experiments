import numpy as np
import caffe
import h5py

model_def = "/om/user/hyo/caffe/train_val_reg_short.prototxt"
model = "/om/user/hyo/caffe/caffe_rotation2d_reg_short_6_iter_30000.caffemodel"
net = caffe.Net(model_def, model, caffe.TEST)

f = h5py.File("/om/user/hyo/caffe/reg/hdf5_reg_train.h5", "r")
d = f.require_dataset('data', shape=(10000,3,227,227), dtype='float32')
actual = []
predict = []
for i in range(10000/50):
    net.blobs['data'].data[...] = d[i*50:(i+1)*50]
    out = net.forward()
    #if net.blobs['loss'].data[0][0][0][0] > 0.1:
    #    print "ERROR: " + str(net.blobs['loss'].data[0][0][0][0]) + " with i=" + str(i)
    label_vec = [[net.blobs['label_vec'].data[i][0][0][0], net.blobs['label_vec'].data[i][1][0][0]] for i in range(50)]
    fc_pose = [[net.blobs['fc_pose'].data[i][0][0][0], net.blobs['fc_pose'].data[i][1][0][0]] for i in range(50)]

    actual = actual + label_vec
    predict = predict + fc_pose

dict = { 'actual': np.array(actual), 'predict': np.array(predict) }
import cPickle
with open("predictions_all_train.pkl", "wb") as f:
    cPickle.dump(dict, f)
